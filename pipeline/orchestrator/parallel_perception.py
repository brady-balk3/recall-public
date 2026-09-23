# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Parallel perception (handbook/20 §2-3): saturate the CPU instead of one thread.

The sequential perception loop runs vision + OCR + motion + emotion on one thread
at 1 fps; OCR (CPU) is ~85-90% of the time, so a long VOD takes ~realtime/2. This
splits the VOD into contiguous time slices and runs a process per slice, each doing
the FULL per-frame work (vision + OCR + motion + facecam emotion) over a single
decode of its range. On an 8c/16t CPU this is ~8-10x on the bottleneck.

Workers use the GPU when one is present with enough free VRAM (plan_perception),
falling back to the original CPU-only pool otherwise. Measured 2026-07-25 on an
8-core box + RTX 4070 Ti SUPER, production settings: 6 CPU workers sustain 5.41x
realtime, 6 GPU workers 11.26x, with identical row coverage and OCR text output.
Both OCR and YOLO move to the GPU together — the earlier "OCR is CPU anyway"
assumption stopped holding once onnxruntime-gpu (CUDA 12) became the installed
build. Emotion stays on CPU regardless (HSEmotion pins CPUExecutionProvider), as
do decode and motion, so workers remain single-CPU-thread pinned in both modes.
Emotion is folded in here (one decode), removing the separate second decode pass.
"""

import os
import sys
import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait

from core.process_governor import get_recall_env

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_HEARTBEAT_SOURCE_SECONDS = 5.0
_HEARTBEAT_EMIT_SECONDS = 5.0
_RATE_WINDOW_SECONDS = 120.0

# Worker liveness watchdog (resource plan pkg 3). Workers publish a source
# timestamp at least every _HEARTBEAT_SOURCE_SECONDS of footage; a worker
# whose position stops advancing is stuck (pathological decode section, OCR
# hang, dead process). WARN surfaces it in telemetry; FAIL kills the pool and
# raises WorkerStallError so the job fails with a diagnosable message instead
# of hanging forever. The warn threshold is generous because a worker's first
# advance waits behind model load (minutes on a cold, slow disk).
_STALL_WARN_SECONDS = float(get_recall_env("RECALL_WORKER_STALL_WARN_SEC", "600"))
_STALL_FAIL_SECONDS = float(get_recall_env("RECALL_WORKER_STALL_FAIL_SEC", "1800"))


class WorkerStallError(RuntimeError):
    """A perception worker stopped making progress past the hard timeout."""


def _missing_ranges(covered_secs, duration, min_gap=30.0):
    """Contiguous missing-second ranges longer than ``min_gap``, as (start, end).

    A perception worker that hits a transient decode failure returns its slice
    early and silently, leaving a hole in the per-second coverage. This finds
    those holes so they can be refilled before the scan ships a gap. Tiny 1-2s
    holes (normal near cuts) are ignored; only real dropped-slice gaps surface.
    Pure so the detection is unit-testable without decoding video.
    """
    dur = int(duration)
    missing = []
    x = 0
    while x < dur:
        if x not in covered_secs:
            start = x
            while x < dur and x not in covered_secs:
                x += 1
            if x - start >= min_gap:
                missing.append((float(start), float(x)))
        else:
            x += 1
    return missing


def _stalled_workers(last_advance, completed, now, threshold):
    """Indices of unfinished workers whose position hasn't advanced in
    ``threshold`` seconds. Pure function so the watchdog logic is testable."""
    if threshold <= 0:
        return []
    return sorted(
        index for index, stamp in last_advance.items()
        if index not in completed and now - stamp >= threshold
    )


def _worker_setup_paths():
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def _perceive_chunk(payload):
    """Worker: perceive one [start, end) slice in a fresh process.

    CRITICAL: pin each worker to a SINGLE CPU thread. OCR/onnxruntime and torch
    each default to using all CPU cores, so N multi-threaded workers oversubscribe
    and thrash (no real speedup). One thread per worker => N clean parallel streams.
    That pinning applies in BOTH device modes — decode, motion, face emotion
    (HSEmotion hardcodes CPUExecutionProvider) and PP-OCR's post-processing stay
    on the CPU regardless.

    ``gpu_ocr`` decides whether the worker can see CUDA. When it can, both
    _ocr_device() and the person detector's onnxruntime providers auto-select
    the GPU, which measured 799 -> 346 ms per frame at production settings
    (2.3x pool throughput). The detector used to route via torch
    (detector._DEVICE); it now runs on onnxruntime like OCR does, so CUDA
    visibility is what gates it in both cases.
    A GPU that fails to init is not fatal: engines.ocr.ocr._get_ocr_engine
    falls back to a CPU engine for that worker.
    """
    if not payload.get("gpu_ocr"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # CPU-only worker
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    # cv2.setNumThreads/OMP below do NOT reach the FFmpeg decoder inside
    # cv2.VideoCapture — it defaults to ~core-count threads PER WORKER, so N
    # workers silently spawn N*cores decode threads. With one slice per
    # worker there's no need for intra-decoder parallelism.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
    _worker_setup_paths()

    # Background CPU scheduling (resource plan pkg 1): the scan must yield to
    # the user's foreground apps instantly, while still eating every idle
    # cycle. Costs ~nothing on an otherwise-idle machine.
    if payload.get("background_priority", True):
        from core.process_governor import enter_background_mode
        enter_background_mode()

    import cv2
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1)

    # Force every onnxruntime session (PP-OCRv6 det/rec, HSEmotion) to 1 intra-op
    # thread by patching the SessionOptions factory the libs call.
    import onnxruntime as _ort
    _OrigSO = _ort.SessionOptions
    def _single_thread_so():
        o = _OrigSO()
        try:
            o.intra_op_num_threads = 1
            o.inter_op_num_threads = 1
        except Exception:
            pass
        return o
    _ort.SessionOptions = _single_thread_so
    from engines.video.frame_extractor import extract_frames_range
    from engines.vision.detector import has_stable_facecam, run_vision, reset_vision_state
    from engines.video.motion import compute_motion_score, prepare_motion_frame
    from engines.ocr.ocr import run_ocr, warm_ocr
    from core.models.signal import OCRSignal, VisionSignal
    import engines.emotion.face_emotion as fe

    # Load the OCR engine BEFORE extract_frames_range opens a cv2.VideoCapture:
    # on Windows the paddlex DLLs must init before OpenCV's ffmpeg DLL or
    # PaddleOCR fails (WinError 1114) and OCR is disabled for the whole slice.
    warm_ocr()

    video_path = payload["video_path"]
    start, end = payload["start"], payload["end"]
    facecam_tracking = payload["facecam_tracking"]
    ocr_min_conf = payload["ocr_min_confidence"]
    ocr_stride = payload["ocr_stride"]
    include_facecam_emotion = payload.get("include_facecam_emotion", True)
    facecam_scan_interval = max(1, int(payload.get("facecam_scan_interval", 1)))
    stable_detections_required = max(1, int(payload.get("stable_facecam_detections", 1)))

    reset_vision_state()
    fe.reset_emotion_state()
    fer = fe._get_model() if include_facecam_emotion else None
    rows = []           # (timestamp, VisionSignal, OCRSignal)
    faces = []          # FaceFrame
    prev_motion = None
    last_bbox = None
    progress_proxy = payload.get("progress_proxy")
    progress_key = payload.get("progress_key")
    last_progress_timestamp = start

    if progress_proxy is not None and progress_key is not None:
        try:
            progress_proxy[progress_key] = float(start)
        except Exception:
            pass

    for timestamp, frame in extract_frames_range(video_path, start, end, fps_sample_rate=1.0):
        if facecam_tracking:
            elapsed = max(0.0, timestamp - start)
            should_scan_facecam = (
                facecam_scan_interval <= 1
                or int(round(elapsed)) % facecam_scan_interval == 0
                or not has_stable_facecam()
            )
            vision = run_vision(
                frame,
                timestamp,
                detect_facecam=should_scan_facecam,
                stable_detections_required=stable_detections_required,
            )
        else:
            vision = VisionSignal(timestamp=timestamp, facecam_box=None, gameplay_box=[0.0, 0.0, 1.0, 1.0])

        motion_frame = prepare_motion_frame(frame, vision.facecam_box if vision else None)
        if vision:
            vision.motion_score = compute_motion_score(prev_motion, motion_frame)
        prev_motion = motion_frame

        if int(round(timestamp)) % ocr_stride == 0:
            ocr = run_ocr(frame, timestamp, min_confidence=ocr_min_conf)
        else:
            ocr = OCRSignal(timestamp=timestamp, text="", confidence=0.0)

        # Facecam emotion can be skipped in the fast first pass and then run only
        # inside candidate regions during refinement.
        if include_facecam_emotion:
            box = vision.facecam_box if vision else None
            panel = fe._crop_facecam(frame, box) if box else None
            face = None
            if panel is not None:
                face, last_bbox = fe._detect_face(panel, last_bbox)
            if fer is None or face is None:
                faces.append(fe.FaceFrame(timestamp, False, 0.0, 0.0, "neutral"))
            else:
                try:
                    import numpy as np
                    emotion, scores = fer.predict_emotions(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), logits=False)
                    scores = np.asarray(scores, dtype="float32").ravel()
                    arousal = float(scores[fe._AROUSAL_IDX]) if scores.size > fe._AROUSAL_IDX else 0.0
                    valence = float(scores[fe._VALENCE_IDX]) if scores.size > fe._VALENCE_IDX else 0.0
                    faces.append(fe.FaceFrame(timestamp, True, arousal, valence, str(emotion)))
                except Exception:
                    faces.append(fe.FaceFrame(timestamp, False, 0.0, 0.0, "neutral"))

        rows.append((timestamp, vision, ocr))

        if progress_proxy is not None and progress_key is not None \
                and timestamp - last_progress_timestamp >= _HEARTBEAT_SOURCE_SECONDS:
            try:
                progress_proxy[progress_key] = float(timestamp)
                last_progress_timestamp = timestamp
            except Exception:
                pass

    if progress_proxy is not None and progress_key is not None:
        try:
            progress_proxy[progress_key] = float(end)
        except Exception:
            pass

    return {"start": start, "rows": rows, "faces": faces}


def _parallel_progress_snapshot(payloads, positions, histories, completed, now, duration):
    """Return coverage, aggregate speed, and a straggler-aware stage ETA."""
    scanned = 0.0
    worker_rates = {}
    for index, payload in enumerate(payloads):
        start = float(payload["start"])
        end = min(float(payload["end"]), float(duration))
        position = end if index in completed else float(positions.get(index, start))
        position = max(start, min(end, position))
        scanned += max(0.0, position - start)

        samples = [
            sample for sample in histories.get(index, [])
            if sample[0] >= now - _RATE_WINDOW_SECONDS
        ]
        if len(samples) >= 2:
            first, last = samples[0], samples[-1]
            elapsed = last[0] - first[0]
            advanced = last[1] - first[1]
            if elapsed >= 3.0 and advanced > 0:
                worker_rates[index] = advanced / elapsed

    known_rates = sorted(rate for rate in worker_rates.values() if rate > 0)
    fallback_rate = known_rates[len(known_rates) // 2] if known_rates else None
    worker_etas = []
    for index, payload in enumerate(payloads):
        if index in completed:
            continue
        start = float(payload["start"])
        end = min(float(payload["end"]), float(duration))
        position = max(start, min(end, float(positions.get(index, start))))
        rate = worker_rates.get(index, fallback_rate)
        if rate and rate > 0:
            worker_etas.append(max(0.0, end - position) / rate)

    stage_eta = max(worker_etas) if worker_etas else (
        0.0 if len(completed) == len(payloads) else None
    )
    active = max(0, len(payloads) - len(completed))
    known_fraction = len(worker_rates) / max(1, active)
    spans = [
        samples[-1][0] - samples[0][0]
        for key, samples in histories.items()
        if key != "coverage" and len(samples) >= 2
    ]
    longest_span = max(spans, default=0.0)
    if stage_eta is None:
        confidence = "unknown"
    elif known_fraction >= 0.8 and longest_span >= 60.0:
        confidence = "high"
    elif known_fraction >= 0.5 and longest_span >= 15.0:
        confidence = "medium"
    else:
        confidence = "low"

    aggregate_speed = None
    coverage_samples = [
        sample for sample in histories.get("coverage", [])
        if sample[0] >= now - _RATE_WINDOW_SECONDS
    ]
    if len(coverage_samples) >= 2:
        first, last = coverage_samples[0], coverage_samples[-1]
        elapsed = last[0] - first[0]
        advanced = last[1] - first[1]
        if elapsed >= 3.0 and advanced > 0:
            aggregate_speed = advanced / elapsed

    return {
        "scanned_seconds": round(min(float(duration), scanned), 1),
        "scan_speed": round(aggregate_speed, 2) if aggregate_speed else None,
        "stage_eta_seconds": round(stage_eta) if stage_eta is not None else None,
        "stage_eta_confidence": confidence,
        "eta_scope": "perception",
        "eta_scope_end_progress": 0.4,
        "active_workers": active,
        "total_workers": len(payloads),
    }


def gpu_perception_enabled(settings=None) -> bool:
    """Kill switch for GPU perception workers (default ON when a GPU fits).

    ``RECALL_GPU_PERCEPTION=0`` or ``settings["gpuPerception"] = False`` forces
    the historical CPU-only workers. The actual decision still depends on free
    VRAM — see plan_perception.
    """
    env = os.environ.get("RECALL_GPU_PERCEPTION", "").strip().lower()
    if env in ("0", "false", "off", "no"):
        return False
    if env in ("1", "true", "on", "yes"):
        return True
    if not settings:
        return True
    raw = settings.get("gpuPerception", settings.get("gpu_perception", True))
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)


def plan_perception(n_workers, settings=None):
    """Resolve ``(workers, gpu_ocr)`` for a perception pass.

    GPU workers are ~2.3x the throughput of CPU workers but cost ~1.2 GB VRAM
    each, so the count may be reduced to fit the card. Falling back to CPU
    keeps the ORIGINAL worker count — this can only ever swap a slower pool for
    a faster one, never shrink the pool without the speedup to pay for it.
    """
    n_workers = max(1, int(n_workers))
    if not gpu_perception_enabled(settings):
        return n_workers, False
    try:
        from core.process_governor import gpu_perception_workers

        gpu_workers = gpu_perception_workers(n_workers)
    except Exception as exc:  # noqa: BLE001 - never block a scan on this
        print(f"GPU perception planning skipped: {exc}")
        return n_workers, False
    if gpu_workers <= 0:
        print(
            "GPU perception: unavailable or too little free VRAM; "
            f"using {n_workers} CPU workers"
        )
        return n_workers, False
    if gpu_workers < n_workers:
        print(
            f"GPU perception: free VRAM fits {gpu_workers} of {n_workers} "
            "workers; running fewer, faster workers"
        )
    return gpu_workers, True


def perceive_parallel(video_path, duration, settings, pipeline_settings, n_workers,
                      emit=None, gpu_ocr=False):
    """Run perception across processes. Returns (rows, faces) merged + time-sorted.

    rows: list of (timestamp, VisionSignal, OCRSignal). faces: list of FaceFrame
    (median-smoothed over present frames, matching analyze_faces).
    """
    n_workers = max(1, int(n_workers))
    span = duration / n_workers
    payloads = []
    for i in range(n_workers):
        payloads.append({
            "video_path": video_path,
            "start": i * span,
            "end": (i + 1) * span if i < n_workers - 1 else duration + 1.0,
            "facecam_tracking": settings.get("facecamTracking", True),
            "ocr_min_confidence": pipeline_settings["ocr_min_confidence"],
            "ocr_stride": pipeline_settings["ocr_stride"],
            "include_facecam_emotion": pipeline_settings.get("include_facecam_emotion", True),
            "facecam_scan_interval": pipeline_settings.get("facecam_scan_interval", 1),
            "stable_facecam_detections": pipeline_settings.get("stable_facecam_detections", 1),
            "background_priority": settings.get("backgroundPriority", True),
            "gpu_ocr": bool(gpu_ocr),
        })

    results = []
    started = time.monotonic()
    completed = set()
    histories = {
        i: [(started, float(payload["start"]))]
        for i, payload in enumerate(payloads)
    }
    histories["coverage"] = [(started, 0.0)]
    last_positions = {
        i: float(payload["start"])
        for i, payload in enumerate(payloads)
    }
    last_advance = {i: started for i in range(len(payloads))}
    last_emit = 0.0

    # Workers only publish source timestamps. The parent remains the sole
    # writer to SQLite/SSE, preventing both DB contention and long silent gaps.
    with multiprocessing.Manager() as manager:
        positions = manager.dict(last_positions)
        for index, payload in enumerate(payloads):
            payload["progress_proxy"] = positions
            payload["progress_key"] = index

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            pending = {
                ex.submit(_perceive_chunk, payload): index
                for index, payload in enumerate(payloads)
            }
            while pending:
                finished, _ = wait(
                    tuple(pending), timeout=2.0, return_when=FIRST_COMPLETED
                )
                now = time.monotonic()

                for future in finished:
                    index = pending.pop(future)
                    results.append(future.result())
                    completed.add(index)
                    positions[index] = min(
                        float(payloads[index]["end"]), float(duration)
                    )

                current_positions = dict(positions)
                for index, position in current_positions.items():
                    position = float(position)
                    if position > last_positions.get(index, float(payloads[index]["start"])):
                        histories[index].append((now, position))
                        histories[index] = histories[index][-40:]
                        last_positions[index] = position
                        last_advance[index] = now

                stalled = _stalled_workers(last_advance, completed, now, _STALL_WARN_SECONDS)
                dead = _stalled_workers(last_advance, completed, now, _STALL_FAIL_SECONDS)
                if dead:
                    detail = "; ".join(
                        f"worker {index} stuck at {last_positions[index]:.0f}s "
                        f"(slice {payloads[index]['start']:.0f}-{min(float(payloads[index]['end']), duration):.0f}s, "
                        f"no progress for {(now - last_advance[index]) / 60:.0f} min)"
                        for index in dead
                    )
                    # The with-block's shutdown(wait=True) would block forever on
                    # a hung worker — kill the pool's processes first. _processes
                    # is private executor API, but there is no public way to
                    # terminate a running task.
                    for proc in list(getattr(ex, "_processes", {}).values()):
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    raise WorkerStallError(
                        f"Perception stalled and was stopped: {detail}. "
                        "The VOD may have a corrupt section — try scanning again, "
                        "or re-download the source file."
                    )

                coverage = sum(
                    max(
                        0.0,
                        min(
                            float(payload["end"]), float(duration),
                            float(current_positions.get(index, payload["start"])),
                        ) - float(payload["start"]),
                    )
                    for index, payload in enumerate(payloads)
                )
                if coverage > histories["coverage"][-1][1]:
                    histories["coverage"].append((now, coverage))
                    histories["coverage"] = histories["coverage"][-40:]

                snapshot = _parallel_progress_snapshot(
                    payloads, current_positions, histories, completed, now, duration
                )
                snapshot["stalled_workers"] = len(stalled)
                if emit and (finished or now - last_emit >= _HEARTBEAT_EMIT_SECONDS):
                    scanned = snapshot["scanned_seconds"]
                    fraction = scanned / max(1.0, float(duration))
                    message = (
                        f"Watching your VOD for big moments · {fraction * 100:.0f}% scanned"
                    )
                    if stalled:
                        message += " · finishing a few slower stretches"
                    emit(
                        "Perception",
                        message,
                        min(0.4, 0.15 + 0.25 * fraction),
                        snapshot,
                    )
                    last_emit = now

    results.sort(key=lambda r: r["start"])
    rows, faces = [], []
    for r in results:
        rows.extend(r["rows"])
        faces.extend(r["faces"])
    rows.sort(key=lambda x: x[0])
    faces.sort(key=lambda f: f.timestamp)

    # Coverage safety-net: a worker that hit a transient decode failure returns
    # its slice early and silently, leaving a hole (observed on stitched Twitch
    # VODs under heavy concurrent load). Detect real gaps and refill them with a
    # targeted re-run — far fewer workers than the main pass, so the contention
    # that caused the drop is gone and the retry succeeds. A gap that survives
    # the refill is logged loudly rather than silently shipping bad data.
    covered = set(int(round(r[0])) for r in rows)
    gaps = _missing_ranges(covered, duration, min_gap=30.0)
    if gaps:
        missing_s = sum(e - s for s, e in gaps)
        msg = (f"Perception coverage gap: {missing_s:.0f}s missing across "
               f"{len(gaps)} range(s) {[(int(s), int(e)) for s, e in gaps]} — refilling")
        print(msg)
        if emit:
            emit("Perception", msg, 0.42)
        refill_workers = max(1, min(n_workers, len(gaps)))
        refill_rows, refill_faces = perceive_regions(
            video_path, gaps, settings, pipeline_settings, refill_workers,
            emit=emit, gpu_ocr=gpu_ocr,
        )
        for r in refill_rows:
            if int(round(r[0])) not in covered:
                rows.append(r)
                covered.add(int(round(r[0])))
        face_secs = set(int(round(f.timestamp)) for f in faces)
        for f in refill_faces:
            if int(round(f.timestamp)) not in face_secs:
                faces.append(f)
                face_secs.add(int(round(f.timestamp)))
        rows.sort(key=lambda x: x[0])
        faces.sort(key=lambda f: f.timestamp)
        still = _missing_ranges(covered, duration, min_gap=30.0)
        if still:
            print(f"WARNING: {sum(e - s for s, e in still):.0f}s of perception "
                  f"still missing after refill: {[(int(s), int(e)) for s, e in still]}")

    # Median-smooth facecam arousal over present frames (matches analyze_faces).
    _worker_setup_paths()
    import numpy as np
    import engines.emotion.face_emotion as fe
    present_idx = [i for i, f in enumerate(faces) if f.face_present]
    if present_idx:
        arr = np.array([faces[i].face_arousal for i in present_idx], dtype="float32")
        sm = fe._median_smooth(arr, 5)
        for j, i in enumerate(present_idx):
            faces[i].face_arousal = float(sm[j])

    return rows, faces


def perceive_regions(video_path, regions, settings, pipeline_settings, n_workers,
                     emit=None, gpu_ocr=False):
    """Run dense perception only inside selected regions.

    Used by fast two-pass mode: the whole VOD gets a cheap first pass, then the
    likely highlight regions get 1 fps OCR + face emotion for final ranking.
    """
    if not regions:
        return [], []

    payloads = []
    for start, end in regions:
        payloads.append({
            "video_path": video_path,
            "start": max(0.0, float(start)),
            "end": max(float(start) + 1.0, float(end)),
            "facecam_tracking": settings.get("facecamTracking", True),
            "ocr_min_confidence": pipeline_settings["ocr_min_confidence"],
            "ocr_stride": 1,
            "include_facecam_emotion": settings.get("facecamTracking", True),
            "facecam_scan_interval": pipeline_settings.get("refine_facecam_scan_interval", 1),
            "stable_facecam_detections": pipeline_settings.get("stable_facecam_detections", 1),
            "background_priority": settings.get("backgroundPriority", True),
            "gpu_ocr": bool(gpu_ocr),
        })

    n_workers = max(1, min(int(n_workers), len(payloads)))
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_perceive_chunk, p) for p in payloads]
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if emit:
                emit("Perception", f"Refined region {done}/{len(payloads)}", 0.45 + 0.12 * (done / len(payloads)))

    results.sort(key=lambda r: r["start"])
    rows, faces = [], []
    seen_rows = set()
    seen_faces = set()
    for r in results:
        for row in r["rows"]:
            key = int(round(row[0]))
            if key not in seen_rows:
                rows.append(row)
                seen_rows.add(key)
        for face in r["faces"]:
            key = int(round(face.timestamp))
            if key not in seen_faces:
                faces.append(face)
                seen_faces.add(key)
    rows.sort(key=lambda x: x[0])
    faces.sort(key=lambda f: f.timestamp)

    if faces:
        _worker_setup_paths()
        import numpy as np
        import engines.emotion.face_emotion as fe
        present_idx = [i for i, f in enumerate(faces) if f.face_present]
        if present_idx:
            arr = np.array([faces[i].face_arousal for i in present_idx], dtype="float32")
            sm = fe._median_smooth(arr, 5)
            for j, i in enumerate(present_idx):
                faces[i].face_arousal = float(sm[j])

    return rows, faces


def recommended_workers(performance_profile=None):
    """Profile- and machine-derived worker count (resource plan pkg 2).

    The governor picks a PHYSICAL-core-derived count for the profile
    (background / balanced / max), then caps it by available RAM: each worker
    is a single-threaded, CPU-bound OCR stream, so scaling off logical threads
    would run 2 workers per core and thrash; scaling off physical cores and
    capping by RAM keeps the machine usable (no SMT oversubscription, no paging
    other apps out on 8-16 GB boxes).
    """
    try:
        _worker_setup_paths()
        from core.process_governor import physical_cpu_count, profile_workers
        cpu = physical_cpu_count()
        workers = profile_workers(performance_profile, cpu_count=cpu)
        print(
            f"Perception workers set to {workers} "
            f"(profile={performance_profile or 'balanced'}, {cpu} physical cores)"
        )
        return workers
    except Exception:
        # Governor unavailable: approximate physical cores (halve logical for
        # the common SMT case) and stay conservative rather than reintroduce
        # the oversubscription this function exists to avoid.
        cpu = os.cpu_count() or 4
        return max(2, min(8, cpu // 2))

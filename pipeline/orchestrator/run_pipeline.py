# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import os
import sys
import math
import uuid
import hashlib
import traceback
import time
from types import SimpleNamespace
from typing import List

# Setup canonical paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.models.signal import UnifiedSignal, VisionSignal, OCRSignal
from core.models.clip import GameClip
from core.models.story import GameStory
from core.storage_manager import StorageManager
from core import cache_keys, signal_cache
from core.timefmt import format_hms as _fmt_hms
from core.geometry import iou as _calculate_iou

# Video Engine
from engines.video.frame_extractor import extract_metadata, extract_frames
from engines.video.motion import compute_motion_score, prepare_motion_frame
# Audio Engine
from engines.audio.voice_spike import extract_audio, analyze_audio_spikes
# Vision Engine
from engines.vision.detector import has_stable_facecam, reset_vision_state, run_vision
# OCR Engine
from engines.ocr.ocr import get_ocr_max_side, run_ocr, warm_ocr
# Caption Engine
import engines.caption.caption_generator as caption_main
# Export Engine (preview-quality renders only; final export is on-demand)
import engines.export.ffmpeg_renderer as export_main
# Clip layout (reused for reaction clips)
from engines.clip.layout import assign_layout, resolve_export_layout
# Reaction Highlight Engine (handbook/19)
from engines.audio.prosody import extract_prosody
from engines.audio.asr_hype import load_lexicon, build_hype_track
from engines.audio.audio_events import analyze_audio_events, audio_events_available
from engines.caption.vod_transcribe import transcribe_vod, transcribe_vod_regions
from engines.caption.whisper_asr import (
    asr_cache_label,
    asr_display_name,
    release_asr_models,
    resolve_whisper_size,
)
from engines.emotion.face_emotion import analyze_faces, scorer_cache_label as face_scorer_label
from engines.chat import twitch_chat
from engines.chat.chat_features import build_chat_evidence, build_chat_track
import engines.reaction.reaction_main as reaction_main
# Local semantic judge (HUMAN_CLIPS Package C). Importing these is always safe:
# llama_cpp itself is only imported inside LlamaBackend on first generate.
import engines.semantic.backend as semantic_backend
import engines.semantic.judge as semantic_judge_mod
import engines.semantic.visual_judge as visual_judge_mod
from engines.reaction.game_detect import detect_game
from engines.reaction.segments import SegmentIndex, detect_game_segments, GameSegment
from engines.vision.layout_map import annotate_gameplay_text, build_layout_map
from engines.vision.gameplay_region import (
    detect_gameplay_box,
    detect_gameplay_box_at,
    detect_gameplay_layout_model,
    gameplay_for_window,
    match_gameplay_layout,
)
import engines.vision.facecam as facecam_mod
import engines.vision.vtuber as vtuber_mod
from core.ranker_service import resolve_active_ranker
from engines.reaction.runtime_settings import (
    DEFAULT_AGGRESSIVENESS,
    DEFAULT_SENSITIVITY,
    auto_clip_count,
    build_pipeline_reaction_settings,
)
from core.source_identity import canonical_source_key
from core.region_plans import merge_priority_regions
from core.device import device_summary
from core.bundle_paths import get_data_dir, get_resource_dir
from core.artifacts import write_chat_evidence_artifact, write_reaction_curve_artifact
from core.more_candidates import (
    MORE_CANDIDATE_LIMIT,
    REVIEW_TIER_SECOND_LOOK,
    ceiling_candidates_from_trace,
    more_candidate_clip_id,
    partition_second_look,
)

from pipeline.orchestrator import parallel_perception

DEFAULT_OCR_STRIDE = 3  # S2: full quality; stride 6 lost 22% of event moments
FAST_SCOUT_OCR_STRIDE = 6  # selected Smart Scan regions refine at stride 1


def _asr_progress_reporter(emit, start_progress: float, end_progress: float):
    """Translate faster-whisper segment progress into honest job telemetry.

    Segment callbacks can be frequent, so events are limited to roughly one
    every two seconds. The final callback always emits. ETA is based only on
    measured audio-seconds per wall-second and remains medium-confidence until
    enough audio has actually been decoded.
    """
    started = time.monotonic()
    last_emitted = [0.0]

    def report(processed_seconds: float, total_seconds: float):
        processed = max(0.0, float(processed_seconds or 0.0))
        total = max(0.0, float(total_seconds or 0.0))
        now = time.monotonic()
        final = total > 0 and processed >= total - 0.25
        if not final and now - last_emitted[0] < 2.0:
            return
        last_emitted[0] = now

        elapsed = max(0.001, now - started)
        speed = processed / elapsed
        ratio = min(1.0, processed / total) if total > 0 else 0.0
        eta = (total - processed) / speed if total > processed and speed > 0 else 0.0
        progress = start_progress + (end_progress - start_progress) * ratio
        confidence = "high" if elapsed >= 20.0 and ratio >= 0.05 else "medium"
        message = f"Transcribing audio · {_fmt_hms(processed)} / {_fmt_hms(total)}"
        if speed > 0:
            message += f" · {speed:.1f}×"
        emit(
            "Reaction", message, progress,
            telemetry={
                "transcribed_seconds": round(processed, 1),
                "transcription_speed": round(speed, 2),
                "stage_eta_seconds": round(max(0.0, eta)),
                "stage_eta_confidence": confidence,
                "eta_scope": "asr",
                "eta_scope_end_progress": end_progress,
            },
        )

    return report


def _rows_to_signals(rows, audio_signals):
    """Build UnifiedSignals from parallel-perception rows, attaching nearest audio."""
    audio_by_t = {int(round(a.timestamp)): a for a in audio_signals}
    out = []
    for ts, vision, ocr in rows:
        out.append(UnifiedSignal(
            timestamp=ts, audio=audio_by_t.get(int(round(ts))), ocr=ocr, vision=vision
        ))
    return out


def _merge_refined_signals(base_signals, refined_rows, audio_signals):
    """Overlay dense regional perception rows onto the cheap whole-VOD pass."""
    if not refined_rows:
        return base_signals
    audio_by_t = {int(round(a.timestamp)): a for a in audio_signals}
    refined_by_t = {
        int(round(ts)): UnifiedSignal(
            timestamp=ts,
            audio=audio_by_t.get(int(round(ts))),
            vision=vision,
            ocr=ocr,
        )
        for ts, vision, ocr in refined_rows
    }
    merged = []
    seen = set()
    for sig in base_signals:
        key = int(round(sig.timestamp))
        merged.append(refined_by_t.get(key, sig))
        seen.add(key)
    for key, sig in refined_by_t.items():
        if key not in seen:
            merged.append(sig)
    merged.sort(key=lambda s: s.timestamp)
    return merged


def _merge_regions(regions, duration, gap=12.0, max_region=90.0):
    """Merge padded candidate ranges without letting one region sprawl forever."""
    if not regions:
        return []
    clipped = [
        (max(0.0, float(start)), min(float(duration), float(end)))
        for start, end in regions
        if end > start
    ]
    if not clipped:
        return []
    clipped.sort()
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap and max(prev_end, end) - prev_start <= max_region:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _regions_from_reaction_clips(
    reaction_clips,
    duration,
    pad_before=12.0,
    pad_after=8.0,
    max_refine_seconds=None,
):
    """Turn first-pass clips into generous regions for second-pass refinement."""
    if not reaction_clips:
        return []

    ranked = sorted(
        reaction_clips,
        key=lambda clip: (
            getattr(clip, "game_label", None) == "terminal_win",
            getattr(clip, "score", 0.0),
            getattr(clip, "reaction_auc", 0.0),
        ),
        reverse=True,
    )
    if max_refine_seconds is not None and ranked:
        # Smart Scan is a recall scout, not the final editor.  Spend most of
        # the bounded region budget on local standouts across the whole VOD so
        # a dense first hour cannot prevent a later moment from receiving face,
        # speech, and chat refinement.  Global leaders still receive a reserve.
        typical_region = 30.0
        slot_budget = max(1, int(max_refine_seconds / typical_region))
        temporal_slots = min(len(ranked), max(1, math.ceil(slot_budget * 2.0 / 3.0)))
        global_slots = min(len(ranked), max(1, slot_budget - temporal_slots))

        terminal = [c for c in ranked if getattr(c, "game_label", None) == "terminal_win"]
        chosen_ids = {id(c) for c in terminal}
        temporal = []
        if duration > 0.0 and temporal_slots > 0:
            bins = [[] for _ in range(temporal_slots)]
            for clip in ranked:
                if id(clip) in chosen_ids:
                    continue
                midpoint = 0.5 * (float(clip.start) + float(clip.end))
                bin_index = min(
                    temporal_slots - 1,
                    max(0, int((midpoint / duration) * temporal_slots)),
                )
                bins[bin_index].append(clip)
            for local_clips in bins:
                if local_clips:
                    temporal.append(local_clips[0])
                    chosen_ids.add(id(local_clips[0]))

        global_reserve = []
        for clip in ranked:
            if len(global_reserve) >= global_slots:
                break
            if id(clip) not in chosen_ids:
                global_reserve.append(clip)
                chosen_ids.add(id(clip))
        ranked = terminal + temporal + global_reserve + [
            clip for clip in ranked if id(clip) not in chosen_ids
        ]
    selected = []
    total = 0.0
    for clip in ranked:
        region = (clip.start - pad_before, clip.end + pad_after)
        candidate = _merge_regions(selected + [region], duration)
        candidate_total = sum(end - start for start, end in candidate)
        if max_refine_seconds is not None and candidate_total > max_refine_seconds and selected:
            continue
        selected.append(region)
        total = candidate_total
        if max_refine_seconds is not None and total >= max_refine_seconds:
            break

    return _merge_regions(selected, duration)


def _scout_candidates(curve, selected_clips):
    """Recover broad first-pass candidates for bounded perception refinement."""
    trace = getattr(curve, "selection_trace", None) or []
    if not trace:
        return list(selected_clips or [])
    candidates = []
    for row in trace:
        reasons = set(row.get("reasons") or [])
        if "semantic_reject" in reasons:
            continue
        peak = float(row.get("peak_value", 0.0) or 0.0)
        if peak < 0.30 and not row.get("strong_startle"):
            continue
        # Scout ranking intentionally ignores negative lobby/padding penalties;
        # refinement exists to collect the richer evidence needed to decide.
        scout_score = max(
            float(row.get("selection_score", 0.0) or 0.0),
            1.5 * peak,
        )
        candidates.append(SimpleNamespace(
            start=float(row.get("start", 0.0)),
            end=float(row.get("end", 0.0)),
            score=scout_score,
            reaction_auc=peak,
            game_label=row.get("game_label"),
        ))
    return candidates or list(selected_clips or [])


def _sequential_perceive(video_path, duration, settings, pipeline_settings, audio_signals, storage, emit):
    """Single-threaded perception loop (fallback when parallel perception is off/fails)."""
    audio_by_t = {int(round(a.timestamp)): a for a in audio_signals}
    unified_signals = []
    previous_motion_frame = None
    # Warm OCR before extract_frames opens a cv2.VideoCapture (Windows DLL
    # load-order: paddlex must init before OpenCV's ffmpeg DLL, else WinError
    # 1114 disables OCR for the run). See engines/ocr/ocr.py:warm_ocr.
    warm_ocr()
    for timestamp, frame_img in extract_frames(video_path, fps_sample_rate=1.0):
        if storage.debug_frames:
            storage.save_debug_frame(frame_img, f"frame_{timestamp:.3f}.jpg")

        if settings.get("facecamTracking", True):
            interval = max(1, int(pipeline_settings.get("facecam_scan_interval", 1)))
            should_scan_facecam = (
                interval <= 1
                or int(round(timestamp)) % interval == 0
                or not has_stable_facecam()
            )
            vision_signal = run_vision(
                frame_img,
                timestamp,
                detect_facecam=should_scan_facecam,
                stable_detections_required=pipeline_settings.get("stable_facecam_detections", 1),
            )
        else:
            vision_signal = VisionSignal(timestamp=timestamp, facecam_box=None,
                                         gameplay_box=[0.0, 0.0, 1.0, 1.0])

        motion_frame = prepare_motion_frame(frame_img, vision_signal.facecam_box if vision_signal else None)
        if vision_signal:
            vision_signal.motion_score = compute_motion_score(previous_motion_frame, motion_frame)
        previous_motion_frame = motion_frame

        if int(round(timestamp)) % pipeline_settings["ocr_stride"] == 0:
            ocr_signal = run_ocr(frame_img, timestamp, min_confidence=pipeline_settings["ocr_min_confidence"])
        else:
            ocr_signal = OCRSignal(timestamp=timestamp, text="", confidence=0.0)

        unified_signals.append(UnifiedSignal(
            timestamp=timestamp, audio=audio_by_t.get(int(round(timestamp))),
            ocr=ocr_signal, vision=vision_signal,
        ))
        progress_val = 0.15 + (0.25 * (timestamp / max(1.0, duration)))
        emit("Perception", f"Processed timestamp {timestamp:.1f}s", min(0.4, progress_val))
    return unified_signals

def emit_progress(phase: str, message: str, progress: float, callback=None, telemetry=None):
    if callback:
        if telemetry is None:
            callback(phase, message, progress)
        else:
            callback(phase, message, progress, telemetry)
    else:
        # Fallback print if no callback is provided
        print(f"[{phase}] {message} ({progress*100:.1f}%)")

def _extract_analysis_audio(video_path: str, wav_path: str, duration: float, emit=None):
    """Extract the analysis WAV, routing multitrack recordings to the mic-most
    stream (plan 22 §4.4). Selection is deterministic per file and entirely
    best-effort — any probe failure falls back to the default mix."""
    if os.path.exists(wav_path):
        return
    from engines.audio.audio_tracks import select_mic_track
    track = None
    try:
        track = select_mic_track(video_path, duration=duration)
    except Exception as exc:  # noqa: BLE001 - probe must never block a scan
        print(f"Mic-track probe skipped: {exc}")
    if track is not None and emit:
        emit("Perception",
             f"Multitrack recording: using audio track {track + 1} (mic) for reaction analysis.",
             0.09)
    extract_audio(video_path, output_path=wav_path, audio_track=track)


def _select_stable_facecam(unified_signals: List[UnifiedSignal]):
    facecam_boxes = [
        sig.vision.facecam_box
        for sig in unified_signals
        if sig.vision and sig.vision.facecam_box
    ]
    if not facecam_boxes:
        return None

    clusters = []
    for box in facecam_boxes:
        matched_cluster = None
        for cluster in clusters:
            if _calculate_iou(box, cluster["anchor"]) >= 0.35:
                matched_cluster = cluster
                break
        if matched_cluster:
            matched_cluster["boxes"].append(box)
        else:
            clusters.append({"anchor": box, "boxes": [box]})

    best_cluster = max(clusters, key=lambda cluster: len(cluster["boxes"]))
    boxes = best_cluster["boxes"]
    return [
        sorted(box[i] for box in boxes)[len(boxes) // 2]
        for i in range(4)
    ]

VISION_CACHE_VERSION = "vision-facecam-panel-v9-scene-layout"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _band(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "balanced"


SUPPORTED_GAMES = {"generic", "fortnite", "valorant", "apex", "cod", "minecraft", "gta"}


def _normalize_game_profile(value) -> str:
    game = str(value or "generic").strip().lower()
    aliases = {
        "default": "generic",
        "other": "generic",
        "apex legends": "apex",
        "apex_legends": "apex",
        "call of duty": "cod",
        "call_of_duty": "cod",
        "warzone": "cod",
        "gta v": "gta",
        "gta_v": "gta",
        "gta online": "gta",
        "grand theft auto": "gta",
    }
    game = aliases.get(game, game)
    return game if game in SUPPORTED_GAMES else "generic"


# Twitch category display names -> internal game id. Substring/keyword matching
# (not exact) because Twitch names vary ("Call of Duty: Warzone", "Grand Theft
# Auto V") and new titles in a series should still route to the right lexicon.
_TWITCH_GAME_KEYWORDS = (
    ("fortnite", "fortnite"),
    ("valorant", "valorant"),
    ("apex legends", "apex"),
    ("call of duty", "cod"),
    ("warzone", "cod"),
    ("modern warfare", "cod"),
    ("black ops", "cod"),
    ("grand theft auto", "gta"),
    ("minecraft", "minecraft"),
)


def _twitch_game_to_id(display_name) -> str:
    """Map a Twitch category display name to a supported game id (or generic)."""
    name = str(display_name or "").strip().lower()
    if not name:
        return "generic"
    for needle, game_id in _TWITCH_GAME_KEYWORDS:
        if needle in name:
            return game_id
    # Fall back to the exact-alias normalizer (handles bare "fortnite" etc.).
    return _normalize_game_profile(name)


def _segments_from_twitch_chapters(chapters, duration: float):
    """Convert Twitch GAME_CHANGE chapters into engine game segments.

    Ground-truth temporal segmentation from Twitch itself, replacing the OCR
    fingerprint sweep. Returns a list of engines.reaction.segments.GameSegment.
    """
    segs = []
    for ch in chapters or []:
        start = max(0.0, float(ch.get("start", 0.0)))
        end = float(ch.get("end", 0.0))
        if duration > 0:
            end = min(end, duration)
        if end <= start:
            continue
        segs.append(GameSegment(start, end, _twitch_game_to_id(ch.get("game"))))
    return segs


def build_scan_profile(settings: dict) -> dict:
    """Creator-facing scan profile plus engine-owned effective knobs.

    Sensitivity, deck yield, and clip boundaries are adaptive engine policy,
    not creator guesses. Only scan depth/performance choices remain settings.
    """
    sensitivity = DEFAULT_SENSITIVITY
    aggressiveness = DEFAULT_AGGRESSIVENESS
    processing_mode = settings.get("processingMode", settings.get("processing_mode", "quality"))
    game = _normalize_game_profile(settings.get("game", "generic"))
    fast_mode = processing_mode == "fast"

    # These bounds are broad scouting safeguards. Final boundaries are derived
    # from action onset, spoken setup/payoff, and reaction settle in the engine.
    profile = {
        "pre_roll": 2.0,
        "post_roll": 5.0,
        "story_min": 15.0,
        "story_max": 60.0,
        "label": "Adaptive clips",
    }

    return {
        "name": "Smart scan" if fast_mode else "Careful scan",
        "mode": processing_mode,
        "sensitivity_band": _band(sensitivity, 0.4, 0.75),
        "density_band": _band(aggressiveness, 0.35, 0.75),
        "clip_length": "adaptive",
        "clip_length_label": profile["label"],
        "game": game,
        "facecam_tracking": bool(settings.get("facecamTracking", True)),
        "details": {
            "sensitivity": round(sensitivity, 3),
            "aggressiveness": round(aggressiveness, 3),
            "audio_spike_threshold": round(_clamp(1.45 - sensitivity * 0.85, 0.55, 1.45), 3),
            # OCR every 3s by default: win/elimination banners persist several
            # seconds and OCR is corroborated by audio/voice/motion in fusion, so
            # 3 vs 2 costs ~no recall while cutting OCR frames by a third. Fast
            # mode stays coarser at 6. Override with ocrStride.
            "ocr_stride": max(1, int(settings.get(
                "ocrStride",
                FAST_SCOUT_OCR_STRIDE if fast_mode else DEFAULT_OCR_STRIDE,
            ))),
            "ocr_min_confidence": round(_clamp(0.62 - sensitivity * 0.32, 0.25, 0.62), 3),
            "story_min_peak_intensity": round(_clamp(0.68 - sensitivity * 0.28 - aggressiveness * 0.10, 0.30, 0.68), 3),
            "story_dropoff_threshold": round(_clamp(0.30 - aggressiveness * 0.16, 0.10, 0.30), 3),
            "include_facecam_emotion": not fast_mode and bool(settings.get("facecamTracking", True)),
            "facecam_scan_interval": max(1, int(settings.get("facecamScanIntervalSec", 10 if fast_mode else 2))),
            "refine_facecam_scan_interval": max(1, int(settings.get("refineFacecamScanIntervalSec", 2))),
            "stable_facecam_detections": max(1, int(settings.get("stableFacecamDetections", 3 if fast_mode else 1))),
            "max_refine_fraction": _clamp(float(settings.get("maxRefineFraction", 0.15)), 0.01, 1.0),
            "max_refine_seconds": max(30.0, float(settings.get("maxRefineSeconds", 1200.0))),
            "fast_speech_scout": bool(settings.get("fastSpeechScout", True)),
            "fast_speech_scout_seconds": max(60.0, float(settings.get("fastSpeechScoutSeconds", 600.0))),
            "game": game,
            "pre_roll": profile["pre_roll"],
            "post_roll": profile["post_roll"],
            "story_min": profile["story_min"],
            "story_max": profile["story_max"],
        },
    }


def _normalize_pipeline_settings(settings: dict) -> dict:
    scan_profile = build_scan_profile(settings)
    details = scan_profile["details"]
    return {
        "scan_profile": scan_profile,
        "pre_roll": details["pre_roll"],
        "post_roll": details["post_roll"],
        "story_min": details["story_min"],
        "story_max": details["story_max"],
        "processing_mode": scan_profile["mode"],
        "game": scan_profile["game"],
        "sensitivity": details["sensitivity"],
        "aggressiveness": details["aggressiveness"],
        "audio_spike_threshold": details["audio_spike_threshold"],
        "ocr_stride": details["ocr_stride"],
        "ocr_min_confidence": details["ocr_min_confidence"],
        "story_min_peak_intensity": details["story_min_peak_intensity"],
        "story_dropoff_threshold": details["story_dropoff_threshold"],
        "include_facecam_emotion": details["include_facecam_emotion"],
        "facecam_scan_interval": details["facecam_scan_interval"],
        "refine_facecam_scan_interval": details["refine_facecam_scan_interval"],
        "stable_facecam_detections": details["stable_facecam_detections"],
        "max_refine_fraction": details["max_refine_fraction"],
        "max_refine_seconds": details["max_refine_seconds"],
        "fast_speech_scout": details["fast_speech_scout"],
        "fast_speech_scout_seconds": details["fast_speech_scout_seconds"],
    }


def _auto_clip_count(duration: float, aggressiveness: float) -> int:
    """Backward-compatible wrapper around the shared production policy."""
    return auto_clip_count(duration, aggressiveness)


def _audio_cache_key(video_path: str) -> str:
    """Content-only key for extracted audio (plan 5.3) — see core.cache_keys.
    Shared with the API service so both sides derive identical paths."""
    return cache_keys.audio_cache_key(video_path)


def _video_cache_key(video_path: str, settings: dict) -> str:
    stat = os.stat(video_path)
    facecam_tracking = settings.get("facecamTracking", True)
    normalized = _normalize_pipeline_settings(settings)
    raw = (
        f"{VISION_CACHE_VERSION}|facecam={facecam_tracking}|"
        f"audio={normalized['audio_spike_threshold']:.3f}|ocr={normalized['ocr_min_confidence']:.3f}|"
        f"ocrstride={normalized['ocr_stride']}|ocrmax={get_ocr_max_side()}|"
        f"mode={normalized['processing_mode']}|"
        f"emotion={normalized['include_facecam_emotion']}|"
        f"facecaminterval={normalized['facecam_scan_interval']}|"
        f"stabledet={normalized['stable_facecam_detections']}|"
        f"{os.path.abspath(video_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    )
    return hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


def _transcript_cache_path(storage, key: str, model_size: str, fast: bool = False, settings: dict | None = None) -> str:
    """``key`` is the settings-independent audio key for full transcripts
    (a slider change must not re-transcribe; the API service rebuilds the
    same path from the video file alone), or the settings-dependent video
    hash for fast-mode scout transcripts (their regions come from settings).
    """
    _ = settings
    return os.path.join(
        storage.cache_dir,
        cache_keys.transcript_cache_name(
            key, asr_cache_label(model_size), signal_cache.CACHE_EXT, fast=fast,
        ),
    )

def _reaction_story_label(rc) -> str:
    """Map a reaction clip onto a caption-engine rule bucket."""
    # Non-gameplay clips (lobby / intermission) get their own bucket so the
    # review UI can filter them and captions don't call them gameplay reactions.
    if getattr(rc, "scene_label", None):
        return "non_gameplay"
    if rc.game_label == "terminal_win":
        return "high_intensity_moment"
    if rc.game_label in ("elimination", "knock"):
        return "gameplay_highlight"
    mb = rc.modality_breakdown or {}
    if mb.get("face", 0.0) > 0.2 or mb.get("voice", 0.0) > 0.45:
        return "reaction_moment"
    return "standard_gameplay"


def _cache_load(path):
    # Versioned JSON cache (core/signal_cache): any mismatch/corruption
    # returns None and the pipeline rebuilds. Legacy .pkl files are never
    # read (pickle is banned from the pipeline) — they just go stale on disk
    # until the cache-cleanup settings action removes them.
    return signal_cache.load(path)


def _reaction_timeline(curve, max_points: int = 2400, game_segments=None):
    """Compact, JSON-able R(t) summary for the UI's VOD-level reaction timeline.

    Downsamples the per-second curve to at most ``max_points`` samples (taking
    the max within each bucket so peaks survive) and normalizes to 0..1. Game
    OCR hits are kept as sparse labeled markers.
    """
    frames = getattr(curve, "frames", None)
    if not frames:
        return None
    score = [max(0.0, float(s)) for s in curve.score]
    peak = max(score) if score else 0.0
    if peak <= 0.0:
        return None
    step = max(1, math.ceil(len(frames) / max_points))
    t, r = [], []
    for i in range(0, len(frames), step):
        bucket = score[i:i + step]
        t.append(round(float(frames[i].timestamp), 1))
        r.append(round(max(bucket) / peak, 3))

    # Sparse game-evidence markers, collapsing repeats of the same label
    # within 10s (OCR banners persist across many consecutive frames).
    markers = []
    for f in frames:
        if not f.game_label or f.game_evidence < 0.4:
            continue
        if markers and markers[-1]["label"] == f.game_label \
                and f.timestamp - markers[-1]["t"] < 10.0:
            continue
        markers.append({"t": round(float(f.timestamp), 1), "label": f.game_label})

    # Labeled game timeline (plan 22 §5.1): non-generic segments only — the
    # UI renders these as named bands so a variety VOD is navigable by game.
    segments = [
        seg.to_dict() for seg in (game_segments or [])
        if getattr(seg, "game", "generic") != "generic"
    ]
    return {"version": 2, "t": t, "r": r, "game_markers": markers, "segments": segments}


def _report_judge_health(health: dict, event_callback=None) -> None:
    """Persist one creator-visible semantic judge health event per scan."""
    if not event_callback:
        return
    judged = int(health.get("judged_candidates", 0) or 0)
    fallbacks = int(health.get("fallback_candidates", 0) or 0)
    failures = int(health.get("batch_failures", 0) or 0)
    payload = dict(health)
    try:
        from core.device import get_torch_device
        payload.setdefault("device", get_torch_device())
    except Exception:  # noqa: BLE001 - health reporting must never block
        pass
    if payload.get("degraded"):
        message = (
            "Semantic judge degraded: "
            f"reviewed {judged} candidate(s), {fallbacks} used signal fallback, "
            f"{failures} batch failure(s)."
        )
        event_type = "semantic_judge_degraded"
    else:
        message = (
            "Semantic judge healthy: "
            f"reviewed {judged} candidate(s), {failures} recovered batch failure(s)."
        )
        event_type = "semantic_judge_health"
    event_callback(event_type, "Reaction", message, payload)


def _report_visual_judge_health(
    health: dict,
    event_callback=None,
    *,
    status: str = None,
    reason: str = None,
) -> None:
    """Persist one creator-visible visual judge health/skip event per scan."""
    if not event_callback:
        return
    payload = dict(health or {})
    try:
        from core.device import get_torch_device
        payload.setdefault("device", get_torch_device())
    except Exception:  # noqa: BLE001 - health reporting must never block
        pass
    resolved = status
    if resolved is None:
        if reason:
            resolved = "skipped"
        elif payload.get("degraded") or int(payload.get("failures", 0) or 0) > 0:
            resolved = "degraded"
        elif payload.get("status") == "disabled":
            resolved = "disabled"
        else:
            resolved = "healthy"
    payload["status"] = resolved
    if reason:
        payload["reason"] = reason
    judged = int(payload.get("judged_candidates", 0) or 0)
    failures = int(payload.get("failures", 0) or 0)
    if resolved == "skipped":
        payload["degraded"] = True
        message = f"Visual judge skipped: {reason or 'unavailable'}."
        event_type = "visual_judge_skipped"
    elif resolved == "degraded":
        payload["degraded"] = True
        message = (
            f"Vision review completed normally for {judged} moment"
            f"{'s' if judged != 1 else ''}; reaction and transcript fallback "
            f"covered {failures} moment{'s' if failures != 1 else ''}."
        )
        event_type = "visual_judge_degraded"
    elif resolved == "disabled":
        payload["degraded"] = False
        message = "Visual judge disabled in settings."
        event_type = "visual_judge_health"
    else:
        payload["degraded"] = False
        message = f"Visual judge healthy: reviewed {judged} candidate(s)."
        event_type = "visual_judge_health"
    event_callback(event_type, "Reaction", message, payload)


def _reaction_select(video_path, unified_signals, duration, settings,
                     pipeline_settings, storage, video_hash, emit, audio_signals=None,
                     cancel_check=None, framing_callback=None, event_callback=None):
    """Reaction Highlight Engine selection path.

    Returns (stories, clips, transcript, timeline) where timeline is the
    compact R(t) summary from _reaction_timeline (or None).

    Builds the multimodal reaction curve (voice prosody + facecam emotion + ASR
    hype + game evidence), selects clips via the reaction engine (optionally a
    learned ranker), and adapts the output to the legacy (stories, clips) contract
    with synthetic stories so caption/export/job_manager stay unchanged.
    """
    # Layout inference (layout-agnostic pass 1): find chat/alert overlay
    # regions from the OCR entry boxes already captured, then annotate every
    # OCR signal with spatially routed "gameplay_text". Game detection, OCR
    # event triggers, and scene classification all prefer the routed text, so
    # on-screen chat / alert widgets can't fire game triggers or misroute the
    # game fingerprint. Annotation is per-run (in memory), never cached.
    layout = build_layout_map(unified_signals)
    routed = annotate_gameplay_text(unified_signals, layout)
    if layout and layout.chat_regions:
        emit(
            "Reaction",
            f"Detected {len(layout.chat_regions)} on-screen overlay region(s); routing game text around them...",
            0.47,
        )
        print(f"Layout map: {len(layout.chat_regions)} chat-like region(s), "
              f"{layout.cells_flagged} cells over {layout.frames_sampled} frames; "
              f"{routed} OCR frames had overlay text filtered.")

    # Game detection (no user selection). Prefer Twitch's OWN VOD metadata --
    # the authoritative game + a GAME_CHANGE chapter timeline -- over
    # fingerprinting the HUD from OCR. OCR detection needs 2+ distinct HUD
    # phrases read cleanly and silently falls back to "generic" on many real
    # VODs (e.g. a Fortnite VOD whose banner text was never OCR'd twice), which
    # then misroutes OCR event triggers, the hype lexicon, and scene labeling.
    # The metadata probe is a ~1s TwitchDownloaderCLI call that works in any scan
    # mode. Overwriting pipeline_settings["game"] in place means every downstream
    # consumer (hype lexicon, scene/lobby classification, OCR event triggers in
    # reaction_main) picks up the game with no further plumbing. Falls back to
    # OCR fingerprinting for non-Twitch sources or when the CLI is unavailable.
    source_url = settings.get("source_url", "")
    vod_meta = None
    if twitch_chat.is_twitch_vod(source_url) and twitch_chat.chat_available():
        try:
            vod_meta = twitch_chat.fetch_twitch_video_meta(
                source_url, cancel_check=cancel_check, temp_dir=storage.temp_dir,
            )
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - metadata is best-effort
            print(f"Twitch VOD metadata skipped: {exc}")

    if vod_meta and vod_meta.get("created_at") and event_callback:
        from core.source_date import normalize_source_date

        source_date = normalize_source_date(vod_meta.get("created_at"))
        if source_date:
            event_callback(
                "source_metadata",
                "Reaction",
                "Recording date recovered from Twitch VOD metadata.",
                {"source_date": source_date},
            )

    game_segments = []
    if vod_meta and (vod_meta.get("game") or vod_meta.get("chapters")):
        chapters = vod_meta.get("chapters") or []
        primary = vod_meta.get("game") or (chapters[0]["game"] if chapters else None)
        pipeline_settings["game"] = _twitch_game_to_id(primary)
        game_segments = _segments_from_twitch_chapters(chapters, duration)
        if pipeline_settings["game"] != "generic":
            emit("Reaction", f"Recognized {primary} from Twitch's VOD info...", 0.48)
        else:
            # Honest copy (plan 22 §3.5): a game with no tuned profile is fine —
            # the reaction engine reads the streamer, not the game.
            emit(
                "Reaction",
                f"Twitch lists this as “{primary}” — no tuned profile yet, "
                "so clips come from your voice, face, and chat reactions.",
                0.48,
            )
    else:
        detection = detect_game(unified_signals)
        pipeline_settings["game"] = _normalize_game_profile(detection.game)
        if detection.game != "generic":
            emit(
                "Reaction",
                f"Recognized {detection.game.title()} from the recording ({detection.confidence*100:.0f}% confident)...",
                0.48,
            )
        else:
            emit(
                "Reaction",
                "Game not recognized — clips still come from your voice, face, and chat reactions.",
                0.48,
            )
        try:
            game_segments = detect_game_segments(unified_signals)
        except Exception as exc:  # noqa: BLE001 - segmentation must never block a scan
            print(f"Game segmentation skipped: {exc}")
            game_segments = []

    # Temporal segmentation (plan 22 §4.1): variety streams switch games (and
    # scenes) mid-VOD. Whether segments came from Twitch chapters or the OCR
    # sweep, label the timeline so OCR triggers + scene lexicons follow the game
    # actually on screen, and refine the chat-overlay layout per segment.
    try:
        seg_index = SegmentIndex(game_segments, default=pipeline_settings["game"])
        if len(seg_index.distinct_games) > 1:
            timeline_text = " · ".join(
                f"{seg.game} {_fmt_hms(seg.start)}–{_fmt_hms(seg.end)}"
                for seg in game_segments if seg.game != "generic"
            )
            emit("Reaction", f"Game timeline: {timeline_text}", 0.485)
            print(f"Game segments: {[s.to_dict() for s in game_segments]}")
            # Per-segment layout maps: re-annotate each segment's OCR routing
            # with a map built from that segment alone (falls back to the
            # global annotation when a segment is too sparse to judge).
            for seg in game_segments:
                seg_slice = [s for s in unified_signals if seg.start <= s.timestamp < seg.end]
                seg_layout = build_layout_map(seg_slice)
                if seg_layout and seg_layout.chat_regions:
                    annotate_gameplay_text(seg_slice, seg_layout)
    except Exception as exc:  # noqa: BLE001 - segmentation must never block a scan
        print(f"Game timeline annotation skipped: {exc}")

    # Audio (persistent cache, shared with prosody + ASR). Keyed on the source
    # file only, so it survives a settings change (plan 5.3).
    if cancel_check:
        cancel_check()
    wav_path = os.path.join(storage.cache_dir, f"{_audio_cache_key(video_path)}_audio.wav")
    _extract_analysis_audio(video_path, wav_path, duration, emit=emit)

    emit("Reaction", "Analyzing voice prosody...", 0.5)
    if cancel_check:
        cancel_check()
    prosody_frames = extract_prosody(wav_path)

    # Laughter/scream burst track (ONNX audio tagger). Cached per VOD; a
    # missing model or any tagger failure just drops the channel — the fusion
    # renormalizes over whatever is present, like face/ASR.
    audio_event_frames = None
    if audio_events_available():
        try:
            # _v3: adds model-independent sudden-loudness onsets for startles
            # (v2 added game_intensity). Old caches decode safely via defaults,
            # but would silently leave the new channel dark, so retire them.
            ev_cache = os.path.join(storage.cache_dir, f"{video_hash}_events_v3{signal_cache.CACHE_EXT}")
            audio_event_frames = _cache_load(ev_cache)
            if audio_event_frames is None:
                emit("Reaction", "Listening for laughter and screams...", 0.52)
                if cancel_check:
                    cancel_check()
                audio_event_frames = analyze_audio_events(wav_path, duration=duration)
                signal_cache.save(ev_cache, audio_event_frames)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - optional channel; never block selection
            print(f"Audio-event channel skipped: {exc}")
            audio_event_frames = None

    stable_facecam = _select_stable_facecam(unified_signals)
    fast_mode = pipeline_settings.get("processing_mode") == "fast"

    # Facecam emotion (cached; re-decodes frames on a cache miss).
    face_frames = None
    if settings.get("facecamTracking", True) and stable_facecam and not fast_mode:
        face_cache = os.path.join(storage.cache_dir, cache_keys.face_cache_name(
            video_hash, face_scorer_label(), signal_cache.CACHE_EXT))
        face_frames = _cache_load(face_cache)
        if face_frames is None:
            emit("Reaction", "Reading facecam emotion...", 0.55)
            per_frame = {
                int(round(s.timestamp)): s.vision.facecam_box
                for s in unified_signals if s.vision and s.vision.facecam_box
            }
            face_frames = analyze_faces(video_path, facecam_box=stable_facecam,
                                        per_frame_boxes=per_frame)
            signal_cache.save(face_cache, face_frames)

    # ASR hype (cached transcript; one VOD-level Whisper pass). Fast mode skips
    # this first; captions can still transcribe selected clips downstream.
    asr_frames = None
    transcript = None
    whisper_model_size = resolve_whisper_size(settings)
    if not fast_mode:
        try:
            tr_cache = _transcript_cache_path(
                storage, _audio_cache_key(video_path), whisper_model_size, settings=settings,
            )
            transcript = _cache_load(tr_cache)
            if transcript is None:
                emit("Reaction", f"Transcribing audio ({asr_display_name()})...", 0.6)
                transcript = transcribe_vod(
                    wav_path,
                    model_size=whisper_model_size,
                    cancel_check=cancel_check,
                    progress_callback=_asr_progress_reporter(emit, 0.60, 0.64),
                    settings=settings,
                )
                signal_cache.save(tr_cache, transcript)
                # Hand the card back before the VLM judge runs. ASR is done for
                # this VOD -- the per-clip caption pass reloads it much later,
                # and a reload is ~9s. MEASURED 2026-08-19: leaving Qwen3-ASR
                # plus the forced aligner resident (~12 GB) through the judge
                # pinned VRAM at 15.9/16.4 GB and dropped the judge from 4-5
                # windows/min to 1-2 -- the same paging collapse MiniCPM hit.
                # Whisper never provoked this because medium.en is ~1.5 GB.
                release_asr_models()
            # Language honesty (plan 22 §3.6-lite): speech analysis (hype
            # lexicon + quote titles) is English-first. Say so instead of
            # silently degrading — the other channels still carry selection.
            lang = (transcript or {}).get("language")
            if lang and lang != "en":
                emit(
                    "Reaction",
                    f"This VOD sounds like '{lang}' — speech analysis is English-first today, "
                    "so clips lean on voice, face, chat, and game signals instead.",
                    0.62,
                )
            lexicon = load_lexicon(
                os.path.join(get_resource_dir(), "configs", "hype_lexicon.json"),
                game=pipeline_settings.get("game", "generic"),
            )
            asr_frames = build_hype_track(transcript, lexicon, duration)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - ASR is optional; never block selection
            print(f"ASR hype skipped: {exc}")

    # Twitch chat velocity (plan 9.1). Optional crowd-reaction channel — only
    # for Twitch VODs, only when TwitchDownloaderCLI is resolvable, and cached
    # per source (settings-independent). The fetch is time-boxed and cancellable
    # (twitch_chat.fetch_twitch_chat) so this optional channel can never block a
    # scan; any failure/timeout just drops it.
    chat_frames = None
    chat_evidence = None
    source_url = settings.get("source_url", "")
    if not fast_mode and twitch_chat.is_twitch_vod(source_url) and twitch_chat.chat_available():
        chat_cache = os.path.join(
            storage.cache_dir,
            f"{_audio_cache_key(video_path)}_chat_evidence_v1{signal_cache.CACHE_EXT}",
        )
        cached = _cache_load(chat_cache)
        if isinstance(cached, dict) and "track" in cached:
            chat_frames = cached.get("track") or None
            chat_evidence = cached.get("evidence")
        else:
            emit("Reaction", "Reading Twitch chat reactions...", 0.57)
            try:
                messages = twitch_chat.fetch_twitch_chat(
                    source_url, cancel_check=cancel_check, temp_dir=storage.temp_dir,
                )
                track = build_chat_track(messages, duration) if messages else []
                chat_evidence = (
                    build_chat_evidence(messages, duration, source_scope="full_vod")
                    if messages else None
                )
                signal_cache.save(chat_cache, {
                    "track": track,
                    "evidence": chat_evidence,
                })
                chat_frames = track or None
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001 - chat is optional; never block selection
                print(f"Chat channel skipped: {exc}")

    ranker, ranker_diagnostics = resolve_active_ranker()
    # No clip_min/clip_max: boundaries are content-driven (arc onset -> payoff
    # -> settle). The engine's wide sanity invariants apply. Crucially, this
    # final selector budget goes through the shared production resolver so the
    # review cap cannot drift away from the actual pipeline again.
    reaction_settings = build_pipeline_reaction_settings(
        duration,
        pipeline_settings,
        settings,
        [segment.to_dict() for segment in game_segments],
        fast_mode=fast_mode,
    )
    reaction_settings["second_look_source_key"] = canonical_source_key(
        source_url or video_path,
        video_path,
    )
    recall_region_plan = settings.get("recallRegionPlan") or {}
    creator_markers = list(recall_region_plan.get("markers") or [])
    if creator_markers:
        reaction_settings["creator_markers"] = creator_markers
        if event_callback:
            event_callback(
                "recall_session_plan",
                "Reaction",
                f"Recall Session supplied {len(creator_markers)} creator marker(s).",
                {
                    "session_id": recall_region_plan.get("session_id"),
                    "marker_count": len(creator_markers),
                    "region_count": len(recall_region_plan.get("regions") or []),
                    "alignment_status": recall_region_plan.get("alignment_status"),
                },
            )

    if fast_mode:
        emit("Reaction", "Smart scan: finding candidate regions...", 0.58)
        first_pass_settings = dict(reaction_settings)
        first_pass_settings["max_clips"] = max(12, reaction_settings["max_clips"] * 2)
        first_pass_curve, first_pass_clips = reaction_main.build_reaction_clips(
            unified_signals, prosody_frames, face_frames=None,
            asr_hype_frames=None, audio_event_frames=audio_event_frames,
            settings=first_pass_settings, ranker=None,
        )
        max_refine_seconds = min(
            pipeline_settings["max_refine_seconds"],
            max(60.0, duration * pipeline_settings["max_refine_fraction"]),
        )
        regions = _regions_from_reaction_clips(
            _scout_candidates(first_pass_curve, first_pass_clips),
            duration,
            max_refine_seconds=max_refine_seconds,
        )
        recall_regions = [
            (float(region["start"]), float(region["end"]))
            for region in recall_region_plan.get("regions") or []
            if isinstance(region, dict)
            and region.get("start") is not None
            and region.get("end") is not None
        ]
        regions = merge_priority_regions(
            recall_regions, regions, duration, max_refine_seconds,
        )
        if regions:
            if pipeline_settings.get("fast_speech_scout", True):
                scout_regions = []
                scout_total = 0.0
                scout_budget = pipeline_settings["fast_speech_scout_seconds"]
                for start, end in regions:
                    region_duration = max(0.0, end - start)
                    if region_duration <= 0:
                        continue
                    if scout_total + region_duration > scout_budget and scout_regions:
                        continue
                    scout_regions.append((start, end))
                    scout_total += region_duration
                    if scout_total >= scout_budget:
                        break
                if scout_regions:
                    speech_started = time.monotonic()
                    try:
                        speech_cache = _transcript_cache_path(
                            storage,
                            video_hash,
                            whisper_model_size,
                            fast=True,
                            settings=settings,
                        )
                        transcript = _cache_load(speech_cache)
                        if transcript is None:
                            emit(
                                "Reaction",
                                f"Smart scan: reading speech in {scout_total/60.0:.1f} min of likely moments...",
                                0.61,
                            )
                            transcript = transcribe_vod_regions(
                                wav_path,
                                scout_regions,
                                model_size=whisper_model_size,
                                cancel_check=cancel_check,
                                progress_callback=_asr_progress_reporter(emit, 0.61, 0.64),
                                settings=settings,
                            )
                            signal_cache.save(speech_cache, transcript)
                        lexicon = load_lexicon(
                            os.path.join(get_resource_dir(), "configs", "hype_lexicon.json"),
                            game=pipeline_settings.get("game", "generic"),
                        )
                        asr_frames = build_hype_track(transcript, lexicon, duration)
                    except InterruptedError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - speech scout should never fail the run
                        print(f"Smart scan speech scout skipped: {exc}")
                    print(f"Smart scan stage timing: speech={time.monotonic() - speech_started:.1f}s")
            refined_seconds = sum(end - start for start, end in regions)
            emit(
                "Perception",
                f"Refining {len(regions)} candidate regions ({refined_seconds/60.0:.1f} min budgeted)...",
                0.45,
            )
            if audio_signals is None:
                audio_signals = analyze_audio_spikes(
                    wav_path,
                    window_duration_sec=1.0,
                    spike_threshold=pipeline_settings["audio_spike_threshold"],
                )
            n_workers = int(settings.get(
                "perceptionWorkers",
                parallel_perception.recommended_workers(settings.get("performanceProfile")),
            ))
            # Planned AFTER the perceptionWorkers override so a user-pinned
            # count is still VRAM-capped before workers reach for the GPU.
            n_workers, gpu_ocr = parallel_perception.plan_perception(n_workers, settings)
            refined_rows, face_frames = parallel_perception.perceive_regions(
                video_path, regions, settings, pipeline_settings,
                n_workers=max(1, min(n_workers, len(regions))), emit=emit,
                gpu_ocr=gpu_ocr,
            )
            if audio_signals is not None:
                unified_signals = _merge_refined_signals(
                    unified_signals, refined_rows, audio_signals
                )
                stable_facecam = _select_stable_facecam(unified_signals)
                # Refined signals carry fresh OCR entries without routing
                # annotations -- re-apply the layout map so their text is
                # routed consistently with the first pass.
                annotate_gameplay_text(unified_signals, layout)

            # --- Smart scan: full perception on the scouted regions (HUMAN_CLIPS
            # plan, Package A). Recover the channels fast mode skipped up front
            # -- facecam emotion and Twitch chat -- but only inside the candidate
            # regions, so the final pass fuses every signal where it matters
            # without paying whole-VOD cost. Region-scoped artifacts cache on
            # (video/audio key, region-set hash): identical regions on a re-scan
            # reuse them; changed regions can't serve stale data.
            region_hash = cache_keys.region_set_hash(regions)

            # Facecam emotion, region-scoped. Prefer the stable-box analyze_faces
            # track (the same channel quality mode feeds fusion) over the
            # per-frame worker faces perceive_regions returned; those remain the
            # fallback if this pass fails or finds nothing. Timestamps outside
            # the regions get no FaceFrame — fusion treats them as face-absent.
            per_frame_boxes = {
                int(round(s.timestamp)): s.vision.facecam_box
                for s in unified_signals if s.vision and s.vision.facecam_box
            }
            if settings.get("facecamTracking", True) and (stable_facecam or per_frame_boxes):
                face_started = time.monotonic()
                try:
                    face_cache = os.path.join(
                        storage.cache_dir,
                        cache_keys.face_cache_name(
                            video_hash, face_scorer_label(),
                            signal_cache.CACHE_EXT, region_hash=region_hash),
                    )
                    region_faces = _cache_load(face_cache)
                    if region_faces is None:
                        emit("Reaction", "Smart scan: reading facecam in likely moments...", 0.62)
                        if cancel_check:
                            cancel_check()
                        region_faces = analyze_faces(
                            video_path, facecam_box=stable_facecam,
                            per_frame_boxes=per_frame_boxes, regions=regions,
                        )
                        signal_cache.save(face_cache, region_faces)
                    if region_faces:
                        face_frames = region_faces
                except InterruptedError:
                    raise
                except Exception as exc:  # noqa: BLE001 - optional channel; never block the scan
                    print(f"Smart scan face pass skipped: {exc}")
                print(f"Smart scan stage timing: face={time.monotonic() - face_started:.1f}s")

            # Twitch chat, region-scoped. Same availability guards, time-box, and
            # cancellation as the quality-mode full fetch; messages are
            # VOD-relative so the merged per-region track feeds fusion unchanged.
            if twitch_chat.is_twitch_vod(source_url) and twitch_chat.chat_available():
                chat_started = time.monotonic()
                chat_cache = os.path.join(
                    storage.cache_dir,
                    f"{_audio_cache_key(video_path)}_chat_evidence_v1_r{region_hash}"
                    f"{signal_cache.CACHE_EXT}",
                )
                cached = _cache_load(chat_cache)
                if isinstance(cached, dict) and "track" in cached:
                    chat_frames = cached.get("track") or None
                    chat_evidence = cached.get("evidence")
                else:
                    emit("Reaction", "Smart scan: reading Twitch chat in likely moments...", 0.63)
                    try:
                        messages = twitch_chat.fetch_twitch_chat_regions(
                            source_url, regions,
                            cancel_check=cancel_check, temp_dir=storage.temp_dir,
                        )
                        track = build_chat_track(messages, duration) if messages else []
                        chat_evidence = (
                            build_chat_evidence(
                                messages, duration, source_scope="smart_regions",
                            )
                            if messages else None
                        )
                        signal_cache.save(chat_cache, {
                            "track": track,
                            "evidence": chat_evidence,
                        })
                        chat_frames = track or None
                    except InterruptedError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - chat is optional; never block the scan
                        print(f"Smart scan chat pass skipped: {exc}")
                print(f"Smart scan stage timing: chat={time.monotonic() - chat_started:.1f}s")
        else:
            emit("Reaction", "Smart scan found no candidate regions.", 0.62)

    # Anonymous chat phrases and counts are durable Memory evidence, while the
    # source messages remain ephemeral. Persist before selection so a later
    # optional judge/export failure cannot erase evidence the scan already read.
    if chat_evidence is not None:
        try:
            chat_job_id = str(
                settings.get("job_id") or pipeline_settings.get("job_id") or "",
            )
            if chat_job_id:
                write_chat_evidence_artifact(chat_job_id, chat_evidence)
        except Exception as exc:  # noqa: BLE001 - evidence capture never blocks selection
            print(f"Chat evidence artifact skipped: {exc}")

    # Resolve the VLM lane BEFORE the text judge. Both judges run in the same
    # scan, and a natively-multimodal model answers text-only prompts on the
    # instance already loaded for vision — so the text judge borrows it rather
    # than loading a second ~2.5 GB GGUF. The backend is lazy (weights load on
    # first generate), so constructing it here costs nothing when neither judge
    # ends up running. VisualJudgeSession below receives this same object.
    vlm_pair = None
    discovery_error = None
    shared_vlm_backend = None
    try:
        vlm_pair = visual_judge_mod.resolve_models()
    except Exception as exc:  # noqa: BLE001 - discovery must never block a scan
        print(f"Visual judge discovery skipped: {exc}")
        discovery_error = str(exc)[:300]
    if vlm_pair:
        shared_vlm_backend = visual_judge_mod.VisualLlamaBackend(
            vlm_pair[0], vlm_pair[1],
        )

    # Local semantic judge (HUMAN_CLIPS Package C): resolved once per run when
    # a model is reachable. Transcript is useful but
    # not required: OCR/chat and disciplined nonverbal signals can carry scares,
    # visual punchlines, and reaction noises with no ASR words. The engine
    # receives a bound callable so it stays model-free like ``ranker``; any
    # failure inside is swallowed there -- a scan never blocks.
    semantic_judge = None
    judge_model_path = semantic_backend.resolve_model()
    if judge_model_path or shared_vlm_backend is not None:
        # Either source enables the text judge: its own GGUF, or the vision
        # lane's shared instance. Gating on models/llm alone would silently
        # disable it on a build that ships only the VLM.
        if shared_vlm_backend is not None:
            judge_backend = visual_judge_mod.TextRoleBackend(shared_vlm_backend)
            print(
                "Semantic judge sharing the vision instance "
                f"({os.path.basename(vlm_pair[0])}); models/llm not loaded."
            )
        else:
            judge_backend = semantic_backend.LlamaBackend(judge_model_path)
        judge_game = pipeline_settings.get("game", "generic")
        judge_transcript = transcript
        judge_chat = chat_frames
        scan_judge_health = {}

        def semantic_judge(judge_candidates_list):
            emit("Reaction", "Local model is reading candidate transcripts...", 0.64)
            judge_started = time.monotonic()
            judge_health = {}
            try:
                verdicts = semantic_judge_mod.judge_candidates(
                    judge_candidates_list, judge_transcript,
                    chat_frames=judge_chat,
                    game_context=judge_game if judge_game != "generic" else None,
                    backend=judge_backend,
                    cancel_check=cancel_check,
                    health=judge_health,
                )
                semantic_judge_mod.accumulate_health(scan_judge_health, judge_health)
                _report_judge_health(scan_judge_health, event_callback)
                if verdicts:
                    emit(
                        "Reaction",
                        f"Local model reviewed {len(verdicts)} candidate moment(s).",
                        0.645,
                    )
                else:
                    emit(
                        "Reaction",
                        "Local model produced no usable transcript verdicts; using signal scoring.",
                        0.645,
                    )
                return verdicts
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001 - judge is optional
                judge_health.update({
                    "candidates_received": len(judge_candidates_list),
                    "eligible_candidates": len(judge_candidates_list),
                    "judged_candidates": 0,
                    "fallback_candidates": len(judge_candidates_list),
                    "batch_attempts": 0,
                    "batch_failures": 1,
                    "generation_attempts": 0,
                    "generation_failures": 0,
                    "context_overflows": 0,
                    "invalid_outputs": 0,
                    "aborted": True,
                    "degraded": True,
                    "error": str(exc)[:500],
                })
                semantic_judge_mod.accumulate_health(scan_judge_health, judge_health)
                _report_judge_health(scan_judge_health, event_callback)
                print(f"Semantic judge skipped: {exc}")
                return {}
            finally:
                print(f"Semantic judge stage timing: judge={time.monotonic() - judge_started:.1f}s")

    # Visual judge + outcome sweep (content axis) is a standard best-effort
    # lane when a local VLM pair exists under models/vlm. It was measured
    # 2026-07-20 on 12 founder VODs
    # (win-priority tighten): recall +0.05, precision +0.03, p@5 +0.02 vs
    # arousal-only. Degrades silently to arousal-only if the model is missing
    # or fails. Verdicts cache per window.
    visual_judge = None
    visual_session = None
    outcome_sweep = None
    # Discovery and the backend were resolved above so the text judge can
    # borrow the same instance; this block only builds the visual session.
    if vlm_pair or discovery_error:
        if vlm_pair:
            vlm_model, vlm_mmproj = vlm_pair
            # Facecam tiles stay off on the 4B judge (measured miss). Re-wire
            # them here when the omni experiment lands so it sees a real face.
            visual_session = visual_judge_mod.VisualJudgeSession(
                video_path,
                transcript,
                chat_frames=chat_frames,
                game_context=(
                    pipeline_settings.get("game")
                    if pipeline_settings.get("game", "generic") != "generic" else None
                ),
                backend=shared_vlm_backend,
                model_path=vlm_model,
                mmproj_path=vlm_mmproj,
                max_candidates=40,
                cancel_check=cancel_check,
            )

            def visual_judge(judge_candidates_list):
                emit("Reaction", "Local vision model is reviewing likely moments...", 0.645)
                started = time.monotonic()
                try:
                    return visual_session(judge_candidates_list)
                finally:
                    print(
                        "Visual judge stage timing: "
                        f"visual={time.monotonic() - started:.1f}s "
                        f"health={visual_session.health()}"
                    )

            # Deck-coverage in reaction_main calls ensure_capacity on this
            # callable when the pre-selection pool has exhausted remaining.
            # Bind the session methods onto the wrapper — a bare nested
            # function otherwise silently judges 0 winners (2026-07-22).
            visual_judge.ensure_capacity = visual_session.ensure_capacity
            visual_judge.session = visual_session

            # The outcome sweep (win banners OCR can't read) shares the
            # session's backend and frame extractor — one model resident.
            sweep_session = visual_judge_mod.OutcomeSweep(
                video_path,
                backend=visual_session.backend,
                frame_extractor=visual_session.frame_extractor,
                cancel_check=cancel_check,
            )

            def outcome_sweep(sweep_duration):
                emit("Reaction", "Local vision model is scanning for match outcomes...", 0.64)
                started = time.monotonic()
                try:
                    return sweep_session(sweep_duration)
                finally:
                    print(
                        "Outcome sweep stage timing: "
                        f"sweep={time.monotonic() - started:.1f}s "
                        f"health={sweep_session.health()}"
                    )
            outcome_sweep.stride_sec = sweep_session.stride_sec
        elif discovery_error:
            _report_visual_judge_health(
                {"error": discovery_error},
                event_callback,
                status="skipped",
                reason="vision model discovery failed",
            )
    else:
        _report_visual_judge_health(
            {},
            event_callback,
            status="skipped",
            reason="vision model files not found",
        )

    emit("Reaction", "Scoring reactions and selecting clips...", 0.65)
    if cancel_check:
        cancel_check()
    # ``transcript`` is whichever ASR pass this mode ran: the full-VOD pass in
    # quality mode, the region-scout transcript in fast mode, or None when ASR
    # was skipped/failed -- sentence-boundary snapping (Package B) degrades to
    # today's silence/cut snapping in that last case.
    # The VLM release lives in a finally: these weights are ~3.4 GB plus a
    # ~0.9 GB vision tower, and a cancelled or failed reaction stage used to
    # skip the close entirely and strand all of it in the long-lived backend
    # process for as long as the app stayed open.
    try:
        curve, reaction_clips = reaction_main.build_reaction_clips(
            unified_signals, prosody_frames, face_frames=face_frames,
            asr_hype_frames=asr_frames, audio_event_frames=audio_event_frames,
            chat_frames=chat_frames,
            settings=reaction_settings, ranker=ranker,
            transcript=transcript,
            semantic_judge=semantic_judge,
            visual_judge=visual_judge,
            outcome_sweep=outcome_sweep,
            ranker_diagnostics=ranker_diagnostics,
        )
    finally:
        if visual_session is not None:
            try:
                _report_visual_judge_health(visual_session.health(), event_callback)
            except Exception:  # noqa: BLE001 - health reporting must never block
                pass
            try:
                visual_session.close()
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass
        elif shared_vlm_backend is not None:
            # Session construction failed after the backend was built, but the
            # text judge may still have loaded weights through it. Nothing else
            # owns the instance in that path, so release it here.
            try:
                shared_vlm_backend.close()
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass

    # Persist the fused curve that candidate generation thresholds. Everything
    # downstream of detect_arcs is already captured in candidates.v1.json, but
    # a moment that never crossed theta leaves no row anywhere, so without this
    # the cost of the generation threshold cannot be measured after the fact.
    # Best-effort: a diagnostics artifact must never fail a scan.
    try:
        curve_job_id = str(
            settings.get("job_id") or pipeline_settings.get("job_id") or "",
        )
        if curve_job_id and curve is not None:
            write_reaction_curve_artifact(
                curve_job_id,
                [frame.timestamp for frame in curve.frames],
                list(curve.score),
            )
    except Exception as exc:  # noqa: BLE001 - diagnostics capture is never fatal
        print(f"Reaction curve artifact skipped: {exc}")

    # Opt-in candidate trace. The app writes this through job_manager, which
    # guarantees one write per job; a direct run_pipeline call bypasses that
    # path entirely and leaves no trace, so an offline experiment has nothing
    # to diff or count rescues from. Only fires when a caller asks, so the
    # production path keeps its single write.
    if settings.get("persist_candidate_trace"):
        try:
            from core.artifacts import write_candidates_artifact
            trace_job_id = str(
                settings.get("job_id") or pipeline_settings.get("job_id") or "",
            )
            trace = list(getattr(curve, "selection_trace", None) or [])
            if trace_job_id and trace:
                write_candidates_artifact(trace_job_id, trace)
                print(f"Candidate trace: {len(trace)} rows -> {trace_job_id}")
        except Exception as exc:  # noqa: BLE001 - never fatal
            print(f"Candidate trace skipped: {exc}")

    # Gameplay viewport detection (layout-agnostic pass 2): on windowed
    # layouts (game capture + chat panel + widgets) export crops must frame
    # the game itself, not the whole stream canvas. None => fullscreen, the
    # existing center-crop behavior. Best-effort: never blocks a scan.
    # Whole-VOD facecam layout model (engines/vision/facecam.py): cluster every
    # frame's box into persistent, spatially-stable, plausible layouts. This
    # rejects transient false positives (a game character / HUD portrait the
    # per-frame tracker snapped to) and supports streamers who use more than one
    # cam position -- each clip is framed with the layout its own window matches.
    facecam_layout_model = facecam_mod.detect_facecam_layout_model(unified_signals)
    facecam_layouts = facecam_layout_model.layouts
    if facecam_layouts:
        # Snap each layout to the live camera feed inside the detected box:
        # tracker boxes routinely overshoot into the static canvas around the
        # plate, which rendered as a squarish PiP with edge bleed.
        facecam_layouts = facecam_mod.refine_layout_boxes(
            video_path, facecam_layouts, duration,
        )
        # Then measure where the subject sits inside each portrait plate. Runs
        # after refinement because the measurement is a fraction of the plate's
        # FINAL height; without it the cam crop guesses which half of a tall
        # plate to keep and clips the face.
        facecam_layouts = facecam_mod.measure_subject_tops(
            video_path, facecam_layouts, duration,
        )
        facecam_layout_model.layouts = facecam_layouts
    if facecam_layouts:
        summary = ", ".join(
            f"{int(round(L.fraction * 100))}% of VOD" for L in facecam_layouts
        )
        print(f"Facecam layouts: {len(facecam_layouts)} ({summary})")
        if len(facecam_layouts) > 1:
            emit("Reaction", f"Detected {len(facecam_layouts)} facecam layouts; framing each clip to the one on screen.", 0.66)

    # Authored canvases can switch geometry between intro/chat/game scenes.
    # Keep a timestamped set of structural content windows and resolve again
    # at each selected clip's peak. The legacy motion detector remains a
    # fallback for borderless windowed captures.
    gameplay_layout_model = None
    gameplay_layouts = []
    fallback_gameplay_box = None
    try:
        gameplay_layout_model = detect_gameplay_layout_model(video_path, duration)
        gameplay_layouts = gameplay_layout_model.layouts
        if not gameplay_layouts:
            fallback_gameplay_box = detect_gameplay_box(
                video_path, duration, facecam_box=stable_facecam,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Gameplay region detection skipped: {exc}")
    if gameplay_layouts:
        emit(
            "Reaction",
            f"Detected {len(gameplay_layouts)} authored stream layout(s); framing each clip to its scene.",
            0.66,
        )
        print("Gameplay layouts: " + ", ".join(
            f"{layout.box} ({layout.support} samples)" for layout in gameplay_layouts
        ))
    elif fallback_gameplay_box:
        emit("Reaction", "Windowed game capture detected; exports will frame the game viewport.", 0.66)
        print(f"Gameplay viewport: {fallback_gameplay_box}")

    stories, clips = [], []
    requested_export_layout = settings.get("exportLayout", "auto")
    resolved_export_layout = resolve_export_layout(
        requested_export_layout,
        reaction_clips,
        facecam_layouts=facecam_layouts,
        game=pipeline_settings.get("game"),
    )
    if requested_export_layout == "auto" and reaction_clips:
        label = "gameplay PiP" if resolved_export_layout == "gameplay_pip" else "stacked split"
        print(f"Auto facecam layout: {label}")
        emit("Reaction", f"Auto framing selected {label} for this VOD.", 0.66)

    # VTuber is an identity choice, never a facecam-shape or title heuristic.
    # The title marker is advisory in the import UI only. A per-session creator
    # confirmation is the sole gate; absent/unknown always keeps normal framing.
    scan_job_id = str(settings.get("job_id") or pipeline_settings.get("job_id") or "orphan")
    session_title = (
        (vod_meta or {}).get("title")
        or settings.get("session_name")
        or ""
    )
    vtuber_confirmed = settings.get("vtuberMode") == "confirmed"
    vtuber_layouts = facecam_layouts
    if vtuber_confirmed:
        confirmed_avatar_layouts = vtuber_mod.detect_confirmed_avatar_layouts(unified_signals)
        if confirmed_avatar_layouts:
            vtuber_layouts = confirmed_avatar_layouts
    vtuber_model = vtuber_mod.prepare_vtuber_model(
        video_path,
        session_title,
        vtuber_layouts,
        duration,
        os.path.join(get_data_dir(), "jobs", scan_job_id),
        confirmed=vtuber_confirmed,
        cancel_check=cancel_check,
    )
    if vtuber_model.get("status") == "confirmed":
        count = len(vtuber_model.get("overlays") or [])
        print(f"VTuber layout confirmed: {count} avatar plate(s)")
        emit(
            "Reaction",
            "VTuber identity confirmed; preparing the top-center cutout layout.",
            0.665,
        )
    elif settings.get("vtuberMode") == "confirmed":
        print(
            "VTuber layout not applied: "
            f"{vtuber_model.get('status', 'uncertain')} (normal facecam retained)"
        )

    # Manual selections made later in the VOD Editor need the same framing
    # evidence as automatic picks. Persist only the compact geometry model:
    # representative facecam boxes, timestamped detections, authored gameplay
    # layouts, and the VOD-stable composition choice. Raw vision frames stay in
    # the normal cache and never enter the UI payload.
    if framing_callback:
        framing_model = {
            "version": 2,
            "resolved_layout": resolved_export_layout,
            "facecam_layouts": [
                {
                    "box": list(item.box),
                    "support": int(item.support),
                    "fraction": float(item.fraction),
                    "stability": float(item.stability),
                    # Persisted because _dominant_fallback gates on it: without
                    # it the VOD Editor rebuilds these layouts with coverage 0.0
                    # and drops the facecam on windows the scan kept it for.
                    "coverage": float(item.coverage),
                    "panel_support": int(item.panel_support),
                    "fallback_support": int(item.fallback_support),
                    # Portrait plates only, and absent when the person detector
                    # could not hold the subject. The VOD Editor must reproduce
                    # the scan's cam framing rather than re-guess it.
                    "subject_top": item.subject_top,
                }
                for item in facecam_layouts
            ],
            "facecam_observations": [
                [round(float(signal.timestamp), 3), list(signal.vision.facecam_box)]
                for signal in unified_signals
                if getattr(signal, "vision", None) is not None
                and getattr(signal.vision, "facecam_box", None)
            ],
            # Unlike the raw v1 list above, this includes explicit fullscreen /
            # unresolved samples. Manual clips must not inherit a corner crop
            # merely because it was dominant elsewhere in the VOD.
            "facecam_timeline": [
                {
                    "timestamp": round(float(item.timestamp), 3),
                    "layout_index": item.layout_index,
                    "state": item.state,
                    "source": item.source,
                }
                for item in facecam_layout_model.observations
            ],
            "gameplay_layouts": [
                {
                    "box": list(item.box),
                    "timestamps": [float(value) for value in item.timestamps],
                    "support": int(item.support),
                    "confidence": float(item.confidence),
                }
                for item in gameplay_layouts
            ],
            "gameplay_observations": [
                {
                    "timestamp": float(item.timestamp),
                    "box": list(item.box) if item.box is not None else None,
                }
                for item in (getattr(gameplay_layout_model, "observations", None) or [])
            ],
            "fallback_gameplay": list(fallback_gameplay_box) if fallback_gameplay_box else None,
            "vtuber": vtuber_model,
        }
        try:
            framing_callback(framing_model)
        except Exception as exc:  # noqa: BLE001 - framing persistence is best-effort
            print(f"Could not persist VOD framing model: {exc}")

    def _gameplay_box_for_window(start: float, end: float, peak_timestamp=None):
        clip_gameplay_box = None
        try:
            local_gameplay_box = detect_gameplay_box_at(
                video_path,
                peak_timestamp if peak_timestamp is not None else ((start + end) / 2.0),
            )
            # A one-frame rectangle never overrides the VOD model. It must
            # match a recurring authored layout; otherwise the explicit scene
            # timeline decides whether this clip is windowed or fullscreen.
            clip_gameplay_box = match_gameplay_layout(
                gameplay_layouts, local_gameplay_box,
            )
        except Exception:  # noqa: BLE001 - nearest sampled layout remains safe
            clip_gameplay_box = None
        if clip_gameplay_box is None:
            if gameplay_layout_model is None:
                clip_gameplay_box = gameplay_for_window(
                    gameplay_layouts, start, end,
                )
            else:
                clip_gameplay_box = gameplay_for_window(
                    gameplay_layouts,
                    start,
                    end,
                    observations=gameplay_layout_model.observations,
                )
            clip_gameplay_box = clip_gameplay_box or fallback_gameplay_box
        return clip_gameplay_box

    def _layout_for_window(start: float, end: float, peak_timestamp=None):
        facecam_box = facecam_mod.facecam_for_window(
            facecam_layouts,
            unified_signals,
            start,
            end,
            observations=facecam_layout_model.observations,
            at=peak_timestamp,
            layout_verifier=lambda box, timestamp: facecam_mod.is_facecam_layout_at(
                video_path, timestamp, box,
            ),
        )
        gameplay_box = _gameplay_box_for_window(start, end, peak_timestamp)
        camera_focus_x = None
        if facecam_box is None:
            probe_time = peak_timestamp if peak_timestamp is not None else (start + end) / 2.0
            camera_focus_x = facecam_mod.fullframe_camera_focus_at(
                video_path, probe_time, camera_region=gameplay_box,
            )
        resolved = assign_layout(
            facecam_box,
            export_layout=resolved_export_layout,
            gameplay_box=gameplay_box,
            fullframe_camera=camera_focus_x is not None,
            camera_focus_x=camera_focus_x,
            facecam_subject_top=facecam_mod.subject_top_for_box(
                facecam_layouts, facecam_box),
        )
        if requested_export_layout == "auto" and resolved.get("facecam"):
            overlay = vtuber_mod.overlay_for_facecam(vtuber_model, facecam_box)
            if overlay is not None:
                resolved = vtuber_mod.apply_overlay(resolved, overlay)
        return resolved

    for rc in reaction_clips:
        story_id = f"s_{uuid.uuid4().hex[:8]}"
        stories.append(GameStory(
            story_id=story_id, start=rc.start, end=rc.end, events=[],
            score=rc.score, label=_reaction_story_label(rc),
        ))
        clips.append(GameClip(
            clip_id=f"c_{uuid.uuid4().hex[:8]}",
            start=rc.start, end=rc.end, story_id=story_id, score=rc.score,
            # Facecam framed per clip window from the VOD layout model: the clip
            # gets the layout its own window matches, a wandered/character window
            # falls back to the real persistent cam, and no-cam windows go
            # full-frame -- so framing never jumps to a game character.
            layout=_layout_for_window(rc.start, rc.end, rc.peak_timestamp),
            reason=rc.reason, peak_timestamp=rc.peak_timestamp,
            modality_breakdown=rc.modality_breakdown, reaction_auc=rc.reaction_auc,
            scene_label=getattr(rc, "scene_label", None),
            features=rc.features,
            hook_score=getattr(rc, "hook_score", 0.0),
            # Semantic judge fields (Package C): flow through so the caption
            # engine can prefer the judge's human title over a quote.
            semantic_title=getattr(rc, "semantic_title", None),
            semantic_hook_line=getattr(rc, "semantic_hook_line", None),
            semantic_moment_type=getattr(rc, "semantic_moment_type", None),
            semantic_verdict=getattr(rc, "semantic_verdict", None),
            # Carry the live marker through, or the moment the creator marked
            # is indistinguishable from one Recall found on its own.
            creator_protected=bool(getattr(rc, "creator_protected", False)),
            recall_marker_ids=list(getattr(rc, "recall_marker_ids", None) or []),
            recall_marker_time=getattr(rc, "recall_marker_time", None),
        ))

    # Ceiling-cut moments get the same framed preview encode as the primary
    # deck during this scan. Theater Review only reveals them later — no
    # on-demand FFmpeg wait when the creator opens More moments.
    #
    # The arousal floor is partitioned out first: those cards are rejected at
    # ~13:1 and would cost a preview encode each for a card nobody reviews.
    # They stay in candidates.v1.json, so the training lane can still reach
    # them through ClipService.deferred_more_candidates.
    job_id = scan_job_id
    ceiling_review, ceiling_deferred = partition_second_look(
        ceiling_candidates_from_trace(
            getattr(curve, "selection_trace", None),
            job_id=job_id,
            limit=MORE_CANDIDATE_LIMIT,
        )
    )
    if ceiling_deferred:
        print(
            f"Second-look floor: holding back {len(ceiling_deferred)} of "
            f"{len(ceiling_review) + len(ceiling_deferred)} ceiling cards"
        )
    for candidate in ceiling_review:
        start = float(candidate["start"])
        end = float(candidate["end"])
        peak = candidate.get("peak_timestamp", start)
        features = candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
        breakdown = (
            candidate.get("modality_breakdown")
            if isinstance(candidate.get("modality_breakdown"), dict)
            else {}
        )
        hook_score = max(0.0, min(1.0, float(features.get("hook_score", 0.0) or 0.0)))
        if hook_score <= 0:
            hook_score = max(0.0, min(1.0, float(features.get("peak_value", 0.0) or 0.0)))
        story_id = f"s_{uuid.uuid4().hex[:8]}"
        shadow_audit = bool(candidate.get("second_look_shadow_primary_audit"))
        shadow_weights = candidate.get("shadow_primary_weights") or []
        audit_weight_text = ", ".join(
            f"{float(weight):.2f}" for weight in shadow_weights
        )
        stories.append(GameStory(
            story_id=story_id, start=start, end=end, events=[],
            score=hook_score,
            label=(
                "Shadow challenger review" if shadow_audit else "Second-look moment"
            ),
        ))
        clips.append(GameClip(
            clip_id=candidate.get("id") or more_candidate_clip_id(job_id, start, end, peak),
            # Preserve the score that actually ordered this candidate.  Hook
            # strength remains a separate 0..1 presentation signal.
            start=start, end=end, story_id=story_id,
            score=float(candidate.get("selection_score", 0.0) or 0.0),
            layout=_layout_for_window(start, end, peak),
            reason=(
                (
                    "Shadow challenger audit: this would enter Primary at "
                    f"personal weight {audit_weight_text}; review it before "
                    "personalization can change future decks."
                ) if shadow_audit else (
                    "This moment was just outside the first 15 and stayed available "
                    "for a second look."
                )
            ),
            peak_timestamp=peak,
            modality_breakdown=breakdown,
            reaction_auc=float(features.get("reaction_auc", 0.0) or 0.0),
            scene_label=candidate.get("scene_label"),
            features=features,
            hook_score=hook_score,
            # A marked moment demoted to second look is still a marked moment.
            creator_protected=bool(candidate.get("creator_protected")),
            recall_marker_ids=list(candidate.get("recall_marker_ids") or []),
            recall_marker_time=candidate.get("recall_marker_time"),
            semantic_verdict=candidate.get("semantic_verdict"),
            review_tier=REVIEW_TIER_SECOND_LOOK,
        ))

    timeline = _reaction_timeline(curve, game_segments=game_segments)
    # Per-candidate audit rows (selection_trace + feature dicts) ride ONLY
    # under a private key: core/job_manager pops it into
    # data/jobs/<job_id>/candidates.v1.json BEFORE the timeline is persisted to
    # reaction_timelines or emitted on the frontend event stream — feature rows
    # are training capture, not UI payload. A None timeline (flat/empty curve)
    # carries nothing; that case has no candidates worth persisting.
    trace = getattr(curve, "selection_trace", None)
    if timeline is not None and trace:
        timeline["_selection_trace"] = trace
        timeline["_selection_diagnostics"] = dict(
            getattr(curve, "ranker_diagnostics", None) or {}
        )
    return stories, clips, transcript, timeline


def run_pipeline(video_path: str, settings: dict = None, progress_callback=None,
                 clip_callback=None, timeline_callback=None, cancel_check=None,
                 framing_callback=None, event_callback=None):
    """Execute the full Recall pipeline.

    clip_callback, when given, is invoked once per selected clip as soon as
    selection finishes — long before captions/export — so the UI's found
    shelf can fill in while the slow tail of the pipeline runs.
    """
    if settings is None:
        settings = {}

    def _check_cancel():
        if cancel_check:
            cancel_check()
    pipeline_settings = _normalize_pipeline_settings(settings)
    pipeline_started = time.perf_counter()
    stage_marks = []
        
    storage = StorageManager(
        delete_temp=settings.get("delete_temp", True),
        debug_frames=settings.get("debug_frames", False),
        debug_vision=settings.get("debug_vision", False),
        debug_ocr=settings.get("debug_ocr", False)
    )
    storage.init_workspace()
    
    # Helper to pass callback easily
    def _emit(phase, msg, prog, telemetry=None):
        emit_progress(phase, msg, prog, progress_callback, telemetry=telemetry)

    def _mark(label, started_at):
        elapsed = time.perf_counter() - started_at
        stage_marks.append((label, elapsed))
        msg = f"{label} completed in {elapsed:.1f}s"
        print(msg)
        return time.perf_counter()
        
    try:
        reset_vision_state()
        video_info = extract_metadata(video_path)
        fps = float(video_info.get("r_frame_rate", "30/1").split("/")[0])
        duration = float(video_info.get("duration", 0.0))
        width = float(video_info.get("width", 1280.0))
        height = float(video_info.get("height", 720.0))
        
        meta_msg = f"Video metadata: {{'fps': {fps}, 'duration': {duration}, 'width': {width}, 'height': {height}}}"
        print(meta_msg)
        _emit("Perception", meta_msg, 0.05)
        device_msg = f"Device summary: {device_summary()}"
        print(device_msg)
        _emit("Perception", device_msg, 0.06)
        
        # Check cache
        video_hash = _video_cache_key(video_path, settings)
        cache_path = os.path.join(storage.cache_dir, f"{video_hash}_signals{signal_cache.CACHE_EXT}")

        stage_started = time.perf_counter()
        unified_signals = signal_cache.load(cache_path)
        if unified_signals is not None:
            msg = "Found cached signals. Skipping heavy perception phase..."
            print(msg)
            _emit("Perception", msg, 0.4)
            stage_started = _mark("Perception cache load", stage_started)
        else:
            # Audio to a persistent cache path, reused by prosody/ASR downstream.
            # Keyed on the source file only (plan 5.3) so it's shared across
            # settings and with the reaction path's identical key.
            wav_path = os.path.join(storage.cache_dir, f"{_audio_cache_key(video_path)}_audio.wav")
            msg = "Extracting audio..."
            print(msg)
            _emit("Perception", msg, 0.1)
            _extract_analysis_audio(video_path, wav_path, duration, emit=_emit)
            stage_started = _mark("Audio extraction", stage_started)

            print("Analyzing audio spikes...")
            audio_started = time.perf_counter()
            audio_signals = analyze_audio_spikes(
                wav_path,
                window_duration_sec=1.0,
                spike_threshold=pipeline_settings["audio_spike_threshold"]
            )
            _mark("Audio spike analysis", audio_started)

            # --- Parallel perception (handbook/20): saturate the CPU. Falls back
            #     to the sequential loop if it fails. Folds emotion in (one decode).
            unified_signals = None
            face_frames_cache = None
            use_parallel = settings.get("parallelPerception", True) and duration > 0
            if use_parallel:
                try:
                    n_workers = int(settings.get(
                        "perceptionWorkers",
                        parallel_perception.recommended_workers(settings.get("performanceProfile")),
                    ))
                    # Planned AFTER the perceptionWorkers override so a user-pinned
                    # count is still VRAM-capped before workers reach for the GPU.
                    n_workers, gpu_ocr = parallel_perception.plan_perception(
                        n_workers, settings,
                    )
                    msg = (
                        f"Perceiving in parallel across {n_workers} "
                        f"{'GPU' if gpu_ocr else 'CPU'} workers..."
                    )
                    print(msg)
                    _emit("Perception", msg, 0.15)
                    rows, face_frames_cache = parallel_perception.perceive_parallel(
                        video_path, duration, settings, pipeline_settings, n_workers,
                        _emit, gpu_ocr=gpu_ocr,
                    )
                    unified_signals = _rows_to_signals(rows, audio_signals)
                except parallel_perception.WorkerStallError:
                    # A worker hung for half an hour — the sequential loop would
                    # hit the same pathological section hours later. Fail the job
                    # with the diagnosable stall message instead.
                    raise
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    print(f"Parallel perception failed; falling back to sequential: {exc}")
                    unified_signals = None

            if unified_signals is None:
                msg = "Extracting frames and processing vision/OCR (sequential)..."
                print(msg)
                _emit("Perception", msg, 0.15)
                unified_signals = _sequential_perceive(
                    video_path, duration, settings, pipeline_settings,
                    audio_signals, storage, _emit
                )

            signal_cache.save(cache_path, unified_signals)
            if face_frames_cache is not None:
                signal_cache.save(
                    os.path.join(storage.cache_dir, cache_keys.face_cache_name(
                        video_hash, face_scorer_label(), signal_cache.CACHE_EXT)),
                    face_frames_cache,
                )
            _mark("Perception", stage_started)

        msg = "Perception execution complete. Selecting clips..."
        print(msg)
        _emit("Event", msg, 0.45)

        stories = None
        clips = None
        vod_transcript = None
        reaction_timeline = None

        # --- REACTION HIGHLIGHT ENGINE (handbook/19) ---
        # The legacy Event->Story->Clip path was removed. A reaction-engine
        # failure now surfaces loudly (InterruptedError still cancels) instead of
        # silently falling back to low-quality rule-selected clips.
        reaction_started = time.perf_counter()
        stories, clips, vod_transcript, reaction_timeline = _reaction_select(
            video_path, unified_signals, duration, settings,
            pipeline_settings, storage, video_hash, _emit,
            audio_signals=audio_signals if "audio_signals" in locals() else None,
            cancel_check=_check_cancel,
            framing_callback=framing_callback,
            event_callback=event_callback,
        )
        primary_clips = [
            clip for clip in clips
            if getattr(clip, "review_tier", None) != REVIEW_TIER_SECOND_LOOK
        ]
        second_look_clips = [
            clip for clip in clips
            if getattr(clip, "review_tier", None) == REVIEW_TIER_SECOND_LOOK
        ]
        # Publish the real R(t) curve as soon as reaction selection produces it.
        # Captions and review-preview renders still run after this point, so the
        # processing screen can become a genuinely live evidence surface.
        if reaction_timeline and timeline_callback:
            timeline_callback(reaction_timeline)
        _mark("Reaction selection", reaction_started)
        msg = (
            f"Reaction engine selected {len(primary_clips)} clips"
            + (f" (+{len(second_look_clips)} second-look)." if second_look_clips else ".")
        )
        print(msg)
        _emit("Clip", msg, 0.7)

        # Selection is final here — announce the primary deck before the slow
        # caption/export tail starts. Second-look stays off the live shelf.
        if clip_callback:
            for c in primary_clips:
                _check_cancel()
                try:
                    clip_callback(c)
                except Exception:
                    traceback.print_exc()

        msg = "Clip execution complete. Moving to Caption Engine..."
        print(msg)
        _emit("Caption", msg, 0.75)
        
        # --- PHASE 6: CAPTION ENGINE ---
        # Both tiers caption here. Second-look used to skip this pass, which
        # left its baked previews with no burned-in subtitles and a generic
        # "Second-look moment" title -- while the same clip revealed through
        # ClipService.prepare_preview (the legacy on-demand path) came out
        # captioned. Running one pass over both tiers is what makes More
        # moments look like the rest of the deck.
        caption_started = time.perf_counter()
        captions = caption_main.process_clips_to_captions(
            primary_clips + second_look_clips, stories, video_path, storage.temp_dir,
            transcript=vod_transcript,
            cancel_check=_check_cancel, caption_style=settings.get("captionStyle"),
            # Detected game id -> lead hashtag on every clip's post metadata.
            game=pipeline_settings.get("game", "generic"),
        )
        _mark("Caption generation", caption_started)
        
        msg = f"Generated {len(captions)} clip captions."
        print(msg)
        _emit("Caption", msg, 0.8)
        
        preview_count = len(primary_clips) + len(second_look_clips)
        msg = (
            f"Caption execution complete. Rendering {preview_count} review previews..."
            if second_look_clips
            else "Caption execution complete. Rendering review previews..."
        )
        print(msg)
        _emit("Export", msg, 0.85)

        # --- PHASE 5: EXPORT ENGINE (preview only) ---
        # Primary deck + Second-look ceiling cuts share this cheap low-res
        # proxy pass. Final-quality rendering stays on demand for exports
        # (ClipService.ensure_rendered). Baking Second-look here means More
        # moments is instant in Theater Review.
        def _export_cb(ev):
            if progress_callback:
                progress_callback(ev.get("phase", "Export"), ev.get("message", ""), ev.get("progress", 0.0))
            else:
                print(f"[{ev.get('phase', 'Export')}] {ev.get('message', '')} ({ev.get('progress', 0.0)*100:.1f}%)")

        export_started = time.perf_counter()
        exported_paths = export_main.export_clips(
            video_path,
            primary_clips + second_look_clips,
            output_dir=storage.outputs_dir,
            progress_callback=_export_cb,
            captions=captions,
            cancel_check=_check_cancel,
            preview=True,
        )
        _mark("Export", export_started)

        msg = f"Rendered {len(exported_paths)} review previews."
        print(msg)
        _emit("Export", msg, 0.95)

        msg = "Pipeline execution complete."
        print(msg)
        print("Pipeline timings:", {label: round(seconds, 1) for label, seconds in stage_marks})
        _emit("Complete", msg, 1.0)
    
    # Finally block ensures cleanup even on failure
    finally:
        storage.cleanup_temp()
        
    # The timestamped transcript is also a durable product artifact: Stream
    # Memory indexes it after clip persistence. Returning it here avoids a
    # second ASR pass and keeps indexing outside the selection/export path.
    return (
        unified_signals, stories, clips, captions, exported_paths, duration,
        reaction_timeline, vod_transcript,
    )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Avatar matting for the VTuber cutout.

A 2D avatar's alpha, frame by frame, from an anime-trained segmentation model
(SkyTNT anime-segmentation ISNet, models/vision/isnetis.onnx, Apache-2.0). It
replaces the GrabCut still mask, which kept whatever background sat behind the
hair when it was cut and dropped dark hair against dark scenery.

- :func:`alpha` mattes one crop.
- :func:`still_mask` builds the RGBA plate the editor previews with: a few
  moments across the stream, merged, so hair that sways stays inside.
- :func:`write_avatar_video` mattes every frame of a clip's window into an
  RGBA avatar video the render lays over the game (ffmpeg_builder). Colours
  are cleaned where the avatar is see-through (thin hair over grass would
  otherwise carry the grass's green): see :func:`foreground`.

Nothing here decides *whether* to cut out an avatar: that stays the creator's
explicit choice (engines/vision/vtuber.py).
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import Callable, Optional, Sequence

import numpy as np

from core.bundle_paths import get_models_dir

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("RECALL_AVATAR_MATTING_MODEL", "isnetis.onnx")
INPUT_SIZE = 1024
# Per-frame matting rate: every frame on a GPU, a sampled rate on the CPU
# (about 0.6 s a frame there); the render holds each mask until the next.
GPU_FPS_CAP = 30.0
CPU_FPS_CAP = 6.0
# Enclosed holes up to this share of the crop are filled (see refine()).
SMALL_HOLE = 0.015
# ...and only when the model half-kept them (mean raw alpha at least this).
HALF_KEPT = 0.2

_session = None
_lock = threading.Lock()


def model_path() -> str:
    if os.path.isabs(MODEL_NAME):
        return MODEL_NAME
    return os.path.join(get_models_dir(), "vision", MODEL_NAME)


def available() -> bool:
    return os.path.isfile(model_path())


def _half_precision_copy(path: str) -> Optional[str]:
    """A half-precision copy of the model for the GPU, made once and cached.

    About 30% faster on tensor cores, alpha identical to within 1e-4; the
    shipped file stays the upstream bytes. Inputs and outputs stay float32.
    """
    from core.bundle_paths import get_data_dir

    import hashlib

    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    tag = f"{os.path.getsize(path)}-{hashlib.sha1(head).hexdigest()[:12]}"
    cached = os.path.join(get_data_dir(), "cache", "models", f"isnetis_fp16_{tag}.onnx")
    if os.path.isfile(cached):
        return cached
    try:
        import onnx
        from onnxruntime.transformers import float16

        os.makedirs(os.path.dirname(cached), exist_ok=True)
        half = float16.convert_float_to_float16(onnx.load(path), keep_io_types=True)
        onnx.save(half, cached + ".tmp")
        os.replace(cached + ".tmp", cached)
        return cached
    except Exception:
        logger.warning("Couldn't make the half-precision matting model; using float32", exc_info=True)
        return None


def _ensure_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            from core.device import get_ort_providers

            path = model_path()
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Avatar matting model not found at {path}.")
            providers = get_ort_providers()
            if providers[0] != "CPUExecutionProvider":
                path = _half_precision_copy(path) or path
            _session = ort.InferenceSession(path, providers=providers)
        return _session


def on_gpu() -> bool:
    session = _ensure_session()
    return session.get_providers()[0] != "CPUExecutionProvider"


def reset_session() -> None:
    global _session
    with _lock:
        _session = None


def refine(raw: np.ndarray) -> np.ndarray:
    """Make a model alpha ready to key with.

    The model leaves faint see-through patches inside the avatar (a pale face
    reads as half background) and a soft floor of noise outside it. A gentle
    curve pushes both to their ends, and small half-kept holes the avatar
    encloses are filled; the soft edge around the silhouette is kept.
    """
    import cv2

    a = np.clip((raw.astype(np.float32) - 0.08) / 0.8, 0.0, 1.0)
    solid = (a > 0.5).astype(np.uint8)
    if solid.any():
        # Flood the outside from a padded border; whatever it can't reach and
        # isn't solid is a hole inside the avatar.
        padded = np.pad(solid, 1)
        flood = padded.copy()
        mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 1)
        holes = (flood[1:-1, 1:-1] == 0).astype(np.uint8)
        # Only patches the model half-kept (a pale face reads ~0.3-0.5), and
        # only small ones. A gap it called background (~0: the game between a
        # twin tail and the neck) is real and stays see-through, however small.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
        limit = SMALL_HOLE * a.size
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] > limit:
                continue
            region = labels == label
            if float(raw[region].mean()) >= HALF_KEPT:
                a[region] = 1.0
    return a


def _blur_fusion(image, fg, bg, a, radius):
    import cv2

    k = (radius, radius)
    blurred_a = cv2.blur(a, k)[..., None]
    blurred_f = cv2.blur(fg * a[..., None], k) / (blurred_a + 1e-5)
    blurred_b = cv2.blur(bg * (1.0 - a[..., None]), k) / ((1.0 - blurred_a) + 1e-5)
    a3 = a[..., None]
    fg = blurred_f + a3 * (image - a3 * blurred_f - (1.0 - a3) * blurred_b)
    return np.clip(fg, 0.0, 1.0), blurred_b


def foreground(bgr: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The avatar's own colours where it's see-through.

    A see-through pixel is avatar and background mixed. Keying it shows that
    mix over the new background, so thin hair over grass stays green. This is
    blur-fusion foreground estimation (Germer et al., 2021, "Fast Multi-Level
    Foreground Estimation"): a wide pass, then a fine one. Only see-through
    pixels change, so it runs at half size (a wide pass at an eighth) and is
    copied back onto them: ~9 ms a frame against ~27 at full size, colours
    within half a step of 255. Returns uint8 BGR.
    """
    import cv2

    h, w = a.shape
    band = a < 0.98
    if not band.any():
        return bgr.copy()
    hw, hh = max(1, w // 2), max(1, h // 2)
    image = cv2.resize(bgr, (hw, hh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    a_half = cv2.resize(a, (hw, hh), interpolation=cv2.INTER_AREA)
    sw, sh = max(1, w // 8), max(1, h // 8)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    small_a = cv2.resize(a_half, (sw, sh), interpolation=cv2.INTER_AREA)
    fg, bg = _blur_fusion(small, small, small, small_a, 12)
    fg = cv2.resize(fg, (hw, hh), interpolation=cv2.INTER_LINEAR)
    bg = cv2.resize(bg, (hw, hh), interpolation=cv2.INTER_LINEAR)
    fg, _ = _blur_fusion(image, fg, bg, a_half, 5)
    fg = cv2.resize((fg * 255.0).round().astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    out = bgr.copy()
    out[band] = fg[band]
    return out


def _prepare(bgr: np.ndarray):
    """Letterbox a crop into the model's 1024x1024 input (contiguous NCHW:
    a strided view makes ONNX Runtime copy it on every run)."""
    import cv2

    h0, w0 = bgr.shape[:2]
    s = INPUT_SIZE
    h, w = (s, max(1, int(s * w0 / h0))) if h0 > w0 else (max(1, int(s * h0 / w0)), s)
    ph, pw = s - h, s - w
    small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    x = np.zeros((1, 3, s, s), np.float32)
    # BGR -> RGB planes, scaled to [0, 1], straight into place.
    for plane, channel in enumerate((2, 1, 0)):
        x[0, plane, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = small[:, :, channel] * (1.0 / 255.0)
    return x, (h0, w0, h, w, ph, pw)


def _infer(x: np.ndarray) -> np.ndarray:
    session = _ensure_session()
    return session.run(None, {session.get_inputs()[0].name: x})[0][0, 0]


def _finish(out: np.ndarray, meta) -> np.ndarray:
    import cv2

    h0, w0, h, w, ph, pw = meta
    out = out[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    return refine(cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR))


def alpha(bgr: np.ndarray) -> np.ndarray:
    """The avatar's alpha in ``bgr`` (an avatar crop), float32 in [0, 1]."""
    x, meta = _prepare(bgr)
    return _finish(_infer(x), meta)


def _crop_px(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = (float(v) for v in box[:4])
    x0, y0 = int(round(max(0.0, x) * width)), int(round(max(0.0, y) * height))
    x1 = int(round(min(1.0, x + w) * width))
    y1 = int(round(min(1.0, y + h) * height))
    return x0, y0, max(x0 + 2, x1), max(y0 + 2, y1)


def still_mask(
    video_path: str,
    box: Sequence[float],
    timestamps: Sequence[float],
    output_path: str,
) -> bool:
    """Write the editor's RGBA avatar plate for ``box``: the merged alpha of a
    few moments (hair that moves stays in), over the middle moment's pixels."""
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return False
    crops, alphas = [], []
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return False
        x0, y0, x1, y1 = _crop_px(box, width, height)
        for t in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            roi = frame[y0:y1, x0:x1]
            if roi.size == 0 or min(roi.shape[:2]) < 32:
                continue
            crops.append(roi)
            alphas.append(alpha(roi))
    finally:
        capture.release()
    if not alphas:
        return False
    merged = np.max(np.stack(alphas), axis=0)
    if float(merged.mean()) < 0.05:
        return False  # nothing avatar-like in the box
    rgba = cv2.cvtColor(crops[len(crops) // 2], cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (merged * 255.0).round().astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    return bool(cv2.imwrite(output_path, rgba))


def write_avatar_video(
    video_path: str,
    box: Sequence[float],
    start: float,
    end: float,
    output_path: str,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Optional[float]:
    """Matte every frame of [start, end] inside ``box`` into an RGBA avatar video.

    Each frame is the avatar's cleaned colour (:func:`foreground`) with its
    alpha. Returns the video's frame rate, or None if it couldn't be made (the
    render then keys with the still plate). The window is decoded by ffmpeg
    with the same ``-ss`` seek the render uses, so mask frame N is clip frame
    N; OpenCV's seek lands seconds off on long VODs.
    """
    import cv2
    from core.ffmpeg_path import get_ffmpeg_path

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        src_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    finally:
        capture.release()
    if width <= 0 or height <= 0 or end <= start:
        return None

    decoder = encoder = None
    cancelled = False

    def check() -> None:
        nonlocal cancelled
        try:
            cancel_check()
        except BaseException:
            cancelled = True
            raise

    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        x0, y0, x1, y1 = _crop_px(box, width, height)
        cw, ch = x1 - x0, y1 - y0
        cap_fps = GPU_FPS_CAP if on_gpu() else CPU_FPS_CAP
        mask_fps = src_fps / max(1, int(round(src_fps / min(src_fps, cap_fps))))
        sample = f",fps={mask_fps:.6f}" if mask_fps < src_fps - 0.01 else ""
        decoder = subprocess.Popen(
            [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
                "-ss", str(start), "-t", str(end - start), "-i", video_path,
                # Crop after converting to BGR: on YUV, crop rounds to even
                # sizes and every frame would come out a few bytes off.
                "-an", "-vf", f"format=bgr24,crop={cw}:{ch}:{x0}:{y0}{sample}",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=no_window,
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        encoder = subprocess.Popen(
            [
                get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{cw}x{ch}",
                "-r", f"{mask_fps:.6f}", "-i", "-",
                "-c:v", "ffv1", "-pix_fmt", "bgra", output_path,
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=no_window,
        )
        frame_bytes = cw * ch * 3
        written = 0
        # Four stages at once, each on its own thread, in order: ffmpeg decodes
        # ahead into its pipe; this thread prepares the model input; one thread
        # owns the GPU; one cleans up and writes. ONNX Runtime, OpenCV and
        # pipe writes release the GIL, so a frame costs about its slowest
        # stage (the model, ~19 ms on a GPU), not the sum.
        from concurrent.futures import ThreadPoolExecutor

        def write(frame: np.ndarray, raw_future, meta) -> None:
            a = _finish(raw_future.result(), meta)
            bgra = np.dstack([foreground(frame, a), (a * 255.0).round().astype(np.uint8)])
            encoder.stdin.write(bgra.tobytes())

        pending = []
        with ThreadPoolExecutor(max_workers=1) as gpu, ThreadPoolExecutor(max_workers=1) as writer:
            while True:
                if cancel_check and written % 30 == 0:
                    check()
                buf = decoder.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frame = np.frombuffer(buf, np.uint8).reshape(ch, cw, 3)
                x, meta = _prepare(frame)
                pending.append(writer.submit(write, frame, gpu.submit(_infer, x), meta))
                written += 1
                # Keep the queue short (bounded memory), and surface errors.
                while len(pending) > 4:
                    pending.pop(0).result()
            for job in pending:
                job.result()
        encoder.stdin.close()
        decoder.wait(timeout=60)
        if encoder.wait(timeout=120) != 0 or written == 0:
            err = encoder.stderr.read().decode("utf-8", "replace")[-400:] if encoder.stderr else ""
            logger.warning("Avatar video failed (%d frames): %s", written, err)
            return None
        return mask_fps
    except Exception:
        if cancelled:
            raise  # a cancelled render stops; it doesn't fall back
        logger.exception("Avatar video failed")
        return None
    finally:
        for proc in (decoder, encoder):
            if proc and proc.poll() is None:
                proc.kill()

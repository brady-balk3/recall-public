# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import os
import re
from typing import Optional

import cv2
import numpy as np

from core.bundle_paths import get_models_dir
from core.device import get_ort_providers, get_torch_device
from core.models.signal import OCRSignal

# PP-OCRv6 small via PaddleOCR's ONNX Runtime engine. We intentionally avoid
# PaddleOCR's default paddle_static runtime because packaged Windows builds
# should not require paddlepaddle.
_OCR_ENGINE = None
_OCR_DISABLED_REASON: Optional[str] = None
_TEXT_DETECTION_MODEL = "PP-OCRv6_small_det"
_TEXT_RECOGNITION_MODEL = "PP-OCRv6_small_rec"
_TEXT_DETECTION_DIR = f"{_TEXT_DETECTION_MODEL}_onnx"
_TEXT_RECOGNITION_DIR = f"{_TEXT_RECOGNITION_MODEL}_onnx"
DEFAULT_OCR_MAX_SIDE = 1280  # S2 benchmark: full event recall, 27.5% faster than 1920px


def _normalize_text(text: str) -> str:
    normalized = text.upper()
    normalized = normalized.replace("|", "I").replace("!", "I")
    normalized = re.sub(r"[^A-Z0-9# ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _model_dir(name: str) -> str:
    return os.path.join(get_models_dir(), "ocr", name)


def _ocr_device() -> str:
    """Device string for PaddleOCR's onnxruntime engine ('gpu'/'cpu').

    Default is GPU when CUDA EP is available (gamer-PC path). Override with
    RECALL_OCR_GPU=0/false/off to force CPU, or =1/true/on to require the GPU
    attempt. Init failures must NOT disable OCR for the whole scan — see
    ``_get_ocr_engine`` CPU fallback. Parallel workers that hide the GPU via
    CUDA_VISIBLE_DEVICES stay on CPU because get_torch_device() is false there.
    """
    env = os.environ.get("RECALL_OCR_GPU", "auto").strip().lower()
    if env in ("0", "false", "off", "no", "cpu"):
        return "cpu"
    force_gpu = env in ("1", "true", "yes", "on", "gpu")
    if force_gpu or env in ("", "auto"):
        if get_torch_device() == "cuda" and "CUDAExecutionProvider" in get_ort_providers():
            return "gpu"
        if force_gpu:
            print("OCR: RECALL_OCR_GPU requested GPU but CUDA EP unavailable; using CPU")
    return "cpu"


def _build_paddle_ocr(device: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        engine="onnxruntime",
        device=device,
        text_detection_model_name=_TEXT_DETECTION_MODEL,
        text_detection_model_dir=_model_dir(_TEXT_DETECTION_DIR),
        text_recognition_model_name=_TEXT_RECOGNITION_MODEL,
        text_recognition_model_dir=_model_dir(_TEXT_RECOGNITION_DIR),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _get_ocr_engine():
    global _OCR_ENGINE, _OCR_DISABLED_REASON
    if _OCR_ENGINE is not None or _OCR_DISABLED_REASON is not None:
        return _OCR_ENGINE

    device = _ocr_device()
    try:
        _OCR_ENGINE = _build_paddle_ocr(device)
        print(f"OCR accelerator: device={device}, providers={get_ort_providers()}")
    except Exception as exc:  # noqa: BLE001
        if device == "gpu":
            print(f"OCR GPU init failed ({exc}); falling back to CPU")
            try:
                _OCR_ENGINE = _build_paddle_ocr("cpu")
                print(f"OCR accelerator: device=cpu (fallback), providers={get_ort_providers()}")
            except Exception as cpu_exc:  # noqa: BLE001
                _OCR_DISABLED_REASON = f"ppocrv6_onnx_unavailable: {cpu_exc}"
                print(f"OCR disabled: {_OCR_DISABLED_REASON}")
                return None
        else:
            _OCR_DISABLED_REASON = f"ppocrv6_onnx_unavailable: {exc}"
            print(f"OCR disabled: {_OCR_DISABLED_REASON}")
            return None

    return _OCR_ENGINE


def warm_ocr() -> bool:
    """Load the OCR engine now, and return True if OCR is available.

    MUST be called before the first cv2.VideoCapture decode. On Windows the
    paddlex/onnxruntime native DLLs have to initialize before OpenCV's video
    (ffmpeg) DLL is loaded into the process; if a VideoCapture opens first,
    PaddleOCR's init fails with `WinError 1114` and OCR is DISABLED for the whole
    run (no kill/win events). Warming the engine up-front forces the safe load
    order. Perception decodes video, so every perceive path warms OCR first.
    """
    return _get_ocr_engine() is not None


def _page_data(page):
    page_json = getattr(page, "json", None)
    if isinstance(page_json, dict):
        return page_json.get("res", page_json)
    if isinstance(page, dict):
        return page.get("res", page)
    return {}


def _entry_box(boxes, idx: int):
    if idx >= len(boxes):
        return None
    box = boxes[idx]
    return box.tolist() if hasattr(box, "tolist") else box


def get_ocr_max_side() -> int:
    """Longest-side cap (px) for the OCR input frame. 0 disables downscaling."""
    try:
        return int(os.environ.get("RECALL_OCR_MAX_SIDE", str(DEFAULT_OCR_MAX_SIDE)))
    except ValueError:
        return DEFAULT_OCR_MAX_SIDE


def _downscale_for_ocr(frame_img: np.ndarray):
    """Shrink the frame to the OCR size cap; return (scaled_img, source_scale).

    OCR cost scales with pixel count, and PP-OCRv6 reads large HUD banners well
    below source resolution. Only the ANALYSIS frame is scaled here -- exports
    re-read the original video at full quality (engines/export/ffmpeg_builder.py
    scales *up* to 1080x1920), so output resolution is untouched. ``source_scale``
    maps a coordinate in the scaled image back to source pixels, so the boxes we
    hand downstream (layout_map spatial routing) stay in the original frame's
    space. Tunable via RECALL_OCR_MAX_SIDE; small kill-feed text is the only
    recall risk, so keep the cap generous.
    """
    max_side = get_ocr_max_side()
    if max_side <= 0:
        return frame_img, 1.0
    h, w = frame_img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame_img, 1.0
    scale = max_side / float(longest)
    scaled = cv2.resize(
        frame_img,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,  # best quality when shrinking
    )
    return scaled, 1.0 / scale


def _rescale_box(box, factor: float):
    """Map an OCR box from scaled-image coords back to source coords."""
    if box is None or factor == 1.0:
        return box
    # Flat [x1, y1, x2, y2] (rec_boxes) or a 4-point polygon (rec_polys).
    if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        return [v * factor for v in box]
    return [[p[0] * factor, p[1] * factor] for p in box]


def run_ocr(frame_img: np.ndarray, timestamp: float, min_confidence: float = 0.5) -> OCRSignal:
    """Run OCR on a full frame and return the merged high-confidence text.

    PP-OCRv6 performs its own text detection across the whole frame, so one pass
    covers HUD text, victory banners, kill feed, and rank/crown indicators.
    """
    engine = _get_ocr_engine()
    if engine is None:
        return OCRSignal(
            timestamp=timestamp,
            text="",
            confidence=0.0,
            metadata={"disabled": True, "reason": _OCR_DISABLED_REASON or "unknown"},
        )

    scaled_img, source_scale = _downscale_for_ocr(frame_img)
    try:
        result = engine.predict(scaled_img)
    except Exception as exc:  # noqa: BLE001
        return OCRSignal(
            timestamp=timestamp,
            text="",
            confidence=0.0,
            metadata={"disabled": True, "reason": f"ocr_runtime_error: {exc}"},
        )

    raw_entries = []
    for page in (result or []):
        data = _page_data(page)
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        boxes = data.get("rec_boxes")
        if boxes is None:
            boxes = data.get("rec_polys")
        if boxes is None:
            boxes = []
        for idx, text in enumerate(texts):
            confidence = float(scores[idx] if idx < len(scores) else 0.0)
            if isinstance(text, str) and confidence >= min_confidence:
                raw_entries.append({
                    "text": text,
                    "confidence": confidence,
                    "box": _rescale_box(_entry_box(boxes, idx), source_scale),
                })

    if not raw_entries:
        return OCRSignal(timestamp=timestamp, text="", confidence=0.0)

    joined = " ".join(entry["text"] for entry in raw_entries)
    return OCRSignal(
        timestamp=timestamp,
        text=_normalize_text(joined),
        confidence=max(entry["confidence"] for entry in raw_entries),
        metadata={"raw_text": joined, "entries": raw_entries},
    )

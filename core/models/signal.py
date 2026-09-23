# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

@dataclass
class AudioSignal:
    timestamp: float
    amplitude: float
    is_spike: bool
    baseline: float = 0.0
    z_score: float = 0.0
    spike_score: float = 0.0

@dataclass
class OCRSignal:
    timestamp: float
    text: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VisionSignal:
    timestamp: float
    facecam_box: List[float]
    gameplay_box: List[float]
    motion_score: float = 0.0
    # How the facecam box was established. ``panel`` means an authored overlay
    # rectangle containing a person; ``person_fallback`` is the deliberately
    # looser edge-person recovery path. None keeps older signal caches valid.
    facecam_source: Optional[str] = None

@dataclass
class UnifiedSignal:
    timestamp: float
    audio: Optional[AudioSignal]
    ocr: Optional[OCRSignal]
    vision: Optional[VisionSignal]

    def to_dict(self):
        return asdict(self)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List

@dataclass
class GameEvent:
    timestamp_start: float
    timestamp_end: float
    event_type: str
    confidence: float
    signals: List[str]
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

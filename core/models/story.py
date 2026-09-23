# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from dataclasses import dataclass, asdict
from typing import List
from core.models.event import GameEvent

@dataclass
class GameStory:
    story_id: str
    start: float
    end: float
    events: List[GameEvent]
    score: float
    label: str

    def to_dict(self):
        # We need to manually convert the embedded events to dicts as well
        d = asdict(self)
        d['events'] = [e.to_dict() for e in self.events]
        return d

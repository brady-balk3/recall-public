# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional

@dataclass
class GameClip:
    clip_id: str
    start: float
    end: float
    story_id: str
    score: float
    layout: Dict[str, Any]
    hook_score: float = 0.0

    # Reaction-engine provenance (plan §4). Optional so the legacy clip path is
    # unaffected; populated when clips come from the reaction engine.
    reason: Optional[str] = None
    peak_timestamp: Optional[float] = None
    modality_breakdown: Dict[str, float] = field(default_factory=dict)
    reaction_auc: float = 0.0
    # Non-gameplay scene code (lobby / intermission), else None. Lets review
    # label + filter clips that fired on social hype with no gameplay behind them.
    scene_label: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)  # learned-ranker label capture
    # Semantic judge output (HUMAN_CLIPS Package C), populated only when the
    # local LLM judged this clip; the caption engine prefers semantic_title
    # over the transcript-quote fallback when present.
    semantic_title: Optional[str] = None
    semantic_hook_line: Optional[str] = None
    semantic_moment_type: Optional[str] = None
    semantic_verdict: Optional[str] = None
    # "second_look" marks ceiling-cut moments rendered with the primary deck
    # but hidden until Theater Review reveals More moments.
    review_tier: Optional[str] = None
    # Recall Session provenance. Without these the clip that a creator marked
    # live is stored as an ordinary session discovery, so review badges it
    # "Live-assisted" instead of "Remembered live" -- the one piece of evidence
    # that pressing Remember did anything.
    creator_protected: bool = False
    recall_marker_ids: list = field(default_factory=list)
    recall_marker_time: Optional[float] = None

    def to_dict(self):
        return asdict(self)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from dataclasses import dataclass, asdict, field
from typing import List, Optional


@dataclass
class ReactionFrame:
    """A single per-second sample of every reaction modality.

    All ``*_arousal`` / ``*_hype`` / ``game_evidence`` fields are raw (pre per-VOD
    normalization). The Reaction Engine robust-z normalizes them across the whole
    VOD before fusing into the ``ReactionCurve.score`` track.
    """
    timestamp: float
    face_present: bool = False
    face_arousal: float = 0.0       # 0..1, emotional intensity from FER (0 if no face)
    face_valence: float = 0.0       # -1..1, negative..positive
    face_emotion: str = "neutral"   # argmax label: happy/surprise/anger/neutral/...
    voice_arousal: float = 0.0      # 0..1, prosody-based excitement
    speech_hype: float = 0.0        # 0..1, lexicon+sentiment over ASR words near t
    burst: float = 0.0              # 0..1, laughter/scream tagger score at t
    burst_label: Optional[str] = None  # "laughter"|"scream"|"loud_onset" when meaningful
    loud_onset: float = 0.0         # 0..1, sudden relative loudness/startle edge at t
    startle: float = 0.0            # 0..1, loud onset synchronized with sudden Fear/Surprise
    chat: float = 0.0               # raw Twitch-chat intensity (rate + emote burst) at t
    motion: float = 0.0             # raw gameplay frame-difference motion at t (facecam masked)
    game_intensity: float = 0.0     # 0..1, game-side combat SFX / intense-music score at t
    game_evidence: float = 0.0      # 0..1, OCR tier weight (kill/knock/win) at/near t
    game_label: Optional[str] = None  # "elimination"|"knock"|"terminal_win"|... or None

    def to_dict(self):
        return asdict(self)


@dataclass
class ReactionCurve:
    """The fused, per-VOD-normalized reaction signal R(t) and its source frames."""
    fps: float                      # sampling rate of the track (1.0 baseline)
    frames: List[ReactionFrame] = field(default_factory=list)
    score: List[float] = field(default_factory=list)  # R(t), per-VOD normalized
    # Compact audit of candidate outcomes.  This makes a miss answerable from
    # the returned engine artifact instead of requiring a monkeypatch replay.
    selection_trace: List[dict] = field(default_factory=list)

    @property
    def timestamps(self) -> List[float]:
        return [f.timestamp for f in self.frames]

    def to_dict(self):
        return {
            "fps": self.fps,
            "frames": [f.to_dict() for f in self.frames],
            "score": list(self.score),
            "selection_trace": list(self.selection_trace),
        }

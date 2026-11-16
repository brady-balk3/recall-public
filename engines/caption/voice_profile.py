# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""A streamer's voice, learned from their own scans, with nothing to set up.

Per streamer (a Twitch VOD's channel name; a local recording has none and
stays per-VOD) Recall keeps the voice centroid and the streamer-voice
detector of their last KEEP_VODS filtered VODs. streamer_voice uses it when a
VOD's own dominant voice is NOT the streamer -- a squad that out-talks them --
which the per-VOD bootstrap cannot see by itself (see
streamer_voice.PROFILE_AGREE for the measurements).

Only vectors and weights are stored: a 256-float centroid and a 12.5k-weight
MLP per VOD, ~70 KB a streamer. Never audio, never words. The file name is a
hash of the channel name. "Wipe all data" deletes the folder.

Poisoning guard: a VOD joins the profile only when its own detector was
trained (the gate filtered it on its own centroid) and either (a) no profile
exists yet and its dominant cluster is clearly dominant (near share >=
SEED_MIN_NEAR), or (b) its centroid agrees with the profile (cosine >=
PROFILE_AGREE). A VOD whose dominant voice is someone else is never added.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from engines.caption import streamer_voice as sv

PROFILE_VERSION = 1
DIRNAME = "voice_profiles"

# The newest VODs win. MEASURED (research_streamer_activity.py `detect16`):
# a detector trained on a streamer's other VODs transfers to a held-out one at
# frame AUC 0.976-0.988, so a few VODs carry the voice; keeping only recent
# ones follows a changed mic or room instead of averaging it away.
KEEP_VODS = 4
# Seeding needs a clearly dominant voice. MEASURED: real squad VODs put
# 0.38-0.54 of their full windows near their own centroid; the thinned VODs
# whose centroid had flipped to a teammate 0.24-0.33.
SEED_MIN_NEAR = 0.35

_LOCK = threading.Lock()


def normalize_streamer(name) -> Optional[str]:
    text = " ".join(str(name or "").split()).lower()
    return text or None


def profile_dir(data_root: str) -> str:
    return os.path.join(data_root, DIRNAME)


def profile_path(data_root: str, streamer: str) -> Optional[str]:
    key = normalize_streamer(streamer)
    if not key:
        return None
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    from core import signal_cache

    return os.path.join(profile_dir(data_root), f"{digest}{signal_cache.CACHE_EXT}")


@dataclass
class ProfileEntry:
    audio_key: str
    centroid: np.ndarray
    mlp: object
    added_at: float


@dataclass
class VoiceProfile:
    entries: List[ProfileEntry] = field(default_factory=list)

    @property
    def centroid(self) -> Optional[np.ndarray]:
        if not self.entries:
            return None
        mean = np.mean([e.centroid for e in self.entries], axis=0)
        norm = float(np.linalg.norm(mean))
        return (mean / norm).astype(np.float32) if norm > 0 else None

    @property
    def mlps(self) -> list:
        return [e.mlp for e in self.entries if e.mlp is not None]

    def without(self, audio_key: str) -> "VoiceProfile":
        return VoiceProfile([e for e in self.entries if e.audio_key != audio_key])

    def to_cache(self) -> dict:
        return {"version": PROFILE_VERSION, "entries": [
            {"audio_key": e.audio_key, "added_at": e.added_at,
             "centroid": sv.pack_array(np.asarray(e.centroid, np.float32)),
             "mlp": sv.pack_mlp(e.mlp)} for e in self.entries]}

    @classmethod
    def from_cache(cls, entry) -> Optional["VoiceProfile"]:
        if not isinstance(entry, dict) or entry.get("version") != PROFILE_VERSION:
            return None
        rows = entry.get("entries")
        if not isinstance(rows, list):
            return None
        out = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("audio_key"), str):
                return None
            centroid = sv.unpack_array(row.get("centroid"), "<f4", 1)
            mlp = sv.unpack_mlp(row.get("mlp"))
            if centroid is None or centroid.shape != (256,) or mlp is None:
                return None
            try:
                added = float(row.get("added_at", 0.0))
            except (TypeError, ValueError):
                return None
            out.append(ProfileEntry(row["audio_key"], centroid, mlp, added))
        return cls(out)


def load_profile(data_root: Optional[str], streamer) -> Optional[VoiceProfile]:
    """The streamer's profile, or None (no name, none yet, or unreadable)."""
    if not data_root:
        return None
    path = profile_path(data_root, streamer)
    if not path:
        return None
    from core import signal_cache

    profile = VoiceProfile.from_cache(signal_cache.load(path))
    return profile if profile and profile.entries else None


def update_profile(data_root: Optional[str], streamer, audio_key: str, voice) -> str:
    """Add this VOD to the streamer's profile when the guard allows; returns
    what happened ("added", "replaced", or "skipped: <why>")."""
    if not data_root or not normalize_streamer(streamer):
        return "skipped: no streamer"
    gate = getattr(voice, "gate", None) or {}
    if voice is None or not voice.active or gate.get("source") != "vod" or voice.mlp is None:
        return "skipped: not filtered on its own voice"
    from core import signal_cache

    path = profile_path(data_root, streamer)
    with _LOCK:
        current = VoiceProfile.from_cache(signal_cache.load(path)) or VoiceProfile()
        others = current.without(audio_key)
        if others.entries:
            cosine = float(np.asarray(voice.centroid, np.float32) @ others.centroid)
            if cosine < sv.PROFILE_AGREE:
                return f"skipped: disagrees with the profile (cosine {cosine:.2f})"
        elif float(gate.get("near", 0.0)) < SEED_MIN_NEAR:
            return f"skipped: dominant voice too weak to seed ({gate.get('near', 0.0):.2f})"
        replaced = len(others.entries) != len(current.entries)
        entries = others.entries + [ProfileEntry(audio_key, np.asarray(voice.centroid, np.float32),
                                                 voice.mlp, time.time())]
        os.makedirs(profile_dir(data_root), exist_ok=True)
        signal_cache.save(path, VoiceProfile(entries[-KEEP_VODS:]).to_cache())
    return "replaced" if replaced else "added"


def clear_profiles(data_root: str) -> int:
    """Delete every stored profile; returns the bytes removed."""
    folder = profile_dir(data_root)
    removed = 0
    with _LOCK:
        if os.path.isdir(folder):
            for dirpath, _, names in os.walk(folder):
                for name in names:
                    try:
                        removed += os.path.getsize(os.path.join(dirpath, name))
                    except OSError:
                        pass
            shutil.rmtree(folder, ignore_errors=True)
    return removed

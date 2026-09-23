# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Optional Twitch VOD chat-replay fetcher (plan 9.1).

Fetches a VOD's chat replay by shelling out to **TwitchDownloaderCLI**'s
``chatdownload`` command (resolved via core/twitchdl_path.py). It writes the
replay to a JSON file which we parse into ``ChatMessage``s.

Chat is a best-effort optional channel: if the CLI isn't available, the URL
isn't a Twitch VOD, or anything fails/times out, this returns ``None`` and the
reaction pipeline simply runs without the chat channel -- exactly like a VOD
with no facecam or no ASR.

History: this previously used the ``chat_downloader`` PyPI package, but its
Twitch handler is broken against Twitch's current API (every request raises
``KeyError: 'data'``) and it retries with exponential backoff forever, hanging
the whole pipeline with no timeout. TwitchDownloaderCLI uses the current API,
downloads in seconds, and exits cleanly. The fetch here is additionally wrapped
in a hard wall-clock timeout with cancellation so an optional channel can never
block a scan again, whatever the backend does.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from typing import Callable, List, Optional

from engines.chat.chat_features import ChatMessage
from core.twitchdl_path import get_twitchdl_path

_TWITCH_VOD_RE = re.compile(r"twitch\.tv/videos/(\d+)", re.IGNORECASE)

# A full 3.7h chat downloads in well under a minute; cap generously so a slow
# network can't hang a scan, but a genuine large chat still completes.
DEFAULT_TIMEOUT_SEC = 240.0


def is_twitch_vod(url: str) -> bool:
    return bool(url) and bool(_TWITCH_VOD_RE.search(url))


def _vod_id(url: str) -> Optional[str]:
    m = _TWITCH_VOD_RE.search(url or "")
    return m.group(1) if m else None


def chat_available() -> bool:
    """True when TwitchDownloaderCLI can be resolved (env/bundle/tools/PATH)."""
    return get_twitchdl_path() is not None


def _emote_count(message: dict) -> int:
    """Count emotes in a TwitchDownloaderCLI message.

    Prefer non-null emoticon fragments (covers messages that carry emotes only
    in ``fragments``); fall back to the flat ``emoticons`` list length.
    """
    frags = message.get("fragments")
    if isinstance(frags, list):
        n = sum(1 for f in frags if isinstance(f, dict) and f.get("emoticon"))
        if n:
            return n
    emoticons = message.get("emoticons")
    return len(emoticons) if isinstance(emoticons, list) else 0


def _temporary_chat_file(temp_dir: Optional[str]) -> str:
    """Reserve a request-owned file without creator or VOD identifiers."""
    fd, path = tempfile.mkstemp(prefix="recall-chat-", suffix=".json", dir=temp_dir)
    os.close(fd)
    return path


def _run_chatdownload(
    vid: str,
    out_path: str,
    cancel_check: Optional[Callable[[], None]],
    timeout: float,
    begin: Optional[float] = None,
    end: Optional[float] = None,
) -> bool:
    """Run TwitchDownloaderCLI ``chatdownload`` to ``out_path``. Returns success.

    Time-boxed and cancellable: the child is polled and always killed on exit,
    so this can never hang a scan. ``begin``/``end`` (seconds) trim the replay —
    used for a tiny metadata-only slice as well as full-chat fetches.
    """
    exe = get_twitchdl_path()
    if not exe:
        return False
    cmd = [
        exe, "chatdownload",
        "-u", vid,
        "-o", out_path,
        "--banner", "false",
        "--embed-images", "false",
        # Third-party emote metadata (embeddedData) is unused here; skipping the
        # BTTV/FFZ/7TV lookups avoids extra network calls and speeds the fetch.
        "--bttv", "false",
        "--ffz", "false",
        "--stv", "false",
        "--collision", "Overwrite",
    ]
    if begin is not None:
        cmd += ["-b", str(begin)]
    if end is not None:
        cmd += ["-e", str(end)]

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + max(1.0, timeout)
        while proc.poll() is None:
            if cancel_check:
                cancel_check()  # may raise InterruptedError
            if time.monotonic() > deadline:
                return False
            time.sleep(0.5)
        return proc.returncode == 0 and os.path.exists(out_path)
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass


def fetch_twitch_video_meta(
    url: str,
    cancel_check: Optional[Callable[[], None]] = None,
    timeout: float = 60.0,
    temp_dir: Optional[str] = None,
) -> Optional[dict]:
    """Fetch a Twitch VOD's authoritative game + game-change timeline, or None.

    TwitchDownloaderCLI embeds a ``video`` block (title, length, ``game``, and a
    ``chapters`` GAME_CHANGE timeline) in every chat download — so a 1-second
    replay slice yields ground-truth game metadata in ~1s, regardless of scan
    mode. This is far more reliable than fingerprinting the HUD from OCR.

    Returns ``{"title": str|None, "streamer": str|None, "game": str|None,
    "created_at": str|None, "length": float|None, "chapters": [
    {"game": str, "start": float, "end": float}, ...]}`` (chapter games are the
    raw Twitch display names; the caller maps them to internal ids).
    """
    vid = _vod_id(url)
    if not vid or not get_twitchdl_path():
        return None
    out_path = None
    try:
        out_path = _temporary_chat_file(temp_dir)
        if not _run_chatdownload(vid, out_path, cancel_check, timeout, begin=0, end=1):
            return None
        with open(out_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh) or {}
        video = metadata.get("video") or {}
        streamer = metadata.get("streamer") or {}
        chapters = []
        for ch in video.get("chapters") or []:
            if not isinstance(ch, dict):
                continue
            name = ch.get("gameDisplayName") or ch.get("description")
            if not name:
                continue
            start = float(ch.get("startMilliseconds", 0)) / 1000.0
            end = start + float(ch.get("lengthMilliseconds", 0)) / 1000.0
            chapters.append({"game": str(name), "start": start, "end": end})
        game = video.get("game")
        created_at = (
            video.get("created_at")
            or video.get("createdAt")
            or video.get("published_at")
            or video.get("publishedAt")
        )
        if not video:
            return None
        return {
            "id": str(video.get("id") or vid),
            "title": str(video["title"]).strip() if video.get("title") else None,
            "streamer": str(streamer["name"]).strip() if streamer.get("name") else None,
            "game": str(game) if game else None,
            "created_at": str(created_at).strip() if created_at else None,
            "length": float(video["length"]) if video.get("length") else None,
            "chapters": chapters,
        }
    except InterruptedError:
        raise
    except Exception:
        return None
    finally:
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass


def _parse_chat_json(path: str, max_messages: int) -> Optional[List[ChatMessage]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    comments = data.get("comments")
    if not isinstance(comments, list):
        return None
    out: List[ChatMessage] = []
    for c in comments:
        if not isinstance(c, dict):
            continue
        t = c.get("content_offset_seconds")
        if t is None:
            continue
        msg = c.get("message") or {}
        out.append(ChatMessage(
            timestamp=float(t),
            text=str(msg.get("body") or ""),
            emotes=_emote_count(msg),
        ))
        if len(out) >= max_messages:
            break
    return out or None


def fetch_twitch_chat_regions(
    url: str,
    regions,
    max_messages: int = 200_000,
    cancel_check: Optional[Callable[[], None]] = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    temp_dir: Optional[str] = None,
) -> Optional[List[ChatMessage]]:
    """Fetch chat for only the given ``(start, end)`` second regions, merged.

    Smart scan (fast mode) fetches chat just for the scouted candidate regions
    instead of the whole replay. One ``chatdownload -b/-e`` per region, all
    sharing a single wall-clock budget (``timeout`` total, not per region) so a
    long region list can't multiply the time-box. ``content_offset_seconds``
    is VOD-relative regardless of the trim, so the merged list aligns with the
    reaction timeline exactly like a full fetch. Messages are de-duplicated on
    (timestamp, text) in case regions ever touch.

    Best-effort like every optional channel: a failed region is skipped, a
    partial fetch returns what arrived, and any total failure returns None.
    ``cancel_check`` may raise InterruptedError, which propagates.
    """
    vid = _vod_id(url)
    if not vid or not get_twitchdl_path():
        return None
    spans = sorted(
        (max(0.0, float(start)), float(end))
        for start, end in (regions or [])
        if float(end) > float(start)
    )
    if not spans:
        return None

    merged: List[ChatMessage] = []
    seen = set()
    deadline = time.monotonic() + max(1.0, timeout)
    for start, end in spans:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or len(merged) >= max_messages:
            break
        out_path = None
        try:
            out_path = _temporary_chat_file(temp_dir)
            if not _run_chatdownload(
                vid, out_path, cancel_check, remaining, begin=start, end=end,
            ):
                continue
            msgs = _parse_chat_json(out_path, max_messages - len(merged)) or []
        except InterruptedError:
            raise
        except Exception:
            continue
        finally:
            try:
                if out_path and os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
        for m in msgs:
            key = (round(m.timestamp, 3), m.text)
            if key not in seen:
                seen.add(key)
                merged.append(m)

    merged.sort(key=lambda m: m.timestamp)
    return merged or None


def fetch_twitch_chat(
    url: str,
    max_messages: int = 200_000,
    cancel_check: Optional[Callable[[], None]] = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    temp_dir: Optional[str] = None,
) -> Optional[List[ChatMessage]]:
    """Fetch a Twitch VOD's chat replay as ``ChatMessage``s, or ``None``.

    ``content_offset_seconds`` on each replay comment is the offset into the VOD,
    which aligns directly with the reaction timeline. Best-effort and defensive:
    any failure/timeout returns None so a scan never fails because of chat.

    ``cancel_check`` (if given) is polled while the CLI runs and may raise
    ``InterruptedError`` to abort; the child process is always killed on exit.
    """
    vid = _vod_id(url)
    if not vid or not get_twitchdl_path():
        return None
    out_path = None
    try:
        out_path = _temporary_chat_file(temp_dir)
        if not _run_chatdownload(vid, out_path, cancel_check, timeout):
            return None
        return _parse_chat_json(out_path, max_messages)
    except InterruptedError:
        raise
    except Exception:
        return None
    finally:
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass

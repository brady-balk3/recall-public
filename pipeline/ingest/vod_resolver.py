# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from core.bundle_paths import get_data_dir
"""Resolve an input source (local file or Twitch VOD URL) to a local media file.

URL ingestion goes through **TwitchDownloaderCLI** (resolved via
core/twitchdl_path.py), not yt-dlp. The product is Twitch-only, and the same CLI
already powers chat + game-metadata fetching, so consolidating on it removes the
yt-dlp dependency and keeps one well-maintained tool that speaks Twitch's current
API. Non-Twitch URLs are rejected with a clear message rather than silently
half-working.
"""

import collections
import queue
import threading
import os
import re
import shutil
import subprocess
import tempfile

from core.twitchdl_path import get_twitchdl_path
from core.ffmpeg_path import get_ffmpeg_path

MEDIA_EXTENSIONS = (".mp4", ".mkv", ".webm")

_TWITCH_VOD_RE = re.compile(r"twitch\.tv/videos/(\d+)", re.IGNORECASE)
_PCT_RE = re.compile(r"(\d{1,3})%")


def is_url(input_source: str) -> bool:
    """Check if the input source is a URL."""
    return input_source.startswith("http://") or input_source.startswith("https://")


def is_twitch_vod(url: str) -> bool:
    return bool(url) and bool(_TWITCH_VOD_RE.search(url))


def _twitch_vod_id(url: str) -> str | None:
    m = _TWITCH_VOD_RE.search(url or "")
    return m.group(1) if m else None


def _asset_key(vod_id: str) -> str:
    # Preserve the historical yt-dlp naming (``twitchvod_v<id>``) so VODs already
    # downloaded under the old resolver are reused instead of re-downloaded.
    return f"twitchvod_v{vod_id}"


def _existing_download(output_dir: str, asset_key: str) -> str | None:
    for ext in MEDIA_EXTENSIONS:
        candidate = os.path.join(output_dir, f"{asset_key}{ext}")
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            return os.path.abspath(candidate)
    return None


def _iter_status_chunks(stream):
    """Yield progress-update chunks from a stream that uses '\\r' for in-place
    updates as well as '\\n' for new lines (TwitchDownloaderCLI does both)."""
    buf = ""
    while True:
        ch = stream.read(1)
        if ch == "":
            if buf:
                yield buf
            return
        if ch in ("\r", "\n"):
            if buf.strip():
                yield buf.strip()
            buf = ""
        else:
            buf += ch
            if len(buf) >= 4096:
                yield buf
                buf = ""


def _download_status(process, cancel_check=None):
    """Drain output separately so a quiet downloader cannot block cancellation."""
    chunks = queue.Queue(maxsize=128)
    stopped = threading.Event()

    def read_output():
        try:
            for chunk in _iter_status_chunks(process.stdout):
                while not stopped.is_set():
                    try:
                        chunks.put(chunk, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stopped.is_set():
                    break
        finally:
            stopped.set()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        while not stopped.is_set() or not chunks.empty() or process.poll() is None:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Download cancelled")
            try:
                yield chunks.get(timeout=0.2)
            except queue.Empty:
                continue
    finally:
        stopped.set()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        reader.join(timeout=1)
        if not reader.is_alive() and process.stdout is not None:
            process.stdout.close()


def resolve_input(input_source: str, output_dir: str | None = None, progress_callback=None, cancel_check=None) -> str:
    """Resolve ``input_source`` to a local media file path.

    Local file -> its absolute path. Twitch VOD URL -> downloaded via
    TwitchDownloaderCLI (cached by VOD id). ``progress_callback(message,
    progress, eta_seconds)`` is called during download; it may raise
    ``InterruptedError`` to cancel, which kills the download promptly.
    """
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Download cancelled")
    if output_dir is None:
        output_dir = os.path.join(get_data_dir(), 'assets')
    if not is_url(input_source):
        if not os.path.exists(input_source):
            raise FileNotFoundError(f"Local file not found: {input_source}")
        return os.path.abspath(input_source)

    vod_id = _twitch_vod_id(input_source)
    if not vod_id:
        raise ValueError(
            "Only Twitch VOD URLs (twitch.tv/videos/<id>) and local files are "
            f"supported. Got: {input_source}"
        )

    exe = get_twitchdl_path()
    if not exe:
        raise RuntimeError(
            "TwitchDownloaderCLI not found. Install it and put it on PATH, in "
            "tools/twitchdownloader/, or set RECALL_TWITCHDL_PATH."
        )

    print(f"Twitch VOD detected. Resolving {input_source}...")
    os.makedirs(output_dir, exist_ok=True)

    asset_key = _asset_key(vod_id)
    existing_file = _existing_download(output_dir, asset_key)
    if existing_file:
        if progress_callback:
            progress_callback("Using cached VOD download.", 0.075, None)
        print(f"Using cached VOD: {existing_file}")
        return existing_file

    output_file = os.path.join(output_dir, f"{asset_key}.mp4")
    # Each attempt owns its staging directory; failed downloads never become
    # reusable cache entries and concurrent requests cannot share chunk files.
    temp_path = tempfile.mkdtemp(prefix=".recall-vod-", dir=output_dir)
    download_file = os.path.join(temp_path, "download.mp4")
    try:
        cmd = [
            exe, "videodownload",
            "-u", vod_id,
            "-o", download_file,
            "--temp-path", os.path.join(temp_path, "chunks"),
            "--collision", "Overwrite",
            "--banner", "false",
            # Explicit rather than relying on the CLI's default -- if Twitch has
            # aged out the source/chunked rendition for this VOD, "best" silently
            # falls back to whatever's left (720p/480p), which then gets upscaled
            # by the export filters and reads as blurry with no visible warning.
            "-q", "best",
        ]
        # TwitchDownloaderCLI muxes with ffmpeg at the end and resolves --ffmpeg-path
        # against its own working dir, so a bare "ffmpeg" (dev/PATH mode) turns into a
        # bogus path in that working dir and crashes. Only pass an ABSOLUTE path;
        # resolve a bare name via PATH, and if we can't, omit the flag and let the CLI
        # fall back to its own resolution.
        ffmpeg = get_ffmpeg_path()
        if ffmpeg and not os.path.isabs(ffmpeg):
            ffmpeg = shutil.which(ffmpeg)
        if ffmpeg and os.path.isabs(ffmpeg) and os.path.exists(ffmpeg):
            cmd += ["--ffmpeg-path", ffmpeg]
        print(f"Downloading VOD to {output_file}...")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # Keep a rolling tail of CLI output so a failure surfaces the real reason
        # (a .NET stack trace, "video not found", etc.) instead of a bare exit code.
        tail = collections.deque(maxlen=40)
        try:
            for chunk in _download_status(proc, cancel_check):
                tail.append(chunk)
                if not progress_callback:
                    continue
                # TwitchDownloaderCLI phases: "Downloading X%" -> "Combining Parts
                # X%" -> "Finalizing". Surface the download % (the long part) and
                # keep the callback firing so a cancel aborts promptly.
                label = chunk.replace("[STATUS] - ", "").replace("[STATUS]", "").strip()
                m = _PCT_RE.search(label)
                if m and "download" in label.lower():
                    ratio = min(1.0, max(0.0, int(m.group(1)) / 100.0))
                    progress_callback(
                        f"Downloading Twitch VOD: {m.group(1)}%",
                        0.01 + ratio * 0.065,
                        None,
                    )
                elif label:
                    progress_callback(label, 0.07, None)
            proc.wait()
        except BaseException:
            # Includes InterruptedError from a cancel inside progress_callback.
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            raise

        if proc.returncode != 0:
            # Prefer the informative line(s) from the CLI's output for the message.
            informative = [c for c in tail if re.search(r"exception|error|cannot find|not found|Win32", c, re.I)]
            detail = " | ".join((informative or list(tail))[:3]).strip()
            raise RuntimeError(
                f"TwitchDownloaderCLI failed (exit {proc.returncode}) for {input_source}"
                + (f": {detail}" if detail else "")
            )
        if not (os.path.isfile(download_file) and os.path.getsize(download_file) > 0):
            raise RuntimeError(f"VOD download produced no file: {output_file}")

        if cancel_check is not None and cancel_check():
            raise InterruptedError("Download cancelled")
        os.replace(download_file, output_file)
        print(f"Download complete: {output_file}")
        _warn_if_low_resolution(output_file, ffmpeg)
        return os.path.abspath(output_file)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)



def _warn_if_low_resolution(video_path: str, ffmpeg_path: str, min_height: int = 900) -> None:
    """Best-effort ffprobe check so a stale/expired VOD rendition (Twitch
    silently serving 720p/480p once the source quality ages out) shows up in
    the logs instead of only surfacing as "blurry export" days later."""
    ffprobe = (ffmpeg_path or "ffprobe").replace("ffmpeg", "ffprobe") if ffmpeg_path else "ffprobe"
    try:
        res = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path],
            capture_output=True, text=True, timeout=15,
        )
        dims = res.stdout.strip()
        if "x" in dims:
            width, height = (int(v) for v in dims.split("x")[:2])
            if height < min_height:
                print(
                    f"WARNING: downloaded VOD is only {width}x{height} -- Twitch likely no "
                    "longer has the source-quality rendition for this VOD. Exports upscaled "
                    "from this file will look soft/blurry regardless of encoder settings."
                )
    except Exception:
        pass  # best-effort diagnostic only, never block the pipeline on it

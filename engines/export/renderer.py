# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import json
from collections import deque
import logging
import os
import queue
import subprocess
import threading
import tempfile
import time
from typing import Callable, Dict, List, Optional

from core.models.clip import GameClip
from core.process_governor import subprocess_creationflags
from engines.export.ffmpeg_builder import build_ffmpeg_command


logger = logging.getLogger(__name__)
RenderProgressCallback = Callable[[Dict[str, float]], None]
CancelCheck = Callable[[], bool]


class FFmpegCancelled(RuntimeError):
    """Raised after a cooperative export cancellation terminates FFmpeg."""


def _progress_seconds(fields: Dict[str, str]) -> Optional[float]:
    """Return FFmpeg's current output timestamp in seconds.

    FFmpeg's historical ``out_time_ms`` field is named misleadingly: like the
    newer ``out_time_us`` field, its value is expressed in microseconds. The
    formatted timestamp is retained as a fallback for older builds.
    """
    for key in ("out_time_us", "out_time_ms"):
        value = fields.get(key)
        if value is not None:
            try:
                return max(0.0, float(value) / 1_000_000.0)
            except (TypeError, ValueError):
                pass

    value = fields.get("out_time")
    if not value:
        return None
    try:
        hours, minutes, seconds = value.split(":", 2)
        return max(0.0, (float(hours) * 3600.0) + (float(minutes) * 60.0) + float(seconds))
    except (TypeError, ValueError):
        return None


def _drain_progress_stdout(stream, output: queue.Queue[Optional[str]]) -> None:
    def enqueue(line: Optional[str]) -> None:
        # Progress is disposable telemetry. Keep the newest lines without
        # blocking pipe drainage (including while the process is cancelled).
        while True:
            try:
                output.put_nowait(line)
                return
            except queue.Full:
                try:
                    output.get_nowait()
                except queue.Empty:
                    pass

    try:
        if stream is not None:
            for line in iter(lambda: stream.readline(4096), ""):
                enqueue(line)
    finally:
        enqueue(None)


def _drain_stderr(stream, output: deque[str]) -> None:
    if stream is not None:
        for line in iter(lambda: stream.readline(4096), ""):
            output.append(line)


def _stop_progress_process(process, threads: List[threading.Thread]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait()
    for thread in threads:
        thread.join(timeout=1.0)


def _progress_wait_seconds(
    *,
    process,
    threads: List[threading.Thread],
    progress_cmd: List[str],
    started_at: float,
    timeout_seconds: Optional[float],
    cancel_check: Optional[CancelCheck],
) -> Optional[float]:
    if cancel_check is not None and cancel_check():
        _stop_progress_process(process, threads)
        raise FFmpegCancelled("FFmpeg export cancelled")
    wait_seconds = 0.25 if cancel_check is not None else None
    if timeout_seconds is None:
        return wait_seconds
    remaining = timeout_seconds - (time.monotonic() - started_at)
    if remaining <= 0:
        _stop_progress_process(process, threads)
        raise subprocess.TimeoutExpired(progress_cmd, timeout_seconds)
    return min(wait_seconds or 0.25, remaining)


def _emit_progress_line(
    raw_line: str,
    fields: Dict[str, str],
    *,
    duration: float,
    progress_callback: RenderProgressCallback,
) -> None:
    key, separator, value = raw_line.strip().partition("=")
    if not separator:
        return
    if key != "progress":
        if key in {"out_time_us", "out_time_ms", "out_time"}:
            fields[key] = value
        return

    rendered_seconds = _progress_seconds(fields)
    if value == "end":
        rendered_seconds = duration
        render_progress = 1.0
    elif rendered_seconds is None:
        fields.clear()
        return
    else:
        render_progress = min(0.995, rendered_seconds / max(duration, 0.001))
    try:
        progress_callback({
            "rendered_seconds": min(duration, max(0.0, rendered_seconds)),
            "render_duration_seconds": duration,
            "render_progress": render_progress,
        })
    except Exception:
        logger.debug("FFmpeg progress callback failed", exc_info=True)
    fields.clear()


def _wait_for_progress_process(
    process,
    *,
    threads: List[threading.Thread],
    progress_cmd: List[str],
    stderr_lines: deque[str],
    started_at: float,
    timeout_seconds: Optional[float],
    cancel_check: Optional[CancelCheck] = None,
) -> None:
    while True:
        wait_seconds = _progress_wait_seconds(
            process=process, threads=threads, progress_cmd=progress_cmd,
            started_at=started_at, timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
        )
        try:
            return_code = process.wait(timeout=wait_seconds)
            break
        except subprocess.TimeoutExpired:
            continue

    for thread in threads:
        thread.join(timeout=1.0)
    if return_code:
        raise subprocess.CalledProcessError(
            return_code,
            progress_cmd,
            stderr="".join(stderr_lines),
        )


def _run_with_progress(
    cmd: List[str],
    *,
    duration: float,
    background: bool,
    progress_callback: RenderProgressCallback,
    timeout_seconds: Optional[float] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> None:
    """Run FFmpeg and translate its ``-progress`` blocks into clip progress."""
    progress_cmd = [*cmd[:-1], "-progress", "pipe:1", "-nostats", cmd[-1]]
    process = subprocess.Popen(  # noqa: S603 - command is fully constructed in-repo
        progress_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=subprocess_creationflags(background=background),
    )
    stdout_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=256)
    stderr_lines: deque[str] = deque(maxlen=256)
    threads = [
        threading.Thread(
            target=_drain_progress_stdout,
            args=(process.stdout, stdout_queue),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stderr,
            args=(process.stderr, stderr_lines),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    started_at = time.monotonic()
    fields: Dict[str, str] = {}
    while True:
        wait_seconds = _progress_wait_seconds(
            process=process,
            threads=threads,
            progress_cmd=progress_cmd,
            started_at=started_at,
            timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
        )
        try:
            raw_line = stdout_queue.get(timeout=wait_seconds)
        except queue.Empty:
            continue
        if raw_line is None:
            break
        _emit_progress_line(
            raw_line,
            fields,
            duration=duration,
            progress_callback=progress_callback,
        )

    _wait_for_progress_process(
        process,
        threads=threads,
        progress_cmd=progress_cmd,
        stderr_lines=stderr_lines,
        started_at=started_at,
        timeout_seconds=timeout_seconds,
        cancel_check=cancel_check,
    )


def render_clip(
    video_path: str,
    clip: GameClip,
    output_dir: str,
    ass_path: str = None,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    preview: bool = False,
    video_fade_in: float = None,
    video_fade_out: float = None,
    audio_fade_in: float = None,
    audio_fade_out: float = None,
    progress_callback: Optional[RenderProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> str:
    """Publish only a successful render; failures leave the old MP4 usable.

    Staging lives on the destination filesystem for atomic MP4 replacement.
    Wait briefly for a detached Windows preview to release the previous MP4.
    The metadata sidecar is a separate replacement, not a two-file transaction.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.join(output_dir, f"clip_{clip.clip_id}.mp4"))
    log_path = os.path.join(output_dir, f"clip_{clip.clip_id}_meta.json")
    with tempfile.TemporaryDirectory(prefix=".recall-render-", dir=output_dir) as staging:
        staged_path = _render_clip_to_directory(
            video_path, clip, staging, ass_path, fade_in, fade_out, preview,
            video_fade_in, video_fade_out, audio_fade_in, audio_fade_out,
            progress_callback, cancel_check,
        )
        if not os.path.isfile(staged_path) or os.path.getsize(staged_path) == 0:
            raise RuntimeError("FFmpeg did not produce a nonempty clip")
        if cancel_check is not None and cancel_check():
            raise FFmpegCancelled("FFmpeg export cancelled before publication")
        # Encoding and metadata serialization have both succeeded at this point.
        # The editor detaches its media before saving. Give the outstanding
        # range response time to close its Windows file handle, then replace
        # the same path. Never accumulate alternate clip files on a lock.
        for attempt in range(21):
            if cancel_check is not None and cancel_check():
                raise FFmpegCancelled("FFmpeg export cancelled before publication")
            try:
                os.replace(staged_path, output_path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 20:
                    raise
                time.sleep(0.1)
        try:
            os.replace(os.path.join(staging, f"clip_{clip.clip_id}_meta.json"), log_path)
        except OSError as exc:
            raise RuntimeError(
                "Clip video was published, but its metadata could not be updated. "
                "Retry the edit before rebuilding or exporting this clip."
            ) from exc
    return output_path


def _render_clip_to_directory(
    video_path: str,
    clip: GameClip,
    output_dir: str,
    ass_path: str = None,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    preview: bool = False,
    video_fade_in: float = None,
    video_fade_out: float = None,
    audio_fade_in: float = None,
    audio_fade_out: float = None,
    progress_callback: Optional[RenderProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> str:
    """Render a single clip to MP4 using FFmpeg and save its metadata log.

    ``preview=True`` renders a cheap low-res review proxy (Clip Library
    review pass) to the same filename a final render would use -- a later
    call with ``preview=False`` overwrites it in place, so no filename churn
    or orphaned files across the preview -> final transition."""
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"clip_{clip.clip_id}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    log_path = os.path.join(output_dir, f"clip_{clip.clip_id}_meta.json")
    layout_type = clip.layout.get("type", "vertical_split")
    facecam_metadata = (
        clip.layout.get("facecam")
        if layout_type in ("vertical_split", "gameplay_pip", "vtuber_overlay")
        else None
    )
    cmd = build_ffmpeg_command(
        video_path,
        output_path,
        clip.start,
        clip.end,
        facecam_metadata,
        ass_path,
        fade_in,
        fade_out,
        layout_type=layout_type,
        gameplay_metadata=clip.layout.get("gameplay"),
        preview=preview,
        video_fade_in=video_fade_in,
        video_fade_out=video_fade_out,
        audio_fade_in=audio_fade_in,
        audio_fade_out=audio_fade_out,
        gameplay_focus_x=float(clip.layout.get("focus_x", 0.5)),
        facecam_subject_top=clip.layout.get("facecam_subject_top"),
        # Set only by the face anchor, which MEASURES where the creator is.
        # Absent -> full_camera keeps its fit-over-blur treatment.
        camera_cover=bool(clip.layout.get("camera_cover")),
        # Script-authored opt-in for PNG/VTuber cutouts. Normal app layouts do
        # not carry this key, so every existing export remains byte-for-byte on
        # its previous graph until the treatment is explicitly approved.
        vtuber_overlay=clip.layout.get("vtuber_overlay"),
    )

    print(f"Rendering {output_filename}...")
    try:
        # Batch preview proxies run at background priority (resource plan
        # pkg 1) — they follow a long scan and shouldn't fight the user's
        # apps. Final exports are user-initiated waits: normal priority.
        if progress_callback is None and cancel_check is None:
            subprocess.run(  # noqa: S603 - command is fully constructed in-repo
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess_creationflags(background=preview),
            )
        else:
            _run_with_progress(
                cmd,
                duration=max(0.001, float(clip.end) - float(clip.start)),
                background=preview,
                progress_callback=progress_callback or (lambda _update: None),
                cancel_check=cancel_check,
            )
    except FFmpegCancelled:
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            logger.debug("Could not delete cancelled render %s", output_path, exc_info=True)
        raise
    except subprocess.CalledProcessError as e:
        error_msg = (
            e.stderr.decode("utf-8", errors="replace")
            if isinstance(e.stderr, bytes)
            else str(e.stderr or "Unknown FFmpeg error")
        )
        print(f"FFmpeg render failed for {clip.clip_id}:\n{error_msg}")
        raise RuntimeError(f"FFmpeg export failed: {error_msg}")

    # Write metadata log alongside the MP4
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(clip.to_dict(), f, indent=2)

    return os.path.abspath(output_path)

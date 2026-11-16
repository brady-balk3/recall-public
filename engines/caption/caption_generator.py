# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import os
from typing import List

from core.models.clip import GameClip
from core.models.story import GameStory
from core.models.caption import ClipCaption
from engines.caption.asr import (
    clip_transcript_to_vod_time,
    generate_ass_for_clip,
    generate_ass_from_transcript,
    transcribe_clip_window,
)
from engines.caption.generator import generate_caption


def _transcript_has_words(transcript: dict, start: float, end: float) -> bool:
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []) or []:
            ws = float(word.get("start", 0.0))
            we = float(word.get("end", 0.0))
            if we >= start and ws <= end:
                return True
    return False

def process_clips_to_captions(clips: List[GameClip], stories: List[GameStory], video_path: str = None, temp_dir: str = None, transcript: dict = None, cancel_check=None, caption_style=None, game: str = "generic", on_caption=None, streamer_voice=None) -> List[ClipCaption]:
    """
    Main entry point for the Caption Engine.
    Converts a stream of GameClips into ClipCaptions using deterministic rules,
    and runs per-clip ASR to generate synced .ass subtitles.

    ``caption_style`` (plan 9.4) controls the burned-in subtitle look (font,
    size, colors, position). A disabled style skips subtitle rendering entirely
    so exports come out clean.

    ``on_caption``, when given, is called with each ClipCaption as soon as it
    is written. Display-only: a failing hook never costs a caption.

    ``streamer_voice`` (engines.caption.streamer_voice.StreamerVoice), when
    given and active, drops the words the streamer did not say from every
    word source here -- the per-clip decode AND the VOD transcript it falls
    back to -- so the title quote and the burned-in captions keep agreeing.
    """
    if not clips:
        return []

    from engines.caption.caption_style import resolve_caption_style
    from engines.caption.streamer_voice import filter_transcript
    style = resolve_caption_style(caption_style)
    # The VOD pass is VOD-timed: filtered once, offset 0. A fallback slice of
    # it must not bring back a teammate line the clip's own decode lost.
    transcript = filter_transcript(transcript, streamer_voice, 0.0)

    # Build a lookup dictionary for stories
    story_map = {s.story_id: s for s in stories}

    captions = []

    for clip in clips:
        if cancel_check:
            cancel_check()
        story = story_map.get(clip.story_id)
        if not story:
            continue

        # One per-clip decode serves BOTH the title quote and the burned-in
        # captions. It used to serve only the captions, while the title was
        # quoted from the VOD-level pass -- the cheapest ASR in the pipeline,
        # decoded with no context and no domain vocabulary. That is why a clip
        # could be titled "Chad, have we seen anything in this room yet?" (see
        # asr.STREAM_ASR_PROMPT), and why the card and the caption
        # burned into the video could disagree about what was said.
        clip_transcript = None
        if video_path and temp_dir:
            try:
                clip_transcript = transcribe_clip_window(
                    video_path, clip.start, clip.end,
                    os.path.join(temp_dir, f"quote_{clip.clip_id}.wav"),
                    cancel_check=cancel_check,
                )
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001 - fall back to the VOD pass
                print(f"Per-clip ASR failed for {clip.clip_id}: {exc}")
        # Whether the decode HEARD anything, before the streamer filter: a
        # clip whose words were all someone else's has no caption, and must
        # not fall back to the VOD pass below.
        decoded_words = (clip_transcript is not None
                         and _transcript_has_words(clip_transcript, 0.0, clip.end - clip.start))
        clip_transcript = filter_transcript(clip_transcript, streamer_voice, clip.start)

        caption = generate_caption(
            clip,
            story,
            transcript=(
                clip_transcript_to_vod_time(clip_transcript, clip.start)
                if clip_transcript is not None else transcript
            ),
            game=game,
        )

        # Burned-in subtitles: always prefer a per-clip medium.en pass so the
        # review deck matches export quality. Fall back to slicing the VOD
        # transcript only when per-clip ASR fails or no video path is available.
        if temp_dir and style.enabled:
            ass_output_path = os.path.join(temp_dir, f"caption_{clip.clip_id}.ass")
            # Keep a top caption off the streamer's face: both layouts put the
            # facecam at the top, so a top-positioned caption drops to center on
            # clips that have one. Every other position is returned unchanged.
            has_top_facecam = bool((getattr(clip, "layout", None) or {}).get("facecam"))
            clip_style = style.facecam_safe(has_top_facecam)
            # The clip's own decode can come back with no words where the VOD pass
            # heard speech; burning that would ship a caption file saying nothing.
            heard_nothing = (
                clip_transcript is not None
                and not decoded_words
                and transcript is not None
                and _transcript_has_words(transcript, clip.start, clip.end)
            )
            try:
                if heard_nothing:
                    generate_ass_from_transcript(
                        transcript, clip.start, clip.end, ass_output_path, style=clip_style,
                    )
                elif video_path:
                    generate_ass_for_clip(
                        video_path, clip.start, clip.end, ass_output_path,
                        cancel_check=cancel_check, style=clip_style,
                        transcript=clip_transcript,
                    )
                elif transcript is not None and _transcript_has_words(transcript, clip.start, clip.end):
                    generate_ass_from_transcript(
                        transcript, clip.start, clip.end, ass_output_path, style=clip_style,
                    )
                else:
                    ass_output_path = None
                if ass_output_path and os.path.exists(ass_output_path):
                    caption.ass_path = ass_output_path
            except InterruptedError:
                raise
            except Exception as e:
                print(f"Failed to generate ASS for clip {clip.clip_id}: {e}")
                # Per-clip failure: try slicing the VOD transcript before giving up.
                if transcript is not None and _transcript_has_words(transcript, clip.start, clip.end):
                    try:
                        generate_ass_from_transcript(
                            transcript, clip.start, clip.end, ass_output_path, style=clip_style,
                        )
                        if os.path.exists(ass_output_path):
                            caption.ass_path = ass_output_path
                    except InterruptedError:
                        raise
                    except Exception as retry_exc:  # noqa: BLE001 - captions stay best-effort
                        print(f"VOD-transcript caption fallback also failed for {clip.clip_id}: {retry_exc}")

        captions.append(caption)
        if on_caption:
            try:
                on_caption(caption)
            except Exception as exc:  # noqa: BLE001 - display hook only
                print(f"Caption hook failed for {clip.clip_id}: {exc}")

    return captions

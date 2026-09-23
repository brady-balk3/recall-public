# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import os
from typing import List

from core.models.clip import GameClip
from core.models.story import GameStory
from core.models.caption import ClipCaption
from engines.caption.whisper_asr import (
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

def process_clips_to_captions(clips: List[GameClip], stories: List[GameStory], video_path: str = None, temp_dir: str = None, transcript: dict = None, cancel_check=None, caption_style=None, game: str = "generic") -> List[ClipCaption]:
    """
    Main entry point for the Caption Engine.
    Converts a stream of GameClips into ClipCaptions using deterministic rules,
    and runs Whisper ASR to generate synced .ass subtitles.

    ``caption_style`` (plan 9.4) controls the burned-in subtitle look (font,
    size, colors, position). A disabled style skips subtitle rendering entirely
    so exports come out clean.
    """
    if not clips:
        return []

    from engines.caption.caption_style import resolve_caption_style
    style = resolve_caption_style(caption_style)

    # Build a lookup dictionary for stories
    story_map = {s.story_id: s for s in stories}

    # Built once per VOD, from regions spread across the whole stream: a single
    # clip can be entirely a friend or entirely narration, so a per-clip profile
    # would learn the wrong voice. None (no model, short VOD, extraction failure)
    # simply means no filtering.
    speaker_profile = None
    if video_path and temp_dir and transcript is not None:
        try:
            from engines.caption import speaker_id
            speaker_profile = speaker_id.profile_for_vod(
                video_path, transcript, temp_dir, cancel_check=cancel_check)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - attribution is best-effort
            print(f"Speaker profile unavailable: {exc}")

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
        # whisper_asr.STREAM_ASR_PROMPT), and why the card and the caption
        # burned into the video could disagree about what was said.
        clip_transcript = None
        if video_path and temp_dir:
            try:
                clip_transcript = transcribe_clip_window(
                    video_path, clip.start, clip.end,
                    os.path.join(temp_dir, f"quote_{clip.clip_id}.wav"),
                    cancel_check=cancel_check,
                    speaker_profile=speaker_profile,
                )
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001 - fall back to the VOD pass
                print(f"Per-clip ASR failed for {clip.clip_id}: {exc}")

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
            try:
                if video_path:
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

    return captions

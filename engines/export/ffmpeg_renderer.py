# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from core.bundle_paths import get_data_dir
import os
import concurrent.futures
from typing import List

from core.models.clip import GameClip
from core.models.caption import ClipCaption
from engines.export.renderer import render_clip

def export_clips(video_path: str, clips: List[GameClip], output_dir: str | None = None, progress_callback=None, captions: List[ClipCaption] = None, cancel_check=None, preview: bool = False) -> List[str]:
    """
    Main entry point for the Export Engine.
    Converts a stream of GameClips into physical MP4 files using FFmpeg.

    ``preview=True`` renders cheap low-res review proxies (used for the Clip
    Library's review pass during the VOD scan) instead of final-quality
    deliverables.
    """
    if output_dir is None:
        output_dir = os.path.join(get_data_dir(), 'exports')
    if not clips:
        return []
        
    # Build dictionary for quick caption lookup by clip_id
    caption_map = {}
    if captions:
        caption_map = {c.clip_id: c.ass_path for c in captions}

    exported_paths = []
    total_clips = len(clips)
    
    # We will submit all render jobs and track their completion
    # Max workers = 3 to avoid overwhelming disk I/O, adjust based on system
    completed_clips = 0
    
    def render_and_report(idx, clip):
        nonlocal completed_clips
        if cancel_check:
            cancel_check()
        ass_path = caption_map.get(clip.clip_id)
        path = render_clip(video_path, clip, output_dir, ass_path, preview=preview)
        if cancel_check:
            cancel_check()
        completed_clips += 1
        
        if progress_callback:
            prog = 0.85 + (0.10 * (completed_clips / total_clips))
            progress_callback({
                "type": "progress",
                "phase": "Export",
                "message": f"Rendered clip {completed_clips}/{total_clips}...",
                "progress": prog
            })
        return path
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i, clip in enumerate(clips):
            if cancel_check:
                cancel_check()
            futures.append(executor.submit(render_and_report, i, clip))
        try:
            for future in concurrent.futures.as_completed(futures):
                if cancel_check:
                    cancel_check()
                exported_paths.append(future.result())
        except InterruptedError:
            for future in futures:
                future.cancel()
            raise
            
    return exported_paths

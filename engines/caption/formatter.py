# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from typing import List

def format_tiktok(title: str, tags: List[str]) -> str:
    """Format for TikTok: Punchy title followed immediately by tags."""
    tag_str = " ".join([f"#{t}" for t in tags])
    return f"{title} {tag_str}"

def format_shorts(title: str, description: str, tags: List[str]) -> str:
    """Format for YouTube Shorts: Title, brief description, then tags."""
    tag_str = " ".join([f"#{t}" for t in tags])
    return f"{title}\n\n{description}\n\n{tag_str}"

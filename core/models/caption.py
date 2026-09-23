# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class ClipCaption:
    caption_id: str
    clip_id: str
    title: str
    description: str
    tags: List[str]
    tiktok_format: str
    shorts_format: str
    ass_path: str = None

    def to_dict(self):
        return asdict(self)

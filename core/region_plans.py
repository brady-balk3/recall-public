# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Pure helpers for handing creator-priority regions to scan engines."""

from __future__ import annotations


def merge_regions(regions, duration, gap=12.0, max_region=90.0):
    if not regions:
        return []
    normalized = sorted(
        (max(0.0, float(start)), min(float(duration), float(end)))
        for start, end in regions
        if float(end) > float(start)
    )
    merged = []
    for start, end in normalized:
        end = min(end, start + max_region)
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap and max(prev_end, end) - prev_start <= max_region:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def merge_priority_regions(priority_regions, ordinary_regions, duration, budget_seconds):
    """Keep creator regions first, then fill the remaining Smart Scan budget."""
    if not priority_regions:
        return ordinary_regions
    priority = merge_regions(
        priority_regions, duration, gap=12.0, max_region=180.0,
    )
    selected = list(priority)
    priority_seconds = sum(end - start for start, end in selected)
    budget = max(float(budget_seconds), priority_seconds)
    for region in ordinary_regions or []:
        candidate = merge_regions(
            selected + [region], duration, gap=12.0, max_region=180.0,
        )
        candidate_seconds = sum(end - start for start, end in candidate)
        if candidate_seconds <= budget + 1e-6:
            selected = candidate
    return selected

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Allowlisted support reports: recording content never enters the formatter."""

import json
import math


STATUSES = frozenset({"pending", "queued", "running", "cancelling", "cancelled", "completed", "failed"})
PHASES = frozenset({
    "Init", "Resolving Input", "Perception", "Reaction", "Event", "Story",
    "Clip", "Caption", "Export", "Complete", "Cancelled", "Cancelling",
    "Failed", "Memory", "Rebuild clips",
})
EVENTS = frozenset({
    "job_started", "job_completed", "job_failed", "job_cancelled",
    "stage_started", "stage_completed", "stage_progress", "clip_found",
    "semantic_judge_health", "semantic_judge_degraded", "visual_judge_health",
    "visual_judge_degraded", "visual_judge_skipped", "memory_indexed",
    "memory_index_failed", "clip_rebuild_started", "clip_rebuild_completed",
})


def _enum(value, allowed):
    return value if isinstance(value, str) and value in allowed else "unknown"


def _number(value):
    # Never coerce user-controlled strings or objects into the report.
    if type(value) in (int, float) and 0 <= value <= 1e12 and math.isfinite(value):
        return round(value, 3)
    return "unknown"


def support_report(job: dict, events, *, packaged: bool, cpu_count=None, ram_gb=None) -> str:
    """Keep operational facts, omit identifiers, source times and free text.

    Unknown fields are omitted automatically, including future nested payloads.
    This is intentionally not a regex scrub of arbitrary logs or transcripts.
    """
    lines = [
        "=== Recall diagnostics ===",
        "privacy: source identifiers, paths, messages, logs and recording content omitted",
        f"packaged_app: {bool(packaged)}",
        f"cpu_count: {_number(cpu_count)}",
        f"available_ram_gb: {_number(ram_gb)}",
        "", "--- job ---",
        f"status: {_enum(job.get('status'), STATUSES)}",
        f"stage: {_enum(job.get('current_stage'), PHASES)}",
    ]
    for key in ("progress", "stage_progress", "duration", "elapsed_seconds", "clips_found"):
        lines.append(f"{key}: {_number(job.get(key))}")
    timings = job.get("stage_timings")
    if isinstance(timings, str):
        try:
            timings = json.loads(timings)
        except (ValueError, TypeError):
            timings = None
    if isinstance(timings, dict):
        lines.extend(["", "--- stage durations (seconds) ---"])
        for phase in sorted(PHASES):
            entry = timings.get(phase)
            if isinstance(entry, dict):
                lines.append(f"{phase}: {_number(entry.get('duration_seconds'))}")
    lines.extend(["", "--- recent events (newest first) ---"])
    for event in list(events or [])[:60]:
        if not isinstance(event, dict):
            continue
        lines.append(
            f"{_enum(event.get('event_type'), EVENTS)} "
            f"phase={_enum(event.get('phase'), PHASES)} "
            f"progress={_number(event.get('progress'))}"
        )
    return "\n".join(lines)

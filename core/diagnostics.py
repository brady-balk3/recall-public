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


JUDGE_HEALTH_EVENTS = (
    "semantic_judge_health", "semantic_judge_degraded",
    "visual_judge_health", "visual_judge_degraded", "visual_judge_skipped",
)
JUDGE_STATUSES = frozenset({"healthy", "degraded", "skipped", "disabled", "failed"})
# Skip reasons are fixed strings written by the pipeline itself, never content.
JUDGE_REASONS = frozenset({"vision model files not found", "vision model discovery failed"})
DEVICES = frozenset({"cpu", "cuda", "mps", "xpu", "directml"})
# Judge health counters (engines/semantic/judge.py, visual_judge.py). Anything
# else in the payload -- model paths, recent failure rows with source times and
# error text -- stays out.
JUDGE_COUNTERS = (
    "candidates_received", "eligible_candidates", "judged_candidates",
    "fallback_candidates", "batch_attempts", "batch_failures",
    "generation_attempts", "generation_failures", "context_overflows",
    "invalid_outputs", "candidate_budget", "candidates_attempted",
    "unique_candidates_attempted", "cache_hits", "failures", "failure_attempts",
    "false_skip_rejudges", "verbal_payoff_rejudges", "verbal_payoff_rejects",
    "remaining_budget", "frame_count",
)


def _device(value):
    if isinstance(value, str):
        base = value.split(":", 1)[0].strip().lower()
        if base in DEVICES:
            return base
    return "unknown"


def _payload(event) -> dict:
    payload = event.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return {}
    return payload if isinstance(payload, dict) else {}


def _judge_health_lines(job: dict, health_events) -> list:
    health = job.get("scan_health") if isinstance(job.get("scan_health"), dict) else {}
    lines = ["", "--- scan health ---",
             f"degraded: {health.get('degraded') is True}",
             f"device: {_device(health.get('device'))}"]
    for judge in ("semantic", "visual"):
        state = health.get(judge) if isinstance(health.get(judge), dict) else {}
        reason = state.get("reason")
        line = f"{judge}: status={_enum(state.get('status'), JUDGE_STATUSES)} degraded={state.get('degraded') is True}"
        if reason in JUDGE_REASONS:
            line += f" reason={reason}"
        lines.append(line)
    for event in list(health_events or [])[:10]:
        if not isinstance(event, dict) or event.get("event_type") not in JUDGE_HEALTH_EVENTS:
            continue
        payload = _payload(event)
        counters = {key: _number(payload[key]) for key in JUDGE_COUNTERS if key in payload}
        extras = [f"degraded={payload.get('degraded') is True}"]
        if payload.get("status") is not None:
            extras.append(f"status={_enum(payload.get('status'), JUDGE_STATUSES)}")
        if payload.get("reason") in JUDGE_REASONS:
            extras.append(f"reason={payload['reason']}")
        lines.append(f"{event['event_type']} {' '.join(extras)} {json.dumps(counters, sort_keys=True)}")
    return lines


def _enum(value, allowed):
    return value if isinstance(value, str) and value in allowed else "unknown"


def _number(value):
    # Never coerce user-controlled strings or objects into the report.
    if type(value) in (int, float) and 0 <= value <= 1e12 and math.isfinite(value):
        return round(value, 3)
    return "unknown"


def _failed_stage(event) -> str:
    """The stage a job_failed event recorded as running, from the allowlist only."""
    return _enum(_payload(event).get("failed_stage"), PHASES)


def support_report(job: dict, events, *, packaged: bool, cpu_count=None, ram_gb=None,
                   health_events=None) -> str:
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
    events = [event for event in list(events or [])[:60] if isinstance(event, dict)]
    # A failed job's stage reads "Failed"; where it actually stopped is the
    # first thing support needs, and a stage name is not recording content.
    failures = [event for event in events if event.get("event_type") == "job_failed"]
    if failures:
        lines.append(f"failed_during: {_failed_stage(failures[0])}")
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
    # Judge health comes from its own lookup: a long export's progress events
    # would otherwise push it out of the recent-events tail.
    if job.get("scan_health") or health_events:
        lines.extend(_judge_health_lines(job, health_events))
    lines.extend(["", "--- recent events (newest first) ---"])
    for event in events:
        lines.append(
            f"{_enum(event.get('event_type'), EVENTS)} "
            f"phase={_enum(event.get('phase'), PHASES)} "
            f"progress={_number(event.get('progress'))}"
        )
    return "\n".join(lines)

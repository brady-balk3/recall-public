// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
export interface DiagnosticJob {
  id: string;
  url: string;
  status: string;
  progress: number;
  createdAt: number;
  stage?: { label?: string; progress?: number };
  currentPhase?: string;
  message?: string;
  errorMessage?: string;
  elapsedSeconds?: number;
  vodDuration?: number;
  clipsFound?: number;
}

/**
 * A useful offline report for failures that happen before the API creates a
 * permanent job row. This is also the fallback when a persisted job's richer
 * backend diagnostics cannot be reached after a restart.
 */
export function buildLocalDiagnostics(job: DiagnosticJob): string {
  const statuses = new Set(["pending", "queued", "running", "cancelling", "cancelled", "completed", "failed"]);
  const phases = new Set(["Init", "Resolving Input", "Perception", "Reaction", "Event", "Story", "Clip", "Caption", "Export", "Complete", "Cancelled", "Cancelling", "Failed", "Memory"]);
  const number = (value: unknown): number | "unknown" =>
    typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1e12
      ? Math.round(value * 1000) / 1000 : "unknown";
  const phase = job.currentPhase;
  const lines = [
    "=== Recall diagnostics (desktop fallback) ===",
    "privacy: source identifiers, paths, messages and recording content omitted",
    "backend_report_available: false",
    "",
    "--- job ---",
    `status: ${statuses.has(job.status) ? job.status : "unknown"}`,
    `stage: ${phase && phases.has(phase) ? phase : "unknown"}`,
    `stage_progress: ${number(job.stage?.progress)}`,
    `overall_progress: ${number(job.progress)}`,
    `elapsed_seconds: ${number(job.elapsedSeconds)}`,
    `vod_duration: ${number(job.vodDuration)}`,
    `clips_found: ${number(job.clipsFound)}`,
    "",
    "note: The scan failed before Recall could retrieve its full backend report, or the local studio was unavailable when this report was copied.",
  ];
  return lines.join("\n");
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import type { ExportOperation, Job, JobStatus, Session } from "./store";

const ACTIVE_SCAN_STATUSES = new Set<JobStatus>(["queued", "analyzing", "detecting", "assembling"]);
const TERMINAL_SCAN_STATUSES = new Set<JobStatus>(["completed", "failed", "cancelled"]);

export function isActiveScanStatus(status: JobStatus | undefined): boolean {
  return !!status && ACTIVE_SCAN_STATUSES.has(status);
}

export function isTerminalScanStatus(status: JobStatus | undefined): boolean {
  return !!status && TERMINAL_SCAN_STATUSES.has(status);
}

export function hasActiveDesktopWork(
  jobs: Job[],
  sessions: Session[],
  exportOperation?: ExportOperation,
): boolean {
  return jobs.some((job) => isActiveScanStatus(job.status))
    || sessions.some((session) => isActiveScanStatus(session.status))
    || exportOperation?.status === "queued"
    || exportOperation?.status === "running";
}

export function scanNotification(name: string, status: JobStatus): { title: string; body: string } | null {
  if (status === "completed") return { title: "Scan finished", body: `${name} is ready for Theater Review.` };
  if (status === "failed") return { title: "Scan needs attention", body: `${name} could not finish. Open Recall for details and recovery steps.` };
  return null;
}

export function batchNotification(statuses: JobStatus[]): { title: string; body: string } {
  const failures = statuses.filter((status) => status === "failed").length;
  const completed = statuses.filter((status) => status === "completed").length;
  if (failures) {
    return {
      title: "Batch finished with issues",
      body: `${completed} ${completed === 1 ? "scan is" : "scans are"} ready and ${failures} ${failures === 1 ? "scan needs" : "scans need"} attention.`,
    };
  }
  return {
    title: "Batch finished",
    body: `${completed} ${completed === 1 ? "scan is" : "scans are"} ready for review.`,
  };
}

export function exportNotification(operation: ExportOperation): { title: string; body: string } | null {
  const label = operation.kind === "reel" ? "Highlight reel" : "Clip export";
  if (operation.status === "completed") {
    return { title: `${label} finished`, body: operation.kind === "reel" ? "Your reel is ready." : `${operation.copied} ${operation.copied === 1 ? "file is" : "files are"} ready.` };
  }
  if (operation.status === "partial") {
    return { title: `${label} finished with issues`, body: `${operation.copied} succeeded and ${operation.errors} failed. Open Recall for details.` };
  }
  if (operation.status === "failed") {
    return { title: `${label} failed`, body: "Recall could not finish the export. Open the app for the failure details." };
  }
  return null;
}

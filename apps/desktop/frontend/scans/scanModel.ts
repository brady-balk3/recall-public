// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** Scan state for creators: where a scan is in the queue, and its activity in plain words. */
import type { Job, JobEvent, Session } from "../lib/store";

export function creatorActivity(message: string, eventType: string, stageLabel?: string) {
  if (/^Device summary:|^Video metadata:/i.test(message)) return null;
  const download = message.match(/Downloading Twitch VOD:\s*(\d+)%/i);
  if (download) {
    const pct = Number(download[1]);
    return pct % 10 === 0 || pct === 100 ? `Downloading Twitch VOD · ${pct}%` : null;
  }
  const finalizing = message.match(/Finalizing Video\s+(\d+)%/i);
  if (finalizing) {
    const pct = Number(finalizing[1]);
    return pct % 25 === 0 || pct === 100 ? `Preparing downloaded video · ${pct}%` : null;
  }
  // Progress heartbeats fire every few seconds with a changing % — the live
  // "current" line and the ring already carry that number, so the running log
  // collapses them all to one steady milestone instead of a flood of pulses.
  if (/% scanned/i.test(message) || /of the VOD covered/i.test(message)) return "Watching your VOD for big moments";
  if (/Perceiving in parallel/i.test(message)) return "Scanning gameplay, reactions, and on-screen moments";
  if (/Extracting audio/i.test(message)) return "Preparing audio for reaction analysis";
  if (/Found cached signals/i.test(message)) return "Reusing the existing analysis for this recording";
  if (eventType === "clip_found") return "Found a moment worth reviewing";
  if (eventType === "stage_completed" && stageLabel) return `${stageLabel} complete`;
  return message;
}

export type ActivityItem = { key: string; timestamp: string; text: string; detail: string; ok: boolean };

// Turn a job's persisted event stream into a creator-facing activity log.
// `events` arrive newest-first. We collapse only *consecutive* identical lines
// so repeated milestones (e.g. the scan heartbeat) show once while genuine
// progression stays visible — the whole history is kept so it never feels like
// activity is being thrown away. Used by both the live feed and the scan log.
export function activityFromEvents(events: JobEvent[], jobId?: string, limit = 40): ActivityItem[] {
  const items: ActivityItem[] = [];
  let lastText = "";
  // The live merge path is ordered already, but restored snapshots and test
  // fixtures are not required to be. "Latest" must remain newest-first.
  const ordered = events
    .filter((entry) => !jobId || entry.jobId === jobId)
    .sort((a, b) => b.at - a.at);
  for (const event of ordered) {
    const raw = event.message || (event.eventType === "stage_completed" ? event.stageLabel || event.phase || "Stage" : "");
    if (!raw) continue;
    const text = creatorActivity(raw, event.eventType, event.stageLabel);
    if (!text || text === lastText) continue;
    lastText = text;
    items.push({
      key: `${event.eventType}-${event.at}`,
      timestamp: new Date(event.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }),
      text,
      detail: event.stageLabel || event.phase || "",
      ok: event.eventType === "stage_completed" || event.eventType === "clip_found",
    });
    if (items.length >= limit) break;
  }
  return items;
}

export function activityItems(job: Job | undefined) {
  return activityFromEvents(job?.events ?? [], job?.id, 24);
}

/** One line of the live feed: a task and the latest word on it. Repeated
 * heartbeats ("Transcribing audio · 1:02:00 / 3:26:16") fold into the task
 * they belong to instead of stacking up as new lines. */
export type ActivityTask = {
  key: string;
  label: string;
  detail: string;
  startedAt: number;
  endedAt: number;
  ok: boolean;
};

/** "Transcribing audio · 3:25:41 / 3:26:16 · 20.8×" -> label and detail. */
export function splitActivity(text: string): [string, string] {
  const trimmed = text.trim().replace(/(\.\.\.|…)$/, "");
  const dot = trimmed.indexOf(" · ");
  if (dot > 0) return [trimmed.slice(0, dot), trimmed.slice(dot + 3)];
  const paren = trimmed.match(/^(.+?)\s*\(([^()]+)\)$/);
  if (paren) return [paren[1], paren[2]];
  return [trimmed, ""];
}

/** How far along a task's detail says it is, 0 to 1; null when it doesn't. */
export function activityFraction(detail: string): number | null {
  const pct = detail.match(/(\d+(?:\.\d+)?)%/);
  if (pct) return Math.max(0, Math.min(1, Number(pct[1]) / 100));
  const span = detail.match(/([\d:]+)\s*\/\s*([\d:]+)/);
  if (span) {
    const done = timestampSeconds(span[1]);
    const total = timestampSeconds(span[2]);
    if (done != null && total) return Math.max(0, Math.min(1, done / total));
  }
  return null;
}

/** The job's events as tasks, newest first. A task ends when the next starts. */
export function foldActivity(events: JobEvent[], jobId?: string, limit = 14): ActivityTask[] {
  const tasks: ActivityTask[] = [];
  const ordered = events
    .filter((entry) => !jobId || entry.jobId === jobId)
    .sort((a, b) => a.at - b.at);
  for (const event of ordered) {
    const raw = event.message || (event.eventType === "stage_completed" ? event.stageLabel || event.phase || "Stage" : "");
    if (!raw) continue;
    const text = creatorActivity(raw, event.eventType, event.stageLabel);
    if (!text) continue;
    const [label, detail] = splitActivity(text);
    const ok = event.eventType === "stage_completed" || event.eventType === "clip_found";
    const last = tasks[tasks.length - 1];
    if (last && last.label === label) {
      last.detail = detail || last.detail;
      last.endedAt = event.at;
      last.ok = last.ok || ok;
      continue;
    }
    if (last) last.endedAt = event.at;
    tasks.push({ key: `${event.eventType}-${event.at}`, label, detail, startedAt: event.at, endedAt: event.at, ok });
  }
  return tasks.reverse().slice(0, limit);
}

export type QueueLaneState = "running" | "waiting" | "done" | "failed" | "cancelled";

export function queueLane(job: Job | undefined, session: Session | undefined): QueueLaneState {
  const status = job?.status ?? session?.status ?? "queued";
  if (status === "completed") return "done";
  if (status === "failed") return "failed";
  if (status === "cancelled") return "cancelled";
  const message = (job?.message || session?.message || "").toLowerCase();
  if (status === "queued" || /waiting for an earlier scan/.test(message)) return "waiting";
  return "running";
}

/** Seconds from an "H:MM:SS" or "M:SS" stamp; null when it isn't one. */
export function timestampSeconds(timestamp: string) {
  const parts = timestamp.split(":").map((part) => Number(part));
  if (!parts.length || parts.some((part) => !Number.isFinite(part) || part < 0)) return null;
  return parts.reduce((total, part) => total * 60 + part, 0);
}

/** Which of the five creator-facing scan stages a job is in, 0 to 4. */
export function stageIndex(job?: Job) {
  if (!job) return 0;
  const phase = `${job.currentPhase ?? ""} ${job.stage?.label ?? ""}`.toLowerCase();
  if (/export|render|preview|package/.test(phase)) return 4;
  if (/caption|writ/.test(phase)) return 3;
  if (/reaction|event|story|clip|select|choos|score|rank|moment/.test(phase)) return 2;
  if (/perception|scan|vision|watch|face|game|ocr|audio|listen|speech/.test(phase)) return 1;
  return Math.min(4, Math.floor((job.progress ?? 0) / 20));
}

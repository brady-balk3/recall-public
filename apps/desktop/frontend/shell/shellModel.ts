// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** What the shell shows at a glance: counts, the running scan, the profile line. */
import type { Job, JobStatus, Session } from "../lib/store";
import { primaryReviewDeckClips } from "../lib/reviewDeck";

const RUNNING: JobStatus[] = ["analyzing", "detecting", "assembling"];

export interface RunningScan {
  id: string;
  title: string;
  /** 0-100 */
  progress: number;
}

export interface ShellCounts {
  /** Library size: every session. */
  sessions: number;
  /** Clips still waiting for a keep or pass, across finished sessions' review decks. */
  toReview: number;
  /** Kept clips that have not been exported yet. */
  toExport: number;
  /** Every kept clip. */
  keepers: number;
  /** Scans waiting behind the running one. */
  queued: number;
  scan?: RunningScan;
}

function undecided(session: Session) {
  return primaryReviewDeckClips(session.clips).filter(
    (clip) => !clip.kept && !clip.passed && !session.savedClipIds.includes(clip.id),
  ).length;
}

export function shellCounts(sessions: Session[], jobs: Job[]): ShellCounts {
  let toReview = 0;
  let toExport = 0;
  let keepers = 0;
  for (const session of sessions) {
    if (session.status === "completed") toReview += undecided(session);
    keepers += session.savedClipIds.length;
    toExport += session.savedClipIds.filter((id) => !session.exportedClipIds.includes(id)).length;
  }
  const runningJob = jobs.find((job) => RUNNING.includes(job.status));
  const scan = runningJob
    ? {
        id: runningJob.id,
        title: sessions.find((session) => session.id === runningJob.id)?.name ?? "Your stream",
        progress: Math.max(0, Math.min(100, runningJob.progress || 0)),
      }
    : undefined;
  return {
    sessions: sessions.length,
    toReview,
    toExport,
    keepers,
    queued: jobs.filter((job) => job.status === "queued").length,
    scan,
  };
}

/** "24 streams · 82 keepers", or a nudge before there is anything to count. */
export function profileLine(counts: Pick<ShellCounts, "sessions" | "keepers">) {
  if (!counts.sessions) return "Drop in your first stream";
  const streams = `${counts.sessions} ${counts.sessions === 1 ? "stream" : "streams"}`;
  const keepers = `${counts.keepers} ${counts.keepers === 1 ? "keeper" : "keepers"}`;
  return `${streams} · ${keepers}`;
}

/** The letter on the avatar: the first letter of the name the creator typed. */
export function avatarLetter(name: string) {
  const letter = Array.from(name.trim())[0];
  return letter ? letter.toLocaleUpperCase() : "R";
}

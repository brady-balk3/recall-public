// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * What Home puts in front of the creator, as pure functions: the moment to
 * watch next, what is waiting, the week so far, and the next milestone.
 */
import type { Clip, Session } from "../lib/store";
import { durationOf, primaryReviewDeckClips, rankScoreOf } from "../lib/reviewDeck";

const DAY = 86_400_000;

/** Hype is the deck score on a 0-99 scale, the number creators see on every card. */
export function hypeOf(clip: Clip) {
  return Math.round(rankScoreOf(clip));
}

const isDecided = (session: Session, clip: Clip) =>
  !!clip.kept || !!clip.passed || session.savedClipIds.includes(clip.id);

/** Review-deck clips still waiting for a keep or pass, loudest first. */
export function undecidedClips(session: Session): Clip[] {
  if (session.status !== "completed") return [];
  return primaryReviewDeckClips(session.clips)
    .filter((clip) => !isDecided(session, clip))
    .sort((a, b) => hypeOf(b) - hypeOf(a));
}

const newest = (a: Session, b: Session) => (b.createdAt || b.updatedAt) - (a.createdAt || a.updatedAt);

/** Finished sessions with something left to review, newest first. */
export function waitingSessions(sessions: Session[]): Session[] {
  return sessions.filter((session) => undecidedClips(session).length > 0).sort(newest);
}

export interface UpNext {
  session: Session;
  /** The loudest undecided moment: the one to watch first. */
  lead: Clip;
  /** Up to three cards for the fan, lead in the middle. */
  fan: Clip[];
  inDeck: number;
  waitingOverall: number;
  /** Whole minutes of footage left across everything waiting. */
  minutesToClear: number;
}

export function upNext(sessions: Session[]): UpNext | undefined {
  const waiting = waitingSessions(sessions);
  const session = waiting[0];
  if (!session) return undefined;
  const deck = undecidedClips(session);
  const all = waiting.flatMap(undecidedClips);
  const seconds = all.reduce((sum, clip) => sum + durationOf(clip), 0);
  const [lead, ...rest] = deck;
  // Fan order is left, front, right.
  const fan = rest.length >= 2 ? [rest[0], lead, rest[1]] : rest.length === 1 ? [lead, rest[0]] : [lead];
  return {
    session,
    lead,
    fan,
    inDeck: deck.length,
    waitingOverall: all.length,
    minutesToClear: Math.max(1, Math.round(seconds / 60)),
  };
}

/** Everything else, newest first: the library at a glance. */
export function recentSessions(sessions: Session[], exclude: Session[], limit = 6): Session[] {
  const skip = new Set(exclude.map((session) => session.id));
  return sessions.filter((session) => !skip.has(session.id)).sort(newest).slice(0, limit);
}

export interface WeekStats {
  streams: number;
  keepers: number;
  topHype: number;
}

export function weekStats(sessions: Session[], now = Date.now()): WeekStats {
  const week = sessions.filter((session) => now - (session.createdAt || session.updatedAt) < 7 * DAY);
  const kept = week.flatMap((session) => session.clips.filter((clip) => session.savedClipIds.includes(clip.id)));
  return {
    streams: week.length,
    keepers: kept.length,
    topHype: kept.reduce((top, clip) => Math.max(top, hypeOf(clip)), 0),
  };
}

const MILESTONES = [10, 25, 50, 100, 250, 500, 1000];

export interface Milestone {
  target: number;
  remaining: number;
  /** 0-1 toward the target. */
  progress: number;
}

export function keeperMilestone(keepers: number): Milestone {
  const target = MILESTONES.find((m) => m > keepers) ?? (Math.floor(keepers / 500) + 1) * 500;
  return { target, remaining: target - keepers, progress: Math.min(1, keepers / target) };
}

export function ordinal(n: number) {
  const mod100 = n % 100;
  const suffix = mod100 >= 11 && mod100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th";
  return `${n}${suffix}`;
}

function partOfDay(hour: number) {
  if (hour < 5) return "night";
  if (hour < 12) return "morning";
  if (hour < 17) return "afternoon";
  if (hour < 21) return "evening";
  return "night";
}

export function greeting(date = new Date()) {
  const part = partOfDay(date.getHours());
  if (part === "night") return "Up late";
  return `Good ${part}`;
}

/** "Saturday night · 3 streams ready" */
export function whenLine(waiting: number, date = new Date()) {
  const day = date.toLocaleDateString(undefined, { weekday: "long" });
  const ready = waiting ? `${waiting} ${waiting === 1 ? "stream" : "streams"} ready` : "All caught up";
  return `${day} ${partOfDay(date.getHours())} · ${ready}`;
}

export type ChecklistStep = "scan" | "review" | "export" | "cut" | "live" | "accent";

export interface ChecklistItem {
  id: ChecklistStep;
  label: string;
  done: boolean;
}

export function checklist(sessions: Session[], accentChosen: boolean): ChecklistItem[] {
  const clips = sessions.flatMap((session) => session.clips);
  return [
    { id: "scan", label: "Scan your first stream", done: sessions.some((s) => s.status === "completed") },
    { id: "review", label: "Review a deck", done: clips.some((c) => c.kept || c.passed) || sessions.some((s) => s.savedClipIds.length > 0) },
    { id: "export", label: "Export a clip", done: sessions.some((s) => s.exportedClipIds.length > 0) },
    { id: "cut", label: "Cut a moment Recall missed", done: clips.some((c) => c.isManual) },
    { id: "live", label: "Mark a moment with Recall Live", done: sessions.some((s) => !!s.recallSessionId) },
    { id: "accent", label: "Make it yours with an accent", done: accentChosen },
  ];
}

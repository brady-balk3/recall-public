// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Keeper milestones: the one bigger beat in the product. Each is celebrated
 * once, when it's crossed live; milestones already behind you when Recall
 * starts are recorded quietly, never replayed.
 */
import { create } from "zustand";
import type { Clip } from "../lib/store";
import { PET_GEAR, PET_UNLOCK_2 } from "../pets/petArt";

export const MILESTONES = [1, 10, 25, 50, 100, 150, 250, 500, 1000];
export const MILESTONES_STORAGE_KEY = "recall-milestones";

export function readAchieved(): number[] {
  try {
    const raw = JSON.parse(localStorage.getItem(MILESTONES_STORAGE_KEY) || "[]");
    return Array.isArray(raw) ? raw.filter((n) => Number.isFinite(n)) : [];
  } catch {
    return [];
  }
}

export function writeAchieved(list: number[]) {
  try { localStorage.setItem(MILESTONES_STORAGE_KEY, JSON.stringify([...new Set(list)].sort((a, b) => a - b))); } catch {}
}

/** The highest milestone crossed going from `before` to `after` keepers, if any. */
export function crossedMilestone(before: number, after: number, achieved: number[] = []): number | undefined {
  const hits = MILESTONES.filter((m) => before < m && after >= m && !achieved.includes(m));
  return hits[hits.length - 1];
}

/** What a milestone unlocks for the crew, in the hype-friend voice. */
export function milestoneUnlock(n: number): string | undefined {
  if (n === PET_UNLOCK_2) return "Your second crew slot is open";
  const gear = PET_GEAR.find(([, , need]) => need === n && need > 0);
  return gear ? `New gear for your crew: ${gear[1].toLowerCase()}` : undefined;
}

export function milestoneLine(n: number) {
  if (n === 1) return "Your first keeper. The first of a lot.";
  if (n < 50) return `${n} moments worth posting. You're finding your rhythm.`;
  return `${n} keepers. That's a highlight reel most streamers never get to.`;
}

interface MomentsStore {
  celebration: { n: number; recent: Clip[] } | null;
  recapOpen: boolean;
  celebrate: (n: number, recent: Clip[]) => void;
  dismiss: () => void;
  openRecap: () => void;
  closeRecap: () => void;
}

export const useMoments = create<MomentsStore>((set) => ({
  celebration: null,
  recapOpen: false,
  celebrate: (n, recent) => set({ celebration: { n, recent } }),
  dismiss: () => set({ celebration: null }),
  openRecap: () => set({ celebration: null, recapOpen: true }),
  closeRecap: () => set({ recapOpen: false }),
}));

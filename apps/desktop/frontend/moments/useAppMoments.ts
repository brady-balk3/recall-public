// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Watches the library for the moments worth reacting to: a new keeper, a 90+
 * keeper, a milestone, a finished scan. The crew reacts; milestones also get
 * the celebration card.
 *
 * The first pass after the library loads only takes a baseline, so opening
 * Recall never replays old news. Before that load the library is empty, and
 * a baseline taken then turned the load itself into a 0 -> 1000 "milestone".
 */
import { useEffect, useRef } from "react";
import type { Job, Session } from "../lib/store";
import { hypeOf } from "../home/homeModel";
import { primaryReviewDeckClips } from "../lib/reviewDeck";
import { PET_LINES, petPick } from "../pets/petArt";
import { petsReact } from "../pets/petStore";
import { crossedMilestone, milestoneUnlock, readAchieved, useMoments, writeAchieved, MILESTONES } from "./milestones";

export function useAppMoments(sessions: Session[], jobs: Job[], loaded = true) {
  const saved = useRef<Set<string> | null>(null);
  const statuses = useRef<Map<string, string> | null>(null);
  const celebrate = useMoments((state) => state.celebrate);

  useEffect(() => {
    if (!loaded) return;
    const ids = new Set(sessions.flatMap((s) => s.savedClipIds));
    const before = saved.current;
    saved.current = ids;
    if (!before) {
      // Milestones already behind this library are recorded, not celebrated.
      const passed = MILESTONES.filter((m) => ids.size >= m);
      if (passed.length) writeAchieved([...readAchieved(), ...passed]);
      return;
    }
    const fresh = [...ids].filter((id) => !before.has(id));
    if (!fresh.length) return;
    const clips = sessions.flatMap((s) => s.clips);
    const newest = fresh.map((id) => clips.find((c) => c.id === id)).filter((c): c is NonNullable<typeof c> => !!c);
    const achieved = readAchieved();
    const hit = crossedMilestone(before.size, ids.size, achieved);
    if (hit) {
      writeAchieved([...achieved, hit]);
      const recent = [...newest, ...clips.filter((c) => ids.has(c.id) && !fresh.includes(c.id))].slice(0, 5);
      celebrate(hit, recent);
      petsReact("unlock", milestoneUnlock(hit) ? `${milestoneUnlock(hit)}!` : `${hit} ${hit === 1 ? "keeper" : "keepers"}!`);
      return;
    }
    const loudest = newest.reduce((top, c) => Math.max(top, hypeOf(c)), 0);
    petsReact(loudest >= 90 ? "wild" : "keep");
  }, [sessions, loaded, celebrate]);

  useEffect(() => {
    const now = new Map(jobs.map((j) => [j.id, j.status]));
    const before = statuses.current;
    statuses.current = now;
    if (!before) return;
    for (const job of jobs) {
      const was = before.get(job.id);
      if (job.status === "completed" && was && was !== "completed") {
        const session = sessions.find((s) => s.id === job.id);
        const n = primaryReviewDeckClips(session?.clips ?? job.clips ?? []).length;
        petsReact("scan", n ? `${petPick(PET_LINES.scan)} ${n} moments` : petPick(PET_LINES.scan));
      }
    }
  }, [jobs, sessions]);
}

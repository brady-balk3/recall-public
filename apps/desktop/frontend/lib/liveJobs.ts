// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Reading scan jobs without redrawing the app for every progress event.
 *
 * A running scan patches its job several times a second (progress, telemetry,
 * speech lanes, candidate verdicts). The shell only acts on which jobs exist
 * and their status, so it reads useJobList; the numbers are read by the views
 * that draw them, through useJobStoreWhileShown so a page kept mounted behind
 * another doesn't redraw for a scan nobody is looking at.
 */
import { useMemo, useRef } from "react";
import { useShallow } from "zustand/react/shallow";
import { useViewActive } from "../shell/viewActive";
import { useJobStore, type Job } from "./store";

type JobState = ReturnType<typeof useJobStore.getState>;

/**
 * The store's jobs array, handed out again only when a job comes or goes,
 * changes status, or gets its clips. Its other fields (progress, stage, ETA,
 * events) are as of that moment: read those where they're drawn.
 */
export function useJobList(): Job[] {
  const key = useJobStore(useShallow((state) => state.jobs.flatMap((job) => [job.id, job.status, job.clips])));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => useJobStore.getState().jobs, [key]);
}

/**
 * useJobStore for a page that stays mounted behind others (ViewPanel). While
 * the page is hidden the value holds and the page doesn't re-render; shown
 * again, it reads the store fresh. The selector must return a value the store
 * owns (a job, an array), a primitive, or an object kept stable by useShallow.
 */
export function useJobStoreWhileShown<T>(selector: (state: JobState) => T): T {
  const shown = useViewActive();
  const last = useRef<{ value: T } | null>(null);
  const value = useJobStore((state) => (shown || !last.current ? selector(state) : last.current.value));
  last.current = { value };
  return value;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The pages that draw a running scan's numbers, reading those numbers from the
 * store themselves. The shell passes them everything else, as before.
 *
 * Scan events land several times a second. When StudioApp read the jobs, each
 * one redrew the shell, the sidebar and the open page; now only the page on
 * screen (and the sidebar's scan card) redraws, and a page kept behind another
 * catches up when it's shown (useJobStoreWhileShown).
 */
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useJobStoreWhileShown } from "../lib/liveJobs";
import type { Job } from "../lib/store";
import { HomeView, type HomeViewProps } from "../home/HomeView";
import { queueWaitSeconds } from "../scans/estimate";
import { LiveScanView, type LiveScanViewProps } from "../scans/LiveScanView";
import { NewScanView, type NewScanViewProps } from "../scans/NewScanView";
import { ScansView, type ScansViewProps } from "../scans/ScansView";
import { runningScanJob, scanProgress } from "./shellModel";

/** Home with the running scan's bar. */
export function LiveHomeView(props: Omit<HomeViewProps, "scan">) {
  // Just what the scan bar draws: Home is a heavy page, and most scan events
  // (lanes, verdicts) change none of it.
  const running = useJobStoreWhileShown(useShallow((state) => {
    const job = runningScanJob(state.jobs);
    return job
      ? { id: job.id, progress: scanProgress(job), stage: job.stage?.label, etaSeconds: job.etaSeconds, clips: job.clips }
      : undefined;
  }));
  const { sessions } = props;
  const scan = useMemo(() => running && {
    title: sessions.find((session) => session.id === running.id)?.name ?? "Your stream",
    progress: running.progress,
    stage: running.stage,
    etaSeconds: running.etaSeconds,
    found: running.clips ?? [],
  }, [running, sessions]);
  return <HomeView {...props} scan={scan} />;
}

/** The import page, with how long a new scan would wait behind the queue. */
export function LiveNewScanView(props: Omit<NewScanViewProps, "queueWait">) {
  const jobs = useJobStoreWhileShown((state) => state.jobs);
  const sessions = useJobStoreWhileShown((state) => state.sessions);
  return <NewScanView {...props} queueWait={queueWaitSeconds(props.rates, jobs, sessions)} />;
}

/** The scans page, with every job's live numbers. */
export function LiveScansView(props: Omit<ScansViewProps, "jobs">) {
  const jobs = useJobStoreWhileShown((state) => state.jobs);
  return <ScansView {...props} jobs={jobs} />;
}

/** One scan up close: the job StudioApp picked, read live. */
export function ProcessingScanView({ pick, onRetryJob, ...props }: Omit<LiveScanViewProps, "job" | "onRetry"> & {
  pick?: { id: string; current: boolean; stub?: Job };
  onRetryJob: (job?: Job) => void;
}) {
  const live = useJobStoreWhileShown((state) => {
    if (!pick || pick.stub) return undefined;
    return pick.current
      ? (state.currentJob?.id === pick.id ? state.currentJob : undefined)
      : state.jobs.find((job) => job.id === pick.id);
  });
  const job = live ?? pick?.stub;
  return <LiveScanView {...props} job={job} onRetry={() => onRetryJob(job)} />;
}

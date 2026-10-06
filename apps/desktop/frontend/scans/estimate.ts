// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * How long a scan takes on this PC, before it starts. The engine measures it
 * from this machine's own recent scans (GET /scan-estimate): wall seconds per
 * second of VOD, with the Twitch download kept apart since a local file skips
 * it. No history yet means no number, never a made-up one.
 */
import { useEffect, useState } from "react";
import { apiFetch } from "../lib/api";
import { fmtDurationHuman } from "../lib/format";
import type { Job, Session } from "../lib/store";

export interface ScanRates {
  per_vod_second: number | null;
  download_per_vod_second: number | null;
  samples: number;
  matched: boolean;
}

const ACTIVE = ["queued", "analyzing", "detecting", "assembling"];
const isTwitch = (url?: string) => /^https?:\/\/(?:www\.)?twitch\.tv\//i.test(url ?? "");

/** This PC's scan rates for the current scan mode. Re-read when a scan
 * finishes, since every finished scan sharpens them. */
export function useScanRates(apiEndpoint: string, mode: string, facecamTracking: boolean, completedCount: number) {
  const [rates, setRates] = useState<ScanRates | null>(null);
  useEffect(() => {
    let alive = true;
    const query = new URLSearchParams({ mode, facecam_tracking: String(facecamTracking) });
    apiFetch(`/scan-estimate?${query}`, undefined, apiEndpoint)
      .then((res) => (res.ok ? res.json() : null))
      .then((data: ScanRates | null) => { if (alive) setRates(data && typeof data === "object" ? data : null); })
      .catch(() => { if (alive) setRates(null); });
    return () => { alive = false; };
  }, [apiEndpoint, mode, facecamTracking, completedCount]);
  return rates;
}

/** Seconds a scan of `duration` takes here, download included for Twitch. */
export function estimateScanSeconds(rates: ScanRates | null, duration: number | null | undefined, twitch: boolean): number | null {
  if (!rates?.per_vod_second || !duration || duration <= 0) return null;
  const download = twitch ? rates.download_per_vod_second ?? 0 : 0;
  return duration * (rates.per_vod_second + download);
}

/** Seconds until a scan added now would start: what's left of the running
 * scan plus every scan waiting ahead of it. Null when any of it is unknown. */
export function queueWaitSeconds(rates: ScanRates | null, jobs: Job[], sessions: Session[]): number | null {
  const active = sessions.filter((session) => ACTIVE.includes(session.status));
  let total = 0;
  for (const session of active) {
    const job = jobs.find((item) => item.id === session.id);
    const duration = job?.vodDuration ?? session.vodDuration;
    const twitch = isTwitch(job?.url ?? session.sourceUrl);
    const running = job && job.status !== "queued" && !/waiting for an earlier scan/i.test(job.message ?? "");
    if (running && job.etaSeconds != null && (job.etaConfidence === "medium" || job.etaConfidence === "high")) {
      total += job.etaSeconds;
      continue;
    }
    const whole = estimateScanSeconds(rates, duration, twitch);
    if (whole == null) return null;
    total += running ? whole * (1 - Math.max(0, Math.min(100, job.progress ?? 0)) / 100) : whole;
  }
  return total;
}

/** "≈ 42m", "≈ 1h 5m", or "under a minute". */
export function fmtEstimate(seconds: number) {
  return seconds < 60 ? "under a minute" : `≈ ${fmtDurationHuman(seconds)}`;
}

/** Wall-clock time `seconds` from now, e.g. "9:40 PM". */
export function clockIn(seconds: number, now = Date.now()) {
  return new Date(now + seconds * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

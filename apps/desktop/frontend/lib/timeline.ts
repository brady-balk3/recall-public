// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import type { ReactionTimeline } from "./store";

export type TimelinePathOptions = {
  /** Cap the number of drawn samples (peak-preserving). */
  maxPoints?: number;
  /** Light boxcar passes to calm dense R(t) noise on large charts. */
  smoothPasses?: number;
};

function downsamplePeak(
  values: number[],
  times: number[] | null,
  maxPoints: number,
): { r: number[]; t: number[] | null } {
  if (values.length <= maxPoints) return { r: values, t: times };
  const outR: number[] = [];
  const outT: number[] | null = times ? [] : null;
  const bucket = values.length / maxPoints;
  for (let i = 0; i < maxPoints; i++) {
    const start = Math.floor(i * bucket);
    const end = Math.max(start + 1, Math.floor((i + 1) * bucket));
    let peak = values[start];
    let peakIdx = start;
    for (let j = start + 1; j < end; j++) {
      if (values[j] > peak) {
        peak = values[j];
        peakIdx = j;
      }
    }
    outR.push(peak);
    if (outT && times) outT.push(times[peakIdx]);
  }
  return { r: outR, t: outT };
}

function smoothSeries(values: number[], passes: number): number[] {
  if (passes <= 0 || values.length < 3) return values;
  let cur = values;
  for (let pass = 0; pass < passes; pass++) {
    const next = cur.slice();
    for (let i = 1; i < cur.length - 1; i++) {
      next[i] = (cur[i - 1] + cur[i] * 2 + cur[i + 1]) / 4;
    }
    cur = next;
  }
  return cur;
}

/** SVG path for the R(t) reaction curve, normalized to the tallest peak.
 * Shared by the scan theater, the review strip, and the Cutting Room so the
 * curve a creator navigates by is always drawn the same way.
 *
 * When `t` is present, X is mapped from absolute time so pins and the curve
 * share one coordinate system. Optional downsampling/smoothing keep dense
 * mid-scan timelines readable on the large processing chart. */
export function timelinePath(
  timeline: ReactionTimeline | null,
  width = 1000,
  height = 56,
  opts?: TimelinePathOptions,
) {
  if (!timeline?.r?.length) return "";
  const rawT = timeline.t?.length === timeline.r.length ? timeline.t : null;
  const maxPoints = opts?.maxPoints ?? 0;
  const sampled = maxPoints > 0
    ? downsamplePeak(timeline.r, rawT, maxPoints)
    : { r: timeline.r, t: rawT };
  const values = smoothSeries(sampled.r, opts?.smoothPasses ?? 0);
  const times = sampled.t;
  const n = values.length;
  if (n < 2) return "";

  const max = Math.max(...values, 0.001);
  const t0 = times ? times[0] : 0;
  const t1 = times ? times[n - 1] : 1;
  const span = Math.max(0.001, t1 - t0);

  return values.map((value, index) => {
    const x = times
      ? ((times[index] - t0) / span) * width
      : (index / Math.max(1, n - 1)) * width;
    const y = height - 5 - (value / max) * (height - 10);
    return `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Reaction-heat primitives — make the R(t) reaction signal (the product's
 * differentiator) visible in the review chrome.
 *
 *  • HeatSparkline — the R(t) reaction curve for a single clip window, burned
 *                    across the card thumbnail with an area fill + peak dot.
 *
 * It reads purely from data already on the Clip / job timeline, degrades to
 * nothing when that data is absent, and themes through Studio Heat tokens.
 */
import { useId } from "react";
import type { ReactionTimeline } from "../lib/store";

/** R(t) reaction curve for a clip's [start, end] window, drawn from the job
 * timeline. Renders nothing when there is no curve to draw. */
export function HeatSparkline({
  timeline,
  start,
  end,
  className,
}: {
  timeline: ReactionTimeline | null | undefined;
  start?: number;
  end?: number;
  className?: string;
}) {
  const gid = useId();
  if (!timeline?.r?.length || !timeline.t?.length) return null;

  const s = start ?? timeline.t[0];
  const e = end ?? timeline.t[timeline.t.length - 1];
  const pts: Array<{ t: number; r: number }> = [];
  for (let i = 0; i < timeline.t.length; i++) {
    if (timeline.t[i] >= s && timeline.t[i] <= e) pts.push({ t: timeline.t[i], r: timeline.r[i] });
  }
  if (pts.length < 2) return null;

  const W = 100;
  const H = 32;
  const span = Math.max(0.001, e - s);
  const max = Math.max(...pts.map((p) => p.r), 0.001);
  const x = (t: number) => ((t - s) / span) * W;
  const y = (r: number) => H - 3 - (r / max) * (H - 6);
  let peak = pts[0];
  for (const p of pts) if (p.r > peak.r) peak = p;

  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)} ${y(p.r).toFixed(1)}`).join(" ");
  const area = `${line} L${x(pts[pts.length - 1].t).toFixed(1)} ${H} L${x(pts[0].t).toFixed(1)} ${H} Z`;

  return (
    <span className={`heat-spark${className ? ` ${className}` : ""}`} aria-hidden="true">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.42" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={area} fill={`url(#${gid})`} />
        <path
          d={line}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="1.6"
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
        <circle cx={x(peak.t)} cy={y(peak.r)} r="2.2" fill="var(--accent-2)" vectorEffect="non-scaling-stroke" />
      </svg>
    </span>
  );
}

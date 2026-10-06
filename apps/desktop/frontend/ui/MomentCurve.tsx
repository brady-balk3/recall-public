// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * A stream's shape at a glance: a soft ridge that rises wherever Recall found
 * a moment, as tall as that moment's hype. Drawn from the clips alone, so it
 * costs nothing per card (no timeline fetch).
 */
import { useId } from "react";
import type { Clip } from "../lib/store";
import { rankScoreOf } from "../lib/reviewDeck";

const SAMPLES = 96;

export function momentCurvePoints(clips: Clip[], duration?: number): number[] {
  const end = duration && duration > 0 ? duration : Math.max(1, ...clips.map((clip) => clip.end_time ?? 0));
  const peaks = clips
    .filter((clip) => clip.start_time != null)
    .map((clip) => ({ at: ((clip.start_time ?? 0) + (clip.end_time ?? clip.start_time ?? 0)) / 2 / end, h: rankScoreOf(clip) / 100 }));
  const width = 1.6 / SAMPLES;
  return Array.from({ length: SAMPLES + 1 }, (_, i) => {
    const x = i / SAMPLES;
    // A low noise floor so the line reads as a stream, not a flat wire.
    let y = 0.08 + 0.04 * Math.sin(i * 1.7) * Math.sin(i * 0.37);
    for (const p of peaks) y = Math.max(y, p.h * Math.exp(-((x - p.at) ** 2) / (2 * width * width)));
    return Math.min(1, y);
  });
}

export function MomentCurve({ clips, duration, className = "curve" }: { clips: Clip[]; duration?: number; className?: string }) {
  const id = useId().replace(/:/g, "");
  const points = momentCurvePoints(clips, duration);
  const w = 300;
  const h = 30;
  const line = points.map((y, i) => `${i ? "L" : "M"}${((i / SAMPLES) * w).toFixed(1)} ${(h - y * (h - 2)).toFixed(1)}`).join(" ");
  return (
    <svg className={className} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id={`${id}f`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#fff" stopOpacity=".32" />
          <stop offset="1" stopColor="#fff" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${line} L${w} ${h} L0 ${h} Z`} fill={`url(#${id}f)`} />
      <path d={line} fill="none" stroke="#fff" strokeWidth="1.4" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  );
}

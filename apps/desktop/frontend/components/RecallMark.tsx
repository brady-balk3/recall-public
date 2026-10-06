// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The Recall mark: viewfinder corners catching a moment as it comes into focus.
 *
 * Two corner marks sit on the rising diagonal (a clip's in-point and out-point);
 * between them three dots rise, the early ones dim and out of focus, the last
 * sharp, solid, and glowing. Pure mono: the mark never takes the accent. It
 * draws in currentColor, so it is white on dark and ink on light.
 */
import { useId } from "react";

export function RecallMark({ size = 24, className }: { size?: number; className?: string }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg
      className={className}
      viewBox="0 0 100 100"
      width={size}
      height={size}
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <filter id={`${id}g`} x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="6" /></filter>
        <filter id={`${id}b0`} x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="2.2" /></filter>
        <filter id={`${id}b1`} x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="1.1" /></filter>
      </defs>
      <circle cx="65" cy="35" r="16.25" fill="currentColor" opacity=".45" filter={`url(#${id}g)`} />
      <path d="M43 84.2H30A14.2 14.2 0 0 1 15.8 70V57" stroke="currentColor" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M52 13.8H65A21.2 21.2 0 0 1 86.2 35V48" stroke="currentColor" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="30" cy="70" r="5.5" fill="currentColor" opacity=".34" filter={`url(#${id}b0)`} />
      <circle cx="47.5" cy="52.5" r="8" fill="currentColor" opacity=".55" filter={`url(#${id}b1)`} />
      <circle cx="65" cy="35" r="12.5" fill="currentColor" />
    </svg>
  );
}

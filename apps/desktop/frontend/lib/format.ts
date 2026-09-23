// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** Shared display formatters (plan 22 §2.2).
 *
 * These were implemented three-plus times across screens with different
 * truncation lengths and (worse) different hour handling — the Library's
 * local fmtClock rendered 87 minutes as "87:10". One implementation, one
 * truth.
 */

/** Last path segment of a file path or URL, ellipsis-truncated. */
export function sourceName(url: string, max = 32): string {
  const parts = (url || "").replace(/\\/g, "/").split("/");
  const name = parts[parts.length - 1] || url || "Unknown source";
  return name.length > max ? `${name.slice(0, max - 1)}…` : name;
}

/** "1:27:10" / "4:06" — clock-style duration, hour-aware. */
export function fmtClock(sec?: number | null): string {
  if (sec == null || !isFinite(sec) || sec < 0) return "0:00";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return h > 0
    ? `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`
    : `${m}:${s.toString().padStart(2, "0")}`;
}

/** "3h 12m" / "45m" — human duration for session cards. */
export function fmtDurationHuman(sec?: number | null): string {
  if (sec == null || !isFinite(sec) || sec < 0) return "0m";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (h > 0) return m > 0 ? `${h}h ${m}m` : `${h}h`;
  return `${Math.max(1, m)}m`;
}

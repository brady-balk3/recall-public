// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** The review deck: which clips a creator reviews first, and in what order. */
import type { Clip } from "./store";

export type ReviewSort = "recommended" | "hook" | "timeline" | "longest";
export const REVIEW_DECK_LIMIT = 15;
export function rankScoreOf(clip: Clip) {
  return Math.max(0, Math.min(99, clip.deckScore ?? clip.confidence ?? 0));
}
export function hookScoreOf(clip: Clip) {
  return Math.max(0, Math.min(100, clip.confidence ?? 0));
}
export function durationOf(clip: Clip) {
  return Math.max(1, Math.round(clip.duration || ((clip.end_time ?? 0) - (clip.start_time ?? 0))));
}
export function sortReviewClips(clips: Clip[], sort: ReviewSort) {
  const ordered = [...clips];
  if (sort === "timeline") {
    return ordered.sort((a, b) => (a.start_time ?? 0) - (b.start_time ?? 0));
  }
  if (sort === "longest") {
    return ordered.sort((a, b) => durationOf(b) - durationOf(a) || rankScoreOf(b) - rankScoreOf(a));
  }
  if (sort === "hook") {
    return ordered.sort((a, b) => hookScoreOf(b) - hookScoreOf(a) || rankScoreOf(b) - rankScoreOf(a));
  }
  return ordered.sort((a, b) => rankScoreOf(b) - rankScoreOf(a) || (a.start_time ?? 0) - (b.start_time ?? 0));
}
export const originalDeckClips = (clips: Clip[]) => clips.filter((clip) => !clip.isOverflowCandidate);
export const secondLookClips = (clips: Clip[]) => clips.filter((clip) => clip.isOverflowCandidate);
export const primaryReviewDeckClips = (clips: Clip[]) =>
  sortReviewClips(originalDeckClips(clips), "recommended").slice(0, REVIEW_DECK_LIMIT);

/** How fast a clip's opening grabs, from its hook score. Not overall quality. */
export function openingBand(score: number) {
  if (score >= 65) return { label: "Strong opening", shortLabel: "Strong", className: "is-high" };
  if (score >= 45) return { label: "Good opening", shortLabel: "Good", className: "is-mid" };
  return { label: "Slow build", shortLabel: "Slow build", className: "is-low" };
}

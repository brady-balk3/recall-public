// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** Shapes shared by the New stream page and the app shell that feeds it. */

export type ImportMode = "single" | "batch";

export type ProbeMeta = {
  duration: number | null;
  thumbnail: string | null;
  title?: string | null;
  creator?: string | null;
  game?: string | null;
  vod_id?: string | null;
  source_date?: string | null;
};

export type BatchRow = {
  id: string;
  url: string;
  title: string;
  titleEdited: boolean;
  probing: boolean;
  probeError: boolean;
  probeMeta: ProbeMeta | null;
  vtuberConfirmed: boolean;
};

export const TWITCH_VOD_URL = /^https?:\/\/(?:www\.)?twitch\.tv\/videos\/\d+(?:[/?#].*)?$/i;

const VTUBER_TITLE_MARKER = /(?:^|[^a-z0-9])(?:v[ -]?tuber|png[ -]?tuber|virtual\s+(?:streamer|youtuber))(?:$|[^a-z0-9])/i;

export function titleSuggestsVtuberMode(title: string | null | undefined): boolean {
  return VTUBER_TITLE_MARKER.test(title || "");
}

export function batchRowReady(row: BatchRow): boolean {
  return !row.probing && !row.probeError && !!row.probeMeta?.thumbnail && Number(row.probeMeta.duration) > 0;
}

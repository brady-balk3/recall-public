// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** Twitch VOD batch-queue helpers (parse, dedupe, session naming). */

const TWITCH_VOD_RE =
  /^https?:\/\/(?:www\.)?twitch\.tv\/videos\/(\d+)(?:[/?#].*)?$/i;

/** Normalize a single string to a canonical Twitch VOD URL, or null. */
export function normalizeTwitchVodUrl(raw: string): string | null {
  const value = raw.trim();
  if (!value) return null;
  const match = value.match(TWITCH_VOD_RE);
  if (!match) return null;
  return `https://www.twitch.tv/videos/${match[1]}`;
}

/** Extract unique Twitch VOD URLs from free text (one per line or whitespace). */
export function parseTwitchVodUrls(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of text.split(/[\s,;]+/)) {
    const url = normalizeTwitchVodUrl(part);
    if (!url || seen.has(url)) continue;
    seen.add(url);
    out.push(url);
  }
  return out;
}

export function twitchVodId(url: string): string | null {
  const match = url.trim().match(TWITCH_VOD_RE);
  return match ? match[1] : null;
}

export type ProbeNameMeta = {
  title?: string | null;
  creator?: string | null;
  vod_id?: string | null;
};

/** Build a session name from probe metadata (title preferred). */
export function sessionNameFromProbe(meta: ProbeNameMeta | null | undefined, url: string): string {
  const title = (meta?.title || "").trim();
  if (title) return title.slice(0, 200);
  const creator = (meta?.creator || "").trim();
  const vodId = (meta?.vod_id || twitchVodId(url) || "").trim();
  if (creator && vodId) return `${creator} · VOD ${vodId}`.slice(0, 200);
  if (vodId) return `Twitch VOD ${vodId}`.slice(0, 200);
  return "Twitch VOD";
}

/** Apply Twitch metadata unless the creator has already named this queue row. */
export function resolveBatchSessionName(
  currentName: string,
  userEdited: boolean,
  meta: ProbeNameMeta | null | undefined,
  url: string,
): string {
  return userEdited ? currentName : sessionNameFromProbe(meta, url);
}

export type BatchStartItem = {
  url: string;
  name: string;
};

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { sourceName } from "./format";

export interface ScanRetryDraft {
  source: string;
  sourceKind: "local" | "twitch";
  twitchUrl: string;
  projectName: string;
}

/** Build, but never auto-start, a safe retry from a failed/cancelled session. */
export function buildScanRetryDraft(source: string, sessionName?: string): ScanRetryDraft {
  const isTwitch = /^https?:\/\/(?:www\.)?twitch\.tv\/videos\/\d+/i.test(source);
  return {
    source,
    sourceKind: isTwitch ? "twitch" : "local",
    twitchUrl: isTwitch ? source : "",
    projectName: sessionName?.trim() || sourceName(source),
  };
}

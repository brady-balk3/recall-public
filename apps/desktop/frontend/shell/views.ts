// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** Every top-level view the app can show. */
export type View =
  | "home"
  | "live"
  | "sessions"
  | "memory"
  | "compilations"
  | "add-vod"
  | "processing"
  | "queue"
  | "review"
  | "theater"
  | "cutting"
  | "clips"
  | "reel"
  | "insights";

/** Ways to add a stream: a local recording, a Twitch VOD link, or a batch of links. */
export type NewStreamKind = "local" | "twitch" | "batch";

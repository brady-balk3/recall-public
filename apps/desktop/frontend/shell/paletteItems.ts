// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * What Ctrl K offers for a query: a pasted Twitch VOD link to scan, a memory
 * search across every scanned stream, matching sessions, and commands.
 * Pure, so the ranking is testable without a DOM.
 */

export type PaletteIcon =
  | "twitch"
  | "memory"
  | "import"
  | "batch"
  | "live"
  | "review"
  | "export"
  | "scissors"
  | "layers"
  | "reel"
  | "settings"
  | "keyboard"
  | "insights"
  | "library"
  | "home"
  | "scans";

export interface PaletteCommand {
  id: string;
  title: string;
  icon: PaletteIcon;
  /** Extra words that should match, e.g. "clips" for Export. */
  keywords?: string;
  run: () => void;
}

export interface PaletteSession {
  id: string;
  title: string;
  sub: string;
  image?: string;
  open: () => void;
}

export interface PaletteItem {
  id: string;
  section: "Link" | "Stream Memory" | "Sessions" | "Do" | "Try remembering";
  title: string;
  sub?: string;
  icon?: PaletteIcon;
  image?: string;
  run: () => void;
}

const TWITCH_VOD = /twitch\.tv\/videos\/\d+/i;

export function isTwitchVodLink(query: string) {
  return TWITCH_VOD.test(query.trim());
}

export interface PaletteInput {
  query: string;
  sessions: PaletteSession[];
  commands: PaletteCommand[];
  examples?: string[];
  scanLink: (url: string) => void;
  searchMemory: (query: string) => void;
}

export function paletteItems({ query, sessions, commands, examples = [], scanLink, searchMemory }: PaletteInput): PaletteItem[] {
  const q = query.trim();
  const ql = q.toLowerCase();
  const out: PaletteItem[] = [];
  const link = isTwitchVodLink(q);

  if (link) {
    out.push({ id: "link", section: "Link", icon: "twitch", title: "Scan this Twitch VOD", sub: q, run: () => scanLink(q) });
  } else if (q) {
    out.push({
      id: "memory",
      section: "Stream Memory",
      icon: "memory",
      title: `Search your streams for “${q}”`,
      sub: "Meaning and words, across every scanned stream",
      run: () => searchMemory(q),
    });
  }

  if (!link) {
    sessions
      .filter((s) => !ql || s.title.toLowerCase().includes(ql))
      .slice(0, ql ? 5 : 3)
      .forEach((s) => out.push({ id: `session:${s.id}`, section: "Sessions", title: s.title, sub: s.sub, image: s.image, icon: "reel", run: s.open }));

    commands
      .filter((c) => !ql || c.title.toLowerCase().includes(ql) || (c.keywords ?? "").toLowerCase().includes(ql))
      .slice(0, ql ? 5 : 8)
      .forEach((c) => out.push({ id: `do:${c.id}`, section: "Do", icon: c.icon, title: c.title, run: c.run }));
  }

  if (!q) {
    examples.slice(0, 2).forEach((example, index) =>
      out.push({ id: `example:${index}`, section: "Try remembering", icon: "memory", title: `“${example}”`, run: () => searchMemory(example) }),
    );
  }
  return out;
}

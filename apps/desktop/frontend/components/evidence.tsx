// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Evidence chips — turn a clip's raw reaction signals into small glyph + label
 * chips, so "why this clip" reads at a glance in the creator's own language.
 * Glyphs are Solar bold-duotone (matching the app icon system); unmapped
 * signals fall back to a text clue upstream.
 */
import type { Clip } from "../lib/store";

const GLYPH: Record<string, string> = {
  trophy: `<path fill="currentColor" d="M12 16c-5.76 0-6.78-5.74-6.96-10.294c-.05-1.266-.076-1.9.4-2.485c.476-.586 1.045-.682 2.184-.874A26.4 26.4 0 0 1 12 2c1.783 0 3.253.157 4.377.347c1.138.192 1.708.288 2.183.874c.476.586.451 1.219.4 2.485C18.78 10.259 17.76 16 12 16" opacity=".5"/><path fill="currentColor" d="m17.64 12.422l2.817-1.565c.752-.418 1.128-.627 1.336-.979C22 9.526 22 9.096 22 8.235v-.073c0-1.043 0-1.565-.283-1.958s-.778-.558-1.768-.888L19 5l-.017.085q-.008.283-.022.621c-.088 2.225-.377 4.733-1.32 6.716M5.04 5.706c.087 2.225.376 4.733 1.32 6.716l-2.817-1.565c-.752-.418-1.129-.627-1.336-.979S2 9.096 2 8.235v-.073c0-1.043 0-1.565.283-1.958s.778-.558 1.768-.888L5 5l.017.087q.008.281.022.62"/><path fill="currentColor" fill-rule="evenodd" d="M5.25 22a.75.75 0 0 1 .75-.75h12a.75.75 0 0 1 0 1.5H6a.75.75 0 0 1-.75-.75" clip-rule="evenodd"/><path fill="currentColor" d="M15.458 21.25H8.543l.297-1.75a1 1 0 0 1 .98-.804h4.36a1 1 0 0 1 .981.804z" opacity=".5"/><path fill="currentColor" d="M12 16q-.39 0-.75-.034v2.73h1.5v-2.73A8 8 0 0 1 12 16m-.854-9.977C11.526 5.34 11.716 5 12 5s.474.34.854 1.023l.098.176c.108.194.162.29.246.354c.085.064.19.088.4.135l.19.044c.738.167 1.107.25 1.195.532s-.164.577-.667 1.165l-.13.152c-.143.167-.215.25-.247.354s-.021.215 0 .438l.02.203c.076.785.114 1.178-.115 1.352c-.23.174-.576.015-1.267-.303l-.178-.082c-.197-.09-.295-.135-.399-.135s-.202.045-.399.135l-.178.082c-.691.319-1.037.477-1.267.303s-.191-.567-.115-1.352l.02-.203c.021-.223.032-.334 0-.438s-.104-.187-.247-.354l-.13-.152c-.503-.588-.755-.882-.667-1.165c.088-.282.457-.365 1.195-.532l.19-.044c.21-.047.315-.07.4-.135c.084-.064.138-.16.246-.354z"/>`,
  mic: `<path fill="currentColor" fill-rule="evenodd" d="M4 9a.75.75 0 0 1 .75.75v1a7.25 7.25 0 1 0 14.5 0v-1a.75.75 0 0 1 1.5 0v1a8.75 8.75 0 0 1-8 8.718v2.282a.75.75 0 0 1-1.5 0v-2.282a8.75 8.75 0 0 1-8-8.718v-1A.75.75 0 0 1 4 9" clip-rule="evenodd"/><path fill="currentColor" fill-rule="evenodd" d="M12 2a5.75 5.75 0 0 0-5.75 5.75v3a5.75 5.75 0 0 0 11.5 0v-3A5.75 5.75 0 0 0 12 2m2 9.5a.75.75 0 0 0 0-1.5h-4a.75.75 0 0 0 0 1.5zm-.25-3.75a.75.75 0 0 1-.75.75h-2A.75.75 0 0 1 11 7h2a.75.75 0 0 1 .75.75" clip-rule="evenodd" opacity=".5"/><path fill="currentColor" d="M14 11.5a.75.75 0 0 0 0-1.5h-4a.75.75 0 0 0 0 1.5zm-1-3A.75.75 0 0 0 13 7h-2a.75.75 0 0 0 0 1.5z"/>`,
  laugh: `<path fill="currentColor" d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2S2 6.477 2 12s4.477 10 10 10" opacity=".5"/><path fill="currentColor" d="M14.898 11.224c.533-.143.792-.908.578-1.708s-.821-1.333-1.355-1.19c-.533.143-.792.907-.577 1.708c.214.8.82 1.333 1.354 1.19m-5.796 1.553c.534-.143.792-.908.578-1.708s-.82-1.333-1.354-1.19s-.792.907-.578 1.708s.82 1.333 1.354 1.19m-.917 2.974a.75.75 0 0 1 .91-.545c1.13.283 2.428.287 3.746-.066c1.318-.354 2.44-1.006 3.278-1.816a.75.75 0 1 1 1.043 1.078a8.4 8.4 0 0 1-1.15.928l.159.322a1.5 1.5 0 1 1-2.693 1.322l-.196-.4l-.053.014c-1.555.417-3.112.42-4.499.073a.75.75 0 0 1-.545-.91"/>`,
  eye: `<path fill="currentColor" d="M2 12c0 1.64.425 2.191 1.275 3.296C4.972 17.5 7.818 20 12 20s7.028-2.5 8.725-4.704C21.575 14.192 22 13.639 22 12c0-1.64-.425-2.191-1.275-3.296C19.028 6.5 16.182 4 12 4S4.972 6.5 3.275 8.704C2.425 9.81 2 10.361 2 12" opacity=".5"/><path fill="currentColor" fill-rule="evenodd" d="M8.25 12a3.75 3.75 0 1 1 7.5 0a3.75 3.75 0 0 1-7.5 0m1.5 0a2.25 2.25 0 1 1 4.5 0a2.25 2.25 0 0 1-4.5 0" clip-rule="evenodd"/>`,
  bolt: `<path fill="currentColor" fill-rule="evenodd" d="M8.732 5.771L5.67 9.914c-1.285 1.739-1.928 2.608-1.574 3.291l.018.034c.375.673 1.485.673 3.704.673c1.233 0 1.85 0 2.236.363l.02.02l3.872-4.57l-.02-.02c-.379-.371-.379-.963-.379-2.148v-.31c0-3.285 0-4.927-.923-5.21s-1.913 1.056-3.892 3.734" clip-rule="evenodd"/><path fill="currentColor" d="M10.453 16.443v.31c0 3.284 0 4.927.923 5.21s1.913-1.056 3.893-3.734l3.062-4.143c1.284-1.739 1.927-2.608 1.573-3.291l-.018-.034c-.375-.673-1.485-.673-3.704-.673c-1.233 0-1.85 0-2.236-.363l-3.872 4.57c.379.371.379.963.379 2.148" opacity=".5"/>`,
  crown: `<path fill="currentColor" fill-rule="evenodd" d="m19.687 14.093l.184-1.704c.097-.91.162-1.51.111-1.889a1.5 1.5 0 0 1-1.117-.52c-.327.201-.753.626-1.394 1.265c-.495.493-.742.739-1.018.777a.83.83 0 0 1-.45-.063c-.254-.112-.424-.416-.763-1.025l-1.79-3.209c-.209-.375-.384-.69-.542-.942c-.273.139-.581.217-.908.217s-.635-.078-.908-.217c-.158.253-.333.567-.543.942L8.76 10.934c-.34.609-.51.913-.764 1.025a.83.83 0 0 1-.45.063c-.275-.038-.522-.284-1.017-.777c-.641-.639-1.067-1.064-1.393-1.265a1.5 1.5 0 0 1-1.118.52c-.051.378.014.979.111 1.889l.184 1.704l.089.85c.252 2.435.46 4.45 1.31 5.21c.946.847 2.364.847 5.2.847h2.176c2.836 0 4.254 0 5.2-.847c.85-.76 1.058-2.775 1.31-5.21q.043-.417.09-.85" clip-rule="evenodd" opacity=".5"/><path fill="currentColor" d="M20 10.5a1.5 1.5 0 1 0-.018 0zM12 3a2 2 0 1 0 0 4a2 2 0 0 0 0-4M2.5 9A1.5 1.5 0 0 0 4 10.5h.018A1.497 1.497 0 0 0 5.5 9a1.5 1.5 0 1 0-3 0m2.349 9.25a18 18 0 0 1-.246-1.5h14.794c-.07.545-.148 1.05-.246 1.5z"/>`,
  fire: `<path fill="currentColor" d="M12.832 21.801c3.126-.626 7.168-2.875 7.168-8.69c0-5.291-3.873-8.815-6.658-10.434c-.619-.36-1.342.113-1.342.828v1.828c0 1.442-.606 4.074-2.29 5.169c-.86.559-1.79-.278-1.894-1.298l-.086-.838c-.1-.974-1.092-1.565-1.87-.971C4.461 8.46 3 10.33 3 13.11C3 20.221 8.289 22 10.933 22q.232 0 .484-.015c.446-.056 0 .099 1.415-.185" opacity=".5"/><path fill="currentColor" d="M8 18.444c0 2.62 2.111 3.43 3.417 3.542c.446-.056 0 .099 1.415-.185C13.871 21.434 15 20.492 15 18.444c0-1.297-.819-2.098-1.46-2.473c-.196-.115-.424.03-.441.256c-.056.718-.746 1.29-1.215.744c-.415-.482-.59-1.187-.59-1.638v-.59c0-.354-.357-.59-.663-.408C9.495 15.008 8 16.395 8 18.445"/>`,
  spark: `<path fill="currentColor" d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z"/>`,
};

/** Raw engine signal → { glyph, short creator label }. */
const SIGNAL_EV: Record<string, { glyph: keyof typeof GLYPH; label: string }> = {
  match_win: { glyph: "trophy", label: "Match win" },
  elimination: { glyph: "bolt", label: "Elimination" },
  knock: { glyph: "bolt", label: "Knock" },
  rank_progress: { glyph: "crown", label: "Rank up" },
  laughter_burst: { glyph: "laugh", label: "Laugh" },
  speech_hype: { glyph: "mic", label: "Hype call" },
  voice_reaction: { glyph: "mic", label: "Voice" },
  facecam_reaction: { glyph: "eye", label: "Facecam" },
  chat_spike: { glyph: "fire", label: "Chat spike" },
  multi_signal: { glyph: "fire", label: "Multi-signal" },
  generic_highlight: { glyph: "spark", label: "Highlight" },
};

const EV_PRIORITY = [
  "match_win", "elimination", "knock", "rank_progress", "laughter_burst",
  "speech_hype", "voice_reaction", "facecam_reaction", "chat_spike",
  "multi_signal", "generic_highlight",
];

export type EvidenceItem = { glyph: string; label: string };

/** Up to `limit` mapped evidence chips for a clip, highest-signal first. */
export function evidenceChipsFor(clip: Clip, limit = 2): EvidenceItem[] {
  const seen = new Set<string>();
  const out: EvidenceItem[] = [];
  const ordered = [...(clip.signals ?? [])].sort(
    (a, b) =>
      (EV_PRIORITY.indexOf(a) < 0 ? 99 : EV_PRIORITY.indexOf(a)) -
      (EV_PRIORITY.indexOf(b) < 0 ? 99 : EV_PRIORITY.indexOf(b)),
  );
  for (const sig of ordered) {
    const ev = SIGNAL_EV[sig];
    if (!ev || seen.has(ev.label)) continue;
    seen.add(ev.label);
    out.push({ glyph: GLYPH[ev.glyph], label: ev.label });
    if (out.length >= limit) break;
  }
  return out;
}

/** A single glyph + label evidence chip. */
export function EvidenceChip({ item }: { item: EvidenceItem }) {
  return (
    <span className="ev-chip">
      <svg className="ev-chip-glyph" viewBox="0 0 24 24" aria-hidden="true" dangerouslySetInnerHTML={{ __html: item.glyph }} />
      <span>{item.label}</span>
    </span>
  );
}

/** A row of evidence chips, or null when nothing maps. */
export function EvidenceChips({ clip, limit = 2 }: { clip: Clip; limit?: number }) {
  const items = evidenceChipsFor(clip, limit);
  if (!items.length) return null;
  return (
    <span className="ev-chip-row">
      {items.map((item) => (
        <EvidenceChip key={item.label} item={item} />
      ))}
    </span>
  );
}

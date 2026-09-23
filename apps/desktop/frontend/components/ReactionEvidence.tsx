// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
// Raw modality means live on different scales (face arousal ~0..0.3, voice
// ~0..1, ...). Normalize by the same nominal "high" values the engine uses to
// pick the dominant modality, so the bars read as "how loud was this channel
// relative to a strong moment".
const MODALITY_NOMINAL: Record<string, number> = {
  voice: 0.5,
  face: 0.25,
  speech: 0.45,
  burst: 0.5,
  chat: 6.0,
  game_audio: 0.5,
  game: 1.0,
};

const MODALITY_LABEL: Record<string, string> = {
  voice: "Voice",
  face: "Facecam",
  speech: "Callouts",
  burst: "Laughter",
  chat: "Chat",
  game_audio: "Game action",
  game: "Game event",
};

const MODALITY_ORDER = ["voice", "face", "speech", "burst", "chat", "game_audio", "game"];

/** Per-modality evidence bars: what drove the reaction score for this clip. */
export function ModalityMix({ breakdown }: { breakdown: Record<string, number> }) {
  const rows = MODALITY_ORDER
    .map((key) => ({
      key,
      label: MODALITY_LABEL[key] ?? key,
      value: Math.max(0, Math.min(1, (breakdown[key] ?? 0) / (MODALITY_NOMINAL[key] ?? 1))),
    }))
    .filter((row) => row.value > 0.02);
  if (rows.length === 0) return null;

  return (
    <div className="modality-mix">
      <span className="modality-mix-title">
        What drove this pick
      </span>
      <div className="modality-mix-rows">
        {rows.map((row) => (
          <div className="modality-mix-row" key={row.key} title={`${row.label}: ${Math.round(row.value * 100)}% of Recall's strong-moment reference`}>
            <span>{row.label}</span>
            <div
              className="modality-mix-track"
              role="img"
              aria-label={`${row.label} contribution ${Math.round(row.value * 100)} percent of a strong moment`}
            >
              <div className={row.key === "game" ? "is-game" : ""} style={{ width: `${Math.max(4, row.value * 100)}%` }} />
            </div>
            <b>{Math.round(row.value * 100)}%</b>
          </div>
        ))}
      </div>
    </div>
  );
}

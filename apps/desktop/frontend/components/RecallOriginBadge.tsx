// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { Target } from "../lib/icons";
import type { RecallProvenance } from "../lib/store";

export function recallOriginCopy(provenance: RecallProvenance) {
  if (provenance.source === "stream_memory") {
    return {
      label: "Found in Memory",
      detail: "You opened this moment from Stream Memory and finished the cut in the VOD Editor.",
    };
  }
  return provenance.origin === "creator_marker"
    ? {
        label: "Remembered live",
        detail: "Recall found this clip around a moment you marked during the session.",
      }
    : {
        label: "Live-assisted",
        detail: "This clip came from a Recall Session scan, but was found independently of your markers.",
      };
}

export default function RecallOriginBadge({
  provenance,
  compact = false,
}: {
  provenance?: RecallProvenance;
  compact?: boolean;
}) {
  if (!provenance) return null;
  const copy = recallOriginCopy(provenance);
  return (
    <span
      className={`recall-origin-badge ${provenance.source === "stream_memory" || provenance.origin === "creator_marker" ? "is-marker" : "is-assisted"} ${compact ? "is-compact" : ""}`}
      title={copy.detail}
      aria-label={`${copy.label}. ${copy.detail}`}
    >
      <Target size={compact ? 10 : 11} aria-hidden="true" />
      {copy.label}
    </span>
  );
}

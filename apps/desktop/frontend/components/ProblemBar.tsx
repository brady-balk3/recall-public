// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useState } from "react";
import { AlertTriangle, X } from "../lib/icons";
import { useJobStore } from "../lib/store";

/**
 * The one visible renderer for the store's log channel.
 *
 * This replaces StatusToaster, which surfaced every log entry as a transient
 * bottom-right popup. Most of that channel is confirmation of something the
 * user just did and watched happen ("Clip changes saved", "Recording
 * selected: …", "Recall will show fewer moments like those"), so the popups
 * were near-constant and carried almost no information -- and because they
 * auto-dismissed, the one kind that DID matter could expire unread.
 *
 * So: only `warn` entries render, and they render in the document flow under
 * the header rather than floating over the content. A failure now stays put
 * until it is dismissed, which is the behavior an error actually wants; the
 * chatty half of the channel renders nowhere. `addLog` remains valid for every
 * caller -- info/success entries still reach the store and its 100-entry
 * history, they simply no longer interrupt anyone.
 */
export default function ProblemBar() {
  const logs = useJobStore((s) => s.logs);
  const [dismissed, setDismissed] = useState<number[]>([]);

  // logs is newest-first and capped at 100, so a dismissed id eventually falls
  // off the end; filtering against the live list keeps the two in step without
  // a second cleanup pass.
  const problems = logs.filter(
    (entry) => entry.type === "warn" && !dismissed.includes(entry.id),
  );
  if (!problems.length) return null;

  const [current, ...older] = problems;
  return (
    <div className="problem-bar" role="alert">
      <AlertTriangle size={15} aria-hidden="true" />
      <p>{current.text}</p>
      {older.length > 0 && (
        <button
          type="button"
          className="problem-bar-rest"
          onClick={() => setDismissed(logs.map((entry) => entry.id))}
        >
          Dismiss all ({problems.length})
        </button>
      )}
      <button
        type="button"
        className="problem-bar-close"
        aria-label="Dismiss this problem"
        onClick={() => setDismissed((ids) => [...ids, current.id])}
      >
        <X size={13} aria-hidden="true" />
      </button>
    </div>
  );
}

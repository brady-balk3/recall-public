// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * "Load session" for the Cutting Room: pick any stream or opened video in the
 * Library to cut. Newest first, filterable by name; streams still downloading
 * (nothing on disk yet) and failed scans aren't offered.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { SessionPoster } from "../components/Posters";
import { fmtDurationHuman } from "../lib/format";
import { primaryReviewDeckClips } from "../lib/reviewDeck";
import type { Session } from "../lib/store";

const isLocal = (session: Session) => !!session.sourceUrl && !/^https?:/i.test(session.sourceUrl);

/** Sessions the Cutting Room can open, newest first. */
export function cuttableSessions(sessions: Session[]): Session[] {
  return sessions
    .filter((session) => session.status === "completed" || (isLocal(session) && session.status !== "failed" && session.status !== "cancelled"))
    .sort((a, b) => (b.updatedAt || b.createdAt) - (a.updatedAt || a.createdAt));
}

function describe(session: Session): string {
  const parts: string[] = [];
  if (session.vodDuration) parts.push(fmtDurationHuman(session.vodDuration));
  if (/not scanned/i.test(session.message ?? "")) parts.push("Not scanned");
  else if (session.status !== "completed") parts.push("Scanning");
  else {
    const moments = primaryReviewDeckClips(session.clips).length;
    parts.push(`${moments} ${moments === 1 ? "moment" : "moments"}`);
  }
  parts.push(new Date(session.createdAt).toLocaleDateString(undefined, { month: "short", day: "numeric" }));
  return parts.join(" · ");
}

export function SessionPicker({ sessions, currentId, onPick, onClose }: {
  sessions: Session[];
  currentId?: string;
  onPick: (session: Session) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const list = useMemo(() => {
    const q = query.trim().toLowerCase();
    return cuttableSessions(sessions).filter((session) => !q || session.name.toLowerCase().includes(q));
  }, [sessions, query]);

  useEffect(() => {
    input.current?.focus();
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Portaled to <body>: the Cutting Room's glass panel would otherwise become
  // the containing block for this fixed overlay and push it off-screen.
  return createPortal(
    <div className="project-modal open session-picker" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div className="project-dialog" role="dialog" aria-modal="true" aria-labelledby="session-picker-title">
        <div>
          <h2 id="session-picker-title" className="disp">Load a session</h2>
          <p>Pick a stream or video from your Library to cut.</p>
        </div>
        <input ref={input} className="field" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter by name" aria-label="Filter sessions by name" />
        <ul className="sp-list">
          {list.length === 0 && <li className="sp-empty">{sessions.length ? "Nothing matches that." : "Your Library is empty. Open a video instead."}</li>}
          {list.map((session) => (
            <li key={session.id}>
              <button type="button" className={`sp-item ${session.id === currentId ? "is-current" : ""}`} onClick={() => onPick(session)}>
                <span className="sp-poster"><SessionPoster session={session} /></span>
                <span className="sp-text">
                  <b>{session.name}</b>
                  <small>{describe(session)}</small>
                </span>
                {session.id === currentId && <span className="sp-tag">Open</span>}
              </button>
            </li>
          ))}
        </ul>
        <div className="project-dialog-actions">
          <button type="button" className="btn ghost" onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

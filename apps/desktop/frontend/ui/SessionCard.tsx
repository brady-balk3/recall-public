// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** A stream as a poster: its best frame, its name, and the shape of its moments. */
import type { ReactNode } from "react";
import { SessionPoster } from "../components/Posters";
import { fmtDurationHuman } from "../lib/format";
import { Check, Folder, Trash2, Twitch } from "../lib/icons";
import { useJobStore, type Job, type Session } from "../lib/store";
import { undecidedClips } from "../home/homeModel";
import { curveSeries, seriesPath } from "../scans/scanStrip";
import { MomentCurve } from "./MomentCurve";

const SCANNING = new Set(["queued", "analyzing", "detecting", "assembling"]);

/** A stream mid-scan has no poster yet: show how far Recall has read it and
 * the reaction curve as it comes in, so the tile isn't a blank placeholder. */
function ScanningPoster({ job }: { job?: Job }) {
  const progress = Math.floor(Math.max(0, Math.min(100, job?.progress ?? 0)));
  const timeline = job?.liveTimeline;
  const end = timeline?.t?.length ? timeline.t[timeline.t.length - 1] : 0;
  const duration = Math.max(job?.vodDuration ?? 0, end);
  const series = curveSeries(timeline, duration);
  const read = Math.max(job?.scannedSeconds ?? 0, job?.transcribedSeconds ?? 0);
  const readTo = duration > 0 ? Math.min(1, read / duration) : progress / 100;
  const path = series ? seriesPath(series) : "";
  return (
    <span className="sess-scan" aria-hidden="true">
      <span className="read" style={{ width: `${readTo * 100}%` }} />
      {series && (
        <svg viewBox="0 0 1000 300" preserveAspectRatio="none">
          <path className="fill" d={`${path}L1000 300L0 300Z`} />
          <path className="line" d={path} vectorEffect="non-scaling-stroke" />
        </svg>
      )}
      {readTo < 1 && <span className="head" style={{ left: `${readTo * 100}%` }} />}
      <span className="pct num">{progress}<small>%</small></span>
    </span>
  );
}

export function sessionDate(session: Session, withWeekday = true) {
  return new Date(session.createdAt || session.updatedAt).toLocaleDateString(undefined, {
    ...(withWeekday ? { weekday: "short" as const } : {}),
    month: "short",
    day: "numeric",
  });
}

export function isFromTwitch(session: Session) {
  return /twitch\.tv/i.test(session.sourceUrl);
}

/** The one badge that best says where a stream stands. */
export function SessionStatusBadge({ session, scanProgress }: { session: Session; scanProgress?: number }) {
  if (session.status === "failed" || session.status === "cancelled") return <span className="badge danger">Needs attention</span>;
  if (session.status !== "completed") {
    return <span className="badge dark">Scanning{scanProgress != null ? ` ${Math.floor(scanProgress)}%` : ""}</span>;
  }
  const waiting = undecidedClips(session).length;
  if (waiting) return <span className="badge heat">{waiting} to review</span>;
  if (session.savedClipIds.length) return <span className="badge keep"><Check aria-hidden="true" />{session.savedClipIds.length} kept</span>;
  return null;
}

export function SessionCard({
  session,
  onOpen,
  badge,
  selecting = false,
  selected = false,
  onToggle,
  onDelete,
}: {
  session: Session;
  onOpen: (session: Session) => void;
  badge?: ReactNode;
  selecting?: boolean;
  selected?: boolean;
  onToggle?: (session: Session) => void;
  onDelete?: (session: Session) => void;
}) {
  const scanning = SCANNING.has(session.status);
  const job = useJobStore((state) => (scanning ? state.jobs.find((j) => j.id === session.id) ?? (state.currentJob?.id === session.id ? state.currentJob : undefined) : undefined));
  return (
    <div className={`sess-wrap ${selected ? "is-selected" : ""}`}>
      <button
        type="button"
        className="sess glass"
        aria-pressed={selecting ? selected : undefined}
        onClick={() => (selecting ? onToggle?.(session) : onOpen(session))}
      >
        <div className="poster">
          {scanning ? <ScanningPoster job={job} /> : <SessionPoster session={session} variant="card" />}
          {badge && <div className="top">{badge}</div>}
          {!scanning && session.clips.length > 0 && <MomentCurve clips={session.clips} duration={session.vodDuration} />}
          <div className="t"><h3 className="disp">{session.name}</h3></div>
        </div>
        <div className="body">
          <span>{sessionDate(session)}</span>
          <span className="grow" />
          {session.vodDuration ? <span className="num">{fmtDurationHuman(session.vodDuration)}</span> : null}
          {isFromTwitch(session) ? <Twitch aria-label="From Twitch" /> : <Folder aria-label="Local recording" />}
        </div>
      </button>
      {selecting ? (
        <span className={`sess-check ${selected ? "on" : ""}`} aria-hidden="true">{selected && <Check />}</span>
      ) : onDelete ? (
        <button type="button" className="sess-del icon-btn" aria-label={`Delete ${session.name}`} title="Delete session" onClick={() => onDelete(session)}>
          <Trash2 aria-hidden="true" />
        </button>
      ) : null}
    </div>
  );
}

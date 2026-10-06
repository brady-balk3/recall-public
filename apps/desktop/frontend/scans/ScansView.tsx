// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Scans: what Recall is working on, and what it has done. While a scan runs,
 * the live card (reaction curve, what it found so far, time left) sits beside
 * the queue. When nothing runs, a short idle strip says so and points at the
 * next thing worth doing. The scan history sits below both. Scan speed lives
 * in Settings.
 *
 * One scan runs at a time; the rest wait their turn. Selecting a queue row
 * shows it in the live card. Finished scans hand off to Review; failed ones
 * explain themselves in plain words and never show a raw error or path.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Poster, SessionPoster } from "../components/Posters";
import { describeScanFailure, describeScanHealth } from "../lib/copy";
import { fmtClock, fmtDurationHuman } from "../lib/format";
import { AlertTriangle, Check, ChevronRight, Plus, Queue, X } from "../lib/icons";
import { primaryReviewDeckClips } from "../lib/reviewDeck";
import { undecidedClips } from "../home/homeModel";
import { fetchReactionTimeline, type Job, type ReactionTimeline, type Session } from "../lib/store";
import type { NewStreamKind } from "../shell/views";
import { foldActivity, queueLane, type QueueLaneState } from "./scanModel";
import { clockIn, estimateScanSeconds, fmtEstimate, type ScanRates } from "./estimate";
import { curveSeries, deckMoments, foundMoments, ProgressRing, ScanStrip, timeLeftSeconds } from "./scanStrip";

export interface ScansViewProps {
  jobIds: string[];
  jobs: Job[];
  sessions: Session[];
  onOpenSession: (session: Session) => void;
  onOpenProcessing: (session: Session) => void;
  onCancelJob: (jobId: string) => Promise<boolean>;
  onDone: () => void;
  onClearQueue: () => void;
  onNew: (kind: NewStreamKind) => void;
  /** This PC's measured scan speed, for how long a waiting scan will take. */
  rates?: ScanRates | null;
}

interface Row {
  id: string;
  index: number;
  job?: Job;
  session?: Session;
  lane: QueueLaneState;
}

const ACTIVE = ["queued", "analyzing", "detecting", "assembling"];
const HISTORY_LIMIT = 12;

function relativeAgo(ms: number) {
  const mins = Math.max(0, Math.round((Date.now() - ms) / 60_000));
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} ${days === 1 ? "day" : "days"} ago`;
}

export function ScansView({ jobIds, jobs, sessions, onOpenSession, onOpenProcessing, onCancelJob, onDone, onClearQueue, onNew, rates = null }: ScansViewProps) {
  const [selectedId, setSelectedId] = useState<string>();
  const [cancellingId, setCancellingId] = useState<string>();
  const [clock, setClock] = useState(() => Date.now());

  const rows: Row[] = useMemo(() => {
    const ids = jobIds.length ? jobIds : sessions.filter((s) => ACTIVE.includes(s.status)).map((s) => s.id);
    return ids.map((id, index) => {
      const job = jobs.find((j) => j.id === id);
      const session = sessions.find((s) => s.id === id);
      return { id, index, job, session, lane: queueLane(job, session) };
    });
  }, [jobIds, jobs, sessions]);

  // The live card only exists while there is live work. A finished scan is
  // history, not the headline: it hands off to Review from the list below.
  const live = rows.some((r) => r.lane === "running" || r.lane === "waiting");

  const history: Session[] = useMemo(() => {
    const inQueue = new Set(live ? rows.map((r) => r.id) : []);
    return sessions
      .filter((s) => !inQueue.has(s.id) && ["completed", "failed"].includes(s.status))
      .sort((a, b) => (b.updatedAt || b.createdAt) - (a.updatedAt || a.createdAt))
      .slice(0, HISTORY_LIMIT);
  }, [live, rows, sessions]);

  useEffect(() => {
    if (!live || (selectedId && rows.some((r) => r.id === selectedId))) return;
    setSelectedId(rows.find((r) => r.lane === "running")?.id ?? rows.find((r) => r.lane === "waiting")?.id);
  }, [live, rows, selectedId]);

  const running = rows.some((r) => r.lane === "running");
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running]);

  const selected = rows.find((r) => r.id === selectedId) ?? rows[0];

  const cancel = async (id: string) => {
    setCancellingId(id);
    try { await onCancelJob(id); } finally { setCancellingId(undefined); }
  };

  return (
    <div className="page scans">
      <div className="ph">
        <div>
          <h1 className="disp">Scans</h1>
          <div className="sub">Recall watches your VODs so you don't have to. One at a time, in the order you add them. Leave it open and the queue keeps moving.</div>
        </div>
        <div className="actions">
          <button type="button" className="btn" onClick={() => onNew("batch")}><Queue aria-hidden="true" />Queue VODs</button>
          <button type="button" className="btn heat" onClick={() => onNew("local")}><Plus aria-hidden="true" />New stream</button>
        </div>
      </div>

      {live && selected ? (
        <div className="scan-grid">
          <SelectedScan
            row={selected}
            rows={rows}
            clock={clock}
            rates={rates}
            cancelling={cancellingId === selected.id}
            onCancel={() => void cancel(selected.id)}
            onOpenSession={onOpenSession}
            onOpenProcessing={onOpenProcessing}
          />
          <section className="panel glass" aria-label="Queue">
            <div className="panel-t">
              <span className="eyebrow">In the queue</span>
              <span className="sp" />
              {jobIds.length > 0 && !running && <button type="button" className="btn sm ghost" onClick={onClearQueue}>Clear</button>}
            </div>
            {rows.map((row) => <QueueRow key={row.id} row={row} rows={rows} selected={row.id === selected.id} onSelect={() => setSelectedId(row.id)} />)}
          </section>
        </div>
      ) : (
        <IdleScan latest={history[0]} job={jobs.find((j) => j.id === history[0]?.id)} onNew={onNew} onOpenSession={onOpenSession} onOpenProcessing={onOpenProcessing} />
      )}

      {history.length > 0 && (
        <section className="scan-history" aria-label="Scan history">
          <div className="panel-t">
            <h2 className="disp">History</h2>
            <span className="sp" />
            <button type="button" className="btn sm ghost" onClick={onDone}>All sessions</button>
          </div>
          <div className="sh-list glass">
            {history.map((session) => (
              <HistoryRow key={session.id} session={session} job={jobs.find((j) => j.id === session.id)} onOpenSession={onOpenSession} onOpenProcessing={onOpenProcessing} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

/** Nothing running: say so, and point at the one thing worth doing next. */
function IdleScan({ latest, job, onNew, onOpenSession, onOpenProcessing }: {
  latest?: Session;
  job?: Job;
  onNew: (kind: NewStreamKind) => void;
  onOpenSession: (session: Session) => void;
  onOpenProcessing: (session: Session) => void;
}) {
  const failed = latest?.status === "failed";
  const failure = describeScanFailure(job?.errorMessage || job?.message || latest?.message);
  const waiting = latest && !failed ? undecidedClips(latest).length : 0;
  const heading = failed ? failure.title
    : waiting > 0 ? `${waiting} ${waiting === 1 ? "moment is" : "moments are"} waiting.`
      : latest ? "You're caught up." : "Nothing scanning.";
  return (
    <section className={`scan-idle glass ${failed ? "is-failed" : ""}`}>
      <span className="eyebrow">{failed ? "Your last scan stopped" : "Nothing scanning"}</span>
      <h2 className="disp">{heading}</h2>
      <p className="t3">
        {failed ? failure.hint
          : waiting > 0 ? <>From <b className="t1">{latest!.name}</b>, scanned {relativeAgo(latest!.createdAt)}.</>
            : "Add a stream and it lands here while Recall works. Queue a few for tonight and they run one after another."}
      </p>
      <div className="w-cta">
        {failed && <button type="button" className="btn heat lg" onClick={() => onOpenProcessing(latest!)}>See what happened</button>}
        {waiting > 0 && <button type="button" className="btn heat lg" onClick={() => onOpenSession(latest!)}>Review {waiting} {waiting === 1 ? "moment" : "moments"}</button>}
        <button type="button" className={`btn lg ${failed || waiting > 0 ? "" : "heat"}`} onClick={() => onNew("local")}><Plus aria-hidden="true" />New stream</button>
        {!failed && waiting === 0 && <button type="button" className="btn lg" onClick={() => onNew("batch")}><Queue aria-hidden="true" />Queue tonight's VODs</button>}
      </div>
    </section>
  );
}

/** One past scan: when it ran, what it found, and what's left to do with it. */
function HistoryRow({ session, job, onOpenSession, onOpenProcessing }: {
  session: Session;
  job?: Job;
  onOpenSession: (session: Session) => void;
  onOpenProcessing: (session: Session) => void;
}) {
  const failed = session.status === "failed";
  const found = primaryReviewDeckClips(session.clips).length;
  const kept = session.savedClipIds.length;
  const waiting = failed ? 0 : undecidedClips(session).length;
  const length = session.vodDuration ? fmtDurationHuman(session.vodDuration) : undefined;
  return (
    <div className={`sh-row ${failed ? "is-failed" : ""}`}>
      <button
        type="button"
        className="sh-main"
        onClick={() => (failed ? onOpenProcessing(session) : onOpenSession(session))}
        aria-label={failed ? `See why ${session.name} stopped` : `Review ${session.name}`}
      >
        <span className="poster"><SessionPoster session={session} variant="card" /></span>
        <span className="sh-name">
          <b>{session.name}</b>
          <small>Scanned {relativeAgo(session.createdAt)}{length ? ` · ${length} of stream` : ""}</small>
        </span>
        {failed ? (
          <span className="sh-fail"><AlertTriangle aria-hidden="true" />{describeScanFailure(job?.errorMessage || job?.message || session.message).title}</span>
        ) : (
          <>
            <span className="sh-n"><b className="num">{found}</b><small>found</small></span>
            <span className="sh-n"><b className="num">{kept}</b><small>kept</small></span>
            <span className="sh-state">
              {waiting > 0 ? <span className="badge heat num">{waiting} to review</span> : <span className="t3"><Check aria-hidden="true" />Reviewed</span>}
            </span>
          </>
        )}
        <ChevronRight className="sh-go" aria-hidden="true" />
      </button>
      {!failed && <button type="button" className="btn sm ghost sh-log" onClick={() => onOpenProcessing(session)}>Scan log</button>}
    </div>
  );
}

function waitingAhead(row: Row, rows: Row[]) {
  return rows.slice(0, row.index).filter((r) => r.lane === "running" || r.lane === "waiting").length;
}

function rowTitle(row: Row) {
  return row.session?.name || row.job?.url?.split(/[\\/]/).pop() || "Queued scan";
}

function QueueRow({ row, rows, selected, onSelect }: { row: Row; rows: Row[]; selected: boolean; onSelect: () => void }) {
  const progress = Math.round(row.job?.progress ?? 0);
  const ahead = waitingAhead(row, rows);
  const sub =
    row.lane === "running" ? `${row.job?.stage?.label || "Scanning"} · ${progress}%`
      : row.lane === "waiting" ? (ahead ? `Waiting · ${ahead} ahead` : "Next up")
        : row.lane === "done" ? `${primaryReviewDeckClips(row.session?.clips ?? []).length} moments ready`
          : row.lane === "failed" ? describeScanFailure(row.job?.errorMessage || row.job?.message || row.session?.message).title
            : "Cancelled";
  return (
    <button type="button" className={`qi ${selected ? "on" : ""}`} aria-pressed={selected} onClick={onSelect}>
      <span className="poster">{row.session && <SessionPoster session={row.session} variant="card" />}</span>
      <span className="qi-copy">
        <b>{rowTitle(row)}</b>
        <small>{sub}</small>
        {row.lane === "running" && <span className="qi-bar" aria-hidden="true"><i style={{ transform: `scaleX(${progress / 100})` }} /></span>}
      </span>
      <span className="t3 qi-end">
        {row.lane === "failed" ? <AlertTriangle className="is-bad" aria-label="Needs attention" /> : row.lane === "done" ? <ChevronRight aria-hidden="true" /> : row.lane === "running" ? <span className="num">{progress}%</span> : null}
      </span>
    </button>
  );
}

function SelectedScan({ row, rows, clock, rates, cancelling, onCancel, onOpenSession, onOpenProcessing }: {
  row: Row;
  rows: Row[];
  clock: number;
  rates: ScanRates | null;
  cancelling: boolean;
  onCancel: () => void;
  onOpenSession: (session: Session) => void;
  onOpenProcessing: (session: Session) => void;
}) {
  const { job, session, lane } = row;
  const [timeline, setTimeline] = useState<ReactionTimeline | null>(null);
  const requested = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (lane !== "done" || requested.current === row.id) return;
    requested.current = row.id;
    setTimeline(null);
    let alive = true;
    void fetchReactionTimeline(row.id).then((t) => { if (alive) setTimeline(t); }).catch(() => {});
    return () => { alive = false; };
  }, [lane, row.id]);

  const title = rowTitle(row);
  const progress = lane === "done" ? 100 : Math.max(0, Math.min(100, Math.round(job?.progress ?? 0)));
  const deck = primaryReviewDeckClips(session?.clips ?? []);
  const moments = lane === "done" ? deckMoments(deck) : foundMoments(job?.foundClips ?? []);
  const curve = lane === "done" ? timeline : job?.liveTimeline ?? null;
  const curveEnd = curve?.t?.length ? curve.t[curve.t.length - 1] : 0;
  const duration = Math.max(job?.vodDuration ?? 0, session?.vodDuration ?? 0, curveEnd);
  const series = curveSeries(curve, duration);
  const twitch = /^https?:\/\/(?:www\.)?twitch\.tv\//i.test(job?.url ?? session?.sourceUrl ?? "");
  const downloading = lane === "running" && twitch && job?.currentPhase === "Resolving Input";
  const readSeconds = Math.max(job?.scannedSeconds ?? 0, job?.transcribedSeconds ?? 0);
  const readTo = lane === "running" && !downloading && duration > 0 && readSeconds > 0 ? Math.min(1, readSeconds / duration) : null;
  const left = lane === "running" ? timeLeftSeconds(job, clock, true, rates, duration, twitch) : null;
  const ahead = waitingAhead(row, rows);
  const failure = describeScanFailure(job?.errorMessage || job?.message || session?.message);
  const health = describeScanHealth(job?.scanHealth);
  const task = lane === "running" ? foldActivity(job?.events ?? [], job?.id, 1)[0] : undefined;
  const now = task ? (task.detail ? `${task.label} · ${task.detail}` : task.label) : job?.message || "Recall is getting ready.";
  const kept = session?.savedClipIds.length ?? 0;

  const eyebrow =
    lane === "running" ? `Scanning · ${job?.stage?.label || job?.currentPhase || "Preparing"}${moments.length ? ` · ${moments.length} found` : ""}`
      : lane === "waiting" ? (ahead ? `Waiting · ${ahead} ${ahead === 1 ? "scan" : "scans"} ahead` : "Next scan in line")
        : lane === "done" ? `Scan finished · ${relativeAgo(session?.updatedAt || session?.createdAt || Date.now())}`
          : lane === "failed" ? "Scan stopped"
            : "Cancelled";
  const strip = lane === "running" || lane === "done";

  return (
    <section className={`live-scan glass is-${lane}`} aria-labelledby="live-scan-title">
      <div className="sc-top">
        <div className="sc-id">
          <div className="row">
            <span className="eyebrow">{eyebrow}</span>
            {lane === "waiting" && <span className="badge">Queued</span>}
            {lane === "failed" && <span className="badge danger">Needs attention</span>}
          </div>
          <h2 className="disp" id="live-scan-title">{title}</h2>
          {lane === "running" && <p className="sc-now">{now}</p>}
          {lane === "done" && <p className="sc-now">{deck.length ? `${deck.length} ${deck.length === 1 ? "moment" : "moments"} to review${kept ? `, ${kept} kept so far` : ""}.` : "Nothing made the cut this time."}</p>}
        </div>
        {strip && (
          <div className="ls-vitals">
            <ProgressRing value={progress} done={lane === "done"} />
            <div className="ls-eta">
              <span>{lane === "done" ? "Stream" : "Time left"}</span>
              <b className="num">{lane === "done" ? (duration ? fmtDurationHuman(duration) : "–") : left == null ? "Measuring…" : fmtEstimate(left)}</b>
              <span className="num">{lane === "done" ? `${moments.length} pinned` : left != null ? `Done around ${clockIn(left, clock)}` : `${fmtClock((job?.elapsedSeconds ?? 0))} so far`}</span>
            </div>
          </div>
        )}
      </div>

      {strip && (
        <ScanStrip
          className="is-compact"
          duration={duration}
          series={series}
          preliminary={lane === "running" && !!series && !!curve?.preliminary}
          readTo={readTo}
          beam={readTo != null && readTo < 1 ? readTo : null}
          beamLabel={readTo != null && readTo < 1 ? fmtClock(readSeconds) : undefined}
          segments={curve?.segments}
          moments={moments}
          hot={null}
          download={downloading ? Math.round(Math.max(0, Math.min(1, job?.downloadProgress ?? 0)) * 100) : null}
          lanes={job?.liveLanes}
          candidates={lane === "running" ? job?.candidates : undefined}
          emptyText={lane === "done" ? "This scan didn't keep a reaction curve." : "The reaction curve draws in once Recall has read the stream."}
        />
      )}

      {lane === "waiting" && (
        <div className="scan-facts">
          <div><b className="disp">{ahead ? `After ${ahead}` : "Next up"}</b><small>{ahead ? `${ahead === 1 ? "scan" : "scans"} ahead of it` : "starts when the current scan finishes"}</small></div>
          <div><b className="disp num">{duration ? fmtDurationHuman(duration) : "–"}</b><small>of stream</small></div>
          {(() => {
            const takes = estimateScanSeconds(rates, duration, twitch);
            return takes != null ? <div><b className="disp num">{fmtEstimate(takes)}</b><small>to scan on this PC</small></div> : null;
          })()}
        </div>
      )}
      {lane === "failed" && (
        <div className="scan-fail">
          <AlertTriangle aria-hidden="true" />
          <div><b>{failure.title}</b><p className="t3">{failure.hint}</p></div>
        </div>
      )}
      {health && lane !== "failed" && <p className={`scan-health is-${health.tone}`}><b>{health.title}.</b> {health.detail}</p>}

      {lane === "done" && deck.length > 0 && (
        <div className="found" aria-label="What Recall found">
          {deck.slice(0, 8).map((clip) => (
            <div key={clip.id} className="found-clip">
              <span className="fr"><Poster clip={clip} className="rl-media" /></span>
              <h3>{clip.hookLine || clip.title}</h3>
            </div>
          ))}
        </div>
      )}

      <div className="w-cta">
        {lane === "done" && session && (
          <>
            <button type="button" className="btn heat lg" onClick={() => onOpenSession(session)}>Review {deck.length} {deck.length === 1 ? "moment" : "moments"}</button>
            <button type="button" className="btn lg" onClick={() => onOpenProcessing(session)}>Scan log</button>
          </>
        )}
        {lane === "running" && session && (
          <>
            <button type="button" className="btn lg" onClick={() => onOpenProcessing(session)}>Open live scan</button>
            <button type="button" className="btn lg ghost" disabled={cancelling} onClick={onCancel}><X aria-hidden="true" />{cancelling ? "Cancelling…" : "Cancel scan"}</button>
          </>
        )}
        {lane === "waiting" && (
          <button type="button" className="btn lg ghost" disabled={cancelling} onClick={onCancel}><X aria-hidden="true" />{cancelling ? "Removing…" : "Remove from queue"}</button>
        )}
        {lane === "failed" && session && (
          <button type="button" className="btn heat lg" onClick={() => onOpenProcessing(session)}>See what happened</button>
        )}
      </div>
    </section>
  );
}

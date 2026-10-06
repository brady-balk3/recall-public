// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * One scan, up close. The VOD runs left to right as a strip: the part Recall
 * has read fills in behind a moving head, the reaction curve draws over it,
 * and each moment pins onto the curve the second it's found. Found moments
 * stack up below as cards; the aside folds the feed into one line per task.
 *
 * A stopped scan explains itself in plain words with the fix first and the raw
 * error folded away. A finished one hands off to review and keeps its log.
 */
import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "../lib/api";
import { writeClipboardText } from "../lib/clipboard";
import { describeScanFailure, describeScanHealth } from "../lib/copy";
import { buildLocalDiagnostics } from "../lib/diagnostics";
import { fmtClock, fmtDurationHuman } from "../lib/format";
import { AlertTriangle, ArrowLeft, Check, ChevronDown, Copy, FolderOpen, Play, RefreshCw, X } from "../lib/icons";
import { primaryReviewDeckClips } from "../lib/reviewDeck";
import { useJobStore, type Job, type Session } from "../lib/store";
import { clockIn, fmtEstimate, type ScanRates } from "./estimate";
import { ActivityList, useScanLog } from "./ScanLog";
import { activityFraction, activityItems, foldActivity, stageIndex, type ActivityTask } from "./scanModel";
import { candidatesChecked, curveSeries, deckMoments, foundMoments, ProgressRing, ScanStrip, Sparkline, timeLeftSeconds, useReactionTimeline, type Moment } from "./scanStrip";

export interface LiveScanViewProps {
  job?: Job;
  session?: Session;
  cancelRequested: boolean;
  onCancel: () => void;
  onRetry: () => void;
  /** A Twitch link that won't download: scan a downloaded copy instead. */
  onUseFile?: () => void;
  onBack: () => void;
  onReview?: (session: Session) => void;
  /** This PC's measured scan speed: the time-left estimate until the engine's own is sure. */
  rates?: ScanRates | null;
}

const TERMINAL = new Set(["completed", "failed", "cancelled"]);
const STAGES = [
  ["Get recording", "Opening the file and mapping its tracks"],
  ["Read stream", "Voice, facecam, gameplay, and what's on screen"],
  ["Find moments", "Building the reaction curve and ranking clips"],
  ["Write captions", "Turning each moment into a review-ready draft"],
  ["Cut previews", "Vertical previews and posters for review"],
] as const;

const isTwitch = (url?: string) => /^https?:\/\/(?:www\.)?twitch\.tv\/videos\/\d+/i.test(url ?? "");

function titleOf(job?: Job, session?: Session) {
  if (session?.name) return session.name;
  const url = job?.url ?? "";
  if (isTwitch(url)) return "Twitch VOD";
  return url.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "This scan";
}

const clockOf = (ms: number) => new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
const fmtTook = (s: number) => (s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s`);

export function LiveScanView(props: LiveScanViewProps) {
  const { job, session, onBack } = props;
  const status = job?.status ?? session?.status;
  return (
    <div className="page livescan">
      <button type="button" className="btn ghost sm ls-back" onClick={onBack}><ArrowLeft aria-hidden="true" />Scans</button>
      {!job && !session ? (
        <section className="scan-idle glass">
          <span className="eyebrow">Nothing scanning</span>
          <h2 className="disp">No scan is open.</h2>
          <p className="t3">Pick one from Scans to watch it here.</p>
        </section>
      ) : status === "failed" ? (
        <StoppedScan {...props} />
      ) : status === "cancelled" ? (
        <CancelledScan {...props} />
      ) : status === "completed" ? (
        <FinishedScan {...props} />
      ) : (
        <RunningScan {...props} />
      )}
    </div>
  );
}

function useTicker(running: boolean, key?: string) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    setNow(Date.now());
    if (!running) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running, key]);
  return now;
}

interface RailProps {
  moments: Moment[];
  series: number[] | null;
  duration: number;
  onHot: (id: string | null) => void;
  emptyText: string;
}

function MomentRail({ moments, series, duration, onHot, emptyText }: RailProps) {
  if (!moments.length) {
    return (
      <div className="ls-rail">
        <div className="ls-ghost lead">{emptyText}</div>
        <div className="ls-ghost" /><div className="ls-ghost" /><div className="ls-ghost" />
      </div>
    );
  }
  return (
    <ol className="ls-rail" aria-label="Moments found">
      {moments.map((m) => (
        <li key={m.id} className={`ls-card ${m.ready ? "is-ready" : m.captioned ? "is-captioned" : ""}`} onMouseEnter={() => onHot(m.id)} onMouseLeave={() => onHot(null)}>
          <div className="thumb">
            {m.thumbUrl && <img src={m.thumbUrl} alt="" loading="lazy" />}
            <Sparkline series={series} t={m.t + m.duration / 2} duration={duration} />
            <span className="no">{m.no}</span>
            <span className="ts num">{fmtClock(m.t)}</span>
            {m.ready && !m.thumbUrl && <span className="play"><Play aria-hidden="true" /></span>}
          </div>
          <div className="body">
            <b className="ttl">{m.title}</b>
            <div className="meta"><span className="state">{m.ready ? "Preview ready" : m.captioned ? "Captioned" : "Found"}</span><span className="num">{fmtClock(m.duration)}</span></div>
          </div>
        </li>
      ))}
    </ol>
  );
}

function StageList({ index, fill, note }: { index: number; fill: (i: number) => number; note: (i: number) => string }) {
  return (
    <ol className="ls-steps" aria-label="Scan stages">
      {STAGES.map(([label], i) => {
        const state = i < index ? "done" : i === index ? "run" : "";
        return (
          <li key={label} className={`ls-step ${state}`} aria-current={i === index ? "step" : undefined}>
            <span className="bar"><i style={{ transform: `scaleX(${fill(i)})` }} /></span>
            <span className="nm">{i < index && <Check aria-hidden="true" />}{label}</span>
            <span className="st num">{note(i)}</span>
          </li>
        );
      })}
    </ol>
  );
}

function TaskFeed({ tasks, running }: { tasks: ActivityTask[]; running: boolean }) {
  return (
    <ul className="ls-tasks">
      {tasks.map((task, i) => {
        const active = running && i === 0;
        const took = (task.endedAt - task.startedAt) / 1000;
        const frac = active ? activityFraction(task.detail) : null;
        return (
          <li key={task.key} className={`ls-task ${active ? "is-active" : ""}`}>
            <span className="ic" aria-hidden="true">{active ? <span className="spin" /> : took >= 1 || task.ok ? <Check /> : <span className="ev" />}</span>
            <span className="lb">{task.label}</span>
            <time className="num">{!active && took >= 5 ? fmtTook(took) : clockOf(task.startedAt)}</time>
            {task.detail && <span className="dt num">{task.detail}</span>}
            {frac != null && <span className="mb" aria-hidden="true"><i style={{ transform: `scaleX(${frac})` }} /></span>}
          </li>
        );
      })}
    </ul>
  );
}

function TechChips({ job, elapsed, speed }: { job?: Job; elapsed: number; speed: string | null }) {
  const chips: [string, string][] = [];
  if (job?.scanHealth?.device) chips.push(["device", job.scanHealth.device]);
  if (job?.totalWorkers) chips.push(["workers", `${job.activeWorkers ?? 0}/${job.totalWorkers}`]);
  if (speed) chips.push(["speed", speed]);
  if (job?.scanHealth?.visual?.status) chips.push(["vision", String(job.scanHealth.visual.status)]);
  if (job?.scanHealth?.semantic?.status) chips.push(["ai review", String(job.scanHealth.semantic.status)]);
  chips.push(["elapsed", fmtClock(elapsed)]);
  if (job?.currentPhase) chips.push(["phase", job.currentPhase]);
  return (
    <section className="panel glass ls-tech" aria-label="Technical details">
      {chips.map(([k, v]) => <span key={k}>{k} <b className="num">{v}</b></span>)}
    </section>
  );
}

function RunningScan({ job, session, cancelRequested, onCancel, rates }: LiveScanViewProps) {
  const running = !!job && !TERMINAL.has(job.status);
  const now = useTicker(running, job?.id);
  const [hot, setHot] = useState<string | null>(null);
  const index = stageIndex(job);
  const progress = Math.max(0, Math.min(100, job?.progress ?? 0));
  const moments = foundMoments(job?.foundClips ?? []);
  const twitch = isTwitch(job?.url);
  const downloading = twitch && job?.currentPhase === "Resolving Input";
  const downloadPct = Math.round(Math.max(0, Math.min(1, job?.downloadProgress ?? 0)) * 100);
  const timeline = job?.liveTimeline;
  const timelineEnd = timeline?.t?.length ? timeline.t[timeline.t.length - 1] : 0;
  const duration = Math.max(job?.vodDuration ?? 0, timelineEnd, session?.vodDuration ?? 0);
  const series = curveSeries(timeline, duration);
  const transcribing = job?.currentPhase === "Reaction" && job?.transcribedSeconds != null;
  const readSeconds = Math.max(job?.scannedSeconds ?? 0, job?.transcribedSeconds ?? 0);
  const readTo = duration > 0 && readSeconds > 0 ? Math.min(1, readSeconds / duration) : null;
  const stageProgress = downloading ? downloadPct : Math.max(0, Math.min(100, job?.stage?.progress ?? 0));
  const elapsed = (job?.elapsedSeconds ?? 0) + (running && job?.elapsedSyncedAt ? Math.max(0, (now - job.elapsedSyncedAt) / 1000) : 0);
  const left = timeLeftSeconds(job, now, running, rates, duration, twitch);
  const speed = transcribing && job?.transcriptionSpeed ? `${job.transcriptionSpeed.toFixed(1)}× realtime` : job?.scanSpeed ? `${job.scanSpeed.toFixed(1)}× realtime` : null;
  const health = describeScanHealth(job?.scanHealth);
  const tasks = foldActivity(job?.events ?? [], job?.id);
  const queued = job?.status === "queued" || /waiting for an earlier scan/i.test(job?.message ?? "");
  const beam = !downloading && readTo != null && readTo < 1 ? readTo : null;

  const activeNote = () => {
    if (index === 0) return downloading ? `${downloadPct}%` : "Opening the recording";
    if (index === 1) return readSeconds && duration ? `${fmtClock(readSeconds)} of ${fmtClock(duration)}` : `${Math.round(stageProgress)}%`;
    if (index === 2) {
      if (transcribing && (job?.transcribedSeconds ?? 0) < duration) return `Transcribed ${fmtClock(job?.transcribedSeconds ?? 0)} of ${fmtClock(duration)}`;
      if (job?.candidates?.length) return `${candidatesChecked(job.candidates)} of ${job.candidates.length} checked`;
      return moments.length ? `${moments.length} found so far` : "Scoring reactions";
    }
    return `${Math.round(stageProgress)}%`;
  };

  return (
    <div className="ls-grid">
      <section className="ls-stage glass" aria-live="polite" aria-labelledby="ls-title">
        <header className="ls-head">
          <div className="ls-id">
            <span className="eyebrow"><span className="ls-dot" />{queued ? "In the queue" : "Scanning now"} · {twitch ? "Twitch VOD" : "Local recording"}</span>
            <h2 className="disp" id="ls-title">{titleOf(job, session)}</h2>
            <p className="ls-now">{cancelRequested ? "Recall is finishing the current step, then it closes the scan." : job?.message || STAGES[index][1]}</p>
          </div>
          <div className="ls-vitals">
            <ProgressRing value={progress} />
            <div className="ls-eta">
              <span>{queued ? "Starts after" : "Time left"}</span>
              <b className="num">{queued ? "the scan ahead" : left == null ? "Measuring…" : fmtEstimate(left)}</b>
              <span className="num">{!queued && left != null ? `Done around ${clockIn(left, now)}` : `${fmtClock(elapsed)} so far`}</span>
            </div>
            <button type="button" className="btn sm ghost ls-cancel" onClick={onCancel} disabled={cancelRequested || !running}>
              <X aria-hidden="true" />{cancelRequested ? "Stopping…" : queued ? "Remove from queue" : "Cancel scan"}
            </button>
          </div>
        </header>

        <ScanStrip
          duration={duration}
          series={series}
          preliminary={!!series && !!timeline?.preliminary}
          readTo={downloading ? null : readTo}
          beam={beam}
          beamLabel={beam != null ? fmtClock(readSeconds) : undefined}
          segments={timeline?.segments}
          moments={moments}
          hot={hot}
          download={downloading ? downloadPct : null}
          lanes={job?.liveLanes}
          candidates={job?.candidates}
          emptyText={queued ? "Recall starts reading once the scan ahead of it finishes." : "The reaction curve draws in here once Recall has read the stream."}
        />

        <StageList
          index={index}
          fill={(i) => (i < index ? 1 : i === index ? Math.max(0.04, stageProgress / 100) : 0)}
          note={(i) => (i < index ? "Done" : i === index ? activeNote() : i === index + 1 ? "Up next" : "")}
        />
        {health && <p className={`scan-health is-${health.tone}`}><b>{health.title}.</b> {health.detail}</p>}

        <div className="ls-found-h">
          <h3 className="disp">Found so far</h3>
          {moments.length > 0 && <span className="count num">{moments.length}</span>}
          {moments.length > 0 && <span className="sub">Ranking happens at the end</span>}
        </div>
        <MomentRail moments={moments} series={series} duration={duration} onHot={setHot} emptyText="Moments pin here the second Recall spots one." />
      </section>

      <aside className="ls-aside">
        <section className="panel glass ls-doing" aria-label="What Recall is doing">
          <span className="eyebrow">What Recall is doing</span>
          {tasks.length ? <TaskFeed tasks={tasks} running={running && !queued} /> : <p className="t3">Each step shows up here as it starts.</p>}
        </section>
        <TechChips job={job} elapsed={elapsed} speed={speed} />
        <p className="t3 ls-leave">You can leave this screen. The scan keeps going, even with the window closed.</p>
      </aside>
    </div>
  );
}

function useDiagnostics(job?: Job) {
  const apiEndpoint = useJobStore((s) => s.settings.apiEndpoint);
  const [state, setState] = useState<"idle" | "copying" | "copied" | "copied-local" | "error">("idle");
  const copy = useCallback(async () => {
    if (!job?.id || state === "copying") return;
    setState("copying");
    let text = buildLocalDiagnostics(job);
    let local = true;
    try {
      const res = await apiFetch(`/jobs/${job.id}/diagnostics`, undefined, apiEndpoint);
      if (res.ok) {
        const data = await res.json();
        if (typeof data.text === "string" && data.text.trim()) { text = data.text; local = false; }
      }
    } catch {
      // A start-up failure may have no engine job at all; the local report
      // still carries the source, stage, and reason.
    }
    try {
      await writeClipboardText(text);
      setState(local ? "copied-local" : "copied");
    } catch {
      setState("error");
    }
    window.setTimeout(() => setState("idle"), 2500);
  }, [job, apiEndpoint, state]);
  const label = state === "copying" ? "Building the report…"
    : state === "copied" ? "Copied. Send it to the developer"
      : state === "copied-local" ? "Copied what Recall had"
        : state === "error" ? "Clipboard blocked, try again"
          : "Copy diagnostics report";
  return { copy, label, busy: state === "copying" };
}

function StoppedScan({ job, session, onRetry, onUseFile }: LiveScanViewProps) {
  const failure = describeScanFailure(job?.errorMessage || job?.message || session?.message);
  const diagnostics = useDiagnostics(job);
  const twitch = isTwitch(job?.url ?? session?.sourceUrl);
  const linkFailed = /couldn't be downloaded/.test(failure.title);
  return (
    <section className="panel glass ls-stopped" role="alert">
      <span className="badge danger"><AlertTriangle aria-hidden="true" />{failure.stage ? `Stopped during ${failure.stage}` : "Scan stopped"}</span>
      <h2 className="disp">{failure.title.replace(/\.$/, "")}</h2>
      <p className="ls-hint">{failure.hint}</p>
      <p className="t3 ls-which">{titleOf(job, session)}</p>
      <div className="row wrap">
        <button type="button" className="btn heat" onClick={onRetry}><RefreshCw aria-hidden="true" />Try again</button>
        {twitch && onUseFile && <button type="button" className="btn" onClick={onUseFile}><FolderOpen aria-hidden="true" />Use a downloaded file</button>}
        <button type="button" className="btn ghost" onClick={() => void diagnostics.copy()} disabled={diagnostics.busy} aria-live="polite"><Copy aria-hidden="true" />{diagnostics.label}</button>
      </div>
      {twitch && linkFailed && (
        <div className="fix">
          <b>How to fix it</b>
          <ol>
            <li><span>1</span><div><b>Download the VOD</b><small>Creator Dashboard, then Content, Past broadcasts, and Download on the VOD.</small></div></li>
            <li><span>2</span><div><b>Choose the file here</b><small>Use a downloaded file brings it in with this stream's name, ready to scan.</small></div></li>
            <li><span>3</span><div><b>Or make the VOD public</b><small>Then press Try again. Sub-only and deleted VODs can't be fetched.</small></div></li>
          </ol>
        </div>
      )}
      {failure.raw && (
        <details className="ls-raw">
          <summary>Technical details<ChevronDown aria-hidden="true" /></summary>
          <pre>{failure.raw}</pre>
        </details>
      )}
    </section>
  );
}

function CancelledScan({ job, session, onRetry }: LiveScanViewProps) {
  return (
    <section className="panel glass ls-stopped" role="status">
      <span className="eyebrow">Cancelled</span>
      <h2 className="disp">{titleOf(job, session)}</h2>
      <p className="ls-hint">Recall stopped this scan before it finished. Anything else in the queue carried on.</p>
      <div className="row"><button type="button" className="btn heat" onClick={onRetry}><RefreshCw aria-hidden="true" />Scan it again</button></div>
    </section>
  );
}

function FinishedScan({ job, session, onReview }: LiveScanViewProps) {
  const [hot, setHot] = useState<string | null>(null);
  const deck = primaryReviewDeckClips(session?.clips ?? []);
  const moments = deckMoments(deck);
  const log = useScanLog(session?.id ?? job?.id);
  const live = activityItems(job);
  const { load } = log;
  useEffect(() => { if (!live.length) load(); }, [live.length, load]);
  const items = live.length ? live : log.items ?? [];
  const timeline = useReactionTimeline(session?.id ?? job?.id, job?.liveTimeline, true);
  const timelineEnd = timeline?.t?.length ? timeline.t[timeline.t.length - 1] : 0;
  const duration = Math.max(session?.vodDuration ?? 0, job?.vodDuration ?? 0, timelineEnd);
  const series = curveSeries(timeline, duration);
  const kept = session?.savedClipIds.length ?? 0;
  const took = job?.elapsedSeconds;
  return (
    <div className="ls-grid">
      <section className="ls-stage glass is-done" aria-labelledby="ls-title">
        <header className="ls-head">
          <div className="ls-id">
            <span className="eyebrow"><span className="ls-dot" />Scan finished · {isTwitch(job?.url ?? session?.sourceUrl) ? "Twitch VOD" : "Local recording"}</span>
            <h2 className="disp" id="ls-title">{titleOf(job, session)}</h2>
            <p className="ls-now">{deck.length ? `${deck.length} ${deck.length === 1 ? "moment is" : "moments are"} ready to review.` : "Recall didn't find a moment worth reviewing in this one."}</p>
          </div>
          <div className="ls-vitals">
            <ProgressRing value={100} done />
            <div className="ls-eta">
              <span>{took ? "Took" : "Stream"}</span>
              <b className="num">{took ? fmtDurationHuman(took) : duration ? fmtDurationHuman(duration) : "–"}</b>
              <span className="num">{kept ? `${kept} kept so far` : took && duration ? `${fmtDurationHuman(duration)} of stream` : ""}</span>
            </div>
            {session && onReview && deck.length > 0 && (
              <button type="button" className="btn heat ls-review" onClick={() => onReview(session)}><Play aria-hidden="true" />Review {deck.length} {deck.length === 1 ? "moment" : "moments"}</button>
            )}
          </div>
        </header>

        <ScanStrip
          duration={duration}
          series={series}
          readTo={null}
          beam={null}
          segments={timeline?.segments}
          moments={moments}
          hot={hot}
          download={null}
          lanes={job?.liveLanes}
          emptyText="This scan didn't keep a reaction curve."
        />

        <div className="ls-found-h">
          <h3 className="disp">Moments</h3>
          {moments.length > 0 && <span className="count num">{moments.length}</span>}
          {moments.length > 0 && <span className="sub">In stream order. Review ranks them</span>}
        </div>
        <MomentRail moments={moments} series={series} duration={duration} onHot={setHot} emptyText="Nothing made the cut this time." />
      </section>
      <aside className="ls-aside">
        <section className="panel glass ls-doing" aria-label="Scan log">
          <span className="eyebrow">Scan log</span>
          {log.loading && !items.length && <p className="t3">Loading the scan history…</p>}
          {log.failed && !items.length && <p className="t3">The scan history isn't available for this session.</p>}
          {items.length > 0 && <ActivityList items={items.slice(0, 30)} />}
          {!log.loading && !log.failed && log.items && !items.length && <p className="t3">No scan activity was recorded.</p>}
        </section>
      </aside>
    </div>
  );
}

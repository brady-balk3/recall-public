// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The stream strip and its parts, shared by the live scan screen and the
 * Scans page card: the VOD left to right with the read coverage, the reaction
 * curve, numbered moment pins, game bands, and the read head; plus the
 * progress ring and the time-left estimate.
 */
import { useEffect, useId, useState } from "react";
import { fmtClock } from "../lib/format";
import { Check } from "../lib/icons";
import { fetchReactionTimeline, type Clip, type FoundClip, type Job, type ReactionTimeline, type ScanCandidate, type ScanLanes } from "../lib/store";
import { estimateScanSeconds, type ScanRates } from "./estimate";
import { timestampSeconds } from "./scanModel";

/** A moment on the strip and in the rail, numbered in stream order. */
export type Moment = { id: string; no: number; t: number; duration: number; title: string; ready: boolean; captioned?: boolean; thumbUrl?: string };

export function foundMoments(found: FoundClip[]): Moment[] {
  return found
    .flatMap((f) => {
      const t = timestampSeconds(f.timestamp);
      return t == null ? [] : [{ id: f.clipId, no: 0, t, duration: f.duration, title: f.title, ready: !f.pending, captioned: !!f.captioned }];
    })
    .sort((a, b) => a.t - b.t)
    .map((m, i) => ({ ...m, no: i + 1 }));
}

export function deckMoments(clips: Clip[]): Moment[] {
  return clips
    .filter((clip) => clip.start_time != null)
    .map((clip) => ({ id: clip.id, no: 0, t: clip.start_time ?? 0, duration: clip.duration, title: clip.title, ready: true, thumbUrl: clip.thumbUrl }))
    .sort((a, b) => a.t - b.t)
    .map((m, i) => ({ ...m, no: i + 1 }));
}

// The strip draws on a 1000 x 300 box; the curve is resampled onto the VOD's
// own clock so pins, the read head, and the curve all share one x axis.
const W = 1000;
const H = 300;
const BINS = 240;
const yOf = (v: number) => H - 6 - v * (H - 44);

/** R(t) on the VOD's clock: peak-preserving bins, one light smoothing pass, 0 to 1. */
export function curveSeries(timeline: ReactionTimeline | null | undefined, duration: number): number[] | null {
  const r = timeline?.r;
  if (!timeline || !r || r.length < 2 || duration <= 0) return null;
  const t = timeline.t?.length === r.length ? timeline.t : r.map((_, i) => (i / (r.length - 1)) * duration);
  const bins = new Array<number>(BINS).fill(-1);
  t.forEach((time, i) => {
    const b = Math.max(0, Math.min(BINS - 1, Math.floor((time / duration) * BINS)));
    bins[b] = Math.max(bins[b], r[i]);
  });
  let carry = bins.find((v) => v >= 0) ?? 0;
  const filled = bins.map((v) => (v >= 0 ? (carry = v) : carry));
  const smooth = filled.map((v, i) => ((filled[i - 1] ?? v) + v * 2 + (filled[i + 1] ?? v)) / 4);
  const peak = Math.max(...smooth, 0.001);
  return smooth.map((v) => v / peak);
}

export const valueAt = (series: number[], t: number, duration: number) =>
  series[Math.max(0, Math.min(series.length - 1, Math.floor((t / duration) * series.length)))];

export const seriesPath = (series: number[], w = W, y: (v: number) => number = yOf) =>
  series.map((v, i) => `${i ? "L" : "M"}${(((i + 0.5) / series.length) * w).toFixed(1)} ${y(v).toFixed(1)}`).join("");


/** The final curve for a finished scan; the streamed one while it runs. */
export function useReactionTimeline(id: string | undefined, live: ReactionTimeline | undefined, fetchIt: boolean) {
  const [fetched, setFetched] = useState<ReactionTimeline | null>(null);
  useEffect(() => {
    setFetched(null);
    if (!id || !fetchIt) return;
    let alive = true;
    void fetchReactionTimeline(id).then((timeline) => { if (alive) setFetched(timeline); });
    return () => { alive = false; };
  }, [id, fetchIt]);
  return fetched ?? live ?? null;
}

/** A per-graphic id for its heat gradients. Ids must be unique on the page:
 * a shared one resolves to the first match, and a match inside a hidden view
 * panel paints nothing. */
function useHeatId() {
  return `heat${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
}

/** The accent's horizontal (-h) and fading vertical (-v) gradients. */
function HeatDefs({ id }: { id: string }) {
  return (
    <defs>
      <linearGradient id={`${id}-h`} x1="0" x2="1" y1="0" y2="0">
        <stop offset="0" style={{ stopColor: "var(--h1)" }} />
        <stop offset=".55" style={{ stopColor: "var(--h2)" }} />
        <stop offset="1" style={{ stopColor: "var(--h3)" }} />
      </linearGradient>
      <linearGradient id={`${id}-v`} x1="0" x2="0" y1="0" y2="1">
        <stop offset="0" style={{ stopColor: "var(--h2)", stopOpacity: 0.4 }} />
        <stop offset="1" style={{ stopColor: "var(--h2)", stopOpacity: 0 }} />
      </linearGradient>
    </defs>
  );
}

const RING = 2 * Math.PI * 36;

export function ProgressRing({ value, done }: { value: number; done?: boolean }) {
  const heat = useHeatId();
  return (
    <div className={`ls-ring ${done ? "is-done" : ""}`} role="img" aria-label={done ? "Scan finished" : `${Math.floor(value)}% done`}>
      <svg viewBox="0 0 84 84" aria-hidden="true">
        <HeatDefs id={heat} />
        <circle className="track" cx="42" cy="42" r="36" />
        <circle className="bar" cx="42" cy="42" r="36" style={done ? undefined : { stroke: `url(#${heat}-h)` }} strokeDasharray={RING} strokeDashoffset={RING * (1 - (done ? 100 : value) / 100)} />
      </svg>
      {done ? <span className="ok"><Check aria-hidden="true" /></span> : <b className="num">{Math.floor(value)}<small>%</small></b>}
    </div>
  );
}

export interface StripProps {
  className?: string;
  duration: number;
  series: number[] | null;
  preliminary?: boolean;
  /** How much of the VOD Recall has read, 0 to 1. */
  readTo: number | null;
  beam: number | null;
  beamLabel?: string;
  segments?: ReactionTimeline["segments"];
  moments: Moment[];
  hot: string | null;
  download: number | null;
  emptyText: string;
  /** Speech bars and the chat line, once the engine has read them. */
  lanes?: ScanLanes | null;
  /** Likely moments queued for the vision check, with their verdicts. */
  candidates?: ScanCandidate[];
}

/** 0..1 lane values as bars standing on the strip's floor. */
function barsPath(values: number[], height: number) {
  const w = W / values.length;
  return values
    .map((v, i) => (v > 0.01 ? `M${(i * w + w * 0.2).toFixed(1)} ${H}h${(w * 0.6).toFixed(1)}v${(-v * height).toFixed(1)}h${(-w * 0.6).toFixed(1)}Z` : ""))
    .join("");
}

function linePath(values: number[], height: number) {
  return values.map((v, i) => `${i ? "L" : "M"}${(((i + 0.5) / values.length) * W).toFixed(1)} ${(H - 6 - v * height).toFixed(1)}`).join("");
}

const CANDIDATE_CLASS: Record<ScanCandidate["status"], string> = {
  waiting: "", checking: "is-checking", post: "is-kept", maybe: "is-maybe", skip: "is-passed", error: "is-passed",
};

export function ScanStrip({ className = "", duration, series, preliminary, readTo, beam, beamLabel, segments, moments, hot, download, emptyText, lanes, candidates }: StripProps) {
  const heat = useHeatId();
  const path = series ? seriesPath(series) : "";
  const checking = candidates?.find((c) => c.status === "checking");
  // The read head leads while Recall is reading; after that the head hops to
  // whichever likely moment the vision model is looking at.
  const head = beam ?? (checking && duration > 0 ? (checking.start + checking.end) / 2 / duration : null);
  const headLabel = beam != null ? beamLabel : checking ? `checking ${fmtClock(checking.start)}` : undefined;
  const speech = lanes?.speech ?? null;
  // Mid-ASR the lane only covers what has been heard; the rest is "not yet".
  const heardTo = lanes?.speech_until != null && duration > 0 ? Math.min(1, lanes.speech_until / duration) : null;
  const chat = lanes?.chat ?? null;
  const at = (t: number) => `${Math.max(0, Math.min(100, (t / duration) * 100))}%`;
  return (
    <div className={`ls-strip ${className}`} aria-hidden="true">
      <div className="ls-legend">
        <span className={readTo != null || series ? "on" : ""}><i className="rd" />Read</span>
        <span className={speech ? "on" : ""}><i className="sp" />Speech</span>
        <span className={chat ? "on" : ""}><i className="ch" />{lanes?.chat_scope === "smart_regions" ? "Chat, likely moments" : "Chat"}</span>
        <span className={series ? "on" : ""}><i className="re" />Reaction</span>
        {preliminary && <span className="on ls-prelim">First pass. It sharpens as Recall checks each moment</span>}
      </div>
      {duration > 0 && <span className="ls-strip-meta num">{fmtClock(duration)} of stream</span>}
      <div className="ls-plot">
        <div className="ls-gridlines" />
        {readTo != null && <div className="ls-read" style={{ width: `${readTo * 100}%` }} />}
        {duration > 0 && candidates?.map((c) => (
          <span
            key={c.start}
            className={`ls-cand ${CANDIDATE_CLASS[c.status]}`}
            style={{ left: `${(c.start / duration) * 100}%`, width: `${Math.max(0.5, ((c.end - c.start) / duration) * 100)}%` }}
          />
        ))}
        {(speech || chat) && (
          <svg className="ls-lanes" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
            {speech && <path className="speech" d={barsPath(heardTo != null ? speech.map((v, i) => ((i + 1) / speech.length <= heardTo ? v : 0)) : speech, 54)} />}
            {chat && <path className="chat" d={linePath(chat, H * 0.55)} vectorEffect="non-scaling-stroke" />}
          </svg>
        )}
        {series ? (
          <svg className="ls-viz" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
            <HeatDefs id={heat} />
            <path className="fill" d={`${path}L${W} ${H}L0 ${H}Z`} style={{ fill: `url(#${heat}-v)` }} />
            <path className="line" d={path} vectorEffect="non-scaling-stroke" style={{ stroke: `url(#${heat}-h)` }} />
          </svg>
        ) : download == null && !speech && !chat && <p className="ls-empty">{emptyText}</p>}
        {duration > 0 && moments.map((m) => (
          <span
            key={m.id}
            className={`ls-pin ${m.ready ? "" : "is-pending"} ${hot === m.id ? "is-hot" : ""}`}
            style={{ left: at(m.t + m.duration / 2), top: `${((series ? yOf(valueAt(series, m.t + m.duration / 2, duration)) : H * 0.8) / H) * 100}%` }}
          >
            <b>{m.no}</b><i />
          </span>
        ))}
        {head != null && (
          <>
            <span className={`ls-beam ${beam == null ? "is-hop" : ""}`} style={{ left: `${head * 100}%` }} />
            {headLabel && <span className="ls-beam-tag num" style={{ left: `${head * 100}%` }}>{headLabel}</span>}
          </>
        )}
      </div>
      {duration > 0 && !!segments?.length && (
        <div className="ls-segs">
          {segments.map((seg) => (
            <span key={`${seg.start}-${seg.game}`} style={{ left: at(seg.start), width: `${((seg.end - seg.start) / duration) * 100}%` }}>{seg.game}</span>
          ))}
        </div>
      )}
      {duration > 0 && (
        <div className="ls-axis num">
          {[0, 0.25, 0.5, 0.75, 1].map((f) => <span key={f} style={{ left: `${f * 100}%` }}>{fmtClock(duration * f)}</span>)}
        </div>
      )}
      {download != null && (
        <div className="ls-dl">
          <div className="ls-dl-fill" style={{ width: `${download}%` }} />
          <b className="num">Downloading the VOD · {download}%</b>
          <span>The curve starts drawing once it's saved.</span>
        </div>
      )}
    </div>
  );
}

export function Sparkline({ series, t, duration }: { series: number[] | null; t: number; duration: number }) {
  const heat = useHeatId();
  if (!series || duration <= 0) return null;
  const around = Array.from({ length: 40 }, (_, i) => valueAt(series, Math.max(0, Math.min(duration, t - 240 + i * 12)), duration));
  const lo = Math.min(...around);
  const span = Math.max(...around) - lo || 1;
  const d = seriesPath(around, 200, (v) => 58 - ((v - lo) / span) * 50);
  return (
    <svg className="ls-spark" viewBox="0 0 200 60" preserveAspectRatio="none" aria-hidden="true">
      <HeatDefs id={heat} />
      <path className="fill" d={`${d}L200 60L0 60Z`} style={{ fill: `url(#${heat}-v)` }} />
      <path className="line" d={d} vectorEffect="non-scaling-stroke" style={{ stroke: `url(#${heat}-h)` }} />
      <line className="mid" x1="100" x2="100" y1="0" y2="60" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** Seconds left: the engine's own estimate once it's sure, this PC's measured
 * speed before that, null when neither knows. */
export function timeLeftSeconds(job: Job | undefined, now: number, running: boolean, rates: ScanRates | null | undefined, duration: number, twitch: boolean) {
  const eta = job?.etaSeconds == null ? null : Math.max(0, job.etaSeconds - (running && job.etaSyncedAt ? Math.max(0, (now - job.etaSyncedAt) / 1000) : 0));
  if (eta != null && (job?.etaConfidence === "medium" || job?.etaConfidence === "high")) return eta;
  const whole = estimateScanSeconds(rates ?? null, duration, twitch);
  if (whole == null) return null;
  const elapsed = (job?.elapsedSeconds ?? 0) + (running && job?.elapsedSyncedAt ? Math.max(0, (now - job.elapsedSyncedAt) / 1000) : 0);
  return Math.max(60, whole - elapsed);
}

/** How many queued likely moments the vision model has finished with. */
export function candidatesChecked(candidates?: ScanCandidate[]) {
  return (candidates ?? []).filter((c) => c.status !== "waiting" && c.status !== "checking").length;
}

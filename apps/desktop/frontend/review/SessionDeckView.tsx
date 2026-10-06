// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * One stream's review deck: the reaction curve with every moment marked on
 * it, then the moments themselves as vertical cards, ranked. Keep, maybe, or
 * pass right from a card, or open one to watch it.
 *
 * Moments just outside the deck ("second look") stay hidden until asked for.
 * Previews cleared to save space say so, with the one action that brings them
 * back. The scan log folds in at the bottom.
 */
import { useEffect, useState, type CSSProperties } from "react";
import { Poster } from "../components/Posters";
import RecallOriginBadge from "../components/RecallOriginBadge";
import { apiFetch } from "../lib/api";
import { fmtClock, fmtDurationHuman } from "../lib/format";
import { ArrowLeft, Check, Clock, Film, Info, Layers, Play, RefreshCw, Scissors, X } from "../lib/icons";
import { durationOf, hookScoreOf, openingBand, primaryReviewDeckClips, secondLookClips, sortReviewClips, type ReviewSort } from "../lib/reviewDeck";
import { useJobStore, type Clip, type ReactionTimeline, type Session } from "../lib/store";
import { timelinePath } from "../lib/timeline";
import { MomentCurve } from "../ui/MomentCurve";
import { ScanLog } from "../scans/ScanLog";

export type ReviewFilter = "all" | "kept" | "maybe" | "unreviewed";

export interface SessionDeckViewProps {
  session?: Session;
  timeline: ReactionTimeline | null;
  filter: ReviewFilter;
  setFilter: (filter: ReviewFilter) => void;
  sort: ReviewSort;
  setSort: (sort: ReviewSort) => void;
  onBack: () => void;
  onOpen: (clip: Clip) => void;
  onKeep: (clip: Clip) => void;
  onCut: (clip: Clip) => void;
  onMaybe: (clip: Clip) => void;
  onClips: () => void;
  onCuttingRoom: () => void;
  moreCandidateCount: number;
  moreCandidatesEnabled: boolean;
  moreCandidatesLoading: boolean;
  onToggleMoreCandidates: () => void;
}

const FILTERS: Array<[ReviewFilter, string]> = [["all", "All"], ["unreviewed", "To review"], ["kept", "Kept"], ["maybe", "Maybe"]];
const SORTS: Array<[ReviewSort, string]> = [["recommended", "Recommended"], ["hook", "Strongest opening"], ["timeline", "VOD order"], ["longest", "Longest first"]];

function sessionMeta(session: Session) {
  const parts: string[] = [];
  const date = session.sourceDate ? new Date(session.sourceDate) : new Date(session.createdAt);
  if (!Number.isNaN(date.getTime())) parts.push(date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }));
  if (session.vodDuration) parts.push(fmtDurationHuman(session.vodDuration));
  parts.push(/^https?:/i.test(session.sourceUrl) ? "Twitch VOD" : "Local recording");
  return parts.join(" · ");
}

export function SessionDeckView(props: SessionDeckViewProps) {
  const { session, timeline, filter, setFilter, sort, setSort, onBack, onOpen, onKeep, onCut, onMaybe, onClips, onCuttingRoom } = props;
  if (!session) {
    return (
      <div className="page deck2">
        <button type="button" className="btn ghost sm dk-back" onClick={onBack}><ArrowLeft aria-hidden="true" />Review</button>
        <section className="scan-idle glass">
          <span className="eyebrow">Nothing open</span>
          <h2 className="disp">Pick a stream to review.</h2>
          <p className="t3">Its moments land here the second the scan finishes.</p>
        </section>
      </div>
    );
  }

  const kept = new Set(session.savedClipIds);
  const deck = primaryReviewDeckClips(session.clips);
  const second = secondLookClips(session.clips);
  const ranks = new Map(deck.map((clip, index) => [clip.id, index + 1]));
  const decided = (clip: Clip) => kept.has(clip.id) || !!clip.passed || !!clip.maybe;
  const counts: Record<ReviewFilter, number> = {
    all: deck.length,
    unreviewed: deck.filter((clip) => !decided(clip)).length,
    kept: deck.filter((clip) => kept.has(clip.id)).length,
    maybe: deck.filter((clip) => clip.maybe).length,
  };
  const matches = (clip: Clip) => filter === "all" ? true
    : filter === "kept" ? kept.has(clip.id)
      : filter === "maybe" ? !!clip.maybe
        : !decided(clip);
  const list = sortReviewClips(deck.filter(matches), sort);
  const secondList = props.moreCandidatesEnabled ? sortReviewClips(second.filter(matches), sort) : [];
  const keptRuntime = session.clips.filter((clip) => kept.has(clip.id)).reduce((sum, clip) => sum + durationOf(clip), 0);
  const next = sortReviewClips(deck, "recommended").find((clip) => !decided(clip)) ?? deck[0];
  const allDone = deck.length > 0 && counts.unreviewed === 0;

  return (
    <div className="page deck2">
      <button type="button" className="btn ghost sm dk-back" onClick={onBack}><ArrowLeft aria-hidden="true" />Review</button>
      <header className="deck-head">
        <div className="dk-title">
          <div className="row wrap dk-status">
            {allDone ? <span className="badge keep"><Check aria-hidden="true" />Reviewed</span> : <span className="badge heat num">{counts.unreviewed} to review</span>}
            <span className="t3">{sessionMeta(session)}</span>
          </div>
          <h1 className="disp">{session.name}</h1>
        </div>
        <div className="row wrap dk-actions">
          <button type="button" className="btn" onClick={onCuttingRoom} title="Scrub the whole stream and cut a moment Recall missed"><Scissors aria-hidden="true" />Cut a missed moment</button>
          <button type="button" className="btn" onClick={onClips} disabled={!kept.size}><Film aria-hidden="true" />Kept clips</button>
          {next && <button type="button" className="btn heat lg" onClick={() => onOpen(next)}><Play aria-hidden="true" />{allDone ? "Watch again" : counts.unreviewed === deck.length ? "Start review" : "Keep reviewing"}</button>}
        </div>
      </header>

      <PreviewsCleared session={session} />

      <DeckCurve session={session} timeline={timeline} kept={kept} showSecond={props.moreCandidatesEnabled} onOpen={onOpen} />

      <div className="filters">
        <div className="seg" role="radiogroup" aria-label="Show">
          {FILTERS.map(([value, label]) => (
            <button key={value} type="button" role="radio" aria-checked={filter === value} onClick={() => setFilter(value)}>{label}<span className="c num">{counts[value]}</span></button>
          ))}
        </div>
        {props.moreCandidateCount > 0 && (
          <button
            type="button"
            className={`btn sm ${props.moreCandidatesEnabled ? "is-on" : ""}`}
            aria-pressed={props.moreCandidatesEnabled}
            disabled={props.moreCandidatesLoading}
            onClick={props.onToggleMoreCandidates}
            title="The strongest moments saved just outside the deck"
          >
            {props.moreCandidatesLoading ? <RefreshCw className="is-spinning" aria-hidden="true" /> : <Layers aria-hidden="true" />}
            Second look · {props.moreCandidatesEnabled ? second.length || props.moreCandidateCount : props.moreCandidateCount}
          </button>
        )}
        <span className="sp" />
        <span className="t3 dk-runtime">Kept runtime <b className="num">{fmtClock(keptRuntime)}</b></span>
        <select className="field dk-sort" value={sort} onChange={(event) => setSort(event.currentTarget.value as ReviewSort)} aria-label="Sort moments">
          {SORTS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </div>

      {list.length ? (
        <div className="dk-clips">
          {list.map((clip, index) => (
            <DeckCard key={clip.id} clip={clip} index={index} rank={ranks.get(clip.id)} kept={kept.has(clip.id)} onOpen={onOpen} onKeep={onKeep} onMaybe={onMaybe} onCut={onCut} />
          ))}
        </div>
      ) : (
        <section className="dk-empty glass">
          <p>{filter === "unreviewed" ? "Every moment has a call. Nice." : "Nothing matches this filter."}</p>
          <button type="button" className="btn sm" onClick={() => setFilter("all")}>Show all {deck.length}</button>
        </section>
      )}

      {secondList.length > 0 && (
        <>
          <div className="sec-t"><h2 className="disp">Second look</h2><span className="c">Saved just outside the deck</span></div>
          <div className="dk-clips">
            {secondList.map((clip, index) => (
              <DeckCard key={clip.id} clip={clip} index={index} kept={kept.has(clip.id)} onOpen={onOpen} onKeep={onKeep} onMaybe={onMaybe} onCut={onCut} />
            ))}
          </div>
        </>
      )}

      <ScanLog jobId={session.id} />
    </div>
  );
}

function DeckCard({ clip, index, rank, kept, onOpen, onKeep, onMaybe, onCut }: {
  clip: Clip;
  index: number;
  rank?: number;
  kept: boolean;
  onOpen: (clip: Clip) => void;
  onKeep: (clip: Clip) => void;
  onMaybe: (clip: Clip) => void;
  onCut: (clip: Clip) => void;
}) {
  const score = hookScoreOf(clip);
  const opening = openingBand(score);
  const state = kept ? "kept" : clip.passed ? "pass" : clip.maybe ? "maybe" : "";
  const title = clip.title;
  return (
    <article className={`dk-clip ${state} ${rank ? "" : "second"}`} data-clip-id={clip.id} style={{ "--i": Math.min(index, 14) } as CSSProperties}>
      <div className="dk-media">
      <button type="button" className="fr" onClick={() => onOpen(clip)} aria-label={`Open ${title}${rank ? `, pick ${rank}` : ", second look"}, ${opening.label}`}>
        <Poster clip={clip} className="rl-media" />
        <span className={`rank disp ${rank ? "" : "is-second"}`}>{rank ? String(rank).padStart(2, "0") : "+"}</span>
        <span className="dur num">{fmtClock(durationOf(clip))}</span>
        {state && (
          <span className="state">
            {state === "kept" ? <span className="badge keep"><Check aria-hidden="true" />Kept</span>
              : state === "maybe" ? <span className="badge maybe">Maybe</span>
                : <span className="badge">Passed</span>}
          </span>
        )}
      </button>
      <div className="quick" role="group" aria-label={`Decide on ${title}`}>
        <button type="button" aria-pressed={!!clip.passed} onClick={() => onCut(clip)} title="Pass · X" aria-label="Pass"><X aria-hidden="true" /></button>
        <button type="button" aria-pressed={!!clip.maybe} onClick={() => onMaybe(clip)} title="Maybe · M" aria-label="Maybe"><Clock aria-hidden="true" /></button>
        <button type="button" className="k" aria-pressed={kept} onClick={() => onKeep(clip)} title="Keep · Space" aria-label="Keep"><Check aria-hidden="true" /></button>
      </div>
      </div>
      <h3>{title}</h3>
      <div className="m">
        <span className="num">{clip.timestamp}</span>
        <span className={`dk-open ${opening.className}`} title="How fast the opening grabs, not overall quality">{opening.shortLabel}</span>
        <RecallOriginBadge provenance={clip.recallProvenance} compact />
      </div>
      <div className="hy" aria-hidden="true"><i style={{ transform: `scaleX(${Math.max(0.04, Math.min(1, score / 100))})` }} /></div>
    </article>
  );
}

function DeckCurve({ session, timeline, kept, showSecond, onOpen }: {
  session: Session;
  timeline: ReactionTimeline | null;
  kept: Set<string>;
  showSecond: boolean;
  onOpen: (clip: Clip) => void;
}) {
  const end = timeline?.t?.length ? timeline.t[timeline.t.length - 1] : undefined;
  const duration = Math.max(1, session.vodDuration ?? end ?? 1);
  const path = timelinePath(timeline, 1000, 130, { maxPoints: 260, smoothPasses: 1 });
  const marks = session.clips.filter((clip) => (showSecond || !clip.isOverflowCandidate) && clip.start_time != null);
  const hours = duration / 3600;
  const step = hours > 6 ? 2 : 1;
  const ticks = hours >= 1
    ? Array.from({ length: Math.floor(hours / step) + 1 }, (_, i) => i * step * 3600)
    : [0, duration / 2, duration];

  return (
    <section className="curvebox glass" aria-label="Reaction curve">
      <div className="row dk-curve-head">
        <span className="eyebrow">Reaction curve</span>
        <span className="t3">voice, face, chat, and what's on screen, scored together</span>
        <span className="sp" />
        <span className="dk-legend t3">
          <span><i className="lg-review" />To review</span>
          <span><i className="lg-kept" />Kept</span>
          <span><i className="lg-maybe" />Maybe</span>
        </span>
      </div>
      <div className="dk-curve" aria-hidden="true">
        {path ? (
          <svg viewBox="0 0 1000 130" preserveAspectRatio="none">
            <path d={`${path} L1000 130 L0 130 Z`} className="sw-fill" />
            <path d={path} className="sw-line" vectorEffect="non-scaling-stroke" />
          </svg>
        ) : (
          <MomentCurve clips={session.clips} duration={duration} className="sw-curve" />
        )}
      </div>
      <div className="curve-marks">
        {marks.map((clip) => {
          const state = kept.has(clip.id) ? "kept" : clip.passed ? "pass" : clip.maybe ? "maybe" : "";
          return (
            <button
              key={clip.id}
              type="button"
              className={`cm ${state} ${clip.isOverflowCandidate ? "is-second" : ""}`}
              style={{ left: `${((clip.start_time ?? 0) / duration) * 100}%`, width: `${Math.max(0.5, (durationOf(clip) / duration) * 100)}%` }}
              title={`${clip.timestamp} · ${clip.title}`}
              aria-label={`Open ${clip.title} at ${clip.timestamp}`}
              onClick={() => onOpen(clip)}
            />
          );
        })}
      </div>
      <div className="curve-axis" aria-hidden="true">
        {ticks.map((seconds) => <span key={seconds} style={{ left: `${(seconds / duration) * 100}%` }}>{fmtClock(seconds)}</span>)}
      </div>
    </section>
  );
}

/** Previews cleared from disk: say so, keep the decisions, offer the rebuild.
 * Never rebuilds on its own; a rebuild is minutes of FFmpeg, and a Twitch
 * source has to come back first. */
function PreviewsCleared({ session }: { session: Session }) {
  const settings = useJobStore((state) => state.settings);
  const addLog = useJobStore((state) => state.addLog);
  const hydrateSessionClips = useJobStore((state) => state.hydrateSessionClips);
  const refreshJobs = useJobStore((state) => state.refreshJobs);
  const [busy, setBusy] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const media = session.media;

  useEffect(() => {
    if (!rebuilding) return;
    let alive = true;
    const timer = window.setInterval(async () => {
      try {
        const response = await apiFetch(`/jobs/${session.id}/media-status`, undefined, settings.apiEndpoint);
        if (!alive || !response.ok) return;
        const status = await response.json() as { rebuild?: { status?: string } };
        if (!alive) return;
        if (status?.rebuild?.status !== "rebuilding") {
          setRebuilding(false);
          await refreshJobs();
          await hydrateSessionClips(session.id);
          addLog(`Clip previews rebuilt for ${session.name}.`, "success");
        }
      } catch {
        // A transient engine hiccup: keep polling rather than calling it failed.
      }
    }, 3000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [rebuilding, session.id, session.name, settings.apiEndpoint, addLog, hydrateSessionClips, refreshJobs]);

  if (!media || (media.rebuildable === 0 && media.unavailable === 0)) return null;
  const missing = media.rebuildable + media.unavailable;

  const startRebuild = async () => {
    setBusy(true);
    try {
      const response = await apiFetch(`/jobs/${session.id}/media/rebuild`, { method: "POST" }, settings.apiEndpoint);
      const result = await response.json().catch(() => ({})) as { status?: string; total?: number; restorable?: boolean; detail?: unknown };
      if (!response.ok) {
        const detail = result?.detail;
        throw new Error(typeof detail === "string" ? detail : "Recall couldn't start the rebuild.");
      }
      if (result?.status === "rebuilding" || result?.status === "already_running") {
        setRebuilding(true);
        addLog(`Rebuilding ${result.total ?? missing} clip previews.`, "info");
      } else if (result?.status === "source_required") {
        addLog(result.restorable
          ? "Restore this stream's Twitch source first. The Cutting Room has the restore button."
          : "The source recording is gone, so these previews can't be rebuilt.", "warn");
      } else {
        addLog("Nothing left to rebuild for this stream.", "info");
      }
    } catch (cause) {
      addLog(cause instanceof Error ? cause.message : "Recall couldn't start the rebuild.", "warn");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="notice glass" role="status" data-state={media.state}>
      <Info aria-hidden="true" />
      <div className="notice-copy">
        <b>{missing} of {media.total} previews were cleared to save space</b>
        <p>
          {media.source_state === "local"
            ? "Your calls, scores, and timestamps are all kept. Rebuild the previews whenever you want them back."
            : media.source_state === "restorable"
              ? "Your calls, scores, and timestamps are all kept. Restore the Twitch source in the Cutting Room first, then rebuild."
              : "Your calls, scores, and timestamps are all kept, but the source recording is gone too, so the video can't come back."}
        </p>
      </div>
      {media.source_state === "local" && (
        <button type="button" className="btn sm" onClick={() => void startRebuild()} disabled={busy || rebuilding}>
          <RefreshCw aria-hidden="true" />{rebuilding ? "Rebuilding…" : busy ? "Starting…" : `Rebuild ${media.rebuildable} previews`}
        </button>
      )}
    </div>
  );
}

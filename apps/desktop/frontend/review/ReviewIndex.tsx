// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Review: every deck still waiting, loudest first, one click to start. Below
 * them, the maybes you haven't settled, what your decisions say so far, and
 * the streams you've finished.
 */
import { Poster, SessionPoster } from "../components/Posters";
import { fmtClock, fmtDurationHuman } from "../lib/format";
import { Play } from "../lib/icons";
import { primaryReviewDeckClips } from "../lib/reviewDeck";
import type { Clip, Session } from "../lib/store";
import { hypeOf, undecidedClips, waitingSessions } from "../home/homeModel";
import { SessionList } from "../library/LibraryView";
import { KeepersReel } from "../ui/KeepersReel";
import { MomentCurve } from "../ui/MomentCurve";
import { isFromTwitch, sessionDate } from "../ui/SessionCard";

export interface ReviewIndexProps {
  sessions: Session[];
  learningMessage?: string;
  onStart: (session: Session) => void;
  onOpenDeck: (session: Session) => void;
  onWatch: (session: Session, clip: Clip) => void;
  onDelete: (session: Session) => void;
  onLearning: () => void;
  onLibrary: () => void;
  onNew: () => void;
}

export interface TasteStats {
  keepRate: number;
  decisions: number;
  /** Mean keeper length in seconds. */
  keeperLength: number;
}

export function tasteStats(sessions: Session[]): TasteStats {
  let kept = 0;
  let decided = 0;
  let keptSeconds = 0;
  for (const session of sessions) {
    for (const clip of session.clips) {
      const isKept = !!clip.kept || session.savedClipIds.includes(clip.id);
      if (isKept) {
        kept += 1;
        keptSeconds += clip.duration || 0;
      }
      if (isKept || clip.passed) decided += 1;
    }
  }
  return {
    keepRate: decided ? Math.round((kept / decided) * 100) : 0,
    decisions: decided,
    keeperLength: kept ? Math.round(keptSeconds / kept) : 0,
  };
}

function maybes(sessions: Session[]) {
  return sessions.flatMap((session) =>
    session.clips.filter((clip) => clip.maybe && !clip.kept && !clip.passed && !session.savedClipIds.includes(clip.id)).map((clip) => ({ clip, session })),
  );
}

export function ReviewIndex({ sessions, learningMessage, onStart, onOpenDeck, onWatch, onDelete, onLearning, onLibrary, onNew }: ReviewIndexProps) {
  const waiting = waitingSessions(sessions);
  const total = waiting.reduce((sum, s) => sum + undecidedClips(s).length, 0);
  const minutes = Math.max(1, Math.round(waiting.flatMap(undecidedClips).reduce((sum, c) => sum + (c.duration || 0), 0) / 60));
  const unsettled = maybes(sessions);
  const taste = tasteStats(sessions);
  const reviewed = sessions
    .filter((s) => s.status === "completed" && !undecidedClips(s).length && s.clips.length > 0)
    .sort((a, b) => (b.createdAt || b.updatedAt) - (a.createdAt || a.updatedAt))
    .slice(0, 6);

  return (
    <div className="page review-index">
      <div className="ph">
        <div>
          <h1 className="disp">Review</h1>
          <div className="sub">
            {total
              ? `${total} ${total === 1 ? "moment" : "moments"} across ${waiting.length} ${waiting.length === 1 ? "stream" : "streams"}, about ${minutes} ${minutes === 1 ? "minute" : "minutes"}. Go with your gut.`
              : "Nothing waiting. Every moment has a keep or a pass."}
          </div>
        </div>
        <div className="actions">
          {waiting[0] ? (
            <button type="button" className="btn heat lg" onClick={() => onStart(waiting[0])}><Play aria-hidden="true" />Review all · {total}</button>
          ) : (
            <button type="button" className="btn heat lg" onClick={onNew}>Scan a stream</button>
          )}
        </div>
      </div>

      {waiting.length > 0 && (
        <div className="decks">
          {waiting.map((session) => <DeckRow key={session.id} session={session} onStart={onStart} onOpenDeck={onOpenDeck} onWatch={onWatch} />)}
        </div>
      )}

      <div className="rv-lower">
        <div className="rv-maybes">
          <div className="sec-t">
            <h2 className="disp">Maybes to settle</h2>
            <span className="c">{unsettled.length ? `${unsettled.length} ${unsettled.length === 1 ? "clip" : "clips"} you weren't sure about` : "None. You know what you like."}</span>
          </div>
          {unsettled.length > 0 && (
            <KeepersReel
              label="Maybes to settle"
              items={unsettled.map(({ clip, session }) => ({
                key: clip.id,
                label: `${clip.hookLine || clip.title}, from ${session.name}`,
                onOpen: () => onWatch(session, clip),
                content: (
                  <>
                    <Poster clip={clip} className="rl-media" />
                    <span className="rl-hype num">{hypeOf(clip)}</span>
                    <span className="rl-b"><b>{clip.hookLine || clip.title}</b><small>{session.name}</small></span>
                  </>
                ),
              }))}
            />
          )}
        </div>
        <section className="taste glass" aria-labelledby="taste-title">
          <b id="taste-title">Your taste so far</b>
          <div className="wk-stats">
            <div><b className="disp num">{taste.keepRate}%</b><small>keep rate</small></div>
            <div><b className="disp num">{taste.decisions}</b><small>decisions</small></div>
            <div><b className="disp num">{taste.keeperLength ? fmtClock(taste.keeperLength) : "–"}</b><small>avg keeper</small></div>
          </div>
          <p className="t3">{learningMessage || "Every keep and pass teaches Recall what you like. New decks lead with it."}</p>
          <button type="button" className="btn sm" onClick={onLearning}>Manage learning</button>
        </section>
      </div>

      {reviewed.length > 0 && (
        <>
          <div className="sec-t">
            <h2 className="disp">Recently reviewed</h2>
            <span className="sp" />
            <button type="button" className="btn sm ghost" onClick={onLibrary}>Whole library</button>
          </div>
          <SessionList sessions={reviewed} onOpen={onOpenDeck} onDelete={onDelete} selecting={false} selectedIds={new Set()} onToggle={() => {}} />
        </>
      )}
    </div>
  );
}

function DeckRow({ session, onStart, onOpenDeck, onWatch }: { session: Session; onStart: ReviewIndexProps["onStart"]; onOpenDeck: ReviewIndexProps["onOpenDeck"]; onWatch: ReviewIndexProps["onWatch"] }) {
  const deck = primaryReviewDeckClips(session.clips);
  const left = undecidedClips(session);
  const decided = deck.length - left.length;
  const top = left.slice(0, 3);
  return (
    <article className="deck glass" aria-labelledby={`deck-${session.id}`}>
      <button type="button" className="deck-poster" onClick={() => onOpenDeck(session)} aria-label={`Open the ${session.name} deck`}>
        <SessionPoster session={session} variant="card" />
        {session.clips.length > 0 && <MomentCurve clips={session.clips} duration={session.vodDuration} />}
      </button>
      <div className="deck-copy">
        <h2 className="disp" id={`deck-${session.id}`}>{session.name}</h2>
        <span className="t3 deck-meta">
          {sessionDate(session)} · {isFromTwitch(session) ? "Twitch" : "Recording"}{session.vodDuration ? ` · ${fmtDurationHuman(session.vodDuration)}` : ""}
        </span>
        <span className="deck-progress">
          <span className="deck-bar" aria-hidden="true"><i style={{ transform: `scaleX(${deck.length ? decided / deck.length : 0})` }} /></span>
          <span className="t3 num">{decided ? `${decided} of ${deck.length} decided` : `${deck.length} moments, none decided yet`}</span>
        </span>
        <span className="deck-cta">
          <button type="button" className="btn heat" onClick={() => onStart(session)}><Play aria-hidden="true" />{decided ? "Continue" : "Start"} · {left.length}</button>
          <button type="button" className="btn ghost" onClick={() => onOpenDeck(session)}>Open deck</button>
        </span>
      </div>
      <div className="deck-thumbs">
        {top.map((clip) => (
          <button key={clip.id} type="button" className="deck-thumb" onClick={() => onWatch(session, clip)} aria-label={`Watch ${clip.hookLine || clip.title}`}>
            <Poster clip={clip} className="rl-media" />
            <span className="rl-hype num">{hypeOf(clip)}</span>
          </button>
        ))}
      </div>
    </article>
  );
}

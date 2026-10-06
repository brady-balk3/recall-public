// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Your week, as a story: five full-screen slides that advance on their own
 * (or on click, Space, the arrow keys). Everything on them comes from the
 * last seven days of your library.
 */
import { useEffect, useMemo, useState } from "react";
import { Poster, SessionPoster } from "../components/Posters";
import { ChevronLeft, ChevronRight, ExportIcon, X } from "../lib/icons";
import { primaryReviewDeckClips } from "../lib/reviewDeck";
import type { Clip, Session } from "../lib/store";
import { hypeOf } from "../home/homeModel";
import { coachFor, type HookGrade } from "../review/PostingCoach";
import { MomentCurve } from "../ui/MomentCurve";
import { useMoments } from "./milestones";

const WEEK = 7 * 86_400_000;
const SLIDE_MS = 5200;

export interface RecapData {
  sessions: Session[];
  moments: Clip[];
  kept: Clip[];
  best?: Clip;
  bestSession?: Session;
  hook?: Clip;
  seconds: number;
  pops: number;
  grades: Record<HookGrade, number>;
}

export function recapData(sessions: Session[], now = Date.now()): RecapData {
  const week = sessions.filter((s) => s.status === "completed" && now - (s.createdAt || s.updatedAt) < WEEK);
  const moments = week.flatMap((s) => primaryReviewDeckClips(s.clips));
  const kept = week.flatMap((s) => s.clips.filter((c) => s.savedClipIds.includes(c.id)));
  const best = [...moments].sort((a, b) => hypeOf(b) - hypeOf(a))[0];
  const hook = [...moments].sort((a, b) => coachFor(b).score - coachFor(a).score)[0];
  const grades: Record<HookGrade, number> = { A: 0, B: 0, C: 0, D: 0 };
  moments.forEach((m) => { grades[coachFor(m).grade] += 1; });
  return {
    sessions: week,
    moments,
    kept,
    best,
    bestSession: best ? week.find((s) => s.clips.some((c) => c.id === best.id)) : undefined,
    hook,
    seconds: week.reduce((sum, s) => sum + (s.vodDuration ?? 0), 0),
    pops: moments.filter((m) => hypeOf(m) >= 80).length,
    grades,
  };
}

export function WeeklyRecap({ sessions, onExport }: { sessions: Session[]; onExport: () => void }) {
  const open = useMoments((state) => state.recapOpen);
  const close = useMoments((state) => state.closeRecap);
  const [index, setIndex] = useState(0);
  const data = useMemo(() => recapData(sessions), [sessions]);
  const slides = 5;

  useEffect(() => { if (open) setIndex(0); }, [open]);
  useEffect(() => {
    if (!open || index >= slides - 1) return;
    const timer = window.setTimeout(() => setIndex((i) => Math.min(slides - 1, i + 1)), SLIDE_MS);
    return () => window.clearTimeout(timer);
  }, [open, index]);
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
      else if (event.key === "ArrowRight" || event.key === " ") { event.preventDefault(); setIndex((i) => Math.min(slides - 1, i + 1)); }
      else if (event.key === "ArrowLeft") setIndex((i) => Math.max(0, i - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, close]);

  if (!open) return null;

  const hours = Math.floor(data.seconds / 3600);
  const mins = Math.round((data.seconds % 3600) / 60);
  const maxGrade = Math.max(1, ...Object.values(data.grades));
  const empty = !data.sessions.length;

  const slide = empty ? (
    <>
      <small>Your week on stream</small>
      <b className="disp big">Quiet week.</b>
      <p>Scan a stream and next week's recap fills itself in.</p>
    </>
  ) : [
    <>
      <small>Your week on stream</small>
      <b className="disp big num">{hours}h {mins}m</b>
      <p>{data.sessions.length} {data.sessions.length === 1 ? "stream" : "streams"}. Recall watched every minute so you didn't have to.</p>
      <div className="rc-posters">{data.sessions.slice(0, 5).map((s, k) => <span key={s.id} style={{ ["--i" as string]: k }}><SessionPoster session={s} variant="card" /></span>)}</div>
    </>,
    <>
      <small>Chat lost it</small>
      <b className="disp big num">{data.pops} {data.pops === 1 ? "time" : "times"}</b>
      <p>Moments that hit 80+ hype. Here's how the week felt.</p>
      <div className="rc-curve"><MomentCurve clips={data.moments} className="rc-curve-svg" /></div>
    </>,
    data.best ? (
      <>
        <small>Moment of the week</small>
        <div className="rc-hero"><Poster clip={data.best} className="rl-media" /><span className="disp num">{hypeOf(data.best)}</span></div>
        <b className="rc-t">{data.best.hookLine || data.best.title}</b>
        <p>{data.bestSession?.name} · {data.best.timestamp}</p>
      </>
    ) : null,
    data.hook ? (
      <>
        <small>Your hook game</small>
        <b className="disp big">Hook {coachFor(data.hook).grade}</b>
        <p>“{data.hook.hookLine || data.hook.title}” is your strongest opener this week. Openings like that are your best posts.</p>
        <div className="rc-grades">
          {(Object.keys(data.grades) as HookGrade[]).map((g) => (
            <span key={g}><i style={{ height: `${Math.max(6, (data.grades[g] / maxGrade) * 160)}px` }} /><b>{g}</b><small className="num">{data.grades[g]}</small></span>
          ))}
        </div>
      </>
    ) : null,
    <>
      <small>Keep it rolling</small>
      <b className="disp big num">{data.kept.length} {data.kept.length === 1 ? "keeper" : "keepers"}</b>
      <p>{data.kept.length ? "Your best of the week, ready to post." : "Nothing kept yet this week. Your next deck is waiting."}</p>
      {data.kept.length > 0 && (
        <div className="rc-card">
          {data.kept.slice(0, 4).map((c, k) => <span key={c.id} style={{ ["--i" as string]: k }}><Poster clip={c} className="rl-media" /></span>)}
          <b className="disp">My week on stream</b>
        </div>
      )}
      <button type="button" className="btn heat" onClick={(event) => { event.stopPropagation(); close(); onExport(); }}><ExportIcon aria-hidden="true" />Export my keepers</button>
    </>,
  ][index];

  return (
    <div className="recap" role="dialog" aria-modal="true" aria-label="Your week so far" onClick={() => setIndex((i) => Math.min(slides - 1, i + 1))}>
      <div className="rc-bars" aria-hidden="true">
        {Array.from({ length: empty ? 1 : slides }, (_, k) => <i key={k} className={k < index ? "done" : k === index ? "on" : ""}><b /></i>)}
      </div>
      <button type="button" className="rc-x icon-btn" aria-label="Close" onClick={(event) => { event.stopPropagation(); close(); }}><X aria-hidden="true" /></button>
      <div className="rc-slide" key={index} onClick={(event) => event.stopPropagation()}>{slide ?? <><small>Your week</small><b className="disp big">That's the week.</b></>}</div>
      {!empty && (
        <div className="rc-nav">
          <button type="button" aria-label="Previous" onClick={(event) => { event.stopPropagation(); setIndex((i) => Math.max(0, i - 1)); }}><ChevronLeft aria-hidden="true" /></button>
          <span className="num t3">{index + 1} / {slides}</span>
          <button type="button" aria-label="Next" onClick={(event) => { event.stopPropagation(); setIndex((i) => Math.min(slides - 1, i + 1)); }}><ChevronRight aria-hidden="true" /></button>
        </div>
      )}
    </div>
  );
}

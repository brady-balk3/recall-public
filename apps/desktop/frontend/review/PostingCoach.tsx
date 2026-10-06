// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Theater's left column extras: how hard the room reacted (hype), and a
 * posting coach that grades how fast the clip grabs a scroller.
 *
 * The grade reads the hook score Recall already computes for every clip (the
 * same number behind "Strong opening"), so it never disagrees with the chips.
 * "Post this first" goes to the strongest opener in the deck.
 */
import type { Clip } from "../lib/store";
import { hookScoreOf, rankScoreOf } from "../lib/reviewDeck";

export type HookGrade = "A" | "B" | "C" | "D";

export interface CoachAdvice {
  grade: HookGrade;
  score: number;
  strength: "fast" | "strong" | "steady" | "slow";
  /** Seconds from the clip's first frame to its reaction peak, when known. */
  landsIn?: number;
  /** Seconds to trim off the front so the peak lands early, when that helps. */
  trim?: number;
  advice: string;
}

const GRADES: HookGrade[] = ["A", "B", "C", "D"];
const STRENGTH: Record<HookGrade, CoachAdvice["strength"]> = { A: "fast", B: "strong", C: "steady", D: "slow" };
/** Where a scroller decides: a peak inside this window reads as an instant hook. */
const HOOK_WINDOW_S = 1.5;
const SLOW_LANDING_S = 6;

/**
 * Grade = the hook score Recall computes for every clip, nudged by when the
 * moment actually lands (the reaction peak's offset into the clip): a peak in
 * the first second and a half lifts the grade one step; one after six seconds
 * drops it one. The advice says what to do about it.
 */
export function coachFor(clip: Clip): CoachAdvice {
  const score = Math.round(hookScoreOf(clip));
  let index = score >= 80 ? 0 : score >= 65 ? 1 : score >= 45 ? 2 : 3;
  const landsIn = clip.peakTimestamp != null && clip.start_time != null
    ? Math.max(0, clip.peakTimestamp - clip.start_time)
    : undefined;
  if (landsIn != null) {
    if (landsIn <= HOOK_WINDOW_S) index = Math.max(0, index - 1);
    else if (landsIn > SLOW_LANDING_S) index = Math.min(3, index + 1);
  }
  const grade = GRADES[index];
  const trim = landsIn != null && landsIn > 3 ? Math.round((landsIn - HOOK_WINDOW_S) * 2) / 2 : undefined;
  let advice: string;
  if (trim) advice = `The big moment lands at ${landsIn!.toFixed(1)}s. Trim about ${trim}s off the front so it hits right away.`;
  else if (grade === "A") advice = "Grabs from the first second. Post it as is.";
  else if (grade === "B") advice = "Strong start. Post it as is, or trim a beat off the front.";
  else if (grade === "C") advice = "Takes a moment to land. Try starting closer to the peak.";
  else advice = "A slow build. Trim to the peak before you post it.";
  return { grade, score, strength: STRENGTH[grade], landsIn, trim, advice };
}

/** The clip in a deck that should go up first: the strongest opener, hype breaking ties. */
export function postFirstId(clips: Clip[]): string | undefined {
  let best: Clip | undefined;
  for (const clip of clips) {
    if (!best || hookScoreOf(clip) > hookScoreOf(best) || (hookScoreOf(clip) === hookScoreOf(best) && rankScoreOf(clip) > rankScoreOf(best))) best = clip;
  }
  return best?.id;
}

const SEGMENTS = 20;

export function HypeMeter({ clip }: { clip: Clip }) {
  const hype = Math.round(rankScoreOf(clip));
  const lit = Math.round((hype / 100) * SEGMENTS);
  return (
    <div className="hype-card" role="meter" aria-label="Hype" aria-valuemin={0} aria-valuemax={99} aria-valuenow={hype}>
      <div className="hype-h">
        <span><b>Hype</b> · how hard the room reacted</span>
        <b className="disp num">{hype}</b>
      </div>
      <div className="hype-bar" aria-hidden="true">
        {Array.from({ length: SEGMENTS }, (_, i) => <i key={i} className={i < lit ? "on" : ""} />)}
      </div>
    </div>
  );
}

export function PostingCoach({ clip, postFirst }: { clip: Clip; postFirst: boolean }) {
  const coach = coachFor(clip);
  const r = 17;
  const c = 2 * Math.PI * r;
  return (
    <section className="coach" aria-label="Posting coach">
      <div className="coach-h">
        <span>Posting coach</span>
        {postFirst && <span className="badge heat">Post this first</span>}
      </div>
      <div className="coach-row">
        <span className="coach-ring">
          <svg viewBox="0 0 40 40" width="46" height="46" aria-hidden="true">
            <circle cx="20" cy="20" r={r} className="bg" />
            <circle cx="20" cy="20" r={r} className="fg" strokeDasharray={c} strokeDashoffset={c * (1 - coach.score / 100)} />
          </svg>
          <b className="num">{coach.score}</b>
        </span>
        <span>
          <b className="disp">Hook {coach.grade}</b>
          <small>{coach.landsIn != null ? `Lands in ${coach.landsIn.toFixed(1)}s · ${coach.strength}` : `Opening is ${coach.strength}`}</small>
        </span>
      </div>
      <p>{coach.advice}</p>
    </section>
  );
}

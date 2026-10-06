// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Insights: what Recall has picked up about your streams, drawn only from the
 * library you actually have. The numbers up top, then when you stream, when
 * the room pops, why your keepers were flagged, and the found to kept to
 * exported story with what Recall is learning from it.
 */
import { Poster } from "../components/Posters";
import { clueLabel, learningStateDescription, learningStateLabel } from "../lib/copy";
import { fmtDurationHuman } from "../lib/format";
import { Film, Play, Plus } from "../lib/icons";
import { durationOf, primaryReviewDeckClips } from "../lib/reviewDeck";
import { useJobStore, type Clip, type Session } from "../lib/store";
import { hypeOf } from "../home/homeModel";
import { useMoments } from "../moments/milestones";
import { MomentCurve } from "../ui/MomentCurve";

const DAY = 86_400_000;
const WEEKS = 18;
/** A new library still gets a readable grid, not a single column. */
const MIN_WEEKS = 6;
const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

interface Props {
  sessions: Session[];
  onStart: () => void;
  onSettings: () => void;
}

const keptOf = (session: Session) => session.clips.filter((clip) => session.savedClipIds.includes(clip.id));
const sessionDay = (session: Session) => {
  const date = new Date(session.sourceDate || session.createdAt);
  date.setHours(0, 0, 0, 0);
  return date.getTime();
};

/** Consecutive finished streams, newest first, that each gave at least one keeper. */
function keeperStreak(completed: Session[]) {
  let streak = 0;
  for (const session of [...completed].sort((a, b) => b.createdAt - a.createdAt)) {
    if (!keptOf(session).length) break;
    streak += 1;
  }
  return streak;
}

/** The longest run of finished streams, oldest to newest, that each gave a keeper. */
function bestKeeperStreak(completed: Session[]) {
  let best = 0;
  let run = 0;
  for (const session of [...completed].sort((a, b) => a.createdAt - b.createdAt)) {
    run = keptOf(session).length ? run + 1 : 0;
    best = Math.max(best, run);
  }
  return best;
}

export function InsightsView({ sessions, onStart, onSettings }: Props) {
  const learningStatus = useJobStore((state) => state.learningStatus);
  const openRecap = useMoments((state) => state.openRecap);
  const completed = sessions.filter((session) => session.status === "completed");

  if (!completed.length) {
    return (
      <div className="page insights2">
        <div className="ph"><div><h1 className="disp">Insights</h1><div className="sub">What Recall picks up about your streams, once it has a few to look at.</div></div></div>
        <section className="empty glass">
          <h3 className="disp">Your story starts with one scan</h3>
          <p>Scan a stream and review it. This page fills in with when the room pops, what you keep, and what Recall learns from it.</p>
          <button type="button" className="btn heat" onClick={onStart}><Plus aria-hidden="true" />New stream</button>
        </section>
      </div>
    );
  }

  const deck = completed.flatMap((session) => primaryReviewDeckClips(session.clips).map((clip) => ({ clip, session })));
  const kept = completed.flatMap((session) => keptOf(session).map((clip) => ({ clip, session })));
  const passed = deck.filter(({ clip }) => clip.passed).length;
  const decided = deck.filter(({ clip, session }) => clip.passed || clip.maybe || session.savedClipIds.includes(clip.id)).length;
  const exported = completed.reduce((sum, session) => sum + session.exportedClipIds.length, 0);
  const reviewedSeconds = completed.reduce((sum, session) => sum + (session.vodDuration ?? 0), 0);
  const keepRate = decided ? Math.round((kept.length / decided) * 100) : 0;
  const best = [...kept].sort((a, b) => hypeOf(b.clip) - hypeOf(a.clip))[0];
  const latest = [...completed].sort((a, b) => b.createdAt - a.createdAt)[0];
  const streak = keeperStreak(completed);
  const bestStreak = bestKeeperStreak(completed);

  return (
    <div className="page insights2">
      <div className="ph">
        <div>
          <h1 className="disp">Insights</h1>
          <div className="sub">{fmtDurationHuman(reviewedSeconds)} of stream, {kept.length} {kept.length === 1 ? "keeper" : "keepers"}, {keepRate}% keep rate. What Recall has picked up about your streams.</div>
        </div>
        <div className="actions"><button type="button" className="btn heat" onClick={openRecap}><Play aria-hidden="true" />Your week</button></div>
      </div>

      <div className="bento">
        <div className="bt glass big">
          <small>Keepers</small>
          <b className="disp num">{kept.length}</b>
          <MomentCurve clips={latest.clips} duration={latest.vodDuration} className="bt-curve" />
        </div>
        <div className="bt glass"><small>Reviewed</small><b className="disp num">{fmtDurationHuman(reviewedSeconds)}</b><span className="t3">of stream, {completed.length} {completed.length === 1 ? "session" : "sessions"}</span></div>
        {streak > 0 ? (
          <div className="bt glass"><small>Streak</small><b className="disp num">{streak}</b><span className="t3">{streak === 1 ? "stream" : "streams"} in a row with a keeper{bestStreak > streak ? `, best ${bestStreak}` : ""}</span></div>
        ) : bestStreak > 1 ? (
          <div className="bt glass"><small>Best streak</small><b className="disp num">{bestStreak}</b><span className="t3">streams in a row with a keeper. One keeper starts the next run</span></div>
        ) : (
          <div className="bt glass"><small>Found</small><b className="disp num">{deck.length}</b><span className="t3">{deck.length === 1 ? "moment" : "moments"} worth a look so far</span></div>
        )}
        {best ? (
          <div className="bt glass clipbt">
            <span className="clipbt-fr"><Poster clip={best.clip} className="rl-media" /></span>
            <span className="clipbt-copy"><small>Your loudest keeper</small><b>{best.clip.title}</b><span className="t3">{best.session.name}</span></span>
            <span className="hype-chip num">{hypeOf(best.clip)}</span>
          </div>
        ) : (
          <div className="bt glass"><small>Your loudest keeper</small><b className="disp">None yet</b><span className="t3">Keep a moment and it shows up here</span></div>
        )}
        <div className="bt glass"><small>Exported</small><b className="disp num">{exported}</b><span className="t3">clips ready to post</span></div>
        <div className="bt glass"><small>Keep rate</small><b className="disp num">{keepRate}%</b><span className="t3">of {decided} {decided === 1 ? "call" : "calls"} you made</span></div>
      </div>

      <div className="ins-grid">
        <StreamCalendar sessions={completed} />
        <WhenItPops deck={deck.map(({ clip }) => clip)} />
        <WhyFlagged kept={kept.map(({ clip }) => clip)} />
      </div>

      <div className="sec-t"><h2 className="disp">Your story</h2><span className="c">Found, kept, and exported, and what Recall is learning</span></div>
      <div className="story">
        <section className="panel glass">
          <div className="funnel">
            {([
              ["Footage reviewed", fmtDurationHuman(reviewedSeconds), 1, true],
              ["Moments found", String(deck.length), 1, false],
              ["Kept", String(kept.length), kept.length / Math.max(1, deck.length), false],
              ["Exported", String(exported), exported / Math.max(1, deck.length), false],
            ] as const).map(([label, value, share, neutral]) => (
              <div className="fstep" key={label}>
                <span className="t2">{label}</span>
                <div className="fb"><i className={neutral ? "n" : ""} style={{ width: `${Math.max(3, Math.min(100, share * 100))}%` }} /></div>
                <b className="disp num">{value}</b>
              </div>
            ))}
          </div>
          <p className="t3 story-foot">
            Kept clips add up to {fmtDurationHuman(kept.reduce((sum, { clip }) => sum + durationOf(clip), 0))} of highlights.
          </p>
        </section>
        <section className="panel glass learning">
          <span className="eyebrow">What Recall is learning</span>
          <div className="disp learning-h">{completed.length < 2 ? "Early days" : learningStateLabel(learningStatus?.state)}</div>
          <p className="t2">{completed.length < 2 ? "Keep and pass on a couple of streams and Recall starts sorting future decks the way you would." : learningStateDescription(learningStatus?.state)}</p>
          <p className="t3 learning-basis">
            Based on {kept.length} {kept.length === 1 ? "keep" : "keeps"} and {passed} {passed === 1 ? "pass" : "passes"} across {completed.length} {completed.length === 1 ? "session" : "sessions"}.
          </p>
          <div><button type="button" className="btn sm" onClick={onSettings}>Manage learning</button></div>
        </section>
      </div>
    </div>
  );
}

function StreamCalendar({ sessions }: { sessions: Session[] }) {
  const end = new Date();
  end.setHours(0, 0, 0, 0);
  // Only as many weeks as there is history for, so a new library isn't
  // mostly empty squares.
  const first = Math.min(...sessions.map(sessionDay));
  const weeksOfHistory = Math.ceil((end.getTime() - first) / (7 * DAY)) + 1;
  const weeks = Math.max(MIN_WEEKS, Math.min(WEEKS, weeksOfHistory));
  const start = new Date(end);
  start.setDate(start.getDate() - (weeks * 7 - 1) - start.getDay());
  const byDay = new Map<number, number>();
  const keptDays = new Set<number>();
  const weekdays = new Array<number>(7).fill(0);
  for (const session of sessions) {
    const day = sessionDay(session);
    if (day < start.getTime()) continue;
    byDay.set(day, (byDay.get(day) ?? 0) + keptOf(session).length + 1);
    if (keptOf(session).length) keptDays.add(day);
    weekdays[new Date(day).getDay()] += 1;
  }
  const max = Math.max(1, ...byDay.values());
  const cells: Array<{ day: number; value: number }> = [];
  for (let day = start.getTime(); day <= end.getTime(); day += DAY) cells.push({ day, value: byDay.get(day) ?? 0 });
  const streamDays = cells.filter((cell) => cell.value).length;
  const busiest = weekdays.indexOf(Math.max(...weekdays));
  const span = weeksOfHistory < WEEKS ? `since you started, ${weeks} weeks` : `last ${WEEKS} weeks`;
  return (
    <section className="lab-card glass wide2" aria-label="Streams and keepers by day">
      <div className="lab-h"><b>Streams and keepers</b><span className="t3">{span}</span></div>
      <div className="cal-body">
        <div className="cal-wrap">
          <div className="cal-days t3" aria-hidden="true"><span /><span>Mon</span><span /><span>Wed</span><span /><span>Fri</span><span /></div>
          <div className="cal">
            {cells.map(({ day, value }) => (
              <i
                key={day}
                style={value ? { background: `color-mix(in oklab, var(--h2) ${20 + (value / max) * 80}%, transparent)` } : undefined}
                title={`${new Date(day).toDateString()}${value ? " · streamed" : ""}`}
              />
            ))}
          </div>
        </div>
        <dl className="cal-sum">
          <div><dt>Stream days</dt><dd className="disp num">{streamDays}</dd></div>
          <div><dt>With a keeper</dt><dd className="disp num">{keptDays.size}</dd></div>
          {streamDays >= 3 && <div><dt>You stream most on</dt><dd className="disp">{WEEKDAYS[busiest]}s</dd></div>}
        </dl>
      </div>
      <div className="cal-leg"><span className="t3">Less</span>{[0, 25, 50, 75, 100].map((p) => <i key={p} style={{ background: p ? `color-mix(in oklab, var(--h2) ${20 + p * 0.8}%, transparent)` : "var(--fill)" }} />)}<span className="t3">More</span></div>
    </section>
  );
}

function WhenItPops({ deck }: { deck: Clip[] }) {
  const bins = Array<number>(12).fill(0);
  for (const clip of deck) {
    if (clip.start_time == null) continue;
    bins[Math.min(11, Math.floor(clip.start_time / 1800))] += 1;
  }
  const max = Math.max(1, ...bins);
  const peak = bins.indexOf(max);
  // Bins are half hours: 0 -> "0m", 1 -> "30m", 3 -> "1h 30m".
  const at = (bin: number) => {
    const minutes = bin * 30;
    const h = Math.floor(minutes / 60);
    const m = minutes % 60;
    return [h ? `${h}h` : "", m || !h ? `${m}m` : ""].filter(Boolean).join(" ");
  };
  const when = peak === 0 ? <>in the first <b>30m</b></> : <><b>{at(peak)} to {at(peak + 1)}</b> in</>;
  return (
    <section className="lab-card glass" aria-label="Moments by stream hour">
      <div className="lab-h"><b>When the room pops</b><span className="t3">moments by stream hour</span></div>
      <div className="bars">
        {bins.map((value, index) => (
          <span key={index} className={index === peak && value ? "pk" : ""}>
            <i style={{ height: `${Math.max(4, (value / max) * 100)}%` }} title={`${value} ${value === 1 ? "moment" : "moments"}`} />
            <small>{index % 2 ? "" : `${index / 2}h`}</small>
          </span>
        ))}
      </div>
      {deck.length > 0 && <p className="lab-note">Your moments cluster {when}. Recall Live marks there are worth the most.</p>}
    </section>
  );
}

const DONUT = ["var(--t1)", "color-mix(in oklab, var(--t1) 72%, transparent)", "color-mix(in oklab, var(--t1) 48%, transparent)", "color-mix(in oklab, var(--t1) 30%, transparent)", "color-mix(in oklab, var(--t1) 16%, transparent)"];

function WhyFlagged({ kept }: { kept: Clip[] }) {
  const counts = new Map<string, number>();
  for (const clip of kept) for (const signal of clip.signals ?? []) {
    const label = clueLabel(signal);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  const total = top.reduce((sum, [, n]) => sum + n, 0);
  const R = 42;
  const C = 2 * Math.PI * R;
  let offset = 0;
  return (
    <section className="lab-card glass" aria-label="Why your keepers were flagged">
      <div className="lab-h"><b>Why your keepers were flagged</b></div>
      {top.length ? (
        <div className="donut">
          <svg viewBox="0 0 120 120" aria-hidden="true">
            {top.map(([label, n], index) => {
              const length = (n / total) * C;
              const arc = <circle key={label} r={R} cx="60" cy="60" fill="none" stroke={DONUT[index]} strokeWidth="16" strokeDasharray={`${Math.max(0, length - 2)} ${C}`} strokeDashoffset={-offset} transform="rotate(-90 60 60)" />;
              offset += length;
              return arc;
            })}
            <text x="60" y="58" textAnchor="middle" className="d-n">{kept.length}</text>
            <text x="60" y="74" textAnchor="middle" className="d-l">keepers</text>
          </svg>
          <div className="d-leg">
            {top.map(([label, n], index) => <span key={label}><i style={{ background: DONUT[index] }} />{label}<b className="num">{Math.round((n / total) * 100)}%</b></span>)}
          </div>
        </div>
      ) : (
        <p className="lab-note"><Film aria-hidden="true" /> Keep a few moments and the reasons show up here.</p>
      )}
    </section>
  );
}

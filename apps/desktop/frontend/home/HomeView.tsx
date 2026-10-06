// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Home: what to do next, front and center.
 *
 * A compact greeting and the scan bar, then the loudest moment waiting for a
 * decision (fanned over a blurred frame of it), a live column (the running
 * scan, Recall Live, the week), everything waiting, the keepers reel, recent
 * sessions, and a getting-started checklist that disappears once it's done.
 */
import { useState, type CSSProperties } from "react";
import { Poster, pickSessionPosterClip } from "../components/Posters";
import { fmtClock } from "../lib/format";
import {
  BarChart3,
  Broadcast,
  Check,
  ChevronRight,
  Film,
  Folder,
  Play,
  Pulse,
  Queue,
  X,
} from "../lib/icons";
import type { Clip, Session } from "../lib/store";
import type { RecallLiveState } from "../lib/useRecallLive";
import { KeepersReel } from "../ui/KeepersReel";
import { MomentCurve } from "../ui/MomentCurve";
import { SessionCard, SessionStatusBadge } from "../ui/SessionCard";
import { ScanBar } from "./ScanBar";
import { useMoments } from "../moments/milestones";
import type { NewStreamKind, View } from "../shell/views";
import {
  checklist,
  greeting,
  hypeOf,
  keeperMilestone,
  ordinal,
  recentSessions,
  undecidedClips,
  upNext,
  waitingSessions,
  weekStats,
  whenLine,
  type ChecklistStep,
  type UpNext,
} from "./homeModel";

export interface HomeScan {
  title: string;
  /** 0-100 */
  progress: number;
  stage?: string;
  etaSeconds?: number | null;
  found: Clip[];
}

export interface HomeViewProps {
  displayName: string;
  onRename: (next: string) => void;
  sessions: Session[];
  scan?: HomeScan;
  queued: number;
  live: RecallLiveState;
  accentChosen: boolean;
  onNew: (kind: NewStreamKind) => void;
  onScanLink: (url: string) => void;
  onOpenSession: (session: Session) => void;
  onWatch: (session: Session, clip: Clip) => void;
  onView: (view: View) => void;
  onChecklist: (step: ChecklistStep) => void;
}

const CHECKLIST_DISMISSED_KEY = "recall-home-checklist-dismissed";

export function HomeView(props: HomeViewProps) {
  const { displayName, onRename, sessions, scan, queued, live, accentChosen, onNew, onScanLink, onOpenSession, onWatch, onView, onChecklist } = props;
  const waiting = waitingSessions(sessions);
  const next = upNext(sessions);
  const recent = recentSessions(sessions, waiting);
  const keepers = sessions
    .flatMap((session) => session.clips.filter((clip) => session.savedClipIds.includes(clip.id)).map((clip) => ({ clip, session })))
    .sort((a, b) => hypeOf(b.clip) - hypeOf(a.clip));
  const moments = waiting.reduce((sum, session) => sum + undecidedClips(session).length, 0);

  return (
    <div className="page home">
      <header className="dh-head">
        <div className="dh-greet">
          <div className="w-when">{whenLine(waiting.length)}</div>
          <h1 className="disp">
            {greeting()}, <EditableName value={displayName} onCommit={onRename} /><span className="dot-h">.</span>
          </h1>
        </div>
        <div className="dh-scan">
          <ScanBar onScanLink={onScanLink} onChooseFile={() => onNew("local")} />
        </div>
      </header>

      <div className="dh-grid">
        {next ? (
          <UpNextHero next={next} onWatch={onWatch} onOpenSession={onOpenSession} />
        ) : (
          <EmptyHero hasSessions={sessions.some((s) => s.status === "completed")} sessions={sessions} onNew={onNew} onView={onView} />
        )}
        <aside className="dh-side">
          <NowScan scan={scan} queued={queued} onNew={onNew} onView={onView} />
          <LiveCard live={live} onOpen={() => onView("live")} />
          <WeekCard sessions={sessions} keepers={keepers.length} />
        </aside>
      </div>

      {waiting.length > 0 && (
        <>
          <div className="sec-t">
            <h2 className="disp">Waiting for you</h2>
            <span className="c">{plural(waiting.length, "stream")} · {plural(moments, "moment")}</span>
            <span className="sp" />
            <button type="button" className="btn sm" onClick={() => onView("review")}>Open Review <ChevronRight aria-hidden="true" /></button>
          </div>
          <div className="sessions dh-wait">
            {waiting.map((session) => (
              <SessionCard key={session.id} session={session} onOpen={onOpenSession} badge={<span className="badge heat">{undecidedClips(session).length} to review</span>} />
            ))}
          </div>
        </>
      )}

      {keepers.length > 0 && (
        <>
          <div className="sec-t">
            <h2 className="disp">Your keepers</h2>
            <span className="c">{plural(keepers.length, "clip")} you said yes to · drag to browse</span>
            <span className="sp" />
            <button type="button" className="btn sm" onClick={() => onView("clips")}>Export <ChevronRight aria-hidden="true" /></button>
          </div>
          <KeepersReel
            label="Your keepers"
            items={keepers.slice(0, 30).map(({ clip, session }) => ({
              key: clip.id,
              label: `${clip.hookLine || clip.title}, hype ${hypeOf(clip)}`,
              onOpen: () => onWatch(session, clip),
              content: (
                <>
                  <Poster clip={clip} className="rl-media" />
                  <span className="rl-hype num">{hypeOf(clip)}</span>
                  <span className="rl-b">
                    <b>{clip.hookLine || clip.title}</b>
                    <small className="num">{fmtClock(clip.duration)}</small>
                  </span>
                </>
              ),
            }))}
          />
        </>
      )}

      {recent.length > 0 && (
        <>
          <div className="sec-t">
            <h2 className="disp">Recent sessions</h2>
            <span className="c">{sessions.length} in your library</span>
            <span className="sp" />
            <button type="button" className="btn sm" onClick={() => onView("sessions")}>Open library <ChevronRight aria-hidden="true" /></button>
          </div>
          <div className="sessions dh-sess">
            {recent.map((session) => <SessionCard key={session.id} session={session} onOpen={onOpenSession} badge={<SessionStatusBadge session={session} scanProgress={scan && scan.title === session.name ? scan.progress : undefined} />} />)}
          </div>
        </>
      )}

      <Checklist sessions={sessions} accentChosen={accentChosen} onStep={onChecklist} />

      <div className="home-foot">
        <span className="t3">Want the long view?</span>
        <button type="button" className="btn sm ghost" onClick={() => onView("insights")}><BarChart3 aria-hidden="true" />Insights</button>
        <button type="button" className="btn sm ghost" onClick={() => onView("reel")}><Film aria-hidden="true" />Highlight reel</button>
      </div>
    </div>
  );
}

function plural(n: number, word: string) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/* ── greeting ── */

function EditableName({ value, onCommit }: { value: string; onCommit: (next: string) => void }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  if (!editing) {
    return (
      <button type="button" className="name" title="Change your name" aria-label={`Your name: ${value}. Change it.`} onClick={() => { setDraft(value === "Creator" ? "" : value); setEditing(true); }}>
        {value}
      </button>
    );
  }
  const commit = () => { setEditing(false); onCommit(draft.trim()); };
  return (
    <input
      className="name-input"
      autoFocus
      value={draft}
      maxLength={32}
      placeholder="Creator"
      aria-label="Your name"
      style={{ width: `${Math.max(4, draft.length || 7) + 0.5}ch` }}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") { event.preventDefault(); commit(); }
        if (event.key === "Escape") { event.preventDefault(); setEditing(false); }
      }}
    />
  );
}

/* ── hero ── */

function UpNextHero({ next, onWatch, onOpenSession }: { next: UpNext; onWatch: HomeViewProps["onWatch"]; onOpenSession: HomeViewProps["onOpenSession"] }) {
  const { session, lead, fan, inDeck, waitingOverall, minutesToClear } = next;
  const heroStyle = lead.thumbUrl ? ({ "--hero": `url("${lead.thumbUrl}")` } as CSSProperties) : undefined;
  const headline = lead.hookLine || lead.title;
  return (
    <section className="dh-hero" style={heroStyle} aria-labelledby="up-next-title">
      <div className="dh-fan">
        <div className={`fan n${fan.length}`}>
          {fan.map((clip) => {
            const front = clip.id === lead.id;
            // Rank in the deck: the lead is #01, the others follow in deck order.
            const rank = front ? 1 : fan.filter((c) => c.id !== lead.id).indexOf(clip) + 2;
            return (
              <button
                key={clip.id}
                type="button"
                className={`fcard ${front ? "front" : ""}`}
                onClick={() => onWatch(session, clip)}
                aria-label={`Watch ${clip.hookLine || clip.title}`}
                tabIndex={front ? 0 : -1}
              >
                <Poster clip={clip} className="fc-media" eager />
                <span className="fc-top">
                  <b className="rk disp num">#{String(rank).padStart(2, "0")}</b>
                  <span className="hype num">{hypeOf(clip)}</span>
                </span>
                <span className="fc-b">
                  <b>{clip.hookLine || clip.title}</b>
                  <small className="num">{fmtClock(clip.duration)}</small>
                </span>
              </button>
            );
          })}
        </div>
      </div>
      <div className="dh-copy">
        <span className="dh-kick"><span className="live-dot" aria-hidden="true" />Up next · {session.name}</span>
        <h2 className="disp" id="up-next-title">“{headline}”</h2>
        <p>
          It peaks at <b className="num">{lead.timestamp}</b>. Hype <b className="num">{hypeOf(lead)}</b>, the loudest moment in this stream. Start there.
        </p>
        <div className="dh-stats">
          <div><b className="disp num">{inDeck}</b><small>in this deck</small></div>
          <div><b className="disp num">{waitingOverall}</b><small>waiting overall</small></div>
          <div><b className="disp num">~{minutesToClear}m</b><small>to clear it all</small></div>
        </div>
        <div className="w-cta">
          <button type="button" className="btn heat lg" onClick={() => onWatch(session, lead)}><Play aria-hidden="true" />Watch it</button>
          <button type="button" className="btn lg" onClick={() => onOpenSession(session)}>Review all {inDeck}</button>
        </div>
      </div>
    </section>
  );
}

function EmptyHero({ hasSessions, sessions, onNew, onView }: { hasSessions: boolean; sessions: Session[]; onNew: HomeViewProps["onNew"]; onView: HomeViewProps["onView"] }) {
  const backdrop = sessions.map(pickSessionPosterClip).find((clip) => clip?.thumbUrl)?.thumbUrl;
  const heroStyle = backdrop ? ({ "--hero": `url("${backdrop}")` } as CSSProperties) : undefined;
  return (
    <section className="dh-hero is-empty" style={heroStyle}>
      <div className="dh-copy">
        {hasSessions ? (
          <>
            <span className="dh-kick">All caught up</span>
            <h2 className="disp">Every moment has a keep or a pass.</h2>
            <p>Your keepers are ready to post. Export them, or scan the next stream and Recall goes looking again.</p>
            <div className="w-cta">
              <button type="button" className="btn heat lg" onClick={() => onView("clips")}>Export keepers</button>
              <button type="button" className="btn lg" onClick={() => onNew("local")}>New stream</button>
            </div>
          </>
        ) : (
          <>
            <span className="dh-kick">Let's find your moments</span>
            <h2 className="disp">Your best clips are in there somewhere.</h2>
            <p>Paste a Twitch VOD link above or pick a recording. Recall watches the whole stream and brings back the moments worth posting.</p>
            <div className="w-cta">
              <button type="button" className="btn heat lg" onClick={() => onNew("local")}><Folder aria-hidden="true" />Choose a recording</button>
              <button type="button" className="btn lg" onClick={() => onNew("batch")}><Queue aria-hidden="true" />Queue tonight's VODs</button>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

/* ── side column ── */

function readyBy(etaSeconds?: number | null) {
  if (!etaSeconds || etaSeconds <= 0) return null;
  return new Date(Date.now() + etaSeconds * 1000).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function NowScan({ scan, queued, onNew, onView }: { scan?: HomeScan; queued: number; onNew: HomeViewProps["onNew"]; onView: HomeViewProps["onView"] }) {
  if (!scan) {
    return (
      <div className="now-scan glass idle">
        <span className="lab">Scans</span>
        <h3 className="disp">{queued ? `${queued} waiting to scan` : "Nothing scanning"}</h3>
        <p className="t3">{queued ? "They start one after another. Leave Recall open and they'll be ready to review." : "Paste a link above, or line up a few VODs for tonight."}</p>
        <button type="button" className="btn sm" onClick={() => (queued ? onView("queue") : onNew("batch"))}>
          {queued ? <><Pulse aria-hidden="true" />Open Scans</> : <><Queue aria-hidden="true" />Queue VODs</>}
        </button>
      </div>
    );
  }
  const ready = readyBy(scan.etaSeconds);
  const minutesLeft = scan.etaSeconds ? Math.max(1, Math.round(scan.etaSeconds / 60)) : null;
  return (
    <button type="button" className="now-scan glass" onClick={() => onView("queue")}>
      <span className="ns-top">
        <span className="lab"><span className="live-dot" aria-hidden="true" />Scanning</span>
        <span className="sp" />
        {ready && <span className="t3 ns-ready">ready by <b className="num">{ready}</b></span>}
      </span>
      <h3 className="disp">{scan.title}</h3>
      <span className="ns-curve" aria-hidden="true">
        <MomentCurve clips={scan.found} className="ns-curve-svg" />
        <span className="head" style={{ left: `${scan.progress}%` }} />
      </span>
      <span className="ns-row">
        <span className="t3">{scan.stage || "Finding exciting moments"}</span>
        <span className="num t3">{Math.floor(scan.progress)}%{minutesLeft ? ` · ${minutesLeft} min left` : ""}</span>
      </span>
      {scan.found.length > 0 && (
        <span className="ns-found">
          <span className="t3">Just found</span>
          {scan.found.slice(-4).map((clip) => <span key={clip.id} className="nf"><Poster clip={clip} /></span>)}
          <b className="num">{scan.found.length}</b>
        </span>
      )}
    </button>
  );
}

function LiveCard({ live, onOpen }: { live: RecallLiveState; onOpen: () => void }) {
  return (
    <button type="button" className={`side-card glass ${live.active ? "on-air" : ""}`} onClick={onOpen}>
      <span className={`ico ${live.active ? "on" : ""}`}><Broadcast aria-hidden="true" /></span>
      <span>
        <b>{live.active ? "Recall Live is listening" : "Going live later?"}</b>
        <small>
          {live.active
            ? live.marks ? `${plural(live.marks, "moment")} marked so far` : "Nothing marked yet"
            : live.hotkey ? <>Hit <kbd>{live.hotkey}</kbd> when something pops off</> : "Mark moments while you stream"}
        </small>
      </span>
      <ChevronRight className="chev" aria-hidden="true" />
    </button>
  );
}

function WeekCard({ sessions, keepers }: { sessions: Session[]; keepers: number }) {
  const openRecap = useMoments((state) => state.openRecap);
  const week = weekStats(sessions);
  const goal = keeperMilestone(keepers);
  return (
    <section className="week glass" aria-labelledby="week-title">
      <div className="wk-h">
        <b id="week-title">Your week</b>
        <button type="button" className="btn sm" onClick={openRecap}><Play aria-hidden="true" />Watch recap</button>
      </div>
      <div className="wk-stats">
        <div><b className="disp num">{week.streams}</b><small>{week.streams === 1 ? "stream" : "streams"}</small></div>
        <div><b className="disp num">{week.keepers}</b><small>{week.keepers === 1 ? "keeper" : "keepers"}</small></div>
        <div><b className="disp num">{week.topHype || "–"}</b><small>top hype</small></div>
      </div>
      <div className="wk-goal">
        <span className="wk-goal-row"><span>{goal.remaining} to your {ordinal(goal.target)} keeper</span><span className="num t3">{keepers}/{goal.target}</span></span>
        <span className="wk-bar" role="progressbar" aria-valuemin={0} aria-valuemax={goal.target} aria-valuenow={keepers} aria-label="Keepers toward the next milestone">
          <i style={{ transform: `scaleX(${goal.progress})` }} />
        </span>
      </div>
    </section>
  );
}

/* ── getting started ── */

function readDismissed() {
  try { return localStorage.getItem(CHECKLIST_DISMISSED_KEY) === "1"; } catch { return false; }
}

function Checklist({ sessions, accentChosen, onStep }: { sessions: Session[]; accentChosen: boolean; onStep: (step: ChecklistStep) => void }) {
  const [dismissed, setDismissed] = useState(readDismissed);
  const items = checklist(sessions, accentChosen);
  const done = items.filter((item) => item.done).length;
  if (dismissed || done === items.length) return null;
  const dismiss = () => {
    setDismissed(true);
    try { localStorage.setItem(CHECKLIST_DISMISSED_KEY, "1"); } catch {}
  };
  const r = 15;
  const c = 2 * Math.PI * r;
  return (
    <section className="start glass" aria-labelledby="start-title">
      <div className="st-hd">
        <div>
          <b id="start-title">Get more out of Recall</b>
          <small>{done} of {items.length} done. The rest are one click each.</small>
        </div>
        <svg className="st-ring" viewBox="0 0 36 36" width="36" height="36" aria-hidden="true">
          <circle cx="18" cy="18" r={r} className="bg" />
          <circle cx="18" cy="18" r={r} className="fg" strokeDasharray={c} strokeDashoffset={c * (1 - done / items.length)} />
        </svg>
        <button type="button" className="icon-btn" aria-label="Hide these tips" onClick={dismiss}><X aria-hidden="true" /></button>
      </div>
      <div className="st-steps">
        {items.map((item) => (
          <button key={item.id} type="button" className={`stp ${item.done ? "ok" : ""}`} disabled={item.done} onClick={() => onStep(item.id)}>
            <span className="stp-c">{item.done && <Check aria-hidden="true" />}</span>
            <span>{item.label}</span>
            {!item.done && <ChevronRight aria-hidden="true" />}
          </button>
        ))}
      </div>
    </section>
  );
}

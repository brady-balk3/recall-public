// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Reels: one stream's kept moments in VOD order with hard cuts, as a single
 * vertical video. Every kept clip starts in; dropping one only affects this
 * reel and never un-keeps it. New keeps join automatically.
 */
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { Poster } from "../components/Posters";
import { plural } from "../lib/copy";
import { fmtClock } from "../lib/format";
import { Check, Film, Play, Plus, X } from "../lib/icons";
import { durationOf } from "../lib/reviewDeck";
import type { Clip, Session } from "../lib/store";
import type { View } from "../shell/views";
import { ExportTabs } from "./ExportTabs";

export interface ReelViewProps {
  sessions: Session[];
  session?: Session;
  clips: Clip[];
  compiling: boolean;
  clipsCount: number;
  compilationsCount?: number;
  onCompile: (clipIds: string[]) => void;
  onSelectSession: (sessionId: string) => void;
  onOpenReview: () => void;
  onView: (view: View) => void;
}

export function ReelView({ sessions, session, clips, compiling, clipsCount, compilationsCount, onCompile, onSelectSession, onOpenReview, onView }: ReelViewProps) {
  const [dropped, setDropped] = useState<Set<string>>(new Set());
  useEffect(() => { setDropped(new Set()); }, [session?.id]);

  const candidates = useMemo(
    () => sessions.filter((item) => item.status === "completed").sort((a, b) => (b.createdAt || b.updatedAt) - (a.createdAt || a.updatedAt)),
    [sessions],
  );
  // No session picked yet: start on the newest one that can make a reel.
  useEffect(() => {
    if (session) return;
    const first = candidates.find((item) => item.savedClipIds.length >= 2);
    if (first) onSelectSession(first.id);
  }, [session, candidates, onSelectSession]);

  const inReel = clips.filter((clip) => !dropped.has(clip.id));
  const total = inReel.reduce((sum, clip) => sum + durationOf(clip), 0);
  const canCompile = inReel.length >= 2;
  const toggle = (id: string) => setDropped((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  return (
    <div className="page export reel2">
      <ExportTabs active="reel" clipsCount={clipsCount} compilationsCount={compilationsCount} onView={onView} />

      {!candidates.length ? (
        <section className="empty glass">
          <h3 className="disp">No reels yet</h3>
          <p>Finish a scan and keep at least two moments. Its reel builds itself here.</p>
          <button type="button" className="btn" onClick={() => onView("review")}>Open Review</button>
        </section>
      ) : (
        <div className="reel-grid">
          <section className="panel glass reel-main" aria-label="Highlight reel">
            <div className="panel-t reel-head">
              <div>
                <h2 className="disp">Highlight reel</h2>
                <div className="t3">One stream, its kept moments in VOD order, hard cuts.</div>
              </div>
              <span className="sp" />
              <select
                className="field reel-pick"
                value={session?.id ?? ""}
                onChange={(event) => onSelectSession(event.target.value)}
                aria-label="Stream for the reel"
              >
                {!session && <option value="">Pick a stream</option>}
                {candidates.map((item) => {
                  const kept = item.savedClipIds.length;
                  return (
                    <option key={item.id} value={item.id} disabled={kept < 2}>
                      {item.name}, {kept < 2 ? "needs more moments" : plural(kept, "kept clip")}
                    </option>
                  );
                })}
              </select>
            </div>

            {!session || clips.length < 2 ? (
              <div className="empty reel-empty">
                <h3 className="disp">Keep a few more</h3>
                <p>{session ? `${session.name} has ${plural(clips.length, "kept clip")}. A reel needs at least two.` : "Pick a stream with two or more kept moments."}</p>
                {session && <button type="button" className="btn" onClick={onOpenReview}><Play aria-hidden="true" />Open review</button>}
              </div>
            ) : (
              <>
                <ol className="seq" aria-label="Reel clips in VOD order">
                  {clips.map((clip, index) => {
                    const out = dropped.has(clip.id);
                    const position = inReel.indexOf(clip) + 1;
                    return (
                      <li key={clip.id} className={`seq-clip ${out ? "out" : ""}`}>
                        <span className="fr">
                          <Poster clip={clip} className="rl-media" eager={index < 6} />
                          {!out && <span className="rank disp">{position}</span>}
                          <button
                            type="button"
                            className="seq-toggle"
                            role="checkbox"
                            aria-checked={!out}
                            aria-label={`${out ? "Add" : "Remove"} ${clip.title} ${out ? "to" : "from"} the reel`}
                            title={out ? "Add back to this reel" : "Remove from this reel"}
                            onClick={() => toggle(clip.id)}
                          >
                            {out ? <Plus aria-hidden="true" /> : <X aria-hidden="true" />}
                          </button>
                        </span>
                        <h3>{clip.title}</h3>
                        <span className="m num">{clip.timestamp} · {fmtClock(durationOf(clip))}</span>
                      </li>
                    );
                  })}
                </ol>
                <div className="scale" aria-hidden="true">
                  {inReel.map((clip, index) => (
                    <i key={clip.id} style={{ "--w": Math.max(1, durationOf(clip)) } as CSSProperties}><span>{index + 1}</span></i>
                  ))}
                </div>
                <div className="row t3 num scale-ax"><span>0:00</span><span className="sp" /><span>{fmtClock(total)}</span></div>
                {dropped.size > 0 && (
                  <button type="button" className="btn sm ghost reel-reset" onClick={() => setDropped(new Set())}>Add all {clips.length} back</button>
                )}
              </>
            )}
          </section>

          <aside className="reel-aside">
            <div className="reel-preview">
              {inReel[0] ? <Poster clip={inReel[0]} className="rl-media" /> : <Film aria-hidden="true" />}
              {inReel.length > 0 && <span className="reel-preview-meta num"><span>1 / {inReel.length}</span><span>{fmtClock(total)}</span></span>}
            </div>
            <div className="panel glass reel-facts">
              <div className="row"><span className="t3">Moments</span><span className="sp" /><b className="num">{inReel.length}</b></div>
              <div className="row"><span className="t3">Runtime</span><span className="sp" /><b className="num">{fmtClock(total)}</b></div>
              <div className="row"><span className="t3">Transitions</span><span className="sp" /><b>Hard cuts</b></div>
              <div className="row"><span className="t3">Format</span><span className="sp" /><b>1080×1920</b></div>
            </div>
            <button type="button" className="btn heat lg reel-go" disabled={!canCompile || compiling} onClick={() => onCompile(inReel.map((clip) => clip.id))}>
              {compiling ? "Compiling reel…" : <><Check aria-hidden="true" />{canCompile ? `Compile ${inReel.length}-clip reel` : "Needs two clips"}</>}
            </button>
            <small className="t3 reel-note">You'll choose a folder when you compile.</small>
          </aside>
        </div>
      )}
    </div>
  );
}

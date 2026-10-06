// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useCallback, useEffect, useMemo, useState } from "react";
import { Broadcast, Check, Film, FolderOpen, Play, Target, Video, X } from "../lib/icons";
import RememberHotkeySetting, { displayHotkey } from "./RememberHotkeySetting";
import { fmtClock } from "../lib/format";
import { plural } from "../lib/copy";
import {
  buildRecallRegionPlan,
  deleteRecallEvent,
  getActiveRecallSession,
  getRecallSession,
  listRecallSessions,
  rememberMoment,
  startRecallSession,
  stopRecallSession,
  type RecallRegionPlan,
  type RecallSession,
  type RecallSessionEvent,
} from "../lib/recallSessions";

type RememberResult = {
  ok?: boolean;
  payload?: unknown;
};

type ScanSourceType = "file" | "twitch";
/** What the creator is marking. The engine calls these `twitch` and
 * `local_replay`; those are storage words, not something anyone is doing. */
type MarkSource = "stream" | "recording";

/** Just enough of a scan session to say whether a Recall Live session was ever
 * scanned, without dragging the whole store type into this component. */
type ScannedSession = {
  id: string;
  status: string;
  recallSessionId?: string;
  clips: { id: string }[];
};

interface LiveSessionPanelProps {
  onScanRecording: (
    path: string,
    sessionId: string,
    sourceType: ScanSourceType,
  ) => Promise<void> | void;
  /** Scan sessions, used only to mark which earlier sessions have been scanned. */
  sessions?: ScannedSession[];
}

const RECORDING_HINT = "Play a recording from 0:00, then start Recall at the same moment.";
const STREAM_HINT = "Start this any time during a live stream, then attach the VOD when it publishes.";
const EARLIER_SESSION_LIMIT = 6;

function recordingName(path: string): string {
  const leaf = path.replace(/\\/g, "/").split("/").pop() || "Recall session";
  return leaf.replace(/\.[^.]+$/, "") || leaf;
}

function startedAt(session: RecallSession | null): number {
  return session ? new Date(session.started_at_utc).getTime() : 0;
}

function elapsedSeconds(session: RecallSession | null, now: number): number {
  if (!session) return 0;
  const end = session.ended_at_utc ? new Date(session.ended_at_utc).getTime() : now;
  return Math.max(0, (end - startedAt(session)) / 1000);
}

function elapsedLabel(session: RecallSession | null, now: number): string {
  const seconds = Math.floor(elapsedSeconds(session, now));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(rest)}`;
}

function marksOf(session: RecallSession | null): RecallSessionEvent[] {
  return (session?.events ?? []).filter((event) => event.kind === "remember");
}

/** Wall-clock time of day, which is how a creator remembers a moment
 * ("that was just after nine") when the session clock means nothing to them. */
function clockOfDay(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "--:--";
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function shortDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const today = new Date();
  const sameDay = date.toDateString() === today.toDateString();
  const yesterday = new Date(today.getTime() - 86_400_000).toDateString() === date.toDateString();
  const time = clockOfDay(iso);
  if (sameDay) return `Today ${time}`;
  if (yesterday) return `Yesterday ${time}`;
  return `${date.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })} ${time}`;
}

/** The tape: one tick per mark, placed by real elapsed offset.
 *
 * Deliberately NOT the R(t) reaction curve. No video has been read yet, so
 * there is no reaction signal to draw and inventing one would be a lie about
 * what Recall knows. What IS true during a broadcast is when the creator
 * reacted, and a comb over the session's own span shows exactly that —
 * including how the marks cluster, which a count cannot say. */
function MarkTape({
  marks, session, now, live, unmappedIds,
}: {
  marks: RecallSessionEvent[];
  session: RecallSession | null;
  now: number;
  live: boolean;
  unmappedIds: Set<string>;
}) {
  const span = Math.max(1, elapsedSeconds(session, now));
  const start = startedAt(session);
  const ruler = useMemo(() => {
    // Round steps so labels land on readable times whatever the span is, and
    // never more than four of them: the step ladder has to be derived from the
    // span rather than picked from a fixed list, or a long span runs off the
    // end of the list and emits a gridline every few pixels.
    const ladder = [60, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400];
    const step = ladder.find((value) => span / value <= 4)
      ?? Math.ceil(span / 4 / 86400) * 86400;
    const stops: number[] = [];
    for (let at = step; at < span * 0.94 && stops.length < 4; at += step) stops.push(at);
    return stops;
  }, [span]);

  return (
    <div className="mark-tape">
      <div className="mark-tape-grid" aria-hidden="true">
        {ruler.map((at) => <i key={at} style={{ left: `${(at / span) * 100}%` }} />)}
      </div>
      {live && <span className="mark-tape-now" aria-hidden="true" />}
      <span className="mark-tape-base" aria-hidden="true" />
      {marks.map((mark, index) => {
        const offset = (new Date(mark.occurred_at_utc).getTime() - start) / 1000;
        const left = Math.max(0, Math.min(100, (offset / span) * 100));
        const classes = [
          "mark-tape-tick",
          index === marks.length - 1 && live ? "is-latest" : "",
          unmappedIds.has(mark.id) ? "is-outside" : "",
        ].filter(Boolean).join(" ");
        return <span key={mark.id} className={classes} style={{ left: `${left}%` }}><i /></span>;
      })}
      <div className="mark-tape-ruler" aria-hidden="true">
        <span>0:00</span>
        {ruler.map((at) => (
          <span key={at} style={{ left: `${(at / span) * 100}%` }} className="is-step">{fmtClock(at)}</span>
        ))}
        <span>{live ? "now" : fmtClock(span)}</span>
      </div>
    </div>
  );
}

export default function RecallLivePanel({ onScanRecording, sessions = [] }: LiveSessionPanelProps) {
  const [session, setSession] = useState<RecallSession | null>(null);
  const [busy, setBusy] = useState<"start" | "remember" | "stop" | "scan" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const [markSource, setMarkSource] = useState<MarkSource>("stream");
  const [vodUrl, setVodUrl] = useState("");
  const [earlier, setEarlier] = useState<RecallSession[]>([]);
  const [plan, setPlan] = useState<RecallRegionPlan | null>(null);
  const hotkey = useMemo(
    () => displayHotkey(window.electronAPI?.getApiConfig?.().rememberHotkey),
    [],
  );

  const refresh = useCallback(async (sessionId: string) => {
    const next = await getRecallSession(sessionId);
    setSession(next);
    return next;
  }, []);

  const loadEarlier = useCallback(async () => {
    try {
      const all = await listRecallSessions(EARLIER_SESSION_LIMIT * 3);
      const ended = all.filter((entry) => entry.status === "ended").slice(0, EARLIER_SESSION_LIMIT);
      // The list route carries no events, so the mark count needs the detail.
      // A handful of loopback reads against local SQLite; cheap, and it is the
      // only number on the row that matters.
      const detailed = await Promise.all(ended.map(async (entry) => {
        try { return await getRecallSession(entry.id); } catch { return entry; }
      }));
      setEarlier(detailed);
    } catch {
      setEarlier([]);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    void getActiveRecallSession()
      .then((active) => {
        if (!mounted || !active) return;
        setSession(active);
        setMarkSource(active.source_platform === "local_replay" ? "recording" : "stream");
      })
      .catch(() => {
        if (mounted) setMessage("The local Recall engine is not available yet.");
      });
    void loadEarlier();
    return () => { mounted = false; };
  }, [loadEarlier]);

  useEffect(() => {
    if (session?.status !== "active") return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [session?.status]);

  useEffect(() => {
    const unsubscribe = window.electronAPI?.onRememberResult?.((raw) => {
      const result = raw as RememberResult;
      if (result?.ok && session?.id) {
        setMessage(null);
        void refresh(session.id).catch(() => {});
      } else if (result && !result.ok) {
        setMessage("That moment was not saved. Make sure this session is still running.");
      }
    });
    return unsubscribe;
  }, [refresh, session?.id]);

  const marks = marksOf(session);
  const markCount = marks.length;
  const active = session?.status === "active";
  const ended = session?.status === "ended";

  // The windows come from the engine's own region planner rather than a
  // second copy of the pre/post-roll constants over here: it already merges
  // overlapping marks and reports which ones it could not place.
  useEffect(() => {
    if (!session || !markCount) { setPlan(null); return; }
    let alive = true;
    void buildRecallRegionPlan({ sessionId: session.id })
      .then((next) => { if (alive) setPlan(next); })
      .catch(() => { if (alive) setPlan(null); });
    return () => { alive = false; };
  }, [session, markCount]);

  const unmappedIds = useMemo(
    () => new Set(plan?.unmapped_event_ids ?? []),
    [plan],
  );

  /** Which Recall Live sessions a scan was actually run against. Matched here
   * rather than fetched: the scan sessions are already loaded, and the link is
   * recorded on the job when the scan starts. */
  const scansByRecallSession = useMemo(() => {
    const map = new Map<string, ScannedSession>();
    for (const entry of sessions) {
      if (!entry.recallSessionId) continue;
      const existing = map.get(entry.recallSessionId);
      // Prefer a completed scan over an abandoned earlier attempt.
      if (!existing || (entry.status === "completed" && existing.status !== "completed")) {
        map.set(entry.recallSessionId, entry);
      }
    }
    return map;
  }, [sessions]);
  const planReady = !!plan && plan.regions.length > 0;

  const start = async () => {
    setBusy("start");
    try {
      const next = await startRecallSession(markSource === "stream" ? {
        sourcePlatform: "twitch",
        title: "Live stream session",
        // No initialStreamTimeSeconds on purpose: the viewer joined at an
        // unknown stream position, so the VOD's real start time (resolved at
        // attach) is the only honest anchor.
        metadata: { mark_source: "live_stream", ui_surface: "recall_live_tab" },
      } : {
        sourcePlatform: "local_replay",
        title: "Recording session",
        initialStreamTimeSeconds: 0,
        metadata: { mark_source: "local_replay", ui_surface: "recall_live_tab" },
      });
      setSession(next);
      setNow(Date.now());
      setMessage(null);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not start a Recall session.");
    } finally {
      setBusy(null);
    }
  };

  const remember = async () => {
    if (!session || !active) return;
    setBusy("remember");
    try {
      await rememberMoment({ sessionId: session.id, source: "recall_live_tab" });
      await refresh(session.id);
      setMessage(null);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not save that moment.");
    } finally {
      setBusy(null);
    }
  };

  const stop = async () => {
    if (!session || !active) return;
    setBusy("stop");
    try {
      const next = await stopRecallSession(session.id);
      setSession(next);
      setMessage(markCount
        ? null
        : "Session ended without any marks. Start another and press the shortcut at least once.");
      void loadEarlier();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not stop this session.");
    } finally {
      setBusy(null);
    }
  };

  const scan = async () => {
    if (!session || !ended || markCount < 1) return;
    setBusy("scan");
    try {
      let path: string | null;
      let sourceType: ScanSourceType;
      if (markSource === "stream") {
        path = vodUrl.trim();
        sourceType = "twitch";
        if (!path) {
          setMessage("Paste the Twitch VOD link for the stream you marked.");
          return;
        }
      } else {
        // Only the local-file path needs the desktop shell; attaching a URL
        // works anywhere.
        if (!window.electronAPI) {
          setMessage("Open Recall in the desktop app to select the recording.");
          return;
        }
        path = await window.electronAPI.openFileDialog();
        sourceType = "file";
        if (!path) return;
      }
      setMessage(null);
      await onScanRecording(path, session.id, sourceType);
      setMessage(`Scanning ${recordingName(path)} around ${plural(markCount, "marked moment")}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not start the scan.");
    } finally {
      setBusy(null);
    }
  };

  const [removing, setRemoving] = useState<string | null>(null);
  const removeMark = async (eventId: string) => {
    if (!session || removing) return;
    setRemoving(eventId);
    try {
      await deleteRecallEvent({ sessionId: session.id, eventId });
      await refresh(session.id);
      setMessage(null);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not remove that mark.");
    } finally {
      setRemoving(null);
    }
  };

  const reopen = async (entry: RecallSession) => {
    setSession(entry);
    setMarkSource(entry.source_platform === "local_replay" ? "recording" : "stream");
    setVodUrl("");
    setMessage(null);
  };

  const resetSession = () => { setSession(null); setVodUrl(""); setPlan(null); setMessage(null); void loadEarlier(); };
  const keys = (hotkey && hotkey !== "Remember button" ? hotkey : "").split(" + ").filter(Boolean);
  const markList = (
    <div className="marks">
      {(active ? [...marks].reverse() : marks).map((mark, index) => {
        const ordinal = active ? markCount - index : index + 1;
        const offset = (new Date(mark.occurred_at_utc).getTime() - startedAt(session)) / 1000;
        const outside = unmappedIds.has(mark.id);
        return (
          <div key={mark.id} className={`mark ${outside ? "is-outside" : ""}`}>
            <span className="mark-n num">#{String(ordinal).padStart(2, "0")}</span>
            <b className="tcode num">{fmtClock(Math.max(0, offset))}</b>
            <span className="t3">
              {clockOfDay(mark.occurred_at_utc)} · {mark.source === "recall_live_tab" ? "button" : "shortcut"}
              {outside && <em className="mark-out"> · outside the recording</em>}
            </span>
            <button
              type="button" className="icon-btn"
              disabled={removing !== null}
              aria-label={`Remove the mark at ${fmtClock(Math.max(0, offset))}`}
              onClick={() => void removeMark(mark.id)}
            >
              <X aria-hidden="true" />
            </button>
          </div>
        );
      })}
      {!markCount && (
        <div className="marks-empty">
          <b>No marks yet</b>
          <small className="t3">
            {active && hotkey
              ? `Press ${hotkey} the moment something happens. Every mark lands here.`
              : "Every moment you mark lands here as you make it."}
          </small>
        </div>
      )}
    </div>
  );

  if (!session) {
    const marks = [0.18, 0.41, 0.63, 0.86];
    return (
      <div className="live-idle">
        <section className="live-explain glass" aria-label="How a mark works">
          <div className="le-copy">
            <span className="eyebrow">How a mark works</span>
            <h2 className="disp">Press your key when it pops off.</h2>
            <p className="t2">Recall keeps the 90 seconds before and the 30 after. Only those windows get scanned, so a 4 hour stream reads in minutes.</p>
            <div className="le-key">
              {keys.length ? (
                <div className="caps">{keys.map((key, index) => <span key={key} className="caps-k">{index > 0 && <span className="plus">+</span>}<kbd className="kcap">{key}</kbd></span>)}</div>
              ) : <b>Use the Remember button</b>}
              <small className="t3">Works in-game, even with Recall in the background.</small>
            </div>
          </div>
          <div className="le-viz" aria-hidden="true">
            <div className="hm-viz">
              {marks.map((at) => <span key={at} className="win" style={{ left: `calc(${at * 100}% - 9%)`, width: "12%" }} />)}
              {marks.map((at) => <span key={`p${at}`} className="pin" style={{ left: `${at * 100}%` }}><i /></span>)}
              <span className="le-span before" style={{ left: `calc(${marks[1] * 100}% - 9%)`, width: "9%" }}>90s</span>
              <span className="le-span after" style={{ left: `${marks[1] * 100}%`, width: "3%" }}>30s</span>
            </div>
            <div className="le-axis num"><span>0:00</span><span>1:00</span><span>2:00</span><span>3:00</span><span>4:00</span></div>
            <div className="hm-legend"><span><i className="lg-pin" />Your mark</span><span><i className="lg-win" />What gets scanned</span></div>
          </div>
        </section>

        <div className="live-grid">
          <section className="panel glass live-setup" aria-label="Start a Recall Live session">
            <span className="lbl" id="live-kind">What are you marking?</span>
            <div className="optgrid" role="radiogroup" aria-labelledby="live-kind">
              <button type="button" role="radio" aria-checked={markSource === "stream"} className={`opt ${markSource === "stream" ? "on" : ""}`} onClick={() => setMarkSource("stream")}>
                <b><Video aria-hidden="true" />A stream, live right now</b>
                <small>Start any time. Attach the VOD when it publishes and Recall lines your marks up with the real stream clock.</small>
              </button>
              <button type="button" role="radio" aria-checked={markSource === "recording"} className={`opt ${markSource === "recording" ? "on" : ""}`} onClick={() => setMarkSource("recording")}>
                <b><Play aria-hidden="true" />A recording you are playing</b>
                <small>Start the file at 0:00 and start Recall at the same moment. Your marks land on the file directly.</small>
              </button>
            </div>
            <div className="live-hotkey"><RememberHotkeySetting /></div>
            <div className="row live-go">
              <button type="button" className="btn heat lg" disabled={busy !== null} onClick={start}>
                <Target aria-hidden="true" />
                {busy === "start" ? "Starting…" : "Start marking"}
              </button>
              <span className="t3">Nothing is recorded. Recall keeps the second you pressed the key, on this PC.</span>
            </div>
            {message && <p className="live-msg" role="status">{message}</p>}
          </section>

          <section className="live-earlier-col" aria-label="Earlier sessions">
            <div className="sec-t"><h2 className="disp">Earlier sessions</h2></div>
            <div className="live-earlier">
              {earlier.map((entry) => {
                const count = marksOf(entry).length;
                const scanned = scansByRecallSession.get(entry.id);
                return (
                  <button type="button" key={entry.id} className="brow glass live-prev" onClick={() => void reopen(entry)}>
                    <span className="live-prev-ic" aria-hidden="true"><Broadcast /></span>
                    <span className="live-prev-copy">
                      <b>{entry.title || "Recall session"}</b>
                      <small>
                        {shortDate(entry.started_at_utc)}
                        {entry.ended_at_utc ? ` · ${fmtClock(elapsedSeconds(entry, now))}` : ""}
                        {" · "}<span className="num">{plural(count, "mark")}</span>
                      </small>
                    </span>
                    {scanned?.status === "completed"
                      ? <span className="badge keep">{plural(scanned.clips.length, "moment")} found</span>
                      : scanned ? <span className="badge">Scan {scanned.status}</span>
                        : <span className="badge heat">Not scanned yet</span>}
                  </button>
                );
              })}
              {!earlier.length && (
                <p className="live-none t3">Sessions you finish stay here until they're scanned, so marks are never lost.</p>
              )}
            </div>
          </section>
        </div>
      </div>
    );
  }

  return (
    <div className="narrow live-session">
      {active ? (
        <section className="live-hero glass" aria-label="Recall Live session">
          <span className="badge live-badge"><span className="dot" /><span>Marking</span> · {markSource === "stream" ? "live stream" : "recording"}</span>
          <strong className="live-clock num" aria-label={`Session duration ${elapsedLabel(session, now)}`}>{elapsedLabel(session, now)}</strong>
          <button type="button" className="remember" disabled={busy !== null} onClick={remember}>
            <span>{busy === "remember" ? "Saving…" : "Remember this"}{hotkey && <kbd>{hotkey}</kbd>}</span>
          </button>
          <p className="t3 live-hero-note">Recall is listening. The shortcut works with this window minimised, so go back to the game.</p>
          {message && <p className="live-msg" role="status">{message}</p>}
        </section>
      ) : (
        <section className="panel glass live-ended" aria-label="Attach the recording">
          <span className="eyebrow">{plural(markCount, "mark")} · started {clockOfDay(session.started_at_utc)}</span>
          <h2 className="disp">Where is the recording?</h2>
          <p className="t2">{markSource === "stream" ? STREAM_HINT : RECORDING_HINT}</p>
          {markSource === "stream" ? (
            <div className="row live-attach">
              <input
                type="url" className="field" value={vodUrl}
                placeholder="https://www.twitch.tv/videos/…"
                aria-label="Twitch VOD link for the stream you marked"
                onChange={(event) => setVodUrl(event.target.value)}
              />
              <button type="button" className="btn heat lg" disabled={busy !== null || markCount < 1} onClick={scan}>
                <Film aria-hidden="true" />{busy === "scan" ? "Opening…" : "Attach VOD & scan"}
              </button>
            </div>
          ) : (
            <div className="row">
              <button type="button" className="btn heat lg" disabled={busy !== null || markCount < 1} onClick={scan}>
                <FolderOpen aria-hidden="true" />{busy === "scan" ? "Opening…" : "Select recording & scan"}
              </button>
            </div>
          )}
          {planReady && (
            <div className="tele">
              <div><span>Search windows</span><b className="num">{plan!.regions.length}</b></div>
              <div><span>Footage to read</span><b className="num">{fmtClock(plan!.total_region_seconds)}</b></div>
              <div><span>Instead of</span><b className="num">{fmtClock(elapsedSeconds(session, now))}</b></div>
            </div>
          )}
          {message && <p className="live-msg" role="status">{message}</p>}
          <div className="row wrap live-ended-foot">
            <button type="button" className="btn sm ghost" disabled={busy !== null} onClick={resetSession}><Check aria-hidden="true" />Start another</button>
            <span className="t3">A full scan of the whole recording is always there from New stream.</span>
          </div>
        </section>
      )}

      <section className="panel glass live-marks" aria-label="Marks in this session">
        <div className="panel-t">
          <h3 className="disp">Marks</h3>
          <span className="badge num">{markCount}</span>
          <span className="t3 live-order">{active ? "newest first" : "in stream order"}</span>
          <span className="sp" />
          {active
            ? (
              <button type="button" className="btn sm ghost danger" disabled={busy !== null} onClick={stop}>
                <X aria-hidden="true" />{busy === "stop" ? "Ending…" : "End session"}
              </button>
            )
            : <span className="badge">Session ended</span>}
        </div>
        <MarkTape marks={marks} session={session} now={now} live={!!active} unmappedIds={unmappedIds} />
        {markList}
        <p className="t3 live-foot">
          {planReady
            ? "Each window is 90s before a mark and 30s after. Marks closer than that share one."
            : "Windows are worked out once your marks can be placed against the recording."}
        </p>
      </section>
    </div>
  );
}

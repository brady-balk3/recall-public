// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useCallback, useEffect, useMemo, useState } from "react";
import { Check, Clock, Film, FolderOpen, Play, Target, Video, X } from "../lib/icons";
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

  return (
    <div className="rlive-body">
      <section className={`rlive-stage ${active ? "is-live" : ""}`} aria-label="Recall Live session">
        {!session && (
          <div className="rlive-console">
            <div>
              <h2>What are you marking?</h2>
              <p className="rlive-console-q">This is how Recall lines your marks up with the recording later.</p>
            </div>

            <div className="rlive-sources" role="radiogroup" aria-label="What you are marking">
              <button
                type="button" role="radio" aria-checked={markSource === "stream"}
                className={`rlive-source ${markSource === "stream" ? "is-selected" : ""}`}
                onClick={() => setMarkSource("stream")}
              >
                <span className="rlive-source-title">
                  <Video size={15} aria-hidden="true" />A stream, live right now
                  <i className="rlive-source-mark" aria-hidden="true" />
                </span>
                <em>Start any time. Attach the VOD when it publishes and Recall anchors your marks to the real stream clock.</em>
              </button>
              <button
                type="button" role="radio" aria-checked={markSource === "recording"}
                className={`rlive-source ${markSource === "recording" ? "is-selected" : ""}`}
                onClick={() => setMarkSource("recording")}
              >
                <span className="rlive-source-title">
                  <Play size={15} aria-hidden="true" />A recording you are playing
                  <i className="rlive-source-mark" aria-hidden="true" />
                </span>
                <em>Start the file at 0:00 and start Recall at the same moment. Your marks land on the file directly.</em>
              </button>
            </div>

            <RememberHotkeySetting />

            <div className="rlive-start">
              <button type="button" className="cta-accent" disabled={busy !== null} onClick={start}>
                <Target size={15} aria-hidden="true" />
                {busy === "start" ? "Starting…" : "Start marking"}
              </button>
              <p className="rlive-fine">
                <b>Nothing is recorded.</b> Recall writes the second you pressed the key, and your clock
                offset, to this PC. No audio, no video, no upload.
              </p>
            </div>
            {message && <p className="rlive-message" role="status">{message}</p>}
          </div>
        )}

        {!!session && (
          <>
            <div className="rlive-instrument">
              <div className="rlive-clockrow">
                <strong className={`rlive-clock ${ended ? "is-ended" : ""}`} aria-label={`Session duration ${elapsedLabel(session, now)}`}>
                  {elapsedLabel(session, now)}
                </strong>
                <span className="rlive-clockmeta">
                  <span className={`rlive-state ${active ? "is-active" : "is-ended"}`}>
                    <i aria-hidden="true" />{active ? "Marking" : "Session ended"}
                  </span>
                  <small>
                    Started {clockOfDay(session.started_at_utc)} · {markSource === "stream" ? "live stream" : "recording"}
                  </small>
                </span>
                <span className="rlive-count">
                  <b className="t-num">{String(markCount).padStart(2, "0")}</b>
                  <small>{markCount === 1 ? "moment marked" : "moments marked"}</small>
                </span>
              </div>

              <MarkTape
                marks={marks} session={session} now={now} live={!!active} unmappedIds={unmappedIds}
              />
              <div className="rlive-tape-legend">
                <em><i aria-hidden="true" />one tick per mark, placed by elapsed time</em>
                {unmappedIds.size > 0 && (
                  <em className="is-outside">
                    <i aria-hidden="true" />{plural(unmappedIds.size, "mark")} outside the recording
                  </em>
                )}
              </div>

              {planReady && (
                <dl className="rlive-projection">
                  <div>
                    <dt>Search windows</dt>
                    <dd className="t-num">{plan!.regions.length}<small>from {plural(plan!.mapped_event_count, "mark")}</small></dd>
                  </div>
                  <div>
                    <dt>Video to scan</dt>
                    <dd className="t-num">
                      {fmtClock(plan!.total_region_seconds)}
                      <small>of {fmtClock(elapsedSeconds(session, now))}</small>
                    </dd>
                  </div>
                  <div className="rlive-projection-say">
                    <span>
                      {active
                        ? <>End the session, attach the recording, and Recall reads <b>{fmtClock(plan!.total_region_seconds)}</b> instead of the whole stream.</>
                        : <>Recall reads <b>{fmtClock(plan!.total_region_seconds)}</b> of video instead of the whole recording. Marks close together share one window.</>}
                    </span>
                  </div>
                </dl>
              )}
            </div>

            {active && (
              <div className="rlive-actionbar">
                <button type="button" className="rlive-remember" disabled={busy !== null} onClick={remember}>
                  <Target size={19} aria-hidden="true" />
                  {busy === "remember" ? "Saving…" : "Remember this"}
                  {hotkey && <kbd className="rlive-remember-key">{hotkey}</kbd>}
                </button>
                <p className="rlive-listening">
                  <b>Recall is listening.</b> The shortcut fires with this window minimised, so you can go
                  back to the game.
                </p>
                <button type="button" className="btn-secondary" disabled={busy !== null} onClick={stop}>
                  <X size={13} aria-hidden="true" />
                  {busy === "stop" ? "Ending…" : "End session"}
                </button>
              </div>
            )}

            {ended && (
              <div className="rlive-attach">
                <div className="rlive-attach-ask">
                  <b>Where is the recording?</b>
                  <small>{markSource === "stream" ? STREAM_HINT : RECORDING_HINT}</small>
                </div>
                {markSource === "stream" && (
                  <input
                    type="url" className="rlive-vod-input" value={vodUrl}
                    placeholder="https://www.twitch.tv/videos/…"
                    aria-label="Twitch VOD link for the stream you marked"
                    onChange={(event) => setVodUrl(event.target.value)}
                  />
                )}
                <div className="rlive-start">
                  <button
                    type="button" className="cta-accent"
                    disabled={busy !== null || markCount < 1} onClick={scan}
                  >
                    {markSource === "stream" ? <Film size={15} aria-hidden="true" /> : <FolderOpen size={15} aria-hidden="true" />}
                    {busy === "scan"
                      ? "Opening…"
                      : markSource === "stream" ? "Attach VOD & scan" : "Select recording & scan"}
                  </button>
                  <button
                    type="button" className="btn-secondary" disabled={busy !== null}
                    onClick={() => { setSession(null); setVodUrl(""); setPlan(null); setMessage(null); void loadEarlier(); }}
                  >
                    <Check size={13} aria-hidden="true" /> Start another
                  </button>
                  <p className="rlive-fine">
                    A full scan of the whole recording stays available from <b>New project</b> if you
                    would rather not rely on the marks.
                  </p>
                </div>
                {message && <p className="rlive-message" role="status">{message}</p>}
              </div>
            )}
            {active && message && <p className="rlive-message" role="status">{message}</p>}
          </>
        )}
      </section>

      <aside className="rlive-rail" aria-label={session ? "Marks in this session" : "Earlier sessions"}>
        {session ? (
          <>
            <div className="rlive-rail-head">
              <b>Marks</b><small>{active ? "newest first" : "in stream order"}</small>
              <span className="t-num">{String(markCount).padStart(2, "0")}</span>
            </div>
            <div className="rlive-rail-list">
              {(active ? [...marks].reverse() : marks).map((mark, index) => {
                const ordinal = active ? markCount - index : index + 1;
                const offset = (new Date(mark.occurred_at_utc).getTime() - startedAt(session)) / 1000;
                const outside = unmappedIds.has(mark.id);
                return (
                  <div key={mark.id} className={`rlive-mark ${outside ? "is-outside" : ""}`}>
                    <span className="rlive-mark-ix t-num">{String(ordinal).padStart(2, "0")}</span>
                    <span className="rlive-mark-body">
                      <b className="t-num">{fmtClock(Math.max(0, offset))}</b>
                      <small>
                        {clockOfDay(mark.occurred_at_utc)}
                        {" · "}
                        {mark.source === "recall_live_tab" ? "button" : "shortcut"}
                      </small>
                    </span>
                    {outside && <span className="rlive-mark-flag">outside</span>}
                    <button
                      type="button" className="rlive-mark-remove"
                      disabled={removing !== null}
                      aria-label={`Remove the mark at ${fmtClock(Math.max(0, offset))}`}
                      onClick={() => void removeMark(mark.id)}
                    >
                      <X size={13} aria-hidden="true" />
                    </button>
                  </div>
                );
              })}
              {!markCount && (
                <div className="rlive-rail-empty">
                  <Target size={18} aria-hidden="true" />
                  <b>No marks yet</b>
                  <small>
                    {active && hotkey
                      ? `Press ${hotkey} the moment something happens. Every mark lands here.`
                      : "Every moment you mark lands here as you make it."}
                  </small>
                </div>
              )}
            </div>
            <div className="rlive-rail-foot">
              {planReady
                ? <>Each window is 90s before a mark and 30s after. Marks closer than that share one.</>
                : <>Windows are worked out once your marks can be placed against the recording.</>}
            </div>
          </>
        ) : (
          <>
            <div className="rlive-rail-head">
              <b>Earlier sessions</b>
              <span className="t-num">{earlier.length}</span>
            </div>
            <div className="rlive-rail-list">
              {earlier.map((entry) => {
                const count = marksOf(entry).length;
                return (
                  <button
                    type="button" key={entry.id} className="rlive-prev"
                    onClick={() => void reopen(entry)}
                  >
                    <span className="rlive-prev-name">
                      <b>{entry.title || "Recall session"}</b>
                      <small>
                        {shortDate(entry.started_at_utc)}
                        {entry.ended_at_utc ? ` · ${fmtClock(elapsedSeconds(entry, now))}` : ""}
                      </small>
                    </span>
                    <span className="rlive-prev-meta">
                      <span className="t-num">{plural(count, "mark")}</span>
                      <i className="rlive-prev-sep" aria-hidden="true" />
                      {(() => {
                        const scan = scansByRecallSession.get(entry.id);
                        if (scan?.status === "completed") {
                          return <span className="rlive-chip is-ok">{plural(scan.clips.length, "moment")} found</span>;
                        }
                        if (scan) return <span className="rlive-chip">Scan {scan.status}</span>;
                        return <span className="rlive-chip is-warn">Not scanned yet</span>;
                      })()}
                    </span>
                  </button>
                );
              })}
              {!earlier.length && (
                <div className="rlive-rail-empty">
                  <Clock size={18} aria-hidden="true" />
                  <b>No sessions yet</b>
                  <small>Sessions you finish stay here, so an unscanned one is never lost.</small>
                </div>
              )}
            </div>
            <div className="rlive-rail-foot">
              Marks stay on this PC. An unscanned session can be attached to its recording at any time.
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

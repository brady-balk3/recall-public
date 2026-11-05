// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * A trim in the theater re-cuts from the stream's recording. When that
 * recording has left the disk, the creator downloads it again right here and
 * watches it come in, instead of leaving review to find the Storage screen.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../lib/api";
import { Download, RefreshCw } from "../lib/icons";

type SourceStatus = {
  source_key?: string | null;
  state: string;
  restore_available: boolean;
  local_path?: string | null;
  last_error?: string | null;
  file_size_bytes?: number;
  restore_progress?: { progress: number; message: string } | null;
};

export type SourceRestoreState = "unknown" | "present" | "missing" | "restoring" | "ready" | "failed";

/** Whether this session's recording is on disk, and a way to bring it back. */
export function useSourceRestore(sessionId: string | undefined, apiEndpoint: string, enabled: boolean) {
  const [state, setState] = useState<SourceRestoreState>("unknown");
  const [status, setStatus] = useState<SourceStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const checked = useRef<string | null>(null);

  const read = useCallback(async () => {
    if (!sessionId) return null;
    const response = await apiFetch(`/jobs/${sessionId}/source-status`, undefined, apiEndpoint).catch(() => null);
    if (!response?.ok) return null;
    const next = await response.json().catch(() => null) as SourceStatus | null;
    if (next) setStatus(next);
    return next;
  }, [sessionId, apiEndpoint]);

  useEffect(() => {
    checked.current = null;
    setState("unknown");
    setStatus(null);
    setError(null);
  }, [sessionId]);

  // Asked once per session, and only once a trim is being set up.
  useEffect(() => {
    if (!enabled || !sessionId || checked.current === sessionId) return;
    checked.current = sessionId;
    void read().then((next) => {
      if (!next) return;
      if (next.local_path) setState("present");
      else if (next.state === "restoring") setState("restoring");
      else setState("missing");
    });
  }, [enabled, sessionId, read]);

  useEffect(() => {
    if (state !== "restoring") return;
    const timer = window.setInterval(async () => {
      const next = await read();
      if (!next) return;
      if (next.local_path) setState("ready");
      else if (next.state === "unavailable") {
        setError(next.last_error || "Twitch wouldn't give the recording back.");
        setState("failed");
      } else if (next.state !== "restoring" && !next.restore_progress) {
        setState("missing");
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [state, read]);

  const start = useCallback(async () => {
    if (!sessionId) return;
    setError(null);
    const response = await apiFetch(`/jobs/${sessionId}/source/restore`, { method: "POST" }, apiEndpoint).catch(() => null);
    const body = await response?.json().catch(() => ({})) ?? {};
    if (!response?.ok) {
      const detail = body?.detail;
      setError(typeof detail === "string" ? detail : detail?.message || "Recall couldn't start the download.");
      setState("failed");
      return;
    }
    if (body.status === "already_available") setState("ready");
    else if (body.status === "blocked_active_scan") { setError("A scan is running. The download can start once it finishes."); setState("failed"); }
    else if (body.status === "blocked_low_disk") { setError("There isn't enough free disk space for this recording."); setState("failed"); }
    else setState("restoring");
    if (body.source) setStatus(body.source);
  }, [sessionId, apiEndpoint]);

  const cancel = useCallback(async () => {
    if (!status?.source_key) return;
    await apiFetch(`/source-assets/${encodeURIComponent(status.source_key)}/restore/cancel`, { method: "POST" }, apiEndpoint).catch(() => undefined);
    setState("missing");
  }, [status?.source_key, apiEndpoint]);

  // A trim error that says the recording is gone is the same news.
  const markMissing = useCallback(() => {
    setState((current) => (current === "restoring" ? current : "missing"));
    void read();
  }, [read]);

  return { state, status, error, start, cancel, markMissing };
}

function gigabytes(bytes?: number) {
  return bytes ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : "";
}

export function SourceRestoreStrip({ restore, pending }: {
  restore: ReturnType<typeof useSourceRestore>;
  /** A trim is waiting on the download and is applied when it lands. */
  pending: boolean;
}) {
  const { state, status, error, start, cancel } = restore;
  if (state === "unknown" || state === "present") return null;
  const progress = Math.max(0, Math.min(1, status?.restore_progress?.progress ?? 0));
  const size = gigabytes(status?.file_size_bytes);

  if (state === "restoring") {
    return (
      <div className="theater-source is-busy" role="status" aria-live="polite">
        <div className="theater-source-row">
          <RefreshCw size={13} className="is-spinning" aria-hidden="true" />
          <span className="theater-source-text">
            Downloading the stream{size ? ` · ${size}` : ""}
            <em>{status?.restore_progress?.message || "Starting the download"}</em>
          </span>
          <strong className="num">{Math.round(progress * 100)}%</strong>
          <button type="button" className="theater-source-ghost" onClick={() => void cancel()}>Cancel</button>
        </div>
        <div className="theater-source-bar" aria-hidden="true"><span style={{ width: `${progress * 100}%` }} /></div>
        {pending && <p className="theater-source-note">Your trim is applied as soon as it lands. Keep reviewing meanwhile.</p>}
      </div>
    );
  }
  if (state === "ready") {
    return (
      <div className="theater-source" role="status">
        <div className="theater-source-row">
          <span className="theater-source-text">The stream is back on your PC.</span>
        </div>
      </div>
    );
  }
  return (
    <div className="theater-source" role={state === "failed" ? "alert" : undefined}>
      <div className="theater-source-row">
        <span className="theater-source-text">
          {state === "failed" ? "The download didn't finish." : "This stream's recording isn't on your PC, so trims can't be cut yet."}
          {error && <em>{error}</em>}
        </span>
        {(status?.restore_available || state === "failed") && (
          <button type="button" className="theater-source-go" onClick={() => void start()}>
            <Download size={13} aria-hidden="true" />{state === "failed" ? "Try again" : `Download it${size ? ` · ${size}` : ""}`}
          </button>
        )}
      </div>
    </div>
  );
}

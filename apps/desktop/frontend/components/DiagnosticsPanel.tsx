// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useCallback, useEffect, useState } from "react";
import { Bug, FileText, FolderOpen } from "lucide-react";

/** Reaching a crash report used to require knowing a keyboard shortcut that was
 *  never bound. The app has a custom title bar, so a menu is not the place for
 *  this -- Settings is, next to everything else a creator changes by hand. */
export default function DiagnosticsPanel() {
  const [info, setInfo] = useState<{ logFile: string; logDir: string } | null>(null);
  const [note, setNote] = useState<string>();

  useEffect(() => {
    let cancelled = false;
    const api = window.electronAPI;
    if (!api || typeof api.getDiagnosticsInfo !== "function") return;
    const pending = api.getDiagnosticsInfo();
    if (!pending || typeof pending.then !== "function") return;
    pending.then(
      (next) => { if (!cancelled && next) setInfo(next); },
      () => { /* diagnostics are best-effort; the panel simply stays hidden */ },
    );
    return () => { cancelled = true; };
  }, []);

  const openLogFolder = useCallback(async () => {
    if (!info) return;
    const result = await window.electronAPI?.openPath?.(info.logDir);
    setNote(result && result.success === false
      ? "No log folder yet — it appears the first time Recall records something."
      : undefined);
  }, [info]);

  const openDevTools = useCallback(async () => {
    await window.electronAPI?.openDevTools?.();
  }, []);

  if (!info) return null;

  return (
    <div className="diagnostics-panel">
      <h3>Diagnostics</h3>
      <p>
        The lifecycle log below records app crashes and engine starts and exits.
        Older logs, engine logs, and developer tools may contain private paths,
        source details, or access tokens. Review those before sharing.
      </p>
      <div className="diagnostics-actions">
        <button type="button" className="btn-secondary" onClick={() => void openLogFolder()}>
          <FolderOpen size={14} aria-hidden="true" /> Open log folder
        </button>
        <button type="button" className="btn-secondary" onClick={() => void openDevTools()}>
          <Bug size={14} aria-hidden="true" /> Open developer tools
        </button>
      </div>
      <p className="diagnostics-path">
        <FileText size={12} aria-hidden="true" />
        <code>{info.logFile}</code>
      </p>
      <p className="diagnostics-path">
        Developer tools also open with <kbd>F12</kbd> or <kbd>Ctrl</kbd> +{" "}
        <kbd>Shift</kbd> + <kbd>I</kbd>.
      </p>
      {note && <p className="diagnostics-note" role="status">{note}</p>}
    </div>
  );
}

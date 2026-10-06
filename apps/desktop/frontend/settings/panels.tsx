// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** Settings building blocks: rows, the storage report, and what Recall has learned. */
import { useCallback, useEffect, useState } from "react";
import { StudioButton, type ChoiceOption } from "../components/StudioControls";
import { apiFetch } from "../lib/api";
import { learningStateDescription, learningStateLabel, plural } from "../lib/copy";
import { Film, HardDrive, RefreshCw, Trash2 } from "../lib/icons";
import { useJobStore } from "../lib/store";

export const PROCESSING_CHOICES: ChoiceOption<"fast" | "quality">[] = [
  { value: "fast", label: "Smart scan", description: "Focuses deep analysis on the moments most likely to matter.", badge: "Faster" },
  { value: "quality", label: "Best quality", description: "Scans the full recording closely and takes longer.", badge: "Thorough" },
];
export const PERFORMANCE_CHOICES: ChoiceOption<"background" | "balanced" | "max">[] = [
  { value: "background", label: "Keep PC responsive", description: "Quieter resource use while you work, play, or stream.", badge: "Quietest" },
  { value: "balanced", label: "Balanced", description: "Good scan speed without taking over the machine.", badge: "Recommended" },
  { value: "max", label: "Full speed", description: "Finishes sooner with heavier CPU and GPU use.", badge: "Fastest" },
];

export function SettingsPanel({ title, copy, children }: { title: string; copy: string; children: React.ReactNode }) { return <div className="settings-panel active"><div className="mock-v1-settings-heading"><h2>{title}</h2><p>{copy}</p></div>{children}</div>; }
/** One setting: what it is on the left, its control on the right. `block`
 * puts a wide control (choice cards, layouts) under the words instead. */
export function SettingRow({ title, copy, children, block = false }: { title: string; copy: string; children: React.ReactNode; block?: boolean }) {
  return <div className={`setrow ${block ? "is-block" : ""}`}><div className="st"><b>{title}</b><small>{copy}</small></div>{children}</div>;
}
export function Shortcut({ label, value }: { label: string; value: string }) {
  return <div className="sc"><span>{label}</span><span className="sc-keys">{value.split(" or ").map((keys, index) => <span key={keys}>{index > 0 && <em>or</em>}<kbd>{keys}</kbd></span>)}</span></div>;
}

const fmtBytes = (b: number) => (b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(1)} GB` : `${Math.max(0, b / 1024 ** 2).toFixed(0)} MB`);
const STORAGE_CATEGORIES = [
  ["clips", "Clips", "success"],
  ["scan_data", "Scan data", "warning"],
  ["session_data", "Session data", "info"],
] as const;

type DownloadedSource = {
  source_key: string;
  vod_id: string;
  twitch_url: string;
  local_path: string | null;
  file_size_bytes: number;
  retention_expires_at: string | null;
  pinned: boolean;
  state: string;
  state_label: string;
  restore_available: boolean;
  last_error?: string | null;
  sessions: { id: string; name: string }[];
};

type StorageReport = {
  recall: {
    cap_bytes: number;
    categories: Record<"clips" | "scan_data" | "session_data", number>;
    total_bytes: number;
    reclaimable_bytes: number;
    overage_bytes: number;
    cleanup_deferred: boolean;
    message: string;
  };
  downloaded_sources: {
    count: number;
    total_bytes: number;
    safe_to_remove_count: number;
    next_scheduled_removal: string | null;
    retention_days: number;
    sources: DownloadedSource[];
  };
};

const fmtStorageDate = (value: string | null) => value
  ? new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(value))
  : "None scheduled";

/** One honest Recall target plus a separate shared Twitch-source lifecycle. */
export function StoragePanel({ commitSettings, onManageClips }: { commitSettings: (patch: Partial<ReturnType<typeof useJobStore.getState>["settings"]>) => void; onManageClips: () => void }) {
  const settings = useJobStore((s) => s.settings);
  const clearScanCache = useJobStore((s) => s.clearScanCache);
  const addLog = useJobStore((s) => s.addLog);
  const [report, setReport] = useState<StorageReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const response = await apiFetch("/system/storage", undefined, settings.apiEndpoint);
      if (!response.ok) throw new Error("Storage inventory is unavailable.");
      setReport(await response.json());
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Storage inventory is unavailable.");
    } finally {
      setLoading(false);
    }
  }, [settings.apiEndpoint]);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (!report?.downloaded_sources.sources.some((source) => source.state === "in_use")) return;
    const timer = window.setInterval(() => void load(), 1500);
    return () => window.clearInterval(timer);
  }, [report?.downloaded_sources.sources, load]);

  const sourceAction = async (source: DownloadedSource, action: "remove" | "pin" | "restore") => {
    const key = `${source.source_key}:${action}`;
    setPending(key);
    setError(null);
    try {
      const response = await apiFetch(
        `/source-assets/${encodeURIComponent(source.source_key)}/${action}`,
        action === "pin"
          ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pinned: !source.pinned }) }
          : { method: "POST" },
        settings.apiEndpoint,
      );
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = body?.detail;
        throw new Error(typeof detail === "string" ? detail : detail?.message || `Could not ${action} this source.`);
      }
      addLog(
        action === "remove" ? "Downloaded source removed. Linked sessions and clips are intact."
          : action === "restore" ? "Source restoration started. Progress is recorded in session activity."
            : source.pinned ? "Source returned to seven-day aging." : "Source will stay on this PC.",
        "success",
      );
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : `Could not ${action} this source.`);
    } finally {
      setPending(null);
    }
  };

  const cleanSafeCache = async () => {
    setPending("cache");
    try {
      await clearScanCache();
      await load();
    } finally {
      setPending(null);
    }
  };

  const recall = report?.recall;
  const sources = report?.downloaded_sources;
  const capBytes = Math.max(1, recall?.cap_bytes || settings.recallStorageCapGb * 1024 ** 3);
  const usagePct = Math.min(100, (recall?.total_bytes || 0) / capBytes * 100);

  return (
    <div className="storage-lifecycle">
      <section className="mock-v1-storage storage-section" aria-labelledby="recall-storage-title">
        <div className="mock-v1-storage-head">
          <span id="recall-storage-title"><HardDrive size={15} /> Recall storage</span>
          {recall && <b className="t-num">{fmtBytes(recall.total_bytes)} <i>/ {fmtBytes(recall.cap_bytes)}</i></b>}
        </div>
        <div className="storage-policy-control">
          <label><span>Storage target</span><span><input className="input-text t-num" type="number" min={1} max={2000} step={1} value={settings.recallStorageCapGb} onChange={(event) => commitSettings({ recallStorageCapGb: Math.max(1, Math.min(2000, Number(event.target.value) || 25)) })} /> GB</span></label>
          <p>Recall automatically cleans only safe scan data. Clips, thumbnails, kept moments, and required session data are never removed.</p>
        </div>
        {recall ? (
          <>
            <div className="settings-storage-bar" aria-label={`${fmtBytes(recall.total_bytes)} of ${fmtBytes(recall.cap_bytes)} used`}><i className="tone-accent" style={{ width: `${Math.max(.7, usagePct)}%` }} /></div>
            <div className="mock-v1-storage-rows">
              {STORAGE_CATEGORIES.map(([key, label, tone]) => <div className="mock-v1-storage-row" key={key}><i className={`tone-${tone}`} /><span>{label}</span><b className="t-num">{fmtBytes(recall.categories[key])}</b></div>)}
              <div className="mock-v1-storage-row is-reclaimable"><i className="tone-muted" /><span>Reclaimable now</span><b className="t-num">{fmtBytes(recall.reclaimable_bytes)}</b></div>
            </div>
            {recall.cleanup_deferred && <p className="storage-note is-warning">Scan data is in use. Cleanup becomes available after the active scan finishes.</p>}
            {recall.overage_bytes > 0 && <div className="storage-overage" role="status"><span>{recall.message}</span><button type="button" className="library-linkbtn" onClick={onManageClips}>Manage clips in Clip Library</button></div>}
          </>
        ) : <span className="form-description">{loading ? "Measuring Recall storage…" : "Storage totals are unavailable."}</span>}
        <div className="mock-v1-inline-actions mock-v1-storage-actions"><StudioButton loading={pending === "cache"} disabled={!recall || recall.reclaimable_bytes <= 0 || recall.cleanup_deferred} onClick={() => void cleanSafeCache()}><RefreshCw size={14} /> Clean safe cache</StudioButton></div>
      </section>

      <section className="mock-v1-storage storage-section" aria-labelledby="downloaded-sources-title">
        <div className="mock-v1-storage-head">
          <span id="downloaded-sources-title"><Film size={15} /> Downloaded sources</span>
          {sources && <b className="t-num">{plural(sources.count, "Twitch VOD")} <i>· {fmtBytes(sources.total_bytes)}</i></b>}
        </div>
        <div className="storage-policy-control">
          <label><span>Retention after last use</span><span><input className="input-text t-num" type="number" min={1} max={365} step={1} value={settings.sourceRetentionDays} onChange={(event) => commitSettings({ sourceRetentionDays: Math.max(1, Math.min(365, Number(event.target.value) || 7)) })} /> days</span></label>
          <p>Sources stay separate from the Recall storage target. Restoring or using one restarts its retention window.</p>
        </div>
        {sources && <div className="source-summary"><span><b className="t-num">{sources.safe_to_remove_count}</b> safe to remove</span><span>Next scheduled removal <b>{fmtStorageDate(sources.next_scheduled_removal)}</b></span></div>}
        <div className="source-asset-list">
          {sources?.sources.length ? sources.sources.map((source) => {
            const busy = pending?.startsWith(source.source_key) || source.state === "in_use";
            const sessionNames = source.sessions.map((session) => session.name).join(", ");
            return <article className="source-asset-row" key={source.source_key}>
              <div className="source-asset-copy"><strong>Twitch VOD {source.vod_id}</strong><span title={sessionNames}>{sessionNames || "No linked sessions"}</span><small className={`is-${source.state}`}><i />{source.state_label}</small></div>
              <b className="source-asset-size t-num">{fmtBytes(source.file_size_bytes)}</b>
              <div className="source-asset-actions">
                {source.local_path && <button type="button" className={source.pinned ? "is-kept" : ""} disabled={busy} onClick={() => void sourceAction(source, "pin")}>{source.pinned ? "Release" : "Keep"}</button>}
                {source.local_path && <button type="button" disabled={busy || source.state !== "safe_to_remove"} title={source.state === "safe_to_remove" ? "Remove this managed download" : source.state_label} onClick={() => void sourceAction(source, "remove")}><Trash2 size={13} /> Remove</button>}
                {!source.local_path && source.restore_available && <button type="button" disabled={busy} onClick={() => void sourceAction(source, "restore")}><RefreshCw size={13} /> Restore</button>}
              </div>
            </article>;
          }) : !loading && <div className="source-empty">No downloaded Twitch sources yet. Local recordings are never listed or managed here.</div>}
        </div>
        {error && <div className="library-editor-error" role="alert">{error}</div>}
      </section>
    </div>
  );
}

/** Local-taste learning progress + reset (plan 22). */
export function LearningPanel() {
  const { learningStatus, learningLoading, loadLearningStatus, resetLearning } = useJobStore();
  const [confirm, setConfirm] = useState(false);
  useEffect(() => { loadLearningStatus(); }, [loadLearningStatus]);
  const count = learningStatus?.label_count ?? 0;
  const goal = learningStatus?.min_labels ?? 25;
  const pct = Math.min(100, Math.round((count / Math.max(1, goal)) * 100));
  return (
    <div className="mock-v1-learning">
      <div className="mock-v1-learning-head">
        <div><h4>Recall is learning</h4><p>{learningStatus?.creator_message || learningStateDescription(learningStatus?.state)}</p></div>
        <span className="mock-v1-learning-badge">{learningStateLabel(learningStatus?.state)}</span>
      </div>
      <div className="mock-v1-learning-meta"><span>{count} of {goal} decisions</span><span>Saved · exported · rejected</span></div>
      <div className="mock-v1-learning-bar"><i style={{ width: `${pct}%` }} /></div>
      <div className="mock-v1-learning-foot">
        <span className="form-description">Learning stays on this computer.</span>
        {confirm ? (
          <div className="mock-v1-inline-actions">
            <StudioButton tone="ghost" onClick={() => setConfirm(false)}>Cancel</StudioButton>
            <StudioButton tone="danger" loading={learningLoading} onClick={async () => { await resetLearning(); setConfirm(false); }}>Reset</StudioButton>
          </div>
        ) : (
          <StudioButton disabled={learningLoading || count === 0} onClick={() => setConfirm(true)}>Reset what Recall learned</StudioButton>
        )}
      </div>
    </div>
  );
}

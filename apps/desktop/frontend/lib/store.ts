// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { create } from "zustand";
import { DEFAULT_API_ENDPOINT, apiFetch, apiUrl, mediaUrl } from "./api";
import { fmtClock } from "./format";

export type JobStatus =
  | "idle"
  | "queued"
  | "analyzing"
  | "detecting"
  | "assembling"
  | "completed"
  | "cancelled"
  | "failed";

export interface Clip {
  id: string;
  title: string;
  /** Append-only identity within the source VOD used in creator filenames. */
  clipNumber?: number;
  /** Internal rendered stem available to custom {original} filename templates. */
  originalStem?: string;
  timestamp: string;
  confidence: number;
  /** Legacy review-order scalar. Used for sorting only, never as confidence. */
  deckScore?: number;
  /** Raw selector authority persisted for diagnostics on new scans. */
  selectionScore?: number;
  thumbnail?: string;
  duration: number;
  start_time?: number;
  end_time?: number;
  videoUrl?: string;
  /** Poster JPEG (reaction-peak frame) for grid cards + filmstrips. */
  thumbUrl?: string;
  description?: string;
  signals?: string[];
  /** Reaction-engine evidence: human-readable "why this clip" line. */
  reason?: string;
  /** Absolute VOD timestamp (seconds) of the reaction peak inside this clip. */
  peakTimestamp?: number;
  /** Raw per-modality means over the clip window: face, voice, speech, game. */
  modalityBreakdown?: Record<string, number>;
  /** Stored reaction/ranker evidence used for truthful creator-facing reasons. */
  features?: Record<string, number>;
  /** Review state, persisted server-side (plan 3.2). */
  kept?: boolean;
  passed?: boolean;
  maybe?: boolean;
  exported?: boolean;
  /** Posting hashtags from the caption engine (game-aware, plan 22 §3.3). */
  postTags?: string[];
  /** Non-gameplay scene code (lobby / intermission), else undefined. */
  scene?: string;
  /** Creator-facing label for a non-gameplay clip, e.g. "Lobby / Just chatting". */
  sceneLabel?: string;
  /** Local semantic judge's short cold-open line, when the judge ran. */
  hookLine?: string;
  /** Local semantic judge's editorial category. */
  momentType?: string;
  /** A ceiling-cut moment revealed into the review deck's second-look tier. */
  isOverflowCandidate?: boolean;
  /** Created by the creator in the VOD Editor, not proposed by Recall. */
  isManual?: boolean;
  /** Play a window of the full source VOD instead of a rendered clip file. */
  sourceWindow?: boolean;
  /** Whether this clip's rendered media is still on disk (GET /clips). The
   * backend stats the file; a set export_path is not proof it survived, since
   * a creator can reclaim the exports folder at any time. */
  mediaState?: ClipMediaState;
  /** Durable origin for clips produced by a Recall Session-assisted scan. */
  recallProvenance?: RecallProvenance;
}

export interface RecallProvenance {
  source: "recall_session" | "stream_memory";
  sessionId?: string;
  origin: "creator_marker" | "session_scan" | "memory_result";
  creatorProtected?: boolean;
  markerIds?: string[];
  markerTime?: number;
  memoryEntryId?: string;
  entryKind?: "transcript" | "clip" | "evidence";
  query?: string;
  sourceStart?: number;
  sourceEnd?: number;
}

/** Mirrors core/clip_media.py. `rebuildable` means the pixels are gone but the
 * clip can be re-rendered; `unavailable` means the source is gone too, and only
 * the metadata and the creator's decision survive. */
export type ClipMediaState = "ready" | "source_window" | "rebuildable" | "unavailable";

/** Per-session clip media health from GET /jobs (embedded) or /media-status. */
export interface SessionMedia {
  ready: number;
  source_window: number;
  rebuildable: number;
  unavailable: number;
  total: number;
  source_state: "local" | "restorable" | "gone";
  state: "ready" | "partial" | "rebuildable" | "unavailable";
}

export interface MoreCandidateResult {
  available: number;
  available_total: number;
  loaded: number;
  clips: Clip[];
}

const SCENE_DISPLAY: Record<string, string> = {
  lobby: "Lobby / Just chatting",
  intermission: "Intermission",
};

/** Compact per-job reaction curve R(t) from GET /jobs/{id}/timeline. */
export interface ReactionTimeline {
  version: number;
  t: number[];
  r: number[];
  game_markers: { t: number; label: string }[];
  /** Labeled game timeline for variety VODs (plan 22 §5.1); absent/empty on
   * single-game or pre-segmentation scans. */
  segments?: { start: number; end: number; game: string }[];
}

export type EtaConfidence = "unknown" | "low" | "medium" | "high";

export interface JobEvent {
  /** Explicit owner used to prevent stale telemetry crossing between scans. */
  jobId: string;
  eventType: string;
  phase?: string;
  stageLabel?: string;
  message?: string;
  progress?: number;
  payload?: any;
  at: number; // epoch ms
}

export interface FoundClip {
  clipId: string;
  title: string;
  timestamp: string;
  duration: number;
  score: number;
  /** True for selection-time candidates whose MP4/thumb don't exist yet. */
  pending?: boolean;
}

export interface Job {
  id: string;
  url: string;
  status: JobStatus;
  progress: number; // 0–100
  stage: {
    label: string;
    progress: number;
  };
  createdAt: number;
  clips?: Clip[];

  // --- Live backend telemetry (populated while running) ---
  currentPhase?: string;
  message?: string;
  elapsedSeconds?: number;
  /** Client timestamp for smoothly advancing elapsedSeconds between snapshots. */
  elapsedSyncedAt?: number;
  etaSeconds?: number | null;
  /** Client timestamp for smoothly counting down etaSeconds between updates. */
  etaSyncedAt?: number;
  etaConfidence?: EtaConfidence;
  etaReliable?: boolean;
  scannedSeconds?: number;
  scanSpeed?: number;
  transcribedSeconds?: number;
  transcriptionSpeed?: number;
  activeWorkers?: number;
  totalWorkers?: number;
  downloadProgress?: number;
  clipsFound?: number;
  vodDuration?: number;
  stageTimings?: Record<string, { started_at?: number; ended_at?: number; duration_seconds?: number }>;
  scanProfile?: {
    name: string;
    mode: string;
    sensitivity_band: string;
    density_band: string;
    clip_length: string;
    clip_length_label: string;
    facecam_tracking: boolean;
    details?: Record<string, unknown>;
  };
  events?: JobEvent[];
  foundClips?: FoundClip[];
  /** Real R(t) curve streamed mid-job via the timeline_ready SSE event. */
  liveTimeline?: ReactionTimeline;
  errorMessage?: string;
  /** Semantic/visual judge + device health for the active scan. */
  scanHealth?: {
    degraded?: boolean;
    device?: string | null;
    semantic?: Record<string, unknown> | null;
    visual?: Record<string, unknown> | null;
  };
}

export interface Session {
  id: string;
  name: string;
  sourceUrl: string;
  /** Recording/VOD calendar date used in creator-facing export filenames. */
  sourceDate?: string;
  /** The Recall Live session whose marks drove this scan, when one did. Lets
   * Recall Live tell a creator which earlier sessions were ever scanned. */
  recallSessionId?: string;
  createdAt: number;
  updatedAt: number;
  status: JobStatus;
  savedClipIds: string[];
  exportedClipIds: string[];
  clips: Clip[];
  /** True once this session's clip list has been authoritatively hydrated. */
  clipsHydrated?: boolean;
  vodDuration?: number; // total source VOD length in seconds
  message?: string; // last backend message — carries the reason for failed scans
  /** Landscape still from the source VOD (GET /jobs/{id}/thumb). */
  posterUrl?: string;
  /** Clip media health, embedded in GET /jobs so the gallery can render an
   * honest card without a status call per session. */
  media?: SessionMedia;
}

/**
 * Eventual file-structure persistence semantics:
 * sessions/<sessionId>/session.json
 * sessions/<sessionId>/source/original.*
 * sessions/<sessionId>/clips/saved/*.mp4
 * sessions/<sessionId>/clips/exported/*.mp4
 * sessions/<sessionId>/analysis/events.json
 */

export interface LogEntry {
  id: number;
  timestamp: string;
  text: string;
  type: "info" | "success" | "warn";
}

let logEntryId = 0;

export interface LearningStatus {
  labels: number;
  label_count: number;
  min_labels: number;
  can_train: boolean;
  active_mode: "personal" | "base" | "signals" | string;
  state: "not_started" | "learning" | "ready_to_personalize" | "personal_active" | string;
  last_trained_at?: string | null;
  has_user_model?: boolean;
  has_base_model?: boolean;
  feature_version?: string;
  creator_message: string;
}

export type ThemeMode = "light" | "dark";
export type AccentTheme = "ember" | "rose" | "violet" | "crimson" | "cobalt" | "sage";
/** How exported clips are framed. Automatic chooses one stable composition per
 * VOD from gameplay motion/event evidence. Consumed by the export engine. */
export type ExportLayout = "auto" | "vertical_split" | "gameplay_pip";

export type ManualClipLayout = ExportLayout | "full_gameplay";
export interface ManualClipOptions {
  start: number;
  end: number;
  title?: string;
  tags?: string[];
  layout: ManualClipLayout;
  focus_x: number;
  captions_enabled: boolean;
  video_fade_in: number;
  video_fade_out: number;
  audio_fade_in: number;
  audio_fade_out: number;
  memory_entry_id?: string;
  memory_query?: string;
}

export interface ManualFraming {
  type: "vertical_split" | "gameplay_pip" | "full_gameplay";
  facecam?: number[] | null;
  gameplay?: number[] | null;
  focus_x: number;
  requested: ManualClipLayout;
  source: "scan_model" | "nearest_clip" | "fallback";
  label: string;
  reason: string;
}

/** Burned-in caption look (plan 9.4). Consumed by the caption engine via job
 * settings; the backend whitelists fonts/sizes/positions and validates colors. */
export type CaptionFont = "Arial Black" | "Impact" | "Arial" | "Verdana" | "Georgia" | "Trebuchet MS" | "Segoe UI Black";
export type CaptionSize = "small" | "medium" | "large";
export type CaptionPosition = "bottom" | "middle" | "top";
export interface CaptionStyle {
  enabled: boolean;
  font: CaptionFont;
  size: CaptionSize;
  textColor: string;
  highlightColor: string;
  position: CaptionPosition;
}

export const DEFAULT_CAPTION_STYLE: CaptionStyle = {
  enabled: true,
  font: "Arial Black",
  size: "medium",
  textColor: "#FFFFFF",
  highlightColor: "#FFFF00",
  position: "bottom",
};

/** Per-platform export preset (plan 9.5). "none" copies files as-is; a platform
 * tags the export + offers a one-click jump to that platform's upload page. */
export type ExportPreset = "none" | "tiktok" | "shorts" | "x";

export type ExportOperationStatus = "queued" | "running" | "completed" | "partial" | "failed" | "cancelled";

const EXPORT_TERMINAL_STATUSES = new Set<ExportOperationStatus>([
  "completed",
  "partial",
  "failed",
  "cancelled",
]);

export interface ExportFailure {
  clipId: string;
  reason: string;
}

export interface ExportOperation {
  id: string;
  kind: "clips" | "reel";
  status: ExportOperationStatus;
  phase: string;
  totalClips: number;
  completedClips: number;
  currentClipId?: string;
  currentIndex: number;
  currentClipProgress?: number;
  currentClipDuration?: number;
  renderedSeconds?: number;
  reelProgress?: number;
  reelDuration?: number;
  progress: number;
  elapsedSeconds?: number;
  etaSeconds?: number;
  copied: number;
  errors: number;
  succeededClipIds: string[];
  failed: ExportFailure[];
  message: string;
  path?: string;
  clips?: number;
  cancelRequested: boolean;
  error?: string;
}

export interface ExportResult {
  copied: number;
  errors: number;
  succeededClipIds: string[];
  failed: ExportFailure[];
  cancelled?: boolean;
  error?: string;
}

export interface CompileReelResult {
  path?: string;
  clips?: number;
  errors?: number;
  succeededClipIds?: string[];
  failed?: ExportFailure[];
  cancelled?: boolean;
  error?: string;
}

function optionalFiniteNumber(value: unknown): number | undefined {
  if (typeof value !== "number" && typeof value !== "string") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function toExportOperation(data: Record<string, unknown>, fallbackTotal = 0): ExportOperation {
  return {
    id: String(data.id || ""),
    kind: data.kind === "reel" ? "reel" : "clips",
    status: String(data.status || "failed") as ExportOperationStatus,
    phase: String(data.phase || data.status || "starting"),
    totalClips: Number(data.total_clips || fallbackTotal || 0),
    completedClips: Number(data.completed_clips || 0),
    currentClipId: data.current_clip_id ? String(data.current_clip_id) : undefined,
    currentIndex: Number(data.current_index || 0),
    currentClipProgress: optionalFiniteNumber(data.current_clip_progress),
    currentClipDuration: optionalFiniteNumber(data.current_clip_duration),
    renderedSeconds: optionalFiniteNumber(data.rendered_seconds),
    reelProgress: optionalFiniteNumber(data.reel_progress),
    reelDuration: optionalFiniteNumber(data.reel_duration),
    progress: Math.max(0, Math.min(1, Number(data.progress || 0))),
    elapsedSeconds: optionalFiniteNumber(data.elapsed_seconds),
    etaSeconds: optionalFiniteNumber(data.eta_seconds),
    copied: Number(data.copied || 0),
    errors: Number(data.errors || 0),
    succeededClipIds: Array.isArray(data.succeeded_clip_ids)
      ? data.succeeded_clip_ids.filter((id): id is string => typeof id === "string")
      : [],
    failed: Array.isArray(data.failed)
      ? data.failed.flatMap((entry) => {
          if (!entry || typeof entry !== "object") return [];
          const row = entry as Record<string, unknown>;
          if (typeof row.clip_id !== "string") return [];
          return [{ clipId: row.clip_id, reason: String(row.reason || "Export failed.") }];
        })
      : [],
    message: String(data.message || "Preparing final files"),
    path: typeof data.path === "string" ? data.path : undefined,
    clips: optionalFiniteNumber(data.clips),
    cancelRequested: Boolean(data.cancel_requested),
    error: typeof data.error === "string" ? data.error : undefined,
  };
}

async function pollExportOperation(
  initial: ExportOperation,
  fallbackTotal: number,
  apiEndpoint: string,
  onUpdate: (operation: ExportOperation) => void,
): Promise<{ operation: ExportOperation; lost: boolean }> {
  let operation = initial;
  let failures = 0;
  while (!EXPORT_TERMINAL_STATUSES.has(operation.status)) {
    await new Promise<void>((resolve) => setTimeout(resolve, 300));
    try {
      const response = await apiFetch(
        `/clips/export-operations/${encodeURIComponent(operation.id)}`,
        {},
        apiEndpoint,
      );
      if (!response.ok) throw new Error(`export status ${response.status}`);
      operation = toExportOperation(await response.json(), fallbackTotal);
      failures = 0;
      onUpdate(operation);
    } catch {
      failures += 1;
      if (failures >= 5) return { operation, lost: true };
    }
  }
  return { operation, lost: false };
}

export interface Settings {
  displayName: string;              // shown in the Start-screen greeting
  processingMode: "fast" | "quality";
  /** How much of the machine a scan may use (backend process governor). */
  performanceProfile: "background" | "balanced" | "max";
  exportLayout: ExportLayout;
  captionStyle: CaptionStyle;
  exportPreset: ExportPreset;
  filenameTemplate: string;
  /** Target for Recall-owned clips, scan data, and required session data. */
  recallStorageCapGb: number;
  /** Twitch source aging window; downloaded sources are outside the target. */
  sourceRetentionDays: number;
  /** Shared preview loudness across Theater, Clip Library, and VOD Editor. */
  mediaVolume: number;
  mediaMuted: boolean;
  /** Keep long local work alive while still allowing the display to turn off. */
  keepAwakeWhileWorking: boolean;
  /** Show OS alerts for newly completed or failed scans/exports. */
  desktopNotifications: boolean;
  apiEndpoint: string;
  theme: ThemeMode;
  /** Creator-selected Studio Heat accent. Local-only presentation preference. */
  accentTheme: AccentTheme;
}

// Settings mirrored to the backend's settings table (plan 3.1). apiEndpoint
// stays local-only — persisting it server-side would be circular.
const PERSISTED_SETTING_KEYS = [
  "displayName",
  "processingMode",
  "performanceProfile",
  "exportLayout",
  "captionStyle",
  "recallStorageCapGb",
  "sourceRetentionDays",
  "theme",
] as const;

// Signals arrive from the backend as a JSON-encoded string; tolerate nulls and
// legacy rows that never stored them.
const parseSignals = (raw: unknown): string[] | undefined => {
  if (!raw) return undefined;
  if (Array.isArray(raw)) return raw.length ? raw : undefined;
  try {
    const parsed = JSON.parse(raw as string);
    return Array.isArray(parsed) && parsed.length ? parsed : undefined;
  } catch {
    return undefined;
  }
};

// modality_breakdown arrives as a JSON-encoded string from /clips (raw column)
// but already parsed from /clips/{id}/edit; tolerate both plus legacy nulls.
const parseBreakdown = (raw: unknown): Record<string, number> | undefined => {
  const obj = (() => {
    if (!raw) return undefined;
    if (typeof raw === "object") return raw as Record<string, number>;
    try { return JSON.parse(raw as string); } catch { return undefined; }
  })();
  if (!obj || typeof obj !== "object") return undefined;
  return Object.values(obj).some((v) => typeof v === "number" && v > 0) ? obj : undefined;
};

const parseFeatures = (raw: unknown): Record<string, number> | undefined => {
  const obj = (() => {
    if (!raw) return undefined;
    if (typeof raw === "object") return raw as Record<string, unknown>;
    try { return JSON.parse(raw as string) as Record<string, unknown>; } catch { return undefined; }
  })();
  if (!obj || typeof obj !== "object") return undefined;
  const numeric = Object.fromEntries(
    Object.entries(obj).filter((entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1])),
  );
  return Object.keys(numeric).length ? numeric : undefined;
};

const parseRecallProvenance = (raw: unknown): RecallProvenance | undefined => {
  const value = (() => {
    if (!raw) return undefined;
    if (typeof raw === "object") return raw as Record<string, unknown>;
    try { return JSON.parse(raw as string) as Record<string, unknown>; } catch { return undefined; }
  })();
  if (!value) return undefined;
  if (value.source === "stream_memory" && typeof value.memory_entry_id === "string") {
    return {
      source: "stream_memory",
      origin: "memory_result",
      memoryEntryId: value.memory_entry_id,
      entryKind: value.entry_kind === "transcript" || value.entry_kind === "clip" || value.entry_kind === "evidence"
        ? value.entry_kind : undefined,
      query: typeof value.query === "string" ? value.query : undefined,
      sourceStart: typeof value.source_start === "number" ? value.source_start : undefined,
      sourceEnd: typeof value.source_end === "number" ? value.source_end : undefined,
    };
  }
  if (value.source !== "recall_session") return undefined;
  const markerIds = Array.isArray(value.marker_ids)
    ? value.marker_ids.filter((id): id is string => typeof id === "string")
    : [];
  const creatorProtected = value.creator_protected === true;
  const origin = value.origin === "creator_marker" || markerIds.length || creatorProtected
    ? "creator_marker"
    : "session_scan";
  return {
    source: "recall_session",
    sessionId: typeof value.session_id === "string" ? value.session_id : undefined,
    origin,
    creatorProtected,
    markerIds,
    markerTime: typeof value.marker_time === "number" && Number.isFinite(value.marker_time)
      ? value.marker_time
      : undefined,
  };
};

/** Map a raw clip row (from GET /clips, POST /edit, or PATCH) into the frontend
 * Clip shape. Single source of truth — the initial load and every edit response
 * flow through here so snake_case never leaks and no mapper drifts. */
export const toClip = (c: any, apiEndpoint: string): Clip => {
  const start = c.start_time || 0;
  const isOverflowCandidate = c.story_label === "recall_more_candidate";
  const isManual = c.story_label === "manual";
  const hookScore = isManual
    ? undefined
    : typeof c.hook_score === "number"
      ? c.hook_score
      : c.deck_score != null && typeof c.score === "number"
        ? c.score
        : undefined;
  // Trust the backend's on-disk verdict when it sent one. Older responses (and
  // tests) omit media_state; those fall back to the historical assumption that
  // a set export_path means a playable file.
  const mediaState: ClipMediaState = c.media_state
    ?? (c.export_path ? "ready" : isOverflowCandidate ? "source_window" : "unavailable");
  const mediaPresent = mediaState === "ready";
  let videoUrl: string | undefined;
  if (c.export_path && mediaPresent) {
    const filename = c.export_path.replace(/\\/g, "/").split("/").pop();
    videoUrl = mediaUrl(filename, apiEndpoint);
  } else if (typeof c.videoUrl === "string") {
    videoUrl = c.videoUrl;
  } else if (isOverflowCandidate && c.job_id && mediaState === "source_window") {
    // Fallback only: modern scans bake framed Second-look previews into the
    // same export pass as the primary deck. Legacy rows (or a failed encode)
    // fall back to seeking the source VOD window.
    videoUrl = apiUrl(`/jobs/${c.job_id}/source`, apiEndpoint);
  }
  if (videoUrl) {
    const cacheBuster = `t=${c.start_time ?? 0}_${c.end_time ?? 0}`;
    videoUrl = videoUrl.includes("?") ? `${videoUrl}&${cacheBuster}` : `${videoUrl}?${cacheBuster}`;
  }
  // Every clip with a rendered MP4 can produce a poster; the endpoint
  // self-heals (generates on first request) so we don't need thumb_path here.
  // But only when the MP4 is actually there -- pointing a hundred cards at a
  // guaranteed 404 is what made a media-less library render blank.
  let thumbUrl = mediaPresent && c.export_path
    ? apiUrl(`/clips/${c.id}/thumb`, apiEndpoint)
    : undefined;
  if (thumbUrl) {
    const cacheBuster = `t=${c.start_time ?? 0}_${c.end_time ?? 0}`;
    thumbUrl = thumbUrl.includes("?") ? `${thumbUrl}&${cacheBuster}` : `${thumbUrl}?${cacheBuster}`;
  }
  return {
    id: c.id,
    title: c.title || "Untitled moment",
    clipNumber: typeof c.clip_number === "number" ? c.clip_number : undefined,
    originalStem: typeof c.export_path === "string"
      ? c.export_path.replace(/\\/g, "/").split("/").pop()?.replace(/\.[^.]+$/, "")
      : undefined,
    // Hour-aware ("1:22:07", not "82:07") — shared formatter (plan 22 §3.0).
    timestamp: fmtClock(start),
    confidence: Math.round((hookScore ?? 0) * 100),
    deckScore: Math.round((c.deck_score ?? c.score ?? 0.8) * 100),
    selectionScore: typeof c.selection_score === "number" ? c.selection_score : undefined,
    duration: Math.round((c.end_time || 0) - (c.start_time || 0)),
    start_time: c.start_time,
    end_time: c.end_time,
    videoUrl,
    thumbUrl,
    description: c.description || undefined,
    signals: parseSignals(c.signals),
    postTags: parseSignals(c.tags),
    reason: c.reason || undefined,
    peakTimestamp: typeof c.peak_timestamp === "number" ? c.peak_timestamp : undefined,
    modalityBreakdown: parseBreakdown(c.modality_breakdown),
    features: parseFeatures(c.features),
    kept: !!c.kept,
    passed: !!c.passed,
    maybe: !!c.maybe,
    exported: !!c.exported_at,
    scene: c.scene || undefined,
    sceneLabel: c.scene ? SCENE_DISPLAY[c.scene] ?? "Lobby / Just chatting" : undefined,
    hookLine: c.hook_line || undefined,
    momentType: c.moment_type || undefined,
    isOverflowCandidate,
    isManual,
    sourceWindow: mediaState === "source_window",
    mediaState,
    recallProvenance: parseRecallProvenance(c.recall_provenance),
  };
};

/** Whether GET /jobs/{id}/thumb has anything left to grab a frame from. The
 * endpoint prefers the source VOD and falls back to a clip export, so a session
 * with neither can only 404 — don't ask. Missing summaries (older API, tests)
 * keep the previous always-ask behaviour. */
const canPosterFromDisk = (media: SessionMedia | undefined): boolean =>
  !media || media.source_state === "local" || media.ready > 0;

const timelineCache = new Map<string, ReactionTimeline | null>();

const terminalJobStatus = (status: JobStatus) =>
  status === "completed" || status === "failed" || status === "cancelled";

function sessionStatusFromBackend(status: string | undefined, progress?: number): JobStatus {
  const progressPct = Math.max(0, Math.min(100, Math.round(Number(progress || 0) * 100)));
  return mapJobStatus(status, progressPct);
}

function backendTimestampMs(value: unknown): number {
  if (typeof value === "number") return value > 10_000_000_000 ? value : value * 1000;
  if (typeof value === "string" && value) {
    const normalized = value.includes("T") ? value : `${value.replace(" ", "T")}Z`;
    const parsed = Date.parse(normalized);
    if (!Number.isNaN(parsed)) return parsed;
  }
  return Date.now();
}

/** Build the live Job lane from a list/detail/SSE backend snapshot. */
function jobFromSnapshot(data: any, existing?: Job): Job {
  const progress = Math.max(0, Math.min(100, Math.round(Number(data.progress || 0) * 100)));
  const stageProgress = data.stage_progress == null
    ? progress
    : Math.max(0, Math.min(100, Math.round(Number(data.stage_progress) * 100)));
  return {
    ...existing,
    id: String(data.id ?? existing?.id ?? ""),
    url: String(data.source_path ?? existing?.url ?? ""),
    status: mapJobStatus(data.status, progress),
    progress,
    stage: {
      label: data.stage_label || data.current_stage || existing?.stage.label || "Preparing scan",
      progress: stageProgress,
    },
    createdAt: backendTimestampMs(data.created_at ?? existing?.createdAt),
    ...telemetryFromSnapshot(data),
  };
}

/** Fetch (and memoize) the VOD-level reaction curve for a job. Returns null
 * for legacy-engine jobs or when the backend is offline. */
export async function fetchReactionTimeline(jobId: string): Promise<ReactionTimeline | null> {
  if (timelineCache.has(jobId)) return timelineCache.get(jobId) ?? null;
  try {
    const res = await apiFetch(`/jobs/${jobId}/timeline`);
    if (!res.ok) return null;
    const data = await res.json();
    const timeline: ReactionTimeline | null =
      data?.timeline && Array.isArray(data.timeline.r) && data.timeline.r.length > 1
        ? data.timeline
        : null;
    if (timeline) timelineCache.set(jobId, timeline);
    return timeline;
  } catch {
    return null; // backend offline — don't cache, retry next time
  }
}

const readStored = <T extends string>(key: string, fallback: T): T => {
  try {
    const v = typeof localStorage !== "undefined" && localStorage.getItem(key);
    return (v as T) || fallback;
  } catch {
    return fallback;
  }
};

const readStoredNumber = (key: string, fallback: number, min: number, max: number): number => {
  try {
    const raw = typeof localStorage !== "undefined" && localStorage.getItem(key);
    const value = raw == null ? fallback : Number(raw);
    return Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback;
  } catch {
    return fallback;
  }
};

const readStoredBoolean = (key: string, fallback: boolean): boolean => {
  try {
    const raw = typeof localStorage !== "undefined" && localStorage.getItem(key);
    return raw == null ? fallback : raw === "true";
  } catch {
    return fallback;
  }
};

// Read a JSON-encoded object from localStorage, merged over the fallback so a
// partial or stale shape (e.g. a caption style saved before a field existed)
// never leaves required keys undefined.
const readStoredJson = <T extends object>(key: string, fallback: T): T => {
  try {
    const raw = typeof localStorage !== "undefined" && localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? { ...fallback, ...parsed } : fallback;
  } catch {
    return fallback;
  }
};

const persistBatchJobIds = (jobIds: string[]) => {
  try {
    localStorage.setItem("recall-latest-batch-job-ids", JSON.stringify(jobIds));
  } catch {
    // Storage can be unavailable in private/test contexts; the live store
    // still owns the queue for the current desktop session.
  }
};

const readStoredBatchJobIds = (): string[] => {
  try {
    const raw = typeof localStorage !== "undefined" && localStorage.getItem("recall-latest-batch-job-ids");
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
};

/** True while the queue is showing a batch restored from a PREVIOUS app run.
 *
 * The Scan queue tracks one live batch. Persisting its ids is what lets an
 * overnight run survive a restart, but nothing ever retired them, so a queue
 * finished days ago still greeted the creator on launch. Ownership tells the
 * two cases apart: a batch this run started stays put no matter what, while a
 * restored one clears itself once every scan in it has reached a terminal
 * state (see StudioApp's retire effect). */
const batchRestoredFromDisk = readStoredBatchJobIds().length > 0;

// Legacy (pre-3.2) review state lived in per-browser localStorage. These
// helpers only exist to migrate old keys into the DB, then delete them.
const readLegacyIds = (key: string): string[] => {
  try {
    const raw = typeof localStorage !== "undefined" && localStorage.getItem(key);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
};

const removeLegacyKey = (key: string) => {
  try {
    if (typeof localStorage !== "undefined") localStorage.removeItem(key);
  } catch {}
};

/** One-time push of a session's legacy localStorage review state into the DB
 * (plan 3.2). Keys are removed only after the backend accepts the batch. */
async function migrateLegacyReviewState(
  sessionId: string,
  clipIds: Set<string>,
  apiEndpoint: string,
): Promise<{ saved: string[]; exported: string[]; passed: string[] }> {
  const pick = (key: string) => readLegacyIds(key).filter((id) => clipIds.has(id));
  const saved = pick(`recall-saved-${sessionId}`);
  const exported = pick(`recall-exported-${sessionId}`);
  const passed = pick(`recall-passed-${sessionId}`);
  const batches: Array<[string, string[], Record<string, boolean>]> = [
    [`recall-saved-${sessionId}`, saved, { kept: true }],
    [`recall-exported-${sessionId}`, exported, { exported: true }],
    [`recall-passed-${sessionId}`, passed, { passed: true }],
  ];
  for (const [key, ids, flags] of batches) {
    if (!readLegacyIds(key).length) continue;
    try {
      if (ids.length) {
        const res = await apiFetch("/clips/state-batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ clip_ids: ids, ...flags }),
        }, apiEndpoint);
        if (!res.ok) continue;
      }
      removeLegacyKey(key);
    } catch {
      // Backend hiccup — leave the key; we'll retry next session load.
    }
  }
  return { saved, exported, passed };
}

interface JobStore {
  currentJob?: Job;
  jobs: Job[];
  logs: LogEntry[];
  settings: Settings;
  learningStatus?: LearningStatus;
  learningLoading: boolean;
  exportOperation?: ExportOperation;

  // Session state layer
  currentSessionId?: string;
  sessions: Session[];
  /** Job ids from the latest overnight batch enqueue (queue view focus). */
  batchJobIds: string[];
  /** The shown batch came from localStorage, not from this run's enqueue. */
  batchRestored: boolean;

  startJob: (
    url: string,
    name?: string,
    options?: {
      vtuberMode?: "off" | "confirmed";
      sourceDate?: string;
      sourceType?: "file" | "twitch";
      recallSessionId?: string;
      vodStartedAtUtc?: string;
    },
  ) => Promise<void>;
  startJobBatch: (items: Array<{
    url: string;
    name: string;
    vtuberMode?: "off" | "confirmed";
    sourceDate?: string;
  }>) => Promise<string[]>;
  setBatchJobIds: (ids: string[]) => void;
  clearBatchJobIds: () => void;
  updateJob: (id: string, patch: Partial<Job>) => void;
  addLog: (text: string, type?: "info" | "success" | "warn") => void;
  clearLogs: () => void;
  setSettings: (patch: Partial<Settings>) => void;
  loadRemoteSettings: () => Promise<void>;
  loadLearningStatus: () => Promise<void>;
  resetLearning: () => Promise<void>;
  clearScanCache: () => Promise<void>;

  setCurrentSessionId: (id?: string) => void;
  loadSessions: () => Promise<void>;
  refreshJobs: () => Promise<void>;
  hydrateSessionClips: (sessionId: string) => Promise<void>;
  loadMoreCandidates: (sessionId: string) => Promise<MoreCandidateResult>;
  prepareClipPreview: (sessionId: string, clipId: string) => Promise<Clip>;
  deleteSession: (sessionId: string) => Promise<void>;
  deleteClips: (sessionId: string, clipIds: string[]) => Promise<void>;
  copyClipsToFolder: (
    clipIds: string[],
    destFolder: string,
    opts?: { preset?: ExportPreset; filenameTemplate?: string },
  ) => Promise<ExportResult>;
  compileReel: (
    clipIds: string[],
    destFolder: string,
    filename?: string,
  ) => Promise<CompileReelResult>;
  cancelExportOperation: () => Promise<void>;
  toggleSaveClip: (sessionId: string, clipId: string) => void;
  setClipPassed: (sessionId: string, clipId: string, passed: boolean) => void;
  setClipMaybe: (sessionId: string, clipId: string, maybe: boolean) => void;
  editClip: (sessionId: string, clipId: string, options: { start_time: number; end_time: number; fade_in?: number; fade_out?: number; video_fade_in?: number; video_fade_out?: number; audio_fade_in?: number; audio_fade_out?: number }) => Promise<Clip>;
  resolveManualFraming: (sessionId: string, options: ManualClipOptions) => Promise<ManualFraming>;
  previewManualClip: (sessionId: string, options: ManualClipOptions) => Promise<Blob>;
  createManualClip: (sessionId: string, options: ManualClipOptions) => Promise<Clip>;
  updateClipTitle: (sessionId: string, clipId: string, title: string) => Promise<Clip>;
}

function exportedStatePatch(state: JobStore, succeeded: Set<string>) {
  return {
    sessions: state.sessions.map((session) => {
      const exportedInSession = session.clips
        .filter((clip) => succeeded.has(clip.id))
        .map((clip) => clip.id);
      if (!exportedInSession.length) return session;
      return {
        ...session,
        exportedClipIds: [...new Set([
          ...session.exportedClipIds,
          ...exportedInSession,
        ])],
        clips: session.clips.map((clip) => (
          succeeded.has(clip.id) ? { ...clip, exported: true } : clip
        )),
        updatedAt: Date.now(),
      };
    }),
    currentJob: state.currentJob
      ? {
          ...state.currentJob,
          clips: (state.currentJob.clips || []).map((clip) => (
            succeeded.has(clip.id) ? { ...clip, exported: true } : clip
          )),
        }
      : state.currentJob,
  };
}

export const useJobStore = create<JobStore>((set, get) => ({
  jobs: [],
  currentJob: undefined,
  logs: [],
  learningStatus: undefined,
  learningLoading: false,
  exportOperation: undefined,
  settings: {
    displayName: readStored("recall-display-name", ""),
    processingMode: "quality",
    performanceProfile: "balanced",
    exportLayout: readStored<ExportLayout>("recall-export-layout", "auto"),
    captionStyle: readStoredJson<CaptionStyle>("recall-caption-style", DEFAULT_CAPTION_STYLE),
    exportPreset: readStored<ExportPreset>("recall-export-preset", "none"),
    filenameTemplate: readStored("recall-filename-template", ""),
    recallStorageCapGb: 25,
    sourceRetentionDays: 7,
    mediaVolume: readStoredNumber("recall-media-volume", 0.35, 0, 1),
    mediaMuted: readStoredBoolean("recall-media-muted", false),
    keepAwakeWhileWorking: readStoredBoolean("recall-keep-awake-working", true),
    desktopNotifications: readStoredBoolean("recall-desktop-notifications", true),
    apiEndpoint: DEFAULT_API_ENDPOINT,
    theme: readStored<ThemeMode>("recall-theme", "dark"),
    accentTheme: readStored<AccentTheme>("recall-accent-theme", "ember"),
  },

  currentSessionId: undefined,
  sessions: [],
  batchJobIds: readStoredBatchJobIds(),
  batchRestored: batchRestoredFromDisk,

  setCurrentSessionId: (currentSessionId) => set({ currentSessionId }),
  setBatchJobIds: (batchJobIds) => {
    persistBatchJobIds(batchJobIds);
    // Anything this run enqueues is ours to keep on screen until it finishes.
    set({ batchJobIds, batchRestored: false });
  },
  clearBatchJobIds: () => {
    persistBatchJobIds([]);
    set({ batchJobIds: [], batchRestored: false });
  },

  // Rehydrate sessions from the backend so completed work survives restarts.
  loadSessions: async () => {
    try {
      const [jobsRes, clipsRes] = await Promise.all([
        apiFetch("/jobs", undefined, get().settings.apiEndpoint),
        apiFetch("/clips", undefined, get().settings.apiEndpoint),
      ]);
      if (!jobsRes.ok || !clipsRes.ok) return;

      const rawJobs = await jobsRes.json();
      const rawClips = await clipsRes.json();

      // SQLite CURRENT_TIMESTAMP is UTC "YYYY-MM-DD HH:MM:SS"
      const parseTs = (ts?: string) => {
        if (!ts) return Date.now();
        const ms = Date.parse(ts.replace(" ", "T") + "Z");
        return Number.isNaN(ms) ? Date.now() : ms;
      };

      const clipsByJob = new Map<string, Clip[]>();
      for (const c of rawClips) {
        const clip = toClip(c, get().settings.apiEndpoint);
        const list = clipsByJob.get(c.job_id) || [];
        list.push(clip);
        clipsByJob.set(c.job_id, list);
      }

      const sessions: Session[] = rawJobs.map((j: any) => {
        const createdAt = parseTs(j.created_at);
        const updatedAt = parseTs(j.updated_at) || createdAt;
        const sessionStatus = sessionStatusFromBackend(j.status, j.progress);
        const clips = clipsByJob.get(j.id) || [];
        // Review state comes from the DB (plan 3.2). Legacy localStorage keys
        // are merged in for this render and pushed to the backend once.
        const clipIdSet = new Set(clips.map((c) => c.id));
        const legacySaved = readLegacyIds(`recall-saved-${j.id}`).filter((id) => clipIdSet.has(id));
        const legacyExported = readLegacyIds(`recall-exported-${j.id}`).filter((id) => clipIdSet.has(id));
        const savedClipIds = [...new Set([...clips.filter((c) => c.kept).map((c) => c.id), ...legacySaved])];
        const exportedClipIds = [...new Set([...clips.filter((c) => c.exported).map((c) => c.id), ...legacyExported])];
        void migrateLegacyReviewState(j.id, clipIdSet, get().settings.apiEndpoint);
        const apiEndpoint = get().settings.apiEndpoint;
        return {
          id: j.id,
          name: j.session_name || (j.source_path || "").replace(/\\/g, "/").split("/").pop() || "Untitled Session",
          sourceUrl: j.source_path || "",
          sourceDate: j.source_date || undefined,
          recallSessionId: j.recall_session_id || undefined,
          createdAt,
          updatedAt,
          status: sessionStatus,
          savedClipIds,
          exportedClipIds,
          clips,
          clipsHydrated: terminalJobStatus(sessionStatus),
          vodDuration: j.duration || undefined,
          message: j.message || undefined,
          // Landscape VOD frame for session cards — generated lazily by the API
          // from the source VOD, or failing that from a clip export. Skipped
          // when neither is left, so a media-less gallery doesn't fire a
          // request per card that can only 404.
          posterUrl: canPosterFromDisk(j.media)
            ? apiUrl(`/jobs/${j.id}/thumb`, apiEndpoint)
            : undefined,
          media: j.media || undefined,
        };
      });

      const previousJobs = get().jobs;
      const activeJobs: Job[] = rawJobs
        .filter((job: any) => ["pending", "running", "cancelling"].includes(job.status))
        .map((job: any) => jobFromSnapshot(
          job,
          previousJobs.find((existing) => existing.id === job.id),
        ));
      const activeIds = new Set(activeJobs.map((job) => job.id));
      const rawIds = new Set(rawJobs.map((job: any) => job.id));
      const localOnlyJobs = previousJobs.filter((job) =>
        !rawIds.has(job.id) && !terminalJobStatus(job.status),
      );
      const jobs = [...activeJobs, ...localOnlyJobs];
      const preferredLiveJob = activeJobs.find((job) => job.status !== "queued") ?? activeJobs[0];

      set((state) => ({
        sessions,
        jobs,
        currentJob: state.currentJob && activeIds.has(state.currentJob.id)
          ? jobs.find((job) => job.id === state.currentJob?.id)
          : preferredLiveJob,
      }));
      reconcileJobSubscriptions(preferredLiveJob?.id);
    } catch {
      // Backend offline — leave sessions as-is.
    }
  },

  /** Refresh the live queue without deserializing the historical clip library. */
  refreshJobs: async () => {
    try {
      const response = await apiFetch("/jobs", undefined, get().settings.apiEndpoint);
      if (!response.ok) return;
      const rawJobs = await response.json();
      if (!Array.isArray(rawJobs)) return;

      const previousJobs = get().jobs;
      const previousIds = new Set(previousJobs.map((job) => job.id));
      const refreshedJobs: Job[] = rawJobs
        .filter((job: any) =>
          ["pending", "running", "cancelling"].includes(job.status) || previousIds.has(job.id),
        )
        .map((job: any) => jobFromSnapshot(
          job,
          previousJobs.find((existing) => existing.id === job.id),
        ));
      const activeJobs = refreshedJobs.filter((job) => !terminalJobStatus(job.status));
      const jobs = [
        ...refreshedJobs,
        ...previousJobs.filter((job) =>
          !rawJobs.some((raw: any) => raw.id === job.id) && !terminalJobStatus(job.status),
        ),
      ];
      const preferredLiveJob = activeJobs.find((job) => job.status !== "queued") ?? activeJobs[0];
      const newlyCompletedIds = refreshedJobs
        .filter((job) => job.status === "completed")
        .filter((job) => {
          const previous = previousJobs.find((candidate) => candidate.id === job.id);
          return previous && !terminalJobStatus(previous.status);
        })
        .map((job) => job.id);

      set((state) => ({
        jobs,
        sessions: state.sessions.map((session) => {
          const raw = rawJobs.find((job: any) => job.id === session.id);
          if (!raw) return session;
          const nextStatus = sessionStatusFromBackend(raw.status, raw.progress);
          const completedNow = nextStatus === "completed" && !terminalJobStatus(session.status);
          return {
            ...session,
            status: nextStatus,
            clipsHydrated: completedNow ? false : session.clipsHydrated,
            message: raw.message ?? session.message,
            vodDuration: raw.duration ?? session.vodDuration,
            updatedAt: backendTimestampMs(raw.updated_at ?? session.updatedAt),
            // Carried through so a finished rebuild actually clears the
            // "previews were cleared" banner instead of leaving it asserting a
            // gap the session no longer has.
            media: raw.media ?? session.media,
            posterUrl: canPosterFromDisk(raw.media)
              ? session.posterUrl ?? apiUrl(`/jobs/${session.id}/thumb`, get().settings.apiEndpoint)
              : undefined,
          };
        }),
        currentJob: state.currentJob && jobs.some((job) => job.id === state.currentJob?.id)
          ? jobs.find((job) => job.id === state.currentJob?.id)
          : preferredLiveJob,
      }));
      reconcileJobSubscriptions(preferredLiveJob?.id);
      await Promise.all(newlyCompletedIds.map((jobId) => loadJobClips(jobId)));
    } catch {
      // Recovery polling is best-effort. Existing live state stays visible.
    }
  },

  hydrateSessionClips: async (sessionId) => {
    await loadJobClips(sessionId);
  },

  loadMoreCandidates: async (sessionId) => {
    const response = await apiFetch(
      `/jobs/${sessionId}/clips/more`,
      { method: "POST" },
      get().settings.apiEndpoint,
    );
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || "Recall could not load more moments.");
    }
    const data = await response.json();
    const clips: Clip[] = Array.isArray(data.clips)
      ? data.clips.map((clip: any) => toClip(clip, get().settings.apiEndpoint))
      : [];
    set((state) => ({
      sessions: state.sessions.map((session) => {
        if (session.id !== sessionId) return session;
        const merged = new Map(session.clips.map((clip) => [clip.id, clip]));
        clips.forEach((clip) => merged.set(clip.id, clip));
        return { ...session, clips: [...merged.values()] };
      }),
      currentJob: state.currentJob?.id === sessionId
        ? {
            ...state.currentJob,
            clips: (() => {
              const merged = new Map((state.currentJob?.clips || []).map((clip) => [clip.id, clip]));
              clips.forEach((clip) => merged.set(clip.id, clip));
              return [...merged.values()];
            })(),
          }
        : state.currentJob,
    }));
    return {
      available: Number(data.available || 0),
      available_total: Number(data.available_total || 0),
      loaded: Number(data.loaded || clips.length),
      clips,
    };
  },

  prepareClipPreview: async (sessionId, clipId) => {
    const response = await apiFetch(
      `/clips/${clipId}/preview`,
      { method: "POST" },
      get().settings.apiEndpoint,
    );
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || "Recall could not prepare this framed preview.");
    }
    const prepared = toClip(await response.json(), get().settings.apiEndpoint);
    set((state) => ({
      sessions: state.sessions.map((session) => session.id === sessionId
        ? { ...session, clips: session.clips.map((clip) => clip.id === clipId ? prepared : clip) }
        : session),
      currentJob: state.currentJob?.id === sessionId
        ? {
            ...state.currentJob,
            clips: (state.currentJob.clips || []).map((clip) => clip.id === clipId ? prepared : clip),
          }
        : state.currentJob,
    }));
    return prepared;
  },

  deleteClips: async (sessionId, clipIds) => {
    try {
      await apiFetch("/clips/delete-batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clip_ids: clipIds }),
      }, get().settings.apiEndpoint);
      get().addLog("Recall will show fewer moments like those.", "info");
      get().loadLearningStatus();
    } catch {}
    set((state) => ({
      sessions: state.sessions.map((s) => {
        if (s.id !== sessionId) return s;
        const nextSaved = s.savedClipIds.filter((id) => !clipIds.includes(id));
        const nextExported = s.exportedClipIds.filter((id) => !clipIds.includes(id));
        return {
          ...s,
          clips: s.clips.filter((c) => !clipIds.includes(c.id)),
          savedClipIds: nextSaved,
          exportedClipIds: nextExported,
        };
      }),
      currentJob:
        state.currentJob?.id === sessionId
          ? { ...state.currentJob, clips: (state.currentJob.clips || []).filter((c) => !clipIds.includes(c.id)) }
          : state.currentJob,
    }));
  },

  copyClipsToFolder: async (clipIds, destFolder, opts) => {
    const preset = opts?.preset && opts.preset !== "none" ? opts.preset : undefined;
    const filename_template = opts?.filenameTemplate?.trim() || undefined;
    const current = get().exportOperation;
    if (current && !EXPORT_TERMINAL_STATUSES.has(current.status)) {
      return {
        copied: 0,
        errors: clipIds.length,
        succeededClipIds: [],
        failed: [],
        error: "Another export is already preparing final files.",
      };
    }

    try {
      const res = await apiFetch("/clips/export-operations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clip_ids: clipIds, dest_folder: destFolder, preset, filename_template }),
      }, get().settings.apiEndpoint);
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        let operation = toExportOperation(data, clipIds.length);
        if (!operation.id) throw new Error("missing export operation id");
        set({ exportOperation: operation });
        const polled = await pollExportOperation(
          operation,
          clipIds.length,
          get().settings.apiEndpoint,
          (update) => set({ exportOperation: update }),
        );
        operation = polled.operation;
        if (polled.lost) {
          return {
            copied: operation.copied,
            errors: Math.max(operation.errors, clipIds.length - operation.copied),
            succeededClipIds: operation.succeededClipIds,
            failed: operation.failed,
            error: "Recall lost the live export status. Finished files remain safe in the selected folder.",
          };
        }

        const succeeded = new Set(operation.succeededClipIds);
        if (succeeded.size) {
          set((state) => exportedStatePatch(state, succeeded));
        }
        return {
          copied: operation.copied,
          errors: operation.errors,
          succeededClipIds: operation.succeededClipIds,
          failed: operation.failed,
          cancelled: operation.status === "cancelled",
          error: operation.status === "cancelled" ? operation.message : operation.error,
        };
      }
      const detail = data?.detail;
      return {
        copied: 0,
        errors: clipIds.length,
        succeededClipIds: [],
        failed: [],
        error: typeof detail === "string" ? detail : "Recall could not prepare the final-quality files.",
      };
    } catch {}
    return {
      copied: 0,
      errors: clipIds.length,
      succeededClipIds: [],
      failed: [],
      error: "Recall could not reach the local studio.",
    };
  },

  compileReel: async (clipIds, destFolder, filename) => {
    const current = get().exportOperation;
    if (current && !EXPORT_TERMINAL_STATUSES.has(current.status)) {
      return { error: "Another export is already preparing final files." };
    }
    try {
      const res = await apiFetch("/clips/reel-operations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clip_ids: clipIds,
          dest_folder: destFolder,
          filename: filename?.trim() || undefined,
        }),
      }, get().settings.apiEndpoint);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        return { error: typeof data?.detail === "string" ? data.detail : data?.error || "Reel compile failed." };
      }
      let operation = toExportOperation(data, clipIds.length);
      if (!operation.id) throw new Error("missing reel operation id");
      set({ exportOperation: operation });
      const polled = await pollExportOperation(
        operation,
        clipIds.length,
        get().settings.apiEndpoint,
        (update) => set({ exportOperation: update }),
      );
      operation = polled.operation;
      if (polled.lost) {
        return {
          path: operation.path,
          clips: operation.clips,
          errors: operation.errors,
          succeededClipIds: operation.succeededClipIds,
          failed: operation.failed,
          error: "Recall lost the live reel status. Any finished file remains safe in the selected folder.",
        };
      }

      const succeeded = new Set(operation.succeededClipIds);
      if (succeeded.size) set((state) => exportedStatePatch(state, succeeded));
      return {
        path: operation.path,
        clips: operation.clips,
        errors: operation.errors,
        succeededClipIds: operation.succeededClipIds,
        failed: operation.failed,
        cancelled: operation.status === "cancelled",
        error: operation.status === "cancelled" ? operation.message : operation.error,
      };
    } catch {
      return { error: "Could not reach the engine to compile the reel." };
    }
  },

  cancelExportOperation: async () => {
    const operation = get().exportOperation;
    if (!operation || operation.cancelRequested || EXPORT_TERMINAL_STATUSES.has(operation.status)) return;
    try {
      const response = await apiFetch(
        `/clips/export-operations/${encodeURIComponent(operation.id)}/cancel`,
        { method: "POST" },
        get().settings.apiEndpoint,
      );
      if (!response.ok) throw new Error(`cancel export ${response.status}`);
      set({
        exportOperation: toExportOperation(
          await response.json(),
          operation.totalClips,
        ),
      });
    } catch {
      get().addLog("Recall could not stop the export. It is still running; try again.", "warn");
    }
  },

  deleteSession: async (sessionId) => {
    let response: Response;
    try {
      response = await apiFetch(`/jobs/${sessionId}`, { method: "DELETE" }, get().settings.apiEndpoint);
    } catch {
      throw new Error("Recall could not reach the local studio, so the session was not deleted.");
    }
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const detail = typeof payload?.detail === "string"
        ? payload.detail
        : "Recall could not delete this session.";
      throw new Error(detail);
    }
    for (const key of ["saved", "exported", "passed", "draft-meta"]) {
      removeLegacyKey(`recall-${key}-${sessionId}`);
    }
    const subscription = jobSubscriptions.get(sessionId);
    subscription?.();
    jobSubscriptions.delete(sessionId);
    const nextBatchJobIds = get().batchJobIds.filter((jobId) => jobId !== sessionId);
    persistBatchJobIds(nextBatchJobIds);
    set((state) => {
      const isActive = state.currentSessionId === sessionId;
      return {
        sessions: state.sessions.filter((s) => s.id !== sessionId),
        jobs: state.jobs.filter((job) => job.id !== sessionId),
        batchJobIds: nextBatchJobIds,
        currentSessionId: isActive ? undefined : state.currentSessionId,
        currentJob: state.currentJob?.id === sessionId ? undefined : state.currentJob,
      };
    });
    timelineCache.delete(sessionId);
  },

  toggleSaveClip: (sessionId, clipId) => {
    const session = get().sessions.find((s) => s.id === sessionId);
    const wasSaved = session?.savedClipIds.includes(clipId) ?? false;
    if (!wasSaved) {
      apiFetch(`/clips/${clipId}/label`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: 1, event: "saved" }),
      }, get().settings.apiEndpoint)
        .then(() => {
          get().addLog("Recall will look for more moments like this.", "success");
          get().loadLearningStatus();
        })
        .catch(() => {});
    }
    if (wasSaved) {
      // Un-keeping retracts the earlier keep signal. The ranker trains on the
      // latest label per clip, so repeated toggling can't pollute training —
      // the final state of the toggle is what sticks (plan 1.3).
      apiFetch(`/clips/${clipId}/label`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: null, event: "saved" }),
      }, get().settings.apiEndpoint)
        .then(() => get().loadLearningStatus())
        .catch(() => {});
    }
    // Persist the keep flag server-side (plan 3.2); local state is optimistic.
    apiFetch(`/clips/${clipId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kept: !wasSaved }),
    }, get().settings.apiEndpoint).catch(() => {});
    set((state) => ({
      sessions: state.sessions.map((s) => {
        if (s.id === sessionId) {
          const exists = s.savedClipIds.includes(clipId);
          const nextSaved = exists
            ? s.savedClipIds.filter((id) => id !== clipId)
            : [...s.savedClipIds, clipId];
          return {
            ...s,
            savedClipIds: nextSaved,
            clips: s.clips.map((c) => (c.id === clipId ? { ...c, kept: !exists } : c)),
            updatedAt: Date.now(),
          };
        }
        return s;
      })
    }));
  },

  setClipPassed: (sessionId, clipId, passed) => {
    apiFetch(`/clips/${clipId}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: passed ? 0 : null, event: "passed" }),
    }, get().settings.apiEndpoint)
      .then(() => {
        if (passed) get().addLog("Recall will rank moments like this lower.", "info");
        get().loadLearningStatus();
      })
      .catch(() => {});
    apiFetch(`/clips/${clipId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ passed }),
    }, get().settings.apiEndpoint).catch(() => {});
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId
          ? { ...s, clips: s.clips.map((c) => (c.id === clipId ? { ...c, passed } : c)) }
          : s,
      ),
    }));
  },

  setClipMaybe: (sessionId, clipId, maybe) => {
    apiFetch(`/clips/${clipId}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: maybe ? 0.5 : null, event: "maybe" }),
    }, get().settings.apiEndpoint)
      .then(() => {
        if (maybe) get().addLog("Recall saved this as a close call, not a rejection.", "info");
        get().loadLearningStatus();
      })
      .catch(() => {});
    apiFetch(`/clips/${clipId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ maybe }),
    }, get().settings.apiEndpoint).catch(() => {});
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId
          ? { ...s, clips: s.clips.map((c) => (c.id === clipId ? { ...c, maybe } : c)) }
          : s,
      ),
    }));
  },

  editClip: async (sessionId, clipId, options) => {
    let res: Response;
    try {
      res = await apiFetch(`/clips/${clipId}/edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(options),
      }, get().settings.apiEndpoint);
    } catch {
      throw new Error("network");
    }
    if (!res.ok) {
      let detail = "";
      try {
        detail = (await res.json())?.detail ?? "";
      } catch {}
      const err = new Error(detail || `Edit failed (HTTP ${res.status})`);
      (err as Error & { status?: number }).status = res.status;
      throw err;
    }
    const updatedClip: Clip = toClip(await res.json(), get().settings.apiEndpoint);
    set((state) => ({
      sessions: state.sessions.map((s) => {
        if (s.id !== sessionId) return s;
        return {
          ...s,
          clips: s.clips.map((c) => (c.id === clipId ? updatedClip : c)),
        };
      }),
      currentJob:
        state.currentJob?.id === sessionId
          ? {
              ...state.currentJob,
              clips: (state.currentJob.clips || []).map((c) => (c.id === clipId ? updatedClip : c)),
            }
          : state.currentJob,
    }));
    return updatedClip;
  },

  // Cut a precise clip from the session's full source VOD (Cutting Room).
  // The backend renders it, files it with this job's clips, and records the
  // label=1 "manual" missed-positive — the supervision channel for moments
  // the engine didn't take on its own.
  resolveManualFraming: async (sessionId, options) => {
    const res = await apiFetch(`/jobs/${sessionId}/clips/manual/framing`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    }, get().settings.apiEndpoint);
    if (!res.ok) {
      let detail = "";
      try { detail = (await res.json())?.detail ?? ""; } catch {}
      throw new Error(detail || `Framing lookup failed (HTTP ${res.status})`);
    }
    return await res.json() as ManualFraming;
  },

  previewManualClip: async (sessionId, options) => {
    const res = await apiFetch(`/jobs/${sessionId}/clips/manual/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    }, get().settings.apiEndpoint);
    if (!res.ok) {
      let detail = "";
      try { detail = (await res.json())?.detail ?? ""; } catch {}
      throw new Error(detail || `Preview failed (HTTP ${res.status})`);
    }
    return await res.blob();
  },

  createManualClip: async (sessionId, options) => {
    let res: Response;
    try {
      res = await apiFetch(`/jobs/${sessionId}/clips/manual`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(options),
      }, get().settings.apiEndpoint);
    } catch {
      throw new Error("Recall couldn't reach the local studio. Make sure it's running.");
    }
    if (!res.ok) {
      let detail = "";
      try {
        const body = await res.json();
        detail = typeof body?.detail === "string" ? body.detail : "";
      } catch {}
      throw new Error(detail || `Clip creation failed (HTTP ${res.status})`);
    }
    const created: Clip = toClip(await res.json(), get().settings.apiEndpoint);
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId
          ? {
              ...s,
              clips: [...s.clips, created],
              savedClipIds: s.savedClipIds.includes(created.id)
                ? s.savedClipIds
                : [...s.savedClipIds, created.id],
              updatedAt: Date.now(),
            }
          : s,
      ),
      currentJob:
        state.currentJob?.id === sessionId
          ? { ...state.currentJob, clips: [...(state.currentJob.clips || []), created] }
          : state.currentJob,
    }));
    get().loadLearningStatus();
    return created;
  },

  updateClipTitle: async (sessionId, clipId, title) => {
    const res = await apiFetch(`/clips/${clipId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }, get().settings.apiEndpoint);
    if (!res.ok) throw new Error(`Rename failed (HTTP ${res.status})`);
    const updatedClip: Clip = toClip(await res.json(), get().settings.apiEndpoint);
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId
          ? { ...s, clips: s.clips.map((c) => (c.id === clipId ? updatedClip : c)) }
          : s,
      ),
      currentJob:
        state.currentJob?.id === sessionId
          ? {
              ...state.currentJob,
              clips: (state.currentJob.clips || []).map((c) => (c.id === clipId ? updatedClip : c)),
            }
          : state.currentJob,
    }));
    return updatedClip;
  },

  addLog: (text, type = "info") => {
    const timestamp = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    set((s) => ({
      logs: [{ id: ++logEntryId, timestamp, text, type }, ...s.logs].slice(0, 100), // Cap at 100 logs
    }));
  },
  clearLogs: () => set({ logs: [] }),
  setSettings: (patch) => {
    set((s) => ({ settings: { ...s.settings, ...patch } }));
    try {
      if (patch.mediaVolume !== undefined) localStorage.setItem("recall-media-volume", String(Math.max(0, Math.min(1, patch.mediaVolume))));
      if (patch.mediaMuted !== undefined) localStorage.setItem("recall-media-muted", String(patch.mediaMuted));
      if (patch.keepAwakeWhileWorking !== undefined) localStorage.setItem("recall-keep-awake-working", String(patch.keepAwakeWhileWorking));
      if (patch.desktopNotifications !== undefined) localStorage.setItem("recall-desktop-notifications", String(patch.desktopNotifications));
    } catch {}
    // Persist scan preferences server-side (plan 3.1) so they survive
    // restarts and agree across the desktop app and browser tabs.
    const persisted: Record<string, unknown> = {};
    for (const key of PERSISTED_SETTING_KEYS) {
      if (key in patch) persisted[key] = (patch as Record<string, unknown>)[key];
    }
    if (Object.keys(persisted).length) {
      apiFetch("/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(persisted),
      }, get().settings.apiEndpoint).catch(() => {});
    }
  },

  loadRemoteSettings: async () => {
    try {
      const res = await apiFetch("/settings", undefined, get().settings.apiEndpoint);
      if (!res.ok) return;
      const data = await res.json();
      const patch: Partial<Settings> = {};
      for (const key of PERSISTED_SETTING_KEYS) {
        if (data[key] !== undefined) (patch as Record<string, unknown>)[key] = data[key];
      }
      if (Object.keys(patch).length) {
        set((s) => ({ settings: { ...s.settings, ...patch } }));
      }
    } catch {
      // Studio offline — local defaults stay in effect.
    }
  },

  loadLearningStatus: async () => {
    set({ learningLoading: true });
    try {
      const res = await apiFetch("/ranker/status", undefined, get().settings.apiEndpoint);
      if (res.ok) {
        set({ learningStatus: await res.json() });
      }
    } catch {
      // Local studio may be offline; keep the last known state.
    } finally {
      set({ learningLoading: false });
    }
  },

  resetLearning: async () => {
    set({ learningLoading: true });
    try {
      const res = await apiFetch("/ranker/reset", { method: "POST" }, get().settings.apiEndpoint);
      if (res.ok) {
        set({ learningStatus: await res.json() });
      }
    } catch {
      // Best effort; the current status stays visible.
    } finally {
      set({ learningLoading: false });
    }
  },

  clearScanCache: async () => {
    try {
      const res = await apiFetch("/system/cache/clear", { method: "POST" }, get().settings.apiEndpoint);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json().catch(() => ({}));
      const mb = typeof data.bytes_removed === "number" ? data.bytes_removed / (1024 * 1024) : undefined;
      const suffix = mb !== undefined ? ` (${mb.toFixed(1)} MB)` : "";
      get().addLog(`Scan cache cleared${suffix}. New VOD scans will recompute analysis.`, "success");
    } catch {
      get().addLog("Could not clear scan cache. Make sure the local studio is running.", "warn");
      throw new Error("clear-cache-failed");
    }
  },

  startJob: async (url, name, options) => {
    get().clearLogs();
    get().addLog(`Starting your analysis...`, "info");
    get().addLog(`Recording selected: ${url}`, "info");
    get().setBatchJobIds([]);
    const jobId = await enqueueScanJob(url, name, {
      focus: true,
      vtuberMode: options?.vtuberMode,
      sourceDate: options?.sourceDate,
      sourceType: options?.sourceType,
      recallSessionId: options?.recallSessionId,
      vodStartedAtUtc: options?.vodStartedAtUtc,
    });
    if (!jobId) {
      get().addLog(`Unable to start analysis. Make sure the local studio is running.`, "warn");
    }
  },

  startJobBatch: async (items) => {
    const cleaned = items
      .map((item) => ({
        url: item.url.trim(),
        name: (item.name || "").trim().slice(0, 200),
        vtuberMode: item.vtuberMode === "confirmed" ? "confirmed" as const : "off" as const,
        sourceDate: item.sourceDate,
      }))
      .filter((item) => item.url);
    if (!cleaned.length) return [];

    get().clearLogs();
    get().addLog(
      `Queuing ${cleaned.length} VOD scan${cleaned.length === 1 ? "" : "s"} for overnight...`,
      "info",
    );

    const realIds: string[] = [];
    for (let index = 0; index < cleaned.length; index += 1) {
      const item = cleaned[index];
      const jobId = await enqueueScanJob(item.url, item.name || undefined, {
        focus: index === 0,
        logEach: true,
        vtuberMode: item.vtuberMode,
        sourceDate: item.sourceDate,
        sourceType: "twitch",
      });
      if (jobId) realIds.push(jobId);
    }

    get().setBatchJobIds(realIds);
    if (realIds.length) {
      get().addLog(
        `Queue ready · ${realIds.length} session${realIds.length === 1 ? "" : "s"} · one scan at a time`,
        "success",
      );
    } else {
      get().addLog("Could not queue any scans. Make sure the local studio is running.", "warn");
    }
    return realIds;
  },

  updateJob: (id, patch) =>
    set((state) => {
      const updatedJobs = state.jobs.map((j) => (j.id === id ? { ...j, ...patch } : j));
      const updatedSessions = state.sessions.map((s) => {
        if (s.id === id) {
          const updatedClips = patch.clips !== undefined ? patch.clips : s.clips;
          const updatedStatus = patch.status !== undefined ? patch.status : s.status;
          const completedNow = updatedStatus === "completed" && !terminalJobStatus(s.status);
          return {
            ...s,
            clips: updatedClips,
            clipsHydrated: patch.clips !== undefined
              ? true
              : completedNow
                ? false
                : s.clipsHydrated,
            status: updatedStatus,
            message: patch.message !== undefined ? patch.message : s.message,
            updatedAt: Date.now()
          };
        }
        return s;
      });
      return {
        jobs: updatedJobs,
        sessions: updatedSessions,
        currentJob:
          state.currentJob?.id === id
            ? { ...state.currentJob, ...patch }
            : state.currentJob,
      };
    }),
}));

// Map a raw backend /jobs/{id} snapshot onto our telemetry fields. Kept
// separate so both the SSE snapshot and the polling fallback stay in sync.
function telemetryFromSnapshot(data: any): Partial<Job> {
  const patch: Partial<Job> = {};
  if (data.current_stage !== undefined) patch.currentPhase = data.current_stage;
  if (data.message !== undefined && data.message !== null) patch.message = data.message;
  if (data.elapsed_seconds !== undefined && data.elapsed_seconds !== null) {
    patch.elapsedSeconds = data.elapsed_seconds;
    patch.elapsedSyncedAt = Date.now();
  }
  patch.etaSeconds = data.eta_seconds ?? null;
  if (data.eta_seconds !== undefined && data.eta_seconds !== null) patch.etaSyncedAt = Date.now();
  if (data.eta_confidence) patch.etaConfidence = data.eta_confidence as EtaConfidence;
  if (data.eta_reliable !== undefined) patch.etaReliable = Boolean(data.eta_reliable);
  if (data.scanned_seconds !== undefined && data.scanned_seconds !== null) patch.scannedSeconds = data.scanned_seconds;
  if (data.scan_speed !== undefined && data.scan_speed !== null) patch.scanSpeed = data.scan_speed;
  if (data.transcribed_seconds !== undefined && data.transcribed_seconds !== null) patch.transcribedSeconds = data.transcribed_seconds;
  if (data.transcription_speed !== undefined && data.transcription_speed !== null) patch.transcriptionSpeed = data.transcription_speed;
  if (data.active_workers !== undefined && data.active_workers !== null) patch.activeWorkers = data.active_workers;
  if (data.total_workers !== undefined && data.total_workers !== null) patch.totalWorkers = data.total_workers;
  if (data.download_progress !== undefined && data.download_progress !== null) patch.downloadProgress = data.download_progress;
  if (data.clips_found !== undefined && data.clips_found !== null) patch.clipsFound = data.clips_found;
  if (data.duration !== undefined && data.duration !== null) patch.vodDuration = data.duration;
  if (data.stage_timings !== undefined && data.stage_timings !== null) patch.stageTimings = data.stage_timings;
  if (data.scan_profile !== undefined && data.scan_profile !== null) patch.scanProfile = data.scan_profile;
  if (data.scan_health !== undefined && data.scan_health !== null) patch.scanHealth = data.scan_health;
  return patch;
}

// created_at may be a float epoch (live events) or a SQLite UTC string
// "YYYY-MM-DD HH:MM:SS" (persisted recent_events). Normalize both to epoch ms.
function eventAtMs(createdAt: unknown): number {
  if (typeof createdAt === "number") return createdAt * 1000;
  if (typeof createdAt === "string") {
    const ms = Date.parse(createdAt.replace(" ", "T") + "Z");
    if (!Number.isNaN(ms)) return ms;
  }
  return Date.now();
}

export function mapEvent(jobId: string, ev: any): JobEvent {
  return {
    // The backend now stamps every event with its owner. Keep the route id as
    // a compatibility fallback for older packaged engines.
    jobId: typeof ev.job_id === "string" ? ev.job_id : jobId,
    eventType: ev.event_type,
    phase: ev.phase ?? undefined,
    stageLabel: ev.stage_label ?? undefined,
    message: ev.message ?? undefined,
    progress: ev.progress ?? undefined,
    payload: ev.payload ?? undefined,
    at: eventAtMs(ev.created_at),
  };
}

// Map raw backend status + progress onto our cinematic JobStatus. Shared by the
// SSE handlers and the polling fallback so status advances consistently.
function mapJobStatus(status: string | undefined, progressPct: number): JobStatus {
  if (status === "completed") return "completed";
  if (status === "failed") return "failed";
  if (status === "cancelled") return "cancelled";
  // Backend "cancelling" is still in-flight — keep a mid-scan band so the UI
  // can show "Stopping safely…" until the terminal cancelled event arrives.
  if (status === "pending") return "queued";
  // running / cancelling / undefined
  if (progressPct < 25) return "analyzing";
  if (progressPct < 75) return "detecting";
  return "assembling";
}

/** Shared create + /jobs/run + SSE subscribe used by single and batch start. */
async function enqueueScanJob(
  url: string,
  name: string | undefined,
  opts: {
    focus: boolean;
    logEach?: boolean;
    vtuberMode?: "off" | "confirmed";
    sourceDate?: string;
    sourceType?: "file" | "twitch";
    recallSessionId?: string;
    vodStartedAtUtc?: string;
  },
): Promise<string | null> {
  const settings = useJobStore.getState().settings;
  const tempId = crypto.randomUUID();
  const job: Job = {
    id: tempId,
    url,
    status: "queued",
    progress: 0,
    createdAt: Date.now(),
    stage: { label: "Queued", progress: 0 },
  };
  const sessionName =
    (name && name.trim()) ||
    url.replace(/\\/g, "/").split("/").pop() ||
    "Untitled session";
  const session: Session = {
    id: tempId,
    name: sessionName.slice(0, 200),
    sourceUrl: url,
    sourceDate: opts.sourceDate,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: "queued",
    savedClipIds: [],
    exportedClipIds: [],
    clips: [],
  };

  useJobStore.setState((s) => ({
    jobs: [job, ...s.jobs],
    sessions: [session, ...s.sessions],
    ...(opts.focus
      ? { currentJob: job, currentSessionId: tempId }
      : {}),
  }));

  let studioReached = false;
  try {
    if (opts.logEach) {
      useJobStore.getState().addLog(`Queued: ${sessionName}`, "info");
    } else {
      useJobStore.getState().addLog(`Connecting to the local studio...`, "info");
    }
    const res = await apiFetch("/jobs/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_path: url,
        source_type: opts.sourceType || (url.startsWith("http") ? "twitch" : "file"),
        source_date: opts.sourceDate,
        recall_session_id: opts.recallSessionId,
        vod_started_at_utc: opts.vodStartedAtUtc,
        session_name: sessionName.slice(0, 200),
        settings: {
          processingMode: settings.processingMode,
          performanceProfile: settings.performanceProfile,
          exportLayout: settings.exportLayout,
          captionStyle: settings.captionStyle,
          // Identity is deliberately per scan and defaults off. Never infer
          // this from webcam geometry or a VOD title.
          vtuberMode: opts.vtuberMode === "confirmed" ? "confirmed" : "off",
        },
      }),
    }, settings.apiEndpoint);
    studioReached = true;

    if (!res.ok) {
      const body = await res.json().catch(() => null);
      const detail = typeof body?.detail === "string" ? body.detail : "";
      if (res.status === 409 && detail) throw new Error(detail);
      if (res.status >= 500) {
        throw new Error(
          "The local studio could not start this scan. Restart Recall and try again. If it still fails, check the launcher window for the API error.",
        );
      }
      throw new Error(detail || `The local studio rejected this scan (${res.status}). Check the source and try again.`);
    }

    const data = await res.json();
    const realId = data.job_id as string;

    if (!opts.logEach) {
      useJobStore.getState().addLog(`Local studio connected. Review ID: ${realId}`, "success");
      useJobStore.getState().addLog(`Getting the local scan ready...`, "info");
    }

    useJobStore.setState((state) => {
      const updatedJobs = state.jobs.map((j) =>
        j.id === tempId ? { ...j, id: realId } : j,
      );
      const updatedSessions = state.sessions.map((s) =>
        s.id === tempId
          ? {
              ...s,
              id: realId,
              name: s.name,
              posterUrl: apiUrl(`/jobs/${realId}/thumb`, settings.apiEndpoint),
            }
          : s,
      );
      return {
        jobs: updatedJobs,
        sessions: updatedSessions,
        currentJob:
          state.currentJob?.id === tempId
            ? { ...state.currentJob, id: realId }
            : state.currentJob,
        currentSessionId:
          state.currentSessionId === tempId ? realId : state.currentSessionId,
      };
    });

    // Recall runs one pipeline at a time. Keep one live stream for the focused
    // job; the lightweight job index promotes/subscribes the next queued item.
    if (opts.focus) reconcileJobSubscriptions(realId);
    return realId;
  } catch (e) {
    console.error(e);
    const message = studioReached
      ? (e instanceof Error ? e.message : "The local studio could not start this scan. Try again.")
      : "Recall couldn't reach the local studio. Close Recall, reopen it, and try again.";
    if (!opts.logEach) {
      useJobStore.getState().addLog(message, "warn");
    }
    useJobStore.getState().updateJob(tempId, {
      status: "failed",
      stage: { label: studioReached ? "Scan could not start" : "Connection failed", progress: 0 },
      message,
      errorMessage: message,
    });
    return null;
  }
}

const foundFromEvent = (ev: JobEvent): FoundClip | undefined => {
  const p = ev.payload;
  if (ev.eventType !== "clip_found" || !p) return undefined;
  return {
    clipId: p.clip_id,
    title: p.title || "Untitled moment",
    timestamp: p.timestamp || "0:00",
    duration: p.duration || 0,
    score: p.score || 0,
    pending: !!p.pending,
  };
};

// Merge an ordered batch of raw events into the current job's history.
function mergeEvents(jobId: string, rawEvents: any[]) {
  if (!rawEvents?.length) return;
  const store = useJobStore.getState();
  const job = store.currentJob?.id === jobId ? store.currentJob : store.jobs.find((j) => j.id === jobId);
  if (!job) return;
  // A batch opens one stream per queued VOD. Treat the event's explicit owner
  // as authoritative so a late message from the previous scan can never be
  // painted into the next scan's activity feed.
  const mapped = rawEvents
    .map((event) => mapEvent(jobId, event))
    .filter((event) => event.jobId === jobId);
  if (!mapped.length) return;
  const prev = job.events ?? [];
  const combined = [...mapped, ...prev];
  // Dedupe on (eventType, phase, message, at) and cap the buffer.
  const seen = new Set<string>();
  const deduped: JobEvent[] = [];
  for (const e of combined) {
    const key = `${e.eventType}|${e.phase}|${e.message}|${Math.round(e.at / 1000)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(e);
  }
  deduped.sort((a, b) => b.at - a.at);
  const events = deduped.slice(0, 120);

  const found = new Map<string, FoundClip>();
  (job.foundClips ?? []).forEach((f) => found.set(f.clipId, f));
  let liveTimeline = job.liveTimeline;
  for (const e of mapped) {
    const fc = foundFromEvent(e);
    // The post-export clip_found supersedes the selection-time candidate
    // announcement for the same moment (payload.replaces = candidate id).
    if (fc && e.payload?.replaces) found.delete(e.payload.replaces);
    if (fc) found.set(fc.clipId, fc);
    // Pin the mid-job reaction curve (handbook/21 §6.3) outside the capped
    // events buffer so it can't be evicted before the scan theater reads it.
    if (e.eventType === "timeline_ready" && e.payload?.timeline) {
      liveTimeline = e.payload.timeline as ReactionTimeline;
    }
  }
  const foundClips = Array.from(found.values());

  let scanHealth = job.scanHealth;
  for (const e of mapped) {
    if (
      e.eventType === "semantic_judge_health"
      || e.eventType === "semantic_judge_degraded"
      || e.eventType === "visual_judge_health"
      || e.eventType === "visual_judge_degraded"
      || e.eventType === "visual_judge_skipped"
    ) {
      const next = {
        degraded: Boolean(scanHealth?.degraded),
        device: (e.payload?.device as string | undefined) ?? scanHealth?.device ?? null,
        semantic: scanHealth?.semantic ?? null,
        visual: scanHealth?.visual ?? null,
      };
      if (e.eventType.startsWith("semantic_judge")) {
        next.semantic = {
          ...(e.payload || {}),
          status: e.eventType === "semantic_judge_degraded" ? "degraded" : (e.payload?.status || "healthy"),
          degraded: e.eventType === "semantic_judge_degraded" || Boolean(e.payload?.degraded),
        };
      } else {
        const status =
          e.eventType === "visual_judge_skipped"
            ? "skipped"
            : e.eventType === "visual_judge_degraded"
              ? "degraded"
              : (e.payload?.status || "healthy");
        next.visual = { ...(e.payload || {}), status, degraded: status === "degraded" || status === "skipped" };
      }
      next.degraded = Boolean(
        (next.semantic && (next.semantic.degraded || next.semantic.status === "degraded"))
        || (next.visual && (next.visual.status === "degraded" || next.visual.status === "skipped")),
      );
      scanHealth = next;
    }
  }

  store.updateJob(jobId, { events, foundClips, liveTimeline, scanHealth });
}

const jobSubscriptions = new Map<string, () => void>();

function reconcileJobSubscriptions(desiredJobId?: string) {
  for (const [jobId, dispose] of jobSubscriptions) {
    if (jobId === desiredJobId) continue;
    dispose();
    jobSubscriptions.delete(jobId);
  }
  if (!desiredJobId || jobSubscriptions.has(desiredJobId)) return;
  const dispose = subscribeJobEvents(desiredJobId, () => {
    if (jobSubscriptions.get(desiredJobId) === dispose) jobSubscriptions.delete(desiredJobId);
  });
  jobSubscriptions.set(desiredJobId, dispose);
}

export function disposeAllJobSubscriptions() {
  for (const dispose of jobSubscriptions.values()) dispose();
  jobSubscriptions.clear();
}

// Live SSE subscription with automatic polling fallback. Returns a disposer.
export function subscribeJobEvents(jobId: string, onClosed?: () => void): () => void {
  const base = useJobStore.getState().settings.apiEndpoint || DEFAULT_API_ENDPOINT;
  let es: EventSource | null = null;
  let disposed = false;
  let fellBack = false;
  let stopPolling: (() => void) | null = null;

  const finish = () => {
    if (disposed) return;
    disposed = true;
    try { es?.close(); } catch {}
    stopPolling?.();
    stopPolling = null;
    onClosed?.();
  };

  const fallbackToPolling = () => {
    if (disposed || fellBack) return;
    fellBack = true;
    try { es?.close(); } catch {}
    stopPolling = pollPipeline(jobId, { onTerminal: finish });
  };

  try {
    es = new EventSource(apiUrl(`/jobs/${jobId}/events`, base));

    es.addEventListener("snapshot", (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        const store = useJobStore.getState();
        const progressPct = Math.round((data.progress || 0) * 100);
        const stagePct = data.stage_progress == null ? progressPct : Math.round(data.stage_progress * 100);
        store.updateJob(jobId, {
          progress: progressPct,
          status: mapJobStatus(data.status, progressPct),
          stage: { label: data.stage_label || data.current_stage || "Processing…", progress: stagePct },
          ...telemetryFromSnapshot(data),
          ...(data.status === "failed" ? { errorMessage: data.message || "Analysis failed." } : {}),
        });
        if (Array.isArray(data.recent_events)) mergeEvents(jobId, data.recent_events);
      } catch {}
    });

    es.addEventListener("message", (e: MessageEvent) => {
      try {
        const ev = JSON.parse(e.data);
        mergeEvents(jobId, [ev]);
        const store = useJobStore.getState();
        const cur = store.currentJob?.id === jobId ? store.currentJob : store.jobs.find((j) => j.id === jobId);
        const patch: Partial<Job> = {};
        if (typeof ev.progress === "number") {
          patch.progress = Math.round(ev.progress * 100);
          patch.stage = { label: ev.stage_label || ev.phase || cur?.stage?.label || "Processing…", progress: patch.progress };
        }
        if (ev.message) patch.message = ev.message;
        if (ev.phase) patch.currentPhase = ev.phase;
        if (ev.payload?.stage_progress !== undefined) {
          const base = patch.stage ?? cur?.stage;
          if (base) patch.stage = { ...base, progress: Math.round(ev.payload.stage_progress * 100) };
        }
        if (ev.payload?.scanned_seconds !== undefined) patch.scannedSeconds = ev.payload.scanned_seconds;
        if (ev.payload?.scan_speed !== undefined) patch.scanSpeed = ev.payload.scan_speed;
        if (ev.payload?.transcribed_seconds !== undefined) patch.transcribedSeconds = ev.payload.transcribed_seconds;
        if (ev.payload?.transcription_speed !== undefined) patch.transcriptionSpeed = ev.payload.transcription_speed;
        if (ev.payload?.active_workers !== undefined) patch.activeWorkers = ev.payload.active_workers;
        if (ev.payload?.total_workers !== undefined) patch.totalWorkers = ev.payload.total_workers;
        if (ev.payload?.download_progress !== undefined) patch.downloadProgress = ev.payload.download_progress;
        if (ev.payload?.clips_found !== undefined) patch.clipsFound = ev.payload.clips_found;
        if (ev.payload?.scan_profile !== undefined) patch.scanProfile = ev.payload.scan_profile;
        if (ev.payload?.duration !== undefined && ev.payload.duration != null) patch.vodDuration = ev.payload.duration;
        if (ev.payload?.eta_seconds !== undefined) {
          patch.etaSeconds = ev.payload.eta_seconds;
          patch.etaSyncedAt = Date.now();
        }
        if (ev.payload?.eta_confidence !== undefined) patch.etaConfidence = ev.payload.eta_confidence as EtaConfidence;
        // Advance the cinematic status while the job is mid-flight. Never
        // revive a terminal job from a late progress heartbeat.
        const terminalNow = cur && (cur.status === "completed" || cur.status === "failed" || cur.status === "cancelled");
        if (cur && !terminalNow && ev.event_type !== "job_completed" && ev.event_type !== "job_cancelled" && ev.event_type !== "job_failed") {
          const pct = patch.progress ?? cur.progress ?? 0;
          patch.status = mapJobStatus("running", pct);
        }
        if (ev.event_type === "clip_found") {
          // Prefer an authoritative payload count when present; otherwise bump
          // once. Avoids N then N+1 when stage_progress already carried clips_found.
          if (typeof ev.payload?.clips_found !== "number" && patch.clipsFound === undefined) {
            patch.clipsFound = (cur?.clipsFound ?? 0) + 1;
          }
          store.addLog(`Possible clip found - ${ev.payload?.title ?? "new clip"}`, "success");
        }
        if (ev.event_type === "job_completed") {
          patch.status = "completed";
          patch.progress = Math.max(patch.progress ?? cur?.progress ?? 0, 100);
          if (Object.keys(patch).length) store.updateJob(jobId, patch);
          void loadJobClips(jobId).catch(() => undefined);
          return;
        }
        if (ev.event_type === "job_failed") {
          patch.status = "failed";
          patch.errorMessage = ev.message || "Analysis failed.";
        }
        if (ev.event_type === "job_cancelled") {
          patch.status = "cancelled";
        }
        if (Object.keys(patch).length) store.updateJob(jobId, patch);
      } catch {}
    });

    es.addEventListener("end", () => {
      try { es?.close(); } catch {}
      // Reconcile authoritative final state + load clips.
      if (!disposed) pollPipeline(jobId, { once: true });
      finish();
    });

    es.onerror = () => {
      // Browser will retry automatically, but if it never opened, fall back.
      if (es && es.readyState === EventSource.CLOSED) fallbackToPolling();
    };
  } catch {
    fallbackToPolling();
  }

  return finish;
}

const jobClipLoads = new Map<string, Promise<void>>();
const CLIP_HYDRATION_RETRY_DELAYS_MS = [250, 1000];

async function fetchJobClipsOnce(jobId: string) {
  const store = useJobStore.getState();
  const clipsRes = await apiFetch(
    `/clips?job_id=${encodeURIComponent(jobId)}`,
    undefined,
    store.settings.apiEndpoint,
  );
  if (!clipsRes.ok) {
    const payload = await clipsRes.json().catch(() => null);
    const detail = typeof payload?.detail === "string"
      ? payload.detail
      : `Clip hydration failed with HTTP ${clipsRes.status}`;
    throw new Error(detail);
  }
  const jobClips = await clipsRes.json();
  const mappedClips: Clip[] = jobClips.map((c: any) => toClip(c, store.settings.apiEndpoint));
  store.addLog(`Loaded ${mappedClips.length} clip${mappedClips.length === 1 ? "" : "s"} for review.`, "success");
  store.updateJob(jobId, { clips: mappedClips });
}

async function fetchJobClips(jobId: string) {
  let lastError: unknown;
  for (let attempt = 0; attempt <= CLIP_HYDRATION_RETRY_DELAYS_MS.length; attempt += 1) {
    try {
      await fetchJobClipsOnce(jobId);
      return;
    } catch (error) {
      lastError = error;
      const delay = CLIP_HYDRATION_RETRY_DELAYS_MS[attempt];
      if (delay === undefined) break;
      await new Promise<void>((resolve) => setTimeout(resolve, delay));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Recall could not load this session's clips.");
}

function loadJobClips(jobId: string): Promise<void> {
  const existing = jobClipLoads.get(jobId);
  if (existing) return existing;
  const pending = fetchJobClips(jobId).finally(() => {
    if (jobClipLoads.get(jobId) === pending) jobClipLoads.delete(jobId);
  });
  jobClipLoads.set(jobId, pending);
  return pending;
}

// One polling tick. Returns true when the job has reached a terminal state.
async function pollTick(jobId: string): Promise<boolean> {
  const store = useJobStore.getState();
  const job = store.currentJob?.id === jobId ? store.currentJob : store.jobs.find((j) => j.id === jobId);
  if (!job) return true;

  const res = await apiFetch(`/jobs/${jobId}`, undefined, store.settings.apiEndpoint);
  if (!res.ok) throw new Error(`Job polling failed with HTTP ${res.status}`);
  const data = await res.json();

  const nextProgress = Math.round((data.progress || 0) * 100);
  const nextStageProgress = data.stage_progress == null ? nextProgress : Math.round(data.stage_progress * 100);
  const status = data.status || "analyzing";

  const mappedStatus: JobStatus = mapJobStatus(status, nextProgress);

  store.updateJob(jobId, {
    progress: nextProgress,
    status: mappedStatus,
    stage: { label: data.stage_label || data.current_stage || "Looking through the recording...", progress: nextStageProgress },
    ...telemetryFromSnapshot(data),
    ...(status === "failed" ? { errorMessage: data.message || "Analysis failed." } : {}),
  });
  if (Array.isArray(data.recent_events)) mergeEvents(jobId, data.recent_events);

  if (mappedStatus === "completed") {
    store.addLog(`Analysis complete. Getting your clips...`, "success");
    await loadJobClips(jobId);
    return true;
  }
  if (mappedStatus === "failed") {
    store.addLog(`Analysis stopped: ${data.message || "Job marked failed."}`, "warn");
    return true;
  }
  if (mappedStatus === "cancelled") {
    store.addLog("Scan cancelled.", "info");
    return true;
  }
  return false;
}

export function pollPipeline(jobId: string, opts?: { once?: boolean; onTerminal?: () => void }): () => void {
  if (opts?.once) {
    pollTick(jobId).catch((e) => console.error("Polling error:", e));
    return () => {};
  }

  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let retryDelay = 1000;

  const schedule = (delay: number) => {
    if (stopped) return;
    timer = setTimeout(run, delay);
  };

  const run = async () => {
    if (stopped) return;
    try {
      const done = await pollTick(jobId);
      if (done) {
        stopped = true;
        opts?.onTerminal?.();
        return;
      }
      retryDelay = 1000;
    } catch (e) {
      console.error("Polling error:", e);
      retryDelay = Math.min(30_000, retryDelay * 2);
    }
    schedule(retryDelay);
  };

  schedule(retryDelay);
  return () => {
    stopped = true;
    if (timer !== undefined) clearTimeout(timer);
  };
}

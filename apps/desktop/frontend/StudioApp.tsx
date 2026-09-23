// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { type CSSProperties, FormEvent, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  ArrowUpDown,
  BarChart3,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  Film,
  FolderOpen,
  HardDrive,
  Home,
  Library,
  Maximize2,
  Minus,
  Play,
  Plus,
  RefreshCw,
  Repeat,
  Scissors,
  Search,
  Settings,
  Target,
  Trash2,
  Video,
  X,
  ArrowLeft,
  Sparkles,
} from "./lib/icons";
import { ModalityMix } from "./components/ReactionEvidence";
import { apiFetch } from "./lib/api";
import { fmtClock, fmtDurationHuman, sourceName } from "./lib/format";
import { learningStateDescription, learningStateLabel, clueLabel, describeScanFailure, describeScanHealth, plural } from "./lib/copy";
import ProblemBar from "./components/ProblemBar";
import ClipLibraryEditor from "./components/ClipLibraryEditor";
import { releaseClipPreview } from "./lib/releaseClipPreview";
import CuttingRoom, { type CuttingRoomMemoryHandoff } from "./components/CuttingRoom";
import { HardwareSettingsPanel } from "./components/HardwareSetup";
import RememberHotkeySetting from "./components/RememberHotkeySetting";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import { OnboardingFlow } from "./components/Onboarding";
import { HeatSparkline } from "./components/heat";
import { EvidenceChips, evidenceChipsFor } from "./components/evidence";
import StudioEmptyState from "./components/StudioEmptyState";
import { ShareHandoffDialog } from "./components/ShareHandoff";
import InsightsStory from "./components/InsightsStory";
import { CaptionEditor } from "./components/CaptionEditor";
import LiveSessionTestPanel from "./components/LiveSessionTestPanel";
import { getActiveRecallSession, type RecallSession } from "./lib/recallSessions";
import RecallOriginBadge, { recallOriginCopy } from "./components/RecallOriginBadge";
import StreamMemory from "./components/StreamMemory";
import CompilationProjects from "./components/CompilationProjects";
import { AccentChoices, ChoiceCards, ExportLayoutChoices, MediaVolumeControl, StateIcon, StudioButton, StudioSwitch, ThemeChoices, useMediaVolume, type ChoiceOption } from "./components/StudioControls";
import { hasCompletedOnboarding } from "./lib/onboarding";
import { timelinePath } from "./lib/timeline";
import { isInteractiveKeyboardTarget } from "./lib/keyboard";
import { writeClipboardText } from "./lib/clipboard";
import { buildLocalDiagnostics } from "./lib/diagnostics";
import { buildScanRetryDraft } from "./lib/scanRecovery";
import { batchNotification, exportNotification, hasActiveDesktopWork, isActiveScanStatus, isTerminalScanStatus, scanNotification } from "./lib/desktopWork";
import { creatorExportFilename } from "./lib/exportName";
import {
  DEFAULT_CAPTION_STYLE,
  disposeAllJobSubscriptions,
  fetchReactionTimeline,
  type CaptionStyle,
  type Clip,
  type ExportOperation,
  type Job,
  type JobEvent,
  type JobStatus,
  type ReactionTimeline,
  type Session,
  mapEvent,
  useJobStore,
} from "./lib/store";
import {
  parseTwitchVodUrls,
  resolveBatchSessionName,
  sessionNameFromProbe,
} from "./lib/twitchBatch";

type View = "home" | "live" | "sessions" | "memory" | "compilations" | "add-vod" | "processing" | "queue" | "review" | "theater" | "cutting" | "clips" | "reel" | "insights";
type ImportMode = "single" | "batch";
type ReviewFilter = "all" | "kept" | "maybe" | "unreviewed";
type ReviewSort = "recommended" | "hook" | "timeline" | "longest";
type SettingsTab = "project" | "general" | "export" | "appearance" | "shortcuts";

const VTUBER_TITLE_MARKER = /(?:^|[^a-z0-9])(?:v[ -]?tuber|png[ -]?tuber|virtual\s+(?:streamer|youtuber))(?:$|[^a-z0-9])/i;

export function titleSuggestsVtuberMode(title: string | null | undefined): boolean {
  return VTUBER_TITLE_MARKER.test(title || "");
}

type ProbeMeta = {
  duration: number | null;
  thumbnail: string | null;
  title?: string | null;
  creator?: string | null;
  game?: string | null;
  vod_id?: string | null;
  source_date?: string | null;
};

type BatchRow = {
  id: string;
  url: string;
  title: string;
  titleEdited: boolean;
  probing: boolean;
  probeError: boolean;
  probeMeta: ProbeMeta | null;
  vtuberConfirmed: boolean;
};

const terminal = new Set(["completed", "failed", "cancelled"]);
const REVIEW_DECK_LIMIT = 15;

/** Official brand glyphs (simple-icons paths) — lucide has no platform logos. */
function TwitchLogo({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M11.571 4.714h1.715v5.143H11.57zm4.715 0H18v5.143h-1.714zM6 0L1.714 4.286v15.428h5.143V24l4.286-4.286h3.428L22.286 12V0zm14.571 11.143l-3.428 3.428h-3.429l-3 3v-3H6.857V1.714h13.714z" />
    </svg>
  );
}

function YouTubeLogo({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
    </svg>
  );
}

function KickLogo({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M2 0h6.5v9.75L15.25 0H22l-7.5 9.75L22 24h-6.75L8.5 13.5V24H2z" />
    </svg>
  );
}

function rankScoreOf(clip: Clip) {
  return Math.max(0, Math.min(99, clip.deckScore ?? clip.confidence ?? 0));
}

function hookScoreOf(clip: Clip) {
  return Math.max(0, Math.min(100, clip.confidence ?? 0));
}

function durationOf(clip: Clip) {
  return Math.max(1, Math.round(clip.duration || ((clip.end_time ?? 0) - (clip.start_time ?? 0))));
}

function useCountUp(value: string | number) {
  const match = String(value).match(/^(-?\d+(?:\.\d+)?)(.*)$/);
  const end = match ? Number(match[1]) : null;
  const suffix = match?.[2] ?? "";
  const decimals = match?.[1].split(".")[1]?.length ?? 0;
  const previous = useRef(end ?? 0);
  const [display, setDisplay] = useState(value);

  useEffect(() => {
    if (end == null || !Number.isFinite(end)) {
      setDisplay(value);
      return;
    }
    const start = previous.current;
    previous.current = end;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || start === end) {
      setDisplay(`${end.toFixed(decimals)}${suffix}`);
      return;
    }
    const startedAt = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const elapsed = Math.min(1, (now - startedAt) / 420);
      const eased = 1 - Math.pow(1 - elapsed, 4);
      const next = start + (end - start) * eased;
      setDisplay(`${next.toFixed(decimals)}${suffix}`);
      if (elapsed < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [end, decimals, suffix, value]);

  return display;
}

type RevealRect = { left: number; top: number; width: number; height: number };
type ScanRevealSnapshot = { jobId: string; cards: Array<{ clipId: string; rect: RevealRect }> };

function sortReviewClips(clips: Clip[], sort: ReviewSort) {
  const ordered = [...clips];
  if (sort === "timeline") {
    return ordered.sort((a, b) => (a.start_time ?? 0) - (b.start_time ?? 0));
  }
  if (sort === "longest") {
    return ordered.sort((a, b) => durationOf(b) - durationOf(a) || rankScoreOf(b) - rankScoreOf(a));
  }
  if (sort === "hook") {
    return ordered.sort((a, b) => hookScoreOf(b) - hookScoreOf(a) || rankScoreOf(b) - rankScoreOf(a));
  }
  return ordered.sort((a, b) => rankScoreOf(b) - rankScoreOf(a) || (a.start_time ?? 0) - (b.start_time ?? 0));
}

const originalDeckClips = (clips: Clip[]) => clips.filter((clip) => !clip.isOverflowCandidate);
const secondLookClips = (clips: Clip[]) => clips.filter((clip) => clip.isOverflowCandidate);
const primaryReviewDeckClips = (clips: Clip[]) =>
  sortReviewClips(originalDeckClips(clips), "recommended").slice(0, REVIEW_DECK_LIMIT);

function relativeTime(value: number) {
  const hours = Math.max(0, Math.floor((Date.now() - value) / 3_600_000));
  if (hours < 1) return "just now";
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function timestampSeconds(timestamp: string) {
  const parts = timestamp.split(":").map((part) => Number(part));
  if (!parts.length || parts.some((part) => !Number.isFinite(part) || part < 0)) return null;
  return parts.reduce((total, part) => total * 60 + part, 0);
}

function useModalFocus<T extends HTMLElement>(open: boolean, onClose: () => void) {
  const dialogRef = useRef<T>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    const dialog = dialogRef.current;
    const priorFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusableSelector = [
      "[data-autofocus]",
      "button:not([disabled])",
      "input:not([disabled])",
      "select:not([disabled])",
      "textarea:not([disabled])",
      "[href]",
      "[tabindex]:not([tabindex='-1'])",
    ].join(",");
    const frame = window.requestAnimationFrame(() => {
      dialog?.querySelector<HTMLElement>(focusableSelector)?.focus();
    });

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current();
        return;
      }
      if (event.key !== "Tab" || !dialog) return;
      const focusable = [...dialog.querySelectorAll<HTMLElement>(focusableSelector)]
        .filter((element) => !element.hidden && element.getClientRects().length > 0);
      if (!focusable.length) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("keydown", onKeyDown);
      priorFocus?.focus();
    };
  }, [open]);

  return dialogRef;
}

function stageIndex(job?: Job) {
  if (!job) return 0;
  const phase = `${job.currentPhase ?? ""} ${job.stage?.label ?? ""}`.toLowerCase();
  if (/export|render|preview|package/.test(phase)) return 4;
  if (/caption|writ/.test(phase)) return 3;
  if (/reaction|event|story|clip|select|choos|score|rank|moment/.test(phase)) return 2;
  if (/perception|scan|vision|watch|face|game|ocr|audio|listen|speech/.test(phase)) return 1;
  return Math.min(4, Math.floor((job.progress ?? 0) / 20));
}

function creatorActivity(message: string, eventType: string, stageLabel?: string) {
  if (/^Device summary:|^Video metadata:/i.test(message)) return null;
  const download = message.match(/Downloading Twitch VOD:\s*(\d+)%/i);
  if (download) {
    const pct = Number(download[1]);
    return pct % 10 === 0 || pct === 100 ? `Downloading Twitch VOD · ${pct}%` : null;
  }
  const finalizing = message.match(/Finalizing Video\s+(\d+)%/i);
  if (finalizing) {
    const pct = Number(finalizing[1]);
    return pct % 25 === 0 || pct === 100 ? `Preparing downloaded video · ${pct}%` : null;
  }
  // Progress heartbeats fire every few seconds with a changing % — the live
  // "current" line and the ring already carry that number, so the running log
  // collapses them all to one steady milestone instead of a flood of pulses.
  if (/% scanned/i.test(message) || /of the VOD covered/i.test(message)) return "Watching your VOD for big moments";
  if (/Perceiving in parallel/i.test(message)) return "Scanning gameplay, reactions, and on-screen moments";
  if (/Extracting audio/i.test(message)) return "Preparing audio for reaction analysis";
  if (/Found cached signals/i.test(message)) return "Reusing the existing analysis for this recording";
  if (eventType === "clip_found") return "Found a moment worth reviewing";
  if (eventType === "stage_completed" && stageLabel) return `${stageLabel} complete`;
  return message;
}

export type ActivityItem = { key: string; timestamp: string; text: string; detail: string; ok: boolean };

// Turn a job's persisted event stream into a creator-facing activity log.
// `events` arrive newest-first. We collapse only *consecutive* identical lines
// so repeated milestones (e.g. the scan heartbeat) show once while genuine
// progression stays visible — the whole history is kept so it never feels like
// activity is being thrown away. Used by both the live feed and the scan log.
export function activityFromEvents(events: JobEvent[], jobId?: string, limit = 40): ActivityItem[] {
  const items: ActivityItem[] = [];
  let lastText = "";
  // The live merge path is ordered already, but restored snapshots and test
  // fixtures are not required to be. "Latest" must remain newest-first.
  const ordered = events
    .filter((entry) => !jobId || entry.jobId === jobId)
    .sort((a, b) => b.at - a.at);
  for (const event of ordered) {
    const raw = event.message || (event.eventType === "stage_completed" ? event.stageLabel || event.phase || "Stage" : "");
    if (!raw) continue;
    const text = creatorActivity(raw, event.eventType, event.stageLabel);
    if (!text || text === lastText) continue;
    lastText = text;
    items.push({
      key: `${event.eventType}-${event.at}`,
      timestamp: new Date(event.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }),
      text,
      detail: event.stageLabel || event.phase || "",
      ok: event.eventType === "stage_completed" || event.eventType === "clip_found",
    });
    if (items.length >= limit) break;
  }
  return items;
}

function activityItems(job: Job | undefined) {
  return activityFromEvents(job?.events ?? [], job?.id, 24);
}

export default function StudioApp() {
  const {
    jobs,
    sessions,
    currentJob,
    currentSessionId,
    settings,
    startJob,
    startJobBatch,
    batchJobIds,
    batchRestored,
    clearBatchJobIds,
    loadSessions,
    refreshJobs,
    hydrateSessionClips,
    loadMoreCandidates,
    prepareClipPreview,
    loadRemoteSettings,
    exportOperation,
    setCurrentSessionId,
    toggleSaveClip,
    setClipPassed,
    setClipMaybe,
    copyClipsToFolder,
    compileReel,
    cancelExportOperation,
    deleteSession,
    addLog,
    setSettings,
    editClip,
    updateClipTitle,
  } = useJobStore(useShallow((state) => ({
    jobs: state.jobs,
    sessions: state.sessions,
    currentJob: state.currentJob,
    currentSessionId: state.currentSessionId,
    settings: state.settings,
    startJob: state.startJob,
    startJobBatch: state.startJobBatch,
    batchJobIds: state.batchJobIds,
    batchRestored: state.batchRestored,
    clearBatchJobIds: state.clearBatchJobIds,
    loadSessions: state.loadSessions,
    refreshJobs: state.refreshJobs,
    hydrateSessionClips: state.hydrateSessionClips,
    loadMoreCandidates: state.loadMoreCandidates,
    prepareClipPreview: state.prepareClipPreview,
    loadRemoteSettings: state.loadRemoteSettings,
    exportOperation: state.exportOperation,
    setCurrentSessionId: state.setCurrentSessionId,
    toggleSaveClip: state.toggleSaveClip,
    setClipPassed: state.setClipPassed,
    setClipMaybe: state.setClipMaybe,
    copyClipsToFolder: state.copyClipsToFolder,
    compileReel: state.compileReel,
    cancelExportOperation: state.cancelExportOperation,
    deleteSession: state.deleteSession,
    addLog: state.addLog,
    setSettings: state.setSettings,
    editClip: state.editClip,
    updateClipTitle: state.updateClipTitle,
  })));
  const [view, setView] = useState<View>("home");
  const [theaterReturn, setTheaterReturn] = useState<View>("review");
  const [cuttingReturn, setCuttingReturn] = useState<View>("review");
  const [memoryEditorHandoff, setMemoryEditorHandoff] = useState<CuttingRoomMemoryHandoff | null>(null);
  const [projectModal, setProjectModal] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Session | null>(null);
  const [bulkDeleteTargets, setBulkDeleteTargets] = useState<Session[]>([]);
  const [editingClip, setEditingClip] = useState<{ session: Session; clip: Clip } | null>(null);
  const [projectName, setProjectName] = useState("");
  const [importMode, setImportMode] = useState<ImportMode>("single");
  const [batchRows, setBatchRows] = useState<BatchRow[]>([]);
  const [batchPaste, setBatchPaste] = useState("");
  const [batchStarting, setBatchStarting] = useState(false);
  const [sourceKind, setSourceKind] = useState<"local" | "twitch">("local");
  const [source, setSource] = useState("");
  const [twitchUrl, setTwitchUrl] = useState("");
  const [probeMeta, setProbeMeta] = useState<ProbeMeta | null>(null);
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState(false);
  const [probeAttempt, setProbeAttempt] = useState(0);
  const [vtuberConfirmed, setVtuberConfirmed] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<string>();
  const [reelSessionId, setReelSessionId] = useState<string>();
  const [selectedClipId, setSelectedClipId] = useState<string>();
  const [reviewFilter, setReviewFilter] = useState<ReviewFilter>("all");
  const [reviewSort, setReviewSort] = useState<ReviewSort>("recommended");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("project");
  const [onboardingOpen, setOnboardingOpen] = useState(() => !hasCompletedOnboarding());
  const [query, setQuery] = useState("");
  const [timeline, setTimeline] = useState<ReactionTimeline | null>(null);
  const [exporting, setExporting] = useState(false);
  // Set only after files actually land on disk — the share step is a handoff for
  // a real export, never a generic "share" entry point.
  const [shareHandoff, setShareHandoff] = useState<{ folder: string; count: number } | null>(null);
  const [compilingReel, setCompilingReel] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);
  const [secondLookSessions, setSecondLookSessions] = useState<Set<string>>(() => {
    try {
      const stored = JSON.parse(localStorage.getItem("recall-second-look-sessions") || "[]");
      return new Set(Array.isArray(stored) ? stored.filter((id): id is string => typeof id === "string") : []);
    } catch {
      return new Set();
    }
  });
  const [moreCandidateStatus, setMoreCandidateStatus] = useState<Record<string, { available: number; availableTotal: number; loaded: number }>>({});
  const [moreCandidatesBusy, setMoreCandidatesBusy] = useState<string>();
  const [preparingPreviewId, setPreparingPreviewId] = useState<string>();
  const videoRef = useRef<HTMLVideoElement>(null);
  const scanRevealRef = useRef<ScanRevealSnapshot | null>(null);
  const openSessionRequestRef = useRef(0);
  const viewRef = useRef(view);
  viewRef.current = view;
  const closeProjectModal = useCallback(() => setProjectModal(false), []);
  const closeDeleteModal = useCallback(() => setDeleteTarget(null), []);
  const closeBulkDeleteModal = useCallback(() => setBulkDeleteTargets([]), []);
  const closeShareHandoff = useCallback(() => setShareHandoff(null), []);
  // Onboarding is not Esc-dismissable: the system check step sets the
  // processing route, and skipping is an explicit control on the later steps.
  const keepOnboardingOpen = useCallback(() => {}, []);
  const createDialogRef = useModalFocus<HTMLFormElement>(projectModal, closeProjectModal);
  const deleteDialogRef = useModalFocus<HTMLDivElement>(!!deleteTarget, closeDeleteModal);
  const bulkDeleteDialogRef = useModalFocus<HTMLDivElement>(bulkDeleteTargets.length > 0, closeBulkDeleteModal);
  const onboardingDialogRef = useModalFocus<HTMLDivElement>(onboardingOpen, keepOnboardingOpen);
  const shareDialogRef = useModalFocus<HTMLDivElement>(!!shareHandoff, closeShareHandoff);

  const sortedSessions = useMemo(
    () => [...sessions].sort((a, b) => sessionRecency(b) - sessionRecency(a)),
    [sessions],
  );
  const activeSession = useMemo(() => {
    const id = selectedSessionId ?? currentSessionId;
    return sessions.find((session) => session.id === id)
      ?? sortedSessions.find((session) => session.status === "completed");
  }, [sessions, selectedSessionId, currentSessionId, sortedSessions]);
  // The scan view reads its data from the live currentJob. A *reopened* failed
  // or cancelled session has no live job, so synthesize a minimal one from the
  // session — otherwise the end-state panel can't explain what happened.
  const processingJob = useMemo<Job | undefined>(() => {
    const activeId = selectedSessionId ?? currentSessionId;
    if (currentJob && (currentJob.id === activeId || !activeSession)) return currentJob;
    const selectedLiveJob = jobs.find((job) => job.id === activeId);
    if (selectedLiveJob) return selectedLiveJob;
    // A session is explicitly open but has no live job of its own.
    //
    // This used to fall through to `return currentJob`, which rendered a
    // *different* VOD's telemetry underneath the session you opened. It showed
    // up during batch scans: once the first job finished and the second began,
    // opening the second from the sidebar displayed the first one's activity
    // feed, because `currentJob` still pointed at the other job and nothing
    // reconciled it against the selection.
    //
    // Synthesize from the session instead. The stub carries no `events`, so the
    // activity feed renders empty rather than borrowing someone else's history.
    if (activeSession) {
      const stageLabel =
        activeSession.status === "cancelled" ? "Cancelled"
        : activeSession.status === "failed" ? "Failed"
        : activeSession.status === "completed" ? "Complete"
        : "Scanning";
      return {
        id: activeSession.id,
        url: activeSession.sourceUrl,
        status: activeSession.status,
        progress: activeSession.status === "completed" ? 100 : 0,
        stage: { label: stageLabel, progress: activeSession.status === "completed" ? 100 : 0 },
        createdAt: activeSession.createdAt,
        message: activeSession.message,
      };
    }
    return currentJob;
  }, [currentJob, activeSession, selectedSessionId, currentSessionId, jobs]);
  const originalActiveClips = useMemo(
    () => primaryReviewDeckClips(activeSession?.clips ?? []),
    [activeSession?.clips],
  );
  const extraActiveClips = useMemo(
    () => secondLookClips(activeSession?.clips ?? []),
    [activeSession?.clips],
  );
  const secondLookEnabled = !!activeSession && secondLookSessions.has(activeSession.id);
  const orderedActiveClips = useMemo(() => [
    ...sortReviewClips(originalActiveClips, reviewSort),
    ...(secondLookEnabled ? sortReviewClips(extraActiveClips, reviewSort) : []),
  ], [originalActiveClips, extraActiveClips, reviewSort, secondLookEnabled]);
  const visibleActiveSession = useMemo(() => activeSession ? {
    ...activeSession,
    clips: secondLookEnabled
      ? [...originalActiveClips, ...extraActiveClips]
      : originalActiveClips,
  } : undefined, [activeSession, originalActiveClips, extraActiveClips, secondLookEnabled]);
  const selectedClip = orderedActiveClips.find((clip) => clip.id === selectedClipId)
    ?? orderedActiveClips[0];

  const keptClips = useMemo(
    () => sessions.flatMap((session) => session.clips
      .filter((clip) => session.savedClipIds.includes(clip.id))
      .map((clip) => ({ clip, session }))),
    [sessions],
  );
  // Export selection lives in the Clip Library and starts EMPTY. It used to be
  // an exclusion set, which meant opening the Library pre-armed every kept clip
  // across every session for export — one stray click away from writing out
  // hundreds of files nobody asked for. Choosing is now explicit, and the
  // per-session control below is what makes choosing a whole VOD one click.
  const [pickedExportIds, setPickedExportIds] = useState<Set<string>>(new Set());
  // Clips can stop being kept while the Library is open; never let a stale id
  // reach an export.
  const selectedExportIds = useMemo(
    () => new Set(keptClips.map(({ clip }) => clip.id).filter((id) => pickedExportIds.has(id))),
    [keptClips, pickedExportIds],
  );
  const toggleExportSelected = (clipId: string) => setPickedExportIds((prev) => {
    const next = new Set(prev);
    if (next.has(clipId)) next.delete(clipId); else next.add(clipId);
    return next;
  });
  const selectAllExport = () => setPickedExportIds(new Set(keptClips.map(({ clip }) => clip.id)));
  const clearExportSelection = () => setPickedExportIds(new Set());
  /** Select or clear every kept clip belonging to one session. */
  const toggleSessionExport = (sessionId: string) => setPickedExportIds((prev) => {
    const ids = keptClips.filter(({ session }) => session.id === sessionId).map(({ clip }) => clip.id);
    const allPicked = ids.length > 0 && ids.every((id) => prev.has(id));
    const next = new Set(prev);
    for (const id of ids) {
      if (allPicked) next.delete(id); else next.add(id);
    }
    return next;
  });
  const filteredSessions = sortedSessions.filter((session) =>
    session.name.toLowerCase().includes(query.trim().toLowerCase()),
  );

  useEffect(() => {
    const theme = settings.theme || "dark";
    const accentTheme = settings.accentTheme || "ember";
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.accent = accentTheme;
    document.body.className = `view-${view}-active`;
    try { localStorage.setItem("recall-theme", theme); } catch {}
    try { localStorage.setItem("recall-accent-theme", accentTheme); } catch {}
    window.electronAPI?.setTitleBarOverlay?.(theme === "dark"
      ? { color: "#131017", symbolColor: "#F6F0E9", height: 48 }
      : { color: "#EDEDED", symbolColor: "#111827", height: 48 });
  }, [view, settings.theme, settings.accentTheme]);

  const activeDesktopWork = useMemo(
    () => hasActiveDesktopWork(jobs, sessions, exportOperation),
    [jobs, sessions, exportOperation],
  );
  useEffect(() => {
    void window.electronAPI?.setWorkProtection?.(
      settings.keepAwakeWhileWorking && activeDesktopWork,
    ).catch(() => {});
  }, [activeDesktopWork, settings.keepAwakeWhileWorking]);
  useEffect(() => () => {
    void window.electronAPI?.setWorkProtection?.(false).catch(() => {});
  }, []);

  const previousScanStatusesRef = useRef<Map<string, JobStatus>>(new Map());
  const previousBatchActiveRef = useRef(false);
  const previousBatchKeyRef = useRef("");
  const previousExportRef = useRef<{ id: string; status: ExportOperation["status"] } | undefined>(undefined);
  useEffect(() => {
    const currentStatuses = new Map<string, JobStatus>();
    for (const session of sessions) currentStatuses.set(session.id, session.status);
    for (const job of jobs) currentStatuses.set(job.id, job.status);
    const notify = (payload: { title: string; body: string } | null) => {
      if (!payload || !settings.desktopNotifications) return;
      void window.electronAPI?.showNotification?.(payload).catch(() => {});
    };

    const batchKey = batchJobIds.join("|");
    if (batchKey !== previousBatchKeyRef.current) {
      previousBatchKeyRef.current = batchKey;
      previousBatchActiveRef.current = false;
    }
    const batchStatuses = batchJobIds.map((id) => currentStatuses.get(id));
    const batchActive = batchStatuses.some(isActiveScanStatus);
    const batchSettled = batchStatuses.length > 0 && batchStatuses.every(isTerminalScanStatus);
    if (batchJobIds.length > 1 && previousBatchActiveRef.current && batchSettled) {
      notify(batchNotification(batchStatuses.filter((status): status is JobStatus => !!status)));
    }

    const batchIds = new Set(batchJobIds);
    for (const [id, status] of currentStatuses) {
      const previous = previousScanStatusesRef.current.get(id);
      if (!isActiveScanStatus(previous) || !isTerminalScanStatus(status)) continue;
      if (batchJobIds.length > 1 && batchIds.has(id)) continue;
      const name = sessions.find((session) => session.id === id)?.name
        || jobs.find((job) => job.id === id)?.url
        || "Your recording";
      notify(scanNotification(name, status));
    }

    const previousExport = previousExportRef.current;
    if (exportOperation
      && previousExport?.id === exportOperation.id
      && (previousExport.status === "queued" || previousExport.status === "running")
      && !["queued", "running"].includes(exportOperation.status)) {
      notify(exportNotification(exportOperation));
    }

    previousScanStatusesRef.current = currentStatuses;
    previousBatchActiveRef.current = batchActive;
    previousExportRef.current = exportOperation
      ? { id: exportOperation.id, status: exportOperation.status }
      : undefined;
  }, [jobs, sessions, batchJobIds, exportOperation, settings.desktopNotifications]);

  useEffect(() => {
    try { localStorage.setItem("recall-second-look-sessions", JSON.stringify([...secondLookSessions])); } catch {}
  }, [secondLookSessions]);

  // Retire a batch left over from a previous run. The queue page is a live
  // tracker, not a history: once a restored batch has no scan still running or
  // waiting, it stops being the thing to watch and the page returns to its
  // empty state. Batches enqueued in THIS run are never auto-retired, so an
  // overnight queue still reads as finished when the creator comes back to it.
  // The sessions themselves are untouched — they stay in Sessions and Library.
  const batchRetireChecked = useRef(false);
  useEffect(() => {
    if (batchRetireChecked.current) return;
    if (!batchRestored || !batchJobIds.length) return;
    // Wait for the rehydrate to land, or every id looks (wrongly) unknown.
    if (!sessions.length && !jobs.length) return;
    // Decide once, on the first load that has data. A restored batch still
    // running is one the creator came back to watch — it must not evaporate
    // under them the moment its last scan lands.
    batchRetireChecked.current = true;
    const settled = batchJobIds.every((id) => {
      const lane = queueLane(
        jobs.find((job) => job.id === id),
        sessions.find((session) => session.id === id),
      );
      return lane === "done" || lane === "failed" || lane === "cancelled";
    });
    if (settled) clearBatchJobIds();
  }, [batchRestored, batchJobIds, sessions, jobs, clearBatchJobIds]);

  useEffect(() => {
    const app = document.getElementById("app-container");
    if (!app) return;
    const modalOpen = projectModal || !!deleteTarget || bulkDeleteTargets.length > 0 || settingsOpen || onboardingOpen;
    app.inert = modalOpen;
    if (modalOpen) app.setAttribute("aria-hidden", "true");
    else app.removeAttribute("aria-hidden");
    return () => {
      app.inert = false;
      app.removeAttribute("aria-hidden");
    };
  }, [projectModal, deleteTarget, bulkDeleteTargets, settingsOpen, onboardingOpen]);

  useEffect(() => {
    void loadSessions();
    void loadRemoteSettings();
  }, [loadSessions, loadRemoteSettings]);

  // Active jobs already stream progress over SSE. Poll only the lightweight job
  // index as a recovery lane; clip libraries hydrate at startup/focus/completion.
  const hasActiveSession = sessions.some((session) => !terminal.has(session.status));
  useEffect(() => {
    if (!hasActiveSession) return;
    const timer = window.setInterval(() => { void refreshJobs(); }, 5000);
    return () => window.clearInterval(timer);
  }, [hasActiveSession, refreshJobs]);

  useEffect(() => () => disposeAllJobSubscriptions(), []);

  useEffect(() => {
    let lastRefreshAt = 0;
    const refresh = () => {
      const now = Date.now();
      if (document.visibilityState !== "visible" || now - lastRefreshAt < 1000) return;
      lastRefreshAt = now;
      void loadSessions();
    };
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [loadSessions]);

  useEffect(() => {
    const prevent = (event: DragEvent) => event.preventDefault();
    const drop = (event: DragEvent) => {
      event.preventDefault();
      const file = event.dataTransfer?.files?.[0] as (File & { path?: string }) | undefined;
      if (!file?.path) {
        addLog("Drop recordings into the Recall desktop app, or paste a Twitch VOD link.", "warn");
        return;
      }
      if (!/\.(mp4|mkv|mov|webm|avi|m4v)$/i.test(file.path)) {
        addLog("That file type is not supported. Choose an MP4, MKV, MOV, WebM, AVI, or M4V recording.", "warn");
        return;
      }
      setSourceKind("local");
      setSource(file.path);
      setProjectName(file.name.replace(/\.[^.]+$/, ""));
      setView("add-vod");
    };
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", drop);
    };
  }, []);

  // Probe the selected source for real duration + poster metadata. Twitch uses
  // the bundled downloader's lightweight metadata/preview path.
  useEffect(() => {
    // Creator identity is a per-source decision. A previous VOD's confirmation
    // must never carry into the next scan.
    setVtuberConfirmed(false);
  }, [source]);

  useEffect(() => {
    setProbeMeta(null);
    setProbeError(false);
    if (!source) return;
    const controller = new AbortController();
    setProbing(true);
    apiFetch("/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_path: source, source_type: sourceKind === "twitch" ? "twitch" : "file" }),
      signal: controller.signal,
    }, settings.apiEndpoint)
      .then((res) => {
        if (!res.ok) throw new Error(`Source probe failed (${res.status})`);
        return res.json();
      })
      .then((data) => {
        if (controller.signal.aborted) return;
        setProbeMeta(data);
        setProbeError(!(data?.thumbnail && Number(data?.duration) > 0));
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setProbeMeta(null);
          setProbeError(true);
        }
      })
      .finally(() => { if (!controller.signal.aborted) setProbing(false); });
    return () => controller.abort();
  }, [source, sourceKind, settings.apiEndpoint, probeAttempt]);

  useEffect(() => {
    if (!activeSession || !["review", "theater", "cutting"].includes(view)) return;
    let cancelled = false;
    fetchReactionTimeline(activeSession.id).then((value) => {
      if (!cancelled) setTimeline(value);
    });
    return () => { cancelled = true; };
  }, [activeSession?.id, view]);

  useEffect(() => {
    if (!activeSession || !["review", "theater"].includes(view)) return;
    const sessionId = activeSession.id;
    let cancelled = false;
    // Status-only: do not flip moreCandidatesBusy (that flag is for reveal/render).
    apiFetch(`/jobs/${sessionId}/clips/more`, undefined, settings.apiEndpoint)
      .then((response) => response.ok ? response.json() : null)
      .then((data) => {
        if (cancelled || !data) return;
        setMoreCandidateStatus((current) => ({
          ...current,
          [sessionId]: {
            available: Number(data.available || 0),
            availableTotal: Number(data.available_total || 0),
            loaded: Number(data.loaded || 0),
          },
        }));
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [activeSession?.id, view, settings.apiEndpoint]);

  useEffect(() => {
    if (!currentJob) return;
    // Batch queue owns navigation while it is open. Individual job updates
    // refresh the selected detail pane without pulling the creator into the
    // single-scan screen or opening review when the first queued job finishes.
    if (view === "queue") {
      return;
    }
    // Mid-scan: only pull into processing if the creator is still on the import
    // step. Leaving to Home/Sessions must stay allowed.
    if (!terminal.has(currentJob.status)) {
      if (view === "add-vod") setView("processing");
      return;
    }
    // Completion auto-advance only from the live scan screen. Including add-vod
    // here re-triggers after a finished job and blocks starting a second session.
    if (currentJob.status !== "completed" || view !== "processing") return;
    // If the processing screen is showing a different live job (queue bridge),
    // don't steal focus when the focused/current job completes.
    const watchingId = selectedSessionId ?? currentSessionId;
    if (watchingId && watchingId !== currentJob.id) return;
    const cards = Array.from(document.querySelectorAll<HTMLElement>("#view-processing .prg-mini[data-clip-id]"))
      .map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          clipId: element.dataset.clipId || "",
          rect: { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
        };
      })
      .filter((item) => item.clipId && item.rect.width > 0 && item.rect.height > 0);
    scanRevealRef.current = cards.length ? { jobId: currentJob.id, cards } : null;
    let cancelled = false;
    void hydrateSessionClips(currentJob.id).then(() => {
      if (cancelled) return;
      setSelectedSessionId(currentJob.id);
      setCurrentSessionId(currentJob.id);
      setView("review");
    }).catch((error) => {
      if (cancelled) return;
      addLog(
        error instanceof Error
          ? `The scan finished, but its clips could not be loaded: ${error.message}`
          : "The scan finished, but its clips could not be loaded. Try opening the session again.",
        "warn",
      );
    });
    return () => { cancelled = true; };
  }, [currentJob?.status, currentJob?.id, view, selectedSessionId, currentSessionId, hydrateSessionClips, addLog]);

  useEffect(() => {
    if (processingJob && terminal.has(processingJob.status)) setCancelRequested(false);
  }, [processingJob?.status, processingJob?.id]);

  useLayoutEffect(() => {
    const snapshot = scanRevealRef.current;
    if (view !== "review" || !snapshot || snapshot.jobId !== activeSession?.id) return;
    scanRevealRef.current = null;
    const targets = Array.from(document.querySelectorAll<HTMLElement>("#view-review .rev-card[data-clip-id]"));
    if (!targets.length) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const byId = new Map(snapshot.cards.map((item) => [item.clipId, item.rect]));

    targets.forEach((target, index) => {
      const finish = target.getBoundingClientRect();
      const source = byId.get(target.dataset.clipId || "");
      if (reduceMotion) {
        target.animate([{ opacity: 0.35 }, { opacity: 1 }], {
          duration: 160,
          easing: "cubic-bezier(0.16, 1, 0.3, 1)",
        });
        return;
      }
      if (source && finish.width > 0 && finish.height > 0) {
        const dx = source.left - finish.left;
        const dy = source.top - finish.top;
        const sx = source.width / finish.width;
        const sy = source.height / finish.height;
        // Rare state-transition signature: the live shelf becomes the exact
        // review card the user can act on. WAAPI keeps the FLIP on the compositor.
        target.animate([
          { opacity: 0.72, transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})` },
          { opacity: 1, transform: "translate(0, 0) scale(1, 1)" },
        ], {
          duration: 440,
          delay: Math.min(index, 3) * 34,
          easing: "cubic-bezier(0.77, 0, 0.175, 1)",
        });
      } else {
        target.animate([
          { opacity: 0, transform: "translateY(8px)" },
          { opacity: 1, transform: "translateY(0)" },
        ], {
          duration: 220,
          delay: Math.min(index, 5) * 34,
          easing: "cubic-bezier(0.16, 1, 0.3, 1)",
        });
      }
    });
  }, [view, activeSession?.id]);

  useEffect(() => {
    setCancelRequested(false);
  }, [currentJob?.id]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        document.getElementById("studio-search-input")?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const openSession = async (session: Session) => {
    const requestId = ++openSessionRequestRef.current;
    const startingView = viewRef.current;
    setEditingClip(null);
    setSelectedSessionId(session.id);
    setCurrentSessionId(session.id);
    // Failed / cancelled scans go to the scan view so the end-state panel can
    // explain what happened; they usually have nothing useful in review.
    if (session.status === "failed" || session.status === "cancelled") {
      setSelectedClipId(sortReviewClips(primaryReviewDeckClips(session.clips), reviewSort)[0]?.id);
      setView("processing");
      return;
    }
    if (!terminal.has(session.status)) {
      setSelectedClipId(sortReviewClips(primaryReviewDeckClips(session.clips), reviewSort)[0]?.id);
      setView("processing");
      return;
    }

    let hydratedSession = session;
    if (session.status === "completed" && !session.clipsHydrated) {
      try {
        await hydrateSessionClips(session.id);
        if (
          requestId !== openSessionRequestRef.current
          || viewRef.current !== startingView
          || useJobStore.getState().currentSessionId !== session.id
        ) return;
        hydratedSession = useJobStore.getState().sessions.find((candidate) => candidate.id === session.id) ?? session;
      } catch (error) {
        if (
          requestId !== openSessionRequestRef.current
          || viewRef.current !== startingView
          || useJobStore.getState().currentSessionId !== session.id
        ) return;
        addLog(
          error instanceof Error
            ? `Recall could not load this session's clips: ${error.message}`
            : "Recall could not load this session's clips. Try opening it again.",
          "warn",
        );
        return;
      }
    }
    setSelectedClipId(sortReviewClips(primaryReviewDeckClips(hydratedSession.clips), reviewSort)[0]?.id);
    setView("review");
  };

  const openMemoryClip = async (jobId: string, clipId: string, returnView: View = "memory") => {
    const session = sessions.find((candidate) => candidate.id === jobId);
    if (!session) {
      addLog("That session is no longer in Recall.", "warn");
      return;
    }
    setSelectedSessionId(jobId);
    setCurrentSessionId(jobId);
    try {
      if (!session.clipsHydrated) await hydrateSessionClips(jobId);
    } catch (error) {
      addLog(
        error instanceof Error ? error.message : "Recall could not load that clip.",
        "warn",
      );
      return;
    }
    const hydrated = useJobStore.getState().sessions.find((candidate) => candidate.id === jobId) ?? session;
    const clip = hydrated.clips.find((candidate) => candidate.id === clipId);
    if (!clip) {
      addLog("That clip is no longer available in the session.", "warn");
      return;
    }
    if (clip.isOverflowCandidate) {
      setSecondLookSessions((current) => new Set(current).add(jobId));
    }
    setSelectedClipId(clipId);
    setTheaterReturn(returnView);
    setView("theater");
  };

  const openMemoryMoment = async (result: { id: string; job_id: string; start_time: number; end_time: number }, memoryQuery: string) => {
    const session = sessions.find((candidate) => candidate.id === result.job_id);
    if (!session) {
      addLog("That session is no longer in Recall.", "warn");
      return;
    }
    setSelectedSessionId(result.job_id);
    setCurrentSessionId(result.job_id);
    setMemoryEditorHandoff({
      entryId: result.id,
      timestamp: result.start_time,
      endTime: result.end_time,
      query: memoryQuery,
    });
    setCuttingReturn("memory");
    setView("cutting");
  };

  // Views that render a specific session; deleting the session they point at
  // leaves them with nothing to show, so we fall back to the sessions list.
  const SESSION_SCOPED_VIEWS = useMemo(
    () => new Set<View>(["processing", "review", "theater", "cutting", "clips", "reel", "insights"]),
    [],
  );

  const removeSessions = useCallback(async (targets: Session[]) => {
    const ids = targets.map((session) => session.id);
    if (!ids.length) return;
    const deletedIds: string[] = [];
    for (const id of ids) {
      try {
        await deleteSession(id);
        deletedIds.push(id);
      } catch (error) {
        addLog(error instanceof Error ? error.message : "Recall could not delete this session.", "warn");
      }
    }
    if (!deletedIds.length) return;
    // Drop local selection only after the backend confirms deletion. A 409 for
    // an active scan must leave the visible session and subscription intact.
    const wasViewing =
      (selectedSessionId && deletedIds.includes(selectedSessionId)) ||
      (reelSessionId && deletedIds.includes(reelSessionId));
    if (selectedSessionId && deletedIds.includes(selectedSessionId)) setSelectedSessionId(undefined);
    if (reelSessionId && deletedIds.includes(reelSessionId)) setReelSessionId(undefined);
    if (wasViewing && SESSION_SCOPED_VIEWS.has(view)) setView("sessions");
    addLog(
      deletedIds.length === 1 ? "Session deleted." : `${deletedIds.length} sessions deleted.`,
      "success",
    );
  }, [selectedSessionId, reelSessionId, view, deleteSession, addLog, SESSION_SCOPED_VIEWS]);

  const confirmDeleteSession = async () => {
    if (!deleteTarget) return;
    const target = deleteTarget;
    setDeleteTarget(null);
    await removeSessions([target]);
  };

  const confirmDeleteMany = async () => {
    if (!bulkDeleteTargets.length) return;
    const targets = bulkDeleteTargets;
    setBulkDeleteTargets([]);
    await removeSessions(targets);
  };

  const openProject = (kind: "local" | "twitch" | "batch") => {
    // Detach from the previous session so a finished currentJob can't own the
    // new import flow (selection otherwise keeps pointing at the last review).
    setSelectedSessionId(undefined);
    setEditingClip(null);
    if (kind === "batch") {
      setImportMode("batch");
      setSourceKind("twitch");
      setProjectName("");
      setSource("");
      setTwitchUrl("");
      setBatchRows([]);
      setBatchPaste("");
      setView("add-vod");
      return;
    }
    setImportMode("single");
    setSourceKind(kind);
    setProjectName("");
    setProjectModal(true);
  };

  const confirmProject = (event: FormEvent) => {
    event.preventDefault();
    setProjectName((value) => value.trim() || "Untitled highlight project");
    setProjectModal(false);
    setImportMode("single");
    setSource("");
    setTwitchUrl("");
    setProbeMeta(null);
    setProbeError(false);
    setSelectedSessionId(undefined);
    setView("add-vod");
  };

  const setImportModeSafe = (mode: ImportMode) => {
    setImportMode(mode);
    if (mode === "batch") {
      setSourceKind("twitch");
      setSource("");
    }
  };

  const probeBatchUrl = useCallback(async (rowId: string, url: string) => {
    setBatchRows((rows) =>
      rows.map((row) =>
        row.id === rowId ? { ...row, probing: true, probeError: false } : row,
      ),
    );
    try {
      const res = await apiFetch("/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_path: url, source_type: "twitch" }),
      }, settings.apiEndpoint);
      if (!res.ok) throw new Error(`probe failed (${res.status})`);
      const data = (await res.json()) as ProbeMeta;
      const ready = !!(data?.thumbnail && Number(data?.duration) > 0);
      setBatchRows((rows) =>
        rows.map((row) => {
          if (row.id !== rowId) return row;
          const title = resolveBatchSessionName(row.title, row.titleEdited, data, url);
          return {
            ...row,
            probing: false,
            probeError: !ready,
            probeMeta: data,
            title,
          };
        }),
      );
    } catch {
      setBatchRows((rows) =>
        rows.map((row) =>
          row.id === rowId
            ? { ...row, probing: false, probeError: true, probeMeta: null }
            : row,
        ),
      );
    }
  }, [settings.apiEndpoint]);

  const addBatchUrls = (text: string) => {
    const urls = parseTwitchVodUrls(text);
    if (!urls.length) {
      addLog("Paste full Twitch VOD URLs, one per line (twitch.tv/videos/…).", "warn");
      return;
    }
    const existing = new Set(batchRows.map((row) => row.url));
    const fresh = urls.filter((url) => !existing.has(url));
    if (!fresh.length) {
      addLog("Those VODs are already in the queue list.", "info");
      setBatchPaste("");
      return;
    }
    const created: BatchRow[] = fresh.map((url) => ({
      id: crypto.randomUUID(),
      url,
      title: sessionNameFromProbe(null, url),
      titleEdited: false,
      probing: true,
      probeError: false,
      probeMeta: null,
      vtuberConfirmed: false,
    }));
    setBatchRows((rows) => [...rows, ...created]);
    setBatchPaste("");
    for (const row of created) void probeBatchUrl(row.id, row.url);
  };

  const beginBatchQueue = async () => {
    const ready = batchRows.filter(
      (row) =>
        !row.probing &&
        !row.probeError &&
        row.probeMeta?.thumbnail &&
        Number(row.probeMeta.duration) > 0,
    );
    if (!ready.length || batchStarting) {
      addLog("Add at least one ready Twitch VOD before starting the queue.", "info");
      return;
    }
    setBatchStarting(true);
    try {
      const ids = await startJobBatch(
        ready.map((row) => ({
          url: row.url,
          name: row.title.trim() || sessionNameFromProbe(row.probeMeta, row.url),
          vtuberMode: row.vtuberConfirmed ? "confirmed" as const : "off" as const,
          sourceDate: row.probeMeta?.source_date || undefined,
        })),
      );
      if (ids.length) {
        setBatchRows([]);
        setBatchPaste("");
        setView("queue");
      }
    } finally {
      setBatchStarting(false);
    }
  };

  const browseSource = async () => {
    const path = await window.electronAPI?.openFileDialog?.();
    if (path) {
      setSource(path);
      setSourceKind("local");
    } else if (!window.electronAPI) {
      addLog("Choose recordings from the Recall desktop app, or paste a Twitch VOD link.", "warn");
    }
  };

  const fetchTwitch = () => {
    const value = twitchUrl.trim();
    if (!/^https?:\/\/(?:www\.)?twitch\.tv\/videos\/\d+(?:[/?#].*)?$/i.test(value)) {
      addLog("Paste a full Twitch VOD URL, such as twitch.tv/videos/123456789.", "warn");
      return;
    }
    setSource(value);
    setSourceKind("twitch");
  };

  const beginScan = () => {
    if (!source || probing || probeError || !probeMeta?.thumbnail || !(Number(probeMeta.duration) > 0)) {
      addLog("Wait for Recall to finish checking the source before starting the scan.", "info");
      return;
    }
    // Clear any prior session selection so the new live job owns the scan view
    // (selectedSessionId otherwise wins over currentJob and can show stale data).
    setSelectedSessionId(undefined);
    setCancelRequested(false);
    startJob(source, projectName || undefined, {
      vtuberMode: vtuberConfirmed ? "confirmed" : "off",
      sourceDate: probeMeta.source_date || undefined,
      sourceType: sourceKind === "twitch" ? "twitch" : "file",
    });
    setView("processing");
  };

  const beginRecallSessionScan = async (
    path: string,
    recallSessionId: string,
    sourceType: "file" | "twitch" = "file",
  ) => {
    setSelectedSessionId(undefined);
    setCancelRequested(false);
    // vodStartedAtUtc is deliberately not sent: for a Twitch source the engine
    // resolves the VOD's real start from its metadata, which beats anything
    // this UI could ask the creator to type.
    await startJob(path, undefined, {
      sourceType,
      recallSessionId,
    });
    setView("processing");
  };

  const cancelJobById = async (jobId: string) => {
    try {
      const response = await apiFetch(`/jobs/${jobId}/cancel`, { method: "POST" }, settings.apiEndpoint);
      if (!response.ok) throw new Error(`cancel failed (${response.status})`);
      addLog("Stopping that scan after the current operation finishes.", "info");
      return true;
    } catch {
      addLog("Recall could not request cancellation for that scan.", "warn");
      return false;
    }
  };

  const cancelScan = async () => {
    const jobId = processingJob?.id ?? currentJob?.id;
    if (!jobId || cancelRequested || terminal.has(processingJob?.status ?? "")) return;
    setCancelRequested(true);
    const ok = await cancelJobById(jobId);
    if (!ok) setCancelRequested(false);
  };

  const retryScan = (job?: Job) => {
    const retrySource = job?.url || activeSession?.sourceUrl || "";
    if (!retrySource) {
      openProject("local");
      return;
    }
    const draft = buildScanRetryDraft(retrySource, activeSession?.name);
    setImportMode("single");
    setSourceKind(draft.sourceKind);
    setSource(draft.source);
    setTwitchUrl(draft.twitchUrl);
    setProjectName(draft.projectName);
    setSelectedSessionId(undefined);
    setCancelRequested(false);
    setProbeMeta(null);
    setProbeError(false);
    // Re-run validation even when this source is already in the import form.
    setProbeAttempt((attempt) => attempt + 1);
    setView("add-vod");
  };

  const toggleKeep = (clip: Clip) => {
    if (!activeSession) return;
    if (!activeSession.savedClipIds.includes(clip.id)) setClipPassed(activeSession.id, clip.id, false);
    if (clip.maybe) setClipMaybe(activeSession.id, clip.id, false);
    toggleSaveClip(activeSession.id, clip.id);
  };

  const cutClip = (clip: Clip) => {
    if (!activeSession) return;
    if (activeSession.savedClipIds.includes(clip.id)) toggleSaveClip(activeSession.id, clip.id);
    if (clip.maybe) setClipMaybe(activeSession.id, clip.id, false);
    setClipPassed(activeSession.id, clip.id, true);
  };

  const toggleMaybe = (clip: Clip) => {
    if (!activeSession) return;
    if (activeSession.savedClipIds.includes(clip.id)) toggleSaveClip(activeSession.id, clip.id);
    if (clip.passed) setClipPassed(activeSession.id, clip.id, false);
    setClipMaybe(activeSession.id, clip.id, !clip.maybe);
  };

  // Theater decisions are idempotent sets (not toggles) so a second Space/click
  // during weak feedback cannot accidentally un-keep a clip.
  const keepClip = (clip: Clip) => {
    const sessionId = activeSession?.id;
    if (!sessionId) return;
    const session = useJobStore.getState().sessions.find((entry) => entry.id === sessionId);
    if (!session) return;
    setClipPassed(session.id, clip.id, false);
    if (clip.maybe) setClipMaybe(session.id, clip.id, false);
    if (!session.savedClipIds.includes(clip.id)) toggleSaveClip(session.id, clip.id);
  };

  const passClip = (clip: Clip) => {
    const sessionId = activeSession?.id;
    if (!sessionId) return;
    const session = useJobStore.getState().sessions.find((entry) => entry.id === sessionId);
    if (!session) return;
    if (session.savedClipIds.includes(clip.id)) toggleSaveClip(session.id, clip.id);
    if (clip.maybe) setClipMaybe(session.id, clip.id, false);
    setClipPassed(session.id, clip.id, true);
  };

  const maybeClip = (clip: Clip) => {
    const sessionId = activeSession?.id;
    if (!sessionId) return;
    const session = useJobStore.getState().sessions.find((entry) => entry.id === sessionId);
    if (!session) return;
    if (session.savedClipIds.includes(clip.id)) toggleSaveClip(session.id, clip.id);
    if (clip.passed) setClipPassed(session.id, clip.id, false);
    if (!clip.maybe) setClipMaybe(session.id, clip.id, true);
  };

  // Which grid the embedded theater player returns to on "Back".
  const openTheater = (clip: Clip, from: View = "review") => {
    setTheaterReturn(from);
    setSelectedClipId(clip.id);
    setView("theater");
  };

  const toggleSecondLook = async () => {
    if (!activeSession || moreCandidatesBusy === activeSession.id) return;
    const sessionId = activeSession.id;
    if (secondLookEnabled) {
      setSecondLookSessions((current) => {
        const next = new Set(current);
        next.delete(sessionId);
        return next;
      });
      if (selectedClip?.isOverflowCandidate) {
        setSelectedClipId(sortReviewClips(originalActiveClips, reviewSort)[0]?.id);
      }
      return;
    }

    let revealed = extraActiveClips;
    if (!revealed.length) {
      setMoreCandidatesBusy(sessionId);
      try {
        const result = await loadMoreCandidates(sessionId);
        revealed = result.clips;
        setMoreCandidateStatus((current) => ({
          ...current,
          [sessionId]: {
            available: result.available,
            availableTotal: result.available_total,
            loaded: result.loaded,
          },
        }));
      } catch (error) {
        addLog(error instanceof Error ? error.message : "Recall could not load more moments.", "warn");
        return;
      } finally {
        setMoreCandidatesBusy((current) => current === sessionId ? undefined : current);
      }
    }
    if (!revealed.length) {
      addLog("There are no more strong candidates saved for this session.", "info");
      return;
    }
    setSecondLookSessions((current) => new Set(current).add(sessionId));
    if (view === "theater") {
      setSelectedClipId(sortReviewClips(revealed, reviewSort)[0]?.id);
    }
    addLog(`${revealed.length} second-look moment${revealed.length === 1 ? "" : "s"} added to this review.`, "success");
  };

  const prepareSelectedPreview = async () => {
    if (!activeSession || !selectedClip?.sourceWindow || preparingPreviewId) return;
    const clipId = selectedClip.id;
    setPreparingPreviewId(clipId);
    try {
      await prepareClipPreview(activeSession.id, clipId);
      addLog("Vertical framing proof ready.", "success");
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Recall could not prepare this framing proof.", "warn");
    } finally {
      setPreparingPreviewId((current) => current === clipId ? undefined : current);
    }
  };

  const exportClipIds = async (ids: string[]) => {
    if (!ids.length || exporting) return;
    const folder = await window.electronAPI?.openFolderDialog?.();
    if (!folder) return;
    setExporting(true);
    try {
      const result = await copyClipsToFolder(ids, folder, {
        preset: settings.exportPreset,
        filenameTemplate: settings.filenameTemplate,
      });
      if (result.succeededClipIds.length || result.failed.length) {
        const succeeded = new Set(result.succeededClipIds);
        const failed = new Set(result.failed.map((entry) => entry.clipId));
        setPickedExportIds((previous) => {
          const next = new Set(previous);
          succeeded.forEach((clipId) => next.delete(clipId));
          failed.forEach((clipId) => next.add(clipId));
          return next;
        });
      }
      if (result.cancelled) {
        addLog(result.error || "Export cancelled.", "warn");
        return;
      }
      if (result.copied) {
        if (result.errors) {
          addLog(
            `${plural(result.copied, "clip")} exported; ${plural(result.errors, "clip")} could not be prepared. Failed clips remain selected for retry.`,
            "warn",
          );
        } else {
          addLog(`${plural(result.copied, "clip")} exported.`, "success");
        }
        setShareHandoff({ folder, count: result.copied });
      } else {
        addLog(result.error || "No clips were exported. Check the destination and try again.", "warn");
      }
    } finally {
      setExporting(false);
    }
  };

  // Theater Review exports every kept clip; the Clip Library exports only the
  // clips the user has left selected (defaults to all kept).
  const exportKept = () => exportClipIds(keptClips.map(({ clip }) => clip.id));
  const exportSelectedClips = () => exportClipIds([...selectedExportIds]);

  const reelSession = useMemo(
    () => sessions.find((session) => session.id === reelSessionId),
    [sessions, reelSessionId],
  );
  const reelKeptClips = useMemo(() => {
    if (!reelSession) return [] as Clip[];
    return reelSession.clips
      .filter((clip) => reelSession.savedClipIds.includes(clip.id))
      .slice()
      .sort((a, b) => (a.start_time ?? 0) - (b.start_time ?? 0));
  }, [reelSession]);

  // clipIds is the reel the creator actually assembled — every kept clip minus
  // the ones they dropped in the storyboard — so it is the authority here, not
  // the session's full kept list.
  const compileSessionReel = async (clipIds: string[]) => {
    if (!reelSession || compilingReel) return;
    const chosen = reelKeptClips.filter((clip) => clipIds.includes(clip.id));
    if (chosen.length < 2) {
      addLog("A highlight reel needs at least 2 clips in it.", "warn");
      return;
    }
    const folder = await window.electronAPI?.openFolderDialog?.();
    if (!folder) return;
    setCompilingReel(true);
    try {
      const stem = `${reelSession.name.replace(/[^\w\- ]+/g, "").trim().slice(0, 40) || "recall"}_reel`;
      const result = await compileReel(
        chosen.map((clip) => clip.id),
        folder,
        stem,
      );
      if (result.error || !result.path) {
        addLog(result.error || "Reel compile failed.", "warn");
        return;
      }
      addLog(
        result.errors
          ? `Highlight reel ready · ${result.clips ?? 0} included, ${result.errors} skipped.`
          : `Highlight reel ready · ${result.clips ?? chosen.length} clips stitched.`,
        result.errors ? "warn" : "success",
      );
    } finally {
      setCompilingReel(false);
    }
  };

  const compileCompilationProject = async (clipIds: string[], title: string) => {
    if (clipIds.length < 2 || compilingReel) return;
    const folder = await window.electronAPI?.openFolderDialog?.();
    if (!folder) return;
    setCompilingReel(true);
    try {
      const stem = `${title.replace(/[^\w\- ]+/g, "").trim().slice(0, 48) || "recall_compilation"}_reel`;
      const result = await compileReel(clipIds, folder, stem);
      if (result.error || !result.path) {
        addLog(result.error || "Compilation export failed.", "warn");
        return;
      }
      addLog(
        result.errors
          ? `Compilation ready · ${result.clips ?? 0} included, ${result.errors} skipped.`
          : `Compilation ready · ${result.clips ?? clipIds.length} clips stitched.`,
        result.errors ? "warn" : "success",
      );
    } finally {
      setCompilingReel(false);
    }
  };

  const totalClips = sessions.reduce((sum, session) => sum + session.clips.length, 0);
  const totalKept = keptClips.length;
  const totalDuration = sessions.reduce((sum, session) => sum + (session.vodDuration ?? 0), 0);
  const keepRate = totalClips ? Math.round(totalKept / totalClips * 100) : 0;
  const bestOpeningScore = Math.max(0, ...originalActiveClips.map(hookScoreOf));
  const displayName = settings.displayName.trim() || "Creator";
  // Both writers of the greeting name — this one and the Settings field — have
  // to persist, because the store seeds `displayName` from this same key on
  // startup (lib/store.ts). Editing in one place and not the other would look
  // like the rename silently reverted on next launch.
  // Re-cut from the theater. editClip re-renders server-side and returns the
  // updated clip, which the store swaps into the session — the <video> is keyed
  // on `clip.videoUrl`, so the new cut loads on its own once that lands.
  const trimClip = async (clip: Clip, start: number, end: number) => {
    if (!activeSession) return;
    await editClip(activeSession.id, clip.id, { start_time: start, end_time: end });
  };

  const saveDisplayName = (value: string) => {
    setSettings({ displayName: value });
    try { localStorage.setItem("recall-display-name", value); } catch {}
  };

  return (
    <>
      <div id="app-container">
        <StudioSidebar
          view={view}
          query={query}
          setQuery={setQuery}
          sessions={filteredSessions}
          activeSessionId={activeSession?.id}
          displayName={displayName}
          onView={(nextView) => { setEditingClip(null); setView(nextView); }}
          onSession={openSession}
          onSettings={() => setSettingsOpen(true)}
        />

        <main>
          <StudioHeader
            view={view}
            sessionName={
              view === "add-vod"
                ? (importMode === "batch" ? "Overnight queue" : (projectName.trim() || "New project"))
                : view === "queue"
                  ? "Scan queue"
                  : view === "reel"
                    ? reelSession?.name
                    : activeSession?.name
            }
            kept={view === "reel" ? reelKeptClips.length : activeSession?.savedClipIds.length ?? 0}
            total={view === "reel" ? reelSession?.clips.length ?? 0 : activeSession?.clips.length ?? 0}
          />
          {/* In the flow under the header, not floating over the content: a
              failure should wait to be read, not slide past the corner of the
              screen. Renders nothing when there is nothing wrong, so it costs
              no space in the normal case. */}
          <ProblemBar />
          {(["add-vod", "processing", "queue", "review"] as View[]).includes(view) && (
            <FlowRail
              view={view === "queue" ? "processing" : view}
              progress={currentJob?.progress ?? 0}
            />
          )}

          <div className={`view-panel ${view === "home" ? "active" : ""}`} id="view-home">
            <HomeView
              displayName={displayName}
              onRename={saveDisplayName}
              sessions={sortedSessions}
              totalKept={totalKept}
              keepRate={keepRate}
              totalDuration={totalDuration}
              onNew={openProject}
              onSession={openSession}
              onDelete={setDeleteTarget}
              onOpenLive={() => setView("live")}
            />
          </div>

          <div className={`view-panel ${view === "live" ? "active" : ""}`} id="view-live">
            {view === "live" && <RecallLiveView onScanRecording={beginRecallSessionScan} sessions={sessions} />}
          </div>

          <div className={`view-panel ${view === "sessions" ? "active" : ""}`} id="view-sessions">
            <SessionsView sessions={sortedSessions} onSession={openSession} onDelete={setDeleteTarget} onDeleteMany={setBulkDeleteTargets} onNew={openProject} />
          </div>

          <div className={`view-panel ${view === "memory" ? "active" : ""}`} id="view-memory">
            {view === "memory" && (
              <StreamMemory
                apiEndpoint={settings.apiEndpoint}
                onOpenClip={openMemoryClip}
                onOpenMoment={openMemoryMoment}
                onOpenCompilations={() => setView("compilations")}
              />
            )}
          </div>

          <div className={`view-panel ${view === "compilations" ? "active" : ""}`} id="view-compilations">
            {view === "compilations" && (
              <CompilationProjects
                apiEndpoint={settings.apiEndpoint}
                compiling={compilingReel}
                onCompile={compileCompilationProject}
                onOpenClip={(jobId, clipId) => openMemoryClip(jobId, clipId, "compilations")}
              />
            )}
          </div>

          <div className={`view-panel ${view === "add-vod" ? "active" : ""}`} id="view-add-vod">
            <ImportView
              importMode={importMode}
              onImportMode={setImportModeSafe}
              source={source}
              sourceKind={sourceKind}
              twitchUrl={twitchUrl}
              setTwitchUrl={setTwitchUrl}
              processingMode={settings.processingMode}
              probeMeta={probeMeta}
              probing={probing}
              probeError={probeError}
              vtuberConfirmed={vtuberConfirmed}
              onVtuberConfirmed={setVtuberConfirmed}
              onBrowse={browseSource}
              onFetch={fetchTwitch}
              onClear={() => setSource("")}
              onRetry={() => setProbeAttempt((attempt) => attempt + 1)}
              onStart={beginScan}
              batchRows={batchRows}
              batchPaste={batchPaste}
              setBatchPaste={setBatchPaste}
              batchStarting={batchStarting}
              onAddBatchUrls={addBatchUrls}
              onBatchTitle={(id, title) =>
                setBatchRows((rows) =>
                  rows.map((row) => (row.id === id ? { ...row, title, titleEdited: true } : row)),
                )
              }
              onBatchVtuber={(id, confirmed) =>
                setBatchRows((rows) =>
                  rows.map((row) => (row.id === id ? { ...row, vtuberConfirmed: confirmed } : row)),
                )
              }
              onRemoveBatchRow={(id) =>
                setBatchRows((rows) => rows.filter((row) => row.id !== id))
              }
              onRetryBatchRow={(row) => void probeBatchUrl(row.id, row.url)}
              onStartBatch={beginBatchQueue}
            />
          </div>

          <div className={`view-panel ${view === "processing" ? "active" : ""}`} id="view-processing">
            <StudioProcessingView job={processingJob} onCancel={cancelScan} onRetry={() => retryScan(processingJob)} cancelRequested={cancelRequested} />
          </div>

          <div className={`view-panel ${view === "queue" ? "active" : ""}`} id="view-queue">
            <StudioQueueView
              jobIds={batchJobIds}
              jobs={jobs}
              sessions={sessions}
              onOpenSession={openSession}
              onOpenProcessing={(session) => {
                setSelectedSessionId(session.id);
                setCurrentSessionId(session.id);
                setCancelRequested(false);
                setView("processing");
              }}
              onCancelJob={cancelJobById}
              onDone={() => setView("sessions")}
              onNew={openProject}
              onClearQueue={() => {
                clearBatchJobIds();
                addLog("Scan queue cleared. Your sessions are still in Sessions.", "info");
              }}
            />
          </div>

          <div className={`view-panel ${view === "review" ? "active" : ""}`} id="view-review">
            <StudioReviewView
              session={visibleActiveSession}
              timeline={timeline}
              filter={reviewFilter}
              setFilter={setReviewFilter}
              sort={reviewSort}
              setSort={setReviewSort}
              onOpen={openTheater}
              onKeep={toggleKeep}
              onCut={cutClip}
              onMaybe={toggleMaybe}
              onClips={() => setView("clips")}
              onCuttingRoom={() => {
                setMemoryEditorHandoff(null);
                setCuttingReturn("review");
                setView("cutting");
              }}
              bestOpeningScore={bestOpeningScore}
              moreCandidateCount={Math.max(
                extraActiveClips.length,
                activeSession ? moreCandidateStatus[activeSession.id]?.available ?? 0 : 0,
              )}
              moreCandidatesEnabled={secondLookEnabled}
              moreCandidatesLoading={moreCandidatesBusy === activeSession?.id}
              onToggleMoreCandidates={toggleSecondLook}
            />
          </div>

          <div className={`view-panel ${view === "cutting" ? "active" : ""}`} id="view-cutting">
            {/* The Cutting Room: scrub the FULL source VOD along the R(t)
                curve and cut moments Recall missed. Only mounted while active
                so its <video> never streams the multi-GB source (or plays
                audio) from a hidden panel. */}
            {view === "cutting" && (
              <CuttingRoom
                session={activeSession}
                timeline={timeline}
                onBack={() => setView(cuttingReturn)}
                onOpenLibrary={() => setView("clips")}
                initialMoment={memoryEditorHandoff}
              />
            )}
          </div>

          <div className={`view-panel ${view === "theater" ? "active" : ""}`} id="view-theater">
            {/* The designed theater: video stage + 9:16 crop guide, the R(t)
                reaction curve, the detected-highlights filmstrip, and the
                analysis/Keep-Pass sidebar. Only mounted while active so its
                autoplaying <video> can't play audio from a hidden panel. Back
                returns to whichever grid we came from (review or clips). */}
            {view === "theater" && (
              <StudioTheaterView
                session={visibleActiveSession}
                clip={selectedClip}
                clips={orderedActiveClips}
                timeline={timeline}
                videoRef={videoRef}
                onBack={() => setView(theaterReturn)}
                onKeep={keepClip}
                onCut={passClip}
                onMaybe={maybeClip}
                onSelect={setSelectedClipId}
                onExport={exportKept}
                onTrim={trimClip}
                onRename={async (clip, title) => {
                  if (!activeSession) return;
                  await updateClipTitle(activeSession.id, clip.id, title);
                  addLog(`Renamed clip to “${title}”.`, "success");
                }}
                exporting={exporting}
                moreCandidateCount={Math.max(
                  extraActiveClips.length,
                  activeSession ? moreCandidateStatus[activeSession.id]?.available ?? 0 : 0,
                )}
                moreCandidatesEnabled={secondLookEnabled}
                moreCandidatesLoading={moreCandidatesBusy === activeSession?.id}
                preparingPreview={preparingPreviewId === selectedClip?.id}
                previewBusy={!!preparingPreviewId}
                onToggleMoreCandidates={toggleSecondLook}
                onPreparePreview={prepareSelectedPreview}
              />
            )}
          </div>

          <div className={`view-panel ${view === "clips" ? "active" : ""}`} id="view-clips">
            {view === "clips" && <StudioClipsView items={keptClips} selectedIds={selectedExportIds} onToggleSelect={toggleExportSelected} onSelectAll={selectAllExport} onClearSelection={clearExportSelection} onToggleSession={toggleSessionExport} onEdit={(session, clip) => setEditingClip({ session, clip })} onExport={exportSelectedClips} exporting={exporting} onEmptyAction={() => activeSession ? setView("review") : openProject("local")} emptyActionLabel={activeSession ? "Open Theater Review" : "Start a session"} />}
          </div>

          <div className={`view-panel ${view === "reel" ? "active" : ""}`} id="view-reel">
            <StudioReelView
              sessions={sortedSessions}
              session={reelSession}
              clips={reelKeptClips}
              compiling={compilingReel}
              onCompile={compileSessionReel}
              onSelectSession={(sessionId) => {
                setReelSessionId(sessionId || undefined);
                if (!sessionId) return;
                setSelectedSessionId(sessionId);
                setCurrentSessionId(sessionId);
              }}
              onOpenReview={() => { if (reelSession) openSession(reelSession); }}
              onPickSession={() => setView("sessions")}
            />
          </div>

          <div className={`view-panel ${view === "insights" ? "active" : ""}`} id="view-insights">
            <InsightsView sessions={sessions} totalClips={totalClips} totalKept={totalKept} keepRate={keepRate} onStart={() => openProject("local")} />
          </div>
        </main>
      </div>

      {exportOperation && ["queued", "running"].includes(exportOperation.status) && (
        <ExportProgressDock operation={exportOperation} onCancel={cancelExportOperation} />
      )}

      <div id="project-create-modal" className={`project-modal ${projectModal ? "open" : ""}`} onMouseDown={(event) => { if (event.target === event.currentTarget) closeProjectModal(); }}>
        <form ref={createDialogRef} className="project-dialog" onSubmit={confirmProject} role="dialog" aria-modal="true" aria-labelledby="project-create-title" aria-describedby="project-create-description" tabIndex={-1}>
          <div><h2 id="project-create-title">Name your project</h2><p id="project-create-description">{sourceKind === "twitch" ? "You’ll paste the VOD link on the next screen." : "You’ll pick the recording on the next screen."}</p></div>
          <input data-autofocus value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="e.g. Ranked grind — July 10" aria-label="Project name" />
          <div className="project-dialog-actions"><button type="button" className="btn-secondary" onClick={closeProjectModal}>Cancel</button><button type="submit" className="cta-accent"><span>Continue</span><ChevronRight size={13} /></button></div>
        </form>
      </div>

      <div id="project-delete-modal" className={`project-modal ${deleteTarget ? "open" : ""}`} onMouseDown={(event) => { if (event.target === event.currentTarget) closeDeleteModal(); }}>
        <div ref={deleteDialogRef} className="project-dialog" role="alertdialog" aria-modal="true" aria-labelledby="project-delete-title" aria-describedby="project-delete-description" tabIndex={-1}>
          <div><h2 id="project-delete-title">Delete this session?</h2><p id="project-delete-description"><strong>{deleteTarget?.name ?? "This session"}</strong> and all {deleteTarget?.clips.length ?? 0} of its clips will be permanently removed. This can’t be undone.</p></div>
          <div className="project-dialog-actions"><button type="button" className="btn-secondary" data-autofocus onClick={closeDeleteModal}>Cancel</button><button type="button" className="mock-v1-danger-btn" onClick={confirmDeleteSession}><Trash2 size={13} />Delete session</button></div>
        </div>
      </div>

      <div id="project-bulk-delete-modal" className={`project-modal ${bulkDeleteTargets.length ? "open" : ""}`} onMouseDown={(event) => { if (event.target === event.currentTarget) closeBulkDeleteModal(); }}>
        <div ref={bulkDeleteDialogRef} className="project-dialog" role="alertdialog" aria-modal="true" aria-labelledby="project-bulk-delete-title" aria-describedby="project-bulk-delete-description" tabIndex={-1}>
          <div>
            <h2 id="project-bulk-delete-title">Delete {bulkDeleteTargets.length} session{bulkDeleteTargets.length === 1 ? "" : "s"}?</h2>
            <p id="project-bulk-delete-description">
              These sessions and all {bulkDeleteTargets.reduce((total, session) => total + session.clips.length, 0)} of their clips will be permanently removed. This can’t be undone.
            </p>
          </div>
          <div className="project-dialog-actions">
            <button type="button" className="btn-secondary" data-autofocus onClick={closeBulkDeleteModal}>Cancel</button>
            <button type="button" className="mock-v1-danger-btn" onClick={confirmDeleteMany}><Trash2 size={13} />Delete {bulkDeleteTargets.length} session{bulkDeleteTargets.length === 1 ? "" : "s"}</button>
          </div>
        </div>
      </div>

      <OnboardingFlow
        open={onboardingOpen}
        apiEndpoint={settings.apiEndpoint}
        dialogRef={onboardingDialogRef}
        onComplete={() => setOnboardingOpen(false)}
      />

      {settingsOpen && <SettingsModal open tab={settingsTab} setTab={setSettingsTab} onClose={() => setSettingsOpen(false)} onNavigate={(next) => { setSettingsOpen(false); setView(next); }} />}

      {/* Export is only half the job — this hands the finished files off to the
          places they get posted. No accounts are connected; it opens pages. */}
      <ShareHandoffDialog
        open={!!shareHandoff}
        folder={shareHandoff?.folder ?? ""}
        count={shareHandoff?.count ?? 0}
        dialogRef={shareDialogRef}
        onClose={closeShareHandoff}
      />

      {/* Full edit controls stay in Clip Library; Theater also owns the quick
          title change that creators make while judging the moment. */}
      {editingClip && (() => {
        const liveSession = sessions.find((session) => session.id === editingClip.session.id) ?? editingClip.session;
        const liveClip = liveSession.clips.find((entry) => entry.id === editingClip.clip.id) ?? editingClip.clip;
        return (
          <ClipEditor
            session={liveSession}
            clip={liveClip}
            onClose={() => setEditingClip(null)}
            onClipUpdated={(clip) => {
              const nextSession = sessions.find((session) => session.id === liveSession.id) ?? liveSession;
              setEditingClip({
                session: {
                  ...nextSession,
                  clips: nextSession.clips.map((entry) => (entry.id === clip.id ? clip : entry)),
                },
                clip,
              });
            }}
          />
        );
      })()}

    </>
  );
}

function ExportProgressDock({ operation, onCancel }: { operation: ExportOperation; onCancel: () => void }) {
  const percent = Math.round(operation.progress * 100);
  const compilingReel = operation.kind === "reel" && operation.phase === "compiling_reel";
  const activeRender = operation.phase === "rendering" || compilingReel;
  const measuredProgress = compilingReel ? operation.reelProgress : operation.currentClipProgress;
  const hasMeasuredProgress = activeRender && measuredProgress !== undefined;
  const eta = operation.etaSeconds !== undefined && operation.etaSeconds >= 5
    ? operation.etaSeconds
    : undefined;
  const baseDetail = compilingReel
    ? `${plural(operation.totalClips, "clip")} prepared`
    : operation.errors
      ? `${plural(operation.completedClips, "clip")} checked · ${plural(operation.errors, "issue")}`
      : `${operation.completedClips} of ${plural(operation.totalClips, "clip")} ${operation.kind === "reel" ? "prepared" : "complete"}`;
  const measuredDetail = hasMeasuredProgress
    ? ` · ${compilingReel ? "reel" : "current clip"} ${Math.round((measuredProgress ?? 0) * 100)}%`
    : "";
  const etaDetail = eta !== undefined
    ? ` · ${eta < 60 ? "under 1m" : `about ${fmtDurationHuman(Math.ceil(eta / 60) * 60)}`} left`
    : "";
  const detail = `${baseDetail}${measuredDetail}${etaDetail}`;
  return (
    <section className="export-progress-dock" aria-label={operation.kind === "reel" ? "Highlight reel progress" : "Final file export progress"}>
      <span className="sr-only" role="status" aria-live="polite">{operation.message}</span>
      <span className="export-progress-icon" aria-hidden="true">{operation.kind === "reel" ? <Film size={16} /> : <HardDrive size={16} />}</span>
      <div className="export-progress-copy">
        <div>
          <strong>{operation.message}</strong>
          <span className="t-num">{percent > 0 ? `${percent}%` : "Working"}</span>
          <button
            type="button"
            className="export-progress-cancel"
            onClick={onCancel}
            disabled={operation.cancelRequested}
            aria-label={operation.cancelRequested ? "Stopping export" : "Cancel export"}
          >
            <X size={11} aria-hidden="true" />
            {operation.cancelRequested ? "Stopping…" : "Cancel"}
          </button>
        </div>
        <small>{detail}</small>
        <div
          className={`export-progress-track ${activeRender && !hasMeasuredProgress ? "is-indeterminate" : ""}`}
          role="progressbar"
          aria-label={operation.kind === "reel" ? "Highlight reel progress" : "Final file export progress"}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
        >
          <i style={{ "--export-progress": operation.progress } as CSSProperties} />
        </div>
      </div>
    </section>
  );
}

/**
 * Recency for every "Recent" ordering -- sidebar projects, Home, Sessions,
 * the reel picker: when the session was *scanned*, not when it was last
 * touched.
 *
 * updatedAt is bumped by ordinary review work -- keeping or passing a single
 * clip rewrites it -- so sorting on it made the session you were working in
 * jump over newer scans and reshuffle the list under the cursor. createdAt is
 * stable, and it is already the date these rows print.
 */
function sessionRecency(session: Session) {
  return session.createdAt || session.updatedAt;
}

type ProjectSort = "recent" | "name" | "clips";
const SORT_META: Record<ProjectSort, { label: string; next: ProjectSort }> = {
  recent: { label: "Recent", next: "name" },
  name: { label: "Name", next: "clips" },
  clips: { label: "Clips", next: "recent" },
};

function StudioSidebar({ view, query, setQuery, sessions, activeSessionId, displayName, onView, onSession, onSettings }: {
  view: View; query: string; setQuery: (value: string) => void; sessions: Session[]; activeSessionId?: string; displayName: string;
  onView: (view: View) => void; onSession: (session: Session) => void; onSettings: () => void;
}) {
  const accentTheme = useJobStore((state) => state.settings.accentTheme || "ember");
  const [projectsOpen, setProjectsOpen] = useState(true);
  const [sort, setSort] = useState<ProjectSort>("recent");
  const sortedProjects = useMemo(() => {
    const arr = [...sessions];
    if (sort === "name") arr.sort((a, b) => a.name.localeCompare(b.name));
    else if (sort === "clips") arr.sort((a, b) => b.clips.length - a.clips.length);
    else arr.sort((a, b) => sessionRecency(b) - sessionRecency(a));
    return arr;
  }, [sessions, sort]);

  // Highlight Reel owns its session gate so creators can always open the page
  // and choose the project they want to compile.
  const nav = [
    ["home", Home, "Home"],
    // The other way to start work: Home imports a finished recording, this
    // marks moments while the stream is still running. Both are entry points,
    // so they sit together -- and a session that runs for hours needs a
    // surface it can be found on, not a card below the fold.
    ["live", Target, "Recall Live"],
    ["sessions", Film, "Sessions"],
    ["memory", Search, "Stream Memory"],
    ["compilations", Repeat, "Compilations"],
    ["queue", Clock, "Scan Queue"],
    ["review", Video, "Theater Review"],
    ["cutting", Scissors, "VOD Editor"],
    ["clips", Library, "Clips Library"],
    ["reel", Sparkles, "Highlight Reel"],
  ] as const;
  const activeNavIndex = nav.findIndex(([key]) => key === "review" ? ["review", "theater"].includes(view) : key === view);
  return (
    <aside>
      {/*
        BASE_URL, not a bare "/…" path. Vite rewrites asset URLs it finds in
        index.html and CSS, but not string literals inside a component — a
        leading-slash src ships verbatim. The packaged app is loaded with
        `win.loadFile` (electron/main.ts), so "/recall_logo_orange_mark.png"
        resolves against the *drive root* and the mark silently 404s; only dev,
        served over http://localhost, ever showed it. BASE_URL is "/" in dev and
        "./" in the build, so one expression is correct in both. public/splash.html
        already hard-codes "./" for the same reason, which is why the splash mark
        rendered in the packaged build while this one did not.
      */}
      <div className="window-header"><div className="brand"><img className="brand-mark" src={`${import.meta.env.BASE_URL}recall_logo_mark_${accentTheme}.png`} alt="" draggable={false} /><span>RECALL</span><div className="brand-badge">STUDIO</div></div></div>
      <label className="search-trigger"><Search size={14} aria-hidden="true" /><input id="studio-search-input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search sessions..." aria-label="Search sessions" /><span className="kb-shortcut" aria-hidden="true">Ctrl+K</span></label>
      <nav className={`nav-section ${activeNavIndex < 0 ? "has-no-active" : ""}`} style={{ "--nav-index": Math.max(0, activeNavIndex) } as CSSProperties}>
        <span className="nav-active-pill" aria-hidden="true" />
        {nav.map(([key, Icon, label]) => {
          const active = key === "review" ? ["review", "theater"].includes(view) : view === key;
          return (
            <button
              key={key}
              className={`nav-item ${active ? "active" : ""}`}
              aria-current={active ? "page" : undefined}
              onClick={() => onView(key as View)}
            >
              <Icon aria-hidden="true" /><span>{label}</span>
            </button>
          );
        })}
      </nav>
      <div className="sidebar-section-header mock-v1-projects-header">
        <button className="mock-v1-projects-toggle" onClick={() => setProjectsOpen((open) => !open)} aria-expanded={projectsOpen}>
          <ChevronDown className={`folder-arrow ${projectsOpen ? "open" : ""}`} style={{ transform: projectsOpen ? "none" : "rotate(-90deg)" }} />
          <span>Projects</span>
          <span className="mock-v1-projects-count">{sessions.length}</span>
        </button>
        <button className="mock-v1-sort-btn" onClick={() => setSort(SORT_META[sort].next)} title={`Sort by ${SORT_META[sort].label.toLowerCase()} — click to change`}><ArrowUpDown size={12} /><span>{SORT_META[sort].label}</span></button>
      </div>
      {projectsOpen && (
        <div className="project-list">
          {sortedProjects.map((session) => {
            const reviewCount = primaryReviewDeckClips(session.clips).length;
            const created = shortSessionDate(session.createdAt || session.updatedAt);
            return <button key={session.id} className={`thread-item ${activeSessionId === session.id ? "active" : ""}`} onClick={() => onSession(session)}><SessionPoster session={session} className="thread-session-poster" /><span className="thread-session-copy"><span className="thread-session-name">{session.name}</span><span className="thread-count">{reviewCount} {reviewCount === 1 ? "clip" : "clips"}{created ? <><span className="thread-count-dot" aria-hidden="true">·</span>{created}</> : null}</span></span></button>;
          })}
          {!sortedProjects.length && <div className="thread-item mock-v1-thread-empty"><span>{query.trim() ? "No matching sessions" : "No sessions yet"}</span></div>}
        </div>
      )}
      <div className="sidebar-footer">
        <button className={`settings-btn ${view === "insights" ? "is-active" : ""}`} onClick={() => onView("insights")}><BarChart3 size={16} /><span>Insights</span></button>
        <button className="settings-btn" onClick={onSettings}><Settings size={16} /><span>Settings</span></button>
        <div className="user-profile"><div className="avatar">{displayName[0]?.toUpperCase()}</div><div className="user-info"><span className="user-name">{displayName}</span></div></div>
      </div>
    </aside>
  );
}

function StudioHeader({ view, sessionName, kept, total }: { view: View; sessionName?: string; kept: number; total: number }) {
  const labels: Record<string, string> = { "add-vod": "New project", live: "Recall Live", processing: "Live scan", queue: "Scan queue", sessions: "Sessions", memory: "Stream Memory", compilations: "Compilations", clips: "Clips Library", reel: "Highlight Reel", insights: "Insights", review: "Review", theater: "Theater", cutting: "VOD Editor" };
  // The session name only belongs in the topbar on views scoped to one session
  // (import flow, live scan, review, theater, reel). List views name themselves; Home
  // shows nothing.
  const sessionScoped = ["add-vod", "processing", "queue", "review", "theater", "cutting", "reel"].includes(view);
  const listView = ["live", "sessions", "memory", "compilations", "clips", "insights"].includes(view);
  return <header><div className="breadcrumbs">{sessionScoped
    ? <><span className="project-crumb">Highlight Sessions</span><span className="divider">›</span><span className="active session-crumb">{sessionName ?? labels[view]}</span></>
    : listView ? <span className="active session-crumb">{view === "sessions" ? "Recent Sessions" : labels[view]}</span> : null}</div><div className="header-actions">{["review", "theater", "cutting"].includes(view) && <div className="badge-status success"><Check size={12} aria-hidden="true" /><span className="t-num">{kept}</span>/{total} Kept</div>}<div className="win-controls"><button className="win-btn" aria-label="Minimize Recall" title="Minimize" onClick={() => window.electronAPI?.minimize?.()}><Minus aria-hidden="true" /></button><button className="win-btn" aria-label="Maximize or restore Recall" title="Maximize or restore" onClick={() => window.electronAPI?.maximize?.()}><Maximize2 aria-hidden="true" /></button><button className="win-btn close" aria-label="Close Recall" title="Close" onClick={() => window.electronAPI?.close?.()}><X aria-hidden="true" /></button></div></div></header>;
}

function FlowRail({ view, progress }: { view: View; progress: number }) {
  const current = view === "add-vod" ? 1 : view === "processing" ? 3 : 4;
  return <div className="flow-topstrip"><div className="flow-rail">{["Source", "Check", "Scan", "Review"].map((label, index) => { const number = index + 1; const state = number < current ? "is-done" : number === current ? (view === "processing" ? "is-progress" : "is-active") : ""; return <span key={label} style={{ display: "contents" }}><div className={`flow-chip ${state}`}><span className="step-dot"><span className="step-dot-num t-num">{number}</span><svg className="step-dot-check" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7" /></svg></span><span>{label}</span>{label === "Scan" && <span className="flow-chip-pct t-num">{Math.round(progress)}%</span>}</div>{number < 4 && <div className={`flow-link ${number < current ? "is-done" : ""}`} />}</span>; })}</div></div>;
}

function ScanRow({ session, onOpen, onDelete }: { session: Session; onOpen: (session: Session) => void; onDelete: (session: Session) => void }) {
  return <div className="scan-row elev-1 lift"><button type="button" className="scan-row-main" onClick={() => onOpen(session)}><span className="scan-info"><span className="scan-game-thumbnail"><SessionPoster session={session} className="scan-session-poster" /></span><span className="scan-details"><span className="scan-name">{session.name}</span><span className="scan-meta">{fmtClock(session.vodDuration ?? 0)} recording <span className="dot" /> Scanned {relativeTime(session.createdAt || session.updatedAt)}</span></span></span><span className="scan-stats"><span className="scan-clips-count"><span className="scan-count-number">{session.clips.length}</span><span className="scan-count-label">clips found</span></span><span className="btn-secondary scan-row-btn">Open session</span><ChevronRight size={18} aria-hidden="true" /></span></button><button type="button" className="mock-v1-scan-delete" aria-label={`Delete ${session.name}`} title="Delete session" onClick={() => onDelete(session)}><Trash2 size={15} aria-hidden="true" /></button></div>;
}

/**
 * The greeting name, editable in place.
 *
 * The Settings field is the same value, but the Start screen is where the name
 * is actually READ, so it is where a wrong one gets noticed -- and hunting
 * through Settings to fix a typo you are looking straight at reads as "the name
 * is hard-coded". studio.css has carried the `#welcome-username` hover
 * treatment since the original mock; only the wiring was missing.
 *
 * Committed on blur and on Enter, abandoned on Escape. Empty is allowed and
 * means "no name": the greeting falls back to "Creator" exactly as the Settings
 * field's own copy promises.
 */
function EditableName({ value, onCommit }: { value: string; onCommit: (next: string) => void }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);

  if (!editing) {
    return (
      <button
        type="button"
        id="welcome-username"
        className="welcome-username-button"
        title="Click to change your name"
        aria-label={`Your name: ${value}. Click to change.`}
        onClick={() => { setDraft(value === "Creator" ? "" : value); setEditing(true); }}
      >
        {value}
      </button>
    );
  }

  const commit = () => { setEditing(false); onCommit(draft.trim()); };
  return (
    <input
      id="welcome-username"
      className="welcome-username-input"
      autoFocus
      value={draft}
      maxLength={32}
      placeholder="Creator"
      aria-label="Your name"
      // Sized to the text so the 56px display line does not jump on entry.
      style={{ width: `${Math.max(6, draft.length || 7) + 1}ch` }}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") { event.preventDefault(); commit(); }
        if (event.key === "Escape") { event.preventDefault(); setEditing(false); }
      }}
    />
  );
}

/** Recall Live: mark moments while the stream runs, scan the recording after.
 *
 * Its own page because a session lives for the length of a broadcast. On Home
 * it was a card taller than the primary action, and once scrolled past there
 * was nothing telling the creator a session was still recording their marks. */
function RecallLiveView({ onScanRecording, sessions }: {
  onScanRecording: (path: string, recallSessionId: string, sourceType?: "file" | "twitch") => Promise<void> | void;
  sessions: Session[];
}) {
  return (
    <div className="recall-live-shell">
      <div className="recall-live-head">
        <h1>Recall Live</h1>
        <span className="recall-live-alpha">ALPHA</span>
        <span>Press a key while you stream. Recall keeps the second, not the video, and finds the clips in the recording afterwards.</span>
      </div>
      <LiveSessionTestPanel onScanRecording={onScanRecording} sessions={sessions} />
    </div>
  );
}

/** Home's reminder that a session is recording marks somewhere else.
 *
 * Rendered only while one is actually running, so Home stays about starting a
 * project on every other visit. */
function HomeLiveStrip({ onOpen }: { onOpen: () => void }) {
  const [live, setLive] = useState<RecallSession | null>(null);

  useEffect(() => {
    let mounted = true;
    const check = () => {
      void getActiveRecallSession()
        .then((session) => { if (mounted) setLive(session?.status === "active" ? session : null); })
        .catch(() => { if (mounted) setLive(null); });
    };
    check();
    const timer = window.setInterval(check, 15_000);
    return () => { mounted = false; window.clearInterval(timer); };
  }, []);

  if (!live) return null;
  const marks = live.events?.filter((event) => event.kind === "remember").length ?? 0;
  return (
    <button type="button" className="home-live-strip" onClick={onOpen}>
      <span className="home-live-dot" aria-hidden="true" />
      <span className="home-live-copy">
        <strong>A Recall Live session is running</strong>
        <small>{marks ? `${plural(marks, "moment")} marked so far.` : "No moments marked yet."}</small>
      </span>
      <span className="home-live-open">Open Recall Live <ChevronRight size={13} aria-hidden="true" /></span>
    </button>
  );
}

function HomeView({ displayName, onRename, sessions, totalKept, keepRate, totalDuration, onNew, onSession, onDelete, onOpenLive }: {
  displayName: string; onRename: (next: string) => void; sessions: Session[]; totalKept: number; keepRate: number; totalDuration: number;
  onNew: (kind: "local" | "twitch" | "batch") => void; onSession: (session: Session) => void; onDelete: (session: Session) => void;
  onOpenLive: () => void;
}) {
  const recent = sessions.filter((session) => session.status === "completed").slice(0, 4);
  return <div className="welcome-container"><div className="welcome-header-section"><h1 className="welcome-title">Welcome back, <EditableName value={displayName} onCommit={onRename} /></h1><p className="welcome-subtitle">Ready to extract your best gameplay highlights? Drag in your stream recording or fetch a Twitch link to start reviewing clips.</p></div><div className="upload-card"><button type="button" className="upload-icon-container" aria-label="Start a new highlight project" onClick={() => onNew("local")}><Plus /></button><h2 className="upload-title">Start a new highlight project</h2><p className="upload-subtext">Name the project, point Recall at a recording or a Twitch VOD, and it handles the rest. Queue several VODs overnight when you streamed more than once.</p><div className="mock-v1-home-actions"><button className="cta-accent" onClick={() => onNew("local")}><FolderOpen size={14} /><span>Select Recording File</span></button><button className="btn-secondary mock-v1-twitch-btn" onClick={() => onNew("twitch")}><TwitchLogo size={14} /><span>Twitch VOD</span></button><button type="button" className="btn-secondary mock-v1-queue-btn" onClick={() => onNew("batch")}><Clock size={14} /><span>Queue VODs</span></button><button type="button" className="btn-secondary mock-v1-youtube-btn mock-v1-platform-soon" disabled title="YouTube VOD ingest coming later" aria-label="YouTube VOD (coming soon)"><YouTubeLogo size={14} /><span>YouTube VOD</span><em>Soon</em></button><button type="button" className="btn-secondary mock-v1-kick-btn mock-v1-platform-soon" disabled title="Kick VOD ingest coming later" aria-label="Kick VOD (coming soon)"><KickLogo size={14} /><span>Kick VOD</span><em>Soon</em></button></div></div><HomeLiveStrip onOpen={onOpenLive} /><div className="quick-presets-section"><h3 className="presets-title"><BarChart3 size={14} /> System &amp; Library Stats</h3><div className="presets-grid"><Metric value={sessions.filter((s) => s.status === "completed").length} label="VODs Analyzed" /><Metric value={totalKept} label="Highlights Kept" tone="success" /><Metric value={`${keepRate}%`} label="Average Keep Rate" tone="warning" /><Metric value={`${(totalDuration / 3600).toFixed(1)} hrs`} label="Recording Reviewed" tone="info" /></div></div><div className="recent-scans-section"><h3 className="recent-scans-header">Recent Review Sessions</h3><div className="scans-list">{recent.map((session) => <ScanRow key={session.id} session={session} onOpen={onSession} onDelete={onDelete} />)}{!recent.length && <div className="recent-scans-empty"><Target size={16} aria-hidden="true" /><span>Your first review session will land here after Recall finds its moments.</span></div>}</div></div></div>;
}

/**
 * Compact absolute date for the sidebar project list, e.g. "Jul 31".
 *
 * Absolute rather than relative on purpose: Sessions and Home already carry
 * "Scanned 8h ago", and a sidebar full of rolling relative times gives you no
 * way to tell two Tuesdays apart. The year only appears when it is not the
 * current one, so the common case stays short.
 */
function shortSessionDate(value: number) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const thisYear = date.getFullYear() === new Date().getFullYear();
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(thisYear ? {} : { year: "numeric" }),
  });
}

function sessionCreatedLabel(value: number) {
  const hours = Math.max(0, Math.floor((Date.now() - value) / 3_600_000));
  if (hours < 1) return "Created: just now";
  if (hours < 24) return `Created: ${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  return `Created: ${days} day${days === 1 ? "" : "s"} ago`;
}

function SessionsView({ sessions, onSession, onDelete, onDeleteMany, onNew }: {
  sessions: Session[];
  onSession: (session: Session) => void;
  onDelete: (session: Session) => void;
  onDeleteMany: (sessions: Session[]) => void;
  onNew: (kind: "local" | "twitch" | "batch") => void;
}) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<ProjectSort>("recent");
  const [selecting, setSelecting] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const sorted = useMemo(() => {
    const arr = [...sessions].filter((session) =>
      session.name.toLowerCase().includes(query.trim().toLowerCase()),
    );
    if (sort === "name") arr.sort((a, b) => a.name.localeCompare(b.name));
    else if (sort === "clips") arr.sort((a, b) => b.clips.length - a.clips.length);
    else arr.sort((a, b) => sessionRecency(b) - sessionRecency(a));
    return arr;
  }, [sessions, query, sort]);

  // Sessions can disappear (deleted elsewhere) or drop out of the filter, so
  // the selection only ever means "visible and picked".
  const visibleIds = useMemo(() => new Set(sorted.map((session) => session.id)), [sorted]);
  const picked = useMemo(() => sorted.filter((session) => selectedIds.has(session.id)), [sorted, selectedIds]);
  useEffect(() => {
    setSelectedIds((prev) => {
      const next = new Set([...prev].filter((id) => visibleIds.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [visibleIds]);

  const exitSelecting = useCallback(() => {
    setSelecting(false);
    setSelectedIds(new Set());
  }, []);

  const toggleSelected = useCallback((session: Session) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(session.id)) next.delete(session.id);
      else next.add(session.id);
      return next;
    });
  }, []);

  const allPicked = sorted.length > 0 && picked.length === sorted.length;

  return (
    <div className="sessions-shell">
      <div className="sessions-head">
        <h1>Recent Sessions</h1>
        <div className="sessions-head-tools">
          <label className="sessions-search">
            <Search size={15} aria-hidden="true" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search"
              aria-label="Search sessions"
            />
          </label>
          <button
            type="button"
            className="sessions-tool-btn"
            onClick={() => setSort(SORT_META[sort].next)}
            title={`Sort: ${SORT_META[sort].label}`}
            aria-label={`Sort by ${SORT_META[sort].label}. Click to change.`}
          >
            <ArrowUpDown size={15} aria-hidden="true" />
          </button>
          {sessions.length > 0 && (
            <button
              type="button"
              className={`sessions-tool-btn ${selecting ? "is-active" : ""}`}
              onClick={() => (selecting ? exitSelecting() : setSelecting(true))}
              title={selecting ? "Cancel selection" : "Select sessions"}
              aria-pressed={selecting}
              aria-label={selecting ? "Cancel selection" : "Select multiple sessions"}
            >
              {selecting ? <X size={15} aria-hidden="true" /> : <Check size={15} aria-hidden="true" />}
            </button>
          )}
          {sessions.length > 0 && (
            <button type="button" className="cta-accent sessions-new-btn" onClick={() => onNew("local")}>
              <Plus size={14} aria-hidden="true" /><span>New session</span>
            </button>
          )}
        </div>
      </div>

      {selecting && sorted.length > 0 && (
        <div className="sessions-select-bar" role="toolbar" aria-label="Session selection">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => setSelectedIds(allPicked ? new Set() : new Set(sorted.map((session) => session.id)))}
          >
            {allPicked ? "Clear all" : `Select all ${sorted.length}`}
          </button>
          <span className="sessions-select-count" role="status">
            {picked.length} selected
          </span>
          <button
            type="button"
            className="mock-v1-danger-btn"
            disabled={!picked.length}
            onClick={() => { onDeleteMany(picked); exitSelecting(); }}
          >
            <Trash2 size={13} aria-hidden="true" />
            Delete{picked.length ? ` ${picked.length}` : ""}
          </button>
        </div>
      )}

      {!sessions.length ? (
        <div className="clips-empty sessions-empty">
          <StudioEmptyState
            variant="sessions"
            icon={<Film />}
            title="Your first session starts with a VOD"
            body="Recall maps the reaction signal, finds the moments worth reviewing, and keeps each scan organized here."
            action={<button type="button" className="cta-accent" onClick={() => onNew("local")}><Plus size={14} /><span>Start a session</span></button>}
          />
        </div>
      ) : !sorted.length ? (
        <div className="sessions-empty-filter" role="status">
          <Search size={18} aria-hidden="true" />
          <span>No sessions match “{query.trim()}”.</span>
        </div>
      ) : (
        <div className="sessions-gallery">
          <div className="sessions-grid">
            {sorted.map((session) => (
              <SessionCard
                key={session.id}
                session={session}
                onOpen={onSession}
                onDelete={onDelete}
                selecting={selecting}
                selected={selectedIds.has(session.id)}
                onToggleSelected={toggleSelected}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SessionCard({ session, onOpen, onDelete, selecting = false, selected = false, onToggleSelected }: {
  session: Session;
  onOpen: (session: Session) => void;
  onDelete: (session: Session) => void;
  selecting?: boolean;
  selected?: boolean;
  onToggleSelected?: (session: Session) => void;
}) {
  const clipCount = session.clips.length;
  const duration = fmtDurationHuman(session.vodDuration);
  const created = sessionCreatedLabel(session.createdAt || session.updatedAt);
  const statusLabel =
    session.status === "failed" ? "Failed"
      : session.status === "completed" || session.status === "cancelled" || session.status === "idle" ? null
        : "Scanning";

  return (
    <article className={`session-card ${selecting ? "is-selecting" : ""} ${selecting && selected ? "is-selected" : ""}`}>
      <button
        type="button"
        className="session-card-main"
        // While selecting, the card body toggles the checkbox instead of
        // opening the session — clicking a card is the obvious way to pick it.
        aria-pressed={selecting ? selected : undefined}
        onClick={() => (selecting ? onToggleSelected?.(session) : onOpen(session))}
      >
        <span className="session-card-media">
          <SessionPoster session={session} variant="card" />
          <span className="session-card-play" aria-hidden="true"><Play size={18} fill="currentColor" /></span>
          {statusLabel && <span className={`session-card-status is-${session.status === "failed" ? "failed" : "live"}`}>{statusLabel}</span>}
        </span>
        <span className="session-card-body">
          <span className="session-card-title">{session.name}</span>
          <span className="session-card-stats" aria-label="Session details">
            <span className="session-card-stat">
              <Clock size={15} aria-hidden="true" />
              <b>{duration}</b>
            </span>
            <span className="session-card-stat">
              <FolderOpen size={15} aria-hidden="true" />
              <b>{clipCount}</b>
              <small>clip{clipCount === 1 ? "" : "s"}</small>
            </span>
            <span className="session-card-stat is-muted">
              <Film size={14} aria-hidden="true" />
              <span>{created.replace(/^Created:\s*/, "")}</span>
            </span>
          </span>
        </span>
      </button>
      {selecting ? (
        <button
          type="button"
          className={`session-card-check ${selected ? "is-on" : ""}`}
          role="checkbox"
          aria-checked={selected}
          aria-label={`Select ${session.name}`}
          onClick={() => onToggleSelected?.(session)}
        >
          {selected && <Check size={13} aria-hidden="true" />}
        </button>
      ) : (
        <button
          type="button"
          className="session-card-delete"
          aria-label={`Delete ${session.name}`}
          title="Delete session"
          onClick={() => onDelete(session)}
        >
          <Trash2 size={14} aria-hidden="true" />
        </button>
      )}
    </article>
  );
}

function Metric({ value, label, tone = "accent" }: { value: string | number; label: string; tone?: string }) {
  const display = useCountUp(value);
  return <div className="preset-btn mock-v1-metric"><span className={`preset-name tone-${tone}`} aria-label={String(value)}>{display}</span><span className="preset-desc">{label}</span></div>;
}

function ImportView({
  importMode, onImportMode, source, sourceKind, twitchUrl, setTwitchUrl, processingMode, probeMeta, probing, probeError,
  vtuberConfirmed, onVtuberConfirmed,
  onBrowse, onFetch, onClear, onRetry, onStart,
  batchRows, batchPaste, setBatchPaste, batchStarting, onAddBatchUrls, onBatchTitle, onBatchVtuber, onRemoveBatchRow, onRetryBatchRow, onStartBatch,
}: {
  importMode: ImportMode;
  onImportMode: (mode: ImportMode) => void;
  source: string; sourceKind: "local" | "twitch"; twitchUrl: string; setTwitchUrl: (value: string) => void; processingMode: "fast" | "quality";
  probeMeta: ProbeMeta | null; probing: boolean; probeError: boolean;
  vtuberConfirmed: boolean; onVtuberConfirmed: (confirmed: boolean) => void;
  onBrowse: () => void; onFetch: () => void; onClear: () => void; onRetry: () => void; onStart: () => void;
  batchRows: BatchRow[];
  batchPaste: string;
  setBatchPaste: (value: string) => void;
  batchStarting: boolean;
  onAddBatchUrls: (text: string) => void;
  onBatchTitle: (id: string, title: string) => void;
  onBatchVtuber: (id: string, confirmed: boolean) => void;
  onRemoveBatchRow: (id: string) => void;
  onRetryBatchRow: (row: BatchRow) => void;
  onStartBatch: () => void;
}) {
  const scanModeLabel = processingMode === "quality" ? "Best quality" : "Smart scan";
  const sourceReady = !!(source && probeMeta?.thumbnail && Number(probeMeta.duration) > 0 && !probing && !probeError);
  const lengthLabel = probing ? "Loading…" : probeMeta?.duration ? fmtClock(probeMeta.duration) : probeError ? "Unavailable" : "--:--:--";
  const twitchValid = /^https?:\/\/(?:www\.)?twitch\.tv\/videos\/\d+(?:[/?#].*)?$/i.test(twitchUrl.trim());
  const batchReady = batchRows.filter(
    (row) => !row.probing && !row.probeError && row.probeMeta?.thumbnail && Number(row.probeMeta.duration) > 0,
  );
  const batchBusy = batchRows.some((row) => row.probing);
  const batchDraftCount = parseTwitchVodUrls(batchPaste).length;
  const vtuberSuggested = titleSuggestsVtuberMode(probeMeta?.title);

  return (
    <div className="imp-shell">
      <div className={`imp-panel ${importMode === "batch" ? "is-batch" : ""}`}>
        <div className="imp-mode-toggle" role="tablist" aria-label="Import mode">
          <button type="button" role="tab" aria-selected={importMode === "single"} className={importMode === "single" ? "is-active" : ""} onClick={() => onImportMode("single")}>Single</button>
          <button type="button" role="tab" aria-selected={importMode === "batch"} className={importMode === "batch" ? "is-active" : ""} onClick={() => onImportMode("batch")}>Batch</button>
        </div>

        {importMode === "batch" ? (
          <div className="imp-batch-workspace">
            <section className="imp-batch-builder">
              <div className="imp-batch-intro">
                <span className="imp-batch-kicker"><Clock size={13} aria-hidden="true" /> Overnight batch</span>
                <h1>Line up tonight&rsquo;s VODs.</h1>
                <p>Paste one public Twitch VOD link per line. Recall checks each source now, then scans them in order while you&rsquo;re away.</p>
              </div>

              <div className="imp-batch-composer">
                <div className="imp-batch-composer-head">
                  <label htmlFor="batch-vod-urls">Twitch VOD links</label>
                  <span>One URL per line</span>
                </div>
                <textarea
                  id="batch-vod-urls"
                  value={batchPaste}
                  onChange={(event) => setBatchPaste(event.target.value)}
                  placeholder={"https://www.twitch.tv/videos/123456789\nhttps://www.twitch.tv/videos/987654321"}
                  aria-label="Twitch VOD URLs"
                  rows={4}
                />
                <div className="imp-batch-composer-foot">
                  <span className={batchDraftCount ? "has-links" : ""}>
                    <i aria-hidden="true" />
                    {batchDraftCount ? `${batchDraftCount} valid ${batchDraftCount === 1 ? "link" : "links"} detected` : "Paste full twitch.tv/videos links"}
                  </span>
                  <button type="button" className="btn-secondary" onClick={() => onAddBatchUrls(batchPaste)} disabled={!batchPaste.trim()}>
                    <Plus size={14} aria-hidden="true" /><span>Add {batchDraftCount || "to queue"}</span>
                  </button>
                </div>
              </div>

              <div className="imp-batch-list-head">
                <span>VODs in this batch</span>
                <small>{batchRows.length ? `${batchReady.length} of ${batchRows.length} ready` : "Nothing added yet"}</small>
              </div>
              <div className={`imp-batch-list ${batchRows.length ? "has-rows" : "is-empty"}`} aria-label="Queued VODs">
                {!batchRows.length && (
                  <div className="imp-batch-empty">
                    <span className="imp-batch-empty-mark"><Plus size={17} aria-hidden="true" /></span>
                    <div>
                      <strong>Your queue starts here</strong>
                      <p>Add a few links above. Titles, creators, thumbnails, and runtimes appear after Recall checks them.</p>
                    </div>
                  </div>
                )}
                {batchRows.map((row, index) => {
                  const ready = !row.probing && !row.probeError && !!row.probeMeta?.thumbnail && Number(row.probeMeta.duration) > 0;
                  return (
                    <article key={row.id} className={`imp-batch-row ${ready ? "is-ready" : ""} ${row.probeError ? "is-error" : ""}`}>
                      <span className="imp-batch-order t-num">{String(index + 1).padStart(2, "0")}</span>
                      <div className="imp-batch-thumb">
                        {row.probeMeta?.thumbnail ? (
                          <img src={row.probeMeta.thumbnail} alt="" />
                        ) : (
                          <span>{row.probing ? <RefreshCw size={15} aria-hidden="true" /> : "!"}</span>
                        )}
                      </div>
                      <div className="imp-batch-fields">
                        <input
                          value={row.title}
                          onChange={(event) => onBatchTitle(row.id, event.target.value)}
                          aria-label="Session name"
                          maxLength={200}
                        />
                        <small>
                          {row.probing
                            ? "Checking source..."
                            : row.probeError
                              ? "Could not check this VOD"
                              : [row.probeMeta?.creator, row.probeMeta?.duration ? fmtClock(row.probeMeta.duration) : null, "Twitch VOD"].filter(Boolean).join(" · ")}
                        </small>
                      </div>
                      <span className="imp-batch-state">{row.probing ? "Checking" : row.probeError ? "Needs attention" : "Ready"}</span>
                      <div className="imp-batch-actions">
                        <button
                          type="button"
                          className={`imp-batch-vtuber ${row.vtuberConfirmed ? "is-confirmed" : ""}`}
                          aria-pressed={row.vtuberConfirmed}
                          disabled={!ready}
                          title={titleSuggestsVtuberMode(row.probeMeta?.title) && !row.vtuberConfirmed ? "The VOD title suggests VTuber mode; confirm only if it uses a virtual avatar." : "Use the avatar cutout only for a virtual-avatar VOD."}
                          onClick={() => onBatchVtuber(row.id, !row.vtuberConfirmed)}
                        >
                          {row.vtuberConfirmed ? <Check size={12} aria-hidden="true" /> : null}
                          VTuber
                        </button>
                        {row.probeError && (
                          <button type="button" className="btn-secondary" onClick={() => onRetryBatchRow(row)}>
                            <RefreshCw size={13} aria-hidden="true" /> Retry
                          </button>
                        )}
                        <button type="button" className="imp-batch-remove" aria-label={`Remove ${row.title}`} onClick={() => onRemoveBatchRow(row.id)}>
                          <X size={14} aria-hidden="true" />
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>

            <aside className="imp-batch-summary">
              <div className="imp-batch-summary-head">
                <span>Tonight&rsquo;s run</span>
                <small>Sequential processing</small>
              </div>
              <div className="imp-batch-summary-counts">
                <span><b className="t-num">{batchRows.length}</b><small>In queue</small></span>
                <span><b className="t-num">{batchReady.length}</b><small>Ready</small></span>
              </div>
              <div className="imp-batch-run-plan">
                <div className={batchRows.length ? "is-done" : "is-current"}><i>1</i><span><b>Add VODs</b><small>Build the ordered list</small></span></div>
                <div className={batchBusy ? "is-current" : batchReady.length ? "is-done" : ""}><i>2</i><span><b>Check sources</b><small>Read titles and runtimes</small></span></div>
                <div className={batchReady.length && !batchBusy ? "is-current" : ""}><i>3</i><span><b>Scan in order</b><small>One VOD uses the machine at a time</small></span></div>
                <div><i>4</i><span><b>Review tomorrow</b><small>Finished sessions wait in Recall</small></span></div>
              </div>
              <div className="imp-batch-mode">
                <span><small>Scan mode</small><b>{scanModeLabel}</b></span>
                <span><small>Starts with</small><b>{batchRows[0]?.title || "First ready VOD"}</b></span>
              </div>
              <button
                type="button"
                className="cta-accent imp-batch-start"
                disabled={!batchReady.length || batchBusy || batchStarting}
                onClick={onStartBatch}
              >
                <Play size={14} fill="currentColor" aria-hidden="true" />
                <span>{batchStarting ? "Starting queue..." : batchBusy ? "Checking sources..." : `Start ${batchReady.length || ""} ${batchReady.length === 1 ? "scan" : "scans"}`.trim()}</span>
              </button>
              <p className="imp-batch-summary-note">Keep Recall open. You can leave this screen after the batch starts.</p>
            </aside>
          </div>
        ) : (
          <>
            <div className="imp-src-cards">
              <section className={`imp-src-card ${sourceKind === "local" ? "is-active" : ""}`} aria-label="Local recording source">
                <span className="imp-src-icon"><FolderOpen size={17} aria-hidden="true" /></span>
                <span className="imp-src-copy"><strong>Local recording</strong><small>MP4, MKV, MOV, WebM, AVI, or M4V</small></span>
                <span className="imp-src-actions"><button type="button" className="btn-secondary" onClick={onBrowse}>Browse files</button></span>
              </section>
              <section className={`imp-src-card mock-v1-twitch-card ${sourceKind === "twitch" ? "is-active" : ""}`} aria-label="Twitch VOD source">
                <span className="imp-src-icon"><TwitchLogo size={17} /></span>
                <span className="imp-src-copy"><strong>Twitch VOD</strong><small>Paste a public twitch.tv/videos link</small></span>
                <span className="imp-src-actions"><input value={twitchUrl} onChange={(event) => setTwitchUrl(event.target.value)} placeholder="https://twitch.tv/videos/…" aria-label="Twitch VOD URL" aria-invalid={twitchUrl.length > 0 && !twitchValid} /><button type="button" className="btn-secondary" onClick={onFetch} disabled={!twitchValid}>Fetch</button></span>
              </section>
            </div>
            <div className={`imp-stage ${source ? "is-loaded" : ""}`}>
              <div className="imp-stage-empty"><StudioEmptyState compact variant="source" icon={<Video />} title="Give Recall a source" body="Drop a recording anywhere, choose a local file, or fetch a Twitch VOD." action={<button type="button" className="btn-secondary" onClick={onBrowse}><FolderOpen size={14} /><span>Choose recording</span></button>} /></div>
              <div className="imp-loaded"><div className="mock-v1-source-preview">{probeMeta?.thumbnail ? <img src={probeMeta.thumbnail} alt="Selected recording preview" /> : <div className="mock-v1-source-probe"><Film size={34} aria-hidden="true" /><strong>{probing ? "Loading source…" : "Source preview unavailable"}</strong>{probeError && <button type="button" className="btn-secondary" onClick={onRetry}><RefreshCw size={13} aria-hidden="true" /> Retry source check</button>}</div>}</div><div className="imp-file-badge"><b>{probeMeta?.title || (sourceKind === "twitch" ? "Twitch VOD" : sourceName(source))}</b><span>{sourceKind === "local" ? "Local recording" : [probeMeta?.creator, probeMeta?.game, "Twitch VOD"].filter(Boolean).join(" · ")}</span></div><button type="button" className="imp-change-btn" onClick={onClear}><X size={11} aria-hidden="true" /> Change source</button></div>
            </div>
            {source && (
              <label className={`imp-vtuber-option ${vtuberConfirmed ? "is-confirmed" : ""}`}>
                <input
                  type="checkbox"
                  checked={vtuberConfirmed}
                  disabled={!sourceReady}
                  onChange={(event) => onVtuberConfirmed(event.target.checked)}
                  aria-describedby="vtuber-mode-description"
                />
                <span className="imp-vtuber-check" aria-hidden="true"><i /></span>
                <span className="imp-vtuber-copy">
                  <span className="imp-vtuber-title">
                    <strong>VTuber / PNGtuber cutout</strong>
                    {vtuberSuggested && <em>Suggested from VOD title</em>}
                  </span>
                  <small id="vtuber-mode-description">Move a virtual avatar to the top of vertical clips. Leave this off for a real facecam.</small>
                </span>
                <span className="imp-vtuber-state">{vtuberConfirmed ? "Confirmed" : "Off"}</span>
              </label>
            )}
            <div className="imp-meta-row">
              <div className={`imp-meta-chip ${source ? "is-filled" : ""}`}><small>Length</small><b className="t-num">{lengthLabel}</b></div>
              <div className={`imp-meta-chip ${source ? "is-filled" : ""}`}><small>Source</small><b>{source ? (sourceKind === "local" ? "Local" : "Twitch") : "--"}</b></div>
              <div className="imp-meta-chip is-filled"><small>Scan mode</small><b>{scanModeLabel}</b></div>
              <button type="button" className="cta-accent imp-start" disabled={!sourceReady} onClick={onStart}><Play size={14} fill="currentColor" aria-hidden="true" /><span>{probing ? "Loading source…" : "Start scan"}</span></button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function queueLane(job: Job | undefined, session: Session | undefined): "running" | "waiting" | "done" | "failed" | "cancelled" {
  const status = job?.status ?? session?.status ?? "queued";
  if (status === "completed") return "done";
  if (status === "failed") return "failed";
  if (status === "cancelled") return "cancelled";
  const message = (job?.message || session?.message || "").toLowerCase();
  if (status === "queued" || /waiting for an earlier scan/.test(message)) return "waiting";
  return "running";
}

/** The queue's own shape, drawn once so the empty page teaches the surface.
 *
 * Not the shared reaction-curve illustration: that previews scan-to-review,
 * which is what Sessions and Review are about. The queue is about order and
 * throughput — any number of sources in, exactly one scanned at a time, one
 * session out each — and a lane says that without a paragraph. */
function QueueLane() {
  return (
    <div className="queue-lane" aria-hidden="true">
      <span className="queue-lane-rail" />
      <div className="queue-lane-stations">
        <div className="queue-lane-station">
          <span className="queue-lane-node"><Film size={22} /></span>
          <strong>Recordings you add</strong>
          <small>Twitch links or local files, in the order you add them</small>
          <span className="queue-lane-cap">any number</span>
        </div>
        <div className="queue-lane-station is-active">
          <span className="queue-lane-node"><HardDrive size={22} /></span>
          <strong>One scan at a time</strong>
          <small>The rest wait their turn. Progress, coverage and time left show here</small>
          <span className="queue-lane-cap">1 at a time</span>
        </div>
        <div className="queue-lane-station">
          <span className="queue-lane-node"><Target size={22} /></span>
          <strong>Moments in Review</strong>
          <small>Each finished recording becomes a session you can keep or cut</small>
          <span className="queue-lane-cap">per recording</span>
        </div>
      </div>
    </div>
  );
}

export function StudioQueueView({
  jobIds, jobs, sessions, onOpenSession, onOpenProcessing, onCancelJob, onDone, onClearQueue, onNew,
}: {
  jobIds: string[];
  jobs: Job[];
  sessions: Session[];
  onOpenSession: (session: Session) => void;
  onOpenProcessing: (session: Session) => void;
  onCancelJob: (jobId: string) => Promise<boolean>;
  onDone: () => void;
  onClearQueue: () => void;
  onNew: (kind: "local" | "twitch" | "batch") => void;
}) {
  const [selectedId, setSelectedId] = useState<string>();
  const [cancellingId, setCancellingId] = useState<string>();
  const [clockNow, setClockNow] = useState(() => Date.now());
  const rows = useMemo(() => {
    const ids = jobIds.length
      ? jobIds
      : sessions
          .filter((session) => ["queued", "analyzing", "detecting", "assembling"].includes(session.status))
          .map((session) => session.id);
    return ids.map((id, index) => {
      const job = jobs.find((item) => item.id === id);
      const session = sessions.find((item) => item.id === id);
      return { id, index, job, session, lane: queueLane(job, session) };
    });
  }, [jobIds, jobs, sessions]);

  useEffect(() => {
    if (selectedId && rows.some((row) => row.id === selectedId)) return;
    setSelectedId(rows.find((row) => row.lane === "running")?.id ?? rows[0]?.id);
  }, [rows, selectedId]);

  // "Never queued anything" and "the batch I was watching just finished" are
  // different facts with different next steps, and the queue used to answer
  // both with the same sentence. Remember the last batch this view actually
  // rendered so the empty state can name what survived — both the explicit
  // Clear action and the restored-batch auto-retire land here. Deliberately a
  // ref, not persisted state: after a restart there is no batch the creator
  // was watching, and claiming one would be a guess.
  const lastBatchIds = useRef<string[]>([]);
  if (rows.length) lastBatchIds.current = rows.map((row) => row.id);
  const finishedBatch = rows.length
    ? []
    : lastBatchIds.current
        .map((id) => sessions.find((session) => session.id === id))
        .filter((session): session is Session => !!session && session.status === "completed");
  const finishedMoments = finishedBatch.reduce((total, session) => total + session.clips.length, 0);

  const hasRunningJob = rows.some((row) => row.lane === "running");
  useEffect(() => {
    if (!hasRunningJob) return;
    const timer = window.setInterval(() => setClockNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [hasRunningJob]);

  const running = rows.filter((row) => row.lane === "running").length;
  const waiting = rows.filter((row) => row.lane === "waiting").length;
  const done = rows.filter((row) => row.lane === "done").length;
  const failed = rows.filter((row) => row.lane === "failed").length;
  const selected = rows.find((row) => row.id === selectedId) ?? rows[0];
  const selectedJob = selected?.job;
  const selectedSession = selected?.session;
  const selectedLane = selected?.lane;
  const selectedProgress = selectedLane === "done" ? 100 : Math.max(0, Math.min(100, Math.round(selectedJob?.progress ?? 0)));
  const selectedStageProgress = selectedLane === "done" ? 100 : Math.max(0, Math.min(100, Math.round(selectedJob?.stage?.progress ?? 0)));
  const selectedName = selectedSession?.name || selectedJob?.url || selected?.id || "Queued scan";
  const selectedStage = selectedJob?.stage?.label || selectedJob?.currentPhase || (
    selectedLane === "waiting" ? "Waiting in queue"
      : selectedLane === "done" ? "Ready for review"
        : selectedLane === "failed" ? "Scan failed"
          : selectedLane === "cancelled" ? "Cancelled"
            : "Preparing scan"
  );
  const selectedFailure = describeScanFailure(selectedJob?.errorMessage || selectedJob?.message || selectedSession?.message);
  const selectedWaitingAhead = selected
    ? rows.slice(0, selected.index).filter((row) => row.lane === "running" || row.lane === "waiting").length
    : 0;
  const selectedMessage = selectedLane === "waiting"
    ? selectedWaitingAhead > 0
      ? `Starts after ${selectedWaitingAhead} earlier ${selectedWaitingAhead === 1 ? "scan" : "scans"} finish.`
      : "This scan is next and will start as soon as the worker is free."
    : selectedLane === "done"
      ? `${selectedSession?.clips.length ?? 0} moments are ready in Theater Review.`
      : selectedLane === "failed"
        ? selectedFailure.title
        : selectedLane === "cancelled"
          ? "This scan was removed from the active queue."
          : (selectedJob?.message || "Recall is preparing the next stage.");
  const selectedElapsed = (selectedJob?.elapsedSeconds ?? 0) + (
    selectedLane === "running" && selectedJob?.elapsedSyncedAt
      ? Math.max(0, (clockNow - selectedJob.elapsedSyncedAt) / 1000)
      : 0
  );
  const selectedEtaSeconds = selectedJob?.etaSeconds == null ? null : Math.max(
    0,
    selectedJob.etaSeconds - (
      selectedLane === "running" && selectedJob.etaSyncedAt
        ? Math.max(0, (clockNow - selectedJob.etaSyncedAt) / 1000)
        : 0
    ),
  );
  const selectedEtaReliable = selectedEtaSeconds != null && (selectedJob?.etaConfidence === "medium" || selectedJob?.etaConfidence === "high");
  const selectedEta = selectedLane === "waiting"
    ? selectedWaitingAhead > 0 ? `After ${selectedWaitingAhead} ${selectedWaitingAhead === 1 ? "scan" : "scans"}` : "Next up"
    : selectedLane === "done" ? "Complete"
      : selectedEtaReliable ? (selectedEtaSeconds < 60 ? "Under a minute" : `About ${fmtClock(selectedEtaSeconds)}`)
        : selectedLane === "running" ? "Calibrating..." : "--";
  const selectedCoverage = selectedJob?.vodDuration
    ? Math.max(0, Math.min(100, ((selectedJob.scannedSeconds ?? 0) / selectedJob.vodDuration) * 100))
    : selectedProgress;
  const selectedActivity = activityItems(selectedJob).slice(0, 6);
  const selectedHealth = describeScanHealth(selectedJob?.scanHealth);
  const selectedFound = selectedJob?.clipsFound ?? selectedJob?.foundClips?.length ?? selectedSession?.clips.length ?? 0;
  const selectedSourceType = /^https?:\/\//i.test(selectedJob?.url || selectedSession?.sourceUrl || "") ? "Twitch VOD" : "Local recording";
  const selectedLaneLabel = selectedLane === "done" ? "Finished"
    : selectedLane === "waiting" ? "Queued"
      : selectedLane === "running" ? "Running"
        : selectedLane === "failed" ? "Failed"
          : "Cancelled";

  const cancelSelected = async (id: string) => {
    if (cancellingId) return;
    setCancellingId(id);
    try {
      await onCancelJob(id);
    } finally {
      setCancellingId(undefined);
    }
  };

  return (
    <main className="queue-shell">
      <div className="queue-head">
        <div className="queue-head-copy">
          <h1>Scan queue</h1>
          <p>One recording at a time, in the order you add them. Leave the window — the queue keeps moving.</p>
        </div>
        {/* Three tiles reading 0 are not telemetry; they restate the empty
            state directly underneath them. They return with the first row. */}
        {!!rows.length && (
          <div className="queue-stats" aria-label="Queue summary">
            <span className={running ? "is-live" : ""}><i aria-hidden="true" /><b className="t-num">{running}</b><small>Running</small></span>
            <span><i aria-hidden="true" /><b className="t-num">{waiting}</b><small>Queued</small></span>
            <span><i aria-hidden="true" /><b className="t-num">{done}</b><small>Finished</small></span>
            {failed > 0 && <span className="is-failed"><i aria-hidden="true" /><b className="t-num">{failed}</b><small>Failed</small></span>}
          </div>
        )}
      </div>

      {!rows.length && (
        <div className="queue-empty">
          {finishedBatch.length ? (
            <section className="queue-empty-card" aria-label="Last batch finished">
              <h2>That batch is done</h2>
              <p className="queue-empty-lede">
                {plural(finishedBatch.length, "recording")} finished and moved to Review, with{" "}
                <b>{plural(finishedMoments, "moment")}</b> between them. Nothing is scanning now.
              </p>
              <div className="queue-done-row">
                {finishedBatch.map((session) => (
                  <button
                    type="button" key={session.id} className="queue-done-tile"
                    onClick={() => onOpenSession(session)}
                  >
                    <span className="queue-done-poster"><SessionPoster session={session} /></span>
                    <span className="queue-done-meta">
                      <strong>{session.name}</strong>
                      <small className="t-num">{plural(session.clips.length, "moment")}</small>
                    </span>
                  </button>
                ))}
              </div>
              <div className="queue-empty-actions">
                <button type="button" className="cta-accent" onClick={() => onOpenSession(finishedBatch[0])}>
                  <Target size={14} aria-hidden="true" /><span>Review {plural(finishedMoments, "moment")}</span>
                </button>
                <button type="button" className="btn-secondary" onClick={() => onNew("batch")}>
                  <Plus size={14} aria-hidden="true" /><span>Queue more recordings</span>
                </button>
                <p className="queue-empty-fine">
                  Clearing the queue only emptied this page. The sessions are in <b>Sessions</b> and
                  their kept clips are in <b>Clip Library</b>.
                </p>
              </div>
            </section>
          ) : (
            <section className="queue-empty-card" aria-label="Nothing queued">
              <h2>Nothing is queued</h2>
              <p className="queue-empty-lede">
                Add the recordings from a night of streaming and Recall works down the list on its own.{" "}
                <b>Every finished scan lands in Review</b> with its moments already found.
              </p>
              <QueueLane />
              <div className="queue-empty-actions">
                <button type="button" className="cta-accent" onClick={() => onNew("batch")}>
                  <Plus size={14} aria-hidden="true" /><span>Queue recordings</span>
                </button>
                <button type="button" className="btn-secondary" onClick={() => onNew("local")}>
                  <FolderOpen size={14} aria-hidden="true" /><span>Add a single file</span>
                </button>
                <p className="queue-empty-fine">
                  This page tracks one batch. Finished scans stay in <b>Sessions</b> after you clear it.
                </p>
              </div>
            </section>
          )}
        </div>
      )}

      {!!rows.length && selected && (
        <div className="queue-workspace">
          <aside className="queue-rail" aria-label="Batch queue">
            <div className="queue-rail-head"><span>Queue</span><b className="t-num">{rows.length}</b></div>
            <div className="queue-list">
              {rows.map((row) => {
                const name = row.session?.name || row.job?.url || row.id;
                const progress = row.lane === "done" ? 100 : Math.round(row.job?.progress ?? 0);
                const stage = row.job?.stage?.label || row.job?.currentPhase || (
                  row.lane === "waiting" ? "Waiting" : row.lane === "done" ? "Review ready" : row.lane
                );
                return (
                  <button
                    type="button"
                    key={row.id}
                    className={`queue-row is-${row.lane} ${selected.id === row.id ? "is-selected" : ""}`}
                    onClick={() => setSelectedId(row.id)}
                    aria-current={selected.id === row.id ? "true" : undefined}
                  >
                    <span className="queue-row-poster" aria-hidden="true">
                      {row.session ? <SessionPoster session={row.session} /> : <Film size={18} />}
                      <small className="t-num">{String(row.index + 1).padStart(2, "0")}</small>
                    </span>
                    <span className="queue-copy">
                      <strong>{name}</strong>
                      <small>
                        {row.lane === "waiting" && `Queued · item ${row.index + 1}`}
                        {row.lane === "running" && `${stage} · ${progress}%`}
                        {row.lane === "done" && `${row.session?.clips.length ?? 0} moments ready`}
                        {row.lane === "failed" && "Needs attention"}
                        {row.lane === "cancelled" && "Removed from queue"}
                      </small>
                      <span className="queue-row-track" aria-hidden="true"><i style={{ transform: `scaleX(${progress / 100})` }} /></span>
                    </span>
                    <span className="queue-row-state" aria-hidden="true"><i /></span>
                  </button>
                );
              })}
            </div>
            <div className="queue-rail-foot">
              <span><i className={running ? "is-live" : ""} aria-hidden="true" />{running ? "Queue is processing" : "Queue is idle"}</span>
              <button type="button" onClick={onDone}>View all sessions <ChevronRight size={13} aria-hidden="true" /></button>
              {!running && !waiting && (
                <button
                  type="button"
                  className="queue-clear"
                  onClick={onClearQueue}
                  title="Empty this queue view. Your scanned sessions are kept."
                >
                  Clear finished queue
                </button>
              )}
            </div>
          </aside>

          <section className={`queue-detail is-${selectedLane}`} aria-live="polite">
            <div className="queue-detail-top">
              <div className="queue-detail-poster">
                {selectedSession
                  ? <SessionPoster session={selectedSession} className="queue-detail-poster-image" />
                  : <Film size={32} aria-hidden="true" />}
                <span className={`queue-status-mark is-${selectedLane}`}><i aria-hidden="true" />{selectedLaneLabel}</span>
              </div>
              <div className="queue-detail-heading">
                <span className="queue-detail-position t-num">Batch item {String(selected.index + 1).padStart(2, "0")} / {String(rows.length).padStart(2, "0")}</span>
                <h2>{selectedName}</h2>
                <p>{selectedMessage}</p>
              </div>
              <div className="queue-detail-actions">
                {(selectedLane === "running" || selectedLane === "waiting") && (
                  <button type="button" className="btn-secondary queue-cancel" disabled={!!cancellingId} onClick={() => void cancelSelected(selected.id)}>
                    {cancellingId === selected.id ? "Stopping..." : selectedLane === "waiting" ? "Remove" : "Cancel scan"}
                  </button>
                )}
                {selectedLane === "running" && selectedSession && <button type="button" className="cta-accent" onClick={() => onOpenProcessing(selectedSession)}>Open live scan</button>}
                {selectedLane === "done" && selectedSession && <button type="button" className="cta-accent" onClick={() => onOpenSession(selectedSession)}>Review moments</button>}
                {selectedLane === "failed" && selectedSession && <button type="button" className="btn-secondary" onClick={() => onOpenProcessing(selectedSession)}>View details</button>}
              </div>
            </div>

            {selectedLane === "running" && (
              <>
                <div className="queue-progress-card">
                  <div className="queue-progress-copy"><span>{selectedStage}</span><b className="t-num">{selectedProgress}%</b></div>
                  <div className="queue-progress-track" aria-label={`${selectedProgress}% complete`}><i style={{ transform: `scaleX(${selectedProgress / 100})` }} /></div>
                  <div className="queue-progress-meta">
                    <span><small>Current stage</small><b>{selectedStage}</b></span>
                    <span><small>Stage progress</small><b className="t-num">{selectedStageProgress}%</b></span>
                    <span><small>Coverage</small><b className="t-num">{Math.round(selectedCoverage)}%</b></span>
                    <span><small>Time remaining</small><b className="t-num">{selectedEta}</b></span>
                  </div>
                </div>

                {selectedHealth && (
                  <div className={`queue-health is-${selectedHealth.tone}`}>
                    <span><i aria-hidden="true" /></span>
                    <div><strong>{selectedHealth.title}</strong><small>{selectedHealth.detail}</small></div>
                  </div>
                )}

                <div className="queue-detail-grid">
                  <section className="queue-activity">
                    <div className="queue-section-head"><span>Latest activity</span><small>Live telemetry</small></div>
                    <div className="queue-activity-list">
                      {selectedActivity.map((item, index) => (
                        <div key={item.key} className={index === 0 ? "is-current" : ""}>
                          <span className="queue-activity-node" aria-hidden="true"><i /></span>
                          <time className="t-num">{item.timestamp}</time>
                          <span><strong>{item.text}</strong>{item.detail && item.detail !== item.text ? <small>{item.detail}</small> : null}</span>
                        </div>
                      ))}
                      {!selectedActivity.length && (
                        <div className="is-empty">
                          <span className="queue-activity-node" aria-hidden="true"><i /></span>
                          <time className="t-num">--:--</time>
                          <span><strong>{selectedStage}</strong><small>{selectedMessage}</small></span>
                        </div>
                      )}
                    </div>
                  </section>

                  <aside className="queue-facts">
                    <div className="queue-section-head"><span>Scan details</span><small>{selectedJob?.scanProfile?.name || "Batch scan"}</small></div>
                    <dl>
                      <div><dt>Elapsed</dt><dd className="t-num">{fmtClock(selectedElapsed)}</dd></div>
                      <div><dt>VOD scanned</dt><dd className="t-num">{selectedJob?.vodDuration ? `${fmtClock(selectedJob.scannedSeconds ?? 0)} / ${fmtClock(selectedJob.vodDuration)}` : selectedSession?.vodDuration ? fmtClock(selectedSession.vodDuration) : "--"}</dd></div>
                      <div><dt>Moments found</dt><dd className="t-num">{selectedFound}</dd></div>
                      <div><dt>Workers</dt><dd className="t-num">{selectedJob?.totalWorkers ? `${selectedJob.activeWorkers ?? 0} / ${selectedJob.totalWorkers}` : "--"}</dd></div>
                      <div><dt>Device</dt><dd>{selectedJob?.scanHealth?.device || "Auto"}</dd></div>
                      <div><dt>Source</dt><dd>{selectedSourceType}</dd></div>
                    </dl>
                  </aside>
                </div>
              </>
            )}

            {selectedLane === "waiting" && (
              <div className="queue-state-panel is-waiting">
                <div className="queue-state-message"><span><Clock size={18} aria-hidden="true" /></span><div><h3>{selectedWaitingAhead ? `Queued behind ${selectedWaitingAhead} ${selectedWaitingAhead === 1 ? "scan" : "scans"}` : "Next scan in line"}</h3><p>{selectedMessage}</p></div></div>
                <dl className="queue-state-facts">
                  <div><dt>Queue position</dt><dd className="t-num">{selected.index + 1} of {rows.length}</dd></div>
                  <div><dt>Source</dt><dd>{selectedSourceType}</dd></div>
                  <div><dt>Expected start</dt><dd>{selectedEta}</dd></div>
                  <div><dt>Scan mode</dt><dd>{selectedJob?.scanProfile?.name || "Batch default"}</dd></div>
                </dl>
              </div>
            )}

            {selectedLane === "done" && (
              <div className="queue-state-panel is-done">
                <div className="queue-state-message"><span><Check size={19} aria-hidden="true" /></span><div><h3>Ready for Theater Review</h3><p>{selectedMessage}</p></div></div>
                <dl className="queue-state-facts">
                  <div><dt>Moments ready</dt><dd className="t-num">{selectedFound}</dd></div>
                  <div><dt>Source</dt><dd>{selectedSourceType}</dd></div>
                  <div><dt>Recording length</dt><dd className="t-num">{selectedSession?.vodDuration ? fmtClock(selectedSession.vodDuration) : "--"}</dd></div>
                  <div><dt>Finished in</dt><dd className="t-num">{selectedElapsed ? fmtClock(selectedElapsed) : "Complete"}</dd></div>
                </dl>
              </div>
            )}

            {selectedLane === "failed" && (
              <div className="queue-state-panel is-failed">
                <div className="queue-state-message"><span><X size={19} aria-hidden="true" /></span><div><h3>{selectedFailure.title}</h3><p>{selectedFailure.hint}</p></div></div>
                <dl className="queue-state-facts">
                  <div><dt>Stopped during</dt><dd>{selectedFailure.stage || selectedStage}</dd></div>
                  <div><dt>Source</dt><dd>{selectedSourceType}</dd></div>
                  <div><dt>Elapsed</dt><dd className="t-num">{selectedElapsed ? fmtClock(selectedElapsed) : "--"}</dd></div>
                </dl>
              </div>
            )}

            {selectedLane === "cancelled" && (
              <div className="queue-state-panel is-cancelled">
                <div className="queue-state-message"><span><X size={19} aria-hidden="true" /></span><div><h3>Removed from this run</h3><p>No more work will be done for this queue item. Other scans will continue normally.</p></div></div>
                <dl className="queue-state-facts"><div><dt>Source</dt><dd>{selectedSourceType}</dd></div></dl>
              </div>
            )}
          </section>
        </div>
      )}
    </main>
  );
}

function StudioProcessingView({ job, onCancel, onRetry, cancelRequested }: {
  job?: Job;
  onCancel: () => void;
  onRetry: () => void;
  cancelRequested: boolean;
}) {
  const [clockNow, setClockNow] = useState(() => Date.now());
  const apiEndpoint = useJobStore((s) => s.settings.apiEndpoint);
  const [diagState, setDiagState] = useState<"idle" | "copying" | "copied" | "copied-local" | "error">("idle");
  const shelfRef = useRef<HTMLDivElement>(null);
  const copyDiagnostics = useCallback(async () => {
    if (!job?.id || diagState === "copying") return;
    setDiagState("copying");
    let text = buildLocalDiagnostics(job);
    let usedLocalFallback = true;
    try {
      const res = await apiFetch(`/jobs/${job.id}/diagnostics`, undefined, apiEndpoint);
      if (res.ok) {
        const data = await res.json();
        if (typeof data.text === "string" && data.text.trim()) {
          text = data.text;
          usedLocalFallback = false;
        }
      }
    } catch {
      // A start-up/connection failure may have no backend job at all. The
      // desktop fallback above still preserves the source, stage, and reason.
    }
    try {
      await writeClipboardText(text);
      setDiagState(usedLocalFallback ? "copied-local" : "copied");
    } catch {
      setDiagState("error");
    }
    window.setTimeout(() => setDiagState("idle"), 2500);
  }, [job, apiEndpoint, diagState]);
  const clockRunning = Boolean(job && !["completed", "cancelled", "failed"].includes(job.status));
  useEffect(() => {
    setClockNow(Date.now());
    if (!clockRunning) return;
    const timer = window.setInterval(() => setClockNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [job?.id, clockRunning]);

  const index = stageIndex(job);
  const progress = Math.max(0, Math.min(100, job?.progress ?? 0));
  const found = job?.foundClips ?? [];
  const twitchDownload = /^https?:\/\/(?:www\.)?twitch\.tv\/videos\/\d+/i.test(job?.url ?? "") && job?.currentPhase === "Resolving Input";
  const downloadPercent = Math.round(Math.max(0, Math.min(1, job?.downloadProgress ?? 0)) * 100);
  const activity = activityItems(job);
  const chartHeight = 240;
  // Large processing chart: keep peaks but calm dense R(t) so the curve reads
  // as a reaction envelope rather than a seismograph once timeline_ready lands.
  const livePath = job?.liveTimeline
    ? timelinePath(job.liveTimeline, 1000, chartHeight, { maxPoints: 180, smoothPasses: 2 })
    : "";
  const timelineEnd = job?.liveTimeline?.t?.length
    ? job.liveTimeline.t[job.liveTimeline.t.length - 1]
    : 0;
  const vodDuration = Math.max(job?.vodDuration ?? 0, timelineEnd);
  const coveragePercent = vodDuration
    ? Math.max(0, Math.min(100, ((job?.scannedSeconds ?? 0) / vodDuration) * 100))
    : 0;
  // Space pins across the plot so clustered finds don't stack into one blob.
  const momentPins = vodDuration ? found.reduce<Array<{
    clip: (typeof found)[number];
    label: string;
    position: number;
  }>>((selected, clip, clipIndex) => {
    const seconds = timestampSeconds(clip.timestamp);
    if (seconds == null) return selected;
    const position = Math.max(2, Math.min(98, (seconds / vodDuration) * 100));
    if (selected.some((item) => Math.abs(item.position - position) < 5)) return selected;
    selected.push({
      clip,
      label: String(clipIndex + 1).padStart(2, "0"),
      position,
    });
    return selected;
  }, []).slice(0, 12) : [];
  const stageProgress = Math.max(0, Math.min(100, job?.stage?.progress ?? 0));
  const currentStagePercent = twitchDownload ? downloadPercent : Math.round(stageProgress);
  const elapsed = (job?.elapsedSeconds ?? 0) + (
    clockRunning && job?.elapsedSyncedAt
      ? Math.max(0, (clockNow - job.elapsedSyncedAt) / 1000)
      : 0
  );
  const transcribing = job?.currentPhase === "Reaction" && job?.transcribedSeconds != null;
  const liveEtaSeconds = job?.etaSeconds == null ? null : Math.max(
    0,
    job.etaSeconds - (
      clockRunning && job.etaSyncedAt ? Math.max(0, (clockNow - job.etaSyncedAt) / 1000) : 0
    ),
  );
  const stages = [
    [twitchDownload ? "Downloading VOD" : "Getting ready", twitchDownload ? "Saving the Twitch recording locally" : "Opening the file and mapping tracks"],
    ["Scanning signals", "Voice, facecam, gameplay, and on-screen events"],
    ["Finding moments", "Building the reaction curve and ranking clips"],
    ["Writing captions", "Turning detected moments into review-ready drafts"],
    ["Rendering previews", "Preparing fast Theater playback and thumbnails"],
  ];
  const status = cancelRequested ? "Stopping" : stages[index]?.[0] || "Queued";
  const etaReliable = liveEtaSeconds != null && (job?.etaConfidence === "medium" || job?.etaConfidence === "high");
  const eta = !etaReliable
    ? "Calibrating…"
    : liveEtaSeconds < 60 ? "Under a minute" : `About ${fmtClock(liveEtaSeconds)}`;

  const curveAwaitingScan = twitchDownload || index === 0;
  const formingTitle = curveAwaitingScan
    ? "Preparing the reaction curve"
    : index >= 3 || coveragePercent >= 100
      ? "Finalizing the reaction curve"
      : "Building the reaction curve";
  const formingDetail = curveAwaitingScan
    ? twitchDownload
      ? `Downloaded ${downloadPercent}% of the Twitch VOD`
      : "Mapping the recording before signal analysis begins."
    : job?.vodDuration
      ? `Scanned ${fmtClock(job.scannedSeconds ?? 0)} of ${fmtClock(job.vodDuration)}`
      : "Reading real reaction signals from the recording.";
  const formingNote = found.length
    ? `${found.length} moment${found.length === 1 ? "" : "s"} found and pinned to the recording.`
    : "Confirmed moments will pin themselves here as Recall finds them.";
  const curveState = livePath
    ? "Curve ready"
    : curveAwaitingScan
      ? "Waiting for scan"
      : coveragePercent >= 100
        ? "Finalizing"
        : "Forming";
  const scanheadPercent = vodDuration ? coveragePercent : progress;
  const readyCount = found.filter((clip) => !clip.pending).length;
  const scanHealthInfo = describeScanHealth(job?.scanHealth);

  useEffect(() => {
    const row = shelfRef.current;
    if (!row || !found.length) return;
    row.scrollTo({ left: row.scrollWidth, behavior: "smooth" });
  }, [found.length, job?.id]);

  if (job?.status === "failed") {
    const failure = describeScanFailure(job?.message || job?.errorMessage);
    return (
      <div className="prg-shell">
        <div className="prg-panel prg-failed" role="alert">
          <div className="prg-fail-head">
            <span className="prg-fail-badge" aria-hidden="true"><X size={20} /></span>
            <div className="prg-fail-copy">
              <h1>{failure.title}</h1>
              <p>{failure.hint}</p>
            </div>
          </div>
          {failure.stage && <div className="prg-fail-stage">Stopped during <b>{failure.stage}</b></div>}
          {failure.raw && (
            <details className="prg-fail-report">
              <summary>Technical details</summary>
              <pre>{failure.raw}</pre>
            </details>
          )}
          <div className="prg-fail-actions">
            <button type="button" className="cta-accent" onClick={onRetry}>Try this source again</button>
            <button type="button" className="btn-secondary" onClick={copyDiagnostics} disabled={diagState === "copying"} aria-live="polite">
              {diagState === "copying" ? "Building report…" : diagState === "copied" ? "Copied — send it to the developer" : diagState === "copied-local" ? "Copied available details" : diagState === "error" ? "Clipboard blocked — try again" : "Copy diagnostics report"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (job?.status === "cancelled") {
    return (
      <div className="prg-shell">
        <div className="prg-panel prg-failed" role="status">
          <div className="prg-fail-head">
            <span className="prg-fail-badge" aria-hidden="true"><X size={20} /></span>
            <div className="prg-fail-copy">
              <h1>Scan cancelled</h1>
              <p>Recall stopped this scan before it finished. Start a new project when you are ready to try again.</p>
            </div>
          </div>
          <div className="prg-fail-actions">
            <button type="button" className="cta-accent" onClick={onRetry}>Try this source again</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="prg-shell">
      <div className="prg-panel" aria-live="polite">
        <div className="prg-runbar">
          <span className="prg-runring" aria-hidden="true"><svg viewBox="0 0 40 40"><circle cx="20" cy="20" r="16" /><circle className="is-progress" cx="20" cy="20" r="16" pathLength="100" strokeDasharray="100" strokeDashoffset={100 - progress} /></svg></span>
          <strong>{cancelRequested ? "Stopping safely" : status}</strong>
          <span className="prg-run-count t-num">{found.length ? `${found.length} moment${found.length === 1 ? "" : "s"} found` : `${Math.round(progress)}% analyzed`}</span>
          <i aria-hidden="true" />
          <span className="prg-run-eta">{etaReliable ? `${eta} left` : eta}</span>
          <span className="prg-run-track" aria-hidden="true"><b style={{ transform: `scaleX(${progress / 100})` }} /></span>
          <em className="t-num">{Math.round(progress)}%</em>
        </div>
        {scanHealthInfo && (
          <div className={`prg-health is-${scanHealthInfo.tone}`} role="status">
            <strong>{scanHealthInfo.title}</strong>
            <span>{scanHealthInfo.detail}</span>
          </div>
        )}
        <div className="prg-stages" aria-label="Scan stages">
          {stages.map(([title, copy], stage) => {
            const state = stage < index ? "is-done" : stage === index ? "is-active" : "";
            return (
              <div key={title} className={`prg-stage ${state}`} aria-current={stage === index ? "step" : undefined}>
                <i className="prg-stage-fill" style={{ width: stage < index ? "100%" : stage === index ? `${twitchDownload && stage === 0 ? Math.max(4, downloadPercent) : Math.max(4, stageProgress)}%` : "0%" }} />
                <span className="step-dot" aria-hidden="true"><span className="step-dot-num t-num">{stage + 1}</span><svg className="step-dot-check" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7" /></svg></span>
                <span className="prg-stage-copy"><b>{title}</b><small>{copy}</small></span>
                <span className="prg-stage-pct t-num">{stage === index ? `${twitchDownload && stage === 0 ? downloadPercent : Math.round(stageProgress)}%` : ""}</span>
              </div>
            );
          })}
        </div>

        <div className="prg-headline">
          <h1>{cancelRequested ? "Stopping the scan safely…" : twitchDownload ? "Downloading Twitch VOD…" : (job?.stage?.label || "Opening your VOD.")}</h1>
          <p>{cancelRequested ? "Recall is finishing the current operation before it closes the job." : (job?.message || "Mapping video and audio tracks before the real work starts.")}</p>
        </div>

        <div className="prg-infochips">
          <Info label="Status" value={status} accent />
          <Info label="Elapsed" value={fmtClock(elapsed)} />
          <Info label="Time left" value={eta} />
          <Info
            label={twitchDownload ? "Downloaded" : transcribing ? "Transcribed" : "Scanned"}
            value={twitchDownload ? `${downloadPercent}%` : fmtClock(transcribing ? job?.transcribedSeconds ?? 0 : job?.scannedSeconds ?? 0)}
          />
          <Info
            label="Speed"
            value={transcribing && job?.transcriptionSpeed
              ? `${job.transcriptionSpeed.toFixed(1)}× ASR`
              : job?.scanSpeed ? `${job.scanSpeed.toFixed(1)}×` : "—"}
          />
          <Info label="Moments" value={found.length} />
        </div>

        <div className="prg-body">
          <div className={`prg-graph ${livePath ? "has-data" : "is-forming"}`}>
            <div className="prg-graph-head">
              <span className="prg-graph-title">Reaction curve</span>
              <span className={`prg-graph-state ${livePath ? "is-live" : ""}`}><i aria-hidden="true" />{curveState}</span>
            </div>
            <div className="prg-yaxis" aria-hidden="true"><span>High</span><span>Medium</span><span>Low</span></div>
            <div className="prg-plot">
              <div className="prg-graph-grid" aria-hidden="true" />
              {livePath ? (
                <svg viewBox={`0 0 1000 ${chartHeight}`} preserveAspectRatio="none" role="img" aria-label="Live reaction intensity across the scanned VOD">
                  <defs><linearGradient id="prgFillReact" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--accent)" stopOpacity=".32" /><stop offset="100%" stopColor="var(--accent)" stopOpacity="0" /></linearGradient></defs>
                  <path d={`${livePath} L1000 ${chartHeight} L0 ${chartHeight} Z`} fill="url(#prgFillReact)" pointerEvents="none" />
                  <path d={livePath} fill="none" stroke="var(--accent)" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" pointerEvents="none" />
                </svg>
              ) : (
                <div className="prg-curve-empty">
                  <span className="prg-curve-empty-icon"><BarChart3 size={20} aria-hidden="true" /></span>
                  <strong>{formingTitle}</strong>
                  <span className="prg-curve-progress-copy t-num">{formingDetail}</span>
                  <span className="prg-curve-track" aria-hidden="true"><i style={{ transform: `scaleX(${coveragePercent / 100})` }} /></span>
                  <small>{formingNote}</small>
                </div>
              )}
              {!livePath && <div className="prg-coverage" style={{ transform: `scaleX(${coveragePercent / 100})` }} aria-hidden="true" />}
              {momentPins.map((moment) => (
                <span
                  key={moment.clip.clipId}
                  className="prg-moment-pin"
                  style={{ left: `${moment.position}%` }}
                  title={`${moment.label} - ${moment.clip.title} at ${moment.clip.timestamp}`}
                  aria-hidden="true"
                ><i />{moment.label}</span>
              ))}
              {!livePath && (
                <div className="prg-scanhead" style={{ left: `${scanheadPercent}%` }} aria-hidden="true" />
              )}
              {!livePath && coveragePercent > 0 && <span className="prg-scanhead-label t-num" style={{
                left: `${coveragePercent}%`,
                transform: coveragePercent < 18 ? "translateX(7px)" : coveragePercent > 88 ? "translateX(calc(-100% - 7px))" : "translateX(-50%)",
              }}>{fmtClock(job?.scannedSeconds ?? 0)} scanned</span>}
            </div>
            <div className="prg-graph-axis" aria-hidden="true"><span>0:00</span><span>{fmtClock(vodDuration * .25)}</span><span>{fmtClock(vodDuration * .5)}</span><span>{fmtClock(vodDuration * .75)}</span><span>{fmtClock(vodDuration)}</span></div>
          </div>
          <div className="prg-feed">
            <span className="prg-feed-title">What Recall is doing</span>
            <div className="prg-feed-current">
              <strong>{status}</strong>
              <span className="prg-feed-current-copy">{job?.message || stages[index]?.[1] || "Preparing the next stage."}</span>
              <span className="prg-feed-progress"><i aria-hidden="true"><b style={{ transform: `scaleX(${currentStagePercent / 100})` }} /></i><em className="t-num">{currentStagePercent}%</em></span>
            </div>
            <div className="prg-ready-card"><span><Check aria-hidden="true" /></span><div>{found.length ? <><strong>{readyCount || found.length} {readyCount === 1 || (!readyCount && found.length === 1) ? "moment" : "moments"} {readyCount ? "ready" : "found"}</strong><small>{readyCount ? "Ready for review after the scan" : "Recall is still preparing previews"}</small></> : <><strong>Finding your best moments</strong><small>Clips appear here the instant Recall spots one</small></>}</div></div>
            <span className="prg-feed-recent">Recent activity</span>
            <div className="prg-feed-log">{activity.map((item) => <div key={item.key} className={`prg-feed-line ${item.ok ? "is-ok" : "is-live"}`}><span className="prg-activity-icon" aria-hidden="true">{item.ok ? <Check /> : null}</span><span className="prg-activity-copy"><span><time>{item.timestamp}</time><strong>{item.text}</strong></span>{item.detail && item.detail !== item.text ? <small>{item.detail}</small> : null}</span></div>)}{!activity.length && <span className="prg-feed-empty">Activity appears here as each stage begins.</span>}</div>
            <details key={job?.id} className="prg-technical">
              <summary>Technical activity <ChevronDown aria-hidden="true" /></summary>
              <dl>
                <div><dt>Phase</dt><dd>{job?.currentPhase || "Preparing"}</dd></div>
                <div><dt>Elapsed</dt><dd className="t-num">{fmtClock(elapsed)}</dd></div>
                <div><dt>{transcribing ? "Transcribed" : "Scanned"}</dt><dd className="t-num">{fmtClock(transcribing ? job?.transcribedSeconds ?? 0 : job?.scannedSeconds ?? 0)}</dd></div>
                <div><dt>Speed</dt><dd className="t-num">{transcribing && job?.transcriptionSpeed ? `${job.transcriptionSpeed.toFixed(1)}x ASR` : job?.scanSpeed ? `${job.scanSpeed.toFixed(1)}x` : "-"}</dd></div>
                <div><dt>Workers</dt><dd className="t-num">{job?.totalWorkers ? `${job.activeWorkers ?? 0}/${job.totalWorkers}` : "-"}</dd></div>
                <div><dt>Device</dt><dd>{job?.scanHealth?.device || "-"}</dd></div>
                <div><dt>Vision</dt><dd>{String(job?.scanHealth?.visual?.status || "-")}</dd></div>
                <div><dt>AI review</dt><dd>{String(job?.scanHealth?.semantic?.status || "-")}</dd></div>
              </dl>
            </details>
          </div>
        </div>

        <div className="prg-shelf">
          <span className="prg-shelf-title">Latest moments · <b>{found.length}</b></span>
          <div className="prg-shelf-row" ref={shelfRef}>{found.map((clip, clipIndex) => {
            const marker = String(clipIndex + 1).padStart(2, "0");
            return <div key={clip.clipId} className="prg-mini" data-clip-id={clip.clipId}><div className="prg-mini-thumb"><Film aria-hidden="true" /><span className="prg-mini-marker t-num">{marker}</span></div><span className="prg-mini-time">{clip.timestamp}</span><span className="prg-mini-label">{clip.title}</span><span className={`prg-mini-state ${clip.pending ? "" : "is-ready"}`}>{clip.pending ? "Moment found" : "Preview ready"}</span></div>;
          })}{!found.length && <span className="prg-shelf-note">Clips appear here as Recall confirms them.</span>}</div>
        </div>

        <div className="prg-foot">
          <button className="prg-tool is-danger" onClick={onCancel} disabled={cancelRequested || terminal.has(job?.status ?? "")}><X size={14} aria-hidden="true" /><span>{cancelRequested ? "Stopping…" : "Cancel scan"}</span></button>
          <span className="prg-foot-spacer" />
          <span className="prg-foot-note">You can leave this screen; Recall keeps the scan running.</span>
        </div>
      </div>
    </div>
  );
}


function Info({ label, value, accent = false }: { label: string; value: string | number; accent?: boolean }) { return <div className="prg-infochip"><small>{label}</small><b className={accent ? "is-accent" : "t-num"}>{value}</b></div>; }

function openingBand(score: number) {
  if (score >= 65) return { label: "Strong opening", shortLabel: "Strong", className: "is-high" };
  if (score >= 45) return { label: "Good opening", shortLabel: "Good", className: "is-mid" };
  return { label: "Slow build", shortLabel: "Slow build", className: "is-low" };
}

function creatorClues(clip: Clip, limit = 3) {
  const clues: string[] = [];
  const add = (value?: string) => {
    if (value && !clues.includes(value)) clues.push(value);
  };
  const payoff = clip.features?.payoff ?? 0;
  const selfContained = clip.features?.self_contained ?? 0;
  if (payoff >= 0.7) add("Clear payoff");
  else if (payoff >= 0.45) add("Payoff lands");
  if (selfContained >= 0.75) add("Works without context");
  else if (selfContained >= 0.5) add("Mostly self-contained");

  const signalPriority = [
    "match_win", "elimination", "knock", "rank_progress", "laughter_burst",
    "speech_hype", "voice_reaction", "facecam_reaction", "chat_spike",
    "low_dead_air", "multi_signal", "generic_highlight",
  ];
  const signals = [...(clip.signals ?? [])].sort((a, b) => {
    const ai = signalPriority.indexOf(a);
    const bi = signalPriority.indexOf(b);
    return (ai < 0 ? signalPriority.length : ai) - (bi < 0 ? signalPriority.length : bi);
  });
  signals.forEach((signal) => add(clueLabel(signal)));
  if (!clues.length) add(momentLabel(clip.momentType) || "Reaction spike");
  return clues.slice(0, limit);
}

function editorialRecommendation(kept: boolean, cut: boolean, maybe: boolean, priority: boolean) {
  if (kept) return { label: "Creator pick", className: "is-kept" };
  if (cut) return { label: "Passed", className: "is-passed" };
  if (maybe) return { label: "Maybe", className: "is-maybe" };
  if (priority) return { label: "Start here", className: "is-top" };
  return { label: "Worth reviewing", className: "is-review" };
}

function StudioReviewCard({ clip, kept, cut, maybe, priority, timeline, index = 0, recommendedRank, onOpen, onKeep, onCut, onMaybe }: {
  clip: Clip;
  kept: boolean;
  cut: boolean;
  maybe: boolean;
  priority: boolean;
  timeline?: ReactionTimeline | null;
  index?: number;
  recommendedRank?: number;
  onOpen: () => void;
  onKeep: () => void;
  onCut: () => void;
  onMaybe: () => void;
}) {
  const score = hookScoreOf(clip);
  const opening = openingBand(score);
  const recommendation = editorialRecommendation(kept, cut, maybe, priority);
  const clues = creatorClues(clip, 2);
  const recallOrigin = clip.recallProvenance ? recallOriginCopy(clip.recallProvenance) : null;
  return (
    <article className={`rev-card ${clip.isOverflowCandidate ? "is-second-look" : ""} ${kept ? "is-kept" : ""} ${cut ? "is-cut" : ""} ${maybe ? "is-maybe" : ""}`} data-clip-id={clip.id} style={{ "--rev-i": index } as CSSProperties}>
      <button type="button" className="rev-card-open" onClick={onOpen} aria-label={`Open ${clip.title}, ${recommendation.label}${recommendedRank ? `, recommended pick ${recommendedRank}` : ", Second look"}, ${opening.label}${recallOrigin ? `, ${recallOrigin.label}` : ""}`}>
        <span className="rev-thumb">
          <Poster clip={clip} className="mock-v1-real-thumb" />
          <HeatSparkline timeline={timeline} start={clip.start_time} end={clip.end_time} />
          <span className={`rev-recommendation ${recommendation.className}`}>{recommendation.label}</span>
          <span className={`rev-band ${opening.className}`} title="Opening strength, not overall clip quality">{opening.label}</span>
          <span className="rev-dur">{fmtClock(durationOf(clip))}</span>
          <span className="rev-kept-flag" aria-label="Kept"><Check aria-hidden="true" /></span>
        </span>
        <span className="rev-card-info">
          {recommendedRank ? (
            <span className={`rev-rank-mark ${kept ? "is-kept" : ""}`} aria-label={`Recommended pick ${recommendedRank}`}>
              <small>Pick</small><b>#{recommendedRank}</b>
            </span>
          ) : (
            <span className="rev-rank-mark is-second-look" aria-label="Second look tier">
              <small>Tier</small><b>2nd</b>
            </span>
          )}
          <span className="rev-card-info-col">
            <span className="rev-card-title">{clip.title}</span>
            <span className="rev-card-time">{clip.timestamp} · {clip.isOverflowCandidate ? "Second look" : momentLabel(clip.momentType) || "Highlight"}</span>
            <RecallOriginBadge provenance={clip.recallProvenance} compact />
            {evidenceChipsFor(clip).length
              ? <EvidenceChips clip={clip} />
              : <span className="rev-card-clues">{clues.join(" · ")}</span>}
          </span>
        </span>
      </button>
      <div className="rev-actions rev-actions-static" aria-label={`Review ${clip.title}`}>
        <button type="button" className={`rev-act is-keep ${kept ? "is-active" : ""}`} aria-pressed={kept} onClick={onKeep}>{kept ? "Kept" : "Keep"}</button>
        <button type="button" className={`rev-act is-maybe ${maybe ? "is-active" : ""}`} aria-pressed={maybe} onClick={onMaybe}>Maybe</button>
        <button type="button" className={`rev-act is-cut ${cut ? "is-active" : ""}`} aria-pressed={cut} onClick={onCut}>{cut ? "Passed" : "Pass"}</button>
      </div>
    </article>
  );
}

/** Tells the creator when a session's clip previews were reclaimed, and offers
 * the one action that brings them back.
 *
 * Never rebuilds on its own. A rebuild is minutes of FFmpeg, and on a Twitch
 * session the source has to be re-downloaded first — so this mirrors the
 * Cutting Room's existing restore banner rather than inventing a second,
 * quieter policy for the same class of problem. */
function SessionMediaBanner({ session }: { session: Session }) {
  const settings = useJobStore((state) => state.settings);
  const addLog = useJobStore((state) => state.addLog);
  const hydrateSessionClips = useJobStore((state) => state.hydrateSessionClips);
  const refreshJobs = useJobStore((state) => state.refreshJobs);
  const [busy, setBusy] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const media = session.media;

  // Poll only while a rebuild is actually in flight. The banner is otherwise
  // driven by the media summary that already rides the sessions response.
  useEffect(() => {
    if (!rebuilding) return;
    let alive = true;
    const timer = window.setInterval(async () => {
      try {
        const response = await apiFetch(
          `/jobs/${session.id}/media-status`, undefined, settings.apiEndpoint,
        );
        if (!alive || !response.ok) return;
        const status = await response.json() as { rebuild?: { status?: string } };
        if (!alive) return;
        if (status?.rebuild?.status !== "rebuilding") {
          setRebuilding(false);
          await refreshJobs();
          await hydrateSessionClips(session.id);
          addLog(`Clip previews rebuilt for ${session.name}.`, "success");
        }
      } catch {
        // Transient API hiccup: keep polling rather than declaring failure.
      }
    }, 3000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [rebuilding, session.id, session.name, settings.apiEndpoint, addLog, hydrateSessionClips, refreshJobs]);

  if (!media || (media.rebuildable === 0 && media.unavailable === 0)) return null;

  const missing = media.rebuildable + media.unavailable;
  const startRebuild = async () => {
    setBusy(true);
    try {
      const response = await apiFetch(
        `/jobs/${session.id}/media/rebuild`, { method: "POST" }, settings.apiEndpoint,
      );
      const result = await response.json().catch(() => ({})) as {
        status?: string; total?: number; restorable?: boolean; detail?: unknown;
      };
      if (!response.ok) {
        const detail = result?.detail;
        throw new Error(typeof detail === "string" ? detail : "Recall could not start the rebuild.");
      }
      if (result?.status === "rebuilding" || result?.status === "already_running") {
        setRebuilding(true);
        addLog(`Rebuilding ${result.total ?? missing} clip previews. Progress is in session activity.`, "info");
      } else if (result?.status === "source_required") {
        addLog(
          result.restorable
            ? "Restore this session's Twitch source first — the VOD Editor has the restore button."
            : "The source recording is gone, so these previews cannot be rebuilt.",
          "warn",
        );
      } else {
        addLog("Nothing left to rebuild for this session.", "info");
      }
    } catch (cause) {
      addLog(cause instanceof Error ? cause.message : "Recall could not start the rebuild.", "warn");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="session-media-banner" role="status" data-state={media.state}>
      <Film size={15} aria-hidden="true" />
      <span className="session-media-banner-copy">
        <b>{missing} of {media.total} clip previews were cleared from disk.</b>{" "}
        {media.source_state === "local"
          ? "Your decisions, scores, and timestamps are all kept. Rebuild the previews from this session's source whenever you want them back."
          : media.source_state === "restorable"
            ? "Your decisions, scores, and timestamps are all kept. Restore the Twitch source in the VOD Editor first, then rebuild."
            : "Your decisions, scores, and timestamps are all kept, but the source recording is gone too, so the video cannot be rebuilt."}
      </span>
      {media.source_state === "local" && (
        <button type="button" className="btn-secondary" onClick={() => void startRebuild()} disabled={busy || rebuilding}>
          <RefreshCw size={13} aria-hidden="true" />
          {rebuilding ? "Rebuilding…" : busy ? "Starting…" : `Rebuild ${media.rebuildable} previews`}
        </button>
      )}
    </div>
  );
}

function StudioReviewView({
  session, timeline, filter, setFilter, sort, setSort, onOpen, onKeep, onCut, onMaybe, onClips, onCuttingRoom, bestOpeningScore,
  moreCandidateCount, moreCandidatesEnabled, moreCandidatesLoading, onToggleMoreCandidates,
}: {
  session?: Session;
  timeline: ReactionTimeline | null;
  filter: ReviewFilter;
  setFilter: (filter: ReviewFilter) => void;
  sort: ReviewSort;
  setSort: (sort: ReviewSort) => void;
  onOpen: (clip: Clip) => void;
  onKeep: (clip: Clip) => void;
  onCut: (clip: Clip) => void;
  onMaybe: (clip: Clip) => void;
  onClips: () => void;
  onCuttingRoom: () => void;
  bestOpeningScore: number;
  moreCandidateCount: number;
  moreCandidatesEnabled: boolean;
  moreCandidatesLoading: boolean;
  onToggleMoreCandidates: () => void;
}) {
  if (!session) return <div className="clips-empty"><StudioEmptyState variant="review" icon={<Target />} title="Review begins when the scan finishes" body="Recall will place the reaction curve and ranked clip deck here as soon as the session is ready." /></div>;
  const kept = new Set(session.savedClipIds);
  const primaryDeck = primaryReviewDeckClips(session.clips);
  const secondLook = secondLookClips(session.clips);
  const recommendedIds = new Set(primaryDeck.slice(0, 3).map((clip) => clip.id));
  const recommendedRanks = new Map(primaryDeck.map((clip, index) => [clip.id, index + 1]));
  const matchesFilter = (clip: Clip) => filter === "all"
    ? true
    : filter === "kept"
      ? kept.has(clip.id)
      : filter === "maybe"
        ? !!clip.maybe
        : !kept.has(clip.id) && !clip.passed && !clip.maybe;
  const visible = [
    ...sortReviewClips(primaryDeck.filter(matchesFilter), sort),
    ...sortReviewClips(secondLook.filter(matchesFilter), sort),
  ];
  const timelineEnd = timeline?.t?.length ? timeline.t[timeline.t.length - 1] : undefined;
  const vodDuration = Math.max(1, session.vodDuration ?? timelineEnd ?? 1);
  const curveHeight = 68;
  const curvePath = timelinePath(timeline, 1000, curveHeight);
  const rulerTickCount = vodDuration >= 7_200 ? 7 : 6;
  const rulerTicks = Array.from({ length: rulerTickCount }, (_, index) => {
    const seconds = (index / Math.max(1, rulerTickCount - 1)) * vodDuration;
    return { seconds, position: (seconds / vodDuration) * 100 };
  });
  const annotations = primaryReviewDeckClips(session.clips)
    .filter((clip) => Number.isFinite(clip.peakTimestamp ?? clip.start_time))
    .reduce<Array<{ clip: Clip; position: number; label: string }>>((selected, clip) => {
      if (selected.length >= 3) return selected;
      const seconds = clip.peakTimestamp ?? (clip.start_time ?? 0) + durationOf(clip) / 2;
      const position = Math.max(0, Math.min(100, (seconds / vodDuration) * 100));
      if (selected.some((item) => Math.abs(item.position - position) < 14)) return selected;
      selected.push({ clip, position, label: momentLabel(clip.momentType) || clip.title });
      return selected;
    }, []);

  return (
    <div className="rev-shell">
      <SessionMediaBanner session={session} />
      <div className="rev-strip">
        <div className="rev-strip-head">
          <span className="rev-strip-title">What Recall saw · <b>{fmtClock(session.vodDuration ?? 0)}</b></span>
          <div className="rev-filter-group" role="group" aria-label="Filter review clips">
            {(["all", "unreviewed", "maybe", "kept"] as ReviewFilter[]).map((value) => (
              <button key={value} type="button" className={`rev-filter ${filter === value ? "is-active" : ""}`} aria-pressed={filter === value} onClick={() => setFilter(value)}>{value === "all" ? "All" : value === "unreviewed" ? "Unreviewed" : value === "maybe" ? "Maybe" : "Kept"}</button>
            ))}
          </div>
          <label className="rev-sort">
            <ArrowUpDown size={13} aria-hidden="true" />
            <span className="sr-only">Sort clips</span>
            <select value={sort} onChange={(event) => setSort(event.currentTarget.value as ReviewSort)} aria-label="Sort review clips">
              <option value="recommended">Recommended</option>
              <option value="hook">Strongest opening</option>
              <option value="timeline">VOD order</option>
              <option value="longest">Longest first</option>
            </select>
          </label>
          {moreCandidateCount > 0 && (
            <button
              type="button"
              className={`rev-more-moments ${moreCandidatesEnabled ? "is-active" : ""}`}
              onClick={onToggleMoreCandidates}
              disabled={moreCandidatesLoading}
              aria-pressed={moreCandidatesEnabled}
              title="Show the strongest moments saved just outside the review deck"
            >
              {moreCandidatesLoading ? <RefreshCw size={13} className="is-spinning" aria-hidden="true" /> : <Sparkles size={13} aria-hidden="true" />}
              <span>More moments</span>
              <small>{moreCandidatesEnabled ? `${secondLook.length || moreCandidateCount} shown` : `+${moreCandidateCount}`}</small>
            </button>
          )}
          <div className="rev-legend" aria-label="Timeline legend">
            <span><i className="is-reaction" />Reaction</span>
            <span><i className="is-kept" />Kept</span>
            <span><i className="is-unreviewed" />Unreviewed</span>
            {moreCandidatesEnabled && secondLook.length > 0 && (
              <span><i className="is-second-look" />Second look</span>
            )}
          </div>
        </div>
        <div className={`rev-curve ${curvePath ? "has-data" : "is-empty"}`}>
          {curvePath ? (
            <>
              <div className="rev-plot">
                <svg viewBox={`0 0 1000 ${curveHeight}`} preserveAspectRatio="none" role="img" aria-label="Reaction intensity and clip windows across this VOD">
                  <defs>
                    <linearGradient id="reviewCurveFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--accent)" stopOpacity=".075" />
                      <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path d={`${curvePath} L1000 ${curveHeight - 4} L0 ${curveHeight - 4} Z`} fill="url(#reviewCurveFill)" pointerEvents="none" />
                  {session.clips.map((clip) => {
                    const x = ((clip.start_time ?? 0) / vodDuration) * 1000;
                    const w = Math.max(5, durationOf(clip) / vodDuration * 1000);
                    const activate = () => onOpen(clip);
                    return <rect key={clip.id} className={`rev-span ${kept.has(clip.id) ? "is-kept" : clip.passed ? "is-cut" : clip.maybe ? "is-maybe" : ""}`} x={x} y="7" width={w} height={curveHeight - 14} rx="2" role="button" tabIndex={0} aria-label={`Open ${clip.title}`} onClick={activate} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); } }} />;
                  })}
                  <path
                    d={curvePath}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="1.1"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    vectorEffect="non-scaling-stroke"
                    opacity=".9"
                    pointerEvents="none"
                  />
                </svg>
                {annotations.map(({ clip, position, label }) => (
                  <button
                    key={clip.id}
                    type="button"
                    className={`rev-annotation ${position < 8 ? "is-start" : position > 92 ? "is-end" : ""}`}
                    style={{ left: `${position}%` }}
                    title={`${label} · ${clip.timestamp}`}
                    aria-label={`Open ${label} at ${clip.timestamp}`}
                    onClick={() => onOpen(clip)}
                  >
                    <span>{label}</span><i aria-hidden="true" />
                  </button>
                ))}
              </div>
              <div className="rev-ruler" aria-label={`VOD runs from 0:00 to ${fmtClock(vodDuration)}`}>
                {rulerTicks.map((tick) => <span key={tick.seconds} style={{ left: `${tick.position}%` }}>{fmtClock(tick.seconds)}</span>)}
              </div>
            </>
          ) : <span className="rev-curve-empty">Signal timeline unavailable for this session.</span>}
        </div>
      </div>

      {visible.length ? (
        <div className="rev-grid">{visible.map((clip, index) => <StudioReviewCard key={clip.id} clip={clip} kept={kept.has(clip.id)} cut={!!clip.passed} maybe={!!clip.maybe} priority={recommendedIds.has(clip.id)} timeline={timeline} index={index} recommendedRank={recommendedRanks.get(clip.id)} onOpen={() => onOpen(clip)} onKeep={() => onKeep(clip)} onMaybe={() => onMaybe(clip)} onCut={() => onCut(clip)} />)}</div>
      ) : (
        <div className="clips-empty rev-filter-empty"><StudioEmptyState compact variant="review" icon={<Target />} title="Nothing matches this filter" body="The full review deck is still intact." action={<button type="button" className="btn-secondary" onClick={() => setFilter("all")}>Show all clips</button>} /></div>
      )}

      <div className="rev-stats">
        <div className="rev-stat"><small>Scanned</small><b>{fmtClock(session.vodDuration ?? 0)}</b></div>
        <div className="rev-stat"><small>Moments</small><b>{session.clips.length}</b></div>
        <div className="rev-stat"><small>Kept</small><b className="is-keep">{kept.size}</b></div>
        <div className="rev-stat"><small>Maybe</small><b className="is-maybe">{session.clips.filter((clip) => clip.maybe).length}</b></div>
        <div className="rev-stat"><small>Best opening</small><b>{openingBand(bestOpeningScore).label}</b></div>
        <div className="rev-stat"><small>Est. export</small><b>{fmtClock(session.clips.filter((clip) => kept.has(clip.id)).reduce((sum, clip) => sum + durationOf(clip), 0))}</b></div>
        <button className="btn-secondary rev-cutting-btn" onClick={onCuttingRoom} title="Scrub the full recording and cut a moment Recall missed"><Scissors size={14} aria-hidden="true" /><span>Open VOD Editor</span></button>
        <button className="cta-accent" disabled={!kept.size} onClick={onClips}><Film size={14} aria-hidden="true" /><span>Open kept clips</span></button>
      </div>

      <ScanLog session={session} />
    </div>
  );
}

// Collapsible retrospective of the scan's activity — the same running feed the
// creator watched live, now persisted with the session. Events are fetched
// lazily the first time the log is opened so the review view stays cheap.
function ScanLog({ session }: { session: Session }) {
  const [items, setItems] = useState<ActivityItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  const onToggle = (event: { currentTarget: HTMLDetailsElement }) => {
    if (!event.currentTarget.open || items !== null || loading) return;
    setLoading(true);
    setFailed(false);
    apiFetch(`/jobs/${session.id}`, undefined, useJobStore.getState().settings.apiEndpoint)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error("scan log unavailable"))))
      .then((data) => {
        const raw = Array.isArray(data.recent_events) ? data.recent_events : [];
        const events = raw.map((ev: any) => mapEvent(session.id, ev)).sort((a: JobEvent, b: JobEvent) => b.at - a.at);
        setItems(activityFromEvents(events, session.id, 60));
      })
      .catch(() => setFailed(true))
      .finally(() => setLoading(false));
  };

  return (
    <details className="rev-scanlog" onToggle={onToggle}>
      <summary>
        <Clock size={13} aria-hidden="true" />
        <span>Scan log</span>
        <small>What Recall did on this scan</small>
        <ChevronDown size={14} aria-hidden="true" />
      </summary>
      <div className="rev-scanlog-body">
        {loading && <span className="rev-scanlog-note">Loading scan history…</span>}
        {failed && <span className="rev-scanlog-note">Scan history isn’t available for this session.</span>}
        {items && items.length > 0 && (
          <ol className="rev-scanlog-list">
            {items.map((item) => (
              <li key={item.key} className={`rev-scanlog-line ${item.ok ? "is-ok" : ""}`}>
                <span className="rev-scanlog-icon" aria-hidden="true">{item.ok ? <Check size={11} /> : null}</span>
                <time>{item.timestamp}</time>
                <strong>{item.text}</strong>
              </li>
            ))}
          </ol>
        )}
        {items && items.length === 0 && !loading && <span className="rev-scanlog-note">No scan activity was recorded for this session.</span>}
      </div>
    </details>
  );
}





const signalLabel = (signal: string) =>
  signal
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\w/, (char) => char.toUpperCase());

const momentLabel = (momentType?: string) => momentType ? signalLabel(momentType) : undefined;

/** Stand-in art for a clip whose rendered media was reclaimed from disk.
 *
 * The row still knows exactly what the moment was — timestamp, strength, the
 * signals that fired — so the card shows that evidence rather than an empty
 * well. "Preview cleared" is a recoverable state and reads like one; only a
 * clip whose source is also unrecoverable says the media is gone for good. */
function ClipMediaPlaceholder({ clip, className }: { clip: Clip; className?: string }) {
  const rebuildable = clip.mediaState === "rebuildable";
  return (
    <span
      className={`clip-media-gone ${className ?? ""}`}
      data-state={clip.mediaState}
      title={rebuildable
        ? "Preview cleared to save space. Rebuild it from this session's source."
        : "The source recording and this preview are both gone. The moment's metadata and your decision are kept."}
    >
      <Film size={16} aria-hidden="true" />
      <span className="clip-media-gone-time t-num">{clip.timestamp}</span>
      <span className="clip-media-gone-label">{rebuildable ? "Preview cleared" : "Media unavailable"}</span>
    </span>
  );
}

function Poster({ clip, className, eager = false }: { clip: Clip; className?: string; eager?: boolean }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [clip.id, clip.thumbUrl]);
  if (clip.thumbUrl && !failed) {
    return <img className={className} src={clip.thumbUrl} alt="" loading={eager ? "eager" : "lazy"} onError={() => setFailed(true)} />;
  }
  if (clip.videoUrl && !clip.sourceWindow) {
    return <video className={className} src={clip.videoUrl} muted preload="metadata" />;
  }
  if (clip.mediaState === "rebuildable" || clip.mediaState === "unavailable") {
    return <ClipMediaPlaceholder clip={clip} className={className} />;
  }
  return <div className={`draft-card-art ${className ?? ""}`} />;
}

function pickSessionPosterClip(session: Session): Clip | undefined {
  const deck = originalDeckClips(session.clips);
  const saved = new Set(session.savedClipIds);
  // Prefer real rendered posters. Fall back to exported clip video (not
  // source-window seeks — those metadata frames are usually blank).
  const withThumb = sortReviewClips(
    deck.filter((clip) => !!clip.thumbUrl),
    "recommended",
  );
  const keptThumb = withThumb.find((clip) => saved.has(clip.id) || clip.kept);
  if (keptThumb) return keptThumb;
  if (withThumb[0]) return withThumb[0];
  const withVideo = sortReviewClips(
    deck.filter((clip) => !!clip.videoUrl && !clip.sourceWindow),
    "recommended",
  );
  return withVideo.find((clip) => saved.has(clip.id) || clip.kept) ?? withVideo[0];
}

function SessionPoster({
  session,
  className,
  variant = "compact",
}: {
  session: Session;
  className?: string;
  /** card = fill the session-card media well; compact = sidebar/list cover */
  variant?: "card" | "compact";
}) {
  const fillClass = variant === "card" ? "session-card-poster" : className;
  const posterClip = pickSessionPosterClip(session);
  // Fetch the VOD still via fetch() (same auth as apiFetch). Only swap it in
  // after a real image blob arrives — never mount a bare <img src=posterUrl>
  // that can sit as a broken-image icon when the source file is gone.
  const [vodSrc, setVodSrc] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    let objectUrl: string | null = null;
    setVodSrc(null);
    if (!session.posterUrl) return () => { alive = false; };
    (async () => {
      try {
        const response = await fetch(session.posterUrl!);
        if (!alive || !response.ok) return;
        const blob = await response.blob();
        if (!alive || !blob.type.startsWith("image/")) return;
        const next = URL.createObjectURL(blob);
        if (!alive) {
          URL.revokeObjectURL(next);
          return;
        }
        objectUrl = next;
        setVodSrc(next);
      } catch {
        // Keep the clip-poster / empty fallback below.
      }
    })();
    return () => {
      alive = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [session.id, session.posterUrl]);

  if (vodSrc) {
    return <img className={fillClass} src={vodSrc} alt="" />;
  }
  if (posterClip) {
    return <Poster clip={posterClip} className={fillClass} />;
  }
  return (
    <span className={`session-poster-fallback ${variant === "card" ? "is-card" : ""} ${className ?? ""}`}>
      <Film aria-hidden="true" />
    </span>
  );
}

function TheaterBackdrop({ clip }: { clip: Clip }) {
  return (
    <div className="theater-backdrop">
      <Poster clip={clip} eager />
    </div>
  );
}

function ProgressDots({
  clips,
  selectedClipId,
  savedSet,
  passedIds,
  maybeIds,
}: {
  clips: Clip[];
  selectedClipId: string;
  savedSet: Set<string>;
  passedIds: Set<string>;
  maybeIds: Set<string>;
}) {
  const MAX_DOTS = 20;
  let visible = clips;
  let hidden = 0;
  if (clips.length > MAX_DOTS) {
    const idx = Math.max(0, clips.findIndex((c) => c.id === selectedClipId));
    const start = Math.min(Math.max(0, idx - Math.floor(MAX_DOTS / 2)), clips.length - MAX_DOTS);
    visible = clips.slice(start, start + MAX_DOTS);
    hidden = clips.length - MAX_DOTS;
  }
  const position = Math.max(0, clips.findIndex((c) => c.id === selectedClipId)) + 1;
  return (
    <div
      className="review-dots"
      role="img"
      aria-label={`Clip ${position} of ${clips.length} — ${savedSet.size} kept, ${maybeIds.size} maybe, ${passedIds.size} passed`}
    >
      {visible.map((clip) => (
        <i
          key={clip.id}
          className={`${clip.id === selectedClipId ? "is-now" : ""} ${savedSet.has(clip.id) ? "is-kept" : ""} ${maybeIds.has(clip.id) ? "is-maybe" : ""} ${passedIds.has(clip.id) ? "is-passed" : ""}`}
        />
      ))}
      {hidden > 0 && <small className="review-dots-more">+{hidden}</small>}
    </div>
  );
}

const THEATER_DECISION_MS = 450;

/** Per-press nudge, in seconds. One second is the smallest step that reads as a
 *  deliberate change on a 10-30s clip; Shift takes the coarse step for the "this
 *  should have started way earlier" case without turning it into ten presses. */
const TRIM_STEP_SEC = 1;
const TRIM_STEP_COARSE_SEC = 5;
/** Never let a nudge produce a clip too short to be a clip. */
const TRIM_MIN_DURATION_SEC = 2;

export function StudioTheaterView({
  session,
  clip,
  clips,
  timeline: _timeline,
  videoRef,
  onBack,
  onKeep,
  onCut,
  onMaybe,
  onSelect,
  onExport,
  onTrim,
  onRename,
  exporting,
  moreCandidateCount,
  moreCandidatesEnabled,
  moreCandidatesLoading,
  preparingPreview,
  previewBusy,
  onToggleMoreCandidates,
  onPreparePreview,
}: {
  session?: Session;
  clip?: Clip;
  clips: Clip[];
  timeline: ReactionTimeline | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  onBack: () => void;
  onKeep: (clip: Clip) => void;
  onCut: (clip: Clip) => void;
  onMaybe: (clip: Clip) => void;
  onSelect: (id: string) => void;
  onExport: () => void;
  onTrim: (clip: Clip, start: number, end: number) => Promise<void>;
  onRename: (clip: Clip, title: string) => Promise<void>;
  exporting: boolean;
  moreCandidateCount: number;
  moreCandidatesEnabled: boolean;
  moreCandidatesLoading: boolean;
  preparingPreview: boolean;
  previewBusy: boolean;
  onToggleMoreCandidates: () => void;
  onPreparePreview: () => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const decisionTimerRef = useRef<number | null>(null);
  const [decision, setDecision] = useState<"keep" | "maybe" | "pass" | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [videoDuration, setVideoDuration] = useState(0);
  const [videoTime, setVideoTime] = useState(0);
  const mediaVolume = useMediaVolume([videoRef], clip?.id);
  // Pending boundary nudges, in seconds relative to the clip's stored window.
  // Held rather than applied per press: every apply re-renders the clip on the
  // backend, so four taps to find the right end point would be four renders.
  const [trim, setTrim] = useState<{ start: number; end: number }>({ start: 0, end: 0 });
  const [trimming, setTrimming] = useState(false);
  const [trimError, setTrimError] = useState<string | null>(null);
  const [titleEditing, setTitleEditing] = useState(false);
  const [titleDraft, setTitleDraft] = useState(clip?.title || "");
  const [titleSaving, setTitleSaving] = useState(false);
  const [titleError, setTitleError] = useState<string | null>(null);
  const exportNameSettings = useJobStore(useShallow((state) => ({
    filenameTemplate: state.settings.filenameTemplate,
    exportPreset: state.settings.exportPreset,
  })));

  const index = clip ? clips.findIndex((item: Clip) => item.id === clip.id) : -1;
  const clipStart = clip?.sourceWindow ? clip.start_time ?? 0 : 0;
  const activeVideoDuration = clip?.sourceWindow ? clip.duration || 0 : videoDuration || clip?.duration || 0;
  const safeVideoTime = Math.max(0, Math.min(videoTime, activeVideoDuration));
  const seekPercent = activeVideoDuration > 0 ? (safeVideoTime / activeVideoDuration) * 100 : 0;

  const score = clip ? hookScoreOf(clip) : 0;
  const savedSet = useMemo(() => new Set<string>(session?.savedClipIds ?? []), [session?.savedClipIds]);
  const passedIds = useMemo(
    () => new Set<string>((session?.clips ?? []).filter((c: Clip) => c.passed).map((c: Clip) => c.id)),
    [session?.clips],
  );
  const maybeIds = useMemo(
    () => new Set<string>((session?.clips ?? []).filter((c: Clip) => c.maybe).map((c: Clip) => c.id)),
    [session?.clips],
  );
  const priorityIds = useMemo(
    () => new Set(primaryReviewDeckClips(session?.clips ?? []).slice(0, 3).map((item) => item.id)),
    [session?.clips],
  );
  const recommendation = clip
    ? editorialRecommendation(savedSet.has(clip.id), passedIds.has(clip.id), maybeIds.has(clip.id), priorityIds.has(clip.id))
    : { label: "", className: "" };
  const opening = openingBand(score);
  const clues = clip ? creatorClues(clip) : [];

  const nextClips = useMemo(() => (index >= 0 ? clips.slice(index + 1, index + 8) : []), [clips, index]);
  const reviewDone = !!clip && index === clips.length - 1 && (savedSet.has(clip.id) || passedIds.has(clip.id) || maybeIds.has(clip.id)) && !decision;

  // --- Boundary nudges -------------------------------------------------------
  // The instinct to re-cut arrives while WATCHING, so it belongs here rather
  // than only in the Clip Library's editor. The backend also treats a boundary
  // change as supervision ("the engine's window was wrong" -- services.py
  // record_boundary_edit), so catching it at the moment of the reaction is
  // worth more than catching it later, or not at all.
  const storedStart = clip?.start_time ?? 0;
  const storedEnd = clip?.end_time ?? storedStart + (clip?.duration ?? 0);
  const sourceLimit = session?.vodDuration;
  const nextStart = storedStart + trim.start;
  const nextEnd = storedEnd + trim.end;
  const trimDirty = trim.start !== 0 || trim.end !== 0;
  const trimmedDuration = Math.max(0, nextEnd - nextStart);
  const previewClipNumber = clip?.clipNumber ?? Math.max(1, index + 1);
  const previewMaxClipNumber = Math.max(
    previewClipNumber,
    ...(session?.clips.map((item, clipIndex) => item.clipNumber ?? clipIndex + 1) ?? [1]),
  );
  const exportNamePreview = clip ? creatorExportFilename({
    title: titleEditing ? titleDraft : clip.title,
    clipNumber: previewClipNumber,
    maxClipNumber: previewMaxClipNumber,
    sourceDate: session?.sourceDate,
    preset: exportNameSettings.exportPreset,
    template: exportNameSettings.filenameTemplate,
    originalStem: clip.originalStem || `clip_${clip.id}`,
  }) : "";

  const cancelTitleEdit = () => {
    setTitleDraft(clip?.title || "");
    setTitleEditing(false);
    setTitleError(null);
  };

  const saveTitle = async (event: FormEvent) => {
    event.preventDefault();
    if (!clip || titleSaving) return;
    const title = titleDraft.trim();
    if (!title) {
      setTitleError("Give this clip a title before saving.");
      return;
    }
    if (title === clip.title) {
      cancelTitleEdit();
      return;
    }
    setTitleSaving(true);
    setTitleError(null);
    try {
      await onRename(clip, title);
      setTitleEditing(false);
    } catch (error) {
      setTitleError(error instanceof Error ? error.message : "Recall could not rename this clip.");
    } finally {
      setTitleSaving(false);
    }
  };

  /** Clamp a proposed nudge against the source bounds and the minimum length. */
  const nudge = (edge: "start" | "end", direction: -1 | 1, coarse: boolean) => {
    if (trimming) return;
    const step = (coarse ? TRIM_STEP_COARSE_SEC : TRIM_STEP_SEC) * direction;
    setTrimError(null);
    setTrim((current) => {
      const proposedStart = storedStart + (edge === "start" ? current.start + step : current.start);
      const proposedEnd = storedEnd + (edge === "end" ? current.end + step : current.end);
      // A clip cannot start before the VOD, end after it, or invert itself.
      if (proposedStart < 0) return current;
      if (sourceLimit !== undefined && proposedEnd > sourceLimit) return current;
      if (proposedEnd - proposedStart < TRIM_MIN_DURATION_SEC) return current;
      return edge === "start"
        ? { ...current, start: current.start + step }
        : { ...current, end: current.end + step };
    });
  };

  const resetTrim = () => { setTrim({ start: 0, end: 0 }); setTrimError(null); };

  const applyTrim = async () => {
    if (!clip || !trimDirty || trimming) return;
    setTrimming(true);
    setTrimError(null);
    const restorePreview = releaseClipPreview(clip.videoUrl);
    try {
      await onTrim(clip, nextStart, nextEnd);
      setTrim({ start: 0, end: 0 });
    } catch (error) {
      setTrimError(error instanceof Error && error.message !== "network"
        ? error.message
        : "Recall could not re-cut this clip.");
    } finally {
      restorePreview();
      setTrimming(false);
      focusTheater();
    }
  };

  const focusTheater = () => {
    videoRef.current?.blur();
    rootRef.current?.focus({ preventScroll: true });
  };

  const decide = (kind: "keep" | "maybe" | "pass") => {
    if (!clip || decision) return;
    if (kind === "keep") onKeep(clip);
    else if (kind === "maybe") onMaybe(clip);
    else onCut(clip);
    setDecision(kind);
    if (decisionTimerRef.current) window.clearTimeout(decisionTimerRef.current);
    const nextId = index >= 0 && index < clips.length - 1 ? clips[index + 1].id : null;
    decisionTimerRef.current = window.setTimeout(() => {
      decisionTimerRef.current = null;
      setDecision(null);
      if (nextId) onSelect(nextId);
    }, THEATER_DECISION_MS);
  };

  useEffect(() => {
    if (!clip) return;
    if (decisionTimerRef.current) {
      window.clearTimeout(decisionTimerRef.current);
      decisionTimerRef.current = null;
    }
    setDecision(null);
    setIsPlaying(true);
    setVideoTime(0);
    // Pending nudges belong to the clip they were made on, never the next one.
    setTrim({ start: 0, end: 0 });
    setTrimError(null);
    setTitleDraft(clip.title || "");
    setTitleEditing(false);
    setTitleSaving(false);
    setTitleError(null);
    const frame = window.requestAnimationFrame(() => focusTheater());
    return () => window.cancelAnimationFrame(frame);
  }, [clip?.id]);

  useEffect(() => () => {
    if (decisionTimerRef.current) window.clearTimeout(decisionTimerRef.current);
  }, []);

  // Sync video play/pause status when state or clip changes
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !clip) return;
    if (isPlaying) {
      video.play().catch(() => setIsPlaying(false));
    } else {
      video.pause();
    }
  }, [isPlaying, clip?.videoUrl, videoRef]);

  // Adjust duration on meta load
  useEffect(() => {
    setVideoDuration(clip?.duration ?? 0);
  }, [clip?.duration, clip?.id]);

  useEffect(() => {
    if (!clip) return;
    const onKey = (event: KeyboardEvent) => {
      if (isInteractiveKeyboardTarget(event.target)) return;

      if (event.key === "Escape") {
        event.preventDefault();
        onBack();
        return;
      }

      if (decision) {
        if (event.key === " " || event.key === "ArrowLeft" || event.key === "ArrowRight"
          || event.key === "Backspace" || event.key.toLowerCase() === "x"
          || event.key.toLowerCase() === "m") {
          event.preventDefault();
          event.stopPropagation();
        }
        return;
      }

      if (event.key === "ArrowRight") {
        event.preventDefault();
        event.stopPropagation();
        if (index < clips.length - 1) onSelect(clips[index + 1].id);
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        event.stopPropagation();
        if (index > 0) onSelect(clips[index - 1].id);
        return;
      }
      if (event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        decide("keep");
        return;
      }
      if (event.key === "Backspace" || event.key.toLowerCase() === "x") {
        event.preventDefault();
        event.stopPropagation();
        decide("pass");
        return;
      }
      if (event.key.toLowerCase() === "m") {
        event.preventDefault();
        event.stopPropagation();
        decide("maybe");
        return;
      }

      // Boundary nudges. Bracket keys move the IN point, comma/period the OUT
      // point — the pairing editors have used for decades — so neither collides
      // with the decision keys above or the arrow-key clip navigation.
      // Shifted forms included on purpose: Shift is the coarse-step modifier, and
      // holding it turns "[" into "{" and "," into "<", so keying off the
      // character alone would silently drop every coarse nudge.
      const trimKeys: Record<string, ["start" | "end", -1 | 1]> = {
        "[": ["start", -1], "]": ["start", 1],
        "{": ["start", -1], "}": ["start", 1],
        ",": ["end", -1], ".": ["end", 1],
        "<": ["end", -1], ">": ["end", 1],
      };
      const move = trimKeys[event.key];
      if (move) {
        event.preventDefault();
        event.stopPropagation();
        nudge(move[0], move[1], event.shiftKey);
        return;
      }
      if (event.key === "Enter" && trimDirty) {
        event.preventDefault();
        event.stopPropagation();
        void applyTrim();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clips, index, decision, clip, onBack, onKeep, onMaybe, onCut, onSelect,
      trimDirty, trimming, storedStart, storedEnd, sourceLimit, trim]);

  const seekClip = (time: number) => {
    const nextTime = Math.max(0, Math.min(time, activeVideoDuration || time));
    if (videoRef.current) {
      videoRef.current.currentTime = clipStart + nextTime;
    }
    setVideoTime(nextTime);
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const togglePlayback = () => {
    setIsPlaying((value) => !value);
    focusTheater();
  };

  if (!session || !clip) {
    return <div className="clips-empty"><StudioEmptyState compact variant="review" icon={<Film />} title="Choose a clip from the review deck" body="Theater opens the selected moment with its reaction evidence and edit controls." /></div>;
  }

  return (
    <div className="review-theater-screen" ref={rootRef} tabIndex={-1}>
      <TheaterBackdrop clip={clip} />
      <div className="theater-top">
        <button type="button" className="theater-back" onClick={onBack}>
          <ArrowLeft size={15} /> All drafts
        </button>
        <div className="theater-top-actions">
          {clip.sourceWindow && (
            <button
              type="button"
              className="theater-more-moments theater-framing-proof"
              onClick={onPreparePreview}
              disabled={previewBusy}
              title={previewBusy && !preparingPreview ? "Another framing proof is finishing" : "Render the exact vertical facecam and gameplay composition"}
            >
              {preparingPreview ? <RefreshCw size={13} className="is-spinning" /> : <Video size={13} />}
              <span>{preparingPreview ? "Rendering proof" : "Preview vertical"}</span>
            </button>
          )}
          {moreCandidateCount > 0 && (
            <button
              type="button"
              className={`theater-more-moments ${moreCandidatesEnabled ? "is-active" : ""}`}
              onClick={onToggleMoreCandidates}
              disabled={moreCandidatesLoading}
              aria-pressed={moreCandidatesEnabled}
              title="Review the strongest moments saved just outside the review deck"
            >
              {moreCandidatesLoading ? <RefreshCw size={13} className="is-spinning" /> : <Sparkles size={13} />}
              <span>More moments</span>
              <small>{moreCandidatesEnabled ? "On" : `+${moreCandidateCount}`}</small>
            </button>
          )}
          <div className="theater-position">
            <ProgressDots clips={clips} selectedClipId={clip.id} savedSet={savedSet} passedIds={passedIds} maybeIds={maybeIds} />
            <span>{index + 1} / {clips.length}</span>
          </div>
        </div>
      </div>

      <section className="theater-center">
        <div className="theater-meta">
          <div className="theater-editorial-row">
            <span className={`theater-recommendation ${recommendation.className}`}>{recommendation.label}</span>
            <span className={`theater-opening ${opening.className}`} title="Opening strength, not overall clip quality">{opening.label}</span>
            {clip.momentType && (
              <span className="moment-type-badge">{momentLabel(clip.momentType)}</span>
            )}
            {clip.isOverflowCandidate && <span className="theater-second-look">Second look</span>}
            <RecallOriginBadge provenance={clip.recallProvenance} />
          </div>

          <div className="theater-title-block">
            {titleEditing ? (
              <form className="theater-title-editor" onSubmit={(event) => void saveTitle(event)}>
                <input
                  autoFocus
                  value={titleDraft}
                  maxLength={200}
                  aria-label="Clip title"
                  aria-invalid={!!titleError}
                  onChange={(event) => { setTitleDraft(event.target.value); setTitleError(null); }}
                  onKeyDown={(event) => {
                    if (event.key === "Escape") {
                      event.preventDefault();
                      cancelTitleEdit();
                    }
                  }}
                />
                <button type="submit" className="theater-title-save" disabled={titleSaving || !titleDraft.trim()}>{titleSaving ? "Saving…" : "Save"}</button>
                <button type="button" className="theater-title-cancel" onClick={cancelTitleEdit} disabled={titleSaving}>Cancel</button>
              </form>
            ) : (
              <div className="theater-title-row">
                <h1>{clip.title || "Untitled moment"}</h1>
                <button type="button" className="theater-title-rename" onClick={() => { setTitleDraft(clip.title || ""); setTitleEditing(true); }} aria-label={`Rename ${clip.title || "clip"}`}>Rename</button>
              </div>
            )}
            {titleError && <p className="theater-title-error" role="alert">{titleError}</p>}
            <div className="theater-filename-preview" title={exportNamePreview}>
              <span>Export name</span>
              <code>{exportNamePreview}</code>
            </div>
          </div>
          {/* Where this sits in the source recording. Reviewers need it to find
              the moment in the VOD (or in their own editor) and to tell two
              similar-looking clips apart; it used to exist only as a fallback
              inside the title string. */}
          <div className="theater-vod-position">
            <Clock size={12} aria-hidden="true" />
            <span className="theater-vod-position-label">In the VOD</span>
            <b className="t-num">
              {clip.timestamp}
              {clip.end_time != null && <> – {fmtClock(clip.end_time)}</>}
            </b>
            <span className="theater-vod-position-sep" aria-hidden="true">·</span>
            <span className="t-num">{fmtClock(durationOf(clip))} long</span>
          </div>
          {clip.hookLine && <p className="theater-hook-line">{clip.hookLine}</p>}
          <div className="theater-why">
            <span className="theater-why-label">Why Recall chose this</span>
            <p className="theater-evidence-line">{clip.reason || "Recall found several signs that this moment is worth reviewing."}</p>
            <div className="theater-chip-row">
              {clues.map((clue) => <span key={clue} className="chip">{clue}</span>)}
            </div>
          </div>
          <details className="theater-analysis">
            <summary>How Recall read this moment</summary>
            <div className="theater-opening-detail">
              <span>Opening signal</span>
              <b>{score} / 100</b>
              <small>This measures the first few seconds, not the clip's overall posting quality.</small>
            </div>
            {clip.modalityBreakdown && <ModalityMix breakdown={clip.modalityBreakdown} />}
          </details>
        </div>

        <div className="theater-frame-zone">
          <div className="theater-frame">
            {clip.videoUrl ? (
              <video
                key={`${clip.id}:${clip.videoUrl}`}
                ref={videoRef}
                src={trimming ? undefined : clip.videoUrl}
                poster={clip.thumbUrl}
                muted={mediaVolume.muted}
                loop={!clip.sourceWindow}
                playsInline
                tabIndex={-1}
                preload="metadata"
                onClick={togglePlayback}
                onLoadedMetadata={(event) => {
                  const duration = event.currentTarget.duration;
                  if (clip.sourceWindow) {
                    event.currentTarget.currentTime = clipStart;
                    setVideoDuration(clip.duration || 0);
                    setVideoTime(0);
                  } else {
                    setVideoDuration(Number.isFinite(duration) ? duration : clip.duration);
                    setVideoTime(event.currentTarget.currentTime || 0);
                  }
                  focusTheater();
                }}
                onTimeUpdate={(event) => {
                  const current = event.currentTarget.currentTime || 0;
                  if (clip.sourceWindow && current >= (clip.end_time ?? clipStart + activeVideoDuration) - 0.04) {
                    event.currentTarget.currentTime = clipStart;
                    setVideoTime(0);
                    return;
                  }
                  setVideoTime(Math.max(0, current - clipStart));
                }}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
              />
            ) : (
              <div className="theater-frame-fallback"><Play size={34} /></div>
            )}
            {decision && (
              <div className={`theater-decision is-${decision}`} role="status" aria-live="polite">
                <strong>{decision === "keep" ? "Keep" : decision === "maybe" ? "Maybe" : "Pass"}</strong>
              </div>
            )}
            {preparingPreview && (
              <div className="theater-preparing-preview" role="status" aria-live="polite">
                <RefreshCw aria-hidden="true" className="is-spinning" />
                <strong>Rendering vertical proof</strong>
                <span>Keep reviewing the source while Recall applies this scene&apos;s framing.</span>
              </div>
            )}
            <MediaVolumeControl
              className="theater-volume"
              orientation="vertical"
              collapsible
              volume={mediaVolume.volume}
              muted={mediaVolume.muted}
              onVolumeChange={mediaVolume.setVolume}
              onToggleMuted={mediaVolume.toggleMuted}
            />
            <div className="theater-scrub" style={{ "--seek": `${seekPercent}%` } as React.CSSProperties}>
              <div className="theater-time-row">
                <span>{formatTime(safeVideoTime)}</span>
                <span>{formatTime(activeVideoDuration)}</span>
              </div>
              <input
                className="theater-seeker"
                type="range"
                min={0}
                max={Math.max(activeVideoDuration, 0.1)}
                step={0.05}
                value={safeVideoTime}
                onChange={(event) => seekClip(Number(event.currentTarget.value))}
                aria-label="Seek clip"
              />
            </div>
          </div>

          {/* Re-cut without leaving the review. Nudges are held and applied in
              one render — see applyTrim. */}
          <div className={`theater-trim ${trimDirty ? "is-dirty" : ""}`}>
            <div className="theater-trim-edge">
              <span className="theater-trim-label">Start</span>
              <button type="button" onClick={(e) => nudge("start", -1, e.shiftKey)} disabled={trimming} aria-label="Move the start earlier, making the clip longer">
                <ChevronLeft size={13} />
              </button>
              <button type="button" onClick={(e) => nudge("start", 1, e.shiftKey)} disabled={trimming} aria-label="Move the start later, making the clip shorter">
                <ChevronRight size={13} />
              </button>
            </div>

            <div className="theater-trim-readout">
              <strong>{formatTime(trimmedDuration)}</strong>
              {trimDirty && (
                <em>
                  {trim.start !== 0 && `start ${trim.start > 0 ? "+" : ""}${trim.start}s`}
                  {trim.start !== 0 && trim.end !== 0 && " · "}
                  {trim.end !== 0 && `end ${trim.end > 0 ? "+" : ""}${trim.end}s`}
                </em>
              )}
            </div>

            <div className="theater-trim-edge">
              <span className="theater-trim-label">End</span>
              <button type="button" onClick={(e) => nudge("end", -1, e.shiftKey)} disabled={trimming} aria-label="Move the end earlier, making the clip shorter">
                <ChevronLeft size={13} />
              </button>
              <button type="button" onClick={(e) => nudge("end", 1, e.shiftKey)} disabled={trimming} aria-label="Move the end later, making the clip longer">
                <ChevronRight size={13} />
              </button>
            </div>

            {trimDirty && (
              <div className="theater-trim-commit">
                <button type="button" className="theater-trim-reset" onClick={resetTrim} disabled={trimming}>Reset</button>
                <button type="button" className="theater-trim-apply" onClick={applyTrim} disabled={trimming}>
                  {trimming ? <><RefreshCw size={13} className="is-spinning" aria-hidden="true" />Re-cutting…</> : "Apply · Enter"}
                </button>
              </div>
            )}
          </div>
          {trimError && <p className="theater-trim-error" role="alert">{trimError}</p>}
        </div>

        <div className="theater-actions">
          <button type="button" className="theater-keep" onClick={() => decide("keep")} disabled={!!decision} aria-label="Keep clip">
            <Check size={34} />
          </button>
          <span className="is-keep">Keep · Space</span>
          <button type="button" className="theater-maybe" onClick={() => decide("maybe")} disabled={!!decision} aria-label="Mark clip as maybe">
            <Clock size={21} />
          </button>
          <span className="is-maybe">Maybe · M</span>
          <button type="button" className="theater-pass" onClick={() => decide("pass")} disabled={!!decision} aria-label="Pass clip">
            <X size={23} />
          </button>
          <span>Pass · X</span>
        </div>
      </section>

      <section className="theater-next">
        {reviewDone ? (
          <div className="theater-wrapup">
            <Sparkles size={17} />
            <strong>Reviewed {clips.length} · kept {savedSet.size} · maybe {maybeIds.size}</strong>
            <StudioButton tone="primary" loading={exporting} onClick={onExport}>Export kept clips</StudioButton>
          </div>
        ) : (
          <>
            <span>Up next</span>
            <div className="theater-next-row">
              {nextClips.map((c: Clip) => (
                <button type="button" key={c.id} onClick={() => onSelect(c.id)} disabled={!!decision}>
                  <Poster clip={c} />
                  <b>{openingBand(hookScoreOf(c)).shortLabel}</b>
                </button>
              ))}
            </div>
            <div className="theater-hints"><kbd>Space</kbd> keep <kbd>M</kbd> maybe <kbd>X</kbd> pass <kbd>←</kbd>/<kbd>→</kbd> next <kbd>[</kbd><kbd>]</kbd> start <kbd>,</kbd><kbd>.</kbd> end <kbd>Esc</kbd> grid</div>
          </>
        )}
      </section>
    </div>
  );
}

function StudioReelView({
  sessions,
  session,
  clips,
  compiling,
  onCompile,
  onSelectSession,
  onOpenReview,
  onPickSession,
}: {
  sessions: Session[];
  session?: Session;
  clips: Clip[];
  compiling: boolean;
  onCompile: (clipIds: string[]) => void;
  onSelectSession: (sessionId: string) => void;
  onOpenReview: () => void;
  onPickSession: () => void;
}) {
  const keptCount = session?.savedClipIds.length ?? 0;
  const [choosingSession, setChoosingSession] = useState(!session);
  const [sessionQuery, setSessionQuery] = useState("");
  // Clips the creator has dropped from THIS session's reel. An exclusion set is
  // right here (unlike the Library): a reel is "this session's kept moments,
  // minus the ones I don't want", so newly kept clips join automatically.
  const [droppedIds, setDroppedIds] = useState<Set<string>>(new Set());
  const matchingSessions = useMemo(() => {
    const needle = sessionQuery.trim().toLowerCase();
    return needle ? sessions.filter((item) => item.name.toLowerCase().includes(needle)) : sessions;
  }, [sessions, sessionQuery]);
  const readySessions = matchingSessions.filter((item) => item.savedClipIds.length >= 2);
  const incompleteSessions = matchingSessions.filter((item) => item.savedClipIds.length < 2);

  const reelClips = useMemo(() => clips.filter((clip) => !droppedIds.has(clip.id)), [clips, droppedIds]);
  const totalSeconds = reelClips.reduce((sum, clip) => sum + durationOf(clip), 0);
  const droppedCount = clips.length - reelClips.length;
  const canCompile = reelClips.length >= 2;

  useEffect(() => {
    if (!session) setChoosingSession(true);
  }, [session]);

  // A different session means a different reel; never carry drops across.
  useEffect(() => { setDroppedIds(new Set()); }, [session?.id]);

  const toggleClip = (clipId: string) => setDroppedIds((prev) => {
    const next = new Set(prev);
    if (next.has(clipId)) next.delete(clipId); else next.add(clipId);
    return next;
  });

  const chooseSession = (sessionId: string) => {
    onSelectSession(sessionId);
    if (sessionId) setChoosingSession(false);
  };

  return (
    <div className="reel-workspace">
      <header className={`reel-page-head ${session && !choosingSession ? "is-reel-ready" : ""}`}>
        <div className="reel-page-title">
          <h1>Highlight Reel</h1>
          <p>Choose one session. Recall will keep its moments in VOD order.</p>
        </div>
        {(choosingSession || !session) && sessions.length > 0 && (
          <div className="reel-page-tools">
            <label className="reel-session-search">
              <Search size={14} aria-hidden="true" />
              <input value={sessionQuery} onChange={(event) => setSessionQuery(event.currentTarget.value)} placeholder="Search sessions" aria-label="Search sessions for a highlight reel" />
            </label>
            {session && <button type="button" className="btn-secondary" onClick={() => setChoosingSession(false)}>Cancel</button>}
          </div>
        )}
        {session && !choosingSession && (
          <div className="reel-session-context" aria-label={`Selected session: ${session.name}`}>
            <span className="reel-session-context-poster" aria-hidden="true"><SessionPoster session={session} className="reel-session-poster-media" /></span>
            <span className="reel-session-context-copy">
              <small>Session</small>
              <strong>{session.name}</strong>
              <span>{keptCount} kept {keptCount === 1 ? "clip" : "clips"}</span>
            </span>
            <button type="button" className="btn-secondary reel-change-session" onClick={() => setChoosingSession(true)}>Change session</button>
          </div>
        )}
      </header>

      {choosingSession || !session ? (
        <section className="reel-session-browser" aria-labelledby="reel-session-browser-title">
          {/* No pager: the gallery auto-fills the available width and the
              catalog scrolls. Six-at-a-time paging over a "1 / 10" counter
              wasted most of a large window and hid sessions behind clicks. */}
          <div className="reel-session-browser-head">
            <div>
              <h2 id="reel-session-browser-title"><StateIcon tone="success" animate={false} icon={<Check size={14} />} />Ready for a reel</h2>
            </div>
            {readySessions.length > 0 && (
              <span className="reel-ready-count t-num">{plural(readySessions.length, "session")}</span>
            )}
          </div>
          {sessions.length ? (
            <div className="reel-session-catalog">
              {readySessions.length ? (
                <div className="reel-ready-gallery">
                  {readySessions.map((item) => {
                    const count = item.savedClipIds.length;
                    return (
                      <button type="button" key={item.id} className={`reel-session-card ${item.id === session?.id ? "is-current" : ""}`} onClick={() => chooseSession(item.id)}>
                        <span className="reel-session-card-media" aria-hidden="true"><SessionPoster session={item} variant="card" /></span>
                        <span className="reel-session-card-body">
                          <strong>{item.name}</strong>
                          <span>{plural(count, "kept clip")} · {new Date(item.updatedAt).toLocaleDateString([], { month: "short", day: "numeric" })}</span>
                          <small><i aria-hidden="true" />Ready</small>
                        </span>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <div className="reel-session-no-match" role="status"><Search size={16} /><span>{sessionQuery.trim() ? `No ready sessions match “${sessionQuery.trim()}”.` : "No sessions have two kept clips yet."}</span></div>
              )}

              {!!incompleteSessions.length && (
                <section className="reel-incomplete-section" aria-labelledby="reel-incomplete-title">
                  <div className="reel-incomplete-head">
                    <h3 id="reel-incomplete-title"><Clock size={15} />Needs more moments</h3>
                    <span>Open one to keep more clips in Theater Review.</span>
                  </div>
                  <div className="reel-incomplete-shelf">
                    {incompleteSessions.map((item) => {
                      const count = item.savedClipIds.length;
                      return (
                        <button type="button" key={item.id} className="reel-incomplete-card" onClick={() => chooseSession(item.id)}>
                          <span className="reel-incomplete-media" aria-hidden="true"><SessionPoster session={item} className="reel-session-poster-media" /></span>
                          <span className="reel-incomplete-copy">
                            <strong>{item.name}</strong>
                            <small>{count} kept {count === 1 ? "clip" : "clips"}</small>
                          </span>
                          <ChevronRight aria-hidden="true" />
                        </button>
                      );
                    })}
                  </div>
                </section>
              )}
            </div>
          ) : (
            <StudioEmptyState
              variant="library"
              icon={<Film />}
              title="No sessions yet"
              body="Finish a scan and keep at least two moments before building a highlight reel."
              action={<button type="button" className="btn-secondary" onClick={onPickSession}><Film size={14} /><span>Open Sessions</span></button>}
            />
          )}
        </section>
      ) : clips.length < 2 ? (
        <section className="reel-gate" aria-label="More kept clips needed">
          <StudioEmptyState
            variant="library"
            icon={<Sparkles />}
            title="Keep a few more moments"
            body={`${session.name} has ${clips.length} kept ${clips.length === 1 ? "clip" : "clips"}. A highlight reel needs at least 2.`}
            action={<button type="button" className="cta-accent" onClick={onOpenReview}><Target size={14} /><span>Open Theater Review</span></button>}
          />
        </section>
      ) : (
        <div className="reel-conform-layout">
          <section className="reel-output-stage" aria-label="Vertical reel preview">
            <div className="reel-output-stage-frame">
              <div className="reel-conform-preview">
                <div className="reel-conform-markers" aria-hidden="true">
                  {reelClips.map((clip, index) => (
                    <span
                      key={clip.id}
                      style={{ "--reel-shot-weight": Math.max(1, durationOf(clip)) } as CSSProperties}
                      className="t-num"
                    >{index + 1}</span>
                  ))}
                </div>
                <div className="reel-conform-frame">
                  <span className="reel-conform-tag reel-conform-resolution t-num">1080 × 1920</span>
                  <div className="reel-conform-strip">
                    {reelClips.map((clip, index) => (
                      <div key={clip.id} className="reel-conform-shot" style={{ "--reel-shot-weight": Math.max(1, durationOf(clip)) } as CSSProperties}>
                        <Poster clip={clip} className="reel-conform-media" eager={index < 3} />
                      </div>
                    ))}
                  </div>
                  <span className="reel-conform-tag reel-conform-duration t-num">{fmtClock(totalSeconds)}</span>
                  <span className="reel-conform-tag reel-conform-ratio t-num">9:16</span>
                </div>
              </div>
              <div className={`reel-ready-state ${canCompile ? "" : "is-blocked"}`} role="status">
                <Check size={13} aria-hidden="true" />
                <span>{canCompile ? `Ready to compile · ${plural(reelClips.length, "clip")}` : "Keep at least 2 clips in the reel"}</span>
              </div>
            </div>
          </section>

          <section className="reel-story-workbench">
            <div className="reel-story-head">
              <h2>Reel sequence</h2>
              <span className="reel-story-head-meta t-num">
                {reelClips.length} of {clips.length} in the reel · locked to VOD order
              </span>
              {droppedCount > 0 && (
                <button type="button" className="library-linkbtn" onClick={() => setDroppedIds(new Set())}>
                  Add all back
                </button>
              )}
            </div>
            {/* Every kept clip is in by default; dropping one is explicit and
                only affects this reel — it never un-keeps the clip. */}
            <ol className="reel-storyboard" aria-label="Reel clips in VOD order">
              {clips.map((clip, index) => {
                const included = !droppedIds.has(clip.id);
                const position = reelClips.findIndex((item) => item.id === clip.id) + 1;
                return (
                  <li key={clip.id} className={`reel-story-card ${included ? "" : "is-dropped"}`}>
                    <span className="reel-story-card-media">
                      <Poster clip={clip} className="reel-story-media" eager={index < 6} />
                      <span className="reel-story-order t-num">{included ? position : "—"}</span>
                      <button
                        type="button"
                        className="reel-story-toggle"
                        role="checkbox"
                        aria-checked={included}
                        aria-label={`${included ? "Remove" : "Add"} ${clip.title} ${included ? "from" : "to"} the reel`}
                        title={included ? "Remove from this reel" : "Add back to this reel"}
                        onClick={() => toggleClip(clip.id)}
                      >
                        {included ? <Check size={13} aria-hidden="true" /> : <Plus size={13} aria-hidden="true" />}
                      </button>
                    </span>
                    <span className="reel-story-card-copy">
                      <strong>{clip.title}</strong>
                      <span><small className="t-num">{clip.timestamp}</small><small className="t-num">{fmtClock(durationOf(clip))}</small></span>
                    </span>
                  </li>
                );
              })}
            </ol>

            <footer className="reel-export-panel">
              <div className="reel-export-facts" aria-label="Reel output details">
                <span><Film size={15} aria-hidden="true" /><b>{plural(reelClips.length, "moment")}</b></span>
                <span><Clock size={15} aria-hidden="true" /><b className="t-num">{fmtClock(totalSeconds)} runtime</b></span>
                <span><Maximize2 size={15} aria-hidden="true" /><b className="t-num">1080 × 1920</b></span>
                <span><ChevronRight size={15} aria-hidden="true" /><b>VOD order</b></span>
                <span><Scissors size={15} aria-hidden="true" /><b>Hard cuts</b></span>
              </div>
              <div className="reel-compile-row">
                <p>{droppedCount > 0 ? `${plural(droppedCount, "clip")} left out. Choose a folder when you compile.` : "Choose a folder when you compile."}</p>
                <StudioButton
                  tone="primary"
                  className="reel-compile-button"
                  icon={<Film size={16} />}
                  loading={compiling}
                  disabled={!canCompile}
                  onClick={() => onCompile(reelClips.map((clip) => clip.id))}
                >
                  {compiling ? "Compiling reel…" : canCompile ? `Compile ${reelClips.length}-clip reel` : "Add another clip to compile"}
                </StudioButton>
              </div>
            </footer>
          </section>
        </div>
      )}
    </div>
  );
}

function StudioClipsView({ items, selectedIds, onToggleSelect, onSelectAll, onClearSelection, onToggleSession, onEdit, onExport, exporting, onEmptyAction, emptyActionLabel }: {
  items: { clip: Clip; session: Session }[];
  selectedIds: Set<string>;
  onToggleSelect: (clipId: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onToggleSession: (sessionId: string) => void;
  onEdit: (session: Session, clip: Clip) => void;
  onExport: () => void;
  exporting: boolean;
  onEmptyAction: () => void;
  emptyActionLabel: string;
}) {
  const selectedCount = items.reduce((sum, { clip }) => sum + (selectedIds.has(clip.id) ? 1 : 0), 0);
  const allSelected = items.length > 0 && selectedCount === items.length;
  const noneSelected = selectedCount === 0;
  // The library spans every session, so the session IS the unit a creator
  // thinks in ("export this VOD's clips"). Group by it and hang the bulk
  // control off each group head.
  const groups = useMemo(() => {
    const order: string[] = [];
    const bySession = new Map<string, { session: Session; clips: Clip[] }>();
    for (const { clip, session } of items) {
      let group = bySession.get(session.id);
      if (!group) {
        group = { session, clips: [] };
        bySession.set(session.id, group);
        order.push(session.id);
      }
      group.clips.push(clip);
    }
    return order.map((id) => bySession.get(id)!);
  }, [items]);
  return (
    <div className="clips-shell">
      <div className="clips-head">
        <h1>Clips Library</h1>
        <span>{items.length ? `${plural(items.length, "kept clip")} · ${selectedCount} selected for export` : "Kept clips land here for final review and export."}</span>
        {items.length > 0 && (
          <div className="library-actions">
            <div className="library-select-controls" role="group" aria-label="Export selection">
              <button type="button" className="library-linkbtn" onClick={onSelectAll} disabled={allSelected}>Select all</button>
              <span aria-hidden="true">·</span>
              <button type="button" className="library-linkbtn" onClick={onClearSelection} disabled={noneSelected}>Clear</button>
            </div>
            <button className="cta-accent mock-v1-library-export" onClick={onExport} disabled={exporting || noneSelected}>{exporting ? "Preparing final files…" : noneSelected ? "Select clips to export" : `Export ${plural(selectedCount, "clip")}`}</button>
          </div>
        )}
      </div>
      {!items.length ? (
        <div className="clips-empty"><StudioEmptyState variant="library" icon={<Library />} title="Build your export reel" body="Keep the moments that make the cut in Theater Review. Their final files will collect here for export." action={<button type="button" className="cta-accent" onClick={onEmptyAction}><Target size={14} /><span>{emptyActionLabel}</span></button>} /></div>
      ) : (
        <div className="library-groups">
          {groups.map(({ session, clips }) => {
            const picked = clips.reduce((sum, clip) => sum + (selectedIds.has(clip.id) ? 1 : 0), 0);
            const allPicked = picked === clips.length;
            return (
              <section className="library-group" key={session.id} aria-label={`Kept clips from ${session.name}`}>
                <header className="library-group-head">
                  <div className="library-group-copy">
                    <h2>{session.name}</h2>
                    <span className="t-num">{plural(clips.length, "kept clip")}{picked ? ` · ${picked} selected` : ""}</span>
                  </div>
                  <button
                    type="button"
                    className={`library-group-select ${allPicked ? "is-on" : ""}`}
                    aria-pressed={allPicked}
                    onClick={() => onToggleSession(session.id)}
                  >
                    <Check size={13} aria-hidden="true" />
                    <span>{allPicked ? "Clear this session" : "Select all in this session"}</span>
                  </button>
                </header>
                <div className="rev-grid library-grid">
                  {clips.map((clip) => {
                    const band = openingBand(hookScoreOf(clip));
                    const selected = selectedIds.has(clip.id);
                    return (
                      <article className={`rev-card library-card is-kept ${selected ? "" : "is-deselected"}`} key={clip.id}>
                        <button type="button" className="library-select" role="checkbox" aria-checked={selected} aria-label={`${selected ? "Deselect" : "Select"} ${clip.title} for export`} onClick={() => onToggleSelect(clip.id)}><Check aria-hidden="true" /></button>
                        <button type="button" className="rev-card-open" onClick={() => onEdit(session, clip)} aria-label={`Edit ${clip.title} from ${session.name}`}>
                          <span className="rev-thumb"><Poster clip={clip} className="mock-v1-real-thumb" /><span className={`rev-band ${band.className}`}>{band.label}</span><span className="rev-dur">{fmtClock(durationOf(clip))}</span><span className="rev-kept-flag"><Check aria-hidden="true" /></span></span>
                          <span className="rev-card-info"><span className="rev-card-info-col"><span className="rev-card-title">{clip.title}</span><span className="rev-card-time t-num">{clip.timestamp}</span><span className="library-card-state">{selected ? "Selected · finalizes on export" : "Not selected for export"}</span></span></span>
                        </button>
                      </article>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** Rename + re-trim a kept clip — the editing that used to live in the theater
 * player, now homed in the Clip Library. Trimming re-renders via editClip. */
function ClipEditor({
  session,
  clip,
  onClose,
  onClipUpdated,
}: {
  session: Session;
  clip: Clip;
  onClose: () => void;
  onClipUpdated: (clip: Clip) => void;
}) {
  const { addLog, copyClipsToFolder, editClip, settings, updateClipTitle } = useJobStore();
  return (
    <ClipLibraryEditor
      session={session}
      clip={clip}
      onClose={onClose}
      onSave={async ({ title, tags, start, end, fades, mediaDirty }) => {
        let nextClip = clip;
        if (title !== clip.title) {
          nextClip = await updateClipTitle(session.id, clip.id, title);
        }
        if (mediaDirty) {
          nextClip = await editClip(session.id, clip.id, {
            start_time: start,
            end_time: end,
            video_fade_in: fades.videoIn,
            video_fade_out: fades.videoOut,
            audio_fade_in: fades.audioIn,
            audio_fade_out: fades.audioOut,
          });
        }
        try {
          const key = `recall-draft-meta-${session.id}`;
          const current = JSON.parse(localStorage.getItem(key) || "{}");
          localStorage.setItem(key, JSON.stringify({ ...current, [clip.id]: { ...current[clip.id], title, tags, fades } }));
        } catch {}
        onClipUpdated(nextClip);
        addLog("Clip changes saved.", "success");
      }}
      onExport={async () => {
        const folder = await window.electronAPI?.openFolderDialog?.();
        if (!folder) return;
        const result = await copyClipsToFolder([clip.id], folder, { preset: settings.exportPreset, filenameTemplate: settings.filenameTemplate });
        if (!result.copied) throw new Error(result.error || "Export failed. Check the destination and try again.");
        addLog("Clip exported.", "success");
      }}
    />
  );
}


function InsightsView({ sessions, totalClips, totalKept, keepRate, onStart }: { sessions: Session[]; totalClips: number; totalKept: number; keepRate: number; onStart: () => void }) {
  return <InsightsStory sessions={sessions} totalClips={totalClips} totalKept={totalKept} keepRate={keepRate} onStart={onStart} />;
  /* Legacy dashboard retained below temporarily while the story view is validated. */
  const completed = sessions.filter((s) => s.status === "completed");
  const allClips = sessions.flatMap((s) => s.clips);
  const reviewedSeconds = sessions.reduce((sum, s) => sum + (s.vodDuration ?? 0), 0);
  const exported = sessions.reduce((sum, s) => sum + s.exportedClipIds.length, 0);
  const avgScore = allClips.length ? Math.round(allClips.reduce((sum, c) => sum + hookScoreOf(c), 0) / allClips.length) : 0;
  const avgPerVod = completed.length ? totalClips / completed.length : 0;
  // Runtime of every kept clip — the highlight reel Recall condensed the VODs into.
  const keptSeconds = sessions.reduce((sum, s) => sum + s.clips.filter((c) => s.savedClipIds.includes(c.id)).reduce((a, c) => a + durationOf(c), 0), 0);

  // Why clips were flagged — the detection signals behind the moments.
  const signals = new Map<string, number>();
  allClips.flatMap((c) => c.signals ?? []).forEach((sig) => signals.set(sig, (signals.get(sig) ?? 0) + 1));
  const topSignals = [...signals.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  const sigMax = Math.max(1, ...topSignals.map(([, n]) => n));

  // Moment-type mix (Elimination / Reaction / …), collapsing "A + B" to its lead.
  const moments = new Map<string, number>();
  allClips.forEach((c) => { const key = (c.momentType || c.reason || "Other").split("+")[0].trim() || "Other"; moments.set(key, (moments.get(key) ?? 0) + 1); });
  const topMoments = [...moments.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  const momMax = Math.max(1, ...topMoments.map(([, n]) => n));

  // Opening mix — descriptive bands, not overall clip grades.
  const bands = [
    { label: "Strong opening", cls: "high", count: allClips.filter((c) => hookScoreOf(c) >= 65).length },
    { label: "Good opening", cls: "mid", count: allClips.filter((c) => { const s = hookScoreOf(c); return s >= 45 && s < 65; }).length },
    { label: "Slow build", cls: "low", count: allClips.filter((c) => hookScoreOf(c) < 45).length },
  ];
  const bandMax = Math.max(1, ...bands.map((b) => b.count));
  const clipMax = Math.max(1, ...completed.map((s) => s.clips.length));

  return <div className="clips-shell"><div className="clips-head"><h1>Insights</h1><span>What Recall is finding across your real sessions.</span></div>
    <div className="ins-tiles">
      <Insight label="VODs analyzed" value={completed.length} copy={`${(reviewedSeconds / 3600).toFixed(1)} hours reviewed`} icon={Film} />
      <Insight label="Moments found" value={totalClips} copy={`~${avgPerVod.toFixed(1)} per VOD`} icon={Target} />
      <Insight label="Clips kept" value={totalKept} copy={`${keepRate}% keep rate`} icon={Check} tone="success" />
      <Insight label="Highlight reel" value={fmtClock(keptSeconds)} copy="Kept-clip runtime" icon={Scissors} />
      <Insight label="Typical opening" value={openingBand(avgScore).label} copy="Across all moments" icon={BarChart3} />
      <Insight label="Exported" value={exported} copy="Ready-to-post clips" icon={Clock} />
    </div>
    <div className="ins-panels">
      <div className="ins-panel"><span className="ins-panel-title">Why clips were flagged</span><div className="ins-bars">{topSignals.length ? topSignals.map(([label, count]) => <div className="ins-bar" key={label}><span>{label}</span><i><u style={{ width: `${count / sigMax * 100}%` }} /></i><b>{count}</b></div>) : <span className="ins-trend">Signal detail appears after the next scan.</span>}</div></div>
      <div className="ins-panel"><span className="ins-panel-title">Moment types</span><div className="ins-bars">{topMoments.length ? topMoments.map(([label, count]) => <div className="ins-bar" key={label}><span>{label}</span><i><u style={{ width: `${count / momMax * 100}%` }} /></i><b>{count}</b></div>) : <span className="ins-trend">No moments yet.</span>}</div></div>
      <div className="ins-panel"><span className="ins-panel-title">Confidence mix</span><div className="ins-bars">{allClips.length ? bands.map((b) => <div className="ins-bar" key={b.label}><span>{b.label}</span><i><u className={`mock-v1-band-${b.cls}`} style={{ width: `${b.count / bandMax * 100}%` }} /></i><b>{b.count}</b></div>) : <span className="ins-trend">No moments yet.</span>}</div></div>
      <div className="ins-panel"><span className="ins-panel-title">Clips found per session</span><div className="ins-chart"><svg viewBox="0 0 500 220" preserveAspectRatio="none"><g className="ins-cols">{completed.slice(0, 8).reverse().map((session, index) => { const height = Math.max(10, session.clips.length / clipMax * 180); return <rect key={session.id} className={index === Math.min(7, completed.length - 1) ? "is-latest" : ""} x={25 + index * 58} y={205 - height} width="34" height={height} rx="5" />; })}</g></svg><div className="ins-chart-axis"><span>Older</span><span>Latest</span></div></div></div>
    </div>
  </div>;
}

function Insight({ label, value, copy, icon: Icon, tone }: { label: string; value: number | string; copy: string; icon?: typeof Film; tone?: string }) {
  return <div className="ins-tile mock-v1-ins-tile"><div className="mock-v1-ins-tile-head"><small>{label}</small>{Icon && <Icon size={14} className={`mock-v1-ins-icon tone-${tone ?? "accent"}`} />}</div><b>{value}</b><span className="ins-trend">{copy}</span></div>;
}

const PROCESSING_CHOICES: ChoiceOption<"fast" | "quality">[] = [
  { value: "fast", label: "Smart scan", description: "Focuses deep analysis on the moments most likely to matter.", badge: "Faster" },
  { value: "quality", label: "Best quality", description: "Scans the full recording closely and takes longer.", badge: "Thorough" },
];
const PERFORMANCE_CHOICES: ChoiceOption<"background" | "balanced" | "max">[] = [
  { value: "background", label: "Keep PC responsive", description: "Quieter resource use while you work, play, or stream.", badge: "Quietest" },
  { value: "balanced", label: "Balanced", description: "Good scan speed without taking over the machine.", badge: "Recommended" },
  { value: "max", label: "Full speed", description: "Finishes sooner with heavier CPU and GPU use.", badge: "Fastest" },
];

function SettingsModal({ open, tab, setTab, onClose, onNavigate }: { open: boolean; tab: SettingsTab; setTab: (tab: SettingsTab) => void; onClose: () => void; onNavigate: (view: View) => void }) {
  const { settings, setSettings } = useJobStore();
  const [saved, setSaved] = useState(false);
  const savedTimer = useRef<number | null>(null);
  const dialogRef = useModalFocus<HTMLDivElement>(open, onClose);
  const tabOrder: SettingsTab[] = ["general", "appearance", "project", "export", "shortcuts"];
  const onTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, current: SettingsTab) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const currentIndex = tabOrder.indexOf(current);
    const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? tabOrder.length - 1 : event.key === "ArrowDown" ? (currentIndex + 1) % tabOrder.length : (currentIndex - 1 + tabOrder.length) % tabOrder.length;
    const next = tabOrder[nextIndex];
    setTab(next);
    window.requestAnimationFrame(() => document.getElementById(`settings-tab-${next}`)?.focus());
  };

  // Mirror the pre-mount localStorage fallbacks the store reads on boot, so a
  // value survives even before the backend settings round-trip returns.
  const persistLocal = (key: string, value: string) => { try { localStorage.setItem(key, value); } catch {} };
  const markSaved = () => {
    setSaved(true);
    if (savedTimer.current) window.clearTimeout(savedTimer.current);
    savedTimer.current = window.setTimeout(() => setSaved(false), 1600);
  };
  const commitSettings = (patch: Partial<typeof settings>) => { setSettings(patch); markSaved(); };
  const caption = settings.captionStyle ?? DEFAULT_CAPTION_STYLE;
  const patchCaption = (patch: Partial<CaptionStyle>) => {
    const next = { ...caption, ...patch };
    setSettings({ captionStyle: next });
    persistLocal("recall-caption-style", JSON.stringify(next));
    markSaved();
  };

  return (
    <div id="settings-modal" className={open ? "open" : ""} onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div ref={dialogRef} className="settings-dialog elev-2" role="dialog" aria-modal="true" aria-label="Recall settings" tabIndex={-1}>
        <div className="settings-body">
          <div className="settings-tabs" role="tablist" aria-label="Settings sections" aria-orientation="vertical">
            <button id="settings-tab-general" role="tab" aria-selected={tab === "general"} aria-controls="settings-panel-general" className={`settings-tab ${tab === "general" ? "active" : ""}`} onKeyDown={(event) => onTabKeyDown(event, "general")} onClick={() => setTab("general")}>General</button>
            <button id="settings-tab-appearance" role="tab" aria-selected={tab === "appearance"} aria-controls="settings-panel-appearance" className={`settings-tab ${tab === "appearance" ? "active" : ""}`} onKeyDown={(event) => onTabKeyDown(event, "appearance")} onClick={() => setTab("appearance")}>Appearance</button>
            <div className="settings-tab-section-header">Projects</div>
            <button id="settings-tab-project" role="tab" aria-selected={tab === "project"} aria-controls="settings-panel-project" className={`settings-tab ${tab === "project" ? "active" : ""}`} onKeyDown={(event) => onTabKeyDown(event, "project")} onClick={() => setTab("project")}>Highlight Sessions</button>
            <button id="settings-tab-export" role="tab" aria-selected={tab === "export"} aria-controls="settings-panel-export" className={`settings-tab ${tab === "export" ? "active" : ""}`} onKeyDown={(event) => onTabKeyDown(event, "export")} onClick={() => setTab("export")}>Export &amp; Captions</button>
            <div className="mock-v1-settings-spacer" />
            <button id="settings-tab-shortcuts" role="tab" aria-selected={tab === "shortcuts"} aria-controls="settings-panel-shortcuts" className={`settings-tab ${tab === "shortcuts" ? "active" : ""}`} onKeyDown={(event) => onTabKeyDown(event, "shortcuts")} onClick={() => setTab("shortcuts")}>Shortcuts</button>
          </div>
          <div id={`settings-panel-${tab}`} className="settings-content-viewport" role="tabpanel" aria-labelledby={`settings-tab-${tab}`}>
            <div className="settings-content-toolbar">
              <div className={`settings-saved-state ${saved ? "is-visible" : ""}`} role="status" aria-live="polite">{saved && <><StateIcon tone="success" icon={<Check size={13} />} /><span>Saved</span></>}</div>
              <button className="settings-close-btn" aria-label="Close settings" title="Close settings" onClick={onClose}><X aria-hidden="true" /></button>
            </div>
            <div className="settings-content-inner">

            {tab === "general" && (
              <SettingsPanel title="General Preferences" copy="Your local creator identity, this PC, and what Recall has learned about your taste.">
                {/*
                  Identity first, hardware second. HardwareSettingsPanel is a
                  tall async block, and sitting above this field it pushed the
                  only control for the Start-screen greeting below the fold —
                  reachable by scrolling, but invisible on open, which reads as
                  "the name is hard-coded and there is no way to change it".
                  The panel's own copy already promises identity first.
                */}
                <label className="form-group">
                  <span className="form-label">Display name</span>
                  <input className="input-text" value={settings.displayName} placeholder="Your name" maxLength={32} onChange={(event) => { commitSettings({ displayName: event.target.value }); persistLocal("recall-display-name", event.target.value); }} />
                  <span className="form-description">Used for the “Welcome back” greeting on the Start screen. Leave it empty to be greeted as “Creator”.</span>
                </label>
                <SettingRow title="Keep this PC awake while working" copy="Prevents system sleep during scans and exports, while still allowing the display to turn off. Recall releases it as soon as the work finishes.">
                  <StudioSwitch checked={settings.keepAwakeWhileWorking} onChange={(keepAwakeWhileWorking) => commitSettings({ keepAwakeWhileWorking })} label="Keep this PC awake while scans and exports are active" />
                </SettingRow>
                <SettingRow title="Completion notifications" copy="Show a native alert when a scan, overnight batch, clip export, or highlight reel finishes or needs attention.">
                  <StudioSwitch checked={settings.desktopNotifications} onChange={(desktopNotifications) => commitSettings({ desktopNotifications })} label="Show completion and failure notifications" />
                </SettingRow>
                <HardwareSettingsPanel active={open && tab === "general"} apiEndpoint={settings.apiEndpoint} />
                <LearningPanel />
              </SettingsPanel>
            )}

            {tab === "project" && (
              <SettingsPanel title="Highlight Sessions" copy="Recall adapts sensitivity, clip count, and clip length to the evidence in each recording. Choose how deeply it scans and how much of this PC it may use. Speech analysis is English-first today — non-English VODs still use voice, face, chat, and game signals.">
                <SettingRow title="Scan speed" copy="Smart scan looks closely at the moments that matter. Best quality scans the whole VOD and takes longer.">
                  <ChoiceCards value={settings.processingMode} options={PROCESSING_CHOICES} onChange={(processingMode) => commitSettings({ processingMode })} label="Scan speed" />
                </SettingRow>
                <SettingRow title="While scanning" copy="How much of this PC a scan may use. Keep PC responsive is for scanning while you stream or play; Full speed is for when you step away.">
                  <ChoiceCards value={settings.performanceProfile} options={PERFORMANCE_CHOICES} onChange={(performanceProfile) => commitSettings({ performanceProfile })} label="While scanning" />
                </SettingRow>
                <StoragePanel commitSettings={commitSettings} onManageClips={() => onNavigate("clips")} />
              </SettingsPanel>
            )}

            {tab === "export" && (
              <SettingsPanel title="Export & Captions" copy="The default shape of generated clips and their burned-in captions.">
                <SettingRow title="Facecam layout" copy="Automatic keeps one composition across the VOD: stacked for context-heavy games, PiP for action-forward gameplay. You can still lock either style.">
                  <ExportLayoutChoices value={settings.exportLayout} onChange={(exportLayout) => { commitSettings({ exportLayout }); persistLocal("recall-export-layout", exportLayout); }} />
                </SettingRow>
                <CaptionEditor caption={caption} onChange={patchCaption} />
              </SettingsPanel>
            )}

            {tab === "appearance" && (
              <SettingsPanel title="Appearance" copy="Tune the studio for your editing space. Your accent carries through actions, selections, focus, and Recall's reaction curve.">
                <SettingRow title="Theme" copy="Dark is tuned for dim editing rooms; light keeps the same compact hierarchy for brighter spaces.">
                  <ThemeChoices value={settings.theme} onChange={(theme) => commitSettings({ theme })} />
                </SettingRow>
                <SettingRow title="Studio color" copy="A warm, low-chroma accent for the parts of Recall that help you act and navigate. Status colors stay consistent.">
                  <AccentChoices value={settings.accentTheme} onChange={(accentTheme) => commitSettings({ accentTheme })} />
                </SettingRow>
              </SettingsPanel>
            )}

            {tab === "shortcuts" && (
              <SettingsPanel title="Keyboard Shortcuts" copy="Rapid controls for theater review.">
                <RememberHotkeySetting />
                <Shortcut label="Keep / approve clip" value="Spacebar" />
                <Shortcut label="Pass clip" value="X or Backspace" />
                <Shortcut label="Next highlight" value="Right Arrow" />
                <Shortcut label="Previous highlight" value="Left Arrow" />
                <Shortcut label="Search" value="Ctrl+K" />
                <Shortcut label="Back to review" value="Esc" />
                <DiagnosticsPanel />
              </SettingsPanel>
            )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SettingsPanel({ title, copy, children }: { title: string; copy: string; children: React.ReactNode }) { return <div className="settings-panel active"><div className="mock-v1-settings-heading"><h2>{title}</h2><p>{copy}</p></div>{children}</div>; }
function SettingRow({ title, copy, children }: { title: string; copy: string; children: React.ReactNode }) { return <div className="mock-v1-setting-row"><div><h4>{title}</h4><p>{copy}</p></div>{children}</div>; }
function Shortcut({ label, value }: { label: string; value: string }) { return <div className="mock-v1-shortcut"><span>{label}</span><span className="kb-shortcut">{value}</span></div>; }

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
function StoragePanel({ commitSettings, onManageClips }: { commitSettings: (patch: Partial<ReturnType<typeof useJobStore.getState>["settings"]>) => void; onManageClips: () => void }) {
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
function LearningPanel() {
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

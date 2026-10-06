// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { type CSSProperties, FormEvent, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Film,
  HardDrive,
  Play,
  RefreshCw,
  Scissors,
  Trash2,
  Video,
  X,
  ArrowLeft,
  Sparkles,
} from "./lib/icons";
import { ModalityMix } from "./components/ReactionEvidence";
import { apiFetch } from "./lib/api";

// The in-app editor (vendored OpenCut) is heavy; it loads the first time a clip is opened in it.
import { fmtClock, fmtDurationHuman } from "./lib/format";
import { clueLabel, plural } from "./lib/copy";
import ProblemBar from "./components/ProblemBar";
import { releaseClipPreview } from "./lib/releaseClipPreview";
import CuttingRoom, { type CuttingRoomMemoryHandoff } from "./cutting/CuttingRoomEditor";
import { OnboardingFlow } from "./onboarding/OnboardingFlow";
import StudioEmptyState from "./components/StudioEmptyState";
import { ShareHandoffDialog } from "./components/ShareHandoff";
import LiveSessionTestPanel from "./components/LiveSessionTestPanel";
import RecallOriginBadge from "./components/RecallOriginBadge";
import StreamMemory, { STARTER_QUERIES } from "./components/StreamMemory";
import CompilationProjects from "./components/CompilationProjects";
import { MediaVolumeControl, StudioButton, useMediaVolume } from "./components/StudioControls";
import { hasCompletedOnboarding } from "./lib/onboarding";
import { isInteractiveKeyboardTarget } from "./lib/keyboard";
import type { NewStreamKind, View } from "./shell/views";
import { Poster, pickSessionPosterClip } from "./components/Posters";
import { Sidebar } from "./shell/Sidebar";
import { WindowChrome } from "./shell/WindowChrome";
import { CommandPalette } from "./shell/CommandPalette";
import type { PaletteCommand, PaletteSession } from "./shell/paletteItems";
import { shellCounts } from "./shell/shellModel";
import { queueLane } from "./scans/scanModel";
import { NewScanView } from "./scans/NewScanView";
import { InsightsView } from "./insights/InsightsView";
import { ReelView } from "./export/ReelView";
import { ExportTabs } from "./export/ExportTabs";
import { queueWaitSeconds, useScanRates } from "./scans/estimate";
import { LiveScanView } from "./scans/LiveScanView";
import { SessionDeckView, type ReviewFilter } from "./review/SessionDeckView";
import { titleSuggestsVtuberMode, type BatchRow, type ImportMode, type ProbeMeta } from "./scans/importModel";
import { ScansView } from "./scans/ScansView";
import { HomeView } from "./home/HomeView";
import { LibraryView } from "./library/LibraryView";
import { ReviewIndex } from "./review/ReviewIndex";
import { ExportView } from "./export/ExportView";
import { SettingsPage, type SettingsSection } from "./settings/SettingsPage";
import { HypeMeter, PostingCoach, postFirstId } from "./review/PostingCoach";
import { waitingSessions, type ChecklistStep } from "./home/homeModel";
import { useRecallLive } from "./lib/useRecallLive";
import { PetLayer, prefersReducedMotion } from "./pets/PetLayer";
import { useCrew } from "./pets/petStore";
import { PET_UNLOCK_2 } from "./pets/petArt";
import { useAppMoments } from "./moments/useAppMoments";
import { Celebration } from "./moments/Celebration";
import { WeeklyRecap } from "./moments/WeeklyRecap";
import { durationOf, hookScoreOf, openingBand, primaryReviewDeckClips, secondLookClips, sortReviewClips, type ReviewSort } from "./lib/reviewDeck";
import { applyAccent, applyCaptionColor } from "./theme/accent";
import { AmbientGlow, GlassDefs, useGlassHighlight } from "./theme/ambient";
import { buildScanRetryDraft } from "./lib/scanRecovery";
import { batchNotification, exportNotification, hasActiveDesktopWork, isActiveScanStatus, isTerminalScanStatus, scanNotification } from "./lib/desktopWork";
import { creatorExportFilename } from "./lib/exportName";
import {
  disposeAllJobSubscriptions,
  fetchReactionTimeline,
  type Clip,
  type ExportOperation,
  type Job,
  type JobStatus,
  type ReactionTimeline,
  type Session,
  useJobStore,
} from "./lib/store";
import {
  parseTwitchVodUrls,
  resolveBatchSessionName,
  sessionNameFromProbe,
} from "./lib/twitchBatch";


export { titleSuggestsVtuberMode };

const terminal = new Set(["completed", "failed", "cancelled"]);

type RevealRect = { left: number; top: number; width: number; height: number };
type ScanRevealSnapshot = { jobId: string; cards: Array<{ clipId: string; rect: RevealRect }> };


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
  // What the Cutting Room has open. Only an explicit choice sets it (Review's
  // "Open Cutting Room", a Memory moment, Open video, Load session); opening
  // the room from the sidebar never guesses the most recent stream.
  const [cuttingSessionId, setCuttingSessionId] = useState<string>();
  const [cuttingTimeline, setCuttingTimeline] = useState<ReactionTimeline | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Session | null>(null);
  const [bulkDeleteTargets, setBulkDeleteTargets] = useState<Session[]>([]);
  // A clip opened for editing in the Cutting Room (Export, Review).
  const [cuttingClip, setCuttingClip] = useState<{ clip: Clip; nonce: number } | null>(null);
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
  const [settingsSection, setSettingsSection] = useState<SettingsSection>("you");
  const [settingsNonce, setSettingsNonce] = useState(0);
  const openSettings = (section: SettingsSection = "you") => { setSettingsSection(section); setSettingsNonce((n) => n + 1); setSettingsOpen(true); };
  const [onboardingOpen, setOnboardingOpen] = useState(() => !hasCompletedOnboarding());
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
  const closeDeleteModal = useCallback(() => setDeleteTarget(null), []);
  const closeBulkDeleteModal = useCallback(() => setBulkDeleteTargets([]), []);
  const closeShareHandoff = useCallback(() => setShareHandoff(null), []);
  // Onboarding is not Esc-dismissable: the system check step sets the
  // processing route, and skipping is an explicit control on the later steps.
  const keepOnboardingOpen = useCallback(() => {}, []);
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
  // This PC's measured scan speed, for "how long will this take" before a scan starts.
  const scanRates = useScanRates(
    settings.apiEndpoint,
    settings.processingMode,
    true,
    sessions.filter((session) => session.status === "completed").length,
  );

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

  const [paletteOpen, setPaletteOpen] = useState(false);
  // Review from the sidebar or Ctrl K shows every deck; opening a session shows its deck.
  const [reviewScope, setReviewScope] = useState<"index" | "deck">("index");
  const learningMessage = useJobStore((state) => state.learningStatus?.creator_message);
  const [memoryHandoff, setMemoryHandoff] = useState<{ text: string; nonce: number }>();
  const shellCountsNow = useMemo(() => shellCounts(sessions, jobs), [sessions, jobs]);
  const recallLive = useRecallLive();
  const crewShow = useCrew((state) => state.show);
  const crewRoam = useCrew((state) => state.roam);
  const crewSize = useCrew((state) => state.crew.length);
  // Pets stay home wherever the creator is watching footage closely.
  const petsStayHome = settingsOpen || onboardingOpen || ["review", "theater", "cutting"].includes(view);
  // With reduced motion the pets never leave the den, so it mustn't say they did.
  const petsRoaming = crewRoam && !!shellCountsNow.scan && !petsStayHome && !prefersReducedMotion();
  const sessionsLoaded = useJobStore((state) => state.sessionsLoaded);
  useAppMoments(sessions, jobs, sessionsLoaded);
  const homeScan = useMemo(() => {
    const running = jobs.find((job) => ["analyzing", "detecting", "assembling"].includes(job.status));
    if (!running || !shellCountsNow.scan) return undefined;
    return {
      title: shellCountsNow.scan.title,
      progress: shellCountsNow.scan.progress,
      stage: running.stage?.label,
      etaSeconds: running.etaSeconds,
      found: running.clips ?? [],
    };
  }, [jobs, shellCountsNow.scan]);

  useGlassHighlight();
  // The footage glow follows what's on screen: the open session, else the newest one.
  const ambientSession = activeSession ?? sortedSessions[0];
  const ambientImage = ambientSession ? pickSessionPosterClip(ambientSession)?.thumbUrl : undefined;

  useEffect(() => {
    const theme = settings.theme || "dark";
    document.documentElement.dataset.theme = theme;
    applyAccent(settings.accent);
    applyCaptionColor(settings.captionColor, settings.accent);
    document.body.className = `view-${view}-active`;
    try { localStorage.setItem("recall-theme", theme); } catch {}
    window.electronAPI?.setTitleBarOverlay?.(theme === "dark"
      ? { color: "#000000", symbolColor: "#FFFFFF", height: 48 }
      : { color: "#F4F4F5", symbolColor: "#0A0A0B", height: 48 });
  }, [view, settings.theme, settings.accent, settings.captionColor]);

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
    // Settings is a page inside the app now, so it no longer makes the app inert.
    const modalOpen = !!deleteTarget || bulkDeleteTargets.length > 0 || onboardingOpen;
    app.inert = modalOpen;
    if (modalOpen) app.setAttribute("aria-hidden", "true");
    else app.removeAttribute("aria-hidden");
    return () => {
      app.inert = false;
      app.removeAttribute("aria-hidden");
    };
  }, [deleteTarget, bulkDeleteTargets, onboardingOpen]);

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
    if (!activeSession || !["review", "theater"].includes(view)) return;
    let cancelled = false;
    fetchReactionTimeline(activeSession.id).then((value) => {
      if (!cancelled) setTimeline(value);
    });
    return () => { cancelled = true; };
  // Status too: a stream opened in the Cutting Room mid-scan gets its curve
  // the moment the scan finishes.
  }, [activeSession?.id, activeSession?.status, view]);

  const cuttingSession = useMemo(
    () => (cuttingSessionId ? sessions.find((session) => session.id === cuttingSessionId) : undefined),
    [sessions, cuttingSessionId],
  );
  // A deleted session leaves the room empty rather than pointing at nothing.
  useEffect(() => {
    if (cuttingSessionId && !cuttingSession && sessions.length) setCuttingSessionId(undefined);
  }, [cuttingSessionId, cuttingSession, sessions.length]);
  useEffect(() => {
    setCuttingTimeline(null);
    if (!cuttingSession || view !== "cutting") return;
    let cancelled = false;
    fetchReactionTimeline(cuttingSession.id).then((value) => { if (!cancelled) setCuttingTimeline(value); });
    return () => { cancelled = true; };
  }, [cuttingSession?.id, cuttingSession?.status, view]);

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
        if (document.querySelector(".oce")) return; // the editor owns the keyboard while open
        event.preventDefault();
        setPaletteOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const openSession = async (session: Session) => {
    const requestId = ++openSessionRequestRef.current;
    setReviewScope("deck");
    const startingView = viewRef.current;
   
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
    setCuttingSessionId(result.job_id);
    setCuttingClip(null);
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

  const openProject = (kind: NewStreamKind) => {
    // Detach from the previous session so a finished currentJob can't own the
    // new import flow (selection otherwise keeps pointing at the last review).
    setSelectedSessionId(undefined);
   
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
    // Straight to New stream: the name field lives on the page now, and an
    // empty name falls back to the VOD's own title.
    setImportMode("single");
    setSourceKind(kind);
    setProjectName("");
    setSource("");
    setTwitchUrl("");
    setProbeMeta(null);
    setProbeError(false);
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

  // A Twitch link that won't download: bring in a downloaded copy under the
  // same stream name, ready for Start scan.
  const useDownloadedFile = async (name?: string) => {
    const path = await window.electronAPI?.openFileDialog?.();
    if (!path) {
      if (!window.electronAPI) addLog("Choose recordings from the Recall desktop app.", "warn");
      return;
    }
    setSelectedSessionId(undefined);
    setImportMode("single");
    setSourceKind("local");
    setProjectName(name ?? "");
    setSource(path);
    setView("add-vod");
  };

  const fetchTwitch = (url?: string) => {
    const value = (url ?? twitchUrl).trim();
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
    // No name typed: use the VOD's own title rather than its URL or file name.
    const name = projectName || (sourceKind === "twitch" ? sessionNameFromProbe(probeMeta, source) : probeMeta.title || undefined);
    startJob(source, name || undefined, {
      vtuberMode: vtuberConfirmed ? "confirmed" : "off",
      sourceDate: probeMeta.source_date || undefined,
      sourceType: sourceKind === "twitch" ? "twitch" : "file",
    });
    setVtuberConfirmed(false);
    setView("processing");
  };

  // Cutting Room: open any local video (a recording, a VOD) to edit it right
  // away. It isn't scanned; "Scan for moments" in the editor does that later
  // if the creator wants moments and Auto framing.
  const openVideoInCuttingRoom = async () => {
    const path = await window.electronAPI?.openFileDialog?.();
    if (!path) {
      if (!window.electronAPI) addLog("Open videos from the Recall desktop app.", "warn");
      return;
    }
    const jobId = await useJobStore.getState().openLocalVideo(path);
    if (jobId) {
      setMemoryEditorHandoff(null);
      setCuttingClip(null);
      setCuttingSessionId(jobId);
      setView("cutting");
    }
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

  const goTo = (next: View) => {
   
    setSettingsOpen(false);
    if (next === "review") setReviewScope("index");
    setView(next);
  };
  const openDeck = (session: Session) => { setReviewScope("deck"); void openSession(session); };
  const startDeck = (session: Session) => {
    const first = primaryReviewDeckClips(session.clips).find((clip) => !clip.kept && !clip.passed && !session.savedClipIds.includes(clip.id));
    if (!first) { openDeck(session); return; }
    setReviewScope("deck");
   
    setSettingsOpen(false);
    setSelectedSessionId(session.id);
    setCurrentSessionId(session.id);
    openTheater(first, "review");
  };
  const watchClip = (session: Session, clip: Clip) => {
   
    setSettingsOpen(false);
    setSelectedSessionId(session.id);
    setCurrentSessionId(session.id);
    openTheater(clip, "home");
  };
  const runChecklistStep = (step: ChecklistStep) => {
    if (step === "scan") openProject("local");
    else if (step === "review") { const next = waitingSessions(sortedSessions)[0]; if (next) void openSession(next); else goTo("review"); }
    else if (step === "export") goTo("clips");
    else if (step === "cut") goTo("cutting");
    else if (step === "live") goTo("live");
    else openSettings("look");
  };
  // A VOD link pasted on Home or into Ctrl K goes straight to the new-scan
  // screen, already checking the VOD. No "name your project" detour: the
  // stream is named from the VOD's own title. (It used to land with the link
  // only typed into the Twitch box, never fetched, so nothing happened.)
  const scanLinkFromPalette = (url: string) => {
    setSelectedSessionId(undefined);
   
    setImportMode("single");
    setSourceKind("twitch");
    setProjectName("");
    setTwitchUrl(url);
    setProbeMeta(null);
    setProbeError(false);
    setSource(url);
    setView("add-vod");
  };
  const paletteSessions: PaletteSession[] = sortedSessions.map((session) => ({
    id: session.id,
    title: session.name,
    sub: [shortSessionDate(session.createdAt || session.updatedAt), session.vodDuration ? fmtDurationHuman(session.vodDuration) : ""].filter(Boolean).join(" · "),
    image: pickSessionPosterClip(session)?.thumbUrl,
    open: () => { setSettingsOpen(false); void openSession(session); },
  }));
  const waitingSession = sortedSessions.find((session) => session.status === "completed" && primaryReviewDeckClips(session.clips).some((clip) => !clip.kept && !clip.passed && !session.savedClipIds.includes(clip.id)));
  const paletteCommands: PaletteCommand[] = [
    ...(waitingSession ? [{ id: "review", title: "Review what's waiting", icon: "review" as const, keywords: "keep pass theater", run: () => { setSettingsOpen(false); void openSession(waitingSession); } }] : []),
    { id: "import", title: "Import a recording", icon: "import", keywords: "new stream file obs", run: () => openProject("local") },
    { id: "twitch", title: "Scan a Twitch VOD", icon: "twitch", keywords: "new stream link", run: () => openProject("twitch") },
    { id: "batch", title: "Queue tonight's VODs", icon: "batch", keywords: "batch several", run: () => openProject("batch") },
    { id: "live", title: "Start Recall Live", icon: "live", keywords: "mark moment hotkey", run: () => goTo("live") },
    { id: "export", title: "Export kept clips", icon: "export", keywords: "clips library", run: () => goTo("clips") },
    { id: "cut", title: "Cut a missed moment", icon: "scissors", keywords: "cutting room editor vod", run: () => goTo("cutting") },
    { id: "compilations", title: "Compilations", icon: "layers", keywords: "montage", run: () => goTo("compilations") },
    { id: "reel", title: "Make a highlight reel", icon: "reel", keywords: "recap", run: () => goTo("reel") },
    { id: "memory", title: "Stream Memory", icon: "memory", keywords: "search remember", run: () => goTo("memory") },
    { id: "library", title: "Library", icon: "library", keywords: "sessions streams", run: () => goTo("sessions") },
    { id: "scans", title: "Scans", icon: "scans", keywords: "queue progress", run: () => goTo("queue") },
    { id: "insights", title: "Insights", icon: "insights", keywords: "stats week", run: () => goTo("insights") },
    { id: "settings", title: "Open settings", icon: "settings", keywords: "preferences", run: () => openSettings() },
    { id: "shortcuts", title: "Keyboard shortcuts", icon: "keyboard", keywords: "keys hotkeys", run: () => openSettings("shortcuts") },
    { id: "home", title: "Home", icon: "home", run: () => goTo("home") },
  ];

  // Every clip edit happens in the Cutting Room, on the clip's own piece of
  // the stream: its title, trim, framing, fades and captions in one place.
  const openClipInCuttingRoom = (sessionId: string, clip: Clip, returnTo: View) => {
    setMemoryEditorHandoff(null);
    setCuttingSessionId(sessionId);
    setCuttingClip({ clip, nonce: Date.now() });
    setCuttingReturn(returnTo);
    setView("cutting");
  };

  return (
    <>
      <GlassDefs />
      <AmbientGlow image={ambientImage} />
      <WindowChrome />
      <PetLayer keepers={shellCountsNow.keepers} scanning={!!shellCountsNow.scan} focus={petsStayHome} hidden={view === "theater"} />
      <Celebration onOpenCrew={() => openSettings("crew")} />
      <WeeklyRecap sessions={sortedSessions} onExport={() => goTo("clips")} />
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        sessions={paletteSessions}
        commands={paletteCommands}
        examples={STARTER_QUERIES}
        scanLink={scanLinkFromPalette}
        searchMemory={(text) => { setMemoryHandoff({ text, nonce: Date.now() }); setSettingsOpen(false); setView("memory"); }}
      />
      <div id="app-container">
        <Sidebar
          view={view}
          counts={shellCountsNow}
          displayName={settings.displayName}
          settingsOpen={settingsOpen}
          onView={goTo}
          onNew={openProject}
          onSearch={() => setPaletteOpen(true)}
          onSettings={() => openSettings()}
          liveActive={recallLive.active}
          den={crewShow ? { roaming: petsRoaming, slotOpen: shellCountsNow.keepers >= PET_UNLOCK_2 && crewSize < 2, onOpenCrew: () => openSettings("crew") } : undefined}
        />

        <main>
          {/* In the flow under the header, not floating over the content: a
              failure should wait to be read, not slide past the corner of the
              screen. Renders nothing when there is nothing wrong, so it costs
              no space in the normal case. */}
          <ProblemBar />
          {settingsOpen && (
            <div className="view-panel active settings-view" id="view-settings">
              <SettingsPage section={settingsSection} sectionNonce={settingsNonce} onNavigate={goTo} />
            </div>
          )}


          <div className={`view-panel ${view === "home" ? "active" : ""}`} id="view-home">
            <HomeView
              displayName={displayName}
              onRename={saveDisplayName}
              sessions={sortedSessions}
              scan={homeScan}
              queued={shellCountsNow.queued}
              live={recallLive}
              accentChosen={settings.accent !== null}
              onNew={openProject}
              onScanLink={scanLinkFromPalette}
              onOpenSession={(session) => { void openSession(session); }}
              onWatch={watchClip}
              onView={goTo}
              onChecklist={runChecklistStep}
            />
          </div>

          <div className={`view-panel ${view === "live" ? "active" : ""}`} id="view-live">
            {view === "live" && <RecallLiveView onScanRecording={beginRecallSessionScan} sessions={sessions} />}
          </div>

          <div className={`view-panel ${view === "sessions" ? "active" : ""}`} id="view-sessions">
            <LibraryView sessions={sortedSessions} onOpen={(session) => { void openSession(session); }} onDelete={setDeleteTarget} onDeleteMany={setBulkDeleteTargets} onNew={openProject} />
          </div>

          <div className={`view-panel ${view === "memory" ? "active" : ""}`} id="view-memory">
            {view === "memory" && (
              <StreamMemory
                apiEndpoint={settings.apiEndpoint}
                onOpenClip={openMemoryClip}
                onOpenMoment={openMemoryMoment}
                onOpenCompilations={() => setView("compilations")}
                initialQuery={memoryHandoff}
              />
            )}
          </div>

          <div className={`view-panel ${view === "compilations" ? "active" : ""}`} id="view-compilations">
            {view === "compilations" && (
              <div className="page export comp2">
              <ExportTabs active="compilations" clipsCount={keptClips.length} onView={goTo} />
              <CompilationProjects
                apiEndpoint={settings.apiEndpoint}
                compiling={compilingReel}
                onCompile={compileCompilationProject}
                onOpenClip={(jobId, clipId) => openMemoryClip(jobId, clipId, "compilations")}
              />
              </div>
            )}
          </div>

          <div className={`view-panel ${view === "add-vod" ? "active" : ""}`} id="view-add-vod">
            <NewScanView
              importMode={importMode}
              onImportMode={setImportModeSafe}
              source={source}
              sourceKind={sourceKind}
              twitchUrl={twitchUrl}
              setTwitchUrl={setTwitchUrl}
              projectName={projectName}
              setProjectName={setProjectName}
              processingMode={settings.processingMode}
              scansAhead={sessions.filter((session) => ["queued", "analyzing", "detecting", "assembling"].includes(session.status)).length}
              rates={scanRates}
              queueWait={queueWaitSeconds(scanRates, jobs, sessions)}
              onLive={() => goTo("live")}
              onScanSettings={() => openSettings("scanning")}
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
            <LiveScanView
              job={processingJob}
              session={sessions.find((session) => session.id === processingJob?.id)}
              cancelRequested={cancelRequested}
              onCancel={cancelScan}
              onRetry={() => retryScan(processingJob)}
              onUseFile={() => void useDownloadedFile(sessions.find((session) => session.id === processingJob?.id)?.name)}
              onBack={() => setView("queue")}
              onReview={openDeck}
              rates={scanRates}
            />
          </div>

          <div className={`view-panel ${view === "queue" ? "active" : ""}`} id="view-queue">
            <ScansView
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
              rates={scanRates}
              onDone={() => setView("sessions")}
              onNew={openProject}
              onClearQueue={() => {
                clearBatchJobIds();
                addLog("Scan queue cleared. Your sessions are still in Sessions.", "info");
              }}
            />
          </div>

          <div className={`view-panel ${view === "review" ? "active" : ""}`} id="view-review">
            {reviewScope === "index" ? (
              <ReviewIndex
                sessions={sortedSessions}
                learningMessage={learningMessage}
                onStart={startDeck}
                onOpenDeck={openDeck}
                onWatch={watchClip}
                onDelete={setDeleteTarget}
                onLearning={() => openSettings("you")}
                onLibrary={() => goTo("sessions")}
                onNew={() => openProject("local")}
              />
            ) : (
            <SessionDeckView
              session={visibleActiveSession}
              timeline={timeline}
              filter={reviewFilter}
              setFilter={setReviewFilter}
              sort={reviewSort}
              setSort={setReviewSort}
              onBack={() => { setReviewScope("index"); setView("review"); }}
              onOpen={openTheater}
              onKeep={toggleKeep}
              onCut={cutClip}
              onMaybe={toggleMaybe}
              onClips={() => setView("clips")}
              onCuttingRoom={() => {
                setMemoryEditorHandoff(null);
                setCuttingClip(null);
                setCuttingSessionId(activeSession?.id);
                setCuttingReturn("review");
                setView("cutting");
              }}
              moreCandidateCount={Math.max(
                extraActiveClips.length,
                activeSession ? moreCandidateStatus[activeSession.id]?.available ?? 0 : 0,
              )}
              moreCandidatesEnabled={secondLookEnabled}
              moreCandidatesLoading={moreCandidatesBusy === activeSession?.id}
              onToggleMoreCandidates={toggleSecondLook}
            />
            )}
          </div>

          <div className={`view-panel ${view === "cutting" ? "active" : ""}`} id="view-cutting">
            {/* The Cutting Room: the OpenCut editor on the session's full VOD,
                streamed from the engine. Only mounted while active so it never
                reads the source (or plays audio) from a hidden panel. */}
            {view === "cutting" && (
              <CuttingRoom
                session={cuttingSession}
                sessions={sortedSessions}
                onLoadSession={(session) => {
                  setMemoryEditorHandoff(null);
                  setCuttingClip(null);
                  setCuttingSessionId(session.id);
                }}
                timeline={cuttingTimeline}
                onBack={() => setView(cuttingReturn)}
                onOpenVideo={() => void openVideoInCuttingRoom()}
                onOpenedVideo={(jobId) => { setMemoryEditorHandoff(null); setCuttingClip(null); setCuttingSessionId(jobId); }}
                onScan={(sessionId) => void useJobStore.getState().scanOpenedVideo(sessionId)}
                initialMoment={memoryEditorHandoff}
                openClip={cuttingClip}
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
                onOpenEditor={(clip) => { if (activeSession) openClipInCuttingRoom(activeSession.id, clip, "theater"); }}
              />
            )}
          </div>

          <div className={`view-panel ${view === "clips" ? "active" : ""}`} id="view-clips">
            {view === "clips" && (
              <ExportView
                items={keptClips}
                selectedIds={selectedExportIds}
                exporting={exporting}
                layout={settings.exportLayout}
                captionsOn={settings.captionStyle?.enabled ?? true}
                onToggleSelect={toggleExportSelected}
                onToggleSession={toggleSessionExport}
                onClearSelection={clearExportSelection}
                onExportSelected={exportSelectedClips}
                onExportIds={(ids) => { void exportClipIds(ids); }}
                onEdit={(session, clip) => openClipInCuttingRoom(session.id, clip, "clips")}
                onView={goTo}
                onChangeFormat={() => openSettings("clips")}
              />
            )}
          </div>

          <div className={`view-panel ${view === "reel" ? "active" : ""}`} id="view-reel">
            <ReelView
              sessions={sortedSessions}
              session={reelSession}
              clips={reelKeptClips}
              compiling={compilingReel}
              clipsCount={keptClips.length}
              onCompile={compileSessionReel}
              onSelectSession={(sessionId) => {
                setReelSessionId(sessionId || undefined);
                if (!sessionId) return;
                setSelectedSessionId(sessionId);
                setCurrentSessionId(sessionId);
              }}
              onOpenReview={() => { if (reelSession) openSession(reelSession); }}
              onView={goTo}
            />
          </div>

          <div className={`view-panel ${view === "insights" ? "active" : ""}`} id="view-insights">
            <InsightsView sessions={sessions} onStart={() => openProject("local")} onSettings={() => openSettings("you")} />
          </div>
        </main>
      </div>

      {exportOperation && ["queued", "running"].includes(exportOperation.status) && (
        <ExportProgressDock operation={exportOperation} onCancel={cancelExportOperation} />
      )}

      <div id="project-delete-modal" className={`project-modal ${deleteTarget ? "open" : ""}`} onMouseDown={(event) => { if (event.target === event.currentTarget) closeDeleteModal(); }}>
        <div ref={deleteDialogRef} className="project-dialog" role="alertdialog" aria-modal="true" aria-labelledby="project-delete-title" aria-describedby="project-delete-description" tabIndex={-1}>
          <div><h2 id="project-delete-title" className="disp">Delete this session?</h2><p id="project-delete-description"><strong>{deleteTarget?.name ?? "This session"}</strong>{clipsClause(deleteTarget?.clips.length ?? 0, "its")} will be permanently removed. This can’t be undone.</p></div>
          <div className="project-dialog-actions"><button type="button" className="btn ghost" data-autofocus onClick={closeDeleteModal}>Cancel</button><button type="button" className="btn danger solid" onClick={confirmDeleteSession}><Trash2 aria-hidden="true" />Delete session</button></div>
        </div>
      </div>

      <div id="project-bulk-delete-modal" className={`project-modal ${bulkDeleteTargets.length ? "open" : ""}`} onMouseDown={(event) => { if (event.target === event.currentTarget) closeBulkDeleteModal(); }}>
        <div ref={bulkDeleteDialogRef} className="project-dialog" role="alertdialog" aria-modal="true" aria-labelledby="project-bulk-delete-title" aria-describedby="project-bulk-delete-description" tabIndex={-1}>
          <div>
            <h2 id="project-bulk-delete-title" className="disp">Delete {bulkDeleteTargets.length} session{bulkDeleteTargets.length === 1 ? "" : "s"}?</h2>
            <p id="project-bulk-delete-description">
              These sessions{clipsClause(bulkDeleteTargets.reduce((total, session) => total + session.clips.length, 0), "their")} will be permanently removed. This can’t be undone.
            </p>
          </div>
          <div className="project-dialog-actions">
            <button type="button" className="btn ghost" data-autofocus onClick={closeBulkDeleteModal}>Cancel</button>
            <button type="button" className="btn danger solid" onClick={confirmDeleteMany}><Trash2 aria-hidden="true" />Delete {bulkDeleteTargets.length} session{bulkDeleteTargets.length === 1 ? "" : "s"}</button>
          </div>
        </div>
      </div>

      <OnboardingFlow
        open={onboardingOpen}
        apiEndpoint={settings.apiEndpoint}
        dialogRef={onboardingDialogRef}
        onComplete={() => setOnboardingOpen(false)}
        onScanLink={scanLinkFromPalette}
        onChooseFile={() => openProject("local")}
      />


      {/* Export is only half the job — this hands the finished files off to the
          places they get posted. No accounts are connected; it opens pages. */}
      <ShareHandoffDialog
        open={!!shareHandoff}
        folder={shareHandoff?.folder ?? ""}
        count={shareHandoff?.count ?? 0}
        dialogRef={shareDialogRef}
        onClose={closeShareHandoff}
      />


    </>
  );
}

/** " and its 12 clips" / " and its only clip" / "" when there are none. */
function clipsClause(count: number, owner: "its" | "their"): string {
  if (count <= 0) return "";
  if (count === 1) return owner === "its" ? " and its only clip" : " and their only clip";
  return ` and all ${count} of ${owner} clips`;
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
    <div className="page live2">
      <div className="ph">
        <div>
          <div className="row"><h1 className="disp">Recall Live</h1><span className="badge live-alpha">Alpha</span></div>
          <div className="sub">Hit your key when something pops off. Recall remembers the second, then reads just those minutes once the VOD is up.</div>
        </div>
      </div>
      <LiveSessionTestPanel onScanRecording={onScanRecording} sessions={sessions} />
    </div>
  );
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
      aria-label={`Clip ${position} of ${clips.length}: ${savedSet.size} kept, ${maybeIds.size} maybe, ${passedIds.size} passed`}
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
  onOpenEditor,
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
  /** Open this clip in the in-app editor. Only offered for a finished render. */
  onOpenEditor?: (clip: Clip) => void;
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
  const postFirstClipId = useMemo(() => postFirstId(clips), [clips]);
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
          <ArrowLeft size={15} /> All moments
        </button>
        <div className="theater-top-actions">
          {onOpenEditor && !clip.sourceWindow && clip.start_time != null && clip.end_time != null && (
            <button
              type="button"
              className="theater-more-moments"
              onClick={() => onOpenEditor(clip)}
              title="Open it in the Cutting Room: trim, framing, fades, captions and more."
            >
              <Scissors size={13} />
              <span>Edit</span>
            </button>
          )}
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
          <div className="theater-rank-row">
            <b className="theater-rank disp num">#{String(index + 1).padStart(2, "0")}</b>
          <div className="theater-editorial-row">
            <span className={`theater-recommendation ${recommendation.className}`}>{recommendation.label}</span>
            <span className={`theater-opening ${opening.className}`} title="Opening strength, not overall clip quality">{opening.label}</span>
            {clip.momentType && (
              <span className="moment-type-badge">{momentLabel(clip.momentType)}</span>
            )}
            {clip.isOverflowCandidate && <span className="theater-second-look">Second look</span>}
            {clip.editedVideoUrl && (
              <span className="theater-edited" title="You finished this clip in the editor. Exports use your edit.">Your edit</span>
            )}
            <RecallOriginBadge provenance={clip.recallProvenance} />
          </div>
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
                <button type="button" className="btn sm ghost theater-title-rename" onClick={() => { setTitleDraft(clip.title || ""); setTitleEditing(true); }} aria-label={`Rename ${clip.title || "clip"}`}>Rename</button>
              </div>
            )}
            {titleError && <p className="theater-title-error" role="alert">{titleError}</p>}
          </div>
          {/* Where this sits in the source recording. Reviewers need it to find
              the moment in the VOD (or in their own editor) and to tell two
              similar-looking clips apart; it used to exist only as a fallback
              inside the title string. */}
          <HypeMeter clip={clip} />
          <PostingCoach clip={clip} postFirst={clip.id === postFirstClipId} />
          <div className="theater-vod-position">
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
          <p className="theater-filename-preview" title={exportNamePreview}>Saves as <span>{exportNamePreview}</span></p>
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

        <div className="theater-actions glass-float">
          <button type="button" className="theater-keep" onClick={() => decide("keep")} disabled={!!decision} aria-label="Keep clip">
            <Check aria-hidden="true" /><span className="disp">Keep</span><kbd>Space</kbd>
          </button>
          <button type="button" className="theater-maybe" onClick={() => decide("maybe")} disabled={!!decision} aria-label="Mark clip as maybe">
            <span className="disp">Maybe</span><kbd>M</kbd>
          </button>
          <button type="button" className="theater-pass" onClick={() => decide("pass")} disabled={!!decision} aria-label="Pass clip">
            <span className="disp">Pass</span><kbd>X</kbd>
          </button>
        </div>
      </section>

      <section className="theater-next">
        {reviewDone ? (
          <div className="theater-wrapup">
            <span className="theater-wrapup-mark" aria-hidden="true"><Check /></span>
            <span className="theater-wrapup-copy"><strong>That's the deck.</strong><span className="num">Reviewed {clips.length} · kept {savedSet.size} · maybe {maybeIds.size}</span></span>
            <StudioButton tone="primary" loading={exporting} onClick={onExport}>Export kept clips</StudioButton>
          </div>
        ) : (
          <>
            <span className="theater-next-label">Up next</span>
            <div className="theater-next-row">
              {nextClips.map((c: Clip) => (
                <button type="button" key={c.id} onClick={() => onSelect(c.id)} disabled={!!decision}>
                  <Poster clip={c} />
                  <b>{openingBand(hookScoreOf(c)).shortLabel}</b>
                </button>
              ))}
            </div>
            <div className="theater-hints"><span><kbd>←</kbd><kbd>→</kbd> move</span><span><kbd>[</kbd><kbd>]</kbd> start</span><span><kbd>,</kbd><kbd>.</kbd> end</span><span><kbd>Esc</kbd> deck</span></div>
          </>
        )}
      </section>
    </div>
  );
}


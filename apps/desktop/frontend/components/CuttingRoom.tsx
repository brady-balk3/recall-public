// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent, type WheelEvent as ReactWheelEvent, type KeyboardEvent as ReactKeyboardEvent } from "react";
import {
  ArrowLeft,
  Check,
  Download,
  Film,
  Pause,
  Play,
  Repeat,
  RotateCcw,
  Scissors,
  Settings,
  Sparkles,
  Target,
  Volume2,
  X,
} from "../lib/icons";
import { apiFetch, apiUrl } from "../lib/api";
import { isInteractiveKeyboardTarget } from "../lib/keyboard";
import { timelinePath } from "../lib/timeline";
import StudioEmptyState from "./StudioEmptyState";
import { MediaVolumeControl, StudioSlider, StudioSwitch, useMediaVolume } from "./StudioControls";
import {
  useJobStore,
  type Clip,
  type ManualClipLayout,
  type ManualClipOptions,
  type ManualFraming,
  type ReactionTimeline,
  type Session,
} from "../lib/store";

const MAX_CLIP_SECONDS = 180;
const MIN_CLIP_SECONDS = 1;
const CURVE_HEIGHT = 92;
const PRECISION_SPANS = [180, 90, 45, 20, 8] as const;
/** Throttle media seeks while scrubbing long local VODs (UI playhead stays live). */
const SCRUB_SEEK_MS = 90;
const RANGE_SELECT_PX = 8;

type FadeState = {
  videoIn: number;
  videoOut: number;
  audioIn: number;
  audioOut: number;
};

type Props = {
  session?: Session;
  timeline: ReactionTimeline | null;
  onBack: () => void;
  onOpenLibrary: () => void;
  initialMoment?: CuttingRoomMemoryHandoff | null;
};

export type CuttingRoomMemoryHandoff = {
  entryId: string;
  timestamp: number;
  endTime: number;
  query: string;
};

type SourceStatus = {
  source_key?: string | null;
  state: string;
  state_label: string;
  restore_available: boolean;
  local_path?: string | null;
  last_error?: string | null;
};

const emptyFades: FadeState = { videoIn: 0, videoOut: 0, audioIn: 0, audioOut: 0 };

function FilmstripImage({ src, label }: { src?: string; label: string }) {
  const [loadedSrc, setLoadedSrc] = useState<string | null>(null);
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const state = src && loadedSrc === src ? "ready" : src && failedSrc === src ? "error" : "loading";

  return (
    <div className={`vod-filmstrip-image is-${state}`} aria-label={label}>
      {src && state !== "error" && (
        <img
          src={src}
          alt=""
          draggable={false}
          onLoad={() => setLoadedSrc(src)}
          onError={() => setFailedSrc(src)}
        />
      )}
      {state === "loading" && <span>Loading source frames…</span>}
      {state === "error" && <span>Source frames unavailable</span>}
    </div>
  );
}

function rangeStyle(start: number, end: number, windowStart: number, windowEnd: number): CSSProperties | null {
  const visibleStart = Math.max(start, windowStart);
  const visibleEnd = Math.min(end, windowEnd);
  if (visibleEnd <= visibleStart || windowEnd <= windowStart) return null;
  return {
    left: `${(visibleStart - windowStart) / (windowEnd - windowStart) * 100}%`,
    width: `${Math.max(0.28, (visibleEnd - visibleStart) / (windowEnd - windowStart) * 100)}%`,
  };
}

function preciseClock(seconds: number) {
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = (safe % 60).toFixed(2).padStart(5, "0");
  return hours > 0
    ? `${hours}:${minutes.toString().padStart(2, "0")}:${remainder}`
    : `${minutes.toString().padStart(2, "0")}:${remainder}`;
}

/** Intervals a creator can navigate by. A ruler always lands on one of these,
 * so its labels are round numbers rather than whatever the window happens to
 * start on. */
const RULER_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];

/** The interval a rail of `span` seconds is labelled at.
 *
 * Choosing it from the span is what makes one ruler serve both rails: eight
 * seconds of precision window and four hours of VOD get labels at the same
 * visual density, and a longer VOD gets a coarser step rather than the same
 * numbers stretched over more time. */
function rulerStep(span: number, slots = 8) {
  return RULER_STEPS.find((value) => value >= span / slots)
    ?? Math.ceil(span / slots / 3600) * 3600;
}

/** Round time labels for a rail covering `span` seconds from `start`. */
function rulerTicks(start: number, span: number, slots = 8) {
  if (!(span > 0)) return [];
  const step = rulerStep(span, slots);
  const ticks: { seconds: number; position: number }[] = [];
  const first = Math.ceil((start - 1e-6) / step) * step;
  // Stop short of the far edge so the last label cannot collide with the
  // rail's own end.
  for (let seconds = first; seconds < start + span - step * 0.35; seconds += step) {
    ticks.push({ seconds, position: (seconds - start) / span * 100 });
  }
  return ticks;
}

/** A ruler label: hour-aware, and only as precise as its own step. */
function tickClock(seconds: number, step: number) {
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const rest = safe % 60;
  const secs = step < 1
    ? rest.toFixed(1).padStart(4, "0")
    : Math.round(rest).toString().padStart(2, "0");
  return hours > 0
    ? `${hours}:${minutes.toString().padStart(2, "0")}:${secs}`
    : `${minutes}:${secs}`;
}

function parseTime(value: string) {
  const parts = value.trim().split(":").map(Number);
  if (!parts.length || parts.length > 3 || parts.some((part) => !Number.isFinite(part) || part < 0)) return null;
  return parts.reduce((total, part) => total * 60 + part, 0);
}

function FadeControl({ label, value, max, onChange }: { label: string; value: number; max: number; onChange: (value: number) => void }) {
  const safe = Math.min(max, Math.max(0, value));
  return (
    <label className="library-editor-fade">
      <span>{label}</span>
      <span className="library-editor-number"><input type="number" min={0} max={max} step={0.1} value={safe} onChange={(event) => onChange(Math.min(max, Math.max(0, Number(event.target.value) || 0)))} /><i>s</i></span>
      <StudioSlider value={safe} min={0} max={Math.max(.1, max)} step={0.1} onChange={onChange} label={`${label} duration`} ticks={["0", `${(max / 2).toFixed(1)}`, `${max.toFixed(1)}s`]} />
    </label>
  );
}

function detailTimelinePath(timeline: ReactionTimeline | null, start: number, end: number) {
  if (!timeline?.t?.length || !timeline.r?.length || end <= start) return "";
  const points = timeline.t
    .map((time, index) => ({ time, reaction: timeline.r[index] ?? 0 }))
    .filter((point) => point.time >= start && point.time <= end);
  if (points.length < 2) return "";
  return points.map((point, index) => {
    const x = ((point.time - start) / (end - start)) * 1000;
    const y = CURVE_HEIGHT - 4 - Math.max(0, Math.min(1, point.reaction)) * (CURVE_HEIGHT - 12);
    return `${index ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
}

/**
 * Full-source editor for finding moments Recall missed. The source remains an
 * honest 16:9 VOD while the output proof uses the same 9:16 renderer as the
 * Clip Library. Every finished cut is inserted as kept and records a manual
 * missed-positive label.
 */
export default function CuttingRoom({ session, timeline, onBack, onOpenLibrary, initialMoment }: Props) {
  const settings = useJobStore((state) => state.settings);
  const createManualClip = useJobStore((state) => state.createManualClip);
  const previewManualClip = useJobStore((state) => state.previewManualClip);
  const resolveManualFraming = useJobStore((state) => state.resolveManualFraming);
  const addLog = useJobStore((state) => state.addLog);

  const sourceVideoRef = useRef<HTMLVideoElement>(null);
  const outputVideoRef = useRef<HTMLVideoElement>(null);
  const overviewRef = useRef<HTMLDivElement>(null);
  const detailRef = useRef<HTMLDivElement>(null);
  const previewRangeRef = useRef<{ in: number; out: number } | null>(null);
  const outputUrlRef = useRef<string | null>(null);
  const detailGestureRef = useRef<{
    startX: number;
    startTime: number;
    selecting: boolean;
    rangeSelect: boolean;
  } | null>(null);
  const appliedMemoryMomentRef = useRef<string | null>(null);
  /** User wants playback; survives scrub pauses and failed play() races. */
  const playIntentRef = useRef(false);
  const scrubStateRef = useRef<{
    active: boolean;
    resumeAfter: boolean;
    lastSeekAt: number;
    pendingTime: number | null;
  }>({ active: false, resumeAfter: false, lastSeekAt: 0, pendingTime: null });

  const [sourceState, setSourceState] = useState<"loading" | "ready" | "missing">("loading");
  const [sourceStatus, setSourceStatus] = useState<SourceStatus | null>(null);
  const [restoringSource, setRestoringSource] = useState(false);
  const [sourceRevision, setSourceRevision] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [mediaDuration, setMediaDuration] = useState(0);
  const [current, setCurrent] = useState(0);
  const [inPoint, setInPoint] = useState<number | null>(null);
  const [outPoint, setOutPoint] = useState<number | null>(null);
  const [dragging, setDragging] = useState<"overview" | "detail" | "in" | "out" | "detail-in" | "detail-out" | null>(null);
  const [previewingRange, setPreviewingRange] = useState(false);
  const [timelineZoom, setTimelineZoom] = useState(2);
  const [precisionAnchor, setPrecisionAnchor] = useState(0);
  const [precisionHoverTime, setPrecisionHoverTime] = useState<number | null>(null);
  const [overviewHoverTime, setOverviewHoverTime] = useState<number | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [title, setTitle] = useState("");
  const [tags, setTags] = useState("");
  const [layoutChoice, setLayoutChoice] = useState<ManualClipLayout>("auto");
  const [focusX, setFocusX] = useState(0.5);
  const [captionsEnabled, setCaptionsEnabled] = useState(settings.captionStyle.enabled);
  const [fades, setFades] = useState<FadeState>(emptyFades);
  const [framing, setFraming] = useState<ManualFraming | null>(null);
  const [framingState, setFramingState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [showOutput, setShowOutput] = useState(false);
  const [outputCurrent, setOutputCurrent] = useState(0);
  const [renderingPreview, setRenderingPreview] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createdClip, setCreatedClip] = useState<Clip | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [memoryMoment, setMemoryMoment] = useState<CuttingRoomMemoryHandoff | null>(initialMoment ?? null);
  const latestCaptionDefaultRef = useRef(settings.captionStyle.enabled);
  latestCaptionDefaultRef.current = settings.captionStyle.enabled;
  const mediaVolume = useMediaVolume([sourceVideoRef, outputVideoRef], `${session?.id ?? "none"}:${outputUrl ?? "source"}`);

  const scanFinished = !!session && ["completed", "cancelled"].includes(session.status);
  const sourceUrl = useMemo(
    () => (session ? apiUrl(`/jobs/${session.id}/source?revision=${sourceRevision}`, settings.apiEndpoint) : undefined),
    [session?.id, settings.apiEndpoint, sourceRevision],
  );
  const loadSourceStatus = useCallback(async () => {
    if (!session) return null;
    try {
      const response = await apiFetch(`/jobs/${session.id}/source-status`, undefined, settings.apiEndpoint);
      if (!response.ok) return null;
      const status = await response.json() as SourceStatus;
      setSourceStatus(status);
      return status;
    } catch {
      return null;
    }
  }, [session?.id, settings.apiEndpoint]);

  const restoreSource = useCallback(async () => {
    if (!session || restoringSource) return;
    setRestoringSource(true);
    setError(null);
    try {
      const response = await apiFetch(
        `/jobs/${session.id}/source/restore`, { method: "POST" }, settings.apiEndpoint,
      );
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = body?.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : detail?.message || "Recall could not start source restoration.",
        );
      }
      addLog("Restoring the Twitch source. Download progress is recorded in session activity.", "info");
      await loadSourceStatus();
    } catch (cause) {
      setRestoringSource(false);
      setError(cause instanceof Error ? cause.message : "Recall could not restore this source.");
    }
  }, [session?.id, restoringSource, settings.apiEndpoint, addLog, loadSourceStatus]);

  const cancelSourceRestore = useCallback(async () => {
    if (!sourceStatus?.source_key) return;
    await apiFetch(
      `/source-assets/${encodeURIComponent(sourceStatus.source_key)}/restore/cancel`,
      { method: "POST" },
      settings.apiEndpoint,
    ).catch(() => undefined);
  }, [sourceStatus?.source_key, settings.apiEndpoint]);

  useEffect(() => {
    if (sourceState === "missing") void loadSourceStatus();
  }, [sourceState, loadSourceStatus]);

  // A media element that already has its metadata will never fire
  // `loadedmetadata` again, and the element outlives a re-render. When that
  // happens -- a warm HTTP cache, a re-mount onto the same src -- the editor
  // sat on "Opening the source recording…" forever with every control
  // disabled, because nothing else ever promotes it to ready. Read the
  // element's own state instead of waiting for an event that has been and gone.
  useEffect(() => {
    if (sourceState !== "loading") return;
    const video = sourceVideoRef.current;
    if (!video) return;
    if (video.error) { setSourceState("missing"); return; }
    if (video.readyState >= 1) {
      if (Number.isFinite(video.duration) && video.duration > 0) setMediaDuration(video.duration);
      setSourceState("ready");
    }
  }, [sourceState, sourceRevision, session?.id]);

  useEffect(() => {
    if (!restoringSource) return;
    const timer = window.setInterval(async () => {
      const status = await loadSourceStatus();
      if (!status) return;
      if (status.local_path) {
        window.clearInterval(timer);
        setRestoringSource(false);
        setSourceState("loading");
        setSourceRevision((value) => value + 1);
        addLog("Twitch source restored and ready in the VOD Editor.", "success");
      } else if (status.state === "unavailable") {
        window.clearInterval(timer);
        setRestoringSource(false);
        setError(status.last_error || "The original Twitch source is currently unavailable.");
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [restoringSource, loadSourceStatus, addLog]);
  const timelineEnd = timeline?.t?.length ? timeline.t[timeline.t.length - 1] : 0;
  const vodDuration = Math.max(1, mediaDuration || session?.vodDuration || timelineEnd || 1);
  const selectionDuration = inPoint != null && outPoint != null ? outPoint - inPoint : null;
  const tooLong = selectionDuration != null && selectionDuration > MAX_CLIP_SECONDS;
  const tooShort = selectionDuration != null && selectionDuration < MIN_CLIP_SECONDS;
  const selectionReady = sourceState === "ready" && selectionDuration != null && !tooLong && !tooShort;
  const fadeMax = Math.max(0, Math.min(5, (selectionDuration ?? 10) / 2));
  const overviewCurve = timelinePath(timeline, 1000, CURVE_HEIGHT);

  const detailSpan = Math.min(vodDuration, PRECISION_SPANS[timelineZoom - 1] ?? 90);
  const detailCenter = Math.max(0, Math.min(vodDuration, precisionAnchor));
  let detailStart = Math.max(0, detailCenter - detailSpan / 2);
  let detailEnd = Math.min(vodDuration, detailStart + detailSpan);
  detailStart = Math.max(0, detailEnd - detailSpan);
  // Quantized bounds keep the visual window stable while video playback
  // advances; a new contact sheet is requested only after explicit seeking,
  // marking, or trimming moves the editor's precision anchor.
  detailStart = Math.floor(detailStart * 2) / 2;
  detailEnd = Math.min(vodDuration, Math.ceil(detailEnd * 2) / 2);
  const detailDuration = Math.max(1, detailEnd - detailStart);
  const [detailFilmstripRange, setDetailFilmstripRange] = useState<{ start: number; end: number } | null>(null);
  useEffect(() => {
    if (sourceState !== "ready") {
      setDetailFilmstripRange(null);
      return;
    }
    const timer = window.setTimeout(() => setDetailFilmstripRange({ start: detailStart, end: detailEnd }), 180);
    return () => window.clearTimeout(timer);
  }, [sourceState, session?.id, detailStart, detailEnd]);
  const detailCurve = detailTimelinePath(timeline, detailStart, detailEnd);
  const playheadPct = Math.max(0, Math.min(100, current / vodDuration * 100));
  const inPct = inPoint == null ? null : Math.max(0, Math.min(100, inPoint / vodDuration * 100));
  const outPct = outPoint == null ? null : Math.max(0, Math.min(100, outPoint / vodDuration * 100));
  const detailPlayPct = Math.max(0, Math.min(100, (current - detailStart) / detailDuration * 100));
  const detailInPct = inPoint == null ? null : Math.max(0, Math.min(100, (inPoint - detailStart) / detailDuration * 100));
  const detailOutPct = outPoint == null ? null : Math.max(0, Math.min(100, (outPoint - detailStart) / detailDuration * 100));
  const detailPlayVisible = current >= detailStart && current <= detailEnd;
  const detailInVisible = inPoint != null && inPoint >= detailStart && inPoint <= detailEnd;
  const detailOutVisible = outPoint != null && outPoint >= detailStart && outPoint <= detailEnd;
  const precisionHoverPct = precisionHoverTime == null ? null : Math.max(0, Math.min(100, (precisionHoverTime - detailStart) / detailDuration * 100));
  const overviewHoverPct = overviewHoverTime == null ? null : Math.max(0, Math.min(100, overviewHoverTime / vodDuration * 100));
  const detailWindowStyle = {
    left: `${detailStart / vodDuration * 100}%`,
    width: `${Math.max(0.45, detailDuration / vodDuration * 100)}%`,
  } as CSSProperties;
  const overviewFilmstripUrl = sourceState === "ready" && session
    ? apiUrl(`/jobs/${session.id}/filmstrip?start=0&end=${vodDuration.toFixed(2)}&frames=36&width=1920&height=104`, settings.apiEndpoint)
    : undefined;
  const detailFilmstripUrl = sourceState === "ready" && session && detailFilmstripRange
    ? apiUrl(`/jobs/${session.id}/filmstrip?start=${detailFilmstripRange.start.toFixed(2)}&end=${detailFilmstripRange.end.toFixed(2)}&frames=24&width=1920&height=160`, settings.apiEndpoint)
    : undefined;
  const autoClips = useMemo(() => (session?.clips ?? []).filter((clip) => !clip.isManual), [session?.clips]);
  const manualClips = useMemo(() => (session?.clips ?? []).filter((clip) => clip.isManual), [session?.clips]);

  // Two rails, two scales. The precision window is seconds wide and the VOD is
  // hours wide, so each carries its own ruler -- a single ruler between them
  // read as the VOD's, and clicking the VOD track where it said 00:30 landed
  // over an hour in.
  const detailStep = rulerStep(detailDuration);
  const detailTicks = rulerTicks(detailStart, detailDuration);
  const overviewStep = rulerStep(vodDuration);
  const overviewTicks = rulerTicks(0, vodDuration);

  // Live capture-region overlay drawn on the 16:9 source. The facecam/gameplay
  // rects are the exact normalized [x, y, w, h] regions the export engine pulls
  // from (engines/clip/layout.py), so this is truthful and costs nothing to
  // draw — no second decode or render. For full-frame gameplay we also draw the
  // real 9:16 portrait crop band: a 9:16 slice of a full-height 16:9 frame is
  // exactly 81/256 of the width, positioned horizontally by the focus slider.
  const cropOverlay = useMemo(() => {
    if (!framing) return null;
    const isRect = (r?: number[] | null): r is number[] => Array.isArray(r) && r.length === 4;
    const regions: { key: "game" | "cam"; label: string; rect: number[] }[] = [];
    if (isRect(framing.gameplay)) regions.push({ key: "game", label: "Game", rect: framing.gameplay });
    if (isRect(framing.facecam)) regions.push({ key: "cam", label: "Cam", rect: framing.facecam });
    let band: { left: number; top: number; width: number; height: number } | null = null;
    if (framing.type === "full_gameplay") {
      const [rx, ry, rw, rh] = isRect(framing.gameplay) ? framing.gameplay : [0, 0, 1, 1];
      const width = Math.min(rw, rh * 81 / 256);
      const focus = Math.max(0, Math.min(1, framing.focus_x ?? focusX));
      band = { left: rx + focus * Math.max(0, rw - width), top: ry, width, height: rh };
    }
    return regions.length || band ? { regions, band } : null;
  }, [framing, focusX]);

  // The crop rects are normalized to the SOURCE FRAME, so they may only be drawn
  // over the video's rendered picture — not over the element that contains it.
  // `.vod-editor-video-wrap.is-source` sets an explicit height alongside
  // `aspect-ratio: 16/9` and `max-width`, so whenever the max-width clamps (a
  // narrow stage, the settings drawer open) the wrap ends up TALLER than 16:9,
  // `object-fit: contain` letterboxes the video inside it, and an `inset: 0`
  // overlay draws every region stretched and pushed down — the facecam box lands
  // below the real cam, which reads as a detection failure. Measure the picture
  // and position the overlay on that instead; when the wrap already matches the
  // source aspect this resolves to the same rect.
  const [pictureBox, setPictureBox] = useState<CSSProperties | null>(null);
  const measurePicture = useCallback(() => {
    const video = sourceVideoRef.current;
    if (!video) return;
    const { clientWidth: boxW, clientHeight: boxH, videoWidth: srcW, videoHeight: srcH } = video;
    if (!boxW || !boxH || !srcW || !srcH) {
      setPictureBox(null);
      return;
    }
    const scale = Math.min(boxW / srcW, boxH / srcH);
    const width = srcW * scale;
    const height = srcH * scale;
    setPictureBox({
      left: (boxW - width) / 2,
      top: (boxH - height) / 2,
      width,
      height,
      right: "auto",
      bottom: "auto",
    });
  }, []);

  useEffect(() => {
    const video = sourceVideoRef.current;
    if (!video || typeof ResizeObserver === "undefined") return;
    measurePicture();
    const observer = new ResizeObserver(measurePicture);
    observer.observe(video);
    return () => observer.disconnect();
  }, [measurePicture, sourceState, showOutput]);

  const buildOptions = useCallback((): ManualClipOptions | null => {
    if (inPoint == null || outPoint == null) return null;
    return {
      start: inPoint,
      end: outPoint,
      title: title.trim(),
      tags: tags.split(",").map((tag) => tag.trim()).filter(Boolean),
      layout: layoutChoice,
      focus_x: focusX,
      captions_enabled: captionsEnabled,
      video_fade_in: fades.videoIn,
      video_fade_out: fades.videoOut,
      audio_fade_in: fades.audioIn,
      audio_fade_out: fades.audioOut,
      memory_entry_id: memoryMoment?.entryId,
      memory_query: memoryMoment?.query,
    };
  }, [inPoint, outPoint, title, tags, layoutChoice, focusX, captionsEnabled, fades, memoryMoment]);

  const clearOutputPreview = useCallback(() => {
    if (outputUrlRef.current) URL.revokeObjectURL(outputUrlRef.current);
    outputUrlRef.current = null;
    setOutputUrl(null);
    setShowOutput(false);
    setOutputCurrent(0);
  }, []);

  const stopRangePreview = useCallback(() => {
    previewRangeRef.current = null;
    setPreviewingRange(false);
  }, []);

  const invalidateMediaProof = useCallback(() => {
    clearOutputPreview();
    setCreatedClip(null);
  }, [clearOutputPreview]);

  const applyVideoTime = useCallback((time: number, force = false) => {
    const video = sourceVideoRef.current;
    if (!video || !Number.isFinite(time)) return;
    const scrub = scrubStateRef.current;
    const now = performance.now();
    if (!force && scrub.active && now - scrub.lastSeekAt < SCRUB_SEEK_MS) {
      scrub.pendingTime = time;
      return;
    }
    scrub.lastSeekAt = now;
    scrub.pendingTime = null;
    if (Math.abs((video.currentTime || 0) - time) > 0.004) {
      video.currentTime = time;
    }
  }, []);

  const seekTo = useCallback((time: number, options?: { scrubbing?: boolean }) => {
    const next = Math.max(0, Math.min(time, vodDuration));
    setCurrent(next);
    applyVideoTime(next, !options?.scrubbing);
  }, [vodDuration, applyVideoTime]);

  const beginScrub = useCallback(() => {
    const scrub = scrubStateRef.current;
    if (scrub.active) return;
    scrub.active = true;
    scrub.resumeAfter = playIntentRef.current;
    scrub.lastSeekAt = 0;
    scrub.pendingTime = null;
    const video = sourceVideoRef.current;
    if (video && !video.paused) video.pause();
  }, []);

  const endScrub = useCallback(() => {
    const scrub = scrubStateRef.current;
    if (!scrub.active) return;
    if (scrub.pendingTime != null) applyVideoTime(scrub.pendingTime, true);
    const resume = scrub.resumeAfter && playIntentRef.current;
    scrub.active = false;
    scrub.resumeAfter = false;
    scrub.pendingTime = null;
    if (!resume) return;
    const video = sourceVideoRef.current;
    if (!video) return;
    void video.play().then(() => {
      if (playIntentRef.current) setPlaying(true);
    }).catch(() => {
      playIntentRef.current = false;
      setPlaying(false);
    });
  }, [applyVideoTime]);

  const togglePlay = useCallback(() => {
    const video = showOutput ? outputVideoRef.current : sourceVideoRef.current;
    if (!video || (!showOutput && sourceState !== "ready")) return;

    // Drive from the button state, not video.paused — large VOD seeks often leave
    // the element paused while a prior play() promise is still racing.
    if (playing) {
      playIntentRef.current = false;
      scrubStateRef.current.resumeAfter = false;
      video.pause();
      setPlaying(false);
      return;
    }

    playIntentRef.current = true;
    setPlaying(true);
    void video.play().catch(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 70));
      if (!playIntentRef.current) return;
      try {
        await video.play();
        setPlaying(true);
      } catch {
        playIntentRef.current = false;
        setPlaying(false);
      }
    });
  }, [showOutput, sourceState, playing]);

  const onSourcePlay = useCallback(() => {
    playIntentRef.current = true;
    setPlaying(true);
  }, []);

  const onSourcePause = useCallback(() => {
    if (scrubStateRef.current.active) return;
    playIntentRef.current = false;
    setPlaying(false);
  }, []);

  const markIn = useCallback(() => {
    if (sourceState !== "ready") return;
    stopRangePreview();
    invalidateMediaProof();
    setInPoint(current);
    setOutPoint((value) => (value != null && value <= current ? null : value));
    setPrecisionAnchor(current);
    setError(null);
  }, [sourceState, current, stopRangePreview, invalidateMediaProof]);

  const markOut = useCallback(() => {
    if (sourceState !== "ready") return;
    stopRangePreview();
    invalidateMediaProof();
    setOutPoint(current);
    setInPoint((value) => (value != null && value >= current ? null : value));
    setPrecisionAnchor(inPoint != null && inPoint < current ? (inPoint + current) / 2 : current);
    setError(null);
  }, [sourceState, current, inPoint, stopRangePreview, invalidateMediaProof]);

  const clearSelection = useCallback(() => {
    stopRangePreview();
    invalidateMediaProof();
    setInPoint(null);
    setOutPoint(null);
    setError(null);
  }, [stopRangePreview, invalidateMediaProof]);

  const previewRange = useCallback(() => {
    if (previewingRange) { stopRangePreview(); return; }
    if (inPoint == null || outPoint == null) return;
    setShowOutput(false);
    previewRangeRef.current = { in: inPoint, out: outPoint };
    setPreviewingRange(true);
    seekTo(inPoint);
    playIntentRef.current = true;
    setPlaying(true);
    void sourceVideoRef.current?.play().catch(() => {
      playIntentRef.current = false;
      setPlaying(false);
    });
  }, [previewingRange, inPoint, outPoint, seekTo, stopRangePreview]);

  const resetDraft = useCallback(() => {
    stopRangePreview();
    clearOutputPreview();
    setInPoint(null);
    setOutPoint(null);
    setTimelineZoom(2);
    setPrecisionAnchor(current);
    setTitle(session ? `Clip from ${session.name}` : "");
    setTags("");
    setLayoutChoice("auto");
    setFocusX(0.5);
    setCaptionsEnabled(latestCaptionDefaultRef.current);
    setFades(emptyFades);
    setFraming(null);
    setFramingState("idle");
    setCreatedClip(null);
    setError(null);
  }, [session, stopRangePreview, clearOutputPreview, current]);

  useEffect(() => {
    previewRangeRef.current = null;
    playIntentRef.current = false;
    scrubStateRef.current = { active: false, resumeAfter: false, lastSeekAt: 0, pendingTime: null };
    sourceVideoRef.current?.pause();
    clearOutputPreview();
    setSourceState("loading");
    setPlaying(false);
    setMediaDuration(0);
    setCurrent(0);
    setInPoint(null);
    setOutPoint(null);
    setTimelineZoom(2);
    setPrecisionAnchor(0);
    setSettingsOpen(false);
    setDragging(null);
    setTitle(session ? `Clip from ${session.name}` : "");
    setTags("");
    setLayoutChoice("auto");
    setFocusX(0.5);
    // Caption settings seed a new source draft; later global changes must not
    // rewrite the creator's in-progress choice or any other draft field.
    setCaptionsEnabled(latestCaptionDefaultRef.current);
    setFades(emptyFades);
    setFraming(null);
    setFramingState("idle");
    setCreatedClip(null);
    setError(null);
    setMemoryMoment(initialMoment ?? null);
    appliedMemoryMomentRef.current = null;
  }, [sourceUrl, clearOutputPreview, initialMoment]);

  useEffect(() => {
    if (!session || sourceState !== "ready" || !memoryMoment || vodDuration <= 0) return;
    const key = `${session.id}:${memoryMoment.entryId}`;
    if (appliedMemoryMomentRef.current === key) return;
    appliedMemoryMomentRef.current = key;
    const anchor = Math.max(0, Math.min(memoryMoment.timestamp, vodDuration));
    let draftStart = Math.max(0, anchor - 10);
    const evidenceEnd = Math.max(anchor, memoryMoment.endTime);
    const draftEnd = Math.min(vodDuration, Math.max(anchor + 20, evidenceEnd + 8));
    if (draftEnd - draftStart < MIN_CLIP_SECONDS) {
      draftStart = Math.max(0, draftEnd - MIN_CLIP_SECONDS);
    }
    seekTo(anchor);
    setPrecisionAnchor(anchor);
    setInPoint(draftStart);
    setOutPoint(draftEnd);
    setTitle(memoryMoment.query.trim()
      ? `Memory: ${memoryMoment.query.trim()}`.slice(0, 200)
      : `Memory moment from ${session.name}`.slice(0, 200));
    setError(null);
  }, [session, sourceState, memoryMoment, vodDuration, seekTo]);

  useEffect(() => () => {
    if (outputUrlRef.current) URL.revokeObjectURL(outputUrlRef.current);
  }, []);

  useEffect(() => {
    if (!session || !selectionReady || inPoint == null || outPoint == null) {
      setFraming(null);
      setFramingState("idle");
      return;
    }
    const options: ManualClipOptions = {
      start: inPoint,
      end: outPoint,
      layout: layoutChoice,
      focus_x: focusX,
      captions_enabled: captionsEnabled,
      video_fade_in: 0,
      video_fade_out: 0,
      audio_fade_in: 0,
      audio_fade_out: 0,
    };
    let cancelled = false;
    setFramingState("loading");
    const timer = window.setTimeout(() => {
      resolveManualFraming(session.id, options)
        .then((value) => {
          if (!cancelled) { setFraming(value); setFramingState("ready"); }
        })
        .catch(() => {
          if (!cancelled) { setFraming(null); setFramingState("error"); }
        });
    }, 220);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [session?.id, selectionReady, inPoint, outPoint, layoutChoice, focusX, resolveManualFraming]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (isInteractiveKeyboardTarget(event.target)) return;
      const key = event.key.toLowerCase();
      if (key === "i") { event.preventDefault(); markIn(); }
      if (key === "o") { event.preventDefault(); markOut(); }
      if (event.code === "Space") { event.preventDefault(); togglePlay(); }
      if (event.key === "ArrowLeft" && !showOutput) { event.preventDefault(); stopRangePreview(); const step = event.altKey ? 0.1 : event.shiftKey ? 5 : 1; const next = current - step; seekTo(next); setPrecisionAnchor(Math.max(0, next)); }
      if (event.key === "ArrowRight" && !showOutput) { event.preventDefault(); stopRangePreview(); const step = event.altKey ? 0.1 : event.shiftKey ? 5 : 1; const next = current + step; seekTo(next); setPrecisionAnchor(Math.min(vodDuration, next)); }
      if (event.key === "Escape") {
        event.preventDefault();
        if (settingsOpen) setSettingsOpen(false);
        else if (inPoint != null || outPoint != null) clearSelection();
        else onBack();
      }
      if ((event.key === "Delete" || event.key === "Backspace") && (inPoint != null || outPoint != null)) {
        event.preventDefault();
        clearSelection();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [markIn, markOut, togglePlay, showOutput, stopRangePreview, seekTo, current, vodDuration, settingsOpen, onBack, inPoint, outPoint, clearSelection]);

  useEffect(() => {
    if (!playing || showOutput) return;
    let frame = 0;
    const tick = () => {
      if (sourceVideoRef.current) setCurrent(sourceVideoRef.current.currentTime || 0);
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [playing, showOutput]);

  const onSourceTimeUpdate = () => {
    const video = sourceVideoRef.current;
    if (!video) return;
    setCurrent(video.currentTime || 0);
    const range = previewRangeRef.current;
    if (range && video.currentTime >= range.out - 0.03) video.currentTime = range.in;
  };

  const generateOutputPreview = async () => {
    if (!session || !selectionReady || renderingPreview) return;
    const options = buildOptions();
    if (!options) return;
    setRenderingPreview(true);
    setError(null);
    sourceVideoRef.current?.pause();
    stopRangePreview();
    try {
      const blob = await previewManualClip(session.id, options);
      clearOutputPreview();
      const url = URL.createObjectURL(blob);
      outputUrlRef.current = url;
      setOutputUrl(url);
      setShowOutput(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Recall could not render that output preview.");
    } finally {
      setRenderingPreview(false);
    }
  };

  const createClip = async () => {
    if (!session || !selectionReady || creating) return;
    const options = buildOptions();
    if (!options) return;
    if (!options.title?.trim()) { setError("Give this clip a title before adding it to the library."); return; }
    setCreating(true);
    setError(null);
    sourceVideoRef.current?.pause();
    outputVideoRef.current?.pause();
    stopRangePreview();
    addLog("Rendering your finished clip from the source recording…", "info");
    try {
      const clip = await createManualClip(session.id, options);
      setCreatedClip(clip);
      setMemoryMoment(null);
      setPrecisionAnchor(((clip.start_time ?? current) + (clip.end_time ?? current)) / 2);
      addLog(`“${clip.title}” is ready in Clip Library. Recall recorded the missed moment.`, "success");
      clearOutputPreview();
      setInPoint(null);
      setOutPoint(null);
      setTitle(`Clip from ${session.name}`);
      setTags("");
      setFades(emptyFades);
      setFraming(null);
      setFramingState("idle");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Recall could not create that clip. Try again.");
    } finally {
      setCreating(false);
    }
  };

  const timeFromPointer = (clientX: number, detail = false) => {
    const rect = (detail ? detailRef.current : overviewRef.current)?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return detail ? detailStart : 0;
    const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return detail ? detailStart + fraction * detailDuration : fraction * vodDuration;
  };

  const clampPrecisionAnchor = (time: number, span = detailDuration) => {
    if (span >= vodDuration) return vodDuration / 2;
    return Math.max(span / 2, Math.min(vodDuration - span / 2, time));
  };

  const panPrecision = (seconds: number) => {
    stopRangePreview();
    setPrecisionAnchor((value) => clampPrecisionAnchor(value + seconds));
  };

  const setPrecisionZoom = (level: number, focalTime = current, focalFraction = 0.5) => {
    const nextLevel = Math.max(1, Math.min(PRECISION_SPANS.length, level));
    const nextSpan = Math.min(vodDuration, PRECISION_SPANS[nextLevel - 1]);
    const nextAnchor = focalTime - (focalFraction - 0.5) * nextSpan;
    setTimelineZoom(nextLevel);
    setPrecisionAnchor(clampPrecisionAnchor(nextAnchor, nextSpan));
  };

  const centerPrecisionOnPlayhead = () => setPrecisionAnchor(clampPrecisionAnchor(current));

  const fitPrecisionSelection = () => {
    if (inPoint == null || outPoint == null) return;
    const targetSpan = Math.min(vodDuration, Math.max(8, outPoint - inPoint + 8));
    let fitLevel = 1;
    PRECISION_SPANS.forEach((span, index) => { if (span >= targetSpan) fitLevel = index + 1; });
    setPrecisionZoom(fitLevel, (inPoint + outPoint) / 2);
  };

  const onPrecisionWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    stopRangePreview();
    const dominantDelta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
    if (event.ctrlKey || event.metaKey) {
      const rect = detailRef.current?.getBoundingClientRect();
      const fraction = rect?.width ? Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) : 0.5;
      const focalTime = detailStart + fraction * detailDuration;
      setPrecisionZoom(timelineZoom + (dominantDelta > 0 ? -1 : 1), focalTime, fraction);
      return;
    }
    panPrecision(dominantDelta / 480 * detailDuration);
  };

  const nudgeBoundary = (bound: "in" | "out", amount: number) => {
    invalidateMediaProof();
    stopRangePreview();
    if (bound === "in" && inPoint != null) {
      const next = Math.max(0, Math.min(inPoint + amount, (outPoint ?? vodDuration) - 0.5));
      setInPoint(next);
      seekTo(next);
    }
    if (bound === "out" && outPoint != null) {
      const next = Math.min(vodDuration, Math.max(outPoint + amount, (inPoint ?? 0) + 0.5));
      setOutPoint(next);
      seekTo(next);
    }
  };

  const onBoundaryKeyDown = (bound: "in" | "out", event: ReactKeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    event.stopPropagation();
    const step = event.altKey ? 0.1 : event.shiftKey ? 5 : 0.5;
    nudgeBoundary(bound, event.key === "ArrowLeft" ? -step : step);
  };

  const beginTrackDrag = (event: ReactPointerEvent<HTMLDivElement>, detail = false) => {
    if (sourceState !== "ready") return;
    event.currentTarget.setPointerCapture(event.pointerId);
    stopRangePreview();
    setShowOutput(false);
    beginScrub();
    setDragging(detail ? "detail" : "overview");
    const time = timeFromPointer(event.clientX, detail);
    if (detail) {
      detailGestureRef.current = {
        startX: event.clientX,
        startTime: time,
        selecting: false,
        // Shift+drag (or Alt+drag) creates a range; plain drag only scrubs.
        rangeSelect: event.shiftKey || event.altKey,
      };
      setPrecisionHoverTime(time);
    }
    seekTo(time, { scrubbing: true });
    if (!detail) setPrecisionAnchor(time);
  };

  const moveTrack = (event: ReactPointerEvent<HTMLDivElement>, detail = false) => {
    const time = timeFromPointer(event.clientX, detail);
    if (detail) setPrecisionHoverTime(time);
    else setOverviewHoverTime(time);
    if (!dragging) return;
    const isDetail = dragging.startsWith("detail");
    if (detail !== isDetail) return;
    if (dragging === "overview") {
      seekTo(time, { scrubbing: true });
      setPrecisionAnchor(time);
    } else if (dragging === "detail") {
      const gesture = detailGestureRef.current;
      if (
        gesture?.rangeSelect
        && (gesture.selecting || Math.abs(event.clientX - gesture.startX) >= RANGE_SELECT_PX)
      ) {
        gesture.selecting = true;
        invalidateMediaProof();
        const start = Math.max(0, Math.min(gesture.startTime, time));
        const end = Math.min(vodDuration, Math.max(gesture.startTime, time));
        setInPoint(start);
        setOutPoint(Math.min(vodDuration, Math.max(start + 0.5, end)));
      }
      seekTo(time, { scrubbing: true });
    } else if (dragging === "in" || dragging === "detail-in") {
      invalidateMediaProof();
      setInPoint(Math.max(0, Math.min(time, (outPoint ?? vodDuration) - 0.5)));
      seekTo(time, { scrubbing: true });
    } else {
      invalidateMediaProof();
      setOutPoint(Math.min(vodDuration, Math.max(time, (inPoint ?? 0) + 0.5)));
      seekTo(time, { scrubbing: true });
    }
  };

  const finishTrackDrag = () => {
    if (dragging === "overview" || dragging === "detail") setPrecisionAnchor(current);
    detailGestureRef.current = null;
    setDragging(null);
    endScrub();
  };

  const setBoundFromInput = (bound: "in" | "out", value: string) => {
    const parsed = parseTime(value);
    if (parsed == null) return;
    invalidateMediaProof();
    if (bound === "in") {
      const next = Math.max(0, Math.min(parsed, (outPoint ?? vodDuration) - 0.5));
      setInPoint(next);
      setPrecisionAnchor(outPoint == null ? next : (next + outPoint) / 2);
    } else {
      const next = Math.min(vodDuration, Math.max(parsed, (inPoint ?? 0) + 0.5));
      setOutPoint(next);
      setPrecisionAnchor(inPoint == null ? next : (inPoint + next) / 2);
    }
  };

  const updateLayout = (value: ManualClipLayout) => {
    invalidateMediaProof();
    setLayoutChoice(value);
  };

  const updateFocus = (value: number) => {
    invalidateMediaProof();
    setFocusX(value);
  };

  const updateCaptions = (value: boolean) => {
    invalidateMediaProof();
    setCaptionsEnabled(value);
  };

  const updateFade = (key: keyof FadeState, value: number) => {
    invalidateMediaProof();
    setFades((currentFades) => ({ ...currentFades, [key]: value }));
  };

  if (!session) {
    return <div className="clips-empty"><StudioEmptyState variant="editor" icon={<Scissors />} title="Open a session before cutting" body="The VOD Editor needs a completed source recording and its reaction timeline." action={<button type="button" className="btn-secondary" onClick={onBack}>Back to review</button>} /></div>;
  }

  if (!scanFinished) {
    return <div className="clips-empty"><StudioEmptyState variant="editor" icon={<Scissors />} title="The editor is waiting on this scan" body="Recall will unlock the full recording and its reaction timeline when analysis finishes." action={<button type="button" className="btn-secondary" onClick={onBack}>Back to session</button>} /></div>;
  }

  const stateLabel = creating ? "Rendering final clip" : createdClip ? "Added to Clip Library" : selectionReady ? "Selection ready" : "Choose a range";
  const stateClass = selectionReady || createdClip ? "is-ready" : "";
  const layoutOptions: { value: ManualClipLayout; label: string; note: string }[] = [
    { value: "auto", label: "Auto", note: "Recall decides" },
    { value: "vertical_split", label: "Stacked", note: "Cam + game" },
    { value: "gameplay_pip", label: "PiP", note: "Game first" },
    { value: "full_gameplay", label: "Gameplay", note: "No facecam" },
  ];

  return (
    <section className={`vod-editor ${settingsOpen ? "has-settings-open" : ""}`} aria-label={`Edit full VOD for ${session.name}`}>
      <header className="vod-editor-topbar">
        <button type="button" className="library-editor-back" onClick={onBack} disabled={creating}><ArrowLeft size={16} /> {initialMoment ? "Back to Memory" : "Back to Review"}</button>
        <div className="vod-editor-context"><strong>New clip</strong><span>from {session.name}</span></div>
        <div className="vod-editor-top-actions">
          <span className={`library-editor-state ${stateClass}`}><i />{stateLabel}</span>
          <button type="button" className={`btn-secondary vod-editor-settings-btn ${settingsOpen ? "is-active" : ""}`} aria-expanded={settingsOpen} aria-controls="vod-editor-output-settings" onClick={() => setSettingsOpen((value) => !value)}><Settings size={14} /> Output settings</button>
          <button type="button" className="btn-secondary vod-editor-preview-btn" onClick={() => void generateOutputPreview()} disabled={!selectionReady || renderingPreview || creating}>{renderingPreview ? <><span className="studio-spinner" />Rendering…</> : <><Play size={14} />Preview</>}</button>
          <button type="button" className="cta-accent vod-editor-create" onClick={() => void createClip()} disabled={!selectionReady || creating}>{creating ? <><span className="studio-spinner" />Rendering…</> : <><Scissors size={14} />Create clip</>}</button>
        </div>
      </header>

      <div className="vod-editor-workspace">
        <section className="vod-editor-stage">
          <div className={`vod-editor-video-wrap ${showOutput ? "is-output" : "is-source"}`}>
            {showOutput && outputUrl ? (
              <video ref={outputVideoRef} src={outputUrl} muted={mediaVolume.muted} playsInline autoPlay onPlay={onSourcePlay} onPause={onSourcePause} onEnded={onSourcePause} onTimeUpdate={(event) => setOutputCurrent(event.currentTarget.currentTime || 0)} />
            ) : sourceState === "missing" ? (
              <div className="library-editor-video-empty vod-editor-source-restore" role="alert">
                <Film size={28} />
                <strong>{restoringSource ? "Restoring source" : "Source recording unavailable"}</strong>
                <span>
                  {restoringSource
                    ? "Recall is downloading the original Twitch VOD. Your deck and review proxy stay available while this finishes."
                    : sourceStatus?.restore_available
                      ? "The downloaded source was removed. Existing clips are safe; restore it before trimming or final rendering."
                      : "The source is not on disk. Existing rendered clips are unaffected."}
                </span>
                {sourceStatus?.restore_available && !restoringSource && (
                  <button type="button" className="cta-accent" onClick={() => void restoreSource()}>
                    <Download size={14} /> Restore source
                  </button>
                )}
                {restoringSource && (
                  <button type="button" className="btn-secondary" onClick={() => void cancelSourceRestore()}>
                    Cancel restoration
                  </button>
                )}
              </div>
            ) : (
              <>
                <video ref={sourceVideoRef} src={sourceUrl} muted={mediaVolume.muted} playsInline preload="metadata" onClick={togglePlay} onLoadedMetadata={(event) => { event.currentTarget.volume = mediaVolume.volume; const duration = event.currentTarget.duration; if (Number.isFinite(duration) && duration > 0) setMediaDuration(duration); setSourceState("ready"); measurePicture(); }} onError={() => setSourceState("missing")} onPlay={onSourcePlay} onPause={onSourcePause} onEnded={onSourcePause} onTimeUpdate={onSourceTimeUpdate} />
                {sourceState === "loading" && <div className="cutr-stage-loading"><span className="studio-spinner" />Opening the source recording…</div>}
                {sourceState === "ready" && framingState === "ready" && cropOverlay && (
                  <div className="vod-editor-crop-overlay" aria-hidden="true" style={pictureBox ?? undefined}>
                    {cropOverlay.regions.map((region) => <div key={region.key} className={`vod-crop-region is-${region.key}`} style={{ left: `${region.rect[0] * 100}%`, top: `${region.rect[1] * 100}%`, width: `${region.rect[2] * 100}%`, height: `${region.rect[3] * 100}%` }}><span>{region.label}</span></div>)}
                    {cropOverlay.band && <div className="vod-crop-band" style={{ left: `${cropOverlay.band.left * 100}%`, top: `${cropOverlay.band.top * 100}%`, width: `${cropOverlay.band.width * 100}%`, height: `${cropOverlay.band.height * 100}%` }}><span>9:16</span></div>}
                  </div>
                )}
              </>
            )}
            {outputUrl && <button type="button" className="vod-editor-view-toggle" onClick={() => { sourceVideoRef.current?.pause(); outputVideoRef.current?.pause(); setShowOutput((value) => !value); }}>{showOutput ? "View source" : "View vertical proof"}</button>}
          </div>

          <div className="vod-editor-transport">
            <div className="vod-editor-transport-left">
              <button type="button" className="vod-editor-play" onClick={togglePlay} disabled={sourceState !== "ready" && !showOutput} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={18} /> : <Play size={18} fill="currentColor" />}</button>
              <button type="button" onClick={() => { const next = current - 5; seekTo(next); setPrecisionAnchor(Math.max(0, next)); }} disabled={showOutput || sourceState !== "ready"} aria-label="Back 5 seconds">−5</button>
              <button type="button" onClick={() => { const next = current + 5; seekTo(next); setPrecisionAnchor(Math.min(vodDuration, next)); }} disabled={showOutput || sourceState !== "ready"} aria-label="Forward 5 seconds">+5</button>
              <span className="vod-editor-clock">{showOutput ? preciseClock(Math.min(outputCurrent, selectionDuration ?? 0)) : preciseClock(current)} <i>/</i> {showOutput ? preciseClock(selectionDuration ?? 0) : preciseClock(vodDuration)}</span>
            </div>
            <div className="vod-editor-markers">
              <button type="button" className="btn-secondary" onClick={markIn} disabled={sourceState !== "ready"}>In <kbd>I</kbd></button>
              <button type="button" className="btn-secondary" onClick={markOut} disabled={sourceState !== "ready"}>Out <kbd>O</kbd></button>
              <button type="button" className="btn-secondary vod-editor-clear-selection" onClick={clearSelection} disabled={inPoint == null && outPoint == null} title="Clear selection">Clear</button>
              <button type="button" className={`btn-secondary ${previewingRange ? "is-active" : ""}`} onClick={previewRange} disabled={inPoint == null || outPoint == null} title="Loop selection"><Repeat size={14} /></button>
            </div>
            <div className="vod-editor-transport-right">
              <MediaVolumeControl className="vod-editor-volume" volume={mediaVolume.volume} muted={mediaVolume.muted} onVolumeChange={mediaVolume.setVolume} onToggleMuted={mediaVolume.toggleMuted} />
              <strong className="vod-editor-sel-dur">{selectionDuration == null ? "—" : preciseClock(selectionDuration)}</strong>
            </div>
          </div>
        </section>

        {settingsOpen && (
          <aside id="vod-editor-output-settings" className="library-editor-inspector vod-editor-settings-drawer" aria-label="Output settings">
            <header><div><Settings size={15} /><strong>Output settings</strong></div><button type="button" onClick={() => setSettingsOpen(false)} aria-label="Close output settings"><X size={15} /></button></header>
            <section>
              <h2><Film size={15} /> Clip details</h2>
              <label><span>Title</span><input value={title} maxLength={120} onChange={(event) => setTitle(event.target.value)} /></label>
              <label><span>Tags</span><input value={tags} placeholder="Funny, clutch, reaction" onChange={(event) => setTags(event.target.value)} /></label>
              <div className="library-editor-timefields vod-editor-timefields">
                <label>Start <input value={inPoint == null ? "—" : preciseClock(inPoint)} onChange={(event) => setBoundFromInput("in", event.target.value)} /></label>
                <label>End <input value={outPoint == null ? "—" : preciseClock(outPoint)} onChange={(event) => setBoundFromInput("out", event.target.value)} /></label>
              </div>
            </section>
            <section className="vod-editor-framing">
              <h2><Target size={15} /> Framing</h2>
              <div className="vod-editor-layout-grid">{layoutOptions.map((option) => <button type="button" key={option.value} className={layoutChoice === option.value ? "is-active" : ""} onClick={() => updateLayout(option.value)}><strong>{option.label}</strong><span>{option.note}</span></button>)}</div>
              <div className={`vod-editor-framing-status is-${framingState}`}>{framingState === "loading" ? <><span className="studio-spinner" /><span>Matching this range to Recall’s framing map…</span></> : framing ? <><Sparkles size={14} /><span><strong>{framing.label}</strong>{framing.reason}</span></> : <><Sparkles size={14} /><span>Set an in and out point to see Recall’s proposed composition.</span></>}</div>
              <div className="vod-editor-focus"><span><b>Gameplay focus</b><i>{focusX < 0.4 ? "Left" : focusX > 0.6 ? "Right" : "Center"}</i></span><StudioSlider value={focusX} min={0} max={1} step={0.01} onChange={updateFocus} label="Gameplay focus" ticks={["Left", "Center", "Right"]} /></div>
            </section>
            <details className="vod-editor-tool-section"><summary><span><Sparkles size={15} /> Captions</span><small>{captionsEnabled ? "On" : "Off"}</small></summary><div className="vod-editor-tool-body"><div className="vod-editor-switch"><span><b>Burn in captions</b><small>{settings.captionStyle.font} · {settings.captionStyle.size} · {settings.captionStyle.position}</small></span><StudioSwitch checked={captionsEnabled} onChange={updateCaptions} label="Burn in captions" /></div></div></details>
            <details className="vod-editor-tool-section"><summary><span><Film size={15} /> Video fades</span><small>{fades.videoIn || fades.videoOut ? "Adjusted" : "None"}</small></summary><div className="vod-editor-tool-body"><FadeControl label="Fade in" value={fades.videoIn} max={fadeMax} onChange={(value) => updateFade("videoIn", value)} /><FadeControl label="Fade out" value={fades.videoOut} max={fadeMax} onChange={(value) => updateFade("videoOut", value)} /></div></details>
            <details className="vod-editor-tool-section"><summary><span><Volume2 size={15} /> Audio fades</span><small>{fades.audioIn || fades.audioOut ? "Adjusted" : "None"}</small></summary><div className="vod-editor-tool-body"><FadeControl label="Fade in" value={fades.audioIn} max={fadeMax} onChange={(value) => updateFade("audioIn", value)} /><FadeControl label="Fade out" value={fades.audioOut} max={fadeMax} onChange={(value) => updateFade("audioOut", value)} /></div></details>
            <footer><button type="button" className="btn-secondary" onClick={resetDraft} disabled={creating}><RotateCcw size={14} /> Reset draft</button></footer>
          </aside>
        )}
      </div>

      <section className="vod-editor-timeline">
        <div className="vod-editor-timeline-toolbar">
          <div className="vod-editor-track-tools">
            <button type="button" onClick={() => panPrecision(-detailDuration * 0.5)} aria-label="Pan backward">‹</button>
            <button type="button" className={!detailPlayVisible ? "is-attention" : ""} onClick={centerPrecisionOnPlayhead} title="Center on playhead">Playhead</button>
            <button type="button" onClick={fitPrecisionSelection} disabled={inPoint == null || outPoint == null}>Fit</button>
            <button type="button" onClick={() => panPrecision(detailDuration * 0.5)} aria-label="Pan forward">›</button>
          </div>
          <div className="vod-editor-zoom" aria-label="Timeline zoom">
            <button type="button" onClick={() => setPrecisionZoom(timelineZoom - 1)} disabled={timelineZoom === 1} aria-label="Zoom out">−</button>
            <input type="range" min={1} max={PRECISION_SPANS.length} step={1} value={timelineZoom} onChange={(event) => setPrecisionZoom(Number(event.target.value))} aria-label="Timeline zoom level" />
            <button type="button" onClick={() => setPrecisionZoom(timelineZoom + 1)} disabled={timelineZoom === PRECISION_SPANS.length} aria-label="Zoom in">+</button>
          </div>
        </div>

        <div className="vod-editor-nle">
          <div className="vod-editor-ruler-row is-precision" aria-hidden="true">
            <div className="vod-editor-track-gutter" />
            <div className="vod-editor-ruler-rail">
              {detailTicks.map((tick) => (
                <span
                  key={tick.seconds}
                  className={tick.position < 3 ? "is-start" : tick.position > 97 ? "is-end" : ""}
                  style={{ left: `${tick.position}%` }}
                >
                  {tickClock(tick.seconds, detailStep)}
                </span>
              ))}
              {detailPlayVisible && (
                <div className="vod-editor-playhead-cap" style={{ left: `${detailPlayPct}%` }} />
              )}
            </div>
          </div>

          <div className="vod-editor-track-row is-source">
            <div className="vod-editor-track-gutter"><span>Precision</span></div>
            <div
              ref={detailRef}
              className={`cutr-track vod-editor-detail ${dragging === "detail" ? "is-selecting" : ""}`}
              style={{ "--playhead": `${detailPlayPct}%`, "--sel-in": `${detailInPct ?? 0}%`, "--sel-out": `${detailOutPct ?? 0}%` } as CSSProperties}
              onPointerDown={(event) => beginTrackDrag(event, true)}
              onPointerMove={(event) => moveTrack(event, true)}
              onPointerUp={finishTrackDrag}
              onPointerCancel={finishTrackDrag}
              onPointerLeave={() => { if (!dragging) setPrecisionHoverTime(null); }}
              onWheel={onPrecisionWheel}
            >
              <FilmstripImage src={detailFilmstripUrl} label={`Source frames from ${preciseClock(detailStart)} to ${preciseClock(detailEnd)}`} />
              <div className="vod-editor-clip-ranges" aria-label="Existing clips in this window">
                {autoClips.map((clip) => {
                  const style = rangeStyle(clip.start_time ?? 0, clip.end_time ?? 0, detailStart, detailEnd);
                  return style && <span key={`detail-auto-${clip.id}`} className="vod-editor-clip-range is-auto" style={style} title={`Auto clip: ${clip.title}`} />;
                })}
                {manualClips.map((clip) => {
                  const style = rangeStyle(clip.start_time ?? 0, clip.end_time ?? 0, detailStart, detailEnd);
                  return style && <span key={`detail-manual-${clip.id}`} className="vod-editor-clip-range is-manual" style={style} title={`VOD editor clip: ${clip.title}`} />;
                })}
              </div>
              {detailCurve && (
                <svg viewBox={`0 0 1000 ${CURVE_HEIGHT}`} preserveAspectRatio="none" aria-hidden="true">
                  <path className="vod-editor-reaction-bed" d={detailCurve} fill="none" vectorEffect="non-scaling-stroke" />
                  <path d={detailCurve} fill="none" stroke="var(--accent)" strokeWidth="1.25" vectorEffect="non-scaling-stroke" />
                </svg>
              )}
              {detailInPct != null && detailOutPct != null && (
                <div className="vod-editor-clip-block">
                  <span className="vod-editor-clip-block-label">{selectionDuration != null ? preciseClock(selectionDuration) : "Clip"}</span>
                </div>
              )}
              {detailPlayVisible && <div className="cutr-playhead" />}
              {precisionHoverPct != null && dragging !== "detail-in" && dragging !== "detail-out" && (
                <div className={`vod-editor-hover-time ${precisionHoverPct < 6 ? "is-near-start" : precisionHoverPct > 94 ? "is-near-end" : ""}`} style={{ left: `${precisionHoverPct}%` }}>{preciseClock(precisionHoverTime ?? 0)}</div>
              )}
              {detailInVisible && (
                <button
                  type="button"
                  className="cutr-handle is-in"
                  style={{ left: `${detailInPct}%` }}
                  aria-label={`Drag in point at ${preciseClock(inPoint ?? 0)}`}
                  onKeyDown={(event) => onBoundaryKeyDown("in", event)}
                  onPointerDown={(event) => {
                    event.stopPropagation();
                    event.currentTarget.setPointerCapture(event.pointerId);
                    detailGestureRef.current = null;
                    stopRangePreview();
                    beginScrub();
                    setDragging("detail-in");
                  }}
                >
                  <i /><i />
                </button>
              )}
              {detailOutVisible && (
                <button
                  type="button"
                  className="cutr-handle is-out"
                  style={{ left: `${detailOutPct}%` }}
                  aria-label={`Drag out point at ${preciseClock(outPoint ?? 0)}`}
                  onKeyDown={(event) => onBoundaryKeyDown("out", event)}
                  onPointerDown={(event) => {
                    event.stopPropagation();
                    event.currentTarget.setPointerCapture(event.pointerId);
                    detailGestureRef.current = null;
                    stopRangePreview();
                    beginScrub();
                    setDragging("detail-out");
                  }}
                >
                  <i /><i />
                </button>
              )}
            </div>
          </div>

          {/* The VOD's own ruler, stepped from its actual length so the labels
              mean the same thing on a 20-minute recording and a five-hour one. */}
          <div className="vod-editor-ruler-row is-map" aria-hidden="true">
            <div className="vod-editor-track-gutter" />
            <div className="vod-editor-ruler-rail">
              {overviewTicks.map((tick) => (
                <span
                  key={tick.seconds}
                  className={tick.position < 3 ? "is-start" : tick.position > 97 ? "is-end" : ""}
                  style={{ left: `${tick.position}%` }}
                >
                  {tickClock(tick.seconds, overviewStep)}
                </span>
              ))}
              <span className="is-end" style={{ left: "100%" }}>{tickClock(vodDuration, overviewStep)}</span>
              <div className="vod-editor-playhead-cap" style={{ left: `${playheadPct}%` }} />
            </div>
          </div>

          <div className="vod-editor-track-row is-map">
            <div className="vod-editor-track-gutter"><span>VOD</span></div>
            <div
              ref={overviewRef}
              className="cutr-track vod-editor-overview"
              style={{ "--playhead": `${playheadPct}%`, "--sel-in": `${inPct ?? 0}%`, "--sel-out": `${outPct ?? 0}%` } as CSSProperties}
              onPointerDown={(event) => beginTrackDrag(event, false)}
              onPointerMove={(event) => moveTrack(event, false)}
              onPointerUp={finishTrackDrag}
              onPointerCancel={finishTrackDrag}
              onPointerLeave={() => setOverviewHoverTime(null)}
            >
              <FilmstripImage src={overviewFilmstripUrl} label="Full VOD overview" />
              <div className="vod-editor-clip-ranges">
                {autoClips.map((clip) => {
                  const style = rangeStyle(clip.start_time ?? 0, clip.end_time ?? 0, 0, vodDuration);
                  return style && <span key={`overview-auto-${clip.id}`} className="vod-editor-clip-range is-auto" style={style} />;
                })}
                {manualClips.map((clip) => {
                  const style = rangeStyle(clip.start_time ?? 0, clip.end_time ?? 0, 0, vodDuration);
                  return style && <span key={`overview-manual-${clip.id}`} className="vod-editor-clip-range is-manual" style={style} />;
                })}
              </div>
              <div className="vod-editor-precision-window" style={detailWindowStyle} aria-hidden="true" />
              {overviewCurve && (
                <svg viewBox={`0 0 1000 ${CURVE_HEIGHT}`} preserveAspectRatio="none" aria-hidden="true">
                  <path d={overviewCurve} fill="none" stroke="var(--accent)" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
                </svg>
              )}
              {inPct != null && outPct != null && <div className="cutr-selection" />}
              <div className="cutr-playhead" />
              {overviewHoverPct != null && (
                <div
                  className={`vod-editor-hover-time ${overviewHoverPct < 6 ? "is-near-start" : overviewHoverPct > 94 ? "is-near-end" : ""}`}
                  style={{ left: `${overviewHoverPct}%` }}
                >
                  {preciseClock(overviewHoverTime ?? 0)}
                </div>
              )}
            </div>
          </div>
        </div>

        {tooLong && <div className="library-editor-error" role="alert">Clips are capped at 3 minutes. Tighten this range by {Math.ceil((selectionDuration ?? 0) - MAX_CLIP_SECONDS)} seconds.</div>}
        {tooShort && <div className="library-editor-error" role="alert">Give the moment at least one second.</div>}
        {error && <div className="library-editor-error" role="alert">{error}</div>}
        {createdClip && <div className="vod-editor-success"><Check size={15} /><span><strong>{createdClip.title}</strong> is ready in Clip Library.</span><button type="button" onClick={onOpenLibrary}>Open Clip Library</button></div>}
      </section>
    </section>
  );
}

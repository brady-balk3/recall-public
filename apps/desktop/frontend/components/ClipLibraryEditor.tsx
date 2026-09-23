// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { releaseClipPreview } from "../lib/releaseClipPreview";
import {
  AlertTriangle,
  ArrowLeft,
  Captions,
  Check,
  Copy,
  Download,
  Film,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Save,
  Target,
  Volume2,
} from "../lib/icons";
import { clueLabel, postText } from "../lib/copy";
import { apiFetch, apiUrl, mediaUrl } from "../lib/api";
import {
  fetchReactionTimeline,
  useJobStore,
  type Clip,
  type ReactionTimeline,
  type Session,
} from "../lib/store";
import { MediaVolumeControl, StateIcon, StudioSlider, useMediaVolume } from "./StudioControls";

type FadeState = {
  videoIn: number;
  videoOut: number;
  audioIn: number;
  audioOut: number;
};

type Props = {
  session: Session;
  clip: Clip;
  onClose: () => void;
  onSave: (changes: {
    title: string;
    tags: string;
    start: number;
    end: number;
    fades: FadeState;
    mediaDirty: boolean;
  }) => Promise<void>;
  onExport: () => Promise<void>;
};

const metaKey = (sessionId: string) => `recall-draft-meta-${sessionId}`;
/** Seconds of VOD context on each side of the current selection. */
const CONTEXT_PAD = 60;
/** Matches backend EditClipRequest max length. */
const MAX_CLIP_DURATION = 180;

function readTags(sessionId: string, clip: Clip) {
  try {
    const raw = localStorage.getItem(metaKey(sessionId));
    const saved = raw ? JSON.parse(raw)?.[clip.id]?.tags : "";
    if (typeof saved === "string" && saved.trim()) return saved;
  } catch {}
  return (clip.postTags?.length ? clip.postTags : clip.signals?.map(clueLabel) ?? []).join(", ");
}

function readFades(sessionId: string, clipId: string): FadeState {
  const empty = { videoIn: 0, videoOut: 0, audioIn: 0, audioOut: 0 };
  try {
    const raw = localStorage.getItem(metaKey(sessionId));
    const saved = raw ? JSON.parse(raw)?.[clipId]?.fades : null;
    if (!saved) return empty;
    return {
      videoIn: Number(saved.videoIn) || 0,
      videoOut: Number(saved.videoOut) || 0,
      audioIn: Number(saved.audioIn) || 0,
      audioOut: Number(saved.audioOut) || 0,
    };
  } catch { return empty; }
}

/** One burned-in word and the window it is on screen for, clip-relative. */
type CaptionWord = { word: string; start: number; end: number };

type CaptionState = {
  /** "edit" = a stored correction, "machine" = the decode, "none" = not read yet. */
  source: "edit" | "machine" | "none";
  text: string;
  words: CaptionWord[];
  machineText: string;
  stale: boolean;
  heldEditText?: string;
};

/** The API's word list, tolerant of the older shape that sent text only. */
function readWords(raw: unknown): CaptionWord[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((entry) => {
      const word = String((entry as { word?: unknown })?.word ?? "").trim();
      const start = Number((entry as { start?: unknown })?.start);
      const end = Number((entry as { end?: unknown })?.end);
      return { word, start: Number.isFinite(start) ? start : 0, end: Number.isFinite(end) ? end : 0 };
    })
    .filter((word) => word.word.length > 0);
}

function wordsToText(words: CaptionWord[]) {
  return words.map((word) => word.word).join(" ");
}

/** Replace one word with what the creator typed, keeping every other word's
 * timing untouched. Typing several words into one chip splits that chip's span
 * by character length, and clearing it deletes the word -- the same two rules
 * the engine applies when it re-times the saved text (caption_edits.py), so the
 * rail on screen matches what the save will produce. */
function replaceWordAt(words: CaptionWord[], index: number, text: string): CaptionWord[] {
  const tokens = text.trim().split(/\s+/).filter(Boolean);
  const target = words[index];
  if (!target) return words;
  const before = words.slice(0, index);
  const after = words.slice(index + 1);
  if (!tokens.length) return [...before, ...after];
  if (tokens.length === 1) return [...before, { ...target, word: tokens[0] }, ...after];
  const span = Math.max(0, target.end - target.start);
  const weight = tokens.reduce((sum, token) => sum + token.length, 0) || tokens.length;
  let cursor = target.start;
  const split = tokens.map((token, position) => {
    const share = span * (token.length / weight);
    const start = cursor;
    cursor = position === tokens.length - 1 ? target.end : start + share;
    return { word: token, start, end: cursor };
  });
  return [...before, ...split, ...after];
}

/** Seconds a word keeps the cursor after its own window closes. Word timings
 * are tight and a spoken phrase arrives with small gaps between its words,
 * while the burned-in line stays up across them -- so without a hold the rail
 * would blink dark between every word. A gap longer than this is real silence
 * and nothing is lit. */
const CAPTION_HOLD = 1.5;

/** Index of the word on screen at `clipTime` (clip-relative seconds), or -1. */
function liveWordIndex(words: CaptionWord[], clipTime: number) {
  let held = -1;
  for (let index = 0; index < words.length; index += 1) {
    const word = words[index];
    if (clipTime < word.start) break;
    if (clipTime < word.end) return index;
    const next = words[index + 1];
    const holdUntil = Math.min(word.end + CAPTION_HOLD, next ? next.start : Infinity);
    held = clipTime < holdUntil ? index : -1;
  }
  return held;
}

function wordClock(seconds: number) {
  const safe = Math.max(0, seconds);
  return `${Math.floor(safe / 60)}:${(safe % 60).toFixed(1).padStart(4, "0")}`;
}

async function loadCaption(clipId: string, transcribe: boolean): Promise<CaptionState> {
  const response = await apiFetch(
    `/clips/${clipId}/caption${transcribe ? "?transcribe=true" : ""}`);
  if (!response.ok) throw new Error("Recall could not read this clip's captions.");
  const data = await response.json();
  return {
    source: data.source, text: data.text ?? "", words: readWords(data.words),
    machineText: data.machine_text ?? "",
    stale: Boolean(data.stale), heldEditText: data.held_edit_text,
  };
}

async function putCaption(clipId: string, text: string) {
  const response = await apiFetch(`/clips/${clipId}/caption`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) throw new Error("Recall could not save this caption.");
  return response.json();
}

function preciseTime(seconds: number) {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const remainder = (safe % 60).toFixed(2).padStart(5, "0");
  return `${minutes.toString().padStart(2, "0")}:${remainder}`;
}

function parseTime(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parts = trimmed.split(":").map((part) => Number(part));
  if (!parts.length || parts.some((part) => !Number.isFinite(part) || part < 0)) return null;
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return parts[0];
}

/** Peak envelope for the visible trim window from the VOD reaction curve. */
function peaksFromTimeline(
  timeline: ReactionTimeline,
  viewStart: number,
  viewEnd: number,
  bins = 128,
): number[] | null {
  if (!timeline.t?.length || !timeline.r?.length || viewEnd <= viewStart) return null;
  const span = viewEnd - viewStart;
  const peaks = Array.from({ length: bins }, (_, index) => {
    const t0 = viewStart + (index / bins) * span;
    const t1 = viewStart + ((index + 1) / bins) * span;
    let peak = 0;
    for (let i = 0; i < timeline.t.length; i += 1) {
      const time = timeline.t[i];
      if (time >= t0 && time < t1) peak = Math.max(peak, timeline.r[i] ?? 0);
    }
    return peak;
  });
  const maxPeak = Math.max(0.001, ...peaks);
  return peaks.map((peak) => peak / maxPeak);
}

/** Map a clip-file waveform onto the absolute VOD view (quiet in the ±pad). */
function mapClipPeaksToView(
  clipPeaks: number[],
  clipAbsStart: number,
  clipAbsEnd: number,
  viewStart: number,
  viewEnd: number,
  bins = 128,
): number[] {
  const clipSpan = Math.max(0.001, clipAbsEnd - clipAbsStart);
  const viewSpan = Math.max(0.001, viewEnd - viewStart);
  return Array.from({ length: bins }, (_, index) => {
    const mid = viewStart + ((index + 0.5) / bins) * viewSpan;
    if (mid < clipAbsStart || mid > clipAbsEnd) return 0.04;
    const frac = (mid - clipAbsStart) / clipSpan;
    const sample = clipPeaks[Math.min(clipPeaks.length - 1, Math.max(0, Math.floor(frac * clipPeaks.length)))] ?? 0;
    return Math.max(0.04, sample);
  });
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

/** Draft-friendly time field so typing isn't overwritten by preciseTime() each keystroke. */
function TimeField({
  label,
  seconds,
  min,
  max,
  onCommit,
  readOnly = false,
}: {
  label: string;
  seconds: number;
  min: number;
  max: number;
  onCommit: (seconds: number) => void;
  readOnly?: boolean;
}) {
  const [text, setText] = useState(() => preciseTime(seconds));
  const focusedRef = useRef(false);

  useEffect(() => {
    if (!focusedRef.current) setText(preciseTime(seconds));
  }, [seconds]);

  const commit = () => {
    focusedRef.current = false;
    const parsed = parseTime(text);
    if (parsed == null) {
      setText(preciseTime(seconds));
      return;
    }
    const next = Math.max(min, Math.min(max, parsed));
    onCommit(next);
    setText(preciseTime(next));
  };

  return (
    <label>
      {label}{" "}
      <input
        value={text}
        readOnly={readOnly}
        onFocus={() => { focusedRef.current = true; }}
        onChange={(event) => setText(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            (event.currentTarget as HTMLInputElement).blur();
          }
        }}
      />
    </label>
  );
}

export default function ClipLibraryEditor({ session, clip, onClose, onSave, onExport }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const apiEndpoint = useJobStore((state) => state.settings.apiEndpoint);
  const sourceUrl = apiUrl(`/jobs/${session.id}/source`, apiEndpoint);
  const mediaVolume = useMediaVolume([videoRef], clip.id);
  const startOriginal = clip.start_time ?? 0;
  const endOriginal = clip.end_time ?? startOriginal + Math.max(1, clip.duration || 1);
  const initialTags = useMemo(() => readTags(session.id, clip), [session.id, clip]);
  const initialFades = useMemo(() => readFades(session.id, clip.id), [session.id, clip.id]);
  const [title, setTitle] = useState(clip.title);
  const [tags, setTags] = useState(initialTags);
  const [start, setStart] = useState(startOriginal);
  const [end, setEnd] = useState(endOriginal);
  const [fades, setFades] = useState<FadeState>(initialFades);
  const [savedSnapshot, setSavedSnapshot] = useState({ title: clip.title, tags: initialTags, start: startOriginal, end: endOriginal, fades: initialFades });
  const [dragging, setDragging] = useState<"start" | "end" | null>(null);
  const [playing, setPlaying] = useState(false);
  const [started, setStarted] = useState(false);
  const [previewTime, setPreviewTime] = useState(0);
  const [audioPeaks, setAudioPeaks] = useState<number[] | null>(null);
  const [clipAudioPeaks, setClipAudioPeaks] = useState<number[] | null>(null);
  const [reactionTimeline, setReactionTimeline] = useState<ReactionTimeline | null>(null);
  const [timelineStatus, setTimelineStatus] = useState<"loading" | "ready" | "missing">("loading");
  const [waveformState, setWaveformState] = useState<"loading" | "ready" | "unavailable">("loading");
  const [waveformKind, setWaveformKind] = useState<"reaction" | "audio" | null>(null);
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [caption, setCaption] = useState<CaptionState | null>(null);
  const [captionWords, setCaptionWords] = useState<CaptionWord[]>([]);
  const [captionText, setCaptionText] = useState("");
  const [captionSaved, setCaptionSaved] = useState("");
  const [captionBusy, setCaptionBusy] = useState<"reading" | null>(null);
  const [captionError, setCaptionError] = useState<string | null>(null);
  const [captionMode, setCaptionMode] = useState<"words" | "text">("words");
  const [editingWord, setEditingWord] = useState<number | null>(null);
  const [wordDraft, setWordDraft] = useState("");
  const wordRailRef = useRef<HTMLDivElement>(null);

  const [filmstripLoadedUrl, setFilmstripLoadedUrl] = useState<string | null>(null);
  const [filmstripFailedUrl, setFilmstripFailedUrl] = useState<string | null>(null);
  const [buffering, setBuffering] = useState(false);
  const [posterAttempt, setPosterAttempt] = useState<"file" | "generated" | "video">("file");
  const [sourceReady, setSourceReady] = useState(false);
  const [sourceProbeDone, setSourceProbeDone] = useState(false);
  const [filmstripWindow] = useState({ start: Math.max(0, startOriginal - CONTEXT_PAD), end: endOriginal + CONTEXT_PAD });

  // Cheap on open: this returns a stored correction if there is one and
  // nothing otherwise. Reading the machine's own words costs an ASR pass and
  // is only spent when the creator asks for it below.
  useEffect(() => {
    let live = true;
    setCaption(null); setCaptionText(""); setCaptionSaved(""); setCaptionError(null);
    setCaptionWords([]); setEditingWord(null); setCaptionMode("words");
    loadCaption(clip.id, false)
      .then((state) => {
        if (!live) return;
        setCaption(state);
        setCaptionWords(state.words);
        setCaptionText(state.text);
        setCaptionSaved(state.text);
      })
      .catch(() => { if (live) setCaption(null); });
    return () => { live = false; };
  }, [clip.id]);

  const readCaption = async () => {
    setCaptionBusy("reading"); setCaptionError(null);
    try {
      const state = await loadCaption(clip.id, true);
      setCaption(state);
      setCaptionWords(state.words);
      setCaptionText(state.text);
      setCaptionSaved(state.text);
      setEditingWord(null);
    } catch {
      // Deliberately not the thrown message: a failed fetch surfaces as
      // "Failed to fetch", which tells the creator nothing and no next step.
      setCaptionError("Recall could not read this clip's captions. Check that the Recall engine is running, then try again.");
    } finally {
      setCaptionBusy(null);
    }
  };

  const duration = Math.max(1, end - start);
  const fadeMax = Math.max(0, Math.min(5, duration / 2));
  const vodDuration = Math.max(
    session.vodDuration || 0,
    endOriginal + CONTEXT_PAD,
    end + CONTEXT_PAD,
    start + duration,
    1,
  );
  // This rail is a fixed coordinate system for the edit. Reframing it from the
  // changing selection makes the untouched edge appear to move too.
  const viewStart = Math.max(0, Math.min(filmstripWindow.start, vodDuration - 1));
  const viewEnd = Math.min(vodDuration, Math.max(filmstripWindow.end, viewStart + 1));
  const viewDuration = Math.max(1, viewEnd - viewStart);
  const startPct = Math.max(0, Math.min(100, (start - viewStart) / viewDuration * 100));
  const endPct = Math.max(startPct, Math.min(100, (end - viewStart) / viewDuration * 100));
  const playAbs = start + previewTime;
  const playPct = Math.max(startPct, Math.min(endPct, (playAbs - viewStart) / viewDuration * 100));
  const captionDirty = caption !== null && captionText.trim() !== captionSaved.trim();
  const dirty = title.trim() !== savedSnapshot.title || tags !== savedSnapshot.tags || start !== savedSnapshot.start || end !== savedSnapshot.end || (Object.keys(fades) as (keyof FadeState)[]).some((key) => fades[key] !== savedSnapshot.fades[key]) || captionDirty;
  // A corrected caption is burned in, so it reaches the creator's file only
  // through a re-render -- the same one a trim or a fade needs.
  const mediaDirty = start !== savedSnapshot.start || end !== savedSnapshot.end || (Object.keys(fades) as (keyof FadeState)[]).some((key) => fades[key] !== savedSnapshot.fades[key]) || captionDirty;

  // What the creator is actually shipping is the rendered 9:16 file: facecam
  // composited over gameplay with the captions burned in. Preview THAT, the
  // way the theater does. The raw source is 16:9 gameplay with neither, and
  // showing it here made the editor disagree with every other surface.
  //
  // The source is still needed for one thing: the rendered file only covers
  // the cut as it was last rendered, so a trim reaching into the +/-60s pad has
  // no rendered media to show. That trim is exactly when we fall back -- and
  // say so, rather than letting the creator think the facecam vanished.
  const renderedUrl = clip.mediaState === "ready" ? clip.videoUrl : undefined;
  const renderStart = startOriginal;
  const renderEnd = endOriginal;
  const trimWithinRender = start >= renderStart - 0.05 && end <= renderEnd + 0.05;
  const previewMode: "rendered" | "source" =
    renderedUrl && (trimWithinRender || !sourceReady) ? "rendered" : "source";
  const previewSrc = previewMode === "rendered"
    ? renderedUrl
    : sourceReady ? sourceUrl : clip.videoUrl;
  // Media time for absolute VOD time t is `t - mediaOffset`. The rendered clip
  // starts at its own zero; the source shares the VOD's clock.
  const mediaOffset = previewMode === "rendered" ? renderStart : 0;
  const showingSource = previewMode === "source" && Boolean(renderedUrl);

  const clampStart = (value: number, currentEnd = end) => {
    const earliest = Math.max(0, currentEnd - MAX_CLIP_DURATION);
    return Math.max(earliest, Math.min(value, currentEnd - 1, vodDuration - 1));
  };
  const clampEnd = (value: number, currentStart = start) => (
    Math.min(vodDuration, Math.max(value, currentStart + 1))
  );

  const setStartClamped = (value: number) => {
    const nextStart = clampStart(value, end);
    setStart(nextStart);
  };

  const setEndClamped = (value: number) => {
    const maxEnd = Math.min(vodDuration, start + MAX_CLIP_DURATION);
    setEnd(Math.min(maxEnd, clampEnd(value, start)));
  };

  const setDurationClamped = (value: number) => {
    const nextDuration = Math.max(1, Math.min(MAX_CLIP_DURATION, value));
    setEnd(Math.min(vodDuration, start + nextDuration));
  };

  const reset = () => {
    setTitle(savedSnapshot.title); setTags(savedSnapshot.tags); setStart(savedSnapshot.start); setEnd(savedSnapshot.end);
    setFades(savedSnapshot.fades); setError(null);
    setCaptionText(captionSaved); setCaptionError(null);
    if (caption) setCaptionWords(caption.words);
    setEditingWord(null); setCaptionMode("words");
  };

  const save = async () => {
    if (!dirty || saving) return;
    if (!title.trim()) { setError("Give this clip a title before saving."); return; }
    if (end - start > MAX_CLIP_DURATION) {
      setError(`Clips can be at most ${MAX_CLIP_DURATION} seconds long.`);
      return;
    }
    setSaving(true); setError(null);
    const restorePreview = releaseClipPreview(clip.videoUrl);
    try {
      // Before onSave, so the re-render it triggers picks the correction up.
      if (captionDirty) {
        const saved = await putCaption(clip.id, captionText.trim());
        // The engine re-times the correction against the decode, so its word
        // list -- not ours -- is what the render will burn in.
        const timed = readWords(saved?.words);
        setCaptionSaved(captionText.trim());
        if (timed.length) { setCaptionWords(timed); setCaptionMode("words"); }
        setEditingWord(null);
        setCaption((current) => (current ? {
          ...current,
          source: captionText.trim() ? "edit" : "machine",
          text: captionText.trim(),
          words: timed.length ? timed : current.words,
          stale: false,
          heldEditText: undefined,
        } : current));
      }
      await onSave({ title: title.trim(), tags, start, end, fades, mediaDirty });
      setSavedSnapshot({ title: title.trim(), tags, start, end, fades: { ...fades } });
    }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Recall could not save this edit. Try again."); }
    finally { restorePreview(); setSaving(false); }
  };

  const exportOne = async () => {
    if (dirty) { setError("Save your changes before exporting this clip."); return; }
    setExporting(true); setError(null);
    const restorePreview = releaseClipPreview(clip.videoUrl);
    try { await onExport(); } catch { setError("Recall could not export this clip. Choose a destination and try again."); }
    finally { restorePreview(); setExporting(false); }
  };

  const copyCaption = async () => {
    const caption = postText(
      { title: title.trim(), description: clip.description, postTags: tags.split(",").map((t) => t.trim()).filter(Boolean) },
      "shorts",
    );
    try {
      await navigator.clipboard.writeText(caption);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch { setError("Recall couldn't copy the caption to your clipboard."); }
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA" || target?.isContentEditable;
      if (event.key === "Escape" && !saving && !exporting) onClose();
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); void save(); }
      if (event.code === "Space" && !typing) {
        event.preventDefault();
        const video = videoRef.current;
        if (video) video.paused ? void video.play() : video.pause();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // New clip (or a re-rendered src) starts on its poster, not a live frame.
  useEffect(() => {
    setStarted(false);
    setPreviewTime(0);
    setPosterAttempt("file");
    setSourceReady(false);
    setSourceProbeDone(false);
    setBuffering(false);
  }, [clip.id, clip.videoUrl, clip.thumbUrl]);

  // Prefer the full VOD source for trim preview + filmstrip so extending the
  // cut into the ±60s pad actually shows that media. Fall back to the rendered
  // clip if the source endpoint is unavailable.
  useEffect(() => {
    let cancelled = false;
    const probe = document.createElement("video");
    probe.preload = "metadata";
    probe.muted = true;
    const onReady = () => {
      if (!cancelled) {
        setSourceReady(true);
        setSourceProbeDone(true);
      }
    };
    const onError = () => {
      if (!cancelled) {
        setSourceReady(false);
        setSourceProbeDone(true);
      }
    };
    probe.addEventListener("loadedmetadata", onReady, { once: true });
    probe.addEventListener("error", onError, { once: true });
    probe.src = sourceUrl;
    return () => {
      cancelled = true;
      probe.removeAttribute("src");
      probe.load();
    };
  }, [sourceUrl, clip.id]);

  // Use the backend's cached contact sheet instead of running sixteen hidden
  // VOD seeks beside the visible player. Competing decoders made both surfaces
  // look stuck on long recordings.
  const filmstripUrl = sourceReady
    ? apiUrl(
      `/jobs/${session.id}/filmstrip?start=${filmstripWindow.start.toFixed(2)}&end=${filmstripWindow.end.toFixed(2)}&frames=16&width=1536&height=68`,
      apiEndpoint,
    )
    : undefined;
  const filmstripState = filmstripUrl && filmstripLoadedUrl === filmstripUrl
    ? "ready"
    : filmstripUrl && filmstripFailedUrl === filmstripUrl
      ? "error"
      : sourceProbeDone && !sourceReady
        ? "error"
        : "loading";

  // Prefer the VOD reaction curve for the ±60s window (no multi‑GB source fetch).
  useEffect(() => {
    let cancelled = false;
    setTimelineStatus("loading");
    setWaveformState("loading");
    fetchReactionTimeline(session.id).then((timeline) => {
      if (cancelled) return;
      setReactionTimeline(timeline);
      setTimelineStatus(timeline ? "ready" : "missing");
    });
    return () => { cancelled = true; };
  }, [session.id]);

  // Clip-file audio as fallback; mapped into the absolute view so pad stays quiet.
  useEffect(() => {
    let cancelled = false;
    const loadWaveform = async () => {
      // Do not fetch and decode the whole rendered clip while the lightweight
      // reaction timeline is available (or still resolving).
      if (timelineStatus !== "missing") return;
      if (!clip.videoUrl) {
        setClipAudioPeaks(null);
        return;
      }
      try {
        const response = await fetch(clip.videoUrl);
        if (!response.ok) throw new Error("media unavailable");
        const bytes = await response.arrayBuffer();
        const AudioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (!AudioContextClass) throw new Error("audio decoding unavailable");
        const context = new AudioContextClass();
        try {
          const buffer = await context.decodeAudioData(bytes.slice(0));
          const bins = 128;
          const peaks = Array.from({ length: bins }, (_, index) => {
            const from = Math.floor(index / bins * buffer.length);
            const to = Math.max(from + 1, Math.floor((index + 1) / bins * buffer.length));
            let peak = 0;
            for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
              const samples = buffer.getChannelData(channel);
              const stride = Math.max(1, Math.floor((to - from) / 160));
              for (let sample = from; sample < to; sample += stride) peak = Math.max(peak, Math.abs(samples[sample] || 0));
            }
            return peak;
          });
          const maxPeak = Math.max(.001, ...peaks);
          if (!cancelled) setClipAudioPeaks(peaks.map((peak) => peak / maxPeak));
        } finally { void context.close(); }
      } catch {
        if (!cancelled) setClipAudioPeaks(null);
      }
    };
    void loadWaveform();
    return () => { cancelled = true; };
  }, [clip.videoUrl, timelineStatus]);

  // Rebuild the visible envelope whenever the trim window moves.
  useEffect(() => {
    const reactionPeaks = reactionTimeline
      ? peaksFromTimeline(reactionTimeline, filmstripWindow.start, filmstripWindow.end)
      : null;
    if (reactionPeaks) {
      setAudioPeaks(reactionPeaks);
      setWaveformKind("reaction");
      setWaveformState("ready");
      return;
    }
    if (clipAudioPeaks?.length) {
      setAudioPeaks(mapClipPeaksToView(
        clipAudioPeaks,
        startOriginal,
        endOriginal,
        filmstripWindow.start,
        filmstripWindow.end,
      ));
      setWaveformKind("audio");
      setWaveformState("ready");
      return;
    }
    setAudioPeaks(null);
    setWaveformKind(null);
    if (timelineStatus === "loading") setWaveformState("loading");
    else if (!clip.videoUrl) setWaveformState("unavailable");
    else setWaveformState(timelineStatus === "missing" ? "unavailable" : "loading");
  }, [reactionTimeline, timelineStatus, clipAudioPeaks, filmstripWindow.start, filmstripWindow.end, startOriginal, endOriginal, clip.videoUrl]);

  // Keep the preview parked inside the current selection, in whichever clock
  // the loaded media is using.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || dragging) return;
    const from = start - mediaOffset;
    const to = end - mediaOffset;
    if (video.currentTime < from || video.currentTime >= to) {
      video.currentTime = Math.max(0, from);
      setPreviewTime(0);
    }
  }, [start, end, mediaOffset, previewSrc, dragging]);

  const updateFromPointer = (clientX: number) => {
    if (!dragging || !trackRef.current) return;
    const rect = trackRef.current.getBoundingClientRect();
    const time = viewStart + Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)) * viewDuration;
    if (dragging === "start") setStartClamped(Math.min(time, end - 1));
    else setEndClamped(Math.max(time, start + 1));
  };

  // Click/scrub the timeline body to seek the preview within the selection.
  const seekPreview = (clientX: number) => {
    const video = videoRef.current;
    if (!video || !trackRef.current) return;
    const rect = trackRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const absTime = viewStart + Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)) * viewDuration;
    const clampedAbs = Math.max(start, Math.min(end - 0.05, absTime));
    video.currentTime = Math.max(0, clampedAbs - mediaOffset);
    setPreviewTime(Math.max(0, clampedAbs - start));
    setStarted(true);
  };

  /** Park the preview on one caption word, so clicking it shows what it sits on. */
  const seekToWord = (word: CaptionWord) => {
    const video = videoRef.current;
    if (!video) return;
    const absTime = Math.max(start, Math.min(end - 0.05, renderStart + word.start));
    video.currentTime = Math.max(0, absTime - mediaOffset);
    setPreviewTime(Math.max(0, absTime - start));
    setStarted(true);
  };

  // Caption word timings are clip-relative to the cut as it was rendered, so
  // the playhead has to be expressed in that same clock to light a word up.
  const captionClipTime = playAbs - renderStart;
  const liveWord = liveWordIndex(captionWords, captionClipTime);
  const railInSync = captionText.trim() === wordsToText(captionWords);

  const openWord = (index: number) => {
    const word = captionWords[index];
    if (!word) return;
    seekToWord(word);
    setEditingWord(index);
    setWordDraft(word.word);
  };

  const commitWord = (index: number) => {
    const next = replaceWordAt(captionWords, index, wordDraft);
    setCaptionWords(next);
    setCaptionText(wordsToText(next));
    setEditingWord(null);
  };

  // Follow playback down the rail, but never fight the creator's own scrolling
  // while they have a word open for editing.
  useEffect(() => {
    if (!playing || editingWord !== null || liveWord < 0) return;
    const rail = wordRailRef.current;
    const chip = rail?.querySelector<HTMLElement>(`[data-word-index="${liveWord}"]`);
    if (!rail || !chip) return;
    const top = chip.offsetTop - rail.clientHeight / 2 + chip.offsetHeight / 2;
    rail.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  }, [liveWord, playing, editingWord]);

  const evidence = clip.modalityBreakdown
    ? Object.entries(clip.modalityBreakdown).filter(([, value]) => Number.isFinite(value)).slice(0, 4)
    : [];
  const openingScore = Math.max(0, Math.min(100, Math.round(clip.confidence ?? 0)));
  const openingMeasured = !clip.isManual && openingScore > 0;
  const openingLabel = !openingMeasured
    ? "Opening not measured"
    : openingScore >= 65
      ? "Strong opening"
      : openingScore >= 45
        ? "Good opening"
        : "Slow build";
  const renderedPosterUrl = mediaUrl(`thumb_${clip.id}.jpg`, apiEndpoint);
  const posterUrl = posterAttempt === "file"
    ? renderedPosterUrl
    : posterAttempt === "generated"
      ? clip.thumbUrl
      : undefined;

  const handlePosterError = () => {
    if (posterAttempt === "file" && clip.thumbUrl && clip.thumbUrl !== renderedPosterUrl) {
      setPosterAttempt("generated");
      return;
    }
    setPosterAttempt("video");
    const video = videoRef.current;
    if (video && video.readyState >= 1) {
      const from = Math.max(0, start - mediaOffset);
      video.currentTime = from + Math.min(duration * 0.35, Math.max(0, duration - 0.05));
    }
  };

  return (
    <div className="library-editor-overlay">
      <section className="library-editor" role="dialog" aria-modal="true" aria-label={`Edit ${clip.title}`}>
        <header className="library-editor-topbar">
          <button type="button" className="library-editor-back" onClick={onClose} disabled={saving || exporting}><ArrowLeft size={16} /> Back to Clip Library</button>
          <input className="library-editor-title" value={title} maxLength={120} aria-label="Clip title" onChange={(event) => setTitle(event.target.value)} />
          <span className={`library-editor-state ${dirty ? "is-dirty" : ""}`}>{dirty ? <i /> : <StateIcon tone="success" icon={<Check size={12} />} />}{dirty ? "Unsaved changes" : "All changes saved"}</span>
          <div className="library-editor-actions">
          <button type="button" className="btn-secondary library-editor-reset" onClick={reset} disabled={!dirty || saving}><RotateCcw size={14} /> Reset</button>
          <button type="button" className="cta-accent library-editor-save" onClick={() => void save()} disabled={!dirty || saving}>{saving ? <><span className="studio-spinner" />Rendering…</> : <><Save size={14} /> Save changes</>}</button>
          <button type="button" className="btn-secondary library-editor-export" onClick={() => void exportOne()} disabled={exporting || saving}><Download size={14} />{exporting ? "Exporting…" : "Export"}</button>
          <button type="button" className="btn-secondary" onClick={() => void copyCaption()} title="Copy title, description, and hashtags for posting">{copied ? <><Check size={14} /> Copied</> : <><Copy size={14} /> Copy caption</>}</button>
          </div>
        </header>

        <div className="library-editor-workspace">
          <section className="library-editor-stage">
            <div className="library-editor-video-wrap">
              {previewSrc ? (
                <>
                  <video
                    ref={videoRef}
                    key={`${clip.id}:${previewSrc}`}
                    src={saving || exporting ? undefined : previewSrc}
                    poster={posterUrl}
                    muted={mediaVolume.muted}
                    playsInline
                    preload="auto"
                    onLoadedMetadata={(event) => {
                      event.currentTarget.volume = mediaVolume.volume;
                      const from = Math.max(0, start - mediaOffset);
                      // With no poster image left to try, the element itself has
                      // to supply the still -- and the first frame of a cut is
                      // usually the least representative one in it.
                      event.currentTarget.currentTime = posterAttempt === "video"
                        ? from + Math.min(duration * 0.35, Math.max(0, duration - 0.05))
                        : from;
                      setPreviewTime(0);
                    }}
                    onCanPlay={() => setBuffering(false)}
                    onPlaying={() => setBuffering(false)}
                    onWaiting={() => setBuffering(true)}
                    onSeeking={() => setBuffering(true)}
                    onSeeked={() => setBuffering(false)}
                    onPlay={() => { setPlaying(true); setStarted(true); }}
                    onPause={() => setPlaying(false)}
                    onTimeUpdate={(event) => {
                      const current = event.currentTarget.currentTime || 0;
                      const from = Math.max(0, start - mediaOffset);
                      const to = end - mediaOffset;
                      if (current >= to - 0.04 || current < from - 0.04) {
                        event.currentTarget.currentTime = from;
                        setPreviewTime(0);
                        return;
                      }
                      setPreviewTime(Math.max(0, current - from));
                    }}
                  />
                  {!started && (
                    // A paused <video> renders black instead of its poster in
                    // Chromium/Electron until the first frame is painted; show the
                    // thumbnail (and a play affordance) over it until playback
                    // starts so the preview is never a black rectangle.
                    <button type="button" className={`library-editor-poster ${posterAttempt === "video" ? "is-video-frame" : ""}`} onClick={() => { const video = videoRef.current; if (video) void video.play(); }} aria-label="Play preview">
                      {posterUrl && <img src={posterUrl} alt="" onError={handlePosterError} />}
                      <span className="library-editor-poster-play"><Play size={30} fill="currentColor" /></span>
                    </button>
                  )}
                  {started && buffering && (
                    <div className="library-editor-buffering" role="status">
                      <span className="studio-spinner" />
                      Loading preview…
                    </div>
                  )}
                  {showingSource && (
                    <div className="library-editor-source-note" role="status">
                      <Film size={13} aria-hidden="true" />
                      <span>You trimmed past the rendered cut, so this is raw source. Facecam and captions come back when you save.</span>
                    </div>
                  )}
                </>
              ) : posterUrl ? <img src={posterUrl} alt="" onError={handlePosterError} /> : <div className="library-editor-video-empty"><Film size={28} /><span>Preview unavailable</span></div>}
              <div className="library-editor-video-fade" style={{ opacity: Math.max(fades.videoIn > 0 ? 1 - previewTime / fades.videoIn : 0, fades.videoOut > 0 ? 1 - Math.max(0, duration - previewTime) / fades.videoOut : 0) }} />
            </div>
            <div className="library-editor-playerbar">
              <button type="button" onClick={() => { const video = videoRef.current; if (video) video.paused ? void video.play() : video.pause(); }} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={16} /> : <Play size={16} fill="currentColor" />}</button>
              <MediaVolumeControl className="library-editor-volume" volume={mediaVolume.volume} muted={mediaVolume.muted} onVolumeChange={mediaVolume.setVolume} onToggleMuted={mediaVolume.toggleMuted} />
              <span>{preciseTime(previewTime)} / {preciseTime(duration)}</span>
              <b>{showingSource ? "Source frames" : "9:16 vertical"}</b>
            </div>
          </section>

          <aside className="library-editor-inspector">
            <section>
              <h2><Film size={15} /> Clip details</h2>
              <label><span>Title</span><input value={title} maxLength={120} onChange={(event) => setTitle(event.target.value)} /></label>
              <label><span>Tags</span><input value={tags} placeholder="Laughter, action peak" onChange={(event) => setTags(event.target.value)} /></label>
              <small>Tags stay with this review workspace. The title is saved to the rendered clip.</small>
            </section>
            <section>
              <h2><Film size={15} /> Video</h2>
              <FadeControl label="Fade in" value={fades.videoIn} max={fadeMax} onChange={(videoIn) => setFades((current) => ({ ...current, videoIn }))} />
              <FadeControl label="Fade out" value={fades.videoOut} max={fadeMax} onChange={(videoOut) => setFades((current) => ({ ...current, videoOut }))} />
              <small>Picture fades transition to black and are baked into the export.</small>
            </section>
            <section>
              <h2><Volume2 size={15} /> Audio</h2>
              <FadeControl label="Fade in" value={fades.audioIn} max={fadeMax} onChange={(audioIn) => setFades((current) => ({ ...current, audioIn }))} />
              <FadeControl label="Fade out" value={fades.audioOut} max={fadeMax} onChange={(audioOut) => setFades((current) => ({ ...current, audioOut }))} />
              <small>Sound fades independently from the picture for cleaner openings and tails.</small>
            </section>
            <section className="library-editor-captions">
              <h2><Captions size={15} /> Captions</h2>

              {caption?.stale && (
                <p className="library-editor-caption-stale" role="status">
                  <AlertTriangle size={14} aria-hidden="true" />
                  <span>You re-cut this clip after fixing its captions, so your words no longer line up with the audio. Recall is using what it heard. Correct it again to put your version back.</span>
                </p>
              )}

              {caption === null || caption.source === "none" ? (
                <>
                  <p className="library-editor-caption-intro">
                    These words are burned into the exported clip. Recall reads them back from the clip's audio, which takes a few seconds.
                  </p>
                  <button
                    type="button"
                    className="library-editor-caption-read"
                    onClick={readCaption}
                    disabled={captionBusy === "reading" || caption === null}
                  >
                    {captionBusy === "reading"
                      ? <><RefreshCw size={14} className="is-spinning" aria-hidden="true" /> Reading this clip…</>
                      : <><Captions size={14} aria-hidden="true" /> Show the words</>}
                  </button>
                </>
              ) : (
                <>
                  <div className="library-editor-caption-head">
                    <span className="library-editor-caption-label">
                      Words in this clip
                      {caption.source === "edit" && !captionDirty && (
                        <i className="library-editor-caption-chip">Corrected</i>
                      )}
                    </span>
                    {captionWords.length > 0 && (
                      <button
                        type="button"
                        className="library-editor-caption-mode"
                        onClick={() => { setEditingWord(null); setCaptionMode(captionMode === "words" ? "text" : "words"); }}
                        disabled={captionMode === "text" && !railInSync}
                        title={captionMode === "text" && !railInSync
                          ? "Recall re-times a rewrite when you save; the timed words come back then."
                          : undefined}
                      >
                        {captionMode === "words" ? "Edit as plain text" : "Back to timed words"}
                      </button>
                    )}
                  </div>

                  {captionMode === "words" && captionWords.length > 0 ? (
                    <>
                      {/* Every word carries the second it lands on screen, so a
                          fix can be aimed at the word the creator just heard go
                          wrong instead of hunted for in a paragraph. */}
                      <div
                        className="library-editor-caption-rail"
                        ref={wordRailRef}
                        role="list"
                        aria-label="Caption words with the time each appears"
                      >
                        {captionWords.map((word, index) => (
                          <span
                            className={`library-editor-word ${index === liveWord ? "is-live" : ""} ${index === editingWord ? "is-editing" : ""}`}
                            key={`${index}-${word.start}`}
                            data-word-index={index}
                            role="listitem"
                          >
                            {index === editingWord ? (
                              <input
                                autoFocus
                                className="library-editor-word-input"
                                value={wordDraft}
                                size={Math.max(3, wordDraft.length + 1)}
                                aria-label={`Word at ${wordClock(word.start)}`}
                                onChange={(event) => setWordDraft(event.target.value)}
                                onBlur={() => commitWord(index)}
                                onKeyDown={(event) => {
                                  if (event.key === "Enter" || event.key === "Tab") {
                                    event.preventDefault();
                                    commitWord(index);
                                    if (event.key === "Tab" && index + 1 < captionWords.length) openWord(index + 1);
                                  }
                                  if (event.key === "Escape") { event.preventDefault(); setEditingWord(null); }
                                }}
                              />
                            ) : (
                              <button
                                type="button"
                                className="library-editor-word-btn"
                                onClick={() => openWord(index)}
                                aria-label={`Edit ${word.word}, on screen at ${wordClock(word.start)}`}
                              >
                                <b>{word.word}</b>
                                <i className="t-num">{wordClock(word.start)}</i>
                              </button>
                            )}
                          </span>
                        ))}
                      </div>
                      <small className="library-editor-caption-hint">
                        Click a word to jump the preview there and fix it. Type several words into one to split it; clear it to drop it.
                      </small>
                    </>
                  ) : (
                    <label className="library-editor-caption-field">
                      <textarea
                        value={captionText}
                        rows={5}
                        spellCheck
                        placeholder="No speech was found in this clip."
                        aria-label="Caption words for this clip"
                        onChange={(event) => setCaptionText(event.target.value)}
                      />
                    </label>
                  )}

                  {captionMode === "text" && !railInSync && captionWords.length > 0 && (
                    <p className="library-editor-caption-heard">
                      Recall re-times this rewrite against the audio when you save, and the timed words come back then.
                    </p>
                  )}

                  {caption.machineText && caption.machineText !== captionText.trim() && (
                    <p className="library-editor-caption-heard">
                      Recall heard <q>{caption.machineText}</q>
                    </p>
                  )}

                  <div className="library-editor-caption-foot">
                    <small>
                      {captionDirty
                        ? "Saving re-renders this clip with your words."
                        : caption.source === "edit"
                          ? "Your words are burned in the next time this clip renders."
                          : "Fix anything Recall misheard, then save."}
                    </small>
                    {captionText.trim() !== caption.machineText && caption.machineText && (
                      <button
                        type="button"
                        className="library-editor-caption-revert"
                        onClick={() => {
                          setCaptionText(caption.machineText);
                          setEditingWord(null);
                          // The machine's own timings are not ours to rebuild;
                          // the save re-times and hands the rail back.
                          if (caption.machineText !== wordsToText(captionWords)) setCaptionMode("text");
                        }}
                      >
                        <RotateCcw size={13} aria-hidden="true" /> Use what Recall heard
                      </button>
                    )}
                  </div>
                </>
              )}

              {captionError && (
                <p className="library-editor-caption-error" role="alert">{captionError}</p>
              )}
            </section>

            <section className="library-editor-why">
              <h2><Target size={15} /> Why this moment</h2>
              <div className="library-editor-hype"><div><strong>{openingLabel}</strong><small>{openingMeasured ? "Opening signal, not overall clip quality" : "No automatic opening signal is available"}</small></div></div>
              <p>{clip.hookLine || clip.reason || "Recall found a strong reaction with a clear in-game payoff."}</p>
              {evidence.map(([label, value]) => <div className="library-editor-signal" key={label}><span>{clueLabel(label)}</span><i><b style={{ width: `${Math.max(4, Math.min(100, value * 100))}%` }} /></i></div>)}
            </section>
          </aside>
        </div>

        <section className="library-editor-timeline">
          <div className="library-editor-timefields">
            <TimeField label="Start" seconds={start} min={0} max={end - 1} onCommit={setStartClamped} />
            <TimeField label="End" seconds={end} min={start + 1} max={Math.min(vodDuration, start + MAX_CLIP_DURATION)} onCommit={setEndClamped} />
            <TimeField label="Duration" seconds={duration} min={1} max={Math.min(MAX_CLIP_DURATION, vodDuration - start)} onCommit={setDurationClamped} />
            <span>Drag either edge · ±{CONTEXT_PAD}s context</span>
          </div>
          <div className="library-editor-ruler">{Array.from({ length: 7 }, (_, index) => <span key={index} style={{ left: `${index / 6 * 100}%` }}>{preciseTime(viewStart + index / 6 * viewDuration).slice(0, 5)}</span>)}</div>
          <div ref={trackRef} className="library-editor-track" style={{ "--trim-start": `${startPct}%`, "--trim-end": `${endPct}%`, "--playhead": `${playPct}%` } as CSSProperties} onPointerDown={(event) => seekPreview(event.clientX)} onPointerMove={(event) => updateFromPointer(event.clientX)} onPointerUp={() => setDragging(null)} onPointerCancel={() => setDragging(null)}>
            <div className={`library-editor-filmstrip is-${filmstripState}`} aria-label={filmstripState === "ready" ? "Frames from around this clip" : "Clip frames loading"}>
              {filmstripUrl && filmstripState !== "error" && (
                <img
                  src={filmstripUrl}
                  alt=""
                  draggable={false}
                  onLoad={() => setFilmstripLoadedUrl(filmstripUrl)}
                  onError={() => setFilmstripFailedUrl(filmstripUrl)}
                />
              )}
              {filmstripState === "loading" && <span>Loading source frames…</span>}
              {filmstripState === "error" && <span>Source frames unavailable</span>}
            </div>
            <div
              className={`library-editor-wave is-${waveformState}${waveformKind === "reaction" ? " is-reaction" : ""}`}
              aria-label={
                waveformState === "ready"
                  ? (waveformKind === "reaction" ? "Reaction signal across this timeline window" : "Audio waveform for this clip")
                  : "Waveform unavailable"
              }
            >
              {audioPeaks?.map((peak, index) => <i key={index} style={{ height: `${Math.max(2, 3 + peak * 28)}px` }} />)}
              {waveformState !== "ready" && (
                <span>{waveformState === "loading" ? "Reading timeline signal…" : "Waveform unavailable"}</span>
              )}
            </div>
            <div className="library-editor-shade is-left" /><div className="library-editor-selection" /><div className="library-editor-shade is-right" /><div className="library-editor-playhead" />
            {(["start", "end"] as const).map((handle) => <button type="button" key={handle} className={`library-editor-handle is-${handle}`} aria-label={`Trim ${handle}`} onPointerDown={(event) => { event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId); setDragging(handle); }}><i /><i /></button>)}
          </div>
          {error && <div className="library-editor-error" role="alert">{error}</div>}
          <div className="library-editor-shortcuts"><span><kbd>Space</kbd> Play / pause</span><span><kbd>Ctrl S</kbd> Save changes</span><span><kbd>Esc</kbd> Back to library</span></div>
        </section>
      </section>
    </div>
  );
}

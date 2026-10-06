// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The Cutting Room: the OpenCut editor, inside Recall, on a stream's full VOD.
 *
 * The VOD is streamed from the engine by byte range (opencut/src/recall/
 * streamed-media.ts), with the engine's waveform and filmstrips on the
 * timeline, so a four-hour stream opens in seconds without loading it. Cut
 * the moments you want on the timeline; "Make clips" sends each piece to the
 * engine, which renders it like any other Recall clip (vertical, facecam
 * framed, captioned) into the Clip Library.
 *
 * A VOD that has left the disk is restored first, the same way the old
 * Cutting Room did; a moment handed over from Memory opens centred.
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { OpenCutAction } from "../opencut/OpenCutEditor";
import type { ImportAdopter, StreamHost } from "@opencut/recall/streamed-media";
import type { RecallCaptionTrackHost, RecallCaptionWord, RecallEditorHost, RecallMoment, RecallOpenClip } from "@opencut/recall/host";
import { hypeOf } from "../home/homeModel";
import StudioEmptyState from "../components/StudioEmptyState";
import { apiFetch, apiUrl, mediaUrl } from "../lib/api";
import { fmtClock } from "../lib/format";
import { Download, FolderOpen, Library, Pulse, RefreshCw, Scissors } from "../lib/icons";
import { useJobStore, type Clip, type ManualClipLayout, type ReactionTimeline, type Session } from "../lib/store";
import type { Box } from "../components/FacecamCropper";
import { FacecamDialog } from "./FacecamDialog";
import { GameplayPan } from "./GameplayWindow";
import { useFraming } from "./useFraming";
import { SessionPicker } from "./SessionPicker";
import { readClipFades, writeClipFades, type ClipFades } from "./clipMeta";

const OpenCutEditor = lazy(() => import("../opencut/OpenCutEditor").then((module) => ({ default: module.OpenCutEditor })));

export type CuttingRoomMemoryHandoff = {
  entryId: string;
  timestamp: number;
  endTime: number;
  query: string;
};

type Props = {
  session?: Session;
  /** The Library, for "Load session". */
  sessions?: Session[];
  onLoadSession?: (session: Session) => void;
  /** Open a local video to edit (no scan). */
  onOpenVideo?: () => void;
  /** A video dropped into the empty editor was registered; open it. */
  onOpenedVideo?: (jobId: string) => void;
  /** Scan a video that was opened without one. */
  onScan?: (sessionId: string) => void;
  /** The stream's reaction curve, drawn over it on the editor's timeline. */
  timeline: ReactionTimeline | null;
  onBack: () => void;
  initialMoment?: CuttingRoomMemoryHandoff | null;
  /** A clip opened for editing (Export, Review): its piece is selected and "Make clip" updates it. */
  openClip?: { clip: Clip; nonce: number } | null;
};

type SourceStatus = {
  source_key?: string | null;
  state: string;
  state_label: string;
  restore_available: boolean;
  local_path?: string | null;
  last_error?: string | null;
};

type SourceState = "checking" | "ready" | "missing" | "restoring" | "offline";

const MIN_CLIP_SECONDS = 1;
const MAX_CLIP_SECONDS = 180;
/** Seconds of stream shown around a moment handed over from Memory. */
const MOMENT_SPAN_SECONDS = 600;
/** Video track height in the Cutting Room (OpenCut's default is 65 px). */
const STREAM_TRACK_HEIGHT = 120;

/** Recall renders clips 1080x1920; edits are mapped onto that canvas. */
const CLIP_CANVAS = { width: 1080, height: 1920 };

const LAYOUTS: { value: ManualClipLayout; label: string; note: string }[] = [
  { value: "auto", label: "Auto", note: "Recall decides" },
  { value: "vertical_split", label: "Stacked", note: "Cam over game" },
  { value: "gameplay_pip", label: "PiP", note: "Game first, cam inset" },
  { value: "full_gameplay", label: "Gameplay", note: "No facecam" },
];

interface ClipFraming {
  layout: ManualClipLayout;
  /** The creator's facecam box on the source frame, normalized [x, y, w, h]. */
  facecam: Box | null;
  /** Gameplay layout: where the vertical window sits, 0 (left) to 1 (right). */
  focusX: number;
  /** Burn captions in; undefined = the app's caption setting. */
  captions?: boolean;
}

const framingKey = (sessionId: string) => `recall-cutting-framing:${sessionId}`;

/**
 * A video's framing, remembered per video: a facecam (or a gameplay window)
 * stays put for a whole stream. `fallbackLayout` applies when nothing is
 * saved yet (an unscanned recording starts on Gameplay).
 */
export function readFraming(sessionId: string, fallbackLayout: ManualClipLayout = "auto"): ClipFraming {
  try {
    const raw = JSON.parse(localStorage.getItem(framingKey(sessionId)) || "null");
    const layout = LAYOUTS.some((option) => option.value === raw?.layout) ? raw.layout as ManualClipLayout : fallbackLayout;
    const facecam = Array.isArray(raw?.facecam) && raw.facecam.length === 4 && raw.facecam.every((v: unknown) => typeof v === "number" && v >= 0 && v <= 1)
      ? raw.facecam as Box
      : null;
    const focusX = typeof raw?.focusX === "number" && raw.focusX >= 0 && raw.focusX <= 1 ? raw.focusX : 0.5;
    const captions = typeof raw?.captions === "boolean" ? raw.captions : undefined;
    return { layout, facecam, focusX, captions };
  } catch {
    return { layout: fallbackLayout, facecam: null, focusX: 0.5 };
  }
}

/** Opened for editing without a scan (the engine's /jobs/open marks it so). */
export const isUnscanned = (session: Session | undefined) =>
  !!session && session.status === "completed" && /not scanned/i.test(session.message ?? "");

function writeFraming(sessionId: string, framing: ClipFraming) {
  try { localStorage.setItem(framingKey(sessionId), JSON.stringify(framing)); } catch { /* not remembered */ }
}

export const streamRefFor = (sessionId: string) => `job:${sessionId}`;
const fadesKeyOf = (fades: ClipFades) => [fades.pictureIn, fades.pictureOut, fades.soundIn, fades.soundOut].map((value) => value.toFixed(2)).join(",");
const pieceKey = (ref: string, piece: { start: number; end: number }) => `${ref}@${piece.start.toFixed(2)}-${piece.end.toFixed(2)}`;
const jobIdOf = (ref: string) => ref.replace(/^job:/, "");

/** The stream's moments as the editor shows them: found ones and the creator's own cuts. */
export function momentsOf(session: Session): RecallMoment[] {
  const saved = new Set(session.savedClipIds);
  return session.clips
    .filter((clip) => clip.start_time != null && clip.end_time != null)
    // Second-look candidates only count once the creator kept one.
    .filter((clip) => !clip.isOverflowCandidate || clip.kept || saved.has(clip.id))
    .map((clip) => ({
      id: clip.id,
      title: clip.hookLine || clip.title,
      start: clip.start_time!,
      end: clip.end_time!,
      peak: clip.peakTimestamp,
      score: clip.isManual ? undefined : hypeOf(clip),
      state: clip.isManual ? "yours"
        : clip.kept || saved.has(clip.id) ? "kept"
          : clip.maybe ? "maybe"
            : clip.passed ? "passed"
              : "waiting",
    }));
}

export default function CuttingRoomEditor({ session, sessions = [], onLoadSession, timeline, onBack, onOpenVideo, onOpenedVideo, onScan, initialMoment, openClip }: Props) {
  const [picking, setPicking] = useState(false);
  const facecamDialog = (open: boolean, frameUrl: string | null) => (open ? (
    <FacecamDialog
      frameUrl={frameUrl}
      value={framing.facecam}
      onSave={(box: Box) => { updateFraming({ ...framing, facecam: box, layout: framing.layout === "full_gameplay" ? "auto" : framing.layout }); setDrawing(false); }}
      onClear={() => { updateFraming({ ...framing, facecam: null }); setDrawing(false); }}
      onClose={() => setDrawing(false)}
    />
  ) : null);
  const picker = picking && onLoadSession ? (
    <SessionPicker
      sessions={sessions}
      currentId={session?.id}
      onPick={(next) => { setPicking(false); onLoadSession(next); }}
      onClose={() => setPicking(false)}
    />
  ) : null;
  const apiEndpoint = useJobStore((state) => state.settings.apiEndpoint);
  const captionsEnabled = useJobStore((state) => state.settings.captionStyle.enabled);
  const captionStyle = useJobStore((state) => state.settings.captionStyle);
  const createManualClip = useJobStore((state) => state.createManualClip);
  const editClip = useJobStore((state) => state.editClip);
  const updateClipTitle = useJobStore((state) => state.updateClipTitle);
  const addLog = useJobStore((state) => state.addLog);
  const [sourceState, setSourceState] = useState<SourceState>("checking");
  // Bumped by "Try again" when the engine didn't answer.
  const [retry, setRetry] = useState(0);
  const [status, setStatus] = useState<SourceStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const unscanned = isUnscanned(session);
  const [framing, setFraming] = useState<ClipFraming>(() => (session ? readFraming(session.id, unscanned ? "full_gameplay" : "auto") : { layout: "auto", facecam: null, focusX: 0.5 }));
  const [drawing, setDrawing] = useState(false);
  const [drawAt, setDrawAt] = useState(0);
  useEffect(() => {
    if (session) setFraming(readFraming(session.id, isUnscanned(session) ? "full_gameplay" : "auto"));
    setDrawing(false);
  }, [session?.id]);
  const captionsOn = framing.captions ?? captionsEnabled;
  const updateFraming = useCallback((next: ClipFraming) => {
    setFraming(next);
    if (session) writeFraming(session.id, next);
  }, [session?.id]);
  // Pieces already turned into clips this visit, so a second press doesn't duplicate them.
  const made = useRef(new Map<string, Clip>());
  // The title and fades each clip was last rendered with, by clip id.
  const madeWith = useRef(new Map<string, { title: string; fades: string; caption: string }>());

  // A clip opened for editing: its caption fixes are read first, so its piece
  // arrives with them. It counts as made, with what it was rendered with.
  const [openClipHost, setOpenClipHost] = useState<RecallOpenClip>();
  useEffect(() => {
    setOpenClipHost(undefined);
    if (!openClip || !session || openClip.clip.start_time == null || openClip.clip.end_time == null) return;
    const { clip, nonce } = openClip;
    const start = clip.start_time as number;
    const end = clip.end_time as number;
    let cancelled = false;
    void (async () => {
      const response = await apiFetch(`/clips/${encodeURIComponent(clip.id)}/caption`, undefined, apiEndpoint).catch(() => null);
      const body = response?.ok ? await response.json().catch(() => null) : null;
      const captionWords = body?.source === "edit" && Array.isArray(body.words) ? body.words as RecallCaptionWord[] : undefined;
      if (cancelled) return;
      const fades = readClipFades(session.id, clip.id);
      madeWith.current.set(clip.id, {
        title: clip.title,
        fades: fadesKeyOf(fades),
        caption: captionWords ? captionWords.map((word) => word.word.trim()).filter(Boolean).join(" ") : "",
      });
      made.current.set(pieceKey(streamRefFor(session.id), { start, end }), clip);
      setOpenClipHost({ key: `${clip.id}:${nonce}`, clipId: clip.id, streamRef: streamRefFor(session.id), start, end, title: clip.title, fades, captionWords });
    })();
    return () => { cancelled = true; };
  }, [openClip?.clip.id, openClip?.nonce, session?.id, apiEndpoint]);

  const scanFinished = !!session && ["completed", "cancelled"].includes(session.status);
  // A local video streams from disk from the start, so it can be cut while its
  // scan runs. A Twitch source has to finish downloading first.
  const localSource = !!session?.sourceUrl && !/^https?:/i.test(session.sourceUrl);
  const scanning = !!session && !scanFinished && session.status !== "failed";
  const editable = scanFinished || (localSource && session?.status !== "failed");
  // The editor-side timeline reader, loaded with the editor chunk.
  const cutting = useRef<typeof import("@opencut/recall/cutting") | null>(null);
  useEffect(() => { void import("@opencut/recall/cutting").then((module) => { cutting.current = module; }); }, []);

  const loadStatus = useCallback(async () => {
    if (!session) return null;
    try {
      const response = await apiFetch(`/jobs/${session.id}/source-status`, undefined, apiEndpoint);
      if (!response.ok) return null;
      const next = await response.json() as SourceStatus;
      setStatus(next);
      return next;
    } catch {
      return null;
    }
  }, [session?.id, apiEndpoint]);

  // Is the VOD on disk? The engine answers /source with 404 when it isn't.
  useEffect(() => {
    if (!session || !editable) return;
    let alive = true;
    made.current = new Map();
    setSourceState("checking");
    setError(null);
    (async () => {
      // no-store: a cached 206 from an earlier visit would hide a deleted VOD.
      const probe = await fetch(apiUrl(`/jobs/${session.id}/source`, apiEndpoint), { headers: { Range: "bytes=0-0" }, cache: "no-store" }).catch(() => null);
      void probe?.body?.cancel();
      if (!alive) return;
      if (probe?.status === 206 || probe?.status === 200) {
        setSourceState("ready");
      } else if (!probe) {
        // No answer at all: the engine is down, not the recording gone.
        setSourceState("offline");
      } else {
        setSourceState("missing");
        void loadStatus();
      }
    })();
    return () => { alive = false; };
  }, [session?.id, editable, apiEndpoint, loadStatus, retry]);

  const restore = useCallback(async () => {
    if (!session) return;
    setError(null);
    try {
      const response = await apiFetch(`/jobs/${session.id}/source/restore`, { method: "POST" }, apiEndpoint);
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = body?.detail;
        throw new Error(typeof detail === "string" ? detail : detail?.message || "Recall could not start restoring this stream.");
      }
      addLog("Restoring the Twitch source. Download progress is recorded in session activity.", "info");
      setSourceState("restoring");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Recall could not restore this stream.");
    }
  }, [session?.id, apiEndpoint, addLog]);

  const cancelRestore = useCallback(async () => {
    if (!status?.source_key) return;
    await apiFetch(`/source-assets/${encodeURIComponent(status.source_key)}/restore/cancel`, { method: "POST" }, apiEndpoint).catch(() => undefined);
    setSourceState("missing");
  }, [status?.source_key, apiEndpoint]);

  useEffect(() => {
    if (sourceState !== "restoring") return;
    const timer = window.setInterval(async () => {
      const next = await loadStatus();
      if (!next) return;
      if (next.local_path) {
        window.clearInterval(timer);
        addLog("Twitch source restored. It's open in the Cutting Room.", "success");
        setSourceState("ready");
      } else if (next.state === "unavailable") {
        window.clearInterval(timer);
        setError(next.last_error || "The original Twitch source is currently unavailable.");
        setSourceState("missing");
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [sourceState, loadStatus, addLog]);

  // Not tied to the open session: a video dropped into the empty editor
  // streams through it before the Cutting Room switches to that video.
  const stream = useMemo<StreamHost>(() => {
    const jobOf = (ref: string) => encodeURIComponent(ref.replace(/^job:/, ""));
    return {
      source: (ref) => apiUrl(`/jobs/${jobOf(ref)}/source`, apiEndpoint),
      waveform: (ref) => apiUrl(`/jobs/${jobOf(ref)}/waveform`, apiEndpoint),
      filmstrip: (ref, r) => apiUrl(
        `/jobs/${jobOf(ref)}/filmstrip?start=${r.start}&end=${r.end}&frames=${r.frames}&width=${r.width}&height=${r.height}`,
        apiEndpoint,
      ),
    };
  }, [apiEndpoint]);

  // Stable across re-renders (a new clip updates the session): the editor
  // treats a new source object as a new source.
  const source = useMemo(() => (session ? { streamRef: streamRefFor(session.id), name: session.name } : null), [session?.id, session?.name]);
  // The editor works on the clip itself: Recall's 9:16 frame, framed live.
  const canvas = CLIP_CANVAS;
  // The stream's track carries frames, the reaction curve and the waveform.
  const trackHeights = useMemo(() => ({ video: STREAM_TRACK_HEIGHT }), []);

  // Each video on the timeline is framed its own way: this one by the panel,
  // a recording dropped in by whatever was last chosen for it.
  const framingFor = useCallback((jobId: string) => (
    jobId === session?.id ? framing : readFraming(jobId, "full_gameplay")
  ), [session?.id, framing]);
  const framingHost = useFraming(apiEndpoint, framingFor);

  // The words each piece's clip will burn in, read the way a render reads them.
  const captionTrack = useMemo<RecallCaptionTrackHost>(() => ({
    enabled: captionsOn,
    style: {
      fontFamily: captionStyle?.font || "Arial Black",
      color: captionStyle?.textColor || "#FFFFFF",
      size: captionStyle?.size || "medium",
      position: captionStyle?.position || "bottom",
    },
    loadWords: async (ref, start, end, listen) => {
      const response = await apiFetch(`/jobs/${encodeURIComponent(jobIdOf(ref))}/captions/window`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start, end, decode: listen }),
      }, apiEndpoint);
      if (!response.ok) {
        const detail = await response.json().then((body) => body?.detail, () => undefined);
        throw new Error(typeof detail === "string" ? detail : "Recall couldn't read this piece's words.");
      }
      return response.json();
    },
  }), [captionsOn, captionStyle, apiEndpoint]);

  const host = useMemo<RecallEditorHost>(() => (session ? {
    layout: {
      options: LAYOUTS,
      value: framing.layout,
      onChange: (value) => updateFraming({ ...framing, layout: value as ManualClipLayout }),
      hasFacecam: !!framing.facecam,
      drawing,
      onToggleDraw: (sourceTime) => {
        setDrawAt(sourceTime ?? 0);
        setDrawing((value) => !value);
      },
      onClearFacecam: () => updateFraming({ ...framing, facecam: null }),
      hint: scanning
        ? "Recall is still scanning this video. Draw the facecam to reframe now; Auto finds it once the scan is done."
        : unscanned && framing.layout !== "full_gameplay" && !framing.facecam
          ? "Not scanned, so Auto can't find a facecam. Use Gameplay for a recording without one, or draw it."
          : framing.layout === "full_gameplay"
            ? "Drag the window on the preview to choose what the vertical clip shows."
            : undefined,
      captions: captionsOn,
      onToggleCaptions: () => updateFraming({ ...framing, captions: !captionsOn }),
    },
    // The preview is the clip, so Gameplay is reframed by dragging the picture.
    previewOverlay: framing.layout === "full_gameplay" ? (
      <GameplayPan focusX={framing.focusX} onChange={(focusX) => updateFraming({ ...framing, focusX })} />
    ) : undefined,
    framing: framingHost,
    captionTrack,
    openClip: openClipHost,
    moments: {
      streamRef: streamRefFor(session.id),
      moments: momentsOf(session),
      curve: timeline?.t?.length && timeline.t.length === timeline.r?.length ? { t: timeline.t, r: timeline.r } : undefined,
      emptyText: scanning
        ? "Recall is scanning this video. Its moments show up here when it finishes; cut your own meanwhile."
        : unscanned
          ? "This video hasn't been scanned. Trim it on the timeline and make your clip, or scan it to find moments."
          : undefined,
    },
  } : { framing: framingHost }), [session, timeline, framing, drawing, updateFraming, scanning, unscanned, captionsOn, framingHost, captionTrack, openClipHost]);

  // A recording dropped into the editor (media bin, drag and drop, paste) is
  // registered like "Open video" (no scan) and streamed from disk, so Make
  // clips can render it. Its clips land on its own session.
  // In the empty editor, a dropped video becomes the session being cut.
  const adoptIntoNewSession = useCallback<ImportAdopter>(async (file) => {
    const path = window.electronAPI?.getPathForFile?.(file);
    if (!path) return null;
    const jobId = await useJobStore.getState().openLocalVideo(path);
    if (jobId) onOpenedVideo?.(jobId);
    return jobId ? { ref: streamRefFor(jobId), size: file.size } : null;
  }, [onOpenedVideo]);

  const adoptImports = useCallback<ImportAdopter>(async (file) => {
    const path = window.electronAPI?.getPathForFile?.(file);
    if (!path) return null;
    const jobId = await useJobStore.getState().openLocalVideo(path);
    return jobId ? { ref: streamRefFor(jobId), size: file.size } : null;
  }, []);

  // The same save Theater's editor uses: the clip's "Your edit", preferred on export.
  const saveEdit = useCallback(async (clipId: string, mp4: ArrayBuffer) => {
    const response = await apiFetch(`/clips/${encodeURIComponent(clipId)}/edited-video`, {
      method: "PUT",
      headers: { "Content-Type": "video/mp4" },
      body: mp4,
    }, apiEndpoint).catch(() => {
      throw new Error("Recall's engine isn't answering. Your clip is made; press Make clips again to add your edits.");
    });
    if (!response.ok) {
      const detail = await response.json().then((body) => body?.detail, () => undefined);
      throw new Error(typeof detail === "string" ? detail : "Recall couldn't keep your edits.");
    }
    const saved = await response.json().catch(() => null) as { filename?: string } | null;
    if (saved?.filename) {
      const url = new URL(mediaUrl(saved.filename, apiEndpoint));
      url.searchParams.set("v", String(Date.now()));
      useJobStore.getState().markClipEdited(clipId, url.toString());
    }
  }, [apiEndpoint]);

  const blankAction = useMemo<OpenCutAction>(() => ({
    label: "Make clips",
    busyLabel: "Making clips…",
    icon: <Scissors aria-hidden="true" />,
    stateFor: () => ({ label: "Make clips", disabled: true, hint: "Open a video, load a session, or drop a recording in to start cutting." }),
    note: "Open a video, load a session, or drop a recording in to start cutting.",
    run: async () => { throw new Error("Open a video or load a session first."); },
  }), []);

  const action = useMemo<OpenCutAction | undefined>(() => {
    if (!session) return undefined;
    const ref = streamRefFor(session.id);
    return {
      label: "Make clips",
      busyLabel: "Making clips…",
      icon: <Scissors aria-hidden="true" />,
      stateFor: (editor) => {
        const module = cutting.current;
        if (!module) return { label: "Make clips" };
        const all = module.streamRefsOnTimeline(editor)
          .flatMap((r) => module.streamElements(editor, r).filter((piece) => !piece.reel));
        // A piece over 3 minutes isn't a clip (yet): it's skipped, not a blocker.
        const long = all.filter((piece) => piece.end - piece.start > MAX_CLIP_SECONDS);
        const pieces = all.length - long.length;
        const stranded = module.unregisteredVideosOnTimeline(editor);
        if (!pieces && long.length) {
          return {
            label: "Make clips",
            disabled: true,
            hint: `Clips can be up to 3 minutes. Trim or split the piece at ${fmtClock(long[0].start)} (it's ${fmtClock(long[0].end - long[0].start)}).`,
          };
        }
        if (!pieces) {
          return {
            label: "Make clips",
            disabled: true,
            hint: stranded.length
              ? `Recall can't render "${stranded[0]}": it was copied into the editor. Drag it in again (or use Open video) and cut that one.`
              : unscanned
                ? "Trim the video's start and end (drag its edges), or split it where a clip starts and ends."
                : "Cut a moment (Moments tab), or split the video where a clip starts and ends.",
          };
        }
        return {
          label: pieces === 1 ? "Make clip" : `Make ${pieces} clips`,
          hint: long.length
            ? `Each piece becomes a vertical Recall clip. ${long.length === 1 ? "The piece" : `${long.length} pieces`} over 3 minutes ${long.length === 1 ? "sits" : "sit"} this one out.`
            : "Each piece becomes a vertical Recall clip, framed and captioned.",
        };
      },
      note: "Each piece you cut from the stream becomes a Recall clip, framed and captioned like the rest. The whole stream is your reel, not a clip.",
      run: async (editor, report) => {
        // Loaded with the editor chunk, which is already open by now.
        const [{ streamElements, streamRefsOnTimeline, customTitle, CLIP_PARAM }, { editsOverPiece, composeOverClip }, { readFades }, captionTrackModule, { markPieceClip }] = await Promise.all([
          import("@opencut/recall/cutting"),
          import("@opencut/recall/carry-edits"),
          import("@opencut/recall/fades"),
          import("@opencut/recall/caption-track"),
          import("@opencut/recall/open-clip"),
        ]);
        // The creator's corrected words for a piece, when they changed any.
        const correctedCaption = (piece: { element: Parameters<typeof captionTrackModule.readCaptions>[0]; start: number; end: number }) => {
          const stored = captionTrackModule.readCaptions(piece.element);
          if (!captionsOn || !stored?.edited || !captionTrackModule.coversWindow(stored, piece)) return "";
          return captionTrackModule.captionText(captionTrackModule.wordsIn(stored, piece));
        };
        // Every registered video on the timeline: this stream, and recordings
        // dropped in (each makes clips on its own session).
        const pieces = streamRefsOnTimeline(editor).flatMap((r) =>
          streamElements(editor, r).filter((piece) => !piece.reel).map((piece) => Object.assign(piece, { ref: r })));
        if (!pieces.length) throw new Error("Cut something first: pick a moment and press Cut, or split the video where a clip should start and end.");
        // Pieces over 3 minutes aren't clips yet (a half of a split stream,
        // say): they're skipped, and the rest are made.
        const tooLong = pieces.filter((piece) => piece.end - piece.start > MAX_CLIP_SECONDS);
        if (tooLong.length === pieces.length) {
          throw new Error(`Clips can be up to 3 minutes. Split or trim the piece at ${fmtClock(tooLong[0].start)} (it's ${fmtClock(tooLong[0].end - tooLong[0].start)}).`);
        }
        // A piece made before (this visit, or an earlier one) isn't made again;
        // its edits are re-applied instead, so changing a title and pressing
        // again updates the clip.
        const sessionFor = (r: string) => useJobStore.getState().sessions.find((item) => item.id === jobIdOf(r));
        // A piece's clip: the one it made (kept on the piece, so a trim re-cuts
        // it), else one at its exact range. Split halves share the mark; only
        // the first gets the clip, the other makes its own.
        const claimed = new Set<string>();
        const clipById = (r: string, id: string) =>
          [...made.current.values()].find((clip) => clip.id === id) ?? sessionFor(r)?.clips.find((clip) => clip.id === id);
        const madeClipFor = (piece: { ref: string; start: number; end: number; element: { params?: Record<string, unknown> } }) => {
          const mark = piece.element.params?.[CLIP_PARAM];
          const marked = typeof mark === "string" && !claimed.has(mark) ? clipById(piece.ref, mark) : undefined;
          const found = marked
            ?? made.current.get(pieceKey(piece.ref, piece))
            ?? (sessionFor(piece.ref)?.clips ?? []).find((clip) => clip.isManual && clip.start_time != null && clip.end_time != null
              && Math.abs(clip.start_time - piece.start) < 0.05 && Math.abs(clip.end_time - piece.end) < 0.05);
          return found && !claimed.has(found.id) ? found : undefined;
        };
        const usable = pieces.filter((piece) => piece.end - piece.start >= MIN_CLIP_SECONDS && piece.end - piece.start <= MAX_CLIP_SECONDS);
        let createdCount = 0;
        let editedCount = 0;
        let updatedCount = 0;
        for (const [index, piece] of usable.entries()) {
          const step = usable.length > 1 ? ` ${index + 1} of ${usable.length}` : "";
          const edits = editsOverPiece(editor, piece.ref, piece, CLIP_CANVAS);
          const own = piece.ref === ref;
          const source = own ? session : sessionFor(piece.ref);
          let clip = madeClipFor(piece);
          if (clip) claimed.add(clip.id);
          const named = customTitle(piece.element, piece.asset);
          const title = named ?? `Clip from ${source?.name ?? "your video"} at ${fmtClock(piece.start)}`;
          const pieceFades = readFades(piece.element);
          const fades = {
            video_fade_in: pieceFades.pictureIn,
            video_fade_out: pieceFades.pictureOut,
            audio_fade_in: pieceFades.soundIn,
            audio_fade_out: pieceFades.soundOut,
          };
          const fadesKey = fadesKeyOf(pieceFades);
          const caption = correctedCaption(piece);
          if (clip) {
            // Made before: bring its title and fades up to date instead.
            const sessionId = jobIdOf(piece.ref);
            // From an earlier visit: its fades as last rendered, if known.
            const was = madeWith.current.get(clip.id) ?? { title: clip.title, fades: fadesKeyOf(readClipFades(sessionId, clip.id)), caption: "" };
            let changed = false;
            if (named && named !== was.title) {
              report(`Renaming clip${step}…`);
              clip = await updateClipTitle(sessionId, clip.id, named);
              changed = true;
            }
            const captionChanged = caption !== was.caption;
            const recut = clip.start_time == null || clip.end_time == null
              || Math.abs(clip.start_time - piece.start) > 0.05 || Math.abs(clip.end_time - piece.end) > 0.05;
            if (captionChanged && !caption) {
              // Back to Recall's own words: clear the stored correction.
              const response = await apiFetch(`/clips/${encodeURIComponent(clip.id)}/caption`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: "" }),
              }, apiEndpoint);
              if (!response.ok) throw new Error(`Recall couldn't reset the captions for the clip at ${fmtClock(piece.start)}.`);
            }
            if (fadesKey !== was.fades || captionChanged || recut) {
              report(recut ? `Re-cutting clip${step}…` : `Re-rendering clip${step}…`);
              // A fixed caption rides along, recorded against the (possibly new) window.
              clip = await editClip(sessionId, clip.id, { start_time: piece.start, end_time: piece.end, ...fades, ...(caption ? { caption_text: caption } : {}) });
              made.current.set(pieceKey(piece.ref, piece), clip);
              writeClipFades(sessionId, clip.id, pieceFades);
              changed = true;
            }
            if (changed) updatedCount += 1;
            madeWith.current.set(clip.id, { title: named ?? was.title, fades: fadesKey, caption });
          }
          if (!clip) {
            report(`Rendering clip${step}…`);
            const coversMoment = own && initialMoment && initialMoment.timestamp >= piece.start && initialMoment.timestamp <= piece.end;
            clip = await createManualClip(jobIdOf(piece.ref), {
              start: piece.start,
              end: piece.end,
              title,
              tags: [],
              layout: framing.layout,
              focus_x: framing.focusX,
              // A drawn facecam belongs to this stream, not to a recording dropped in.
              facecam_override: framing.layout === "full_gameplay" || !own ? null : framing.facecam,
              captions_enabled: captionsOn,
              ...fades,
              caption_text: caption || undefined,
              memory_entry_id: coversMoment ? initialMoment.entryId : undefined,
              memory_query: coversMoment ? initialMoment.query : undefined,
            });
            made.current.set(pieceKey(piece.ref, piece), clip);
            claimed.add(clip.id);
            madeWith.current.set(clip.id, { title, fades: fadesKey, caption });
            writeClipFades(jobIdOf(piece.ref), clip.id, pieceFades);
            createdCount += 1;
          }
          if (piece.element.params?.[CLIP_PARAM] !== clip.id) markPieceClip(editor, piece, clip.id);
          if (edits.count > 0) {
            if (!clip.videoUrl) throw new Error(`Recall's render of the clip at ${fmtClock(piece.start)} isn't available yet, so your edits couldn't be added.`);
            report(`Adding your edits${step}…`);
            const mp4 = await composeOverClip({
              editor,
              edits,
              clipUrl: clip.videoUrl,
              clipName: clip.id,
              clipCanvas: CLIP_CANVAS,
              onProgress: (progress) => report(`Adding your edits${step}… ${Math.round(progress * 100)}%`),
            });
            await saveEdit(clip.id, mp4);
            editedCount += 1;
          }
        }
        const skipped = tooLong.length
          ? ` Skipped ${tooLong.length === 1 ? `the piece at ${fmtClock(tooLong[0].start)}` : `${tooLong.length} pieces`} over 3 minutes.`
          : "";
        if (!createdCount && !editedCount && !updatedCount) return `Those clips are already in your Clip Library.${skipped}`;
        const summary = [
          createdCount ? `${createdCount} ${createdCount === 1 ? "clip" : "clips"} made` : "",
          updatedCount ? `${updatedCount} ${updatedCount === 1 ? "clip" : "clips"} updated` : "",
          editedCount ? `your edits added to ${editedCount}` : "",
        ].filter(Boolean).join(", ");
        const sentence = summary.charAt(0).toUpperCase() + summary.slice(1);
        addLog(`${sentence} (${session.name}).`, "success");
        return `${sentence}. They're in your Clip Library.${skipped}`;
      },
    };
  }, [session, initialMoment, captionsOn, createManualClip, editClip, updateClipTitle, addLog, framing, saveEdit, unscanned, apiEndpoint]);

  if (!session) {
    // No middle screen: the editor opens straight away on a scratch project.
    // Open video / Load session sit in its bar, and a video dropped in opens
    // as its own session, streamed from disk.
    return (
      <Suspense fallback={<div className="clips-empty"><span className="studio-spinner" /> Opening the editor…</div>}>
        <OpenCutEditor
          inline
          ephemeral
          sessionKey="cutting:scratch"
          stream={stream}
          title="Cutting Room"
          canvas={canvas}
          action={blankAction}
          host={host}
          trackHeights={trackHeights}
          shuttle
          adoptImports={adoptIntoNewSession}
          onClose={onBack}
          extraActions={(
            <>
              {onLoadSession && (
                <button type="button" className="oce-btn is-quiet" onClick={() => setPicking(true)} title="Cut a stream or video from your Library">
                  <Library aria-hidden="true" /> Load session
                </button>
              )}
              {onOpenVideo && (
                <button type="button" className="oce-btn" onClick={onOpenVideo} title="Open a recording or VOD from your PC">
                  <FolderOpen aria-hidden="true" /> Open video
                </button>
              )}
            </>
          )}
        />
        {picker}
      </Suspense>
    );
  }
  if (!editable) {
    return <div className="clips-empty"><StudioEmptyState variant="editor" icon={<Scissors />} title="The Cutting Room is waiting on this stream" body={session.status === "failed" ? "This scan stopped before the video was ready. Open it again from Scans, or open a video from your PC." : "Recall opens the stream here once it has finished downloading."} action={<button type="button" className="btn" onClick={onBack}>Back</button>} /></div>;
  }
  if (sourceState === "offline") {
    return (
      <div className="clips-empty">
        <StudioEmptyState
          variant="editor"
          icon={<RefreshCw />}
          title="Recall's engine isn't answering"
          body="Your recording and clips are fine. The editor needs the engine to stream the video; give it a moment and try again."
          action={
            <div className="w-cta">
              <button type="button" className="btn heat" onClick={() => setRetry((n) => n + 1)}><RefreshCw aria-hidden="true" />Try again</button>
              <button type="button" className="btn ghost" onClick={onBack}>Back</button>
            </div>
          }
        />
      </div>
    );
  }
  if (sourceState === "missing" || sourceState === "restoring") {
    const restoring = sourceState === "restoring";
    return (
      <div className="clips-empty">
        <StudioEmptyState
          variant="editor"
          icon={<Download />}
          title={restoring ? "Restoring the stream" : "This stream isn't on your PC anymore"}
          body={error ?? (restoring
            ? "Recall is downloading the original Twitch VOD. Your clips stay available while this finishes."
            : status?.restore_available
              ? "The download was removed to save space. Your clips are safe; restore it to cut more."
              : "The source recording isn't on disk. Clips you already made are unaffected.")}
          action={
            <div className="w-cta">
              {!restoring && status?.restore_available && <button type="button" className="btn heat" onClick={() => void restore()}><Download aria-hidden="true" />Restore the stream</button>}
              {restoring && <button type="button" className="btn" onClick={() => void cancelRestore()}>Cancel</button>}
              <button type="button" className="btn ghost" onClick={onBack}>Back</button>
            </div>
          }
        />
      </div>
    );
  }
  if (sourceState === "checking" || !stream) {
    return <div className="clips-empty"><span className="studio-spinner" /> Opening the stream…</div>;
  }

  return (
    <Suspense fallback={<div className="clips-empty"><span className="studio-spinner" /> Opening the editor…</div>}>
      <OpenCutEditor
        inline
        sessionKey={`vod:${session.id}`}
        title={session.name}
        source={source!}
        stream={stream}
        canvas={canvas}
        action={action}
        focus={initialMoment
          ? { time: initialMoment.timestamp, span: MOMENT_SPAN_SECONDS }
          : openClipHost
            ? { time: (openClipHost.start + openClipHost.end) / 2, span: Math.max(30, (openClipHost.end - openClipHost.start) * 1.6) }
            : null}
        host={host}
        trackHeights={trackHeights}
        shuttle
        adoptImports={adoptImports}
        onClose={onBack}
        extraActions={(
          <>
            {unscanned && onScan && (
              <button type="button" className="oce-btn is-quiet" onClick={() => onScan(session.id)} title="Find this video's moments and facecam. Uses the GPU for a while.">
                <Pulse aria-hidden="true" /> Scan for moments
              </button>
            )}
            {onLoadSession && (
              <button type="button" className="oce-btn is-quiet" onClick={() => setPicking(true)} title="Cut a stream or video from your Library">
                <Library aria-hidden="true" /> Load session
              </button>
            )}
            {onOpenVideo && (
              <button type="button" className="oce-btn is-quiet" onClick={onOpenVideo} title="Open a recording or VOD from your PC">
                <FolderOpen aria-hidden="true" /> Open video
              </button>
            )}
          </>
        )}
      />
      {picker}
      {facecamDialog(drawing, apiUrl(`/jobs/${encodeURIComponent(session.id)}/frame?t=${drawAt.toFixed(2)}&width=1280`, apiEndpoint))}
    </Suspense>
  );
}

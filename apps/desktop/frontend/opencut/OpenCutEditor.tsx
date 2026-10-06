// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall's host for the vendored OpenCut editor (./src, MIT, see ./LICENSE).
 *
 * A full-window surface that opens one of Recall's rendered MP4s for
 * finishing: text, stickers, effects, transitions, trims. Each clip keeps its
 * own OpenCut project (IndexedDB + OPFS), so closing and reopening resumes
 * the edit. "Save to Recall" exports through OpenCut's WebCodecs pipeline and
 * hands the MP4 back; Recall's own render is never touched.
 *
 * The layout mirrors OpenCut's editor page minus its website chrome (header,
 * changelog, feedback, mobile gate, onboarding), which Recall replaces.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Toaster } from "sonner";
import { setOpenCutRouterHost } from "next/navigation";
import { EditorCore } from "@opencut/core";
import { EditorProvider } from "@opencut/components/providers/editor-provider";
import { TooltipProvider } from "@opencut/components/ui/tooltip";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@opencut/components/ui/resizable";
import { AssetsPanel } from "@opencut/components/editor/panels/assets";
import { PropertiesPanel } from "@opencut/components/editor/panels/properties";
import { PreviewPanel } from "@opencut/preview/components";
import { Timeline } from "@opencut/timeline/components";
import { MigrationDialog } from "@opencut/project/components/migration-dialog";
import { usePanelStore } from "@opencut/editor/panel-store";
import { useEditor } from "@opencut/editor/use-editor";
import { usePasteMedia } from "@opencut/media/use-paste-media";
import { usePreviewStore } from "@opencut/preview/preview-store";
import { createPreviewOverlayControl, isPreviewOverlayVisible, mergePreviewOverlaySources } from "@opencut/preview/overlays";
import { getGuidePreviewOverlaySource } from "@opencut/guides";
import { bookmarkNotesPreviewOverlay, getBookmarkPreviewOverlaySource } from "@opencut/timeline/bookmarks/index";
import { processMediaAssets } from "@opencut/media/processing";
import { buildElementFromMedia } from "@opencut/timeline/element-utils";
import { AddMediaAssetCommand } from "@opencut/commands/media";
import { InsertElementCommand } from "@opencut/commands/timeline";
import { BatchCommand } from "@opencut/commands";
import { DEFAULT_NEW_ELEMENT_DURATION } from "@opencut/timeline/creation";
import { mediaTimeFromSeconds, TICKS_PER_SECOND } from "@opencut/wasm";
import { RecallHostContext, type RecallEditorHost } from "@opencut/recall/host";
import { applyFraming } from "@opencut/recall/framing";
import { syncFades } from "@opencut/recall/clip-panel";
import { applyCaptionTrack, coversWindow, readCaptions, storeCaptionWords } from "@opencut/recall/caption-track";
import { captionStyleOverrides } from "@opencut/recall/captions-view";
import { streamElements, streamRefsOnTimeline } from "@opencut/recall/cutting";
import type { RecallCaptionTrackHost, RecallOpenClip } from "@opencut/recall/host";
import { placeOpenClip } from "@opencut/recall/open-clip";
import { remoteSize, resolveStreamUrl, setImportAdopter, setStreamResolver, streamedFile, streamedMedia, type ImportAdopter, type StreamHost } from "@opencut/recall/streamed-media";
import { requestTimelineView, type TimelineViewRequest } from "@opencut/recall/timeline-view";
import { setTrackHeightOverrides } from "@opencut/timeline/components/track-layout";
import { Shuttle } from "@opencut/recall/shuttle";
import { isTypableDOMElement } from "@opencut/utils/browser";
import type { TrackType } from "@opencut/timeline";
import { ArrowLeft, Save } from "../lib/icons";
import "./opencut.css";
import "./editor-shell.css";

export type OpenCutSource =
  | {
      /** A Recall media URL (a rendered clip); fetched once into the project. */
      url: string;
      /** File name shown in OpenCut's media bin. */
      name: string;
    }
  | {
      /**
       * A long source (a full VOD) streamed by byte range, never copied into
       * memory or browser storage. `stream` turns this ref into URLs.
       */
      streamRef: string;
      name: string;
    };

/**
 * What the primary button does instead of exporting an MP4 (e.g. the Cutting
 * Room turns the timeline into Recall clips). `run` resolves to the message
 * shown once it's done; throw to report a failure.
 */
export interface OpenCutAction {
  label: string;
  busyLabel: string;
  /** The button's icon; defaults to Save. */
  icon?: ReactNode;
  /**
   * The button as the timeline stands right now ("Make 3 clips", or disabled
   * with a reason). Re-evaluated whenever the timeline changes.
   */
  stateFor?: (editor: EditorCore) => { label: string; disabled?: boolean; hint?: string };
  /** One line under the title saying what the button does. */
  note: string;
  /** `report` updates the button while it runs ("Rendering 2 of 3…"). */
  run: (editor: EditorCore, report: (label: string) => void) => Promise<string>;
}

export interface OpenCutEditorProps {
  /** Stable key for this edit (e.g. `clip:<id>`); reopening resumes it. */
  sessionKey: string;
  title: string;
  /** What the project starts with. None: an empty editor, ready for a drop. */
  source?: OpenCutSource;
  /**
   * A scratch project: every open starts empty, and the previous scratch
   * project is deleted rather than resumed.
   */
  ephemeral?: boolean;
  /** Canvas the project starts on. Recall clips are 1080x1920. */
  canvas?: { width: number; height: number };
  /** Receives the exported MP4. Throw to report a failed save. Unused with `action`. */
  onSave?: (mp4: Blob) => Promise<void>;
  /** Replaces "Save to Recall" (export an MP4) with a Recall action. */
  action?: OpenCutAction;
  /** Render inside a Recall view instead of over the whole window. */
  inline?: boolean;
  /** More buttons for the top bar, before the primary one. */
  extraActions?: ReactNode;
  /**
   * Where the timeline opens: a moment to centre (and how many seconds to
   * show around it). A streamed source with no focus opens zoomed to fit.
   */
  focus?: TimelineViewRequest | null;
  /** Adopt videos the creator imports (register them, stream from disk). */
  adoptImports?: ImportAdopter;
  /** J/K/L shuttle (forward/back at 1-8x, K stops) instead of OpenCut's 1 s jumps. */
  shuttle?: boolean;
  /** Taller tracks by type (the Cutting Room's stream track). Default: OpenCut's. */
  trackHeights?: Partial<Record<TrackType, number>>;
  onClose: () => void;
  /** What Recall knows about the clip (its captions), for the editor's tabs. */
  host?: RecallEditorHost;
  /**
   * Turns a streamed source's `streamRef` into this launch's URLs (the source,
   * and the engine's waveform and filmstrips for it). Saved projects keep the
   * ref, because engine URLs change every launch.
   */
  stream?: StreamHost;
}

const PROJECTS_KEY = "recall-opencut-projects";
const NO_HOST: RecallEditorHost = {};
/** Projects whose first-open import is in flight. */
const seeding = new Set<string>();

type SaveState = { phase: "idle" } | { phase: "exporting"; progress: number } | { phase: "saving" } | { phase: "saved" } | { phase: "error"; message: string };

function readProjectMap(): Record<string, string> {
  try {
    const parsed = JSON.parse(localStorage.getItem(PROJECTS_KEY) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function rememberProject(sessionKey: string, projectId: string) {
  try {
    localStorage.setItem(PROJECTS_KEY, JSON.stringify({ ...readProjectMap(), [sessionKey]: projectId }));
  } catch {
    /* storage unavailable: the edit just won't resume */
  }
}

/** Resolve this session's OpenCut project, creating it on first open. */
async function resolveProject(sessionKey: string, title: string, ephemeral = false): Promise<{ id: string; fresh: boolean }> {
  const known = readProjectMap()[sessionKey];
  const editor = EditorCore.getInstance();
  if (known && ephemeral) {
    await editor.project.loadAllProjects();
    if (editor.project.getSavedProjects().some((project) => project.id === known)) {
      await editor.project.deleteProjects({ ids: [known] }).catch(() => undefined);
    }
  } else if (known) {
    await editor.project.loadAllProjects();
    if (editor.project.getSavedProjects().some((project) => project.id === known)) return { id: known, fresh: false };
  }
  const id = await editor.project.createNewProject({ name: title });
  rememberProject(sessionKey, id);
  return { id, fresh: true };
}

export function OpenCutEditor({ sessionKey, title, source, ephemeral = false, canvas = { width: 1080, height: 1920 }, onSave, action, inline = false, focus, onClose, host = NO_HOST, stream, trackHeights, shuttle = false, extraActions, adoptImports }: OpenCutEditorProps) {
  // Set during render, not in an effect: the editor loads saved projects
  // (and their streamed assets) in its own child effects, which run first.
  setStreamResolver(stream);
  setTrackHeightOverrides(trackHeights);
  setImportAdopter(adoptImports);
  const [project, setProject] = useState<{ id: string; fresh: boolean } | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [portalHost, setPortalHost] = useState<HTMLElement | null>(null);

  // Menus, dialogs and drag ghosts portal here: inside the .oc-root scope,
  // outside any backdrop-filter that would trap position: fixed.
  useEffect(() => {
    const host = document.createElement("div");
    host.className = "oc-root oc-portal-host";
    document.body.appendChild(host);
    setPortalHost(host);
    return () => host.remove();
  }, []);

  useEffect(() => {
    let cancelled = false;
    setProject(null);
    resolveProject(sessionKey, title, ephemeral)
      .then((resolved) => { if (!cancelled) setProject(resolved); })
      .catch((error: unknown) => { if (!cancelled) setFailure(error instanceof Error ? error.message : "The editor couldn't start."); });
    return () => { cancelled = true; };
  }, [sessionKey, title, ephemeral]);

  useEffect(() => {
    if (!project) return;
    setOpenCutRouterHost({
      projectId: project.id,
      onNavigate: (to) => {
        const next = /^\/editor\/([^/?#]+)/.exec(to)?.[1];
        if (next) {
          rememberProject(sessionKey, next);
          setProject({ id: next, fresh: true });
        } else {
          onClose();
        }
      },
    });
  }, [project, sessionKey, onClose]);

  // A streamed VOD always opens framed: at a moment, or fitted whole.
  const view = focus ?? (source && "streamRef" in source ? {} : null);

  const surface = (
    <div
      className={`oc-root oce ${inline ? "is-inline" : ""}`}
      role={inline ? "region" : "dialog"}
      aria-modal={inline ? undefined : true}
      aria-label={`Edit ${title}`}
    >
      <RecallHostContext.Provider value={host}>
      <TooltipProvider>
        {failure ? (
          <div className="oce-state">
            <p>{failure}</p>
            <button type="button" className="oce-btn" onClick={onClose}>Back</button>
          </div>
        ) : !project ? (
          <div className="oce-state"><p>Opening the editor…</p></div>
        ) : (
          <EditorProvider key={project.id} projectId={project.id}>
            <EditorTopBar title={title} onSave={onSave} action={action} onClose={onClose} extraActions={extraActions} />
            <SeedSource projectId={project.id} fresh={project.fresh} source={source} canvas={canvas} />
            <ApplyView projectId={project.id} view={view} />
            {host.framing && <FramingSync projectId={project.id} framing={host.framing} />}
            {host.captionTrack && <CaptionSync projectId={project.id} captions={host.captionTrack} />}
            {host.openClip && <OpenClipSync projectId={project.id} clip={host.openClip} />}
            {shuttle && <ShuttleKeys />}
            <div className="oce-body">
              <EditorLayout />
            </div>
            <MigrationDialog />
          </EditorProvider>
        )}
        <Toaster position="bottom-center" theme="system" toastOptions={{ className: "oce-toast" }} />
      </TooltipProvider>
      </RecallHostContext.Provider>
    </div>
  );
  if (!portalHost) return null;
  return inline ? surface : createPortal(surface, document.body);
}

/**
 * Keeps the Cutting Room framed: the canvas is the clip's 9:16 frame and every
 * piece shows the way Recall will render it. Re-applies on any timeline or
 * media change, when new layers arrive, and as the playhead crosses into a new
 * stretch of the reel. applyFraming is a no-op when nothing differs.
 */
function FramingSync({ projectId, framing }: { projectId: string; framing: import("@opencut/recall/host").RecallFramingHost }) {
  const editor = useEditor();
  const activeId = useEditor((e) => e.project.getActiveOrNull()?.metadata.id);
  const mediaLoading = useEditor((e) => e.media.isLoadingMedia());
  const scene = useEditor((e) => e.scenes.getActiveSceneOrNull());
  const assets = useEditor((e) => e.media.getAssets().length);
  const canvasKey = useEditor((e) => {
    const size = e.project.getActiveOrNull()?.settings.canvasSize;
    return size ? `${size.width}x${size.height}` : "";
  });
  // Two-minute stretches of the reel: framing follows the playhead without
  // asking the engine again on every frame.
  const stretch = useEditor((e) => Math.floor(e.playback.getCurrentTime() / TICKS_PER_SECOND / 120));

  useEffect(() => {
    if (activeId !== projectId || mediaLoading) return;
    const project = editor.project.getActiveOrNull();
    if (!project) return;
    const { width, height } = framing.canvas;
    if (project.settings.canvasSize.width !== width || project.settings.canvasSize.height !== height) {
      void editor.project.updateSettings({ settings: { canvasSize: { width, height }, canvasSizeMode: "custom" }, pushHistory: false });
      return;
    }
    // Fades follow a piece's ends through trims and splits, then the framing
    // (and its facecam copies) follow the pieces.
    syncFades(editor);
    applyFraming(editor, framing.lookup, stretch * 120 + 60);
  }, [editor, projectId, activeId, mediaLoading, scene, assets, canvasKey, stretch, framing]);
  return null;
}

/** Selects (or cuts) the clip opened for editing, once the stream is on the timeline. */
function OpenClipSync({ projectId, clip }: { projectId: string; clip: RecallOpenClip }) {
  const editor = useEditor();
  const activeId = useEditor((e) => e.project.getActiveOrNull()?.metadata.id);
  const mediaLoading = useEditor((e) => e.media.isLoadingMedia());
  const scene = useEditor((e) => e.scenes.getActiveSceneOrNull());
  const placed = useRef<string | null>(null);
  useEffect(() => {
    if (activeId !== projectId || mediaLoading || placed.current === clip.key) return;
    if (placeOpenClip(editor, clip)) placed.current = clip.key;
  }, [editor, projectId, activeId, mediaLoading, scene, clip]);
  return null;
}

/** Longest piece Recall reads words for (a clip is at most three minutes). */
const CAPTION_MAX_SECONDS = 180;

/**
 * Keeps the caption track showing each piece's words: reads the words for a
 * piece that has none for its window yet (from the scan; decoding waits for
 * the creator to ask), then rebuilds the track.
 */
function CaptionSync({ projectId, captions }: { projectId: string; captions: RecallCaptionTrackHost }) {
  const editor = useEditor();
  const activeId = useEditor((e) => e.project.getActiveOrNull()?.metadata.id);
  const mediaLoading = useEditor((e) => e.media.isLoadingMedia());
  const scene = useEditor((e) => e.scenes.getActiveSceneOrNull());
  const pending = useRef(new Set<string>());

  useEffect(() => {
    if (activeId !== projectId || mediaLoading) return;
    if (captions.enabled) {
      for (const ref of streamRefsOnTimeline(editor)) {
        for (const piece of streamElements(editor, ref)) {
          if (piece.reel || piece.end - piece.start > CAPTION_MAX_SECONDS) continue;
          const window = { start: piece.start, end: piece.end };
          if (coversWindow(readCaptions(piece.element), window)) continue;
          const key = `${piece.element.id}@${piece.start.toFixed(2)}-${piece.end.toFixed(2)}`;
          if (pending.current.has(key)) continue;
          pending.current.add(key);
          void captions.loadWords(ref, piece.start, piece.end, false)
            .then(({ source, words }) => {
              // The piece may have moved on while the words were read.
              const now = streamElements(editor, ref).find((item) => item.element.id === piece.element.id);
              if (!now || Math.abs(now.start - piece.start) > 0.05 || Math.abs(now.end - piece.end) > 0.05) return;
              storeCaptionWords(editor, now, source, words);
            })
            .catch((error) => console.warn("Recall couldn't read this piece's words", error))
            .finally(() => pending.current.delete(key));
        }
      }
    }
    applyCaptionTrack(editor, captions.enabled ? captionStyleOverrides(captions.style) : null);
  }, [editor, projectId, activeId, mediaLoading, scene, captions]);
  return null;
}

/** J/K/L shuttle, claimed before OpenCut's own J/K/L bindings see the keys. */
function ShuttleKeys() {
  const editor = useEditor();
  const [speed, setSpeed] = useState(0);
  useEffect(() => {
    const shuttle = new Shuttle(editor);
    const off = shuttle.onChange(setSpeed);
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.repeat) return;
      const key = event.key.toLowerCase();
      if (key !== "j" && key !== "k" && key !== "l") return;
      const target = event.target as HTMLElement | null;
      if (target && isTypableDOMElement({ element: target })) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      shuttle.press(key);
    };
    // Window capture runs before OpenCut's document-capture keybindings.
    window.addEventListener("keydown", onKey, { capture: true });
    return () => {
      window.removeEventListener("keydown", onKey, { capture: true });
      off();
      shuttle.stop();
    };
  }, [editor]);
  if (speed === 0 || speed === 1) return null;
  return (
    <div className="oce-shuttle" role="status" aria-live="polite">
      {speed < 0 ? `◀ ${-speed}×` : `▶ ${speed}×`}
    </div>
  );
}

/** Frames the timeline once per project open (and when the focus moves). */
function ApplyView({ projectId, view }: { projectId: string; view: TimelineViewRequest | null }) {
  const time = view?.time;
  const span = view?.span;
  const wanted = view !== null;
  useEffect(() => {
    if (wanted) requestTimelineView({ time, span });
  }, [projectId, wanted, time, span]);
  return null;
}

/** Recall's bar in place of OpenCut's website header. */
function EditorTopBar({ title, onSave, action, onClose, extraActions }: Pick<OpenCutEditorProps, "title" | "onSave" | "action" | "onClose" | "extraActions">) {
  const editor = useEditor();
  // Re-render on timeline changes so the action's live label stays current.
  useEditor((e) => e.scenes.getActiveSceneOrNull());
  const live = action?.stateFor?.(editor);
  const exportState = useEditor((e) => e.project.getExportState());
  const [state, setState] = useState<SaveState>({ phase: "idle" });
  const [doneMessage, setDoneMessage] = useState<string | null>(null);
  const [busyText, setBusyText] = useState<string | null>(null);
  const busy = state.phase === "exporting" || state.phase === "saving";
  const shownState: SaveState = exportState.isExporting ? { phase: "exporting", progress: exportState.progress } : state;

  // "Saved" is a moment, not a mode: settle back so the next save reads normally.
  useEffect(() => {
    if (state.phase !== "saved") return;
    const timer = window.setTimeout(() => setState({ phase: "idle" }), action ? 6000 : 2400);
    return () => window.clearTimeout(timer);
  }, [state.phase, action]);

  const close = useCallback(async () => {
    if (exportState.isExporting) editor.project.cancelExport();
    try {
      await editor.project.prepareExit();
    } finally {
      editor.project.closeProject();
      onClose();
    }
  }, [editor, exportState.isExporting, onClose]);

  const save = async () => {
    const active = editor.project.getActiveOrNull();
    if (!active || busy) return;
    if (action) {
      setState({ phase: "saving" });
      setBusyText(null);
      try {
        setDoneMessage(await action.run(editor, setBusyText));
        setState({ phase: "saved" });
      } catch (error) {
        setState({ phase: "error", message: error instanceof Error ? error.message : "That didn't work. Try again." });
      } finally {
        setBusyText(null);
      }
      return;
    }
    if (!onSave) return;
    setState({ phase: "exporting", progress: 0 });
    const result = await editor.project.export({
      options: { format: "mp4", quality: "very_high", fps: active.settings.fps, includeAudio: true },
    });
    editor.project.clearExportState();
    if (result.cancelled) { setState({ phase: "idle" }); return; }
    if (!result.success || !result.buffer) {
      setState({ phase: "error", message: result.error || "The export didn't finish." });
      return;
    }
    setState({ phase: "saving" });
    try {
      await onSave(new Blob([result.buffer], { type: "video/mp4" }));
      setState({ phase: "saved" });
    } catch (error) {
      setState({ phase: "error", message: error instanceof Error ? error.message : "Recall couldn't keep the edit." });
    }
  };

  const label =
    shownState.phase === "exporting" ? `Exporting ${Math.round(shownState.progress * 100)}%`
    : shownState.phase === "saving" ? (action ? busyText ?? action.busyLabel : "Saving…")
    : shownState.phase === "saved" && !action ? "Saved"
    : action ? live?.label ?? action.label : "Save to Recall";

  return (
    <header className="oce-top">
      <button type="button" className="oce-back" onClick={() => void close()}>
        <ArrowLeft aria-hidden="true" /> Back
      </button>
      <div className="oce-title">
        <b className="disp">{title}</b>
        <span title={action?.note}>{action ? live?.hint ?? action.note : "Exports use your edit. Recall's own cut stays as it was."}</span>
      </div>
      <div className="oce-actions">
        {shownState.phase === "error" && <span className="oce-error" role="alert">{shownState.message}</span>}
        {shownState.phase === "saved" && action && doneMessage && <span className="oce-done" role="status">{doneMessage}</span>}
        {shownState.phase === "exporting" && (
          <button type="button" className="oce-btn" onClick={() => editor.project.cancelExport()}>Cancel</button>
        )}
        {extraActions}
        <button
          type="button"
          className="oce-btn is-primary"
          onClick={() => void save()}
          disabled={busy || (!!live?.disabled && shownState.phase !== "saving")}
          aria-busy={busy}
          style={shownState.phase === "exporting" ? { ["--oce-progress" as string]: `${shownState.progress * 100}%` } : undefined}
        >
          {action?.icon ?? <Save aria-hidden="true" />} {label}
        </button>
      </div>
    </header>
  );
}

/**
 * Puts the source on the timeline whenever the project opens with an empty
 * one: first open, or a reopen after the timeline was cleared (the Cutting
 * Room's Open video reuses a video's project, and "nothing happened" is not an
 * answer). A copy already in the media bin is reused rather than imported
 * again. Runs outside undo history, so Undo can't empty it.
 */
function SeedSource({ projectId, fresh, source, canvas }: { projectId: string; fresh: boolean; source?: OpenCutSource; canvas: { width: number; height: number } }) {
  const editor = useEditor();
  // Re-run once the project's saved media has loaded: deciding before that
  // would import a second copy of a video that is about to appear in the bin.
  const mediaLoading = useEditor((e) => e.media.isLoadingMedia());
  const activeId = useEditor((e) => e.project.getActiveOrNull()?.metadata.id);

  useEffect(() => {
    const active = editor.project.getActiveOrNull();
    if (!active || activeId !== projectId || mediaLoading) return;
    const scene = editor.scenes.getActiveSceneOrNull();
    if (!scene) return;
    // Anything on the timeline is the creator's work: never add to it.
    const hasTimeline = [...scene.tracks.overlay, scene.tracks.main, ...scene.tracks.audio].some((track) => track.elements.length > 0);
    if (hasTimeline) return;
    // Keyed by project, not by mount: StrictMode's double effect must not
    // start two imports, nor cancel the one that's running.
    if (seeding.has(projectId)) return;
    seeding.add(projectId);
    const stillOpen = () => editor.project.getActiveOrNull()?.metadata.id === projectId;
    (async () => {
      if (fresh) await editor.project.updateSettings({ settings: { canvasSize: canvas, canvasSizeMode: "custom" }, pushHistory: false });
      if (!source) return;
      const assets = editor.media.getAssets();
      const existing = "streamRef" in source
        ? assets.find((asset) => streamedMedia(asset.file)?.ref === source.streamRef)
        : assets.find((asset) => asset.name === source.name);
      if (existing) {
        const element = buildElementFromMedia({
          mediaId: existing.id,
          mediaType: existing.type,
          name: existing.name,
          duration: existing.duration != null ? mediaTimeFromSeconds({ seconds: existing.duration }) : DEFAULT_NEW_ELEMENT_DURATION,
          startTime: mediaTimeFromSeconds({ seconds: 0 }),
        });
        new InsertElementCommand({ element, placement: { mode: "auto", trackType: "video" } }).execute();
        await editor.project.saveCurrentProject();
        return;
      }
      let file: File;
      if ("streamRef" in source) {
        const size = await remoteSize(resolveStreamUrl(source.streamRef));
        if (!stillOpen()) return;
        file = streamedFile({ ref: source.streamRef, name: source.name, size });
      } else {
        const response = await fetch(source.url);
        if (!response.ok) throw new Error(`Couldn't load the clip (${response.status}).`);
        const blob = await response.blob();
        if (!stillOpen()) return;
        file = new File([blob], source.name, { type: blob.type || "video/mp4" });
      }
      const [asset] = await processMediaAssets({ files: [file] });
      if (!asset || !stillOpen()) return;
      const addMedia = new AddMediaAssetCommand({ projectId, asset });
      const element = buildElementFromMedia({
        mediaId: addMedia.getAssetId(),
        mediaType: asset.type,
        name: asset.name,
        duration: asset.duration != null ? mediaTimeFromSeconds({ seconds: asset.duration }) : DEFAULT_NEW_ELEMENT_DURATION,
        startTime: mediaTimeFromSeconds({ seconds: 0 }),
      });
      new BatchCommand([addMedia, new InsertElementCommand({ element, placement: { mode: "auto", trackType: "video" } })]).execute();
      await editor.project.saveCurrentProject();
    })()
      .catch((error: unknown) => console.error("OpenCut seed failed", error))
      .finally(() => seeding.delete(projectId));
  }, [editor, projectId, activeId, mediaLoading, fresh, source, canvas]);
  return null;
}

/** OpenCut's editor page layout (upstream app/editor/[project_id]/page.tsx). */
function EditorLayout() {
  usePasteMedia();
  const { panels, setPanel } = usePanelStore();
  const activeScene = useEditor((editor) => editor.scenes.getActiveSceneOrNull());
  const currentTime = useEditor((editor) => editor.playback.getCurrentTime());
  const activeGuide = usePreviewStore((state) => state.activeGuide);
  const overlays = usePreviewStore((state) => state.overlays);
  const setOverlayVisibility = usePreviewStore((state) => state.setOverlayVisibility);
  const showBookmarkNotes = isPreviewOverlayVisible({ overlay: bookmarkNotesPreviewOverlay, overlays });

  const overlaySource = useMemo(
    () =>
      mergePreviewOverlaySources({
        sources: [
          getGuidePreviewOverlaySource({ guideId: activeGuide }),
          activeScene
            ? getBookmarkPreviewOverlaySource({ bookmarks: activeScene.bookmarks, time: currentTime, isVisible: showBookmarkNotes })
            : { definitions: [bookmarkNotesPreviewOverlay], instances: [] },
        ],
      }),
    [activeGuide, activeScene, currentTime, showBookmarkNotes],
  );
  const overlayControls = useMemo(
    () => overlaySource.definitions.map((overlay) => createPreviewOverlayControl({ overlay, overlays })),
    [overlaySource.definitions, overlays],
  );

  return (
    <ResizablePanelGroup
      direction="vertical"
      className="size-full gap-[0.18rem]"
      onLayout={(sizes) => {
        setPanel({ panel: "mainContent", size: sizes[0] ?? panels.mainContent });
        setPanel({ panel: "timeline", size: sizes[1] ?? panels.timeline });
      }}
    >
      <ResizablePanel defaultSize={panels.mainContent} minSize={30} maxSize={85} className="min-h-0">
        <ResizablePanelGroup
          direction="horizontal"
          className="size-full gap-[0.19rem] px-3"
          onLayout={(sizes) => {
            setPanel({ panel: "tools", size: sizes[0] ?? panels.tools });
            setPanel({ panel: "preview", size: sizes[1] ?? panels.preview });
            setPanel({ panel: "properties", size: sizes[2] ?? panels.properties });
          }}
        >
          <ResizablePanel defaultSize={panels.tools} minSize={15} maxSize={40} className="min-w-0">
            <AssetsPanel />
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={panels.preview} minSize={30} className="min-h-0 min-w-0 flex-1">
            <PreviewPanel
              overlayControls={overlayControls}
              overlayInstances={overlaySource.instances}
              onOverlayVisibilityChange={setOverlayVisibility}
            />
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={panels.properties} minSize={15} maxSize={40} className="min-w-0">
            <PropertiesPanel />
          </ResizablePanel>
        </ResizablePanelGroup>
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel defaultSize={panels.timeline} minSize={15} maxSize={70} className="min-h-0 px-3 pb-3">
        <Timeline />
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}

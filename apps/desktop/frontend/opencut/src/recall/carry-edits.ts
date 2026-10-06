// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Carrying Cutting Room edits into the clips Recall renders.
 *
 * Recall renders each piece as a vertical clip (layout, facecam, captions).
 * What the creator added over that piece in the Cutting Room (text,
 * stickers, graphics, effects, extra media, and effects on the piece itself)
 * is lifted off the 16:9 timeline, retimed to the clip, mapped onto the 9:16
 * canvas and composited over Recall's render with OpenCut's own exporter.
 * The open project is never switched: the composition is built and exported
 * on the side (the same buildScene/SceneExporter path as a normal export).
 */
import type { EditorCore } from "@opencut/core";
import type { MediaAsset } from "@opencut/media/types";
import type { ExportQuality } from "@opencut/export";
import { createTimelineAudioBuffer } from "@opencut/media/audio";
import { processMediaAssets } from "@opencut/media/processing";
import { SceneExporter } from "@opencut/services/renderer/scene-exporter";
import { buildScene } from "@opencut/services/renderer/scene-builder";
import type { SceneTracks, TimelineElement, TimelineTrack, VideoElement } from "@opencut/timeline";
import { buildElementFromMedia } from "@opencut/timeline/element-utils";
import { TICKS_PER_SECOND, mediaTimeFromSeconds, type MediaTime } from "@opencut/wasm";
import type { StreamElement } from "./cutting";
import { isCaptionTrack } from "./caption-track";
import { isFramingTrack } from "./framing";
import { streamedMedia } from "./streamed-media";

const seconds = (time: number) => time / TICKS_PER_SECOND;
const ticks = (secs: number): MediaTime => mediaTimeFromSeconds({ seconds: secs });
/** Edits shorter than this after clamping to the piece are dropped. */
const MIN_EDIT_SECONDS = 0.05;

export interface CanvasSize {
	width: number;
	height: number;
}

/** How a 16:9 Cutting Room position/scale lands on a clip canvas. */
export function canvasMapping(from: CanvasSize, to: CanvasSize) {
	const scale = to.width / from.width;
	return {
		x: to.width / from.width,
		y: to.height / from.height,
		// Size keeps its share of the frame's width (a title across half the
		// stream stays across half the clip).
		scale,
		// OpenCut already sizes text by canvas height (fontSize x height /
		// reference), so text grows with the taller canvas on its own; undo that.
		textScale: scale * (from.height / to.height),
	};
}

type Mapping = ReturnType<typeof canvasMapping>;

const PARAM_FACTORS: Record<string, "x" | "y" | "scale"> = {
	"transform.positionX": "x",
	"transform.positionY": "y",
	"transform.scaleX": "scale",
	"transform.scaleY": "scale",
};

/** The factor for a param on this element (text sizes by canvas height already). */
function factorFor(key: string, mapping: Mapping, isText: boolean): number {
	const factor = PARAM_FACTORS[key];
	if (!factor) return 1;
	return factor === "scale" && isText ? mapping.textScale : mapping[factor];
}

function mapParams(params: TimelineElement["params"], mapping: Mapping, isText: boolean): TimelineElement["params"] {
	const next = { ...params } as Record<string, unknown>;
	for (const key of Object.keys(PARAM_FACTORS)) {
		if (typeof next[key] === "number") next[key] = (next[key] as number) * factorFor(key, mapping, isText);
	}
	return next as TimelineElement["params"];
}

/** Keyframes: shift by `shift` seconds (head trimmed) and map transform values. */
function mapAnimations(animations: TimelineElement["animations"], mapping: Mapping, shift: number, isText: boolean): TimelineElement["animations"] {
	if (!animations) return animations;
	const out: Record<string, unknown> = {};
	for (const [path, channel] of Object.entries(animations)) {
		const factor = factorFor(path, mapping, isText);
		const mapChannel = (value: unknown): unknown => {
			if (!value || typeof value !== "object") return value;
			if (Array.isArray((value as { keys?: unknown }).keys)) {
				const ch = value as { keys: Record<string, unknown>[] };
				return {
					...ch,
					keys: ch.keys.map((key) => {
						const next: Record<string, unknown> = { ...key, time: ticks(seconds(key.time as number) - shift) };
						if (typeof key.value === "number") next.value = (key.value as number) * factor;
						// Bezier handles hold a value delta (dv) that scales with the value.
						for (const handle of ["leftHandle", "rightHandle"]) {
							const h = key[handle] as { dt: number; dv: number } | undefined;
							if (h && typeof h.dv === "number") next[handle] = { ...h, dv: h.dv * factor };
						}
						return next;
					}),
				};
			}
			// Composite channel data: map each component.
			return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, mapChannel(v)]));
		};
		out[path] = mapChannel(channel);
	}
	return out as TimelineElement["animations"];
}

// Any registered video (this stream or another Recall renders) is footage,
// not an edit to composite over the clip.
const isStreamElement = (element: TimelineElement, assets: Map<string, MediaAsset>, _streamRef: string) =>
	element.type === "video" && !!streamedMedia(assets.get(element.mediaId)?.file);

/**
 * One edit lifted onto a clip that starts at timeline second `at` and runs
 * `length` seconds: retimed, trimmed to fit, mapped to the clip canvas.
 */
function liftEdit(element: TimelineElement, at: number, length: number, mapping: Mapping): TimelineElement | null {
	const start = seconds(element.startTime) - at;
	const end = start + seconds(element.duration);
	const head = Math.max(0, -start);
	const tail = Math.max(0, end - length);
	const duration = seconds(element.duration) - head - tail;
	if (duration < MIN_EDIT_SECONDS) return null;
	const lifted = {
		...element,
		startTime: ticks(Math.max(0, start)),
		duration: ticks(duration),
		trimStart: ticks(seconds(element.trimStart) + head),
		trimEnd: ticks(seconds(element.trimEnd) + tail),
		params: "transform.positionX" in (element.params ?? {}) || "transform.scaleX" in (element.params ?? {})
			? mapParams(element.params, mapping, element.type === "text")
			: element.params,
		animations: mapAnimations(element.animations, mapping, head, element.type === "text"),
	} as TimelineElement;
	return lifted;
}

export interface PieceEdits {
	/** Edits over the piece, lifted onto the clip, per track. */
	tracks: SceneTracks;
	/** The piece's own clip effects, carried onto Recall's render. */
	pieceEffects?: VideoElement["effects"];
	count: number;
}

/**
 * What the creator added over `piece` (a stream piece on the timeline),
 * lifted onto a clip canvas. `count` is 0 when there is nothing to carry.
 */
export function editsOverPiece(editor: EditorCore, streamRef: string, piece: StreamElement, clipCanvas: CanvasSize): PieceEdits {
	const scene = editor.scenes.getActiveScene();
	const project = editor.project.getActive();
	const assets = new Map(editor.media.getAssets().map((asset) => [asset.id, asset]));
	const mapping = canvasMapping(project.settings.canvasSize, clipCanvas);
	const length = piece.end - piece.start;
	let count = 0;
	const lift = <T extends TimelineTrack>(track: T): T => {
		const elements = (track.elements as TimelineElement[])
			.filter((element) => !isStreamElement(element, assets, streamRef) && !("hidden" in element && element.hidden))
			.map((element) => liftEdit(element, piece.at, length, mapping))
			.filter((element): element is TimelineElement => !!element);
		count += elements.length;
		return { ...track, elements } as T;
	};
	const tracks: SceneTracks = {
		overlay: scene.tracks.overlay.filter((track) => !isFramingTrack(track) && !isCaptionTrack(track)).map(lift),
		main: lift(scene.tracks.main),
		audio: scene.tracks.audio.map(lift),
	};
	const pieceEffects = piece.element.effects?.length ? piece.element.effects : undefined;
	if (pieceEffects) count += 1;
	return { tracks, pieceEffects, count };
}

/**
 * Composite `edits` over Recall's rendered clip (an MP4 at `clipUrl`) and
 * return the finished MP4. Uses the project's media for anything the edits
 * reference (stickers, images, imported clips).
 */
export async function composeOverClip({
	editor,
	edits,
	clipUrl,
	clipName,
	clipCanvas,
	quality = "high",
	onProgress,
}: {
	editor: EditorCore;
	edits: PieceEdits;
	clipUrl: string;
	clipName: string;
	clipCanvas: CanvasSize;
	quality?: ExportQuality;
	onProgress?: (progress: number) => void;
}): Promise<ArrayBuffer> {
	const response = await fetch(clipUrl, { cache: "no-store" });
	if (!response.ok) throw new Error(`Recall's render of this clip isn't available (${response.status}).`);
	const blob = await response.blob();
	const [processed] = await processMediaAssets({ files: [new File([blob], clipName, { type: blob.type || "video/mp4" })] });
	if (!processed?.duration) throw new Error("Recall's render of this clip couldn't be read.");
	const clipAsset: MediaAsset = { ...processed, id: `recall-clip-${clipName}` };
	const project = editor.project.getActive();

	const clipElement = {
		...buildElementFromMedia({
			mediaId: clipAsset.id,
			mediaType: "video",
			name: clipName,
			duration: ticks(processed.duration),
			startTime: ticks(0),
		}),
		id: `recall-clip-element-${clipName}`,
		effects: edits.pieceEffects,
	} as VideoElement;

	// Recall's render is the base; the edits' own main-track media go above it.
	const tracks: SceneTracks = {
		overlay: [
			...edits.tracks.overlay,
			...(edits.tracks.main.elements.length ? [{ ...edits.tracks.main, id: `${edits.tracks.main.id}-lifted` } as SceneTracks["overlay"][number]] : []),
		],
		main: { ...edits.tracks.main, elements: [clipElement] },
		audio: edits.tracks.audio,
	};
	const mediaAssets = [...editor.media.getAssets(), clipAsset];
	const duration = ticks(processed.duration);

	onProgress?.(0.02);
	const audioBuffer = await createTimelineAudioBuffer({ tracks, mediaAssets, duration });
	const scene = buildScene({
		tracks,
		mediaAssets,
		duration,
		canvasSize: clipCanvas,
		background: { type: "color", color: "#000000" } as typeof project.settings.background,
	});
	const exporter = new SceneExporter({
		width: clipCanvas.width,
		height: clipCanvas.height,
		fps: project.settings.fps,
		format: "mp4",
		quality,
		shouldIncludeAudio: true,
		audioBuffer: audioBuffer || undefined,
	});
	exporter.on("progress", (progress) => onProgress?.(0.05 + progress * 0.95));
	const buffer = await exporter.export({ rootNode: scene });
	if (!buffer) throw new Error("OpenCut didn't finish rendering your edits.");
	return buffer;
}

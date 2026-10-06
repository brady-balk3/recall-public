// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. The Cutting Room edits on the clip's own 9:16 canvas, framed
 * the way Recall will render it: the engine describes a layout as rectangles
 * (a source region of the 16:9 frame stretched onto a target region of the
 * canvas; engines/export/composition.py plan_layers), and this turns them into
 * plain OpenCut state.
 *
 * - The piece itself carries the first layer (gameplay): a transform that puts
 *   the region on its target, and a rectangle mask that shows only the region.
 * - Every further layer (the facecam) is a copy of the piece on a Recall-owned
 *   track directly above the piece's track, kept in step with it: split, trim
 *   or move a piece and its facecam follows. Those tracks are derived, never
 *   saved as the creator's work, and are skipped when Recall reads pieces.
 *
 * Nothing here enters undo history: framing is a view of the pieces, not an
 * edit to them.
 */
import type { EditorCore } from "@opencut/core";
import type { MediaAsset } from "@opencut/media/types";
import type { RectangleMask } from "@opencut/masks/types";
import type { OverlayTrack, SceneTracks, TimelineElement, TimelineTrack, VideoElement, VideoTrack } from "@opencut/timeline";
import { TICKS_PER_SECOND } from "@opencut/wasm";
import { streamedMedia } from "./streamed-media";

/** [x, y, w, h]: normalized on the 16:9 source, or px on the canvas. */
export type Rect = [number, number, number, number];

export interface FramingLayer {
	role: "gameplay" | "facecam" | "camera" | string;
	source: Rect;
	target: Rect;
}

export interface FramingPiece {
	/** Source seconds of the stream this element shows. */
	start: number;
	end: number;
	/** The whole, untrimmed stream (the reel) rather than a cut piece. */
	reel: boolean;
}

/**
 * Layers for a piece of a stream, or undefined while they're being fetched.
 * `playhead` (timeline seconds) lets the reel be framed for where you are.
 */
export type FramingLookup = (streamRef: string, piece: FramingPiece, playhead: number) => FramingLayer[] | undefined;

export const FRAMING_TRACK_PREFIX = "recall-framing:";
const FRAMING_TRACK_NAME = "Facecam (Recall)";
const FRAMING_ELEMENT_SUFFIX = "~recall-layer";
const SOURCE_ASPECT = 16 / 9;

export function isFramingTrack(track: { id?: string }): boolean {
	return Boolean(track.id?.startsWith(FRAMING_TRACK_PREFIX));
}

const seconds = (value: number) => value / TICKS_PER_SECOND;

function streamPiece(element: VideoElement, asset: MediaAsset): FramingPiece {
	const sourceDuration = element.sourceDuration != null ? seconds(element.sourceDuration) : asset.duration ?? 0;
	const start = seconds(element.trimStart);
	const end = Math.max(start, sourceDuration - seconds(element.trimEnd));
	return { start, end, reel: start < 0.05 && sourceDuration - end < 0.05 };
}

/**
 * Transform and mask that land `layer.source` exactly on `layer.target`.
 * OpenCut fits a video inside the canvas, centres it, then scales and offsets
 * it; masks are fractions of the video, measured from its centre.
 */
export function layerState(layer: FramingLayer, canvas: { width: number; height: number }) {
	const [sx, sy, sw, sh] = layer.source;
	const [tx, ty, tw, th] = layer.target;
	const contain = Math.min(canvas.width / SOURCE_ASPECT, canvas.height);
	const baseW = contain * SOURCE_ASPECT;
	const baseH = contain;
	const scaleX = tw / Math.max(1e-6, sw * baseW);
	const scaleY = th / Math.max(1e-6, sh * baseH);
	const positionX = tx + tw / 2 - canvas.width / 2 - (sx + sw / 2 - 0.5) * baseW * scaleX;
	const positionY = ty + th / 2 - canvas.height / 2 - (sy + sh / 2 - 0.5) * baseH * scaleY;
	const mask: RectangleMask = {
		id: "recall-framing",
		type: "rectangle",
		params: {
			centerX: sx + sw / 2 - 0.5,
			centerY: sy + sh / 2 - 0.5,
			width: sw,
			height: sh,
			rotation: 0,
			scale: 1,
			feather: 0,
			inverted: false,
			strokeColor: "#ffffff",
			strokeWidth: 0,
			strokeAlign: "center",
		},
	};
	return {
		params: {
			"transform.positionX": round(positionX),
			"transform.positionY": round(positionY),
			"transform.scaleX": round(scaleX),
			"transform.scaleY": round(scaleY),
			"transform.rotate": 0,
		},
		mask,
	};
}

const round = (value: number) => Math.round(value * 1000) / 1000;

function sameState(element: VideoElement, state: ReturnType<typeof layerState>): boolean {
	const params = element.params ?? {};
	for (const [key, value] of Object.entries(state.params)) {
		if (Math.abs(Number(params[key] ?? NaN) - value) > 1e-3) return false;
	}
	const mask = element.masks?.find((candidate) => candidate.id === state.mask.id) as RectangleMask | undefined;
	if (!mask || element.masks?.length !== 1) return false;
	return (["centerX", "centerY", "width", "height"] as const).every((key) => Math.abs(mask.params[key] - state.mask.params[key]) < 1e-4);
}

function withState(element: VideoElement, state: ReturnType<typeof layerState>): VideoElement {
	return { ...element, params: { ...element.params, ...state.params }, masks: [state.mask] };
}

/**
 * Bring the timeline's framing in line with `lookup`. Returns true when it
 * changed anything. Safe to call on every timeline change: it does nothing
 * when the framing already matches.
 */
export function applyFraming(editor: EditorCore, lookup: FramingLookup, playhead = 0): boolean {
	const scene = editor.scenes.getActiveSceneOrNull();
	const project = editor.project.getActiveOrNull();
	if (!scene || !project) return false;
	const canvas = project.settings.canvasSize;
	const assets = new Map(editor.media.getAssets().map((asset) => [asset.id, asset]));

	let changed = false;
	const frame = (track: VideoTrack): { track: VideoTrack; layer: VideoTrack | null } => {
		const derived: VideoElement[] = [];
		let trackChanged = false;
		const elements = track.elements.map((element) => {
			if (element.type !== "video") return element;
			const asset = assets.get(element.mediaId);
			const ref = asset ? streamedMedia(asset.file)?.ref : undefined;
			if (!asset || !ref) return element;
			const layers = lookup(ref, streamPiece(element, asset), playhead);
			if (!layers?.length) return element;
			const [first, ...rest] = layers;
			const state = layerState(first, canvas);
			rest.forEach((layer, index) => {
				derived.push({
					...withState(element, layerState(layer, canvas)),
					id: `${element.id}${FRAMING_ELEMENT_SUFFIX}${index}`,
					name: layer.role === "facecam" ? "Facecam" : layer.role,
					// The piece underneath carries the sound.
					isSourceAudioEnabled: false,
				});
			});
			if (sameState(element, state)) return element;
			trackChanged = true;
			return withState(element, state);
		}) as VideoTrack["elements"];
		if (trackChanged) changed = true;
		const layer: VideoTrack | null = derived.length
			? { id: `${FRAMING_TRACK_PREFIX}${track.id}`, type: "video", name: FRAMING_TRACK_NAME, elements: derived, muted: true, hidden: false }
			: null;
		return { track: trackChanged ? { ...track, elements } : track, layer };
	};

	// Overlay tracks draw top-first. Each creator track gets its framing layer
	// directly above it; stale framing tracks are dropped.
	const overlay: OverlayTrack[] = [];
	const previous = new Map(scene.tracks.overlay.filter(isFramingTrack).map((track) => [track.id, track]));
	const place = (layer: VideoTrack | null) => {
		if (!layer) return;
		const before = previous.get(layer.id);
		if (!before || signature(before) !== signature(layer)) changed = true;
		previous.delete(layer.id);
		overlay.push(before && signature(before) === signature(layer) ? before : layer);
	};
	for (const track of scene.tracks.overlay) {
		if (isFramingTrack(track)) continue;
		if (track.type !== "video") { overlay.push(track); continue; }
		const { track: next, layer } = frame(track);
		place(layer);
		overlay.push(next);
	}
	const main = frame(scene.tracks.main);
	place(main.layer);
	if (previous.size) changed = true;
	// Keep the creator's own track order; only framing tracks moved.
	const orderChanged = overlay.map((track) => track.id).join("|") !== scene.tracks.overlay.map((track) => track.id).join("|");
	if (!changed && !orderChanged) return false;
	const tracks: SceneTracks = { ...scene.tracks, overlay, main: main.track };
	editor.timeline.updateTracks(tracks);
	return true;
}

function signature(track: TimelineTrack): string {
	return (track.elements as TimelineElement[]).map((element) => {
		const video = element as VideoElement;
		const mask = video.masks?.[0] as RectangleMask | undefined;
		return [
			element.id, element.startTime, element.duration, element.trimStart, element.trimEnd,
			video.params?.["transform.positionX"], video.params?.["transform.positionY"],
			video.params?.["transform.scaleX"], video.params?.["transform.scaleY"],
			mask?.params.centerX, mask?.params.centerY, mask?.params.width, mask?.params.height,
			// Fades ride along on the facecam copy.
			JSON.stringify(video.animations?.opacity ?? null),
		].join(",");
	}).join(";");
}

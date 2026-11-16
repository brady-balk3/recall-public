// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. The full 16:9 frames behind the Cutting Room's 9:16 preview.
 *
 * The preview renderer decodes each video's whole frame, then crops and places
 * it (framing.ts turns Recall's layout into a transform plus a rectangle mask).
 * The renderer hands this module what it decoded on every frame it draws, so
 * the full-frame pane shows the same frame, at the same moment, with the exact
 * regions the clip is cut from. Nothing is decoded twice.
 */
import type { Mask, RectangleMask } from "@opencut/masks/types";
import { FRAMING_ELEMENT_SUFFIX, FRAMING_MASK_ID, type Rect } from "./framing";

export interface SourceFrame {
	mediaId: string;
	elementId?: string;
	/** The decoded frame, whole. Drawn straight away: the cache reuses it. */
	canvas: CanvasImageSource & { width: number; height: number };
	masks: Mask[];
}

export interface SourceRegion {
	/** "clip": the piece's own window (gameplay). "cam": a facecam layer. */
	role: "clip" | "cam";
	/** Normalized on the source frame. */
	rect: Rect;
}

/** The part of the frame an element shows: its framing mask, or all of it. */
export function regionOf(frame: SourceFrame): SourceRegion {
	const role = frame.elementId?.includes(FRAMING_ELEMENT_SUFFIX) ? "cam" : "clip";
	const mask = frame.masks.find((candidate) => candidate.id === FRAMING_MASK_ID) as RectangleMask | undefined;
	if (!mask) return { role, rect: [0, 0, 1, 1] };
	const { centerX, centerY, width, height } = mask.params;
	return { role, rect: [centerX + 0.5 - width / 2, centerY + 0.5 - height / 2, width, height] };
}

/**
 * What the full-frame pane shows for one rendered frame: the first video the
 * clip is built from (not a facecam copy), and every region taken from it.
 */
export function pickSource(frames: SourceFrame[]): { frame: SourceFrame; regions: SourceRegion[] } | null {
	const base = frames.find((frame) => !frame.elementId?.includes(FRAMING_ELEMENT_SUFFIX)) ?? frames[0];
	if (!base) return null;
	return { frame: base, regions: frames.filter((frame) => frame.mediaId === base.mediaId).map(regionOf) };
}

type Listener = (frames: SourceFrame[]) => void;
const listeners = new Set<Listener>();
let last: SourceFrame[] = [];

/** Called by the preview renderer with everything it decoded for a frame. */
export function publishSourceFrames(frames: SourceFrame[]): void {
	last = frames;
	for (const listener of listeners) listener(frames);
}

/** The last frame's sources, for a late redraw (a mask that just loaded, while paused). */
export function latestSourceFrames(): SourceFrame[] {
	return last;
}

export function subscribeSourceFrames(listener: Listener): () => void {
	listeners.add(listener);
	return () => { listeners.delete(listener); };
}

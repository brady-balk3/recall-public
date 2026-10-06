// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The Cutting Room's timeline, read and edited in terms of the stream.
 *
 * The whole, untrimmed stream on the timeline is the reel: what the creator
 * browses. It is never a clip (unless the whole video is short enough to be
 * one). Everything trimmed from the stream is a piece, and each piece becomes
 * a Recall clip. "Cut" drops a moment's piece on a
 * track above the reel, at the same time, so the two line up.
 */
import type { EditorCore } from "@opencut/core";
import type { MediaAsset } from "@opencut/media/types";
import type { TimelineElement, VideoElement } from "@opencut/timeline";
import { buildElementFromMedia } from "@opencut/timeline/element-utils";
import { TICKS_PER_SECOND, mediaTimeFromSeconds, type MediaTime } from "@opencut/wasm";
import { isFramingTrack } from "./framing";
import { streamedMedia } from "./streamed-media";

export interface StreamPiece {
	/** Source seconds of the streamed VOD. */
	start: number;
	end: number;
}

/**
 * The longest clip Recall makes. A whole, untrimmed video longer than this is
 * the reel; a short recording on its own is a clip.
 */
export const MAX_CLIP_SECONDS = 180;

/** The Recall clip a piece made (its id), kept on the piece. */
export const CLIP_PARAM = "recall.clip";

/** A little context either side of a cut moment, like the scan's own clips. */
const CUT_PAD_SECONDS = 2;

export interface StreamElement {
	element: VideoElement;
	asset: MediaAsset;
	trackId: string;
	/** Source seconds this element shows. */
	start: number;
	end: number;
	/** Where on the timeline it begins, in seconds. */
	at: number;
	reel: boolean;
}

const seconds = (ticks: number) => ticks / TICKS_PER_SECOND;

/**
 * The title the creator gave a piece (its name on the timeline), or undefined
 * while it still carries the video's own name.
 */
export function customTitle(element: VideoElement, asset: MediaAsset): string | undefined {
	// Splitting names the halves "<name> (left)" and "<name> (right)".
	const name = element.name?.trim().replace(/(\s\((left|right)\))+$/, "");
	return name && name !== asset.name && name !== asset.name.replace(/\.[^.]+$/, "") ? name : undefined;
}

const ticks = (secs: number): MediaTime => mediaTimeFromSeconds({ seconds: secs });

export function streamElements(editor: EditorCore, streamRef: string): StreamElement[] {
	const scene = editor.scenes.getActiveSceneOrNull();
	if (!scene) return [];
	const assets = new Map(editor.media.getAssets().map((asset) => [asset.id, asset]));
	// Recall's framing tracks are views of the pieces, never pieces themselves.
	const tracks = [...scene.tracks.overlay.filter((track) => !isFramingTrack(track)), scene.tracks.main, ...scene.tracks.audio];
	const out: StreamElement[] = [];
	for (const track of tracks) {
		for (const element of track.elements as TimelineElement[]) {
			if (element.type !== "video") continue;
			const asset = assets.get(element.mediaId);
			if (!asset || streamedMedia(asset.file)?.ref !== streamRef) continue;
			// trimEnd is source time cut from the end, so this holds under retime too.
			const sourceDuration = element.sourceDuration != null ? seconds(element.sourceDuration) : asset.duration ?? 0;
			const start = seconds(element.trimStart);
			const end = sourceDuration - seconds(element.trimEnd);
			if (end <= start) continue;
			out.push({
				element,
				asset,
				trackId: track.id,
				start,
				end,
				at: seconds(element.startTime),
				reel: element.trimStart === 0 && element.trimEnd === 0 && sourceDuration > MAX_CLIP_SECONDS,
			});
		}
	}
	return out.sort((a, b) => a.at - b.at);
}

/**
 * Every piece of the stream on the timeline, in source seconds, in timeline
 * order. The untrimmed reel is not a piece; other media is ignored.
 */
export function streamPieces(editor: EditorCore, streamRef: string): StreamPiece[] {
	return streamElements(editor, streamRef)
		.filter((item) => !item.reel)
		.map(({ start, end }) => ({ start, end }));
}

/** Every registered video (stream ref) with something on the timeline. */
export function streamRefsOnTimeline(editor: EditorCore): string[] {
	const scene = editor.scenes.getActiveSceneOrNull();
	if (!scene) return [];
	const assets = new Map(editor.media.getAssets().map((asset) => [asset.id, asset]));
	const refs = new Set<string>();
	for (const track of [...scene.tracks.overlay.filter((track) => !isFramingTrack(track)), scene.tracks.main]) {
		for (const element of track.elements as TimelineElement[]) {
			if (element.type !== "video") continue;
			const ref = streamedMedia(assets.get(element.mediaId)?.file)?.ref;
			if (ref) refs.add(ref);
		}
	}
	return [...refs];
}

/**
 * Names of videos on the timeline Recall can't make clips from: copied into
 * the editor before it could register them (no file on disk to render).
 */
export function unregisteredVideosOnTimeline(editor: EditorCore): string[] {
	const scene = editor.scenes.getActiveSceneOrNull();
	if (!scene) return [];
	const assets = new Map(editor.media.getAssets().map((asset) => [asset.id, asset]));
	const names = new Set<string>();
	for (const track of [...scene.tracks.overlay.filter((track) => !isFramingTrack(track)), scene.tracks.main]) {
		for (const element of track.elements as TimelineElement[]) {
			if (element.type !== "video") continue;
			const asset = assets.get(element.mediaId);
			if (asset && !streamedMedia(asset.file)) names.add(asset.name);
		}
	}
	return [...names];
}

/** Where a source second shows on the timeline (the reel first), if anywhere. */
export function timelineTimeForSource(editor: EditorCore, streamRef: string, sourceSec: number): number | null {
	const items = streamElements(editor, streamRef).filter((item) => !item.element.retime && sourceSec >= item.start && sourceSec <= item.end);
	const best = items.find((item) => item.reel) ?? items[0];
	return best ? best.at + (sourceSec - best.start) : null;
}

/** The source second shown at a timeline time (a piece over the reel), if any. */
export function sourceTimeAt(editor: EditorCore, streamRef: string, timelineSec: number): number | null {
	const items = streamElements(editor, streamRef).filter((item) => timelineSec >= item.at && timelineSec <= item.at + (item.end - item.start));
	const best = items.find((item) => !item.reel) ?? items[0];
	return best ? best.start + (timelineSec - best.at) : null;
}

/** Whether a piece already covers most of [start, end]. */
export function isCut(editor: EditorCore, streamRef: string, start: number, end: number): boolean {
	const span = Math.max(0.001, end - start);
	return streamPieces(editor, streamRef).some((piece) => {
		const overlap = Math.min(piece.end, end) - Math.max(piece.start, start);
		return overlap / span >= 0.8;
	});
}

/**
 * Put [start, end] of the stream on the timeline as a piece, lined up with
 * the reel when there is one (else after everything). Returns false when the
 * stream isn't in the project.
 */
export function cutPiece(
	editor: EditorCore,
	streamRef: string,
	range: StreamPiece,
	{ pad = CUT_PAD_SECONDS, name }: { pad?: number; name?: string } = {},
): boolean {
	const items = streamElements(editor, streamRef);
	const asset = items[0]?.asset
		?? editor.media.getAssets().find((candidate) => streamedMedia(candidate.file)?.ref === streamRef);
	if (!asset) return false;
	const sourceDuration = asset.duration ?? 0;
	const start = Math.max(0, range.start - pad);
	const end = sourceDuration > 0 ? Math.min(sourceDuration, range.end + pad) : range.end + pad;
	const aligned = timelineTimeForSource(editor, streamRef, start);
	const at = aligned ?? items.reduce((latest, item) => Math.max(latest, item.at + (item.end - item.start)), 0);
	const element = {
		...buildElementFromMedia({
			mediaId: asset.id,
			mediaType: "video",
			name: name ?? asset.name,
			duration: ticks(end - start),
			startTime: ticks(at),
		}),
		trimStart: ticks(start),
		trimEnd: ticks(Math.max(0, sourceDuration - end)),
		sourceDuration: ticks(sourceDuration),
	} as VideoElement;
	editor.timeline.insertElement({ element, placement: { mode: "auto", trackType: "video" } });
	return true;
}

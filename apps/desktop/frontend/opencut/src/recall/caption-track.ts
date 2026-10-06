// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. The Cutting Room's caption track: the words each piece's
 * clip will burn in, on a Recall-owned text track above the video, styled the
 * way Recall renders them, so the preview shows the clip's captions.
 *
 * - A piece carries its own words (source seconds, with the creator's edits)
 *   in a param, so they survive saving, splitting and trimming with it.
 * - The track is derived from those words and rebuilt when they change. Text
 *   typed into a caption on the preview is read back into the piece's words
 *   first, so editing there sticks.
 * - The track is never carried onto clips as an edit: Recall burns the words
 *   in itself, and "Make clip" sends the corrected text along (captionText).
 */
import type { EditorCore } from "@opencut/core";
import type { MediaAsset } from "@opencut/media/types";
import { buildSubtitleTextElement } from "@opencut/subtitles/build-subtitle-text-element";
import type { SubtitleStyleOverrides } from "@opencut/subtitles/types";
import type { TextElement, TextTrack, TimelineTrack, VideoElement } from "@opencut/timeline";
import { TICKS_PER_SECOND } from "@opencut/wasm";
import { chunkCaptionWords } from "./caption-chunks";
import { streamedMedia } from "./streamed-media";
import type { RecallCaptionWord } from "./host";

export const CAPTION_PARAM = "recall.captions";
export const CAPTION_TRACK_PREFIX = "recall-captions:";
const CAPTION_TRACK_ID = `${CAPTION_TRACK_PREFIX}track`;
const CAPTION_TRACK_NAME = "Captions (Recall)";
export const CAPTION_ELEMENT_SUFFIX = "~recall-cap";

/** Where a piece's words came from: the scan, a decode of its own seconds, or nowhere yet. */
export type CaptionSource = "scan" | "decode" | "none";

export interface PieceCaptions {
	/** The source window these words were read for (seconds). */
	from: number;
	to: number;
	source: CaptionSource;
	/** Timed words, in source seconds. */
	words: RecallCaptionWord[];
	/** The creator changed the words. */
	edited: boolean;
}

export function isCaptionTrack(track: { id?: string }): boolean {
	return Boolean(track.id?.startsWith(CAPTION_TRACK_PREFIX));
}

const seconds = (ticks: number) => ticks / TICKS_PER_SECOND;
const round = (value: number) => Math.round(value * 1000) / 1000;

export function readCaptions(element: VideoElement): PieceCaptions | undefined {
	const raw = element.params?.[CAPTION_PARAM];
	if (typeof raw !== "string") return undefined;
	try {
		const data = JSON.parse(raw) as { f: number; t: number; s: CaptionSource; e?: 1; w: [string, number, number][] };
		return {
			from: data.f,
			to: data.t,
			source: data.s,
			edited: data.e === 1,
			words: data.w.map(([word, start, end]) => ({ word, start, end })),
		};
	} catch {
		return undefined;
	}
}

export function captionParam(captions: PieceCaptions): string {
	return JSON.stringify({
		f: round(captions.from),
		t: round(captions.to),
		s: captions.source,
		...(captions.edited ? { e: 1 } : {}),
		w: captions.words.map((word) => [word.word, round(word.start), round(word.end)]),
	});
}

/** A piece's source window. */
export interface PieceWindow {
	start: number;
	end: number;
}

/** Whether the stored words were read for (at least) this window. */
export function coversWindow(captions: PieceCaptions | undefined, piece: PieceWindow): boolean {
	return Boolean(captions && captions.from <= piece.start + 0.05 && captions.to >= piece.end - 0.05);
}

/** The words inside the piece, the way the engine slices them (touching counts). */
export function wordsIn(captions: PieceCaptions, piece: PieceWindow): RecallCaptionWord[] {
	return captions.words.filter((word) => word.end >= piece.start && word.start <= piece.end);
}

/**
 * Newly read words for a window, keeping the creator's words where the old
 * window already had them: an edit made before a trim outlives the trim.
 */
export function mergeCaptions(previous: PieceCaptions | undefined, window: PieceWindow, source: CaptionSource, words: RecallCaptionWord[]): PieceCaptions {
	if (!previous?.edited) return { from: window.start, to: window.end, source, words, edited: false };
	const outside = words.filter((word) => word.end < previous.from || word.start > previous.to);
	const kept = previous.words.filter((word) => word.end >= window.start && word.start <= window.end);
	return {
		from: Math.min(window.start, previous.from),
		to: Math.max(window.end, previous.to),
		source,
		words: [...outside, ...kept].sort((a, b) => a.start - b.start),
		edited: true,
	};
}

/**
 * The words with the span [from, to] (source seconds) replaced by `text`.
 * Unchanged words keep their times; a changed run shares the old run's span by
 * character length (the engine re-times the same way at render, onto its own
 * decode, so this only has to look right in the preview).
 */
export function replaceSpan(captions: PieceCaptions, from: number, to: number, text: string): PieceCaptions {
	const inside = captions.words.filter((word) => word.start >= from - 1e-3 && word.end <= to + 1e-3);
	const before = captions.words.filter((word) => word.start < from - 1e-3);
	const after = captions.words.filter((word) => !before.includes(word) && !inside.includes(word));
	const next = text.split(/\s+/).map((word) => word.trim()).filter(Boolean);
	const oldText = inside.map((word) => word.word.trim()).join(" ");
	if (next.join(" ") === oldText) return captions;
	let replaced: RecallCaptionWord[];
	if (next.length === inside.length) {
		replaced = inside.map((word, index) => ({ ...word, word: next[index] }));
	} else if (next.length) {
		const spanStart = inside[0]?.start ?? from;
		const spanEnd = inside[inside.length - 1]?.end ?? to;
		const weights = next.map((word) => Math.max(1, word.length));
		const total = weights.reduce((sum, weight) => sum + weight, 0);
		let at = spanStart;
		replaced = next.map((word, index) => {
			const length = ((spanEnd - spanStart) * weights[index]) / total;
			const out = { word, start: at, end: at + length };
			at += length;
			return out;
		});
	} else {
		replaced = [];
	}
	return { ...captions, words: [...before, ...replaced, ...after], edited: true };
}

/**
 * Store words read for a piece's window (clip-relative, as the engine sends
 * them) on the piece. `fresh` drops the creator's edits (start over); else
 * edits made before are kept where they still apply.
 */
export function storeCaptionWords(
	editor: EditorCore,
	piece: { trackId: string; element: VideoElement; start: number; end: number },
	source: CaptionSource,
	words: RecallCaptionWord[],
	{ fresh = false }: { fresh?: boolean } = {},
): void {
	const window = { start: piece.start, end: piece.end };
	const merged = mergeCaptions(fresh ? undefined : readCaptions(piece.element), window, source,
		words.map((word) => ({ word: word.word, start: word.start + piece.start, end: word.end + piece.start })));
	editor.timeline.updateElements({
		updates: [{ trackId: piece.trackId, elementId: piece.element.id, patch: { params: { ...piece.element.params, [CAPTION_PARAM]: captionParam(merged) } } }],
		pushHistory: false,
	});
}

/** Store edited words on the piece, as an undoable edit. */
export function saveCaptions(editor: EditorCore, piece: { trackId: string; element: VideoElement }, captions: PieceCaptions): void {
	editor.timeline.updateElements({
		updates: [{ trackId: piece.trackId, elementId: piece.element.id, patch: { params: { ...piece.element.params, [CAPTION_PARAM]: captionParam(captions) } } }],
	});
}

export function captionText(words: RecallCaptionWord[]): string {
	return words.map((word) => word.word.trim()).filter(Boolean).join(" ");
}

interface CaptionPiece {
	trackId: string;
	element: VideoElement;
	asset: MediaAsset;
	window: PieceWindow;
	/** Timeline seconds where the piece begins. */
	at: number;
}

/** Every piece that has words to show: a trimmed stream element with captions stored. */
function captionPieces(editor: EditorCore): CaptionPiece[] {
	const scene = editor.scenes.getActiveSceneOrNull();
	if (!scene) return [];
	const assets = new Map(editor.media.getAssets().map((asset) => [asset.id, asset]));
	const out: CaptionPiece[] = [];
	for (const track of [...scene.tracks.overlay, scene.tracks.main] as TimelineTrack[]) {
		if (track.type !== "video" || track.id.startsWith("recall-")) continue;
		for (const element of track.elements as VideoElement[]) {
			if (element.type !== "video") continue;
			const asset = assets.get(element.mediaId);
			if (!asset || !streamedMedia(asset.file)) continue;
			const sourceDuration = element.sourceDuration != null ? seconds(element.sourceDuration) : asset.duration ?? 0;
			const start = seconds(element.trimStart);
			const end = sourceDuration - seconds(element.trimEnd);
			if (end <= start) continue;
			out.push({ trackId: track.id, element, asset, window: { start, end }, at: seconds(element.startTime) });
		}
	}
	return out;
}

export interface CaptionChunk {
	/** Source seconds this cue covers. */
	from: number;
	to: number;
	text: string;
}

/** The cues a piece's clip shows, in order (the same chunking the preview uses). */
export function pieceChunks(captions: PieceCaptions, window: PieceWindow): CaptionChunk[] {
	const words = wordsIn(captions, window);
	return chunkCaptionWords({ words }).map((cue) => {
		const inside = words.filter((word) => word.start >= cue.startTime - 1e-3 && word.start < cue.startTime + cue.duration - 1e-3);
		return {
			from: cue.startTime,
			to: inside.length ? inside[inside.length - 1].end : cue.startTime + cue.duration,
			text: cue.text,
		};
	});
}

const flat = (text: unknown) => String(text ?? "").replace(/\s+/g, " ").trim();

/**
 * Bring the caption track in line with the pieces' words. `style` null means
 * captions are off: the track goes. Returns true when it changed anything.
 */
export function applyCaptionTrack(editor: EditorCore, style: SubtitleStyleOverrides | null): boolean {
	const scene = editor.scenes.getActiveSceneOrNull();
	const project = editor.project.getActiveOrNull();
	if (!scene || !project) return false;
	const existing = scene.tracks.overlay.find(isCaptionTrack) as TextTrack | undefined;
	const pieces = captionPieces(editor);

	// Text typed into a caption on the preview goes back into its piece's words.
	if (existing && style) {
		const updates: { trackId: string; elementId: string; patch: Partial<VideoElement> }[] = [];
		for (const piece of pieces) {
			let captions = readCaptions(piece.element);
			if (!captions) continue;
			const chunks = pieceChunks(captions, piece.window);
			let changed = false;
			chunks.forEach((chunk, index) => {
				const shown = existing.elements.find((element) => element.id === `${piece.element.id}${CAPTION_ELEMENT_SUFFIX}${index}`);
				// A cue is named after the text Recall gave it, so a difference is
				// the creator typing; a cue whose words changed elsewhere (and got
				// re-chunked) still matches its name and is simply rebuilt.
				if (!shown || flat(shown.params.content) === flat(shown.name) || flat(shown.name) !== flat(chunk.text)) return;
				captions = replaceSpan(captions as PieceCaptions, chunk.from, chunk.to, flat(shown.params.content));
				changed = true;
			});
			if (changed) {
				updates.push({ trackId: piece.trackId, elementId: piece.element.id, patch: { params: { ...piece.element.params, [CAPTION_PARAM]: captionParam(captions) } } });
			}
		}
		if (updates.length) {
			editor.timeline.updateElements({ updates, pushHistory: true });
			return true;
		}
	}

	const elements: TextElement[] = [];
	if (style) {
		const canvasSize = project.settings.canvasSize;
		for (const piece of pieces) {
			const captions = readCaptions(piece.element);
			if (!captions) continue;
			pieceChunks(captions, piece.window).forEach((chunk, index) => {
				const startTime = piece.at + Math.max(0, chunk.from - piece.window.start);
				const endTime = Math.min(piece.at + (piece.window.end - piece.window.start), piece.at + (chunk.to - piece.window.start));
				if (endTime - startTime < 0.05) return;
				const built = buildSubtitleTextElement({
					index,
					caption: { text: chunk.text, startTime, duration: Math.max(0.3, endTime - startTime), style },
					canvasSize,
				});
				elements.push({ ...built, id: `${piece.element.id}${CAPTION_ELEMENT_SUFFIX}${index}`, name: chunk.text } as TextElement);
			});
		}
	}

	const next: TextTrack | null = elements.length
		? { id: CAPTION_TRACK_ID, type: "text", name: CAPTION_TRACK_NAME, elements, hidden: false }
		: null;
	if (signature(existing) === signature(next) && (!next || scene.tracks.overlay[0]?.id === CAPTION_TRACK_ID)) return false;
	const overlay = scene.tracks.overlay.filter((track) => !isCaptionTrack(track));
	// Captions draw over everything else, like the burned-in ones.
	editor.timeline.updateTracks({ ...scene.tracks, overlay: next ? [next, ...overlay] : overlay });
	return true;
}

function signature(track: TextTrack | null | undefined): string {
	if (!track) return "";
	return track.elements.map((element) => [
		element.id, element.name, element.startTime, element.duration, flat(element.params.content),
		element.params.fontSize, element.params.color, element.params.fontFamily,
	].join(",")).join(";");
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. The burned-in caption look, computed the way the engine
 * writes it (engines/caption/asr.py generate_tiktok_ass and
 * engines/caption/caption_style.py), so the Cutting Room's preview shows the
 * captions a render will carry: up to four words a line, a new line at a
 * pause, words appearing as they are spoken with the current one in the
 * highlight colour, bold text with a black outline, placed by the ASS margins.
 *
 * Pure: no DOM. caption-overlay.tsx draws what this returns.
 */
import type { RecallCaptionStyle, RecallCaptionWord } from "./host";

// engines/caption/asr.py
export const CHUNK_SIZE = 4;
export const CHUNK_GAP_SEC = 0.9;
export const LINGER_PAD_SEC = 0.45;
export const CAPTION_AUDIO_LAG_SEC = 0.12;
const COLLAPSE_DUR_SEC = 0.08;
const MIN_REFLOW_WORD_SEC = 0.22;
const MAX_REFLOW_WORD_SEC = 0.5;

// engines/caption/caption_style.py
export const PLAY_RES_X = 1080;
export const PLAY_RES_Y = 1920;
const SIZE_MAP: Record<RecallCaptionStyle["size"], number> = { small: 56, medium: 72, large: 92 };
const MARGIN_V: Record<RecallCaptionStyle["position"], number> = { bottom: 250, middle: 0, top: 130 };
const MARGIN_LR = 10;
export const DEFAULT_HIGHLIGHT = "#FFFF00";

export interface CaptionPart {
	text: string;
	active: boolean;
}

/** One on-screen state of the burned captions, in clip seconds. */
export interface CaptionEvent {
	start: number;
	end: number;
	parts: CaptionPart[];
}

/** _reflow_collapsed_words: spread runs of zero-length words at a reading pace. */
export function reflowCollapsedWords(words: RecallCaptionWord[], limit?: number): RecallCaptionWord[] {
	const out = words.map((word) => ({ ...word }));
	const n = out.length;
	let i = 0;
	while (i < n) {
		if (out[i].end - out[i].start > COLLAPSE_DUR_SEC) {
			i += 1;
			continue;
		}
		let j = i;
		while (j < n && out[j].end - out[j].start <= COLLAPSE_DUR_SEC) j += 1;
		const runStart = out[i].start;
		const boundary = j < n ? out[j].start : null;
		const count = j - i;
		let per = boundary != null && boundary > runStart
			? Math.max(MIN_REFLOW_WORD_SEC, Math.min(MAX_REFLOW_WORD_SEC, (boundary - runStart) / count))
			: MAX_REFLOW_WORD_SEC;
		let stop = boundary;
		if (boundary == null && limit != null && limit > runStart) {
			stop = limit;
			per = Math.min(per, (limit - runStart) / count);
		}
		let t = runStart;
		for (let k = i; k < j; k += 1) {
			let start = t;
			let end = t + per;
			if (stop != null) {
				start = Math.min(start, stop);
				end = Math.min(end, stop);
			}
			out[k].start = start;
			out[k].end = end;
			t += per;
		}
		i = j;
	}
	return out;
}

/** ASS times are centiseconds, truncated (format_time_ass). */
const cs = (seconds: number) => Math.floor(seconds * 100 + 1e-6) / 100;

/**
 * generate_tiktok_ass's events for clip-relative words. `duration` is the
 * clip's length: nothing is shown past its last frame.
 */
export function captionEvents(input: RecallCaptionWord[], duration?: number): CaptionEvent[] {
	const clean = input
		.map((word) => ({ word: String(word.word ?? ""), start: Number(word.start), end: Number(word.end) }))
		.filter((word) => Number.isFinite(word.start) && Number.isFinite(word.end));
	const limit = duration && duration > 0 ? duration : undefined;
	const words = reflowCollapsedWords(clean, limit);

	const chunks: RecallCaptionWord[][] = [];
	let current: RecallCaptionWord[] = [];
	for (const word of words) {
		if (current.length && (current.length >= CHUNK_SIZE || word.start - current[current.length - 1].end > CHUNK_GAP_SEC)) {
			chunks.push(current);
			current = [];
		}
		current.push(word);
	}
	if (current.length) chunks.push(current);

	const events: CaptionEvent[] = [];
	chunks.forEach((chunk, chunkIndex) => {
		const nextChunkStart = chunks[chunkIndex + 1]?.[0]?.start;
		chunk.forEach((target, i) => {
			const start = Math.max(0, target.start + CAPTION_AUDIO_LAG_SEC);
			const wordEnd = Math.max(start, target.end + CAPTION_AUDIO_LAG_SEC);
			let cappedEnd: number;
			if (i + 1 < chunk.length) {
				cappedEnd = Math.max(0, chunk[i + 1].start + CAPTION_AUDIO_LAG_SEC);
			} else {
				cappedEnd = wordEnd + LINGER_PAD_SEC;
				if (nextChunkStart != null) cappedEnd = Math.min(cappedEnd, Math.max(0, nextChunkStart + CAPTION_AUDIO_LAG_SEC));
			}
			let hardStop: number | null = null;
			if (i + 1 < chunk.length) hardStop = Math.max(0, chunk[i + 1].start + CAPTION_AUDIO_LAG_SEC);
			else if (nextChunkStart != null) hardStop = Math.max(0, nextChunkStart + CAPTION_AUDIO_LAG_SEC);
			cappedEnd = Math.max(cappedEnd, start + 0.12);
			if (hardStop != null) cappedEnd = Math.min(cappedEnd, hardStop);
			if (cappedEnd <= start) cappedEnd = start + 0.05;
			if (limit != null) {
				if (start >= limit) return;
				cappedEnd = Math.min(cappedEnd, limit);
			}
			events.push({
				start: cs(start),
				end: cs(cappedEnd),
				parts: chunk.slice(0, i + 1).map((word) => ({ text: word.word.trim(), active: word === target })),
			});
		});
	});
	return events;
}

/**
 * The event libass shows at `time`. Events can touch or (rarely) overlap;
 * the later one wins, which is what a single bottom line reads as.
 */
export function eventAt(events: CaptionEvent[], time: number): CaptionEvent | null {
	let found: CaptionEvent | null = null;
	for (const event of events) {
		if (event.start > time) break;
		if (time < event.end) found = event;
	}
	return found;
}

/** The ASS style line's numbers, in 1080x1920 script pixels. */
export interface CaptionMetrics {
	/** ASS Fontsize: the font's full line height (ascent + descent). */
	fontSize: number;
	outline: number;
	/** 2 bottom, 5 middle, 8 top (numpad, as ASS). */
	alignment: 2 | 5 | 8;
	marginV: number;
	marginLR: number;
	fontFamily: string;
	color: string;
	highlight: string;
}

/**
 * CaptionStyle, including facecam_safe: a top caption drops to the middle
 * when the clip has a facecam band at the top.
 */
export function captionMetrics(style: RecallCaptionStyle, hasTopFacecam = false): CaptionMetrics {
	const fontSize = SIZE_MAP[style.size] ?? SIZE_MAP.medium;
	const position = style.position === "top" && hasTopFacecam ? "middle" : style.position;
	return {
		fontSize,
		outline: Math.max(2, Math.round(fontSize / 14)),
		alignment: position === "top" ? 8 : position === "middle" ? 5 : 2,
		marginV: MARGIN_V[position] ?? MARGIN_V.bottom,
		marginLR: MARGIN_LR,
		fontFamily: style.fontFamily || "Arial Black",
		color: style.color || "#FFFFFF",
		highlight: style.highlightColor || DEFAULT_HIGHLIGHT,
	};
}

/**
 * Break a line the way libass's default smart wrap (WrapStyle 0) does when it
 * is wider than the frame: lines of near-equal width, the upper one the wider.
 * `measure` returns a run of words' width; returns word index ranges. Four
 * words almost always fit, so this is the rare long-word case.
 */
export function wrapParts(count: number, measure: (from: number, to: number) => number, maxWidth: number): Array<[number, number]> {
	if (count === 0) return [];
	if (count === 1 || measure(0, count) <= maxWidth) return [[0, count]];
	let best: { split: number; score: number } | null = null;
	for (let split = 1; split < count; split += 1) {
		const top = measure(0, split);
		const bottom = measure(split, count);
		// Balanced first; on a tie the wider line goes on top.
		const score = Math.max(top, bottom) + (top < bottom ? 0.5 : 0);
		if (!best || score < best.score) best = { split, score };
	}
	const split = best?.split ?? 1;
	const rest = wrapParts(count - split, (from, to) => measure(from + split, to + split), maxWidth)
		.map(([from, to]) => [from + split, to + split] as [number, number]);
	return [[0, split], ...rest];
}

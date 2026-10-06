// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. Turns Recall's timed caption words into short on-screen
 * cues the way its burned-in captions read: a few words at a time, a new cue
 * at a pause or the end of a sentence, never two cues on screen at once.
 */
import type { RecallCaptionWord } from "./host";

export interface CaptionCue {
	text: string;
	startTime: number;
	duration: number;
}

export const MAX_WORDS_PER_CUE = 3;
/** A silence this long starts a new cue even mid-phrase. */
export const PAUSE_BREAK_SECONDS = 0.35;
const MIN_CUE_SECONDS = 0.3;

export function chunkCaptionWords({
	words,
	offset = 0,
	maxWords = MAX_WORDS_PER_CUE,
	pauseBreak = PAUSE_BREAK_SECONDS,
}: {
	words: RecallCaptionWord[];
	/** Timeline seconds at which the clip's first frame plays. */
	offset?: number;
	maxWords?: number;
	pauseBreak?: number;
}): CaptionCue[] {
	const clean = words
		.map((word) => ({ ...word, word: String(word.word ?? "").trim() }))
		.filter((word) => word.word && Number.isFinite(word.start) && Number.isFinite(word.end))
		.sort((a, b) => a.start - b.start);

	const groups: RecallCaptionWord[][] = [];
	let current: RecallCaptionWord[] = [];
	for (const word of clean) {
		const previous = current[current.length - 1];
		const breakHere =
			current.length >= maxWords ||
			(previous && word.start - previous.end >= pauseBreak) ||
			(previous && /[.!?]$/.test(previous.word));
		if (breakHere && current.length) {
			groups.push(current);
			current = [];
		}
		current.push(word);
	}
	if (current.length) groups.push(current);

	const cues: CaptionCue[] = [];
	for (const [index, group] of groups.entries()) {
		const start = Math.max(0, group[0].start);
		const next = groups[index + 1]?.[0].start;
		let end = Math.max(group[group.length - 1].end, start + MIN_CUE_SECONDS);
		// Hand over cleanly: a cue never runs into the next one.
		if (next != null && end > next) end = Math.max(next, start + 0.05);
		cues.push({
			text: group.map((word) => word.word).join(" "),
			startTime: offset + start,
			duration: end - start,
		});
	}
	return cues;
}

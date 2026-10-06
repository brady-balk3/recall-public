// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Paging for a streamed source's timeline filmstrip.
 *
 * Each thumbnail stands for a fixed stretch of source time. That stretch comes
 * from a ladder growing by LADDER_RATIO, so at any zoom a thumbnail is between
 * 1x and LADDER_RATIO x its natural width and is cropped, never stretched.
 * Thumbnails come in pages of PAGE_FRAMES anchored at source 0, so a page's URL
 * depends only on the ladder step and its index: scrolling, trimming and
 * zooming back to a level all reuse the engine's cached contact sheets.
 */

export const PAGE_FRAMES = 12;
export const LADDER_RATIO = 1.25;
// Down to about one frame, so the deepest zoom still shows one frame per tile.
const LADDER_BASE_SECONDS = 1 / 64;
// The engine's filmstrip limits (apps/api/services.py filmstrip_path).
const MIN_FRAMES = 4;
const MAX_FRAMES = 48;
const MAX_IMAGE_WIDTH = 2560;
const MAX_CELL_WIDTH = 240;
const MAX_IMAGE_HEIGHT = 160;

/** Seconds per thumbnail: the smallest ladder step at least `tileWidthPx` wide. */
export function tileSeconds({
	pixelsPerSecond,
	tileWidthPx,
}: {
	pixelsPerSecond: number;
	tileWidthPx: number;
}): number {
	const target = tileWidthPx / Math.max(1e-9, pixelsPerSecond);
	const step = Math.max(0, Math.ceil(Math.log(target / LADDER_BASE_SECONDS) / Math.log(LADDER_RATIO) - 1e-9));
	return Number((LADDER_BASE_SECONDS * LADDER_RATIO ** step).toFixed(3));
}

export interface FilmstripPage {
	index: number;
	/** Source seconds this page covers. */
	start: number;
	end: number;
	frames: number;
	/** Source seconds per thumbnail on this page (the last page may differ). */
	cellSeconds: number;
}

/** Pages overlapping [fromSec, toSec] of the source, plus `margin` either side. */
export function visiblePages({
	tileSec,
	sourceDuration,
	fromSec,
	toSec,
	margin = 1,
}: {
	tileSec: number;
	sourceDuration: number;
	fromSec: number;
	toSec: number;
	margin?: number;
}): FilmstripPage[] {
	if (!(tileSec > 0) || !(sourceDuration > 0) || toSec < fromSec) return [];
	const pageSeconds = tileSec * PAGE_FRAMES;
	const lastIndex = Math.max(0, Math.ceil(sourceDuration / pageSeconds) - 1);
	const first = Math.max(0, Math.floor(Math.max(0, fromSec) / pageSeconds) - margin);
	const last = Math.min(lastIndex, Math.floor(Math.max(0, toSec) / pageSeconds) + margin);
	const pages: FilmstripPage[] = [];
	for (let index = first; index <= last; index++) {
		const start = Number((index * pageSeconds).toFixed(3));
		if (start >= sourceDuration) break;
		const end = Number(Math.min(sourceDuration, start + pageSeconds).toFixed(3));
		const cells = Math.ceil((end - start) / tileSec - 1e-6);
		const frames = Math.min(MAX_FRAMES, Math.max(MIN_FRAMES, cells));
		pages.push({ index, start, end, frames, cellSeconds: (end - start) / frames });
	}
	return pages;
}

/**
 * Image size to request for a page, in device pixels, keeping each cell at
 * `cellWidthPx` x `heightPx` (CSS px) proportions within the engine's limits.
 */
export function pageImageSize({
	frames,
	cellWidthPx,
	heightPx,
	devicePixelRatio,
}: {
	frames: number;
	cellWidthPx: number;
	heightPx: number;
	devicePixelRatio: number;
}): { width: number; height: number } {
	const aspect = cellWidthPx / heightPx;
	let cell = Math.round(cellWidthPx * devicePixelRatio);
	cell = Math.min(cell, MAX_CELL_WIDTH, Math.floor(MAX_IMAGE_WIDTH / frames), Math.floor(MAX_IMAGE_HEIGHT * aspect));
	return { width: cell * frames, height: Math.max(1, Math.round(cell / aspect)) };
}

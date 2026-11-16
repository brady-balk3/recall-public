// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. Paints one caption-render.ts event onto a 2D canvas the
 * way libass paints the engine's ASS line: bold, black outline of the style's
 * width on both sides of the stroke, the active word in the highlight colour,
 * centred, positioned by the ASS alignment and vertical margin.
 */
import { PLAY_RES_X, PLAY_RES_Y, wrapParts, type CaptionEvent, type CaptionMetrics } from "./caption-render";

/** CSS px for an ASS Fontsize: ASS sizes the full line (ascent + descent), CSS the em. */
const emPerLine = new Map<string, number>();
function fontPx(ctx: CanvasRenderingContext2D, family: string, assSize: number): number {
	let ratio = emPerLine.get(family);
	if (ratio == null) {
		ctx.font = `bold 100px "${family}"`;
		const m = ctx.measureText("Hg");
		const line = (m.fontBoundingBoxAscent ?? 0) + (m.fontBoundingBoxDescent ?? 0);
		ratio = line > 0 ? 100 / line : 1 / 1.2;
		emPerLine.set(family, ratio);
	}
	return assSize * ratio;
}

export function drawCaption(ctx: CanvasRenderingContext2D, event: CaptionEvent, metrics: CaptionMetrics, width: number, height: number): void {
	const sx = width / PLAY_RES_X;
	const sy = height / PLAY_RES_Y;
	const size = fontPx(ctx, metrics.fontFamily, metrics.fontSize) * sy;
	ctx.font = `bold ${size}px "${metrics.fontFamily}", sans-serif`;
	ctx.textBaseline = "alphabetic";
	ctx.textAlign = "left";
	const probe = ctx.measureText("Hg");
	const ascent = probe.fontBoundingBoxAscent ?? size * 0.8;
	const descent = probe.fontBoundingBoxDescent ?? size * 0.2;
	const lineHeight = ascent + descent;
	const space = ctx.measureText(" ").width;

	const words = event.parts;
	const widths = words.map((part) => ctx.measureText(part.text).width);
	const runWidth = (from: number, to: number) => widths.slice(from, to).reduce((sum, w) => sum + w, 0) + space * Math.max(0, to - from - 1);
	const lines = wrapParts(words.length, runWidth, (PLAY_RES_X - 2 * metrics.marginLR) * sx);
	const blockHeight = lines.length * lineHeight;
	let top: number;
	if (metrics.alignment === 8) top = metrics.marginV * sy;
	else if (metrics.alignment === 5) top = (height - blockHeight) / 2;
	else top = height - metrics.marginV * sy - blockHeight;

	const outline = metrics.outline * sy;
	ctx.lineJoin = "round";
	ctx.miterLimit = 2;
	lines.forEach(([from, to], row) => {
		const baseline = top + row * lineHeight + ascent;
		let x = width / 2 - runWidth(from, to) / 2;
		for (let index = from; index < to; index += 1) {
			const part = words[index];
			if (outline > 0) {
				ctx.strokeStyle = "#000";
				ctx.lineWidth = outline * 2;
				ctx.strokeText(part.text, x, baseline);
			}
			x += widths[index] + space;
		}
		x = width / 2 - runWidth(from, to) / 2;
		for (let index = from; index < to; index += 1) {
			const part = words[index];
			ctx.fillStyle = part.active ? metrics.highlight : metrics.color;
			ctx.fillText(part.text, x, baseline);
			x += widths[index] + space;
		}
	});
}

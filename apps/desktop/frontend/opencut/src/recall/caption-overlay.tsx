// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. The burned-in captions on the 9:16 preview, drawn the way
 * the render draws them (caption-render.ts mirrors the engine's ASS): bold
 * text with a black outline, the word being spoken in the highlight colour,
 * words appearing as they are said, at the render's size and margin.
 *
 * OpenCut's text elements can carry neither an outline nor a second colour,
 * so the caption track (caption-track.ts) stays on the timeline as the cue
 * blocks but is hidden from the preview, and this draws the picture instead.
 */
import { useEffect, useRef } from "react";
import type { EditorCore } from "@opencut/core";
import { mediaTimeToSeconds } from "@opencut/wasm";
import { useEditor } from "@opencut/editor/use-editor";
import { CAPTION_PARAM, captionPieces, readCaptions, wordsIn } from "./caption-track";
import { captionEvents, captionMetrics, eventAt, type CaptionEvent } from "./caption-render";
import { drawCaption } from "./caption-draw";
import type { RecallCaptionTrackHost } from "./host";

/** Events per piece, rebuilt only when its words or window change. */
const eventCache = new WeakMap<object, { key: string; events: CaptionEvent[] }>();

function piecesEvents(editor: EditorCore) {
	return captionPieces(editor).flatMap((piece) => {
		const raw = piece.element.params?.[CAPTION_PARAM];
		if (typeof raw !== "string") return [];
		const key = `${raw}|${piece.window.start}|${piece.window.end}`;
		const cached = eventCache.get(piece.element);
		let events = cached?.key === key ? cached.events : null;
		if (!events) {
			const captions = readCaptions(piece.element);
			if (!captions) return [];
			const words = wordsIn(captions, piece.window).map((word) => ({
				word: word.word,
				start: word.start - piece.window.start,
				end: word.end - piece.window.start,
			}));
			events = captionEvents(words, piece.window.end - piece.window.start);
			eventCache.set(piece.element, { key, events });
		}
		return [{ at: piece.at, length: piece.window.end - piece.window.start, events }];
	});
}

export function CaptionOverlay({ captions }: { captions: RecallCaptionTrackHost }) {
	const editor = useEditor();
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const latest = useRef(captions);
	latest.current = captions;
	const redraw = useRef<() => void>(() => {});

	useEffect(() => {
		let frame = 0;
		const draw = () => {
			frame = 0;
			const canvas = canvasRef.current;
			if (!canvas) return;
			const rect = canvas.getBoundingClientRect();
			const dpr = window.devicePixelRatio || 1;
			const w = Math.max(1, Math.round(rect.width * dpr));
			const h = Math.max(1, Math.round(rect.height * dpr));
			if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
			const ctx = canvas.getContext("2d");
			if (!ctx) return;
			ctx.clearRect(0, 0, w, h);
			const host = latest.current;
			if (!host.enabled) return;
			const time = mediaTimeToSeconds({ time: editor.playback.getCurrentTime() });
			for (const piece of piecesEvents(editor)) {
				const local = time - piece.at;
				if (local < 0 || local >= piece.length) continue;
				const event = eventAt(piece.events, local);
				if (event) drawCaption(ctx, event, captionMetrics(host.style, host.topFacecam), w, h);
				break;
			}
		};
		const schedule = () => { if (!frame) frame = requestAnimationFrame(draw); };
		redraw.current = schedule;
		const offs = [
			editor.playback.onUpdate(schedule),
			editor.playback.onSeek(schedule),
			editor.playback.subscribe(schedule),
			editor.timeline.subscribe(schedule),
		];
		const resize = new ResizeObserver(schedule);
		if (canvasRef.current) resize.observe(canvasRef.current);
		schedule();
		return () => {
			offs.forEach((off) => off());
			resize.disconnect();
			if (frame) cancelAnimationFrame(frame);
		};
	}, [editor]);

	// Restyling (or turning captions off) redraws at once, even paused.
	useEffect(() => { redraw.current(); }, [captions]);

	return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 block size-full" aria-hidden="true" />;
}

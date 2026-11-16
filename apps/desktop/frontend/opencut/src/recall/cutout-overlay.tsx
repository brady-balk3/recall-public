// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. The VTuber cutout on the 9:16 preview, drawn the way the
 * render draws it (engines/export/composition.py, the VTuber filter): the
 * avatar's box from the live frame, keyed through the saved alpha mask, faded
 * out over its bottom `lowerFade`, laid over the game at `target`.
 *
 * Frames come from the preview renderer (source-frames.ts), so the avatar
 * moves with the clip under it, mouth and eyes included.
 */
import { useEffect, useRef } from "react";
import type { RecallCutoutHost } from "./host";
import { latestSourceFrames, pickSource, subscribeSourceFrames, type SourceFrame } from "./source-frames";

export function CutoutOverlay({ cutout, canvasWidth }: {
	cutout: RecallCutoutHost;
	/** The clip canvas's width in px (target is in these units). */
	canvasWidth: number;
}) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const mask = useRef<ImageBitmap | null>(null);
	const scratch = useRef<HTMLCanvasElement | null>(null);
	const latest = useRef(cutout);
	latest.current = cutout;
	const draw = useRef<(frames: SourceFrame[]) => void>(() => {});

	useEffect(() => {
		let alive = true;
		mask.current = null;
		cutout.loadMask().then((bitmap) => {
			if (!alive) return;
			mask.current = bitmap;
			draw.current(latestSourceFrames()); // don't wait for the next frame (paused)
		}, () => {});
		return () => { alive = false; };
		// One load per mask; the rest of `cutout` is read through `latest`.
	}, [cutout.maskKey]);

	useEffect(() => {
		draw.current = (frames) => {
		const canvas = canvasRef.current;
		const alpha = mask.current;
		const pick = pickSource(frames);
		if (!canvas || !alpha || !pick) return;
		const { source, target, lowerFade } = latest.current;
		const rect = canvas.getBoundingClientRect();
		const dpr = window.devicePixelRatio || 1;
		const w = Math.max(1, Math.round(rect.width * dpr));
		const h = Math.max(1, Math.round(rect.height * dpr));
		if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
		const ctx = canvas.getContext("2d");
		if (!ctx) return;
		ctx.clearRect(0, 0, w, h);

		// The avatar on its own: box pixels, keyed by the mask, faded at the bottom.
		const scale = w / Math.max(1, canvasWidth);
		const aw = Math.max(1, Math.round(target[2] * scale));
		const ah = Math.max(1, Math.round(target[3] * scale));
		const sheet = scratch.current ?? (scratch.current = document.createElement("canvas"));
		if (sheet.width !== aw || sheet.height !== ah) { sheet.width = aw; sheet.height = ah; }
		const sc = sheet.getContext("2d");
		if (!sc) return;
		const frame = pick.frame.canvas;
		sc.globalCompositeOperation = "source-over";
		sc.clearRect(0, 0, aw, ah);
		sc.drawImage(frame, source[0] * frame.width, source[1] * frame.height, source[2] * frame.width, source[3] * frame.height, 0, 0, aw, ah);
		sc.globalCompositeOperation = "destination-in";
		sc.drawImage(alpha, 0, 0, aw, ah);
		if (lowerFade > 0) {
			const fade = sc.createLinearGradient(0, 0, 0, ah);
			fade.addColorStop(0, "#000");
			fade.addColorStop(Math.max(0, 1 - lowerFade), "#000");
			fade.addColorStop(1, "rgba(0,0,0,0)");
			sc.fillStyle = fade;
			sc.fillRect(0, 0, aw, ah);
		}
		sc.globalCompositeOperation = "source-over";
		ctx.drawImage(sheet, Math.round(target[0] * scale), Math.round(target[1] * scale));
		};
		draw.current(latestSourceFrames());
		return subscribeSourceFrames((frames) => draw.current(frames));
	}, [canvasWidth]);

	return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 block size-full" aria-hidden="true" />;
}

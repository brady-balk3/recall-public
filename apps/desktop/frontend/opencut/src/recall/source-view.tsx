// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. The Cutting Room's full-frame pane: the whole 16:9 frame
 * under the playhead, beside the 9:16 clip preview, with everything the clip
 * leaves out dimmed. Frames come from the preview's own renderer
 * (source-frames.ts), so both panes always show the same moment.
 *
 * - Drag inside the game window to slide it left or right.
 * - Drag the facecam box to move it, its corner to resize it, or drag anywhere
 *   else to draw a new one (Auto found none, or found the wrong thing).
 * - Gameplay layout has no facecam: the whole frame slides the window.
 */
import { useEffect, useRef, useState } from "react";
import { useContainerSize } from "@opencut/hooks/use-container-size";
import { PlayPauseButton, TimecodeDisplay } from "@opencut/preview/components/toolbar";
import { useRecallHost } from "./host";
import type { Rect } from "./framing";
import { pickSource, subscribeSourceFrames, type SourceRegion } from "./source-frames";

type Box = [number, number, number, number];

const MIN_BOX = 0.04;
const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

/** A box from two corners, inside the frame and never smaller than MIN_BOX. */
export function boxFromCorners(ax: number, ay: number, bx: number, by: number): Box {
	const x1 = clamp01(Math.min(ax, bx));
	const y1 = clamp01(Math.min(ay, by));
	const w = Math.max(MIN_BOX, clamp01(Math.max(ax, bx)) - x1);
	const h = Math.max(MIN_BOX, clamp01(Math.max(ay, by)) - y1);
	return [Math.min(x1, 1 - w), Math.min(y1, 1 - h), w, h];
}

/** The box moved by (dx, dy), kept inside the frame. */
export function shiftBox([x, y, w, h]: Box, dx: number, dy: number): Box {
	return [Math.min(1 - w, Math.max(0, x + dx)), Math.min(1 - h, Math.max(0, y + dy)), w, h];
}

/** focus_x for a clip window of width `w` whose left edge is at `left`. */
export function focusForLeft(left: number, w: number): number {
	const travel = 1 - w;
	return travel <= 1e-6 ? 0.5 : clamp01(left / travel);
}

const sameRegions = (a: SourceRegion[], b: SourceRegion[]) =>
	a.length === b.length && a.every((region, i) => region.role === b[i].role && region.rect.every((v, k) => Math.abs(v - b[i].rect[k]) < 1e-4));

type Drag =
	| { kind: "clip"; startX: number; startLeft: number; w: number; live: boolean }
	| { kind: "draw"; ax: number; ay: number }
	| { kind: "move"; sx: number; sy: number; start: Box }
	| { kind: "resize"; start: Box };

export function SourcePanel() {
	const host = useRecallHost().sourceView;
	const areaRef = useRef<HTMLDivElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const surfaceRef = useRef<HTMLDivElement>(null);
	const area = useContainerSize({ containerRef: areaRef });
	const [aspect, setAspect] = useState(16 / 9);
	const [regions, setRegions] = useState<SourceRegion[]>([]);
	const [hasFrame, setHasFrame] = useState(false);
	const [draftBox, setDraftBox] = useState<Box | null>(null);
	const [draftLeft, setDraftLeft] = useState<number | null>(null);
	const drag = useRef<Drag | null>(null);

	// The frame, fitted into the pane.
	const fitW = Math.max(0, Math.min(area.width, area.height * aspect));
	const fitH = aspect > 0 ? fitW / aspect : 0;

	useEffect(() => subscribeSourceFrames((frames) => {
		const pick = pickSource(frames);
		const canvas = canvasRef.current;
		if (!pick || !canvas) return;
		const { width, height } = pick.frame.canvas;
		if (!width || !height) return;
		// Back the canvas at its on-screen size (never above the source's).
		const rect = canvas.getBoundingClientRect();
		const scale = Math.min(window.devicePixelRatio || 1, width / Math.max(1, rect.width));
		const w = Math.max(1, Math.round(rect.width * scale));
		const h = Math.max(1, Math.round(rect.height * scale));
		if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
		canvas.getContext("2d")?.drawImage(pick.frame.canvas, 0, 0, w, h);
		setHasFrame(true);
		setAspect((prev) => (Math.abs(prev - width / height) > 1e-3 ? width / height : prev));
		setRegions((prev) => (sameRegions(prev, pick.regions) ? prev : pick.regions));
	}), []);

	if (!host) return null;

	const clip = regions.find((region) => region.role === "clip");
	const clipRect: Rect | undefined = clip && draftLeft != null ? [draftLeft, clip.rect[1], clip.rect[2], clip.rect[3]] : clip?.rect;
	const shown = regions.map((region) => (region === clip && clipRect ? { ...region, rect: clipRect } : region));
	const sliding = Boolean(host.gameplayOnly && host.onClipFocus);
	const camBox = draftBox ?? host.camBox;
	const framed = shown.some((region) => region.rect[2] < 0.999 || region.rect[3] < 0.999);

	const point = (event: React.PointerEvent) => {
		const r = surfaceRef.current!.getBoundingClientRect();
		return [clamp01((event.clientX - r.left) / r.width), clamp01((event.clientY - r.top) / r.height)] as const;
	};

	const onDown = (event: React.PointerEvent) => {
		if (event.button !== 0 || !surfaceRef.current) return;
		event.preventDefault();
		surfaceRef.current.setPointerCapture(event.pointerId);
		const [px, py] = point(event);
		if (sliding) {
			if (!clipRect) return;
			const [left, , w] = clipRect;
			const inside = px >= left && px <= left + w;
			const startLeft = inside ? left : Math.min(1 - w, Math.max(0, px - w / 2));
			drag.current = { kind: "clip", startX: px, startLeft, w, live: true };
			if (!inside) { setDraftLeft(startLeft); host.onClipFocus?.(focusForLeft(startLeft, w)); }
			return;
		}
		const handle = (event.target as HTMLElement).dataset.handle;
		const inClip = clipRect && px >= clipRect[0] && px <= clipRect[0] + clipRect[2] && py >= clipRect[1] && py <= clipRect[1] + clipRect[3];
		if (camBox && handle === "resize") drag.current = { kind: "resize", start: camBox };
		else if (camBox && handle === "box") drag.current = { kind: "move", sx: px, sy: py, start: camBox };
		else if (inClip && clipRect && clipRect[2] < 0.999 && host.onClipFocus) {
			drag.current = { kind: "clip", startX: px, startLeft: clipRect[0], w: clipRect[2], live: false };
		} else drag.current = { kind: "draw", ax: px, ay: py };
	};

	const onMove = (event: React.PointerEvent) => {
		const d = drag.current;
		if (!d) return;
		const [px, py] = point(event);
		if (d.kind === "clip") {
			const left = Math.min(1 - d.w, Math.max(0, d.startLeft + (px - d.startX)));
			setDraftLeft(left);
			if (d.live) host.onClipFocus?.(focusForLeft(left, d.w));
		} else if (d.kind === "draw") setDraftBox(boxFromCorners(d.ax, d.ay, px, py));
		else if (d.kind === "move") setDraftBox(shiftBox(d.start, px - d.sx, py - d.sy));
		else setDraftBox(boxFromCorners(d.start[0], d.start[1], px, py));
	};

	const onUp = () => {
		const d = drag.current;
		drag.current = null;
		if (d?.kind === "clip") {
			if (!d.live && draftLeft != null) host.onClipFocus?.(focusForLeft(draftLeft, d.w));
			setDraftLeft(null);
		} else if (d && draftBox) host.onCamBox(draftBox);
		setDraftBox(null);
	};

	const onKeyDown = (event: React.KeyboardEvent) => {
		if (!sliding || !clipRect) return;
		const step = event.shiftKey ? 0.1 : 0.02;
		const delta = event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0;
		if (!delta) return;
		event.preventDefault();
		host.onClipFocus?.(clamp01(focusForLeft(clipRect[0], clipRect[2]) + delta));
	};

	const pct = (value: number) => `${value * 100}%`;
	const hole = ([x, y, w, h]: Rect) => `M${x} ${y}h${w}v${h}h${-w}Z`;

	return (
		<div className="panel bg-background flex size-full min-h-0 min-w-0 flex-col rounded-sm border">
			<div ref={areaRef} className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden m-2 mb-0">
				<div
					ref={surfaceRef}
					className="relative"
					style={{ width: fitW, height: fitH, cursor: sliding ? "ew-resize" : "crosshair", touchAction: "none" }}
					role={sliding ? "slider" : "application"}
					aria-label={sliding
						? "Where the vertical clip is cut from. Drag left or right, or use the arrow keys."
						: "The whole frame. Drag the facecam box to move it, its corner to resize it, or drag elsewhere to draw a new one."}
					aria-valuemin={sliding ? 0 : undefined}
					aria-valuemax={sliding ? 100 : undefined}
					aria-valuenow={sliding && clipRect ? Math.round(focusForLeft(clipRect[0], clipRect[2]) * 100) : undefined}
					tabIndex={0}
					onPointerDown={onDown}
					onPointerMove={onMove}
					onPointerUp={onUp}
					onPointerCancel={onUp}
					onKeyDown={onKeyDown}
				>
					<canvas ref={canvasRef} className="absolute inset-0 block size-full bg-black" />
					{hasFrame && framed && (
						<svg className="pointer-events-none absolute inset-0 size-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
							<path d={`M0 0H1V1H0Z ${shown.map((region) => hole(region.rect)).join(" ")}`} fill="rgba(0,0,0,0.68)" fillRule="evenodd" />
							{shown.map((region, i) => (
								<rect
									key={i}
									x={region.rect[0]} y={region.rect[1]} width={region.rect[2]} height={region.rect[3]}
									fill="none"
									stroke={region.role === "clip" ? "#ffffff" : "#7dd3fc"}
									strokeWidth={2}
									vectorEffect="non-scaling-stroke"
								/>
							))}
						</svg>
					)}
					{hasFrame && shown.map((region, i) => (
						<span
							key={i}
							className="pointer-events-none absolute rounded-sm px-1.5 py-0.5 text-[11px] font-medium"
							style={{ left: pct(region.rect[0]), top: pct(region.rect[1]), margin: 4, background: "rgba(0,0,0,0.65)", color: region.role === "clip" ? "#fff" : "#7dd3fc" }}
						>
							{region.role === "clip" ? (shown.length > 1 ? "Game" : "Clip") : "Cam"}
						</span>
					))}
					{hasFrame && !sliding && camBox && (
						<div
							data-handle="box"
							className="absolute"
							style={{
								left: pct(camBox[0]), top: pct(camBox[1]), width: pct(camBox[2]), height: pct(camBox[3]),
								border: "1.5px dashed #7dd3fc", cursor: "move",
							}}
						>
							<span className="pointer-events-none absolute rounded-sm px-1.5 py-0.5 text-[11px] font-medium" style={{ right: 0, bottom: "100%", marginBottom: 3, background: "#7dd3fc", color: "#04131c" }}>
								{host.camLabel ?? "Facecam"}
							</span>
							<span
								data-handle="resize"
								className="absolute"
								style={{ right: -6, bottom: -6, width: 12, height: 12, borderRadius: 3, background: "#7dd3fc", cursor: "nwse-resize" }}
							/>
						</div>
					)}
				</div>
			</div>
			<div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3 px-4 pb-3 pt-3">
				<TimecodeDisplay />
				<PlayPauseButton />
				<span className="text-muted-foreground block w-full min-w-0 truncate text-right text-xs">
					{host.hint ?? (sliding
						? "Drag to slide the clip window"
						: camBox
							? "Drag the window to slide it, or the facecam box to move it"
							: "Drag the window to slide it, or draw a box over the facecam")}
				</span>
			</div>
		</div>
	);
}

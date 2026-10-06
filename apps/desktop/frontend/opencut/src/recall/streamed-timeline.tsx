// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Timeline content for a streamed video (a full VOD): real frames along the
 * element instead of one repeated poster, and the engine's waveform along its
 * bottom edge, which is how a creator spots the loud moments in hours of
 * footage. Only the thumbnail pages near the visible part of the element are
 * requested; everything else stays the poster until scrolled into view.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { VideoElement } from "@opencut/timeline";
import type { MediaAsset } from "@opencut/media/types";
import { buildWaveformSourceKey } from "@opencut/media/waveform-summary";
import { AudioWaveform } from "@opencut/timeline/components/audio-waveform";
import { findScrollParent } from "@opencut/utils/browser";
import { TICKS_PER_SECOND } from "@opencut/wasm";
import { LADDER_RATIO, pageImageSize, tileSeconds, visiblePages } from "./streamed-filmstrip";
import { streamedMedia, streamFilmstripUrl } from "./streamed-media";
import { useRecallHost, type RecallMomentState } from "./host";

const THUMBNAIL_ASPECT_RATIO = 16 / 9;
const WAVEFORM_HEIGHT = "42%";
const WAVEFORM_COLOR = "rgba(255, 255, 255, 0.6)";
const CURVE_HEIGHT = "58%";
const MOMENT_COLORS: Record<RecallMomentState, string> = {
	kept: "var(--keep, #3ecf8e)",
	maybe: "var(--maybe, #f5b544)",
	passed: "rgba(255, 255, 255, 0.35)",
	waiting: "var(--heat, #ff6a3d)",
	yours: "#ffffff",
};

/**
 * The reaction curve over source seconds [from, to], as an SVG area path in a
 * `width` x 100 box: the loudest value per pixel column, scaled to the curve's
 * overall peak so the same moment is the same height at every zoom.
 */
export function reactionPath({ t, r, peak, from, to, width }: {
	t: number[];
	r: number[];
	peak: number;
	from: number;
	to: number;
	width: number;
}): string {
	if (!(to > from) || width < 2 || !(peak > 0) || !t.length) return "";
	const columns = Math.max(2, Math.min(4096, Math.round(width / 2)));
	const tops = new Float32Array(columns);
	let seen = false;
	// Binary-search the first sample in view; t is sorted.
	let lo = 0, hi = t.length;
	while (lo < hi) { const mid = (lo + hi) >> 1; if (t[mid] < from) lo = mid + 1; else hi = mid; }
	for (let i = Math.max(0, lo - 1); i < t.length && t[i] <= to; i++) {
		const column = Math.min(columns - 1, Math.max(0, Math.floor(((t[i] - from) / (to - from)) * columns)));
		const value = Math.max(0, r[i] ?? 0) / peak;
		if (value > tops[column]) tops[column] = value;
		seen = true;
	}
	if (!seen) return "";
	let d = `M0 100`;
	for (let c = 0; c < columns; c++) {
		const x = ((c + 0.5) / columns) * width;
		d += `L${x.toFixed(1)} ${(100 - Math.min(1, tops[c]) * 100).toFixed(1)}`;
	}
	return `${d}L${width.toFixed(1)} 100Z`;
}

/** The element's visible horizontal span, in px from its left edge. */
function useVisibleSpan(container: React.RefObject<HTMLDivElement | null>) {
	const [span, setSpan] = useState<{ left: number; right: number } | null>(null);
	const frame = useRef(0);

	const measure = useCallback(() => {
		cancelAnimationFrame(frame.current);
		frame.current = requestAnimationFrame(() => {
			const node = container.current;
			if (!node) return;
			const rect = node.getBoundingClientRect();
			const parent = findScrollParent({ element: node });
			const bounds = parent ? parent.getBoundingClientRect() : { left: 0, right: window.innerWidth };
			const left = Math.max(0, bounds.left - rect.left);
			const right = Math.min(rect.width, bounds.right - rect.left);
			setSpan((previous) =>
				previous && Math.abs(previous.left - left) < 1 && Math.abs(previous.right - right) < 1
					? previous
					: { left, right },
			);
		});
	}, [container]);

	useLayoutEffect(() => {
		const node = container.current;
		if (!node) return;
		measure();
		const parent = findScrollParent({ element: node });
		parent?.addEventListener("scroll", measure, { passive: true });
		window.addEventListener("resize", measure);
		const observer = new ResizeObserver(measure);
		observer.observe(node);
		if (parent) observer.observe(parent);
		return () => {
			cancelAnimationFrame(frame.current);
			parent?.removeEventListener("scroll", measure);
			window.removeEventListener("resize", measure);
			observer.disconnect();
		};
	}, [container, measure]);

	return { span, measure };
}

export function StreamedVideoContent({
	element,
	mediaAsset,
	trackHeight,
	pixelsPerSecond,
}: {
	element: VideoElement;
	mediaAsset: MediaAsset;
	trackHeight: number;
	pixelsPerSecond: number;
}) {
	const container = useRef<HTMLDivElement>(null);
	const { span, measure } = useVisibleSpan(container);
	const streamed = streamedMedia(mediaAsset.file);
	// Recall's moments and reaction curve for this stream, when it supplies them.
	const momentsHost = useRecallHost().moments;
	const recall = momentsHost && streamed && momentsHost.streamRef === streamed.ref ? momentsHost : undefined;
	const curvePeak = useMemo(() => (recall?.curve ? Math.max(0, ...recall.curve.r) : 0), [recall?.curve]);
	const trimStart = element.trimStart / TICKS_PER_SECOND;
	const durationSec = element.duration / TICKS_PER_SECOND;
	const sourceDuration = mediaAsset.duration ?? (element.sourceDuration ?? 0) / TICKS_PER_SECOND;

	// Zoom changes the element's width without scrolling; re-measure.
	useEffect(() => { measure(); }, [measure, pixelsPerSecond, element.duration, element.trimStart]);

	const tileWidthPx = trackHeight * THUMBNAIL_ASPECT_RATIO;
	// Cells are requested a ladder step wide so they are cropped, never stretched.
	const cellWidthPx = Math.ceil(tileWidthPx * LADDER_RATIO);
	const tileSec = tileSeconds({ pixelsPerSecond, tileWidthPx });
	// Retimed elements don't map source time to position linearly: keep the poster.
	const pages = streamed && span && !element.retime
		? visiblePages({
				tileSec,
				sourceDuration,
				fromSec: trimStart + span.left / pixelsPerSecond,
				toSec: Math.min(trimStart + durationSec, trimStart + span.right / pixelsPerSecond),
			})
		: [];
	const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;

	return (
		<div ref={container} data-streamed-video="" className="absolute inset-0 overflow-hidden" style={{ pointerEvents: "none" }}>
			<div
				className="absolute inset-0"
				style={{
					backgroundColor: "var(--muted)",
					backgroundImage: mediaAsset.thumbnailUrl ? `url(${mediaAsset.thumbnailUrl})` : undefined,
					backgroundRepeat: "repeat-x",
					backgroundSize: `${tileWidthPx}px ${trackHeight}px`,
					backgroundPosition: "left center",
				}}
			/>
			{streamed && pages.map((page) => {
				const size = pageImageSize({ frames: page.frames, cellWidthPx, heightPx: trackHeight, devicePixelRatio: dpr });
				const url = streamFilmstripUrl(streamed.ref, { start: page.start, end: page.end, frames: page.frames, ...size });
				if (!url) return null;
				const cellPx = page.cellSeconds * pixelsPerSecond;
				// Scale up only when a (last-page) cell is wider than the image cell.
				const scale = Math.max(1, cellPx / cellWidthPx);
				const imageCell = cellWidthPx * scale;
				return (
					<div
						key={`${tileSec}:${page.index}`}
						data-filmstrip-page={url}
						className="absolute top-0 h-full"
						style={{ left: (page.start - trimStart) * pixelsPerSecond, width: (page.end - page.start) * pixelsPerSecond }}
					>
						{Array.from({ length: page.frames }, (_, i) => (
							<div
								key={i}
								className="absolute top-0 h-full overflow-hidden"
								style={{
									left: i * cellPx,
									width: Math.ceil(cellPx),
									backgroundImage: `url(${url})`,
									backgroundRepeat: "no-repeat",
									backgroundSize: `${imageCell * page.frames}px ${trackHeight * scale}px`,
									backgroundPosition: `${-(i * imageCell) + (cellPx - imageCell) / 2}px center`,
								}}
							/>
						))}
					</div>
				);
			})}
			{recall && span && !element.retime && (() => {
				const from = trimStart + span.left / pixelsPerSecond;
				const to = Math.min(trimStart + durationSec, trimStart + span.right / pixelsPerSecond);
				const width = Math.max(0, span.right - span.left);
				const d = recall.curve ? reactionPath({ ...recall.curve, peak: curvePeak, from, to, width }) : "";
				return (
					<>
						{d && (
							<svg
								data-reaction-curve=""
								className="absolute top-0"
								style={{ left: span.left, width, height: CURVE_HEIGHT }}
								viewBox={`0 0 ${width} 100`}
								preserveAspectRatio="none"
								aria-hidden="true"
							>
								<path d={d} fill="rgba(255, 106, 61, 0.22)" stroke="rgba(255, 140, 90, 0.9)" strokeWidth={1.25} vectorEffect="non-scaling-stroke" />
							</svg>
						)}
						{recall.moments
							.filter((moment) => moment.end >= from && moment.start <= to)
							.map((moment) => (
								<div
									key={moment.id}
									data-moment-bar={moment.state}
									className="absolute top-0"
									title={moment.title}
									style={{
										left: (moment.start - trimStart) * pixelsPerSecond,
										width: Math.max(3, (moment.end - moment.start) * pixelsPerSecond),
										height: 4,
										borderRadius: 2,
										background: MOMENT_COLORS[moment.state],
										boxShadow: "0 0 0 1px rgba(0,0,0,0.35)",
									}}
								/>
							))}
					</>
				);
			})()}
			{mediaAsset.hasAudio !== false && (
				<div className="absolute inset-x-0 bottom-0" style={{ height: WAVEFORM_HEIGHT }}>
					<AudioWaveform
						sourceKey={buildWaveformSourceKey({ kind: "media", id: element.mediaId })}
						sourceFile={mediaAsset.file}
						pixelsPerSecond={pixelsPerSecond}
						clipDurationSec={durationSec}
						retime={element.retime}
						sourceStartSec={trimStart}
						color={WAVEFORM_COLOR}
					/>
				</div>
			)}
		</div>
	);
}

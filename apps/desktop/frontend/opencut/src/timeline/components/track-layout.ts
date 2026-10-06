import type { TrackType } from "@opencut/timeline";
import {
	KEYFRAME_LANE_HEIGHT_PX,
	TIMELINE_TRACK_GAP_PX,
	TIMELINE_TRACK_HEIGHTS_PX,
} from "./layout";

// Recall: the host can make track types taller (the Cutting Room's stream
// carries frames, a reaction curve and a waveform). Set before the editor
// mounts; every height lookup goes through getTrackHeight.
let heightOverrides: Partial<Record<TrackType, number>> = {};

export function setTrackHeightOverrides(overrides: Partial<Record<TrackType, number>> | undefined): void {
	heightOverrides = overrides ?? {};
}

export function getTrackHeight({ type }: { type: TrackType }): number {
	return heightOverrides[type] ?? TIMELINE_TRACK_HEIGHTS_PX[type];
}

export function getExpandedTrackHeight({
	type,
	expandedLaneCount,
}: {
	type: TrackType;
	expandedLaneCount: number;
}): number {
	return (
		getTrackHeight({ type }) +
		expandedLaneCount * KEYFRAME_LANE_HEIGHT_PX
	);
}

export function getCumulativeHeightBefore({
	tracks,
	trackIndex,
	getExtraHeight,
}: {
	tracks: Array<{ type: TrackType }>;
	trackIndex: number;
	getExtraHeight?: (trackIndex: number) => number;
}): number {
	return tracks
		.slice(0, trackIndex)
		.reduce(
			(sum, track, i) =>
				sum +
				getTrackHeight({ type: track.type }) +
				(getExtraHeight?.(i) ?? 0) +
				TIMELINE_TRACK_GAP_PX,
			0,
		);
}

export function getTotalTracksHeight({
	tracks,
	getExtraHeight,
}: {
	tracks: Array<{ type: TrackType }>;
	getExtraHeight?: (trackIndex: number) => number;
}): number {
	const tracksHeight = tracks.reduce(
		(sum, track, i) =>
			sum + getTrackHeight({ type: track.type }) + (getExtraHeight?.(i) ?? 0),
		0,
	);
	const gapsHeight = Math.max(0, tracks.length - 1) * TIMELINE_TRACK_GAP_PX;
	return tracksHeight + gapsHeight;
}

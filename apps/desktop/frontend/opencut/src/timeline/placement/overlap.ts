import type { TimelineElement } from "@opencut/timeline";
import type { PlacementTimeSpan } from "./types";

interface TrackWithElements {
	id?: string;
	elements: TimelineElement[];
}

function wouldElementOverlap({
	elements,
	startTime,
	endTime,
	excludeElementId,
}: {
	elements: TimelineElement[];
	startTime: number;
	endTime: number;
	excludeElementId?: string;
}): boolean {
	return elements.some((element) => {
		if (excludeElementId && element.id === excludeElementId) {
			return false;
		}

		const elementEnd = element.startTime + element.duration;
		return startTime < elementEnd && endTime > element.startTime;
	});
}

export function canPlaceTimeSpansOnTrack({
	track,
	timeSpans,
}: {
	track: TrackWithElements;
	timeSpans: PlacementTimeSpan[];
}): boolean {
	// Recall: its framing and caption tracks are derived from the pieces and
	// rebuilt on every change, so nothing may be placed on one.
	if (track.id?.startsWith("recall-framing:") || track.id?.startsWith("recall-captions:")) return false;
	return timeSpans.every(({ startTime, duration, excludeElementId }) => {
		return !wouldElementOverlap({
			elements: track.elements,
			startTime,
			endTime: startTime + duration,
			excludeElementId,
		});
	});
}

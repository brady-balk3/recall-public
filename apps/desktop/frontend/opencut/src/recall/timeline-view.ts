// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Lets Recall point the timeline somewhere after it has mounted.
 *
 * The timeline takes its zoom from the project once, when it mounts, and a new
 * project mounts before its source is on the timeline. Left alone, a 4 h VOD
 * lands at the default zoom: five million px wide, which is unreadable and
 * costs gigabytes of renderer memory. A request here is applied by the
 * timeline (timeline/components/index.tsx) as soon as it can.
 */

export interface TimelineViewRequest {
	/** Seconds of timeline to fit across the view; omitted = everything. */
	span?: number;
	/** Timeline seconds to put the playhead on and centre in the view. */
	time?: number;
}

type Listener = (request: TimelineViewRequest) => boolean;

let pending: TimelineViewRequest | null = null;
const listeners = new Set<Listener>();

/** Ask the timeline to show a span / a moment. Kept until a timeline applies it. */
export function requestTimelineView(request: TimelineViewRequest): void {
	pending = request;
	for (const listener of listeners) {
		if (listener(request)) {
			pending = null;
			return;
		}
	}
}

/**
 * A mounted timeline subscribes; a listener returns true once it has applied
 * a request (false if it can't yet, e.g. the timeline is still empty).
 */
export function onTimelineViewRequest(listener: Listener): { retry: () => void; dispose: () => void } {
	listeners.add(listener);
	const retry = () => {
		if (pending && listener(pending)) pending = null;
	};
	retry();
	return { retry, dispose: () => listeners.delete(listener) };
}

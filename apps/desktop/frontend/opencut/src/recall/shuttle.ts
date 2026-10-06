// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * J/K/L shuttle for the Cutting Room, like the old VOD editor had: L plays
 * forward and each press doubles the speed (1x, 2x, 4x, 8x), J does the same
 * backwards, K stops. At 1x forward it is ordinary playback, with audio.
 * Anything else steps the playhead itself (OpenCut's playback has no speed),
 * a few times a second so a streamed four-hour VOD keeps up.
 */
import type { EditorCore } from "@opencut/core";
import { mediaTimeFromSeconds, mediaTimeToSeconds } from "@opencut/wasm";

const SPEEDS = [1, 2, 4, 8];
/** How often the playhead is moved while shuttling, like the old editor's scrub throttle. */
const STEP_MS = 90;

/** The next speed for a J (-1) or L (+1) press, from the current one (0 = stopped). */
export function nextSpeed(current: number, direction: 1 | -1): number {
	if (Math.sign(current) !== direction) return direction;
	const index = SPEEDS.indexOf(Math.abs(current));
	return direction * SPEEDS[Math.min(SPEEDS.length - 1, index + 1)];
}

export class Shuttle {
	private speed = 0;
	private timer: number | null = null;
	private last = 0;
	private listeners = new Set<(speed: number) => void>();

	constructor(private editor: EditorCore) {}

	getSpeed(): number {
		return this.speed;
	}

	onChange(listener: (speed: number) => void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	press(key: "j" | "k" | "l"): void {
		if (key === "k") return this.set(0);
		this.set(nextSpeed(this.speed, key === "l" ? 1 : -1));
	}

	stop(): void {
		this.set(0);
	}

	private set(speed: number): void {
		const playback = this.editor.playback;
		this.speed = speed;
		this.clearTimer();
		if (speed === 1) {
			playback.setScrubbing({ isScrubbing: false });
			if (!playback.getIsPlaying()) playback.play();
		} else {
			if (playback.getIsPlaying()) playback.pause();
			playback.setScrubbing({ isScrubbing: speed !== 0 });
			if (speed !== 0) {
				this.last = performance.now();
				this.timer = window.setInterval(() => this.step(), STEP_MS);
			}
		}
		for (const listener of this.listeners) listener(speed);
	}

	private step(): void {
		const playback = this.editor.playback;
		// The creator pressed play/space or the playhead was taken over: let go.
		if (playback.getIsPlaying()) {
			this.clearTimer();
			this.speed = 1;
			for (const listener of this.listeners) listener(1);
			return;
		}
		const now = performance.now();
		const dt = (now - this.last) / 1000;
		this.last = now;
		const end = mediaTimeToSeconds({ time: this.editor.timeline.getTotalDuration() });
		const current = mediaTimeToSeconds({ time: playback.getCurrentTime() });
		const next = Math.min(end, Math.max(0, current + this.speed * dt));
		playback.seek({ time: mediaTimeFromSeconds({ seconds: next }) });
		if (next <= 0 || next >= end) this.set(0);
	}

	private clearTimer(): void {
		if (this.timer !== null) window.clearInterval(this.timer);
		this.timer = null;
	}
}

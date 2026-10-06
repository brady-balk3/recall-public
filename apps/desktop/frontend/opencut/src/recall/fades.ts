// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. A piece's fades (picture from and to black, sound from and
 * to silence) are plain OpenCut keyframes on its opacity and volume, so the
 * preview plays them and the timeline shows them. Recall owns the keyframes by
 * id, reads the durations back when making the clip (the engine renders the
 * fades itself), and lays them out again when the piece is trimmed or split.
 */
import type { ElementAnimations, ScalarAnimationKey, ScalarChannel } from "@opencut/animation/types";
import type { VideoElement } from "@opencut/timeline";
import { VOLUME_DB_MIN } from "@opencut/timeline/audio-constants";
import { TICKS_PER_SECOND, mediaTimeFromSeconds } from "@opencut/wasm";

export interface PieceFades {
	pictureIn: number;
	pictureOut: number;
	soundIn: number;
	soundOut: number;
}

export const NO_FADES: PieceFades = { pictureIn: 0, pictureOut: 0, soundIn: 0, soundOut: 0 };

const KEY_PREFIX = "recall-fade-";
const PATHS = { picture: "opacity", sound: "volume" } as const;
type Kind = keyof typeof PATHS;
/** Silence, and the full level, per property. */
const FLOOR: Record<Kind, number> = { picture: 0, sound: VOLUME_DB_MIN };

const isOurs = (key: { id: string }) => key.id.startsWith(KEY_PREFIX);

function channel(element: VideoElement, kind: Kind): ScalarChannel | undefined {
	const data = element.animations?.[PATHS[kind]] as ScalarChannel | undefined;
	return data && Array.isArray(data.keys) ? data : undefined;
}

function keyTime(keys: ScalarAnimationKey[], id: string): number | undefined {
	const key = keys.find((candidate) => candidate.id === id);
	return key ? key.time / TICKS_PER_SECOND : undefined;
}

/** The fades Recall laid on this piece, in seconds (0 when there are none). */
export function readFades(element: VideoElement): PieceFades {
	const span = (kind: Kind, edge: "in" | "out") => {
		const keys = channel(element, kind)?.keys ?? [];
		const a = keyTime(keys, `${KEY_PREFIX}${kind}-${edge}-a`);
		const b = keyTime(keys, `${KEY_PREFIX}${kind}-${edge}-b`);
		return a != null && b != null ? Math.max(0, round(b - a)) : 0;
	};
	return {
		pictureIn: span("picture", "in"),
		pictureOut: span("picture", "out"),
		soundIn: span("sound", "in"),
		soundOut: span("sound", "out"),
	};
}

const round = (value: number) => Math.round(value * 100) / 100;

function key(id: string, seconds: number, value: number): ScalarAnimationKey {
	return { id, time: mediaTimeFromSeconds({ seconds }), value, segmentToNext: "linear", tangentMode: "auto" };
}

/**
 * The piece with exactly these fades: Recall's keyframes replaced, anything
 * else on opacity and volume left alone. Fades longer than the piece shrink
 * to fit it (in and out never overlap).
 */
export function withFades(element: VideoElement, fades: PieceFades, fullLevel: { picture: number; sound: number }): VideoElement {
	const length = element.duration / TICKS_PER_SECOND;
	const animations: ElementAnimations = { ...(element.animations ?? {}) };
	for (const kind of ["picture", "sound"] as const) {
		let fadeIn = Math.max(0, kind === "picture" ? fades.pictureIn : fades.soundIn);
		let fadeOut = Math.max(0, kind === "picture" ? fades.pictureOut : fades.soundOut);
		if (fadeIn + fadeOut > length) {
			const share = length / (fadeIn + fadeOut);
			fadeIn *= share;
			fadeOut *= share;
		}
		const theirs = (channel(element, kind)?.keys ?? []).filter((candidate) => !isOurs(candidate));
		const ours: ScalarAnimationKey[] = [];
		const full = fullLevel[kind];
		if (fadeIn > 0) {
			ours.push(key(`${KEY_PREFIX}${kind}-in-a`, 0, FLOOR[kind]), key(`${KEY_PREFIX}${kind}-in-b`, fadeIn, full));
		}
		if (fadeOut > 0) {
			ours.push(key(`${KEY_PREFIX}${kind}-out-a`, length - fadeOut, full), key(`${KEY_PREFIX}${kind}-out-b`, length, FLOOR[kind]));
		}
		const keys = [...theirs, ...ours].sort((a, b) => a.time - b.time);
		if (keys.length) {
			animations[PATHS[kind]] = { ...(channel(element, kind) ?? {}), keys } as ScalarChannel;
		} else {
			delete animations[PATHS[kind]];
		}
	}
	return { ...element, animations: Object.keys(animations).length ? animations : undefined };
}

/** The level a piece plays at between its fades. */
export function fullLevelOf(element: VideoElement): { picture: number; sound: number } {
	const opacity = Number(element.params?.opacity ?? 1);
	const volume = Number(element.params?.volume ?? 0);
	return { picture: Number.isFinite(opacity) ? opacity : 1, sound: Number.isFinite(volume) ? volume : 0 };
}

/**
 * Whether the piece's Recall keyframes sit where its fades say they should.
 * A trim or split moves the piece's ends; the fades follow them.
 */
export function fadesInPlace(element: VideoElement): boolean {
	const fades = readFades(element);
	const expected = withFades(element, fades, fullLevelOf(element));
	for (const kind of ["picture", "sound"] as const) {
		const now = (channel(element, kind)?.keys ?? []).filter(isOurs);
		const want = (channel(expected, kind)?.keys ?? []).filter(isOurs);
		if (now.length !== want.length) return false;
		for (const candidate of want) {
			const found = now.find((existing) => existing.id === candidate.id);
			if (!found || Math.abs(found.time - candidate.time) > 1 || Math.abs(found.value - candidate.value) > 1e-3) return false;
		}
	}
	return true;
}

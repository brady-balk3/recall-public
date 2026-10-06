// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The engine's waveform for a streamed source (core/waveform_peaks.py), as
 * OpenCut's own SourceWaveformSummary, so the timeline draws it unchanged.
 *
 *   "RWF2" | u32 sampleRate | u32 bucketSize | u32 totalSamples | f32 loudLevel
 *          | u8 peak per bucket, on a -80..0 dBFS scale (0 = silence)
 *
 * A VOD's waveform is drawn relative to the VOD's own loud level, not to
 * 0 dBFS: it is there to find where a stream gets loud, and many streams mix
 * around -45 dB, below OpenCut's -40 dB display floor, which would draw hours
 * of footage as a flat line.
 */
import type { SourceWaveformSummary } from "@opencut/media/waveform-summary";
import { streamedMedia, streamWaveformUrl } from "./streamed-media";

const HEADER_BYTES = 20;
const FLOOR_DB = 80;
// The VOD's loud level lands just under the top of the waveform.
const LOUD_LEVEL_DISPLAY = 10 ** (-3 / 20);

export function parseWaveformPeaks({
	sourceKey,
	bytes,
}: {
	sourceKey: string;
	bytes: ArrayBuffer;
}): SourceWaveformSummary {
	const view = new DataView(bytes);
	const magic = String.fromCharCode(...new Uint8Array(bytes, 0, Math.min(4, bytes.byteLength)));
	if (bytes.byteLength < HEADER_BYTES || magic !== "RWF2") {
		throw new Error("Recall's waveform file wasn't recognised.");
	}
	const loudLevel = view.getFloat32(16, true);
	const gain = loudLevel > 0 ? LOUD_LEVEL_DISPLAY / loudLevel : 1;
	const peaks = new Uint8Array(bytes, HEADER_BYTES);
	// One lookup per byte value instead of a pow() per bucket (millions for a VOD).
	const table = new Float32Array(256);
	for (let b = 1; b < 256; b++) {
		table[b] = Math.min(1, 10 ** ((b * FLOOR_DB / 255 - FLOOR_DB) / 20) * gain);
	}
	const amplitudes = new Float32Array(peaks.length);
	for (let i = 0; i < peaks.length; i++) amplitudes[i] = table[peaks[i]];
	return {
		sourceKey,
		sampleRate: view.getUint32(4, true),
		bucketSize: view.getUint32(8, true),
		totalSamples: view.getUint32(12, true),
		amplitudes,
	};
}

/** The engine's summary for a streamed file, or null when Recall has none. */
export async function fetchStreamedWaveform({
	sourceKey,
	file,
}: {
	sourceKey: string;
	file: File;
}): Promise<SourceWaveformSummary | null> {
	const streamed = streamedMedia(file);
	const url = streamed && streamWaveformUrl(streamed.ref);
	if (!url) return null;
	const response = await fetch(url);
	if (!response.ok) throw new Error(`Recall's waveform isn't available (${response.status}).`);
	return parseWaveformPeaks({ sourceKey, bytes: await response.arrayBuffer() });
}

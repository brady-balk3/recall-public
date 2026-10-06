// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Streamed media: a source video that stays where it is on disk and is read by
 * byte range through Recall's engine, never loaded into memory.
 *
 * OpenCut passes a `File` everywhere it handles media. A streamed asset is an
 * empty placeholder `File` registered here against its URL; every decode point
 * asks `sourceFor()` for its mediabunny source, and gets a range-reading
 * `UrlSource` for a streamed file or the usual `BlobSource` for everything
 * else. Paths that would read a whole file (the browser-storage copy, the
 * decoded waveform, the whole-clip audio buffer) check `isStreamed()` and
 * stay out of the way.
 */
import { BlobSource, CustomSource, type Source } from "mediabunny";

export interface StreamedMedia {
	/**
	 * Stable reference Recall understands (e.g. `job:<id>`). This is what a
	 * saved project keeps: engine URLs carry a per-launch port and token.
	 */
	ref: string;
	/** The URL for this launch, resolved from `ref`. */
	url: string;
	/** Real size of the file on disk (the placeholder `File` is 0 bytes). */
	size: number;
}

// Ranges are read on demand; keep what's cached small and bounded.
const CACHE_BYTES = 32 * 1024 * 1024;

const registry = new WeakMap<Blob, StreamedMedia>();

export interface FilmstripRequest {
	/** Source seconds the strip covers; frames are sampled at cell centres. */
	start: number;
	end: number;
	frames: number;
	/** Image size in device pixels. */
	width: number;
	height: number;
}

/**
 * How the editor reaches a streamed source through Recall, for this launch.
 * The engine does what the renderer can't afford for a long VOD: it computes
 * the waveform once and serves contact sheets of real frames.
 */
export interface StreamHost {
	source: (ref: string) => string;
	waveform?: (ref: string) => string;
	filmstrip?: (ref: string, request: FilmstripRequest) => string;
}

let host: StreamHost | undefined;

/** Recall tells the editor how to turn a saved `ref` into this launch's URLs. */
export function setStreamResolver(next: StreamHost | undefined): void {
	host = next;
}

/**
 * The host can adopt a video the creator imports (media bin, drag and drop,
 * paste): Recall registers the file on disk and the editor streams it from
 * there instead of copying it into browser storage. Resolves to the stream
 * ref, or null to import the ordinary way.
 */
export type ImportAdopter = (file: File) => Promise<{ ref: string; size: number } | null>;

let importAdopter: ImportAdopter | undefined;

export function setImportAdopter(next: ImportAdopter | undefined): void {
	importAdopter = next;
}

/** The file to import: a streamed placeholder when the host adopts it. */
export async function adoptImport(file: File): Promise<File> {
	if (!importAdopter || isStreamed(file) || !file.type.startsWith("video/")) return file;
	try {
		const adopted = await importAdopter(file);
		return adopted ? streamedFile({ ref: adopted.ref, name: file.name, type: file.type, size: adopted.size }) : file;
	} catch {
		return file;
	}
}

export function resolveStreamUrl(ref: string): string {
	if (!host) throw new Error("Recall hasn't told the editor where streamed media lives.");
	return host.source(ref);
}

export function streamWaveformUrl(ref: string): string | undefined {
	return host?.waveform?.(ref);
}

export function streamFilmstripUrl(ref: string, request: FilmstripRequest): string | undefined {
	return host?.filmstrip?.(ref, request);
}

export function streamedFile({
	ref,
	name,
	type = "video/mp4",
	size,
}: {
	ref: string;
	name: string;
	type?: string;
	size: number;
}): File {
	const file = new File([], name, { type });
	registry.set(file, { ref, url: resolveStreamUrl(ref), size });
	return file;
}

export function streamedMedia(blob: Blob | undefined | null): StreamedMedia | undefined {
	return blob ? registry.get(blob) : undefined;
}

export function isStreamed(blob: Blob | undefined | null): boolean {
	return !!streamedMedia(blob);
}

/** The file's real size, including a streamed file's size on disk. */
export function mediaSize(blob: Blob): number {
	return registry.get(blob)?.size ?? blob.size;
}

/**
 * The engine is on this machine, so a streamed source behaves like a local
 * file: every read asks for exactly the bytes mediabunny needs, with small
 * file-system-style prefetch. (mediabunny's UrlSource is tuned for the
 * internet: it opens multi-GB ranges and cancels them once it has enough,
 * which over localhost read 36 GB of a 10.9 GB VOD in one short session.)
 */
export function sourceFor(blob: Blob): Source {
	const streamed = registry.get(blob);
	if (!streamed) return new BlobSource(blob);
	const { url, size } = streamed;
	return new CustomSource({
		getSize: () => size,
		read: (start, end) => readRange(url, start, end),
		maxCacheSize: CACHE_BYTES,
		prefetchProfile: "fileSystem",
	});
}

const READ_ATTEMPTS = 3;

/**
 * Exactly bytes [start, end) of a streamed source. The engine is local, but a
 * busy engine can still drop a connection mid-scan; a dropped read retries
 * instead of failing playback or an export.
 */
async function readRange(url: string, start: number, end: number): Promise<Uint8Array> {
	let lastError: unknown;
	for (let attempt = 0; attempt < READ_ATTEMPTS; attempt++) {
		if (attempt) await new Promise((resolve) => setTimeout(resolve, 150 * attempt));
		try {
			// no-store: a restored VOD comes back at the same URL, and cached ranges
			// of the old file must never be mixed into it. The engine is local anyway.
			const response = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` }, cache: "no-store" });
			if (response.status === 206) return new Uint8Array(await response.arrayBuffer());
			lastError = new Error(`Streamed read failed (${response.status}) for bytes ${start}-${end - 1}`);
			// A 4xx (the file moved, the range is wrong) won't fix itself.
			if (response.status < 500) break;
		} catch (error) {
			lastError = error;
		}
	}
	throw lastError;
}

export interface SourceWindow {
	/** Seconds into the source to start decoding. */
	start: number;
	/** Seconds into the source to stop; undefined decodes to the end. */
	end?: number;
}

// A little extra either side so retime and resampling never run short.
const WINDOW_PAD_SECONDS = 0.5;

/**
 * The part of a streamed source a timeline element actually uses, in source
 * seconds. Decoding a whole multi-hour track to mix a 30s cut would hold
 * gigabytes of samples; a streamed file only ever decodes its window.
 * Returns undefined for ordinary files, which keep upstream's behaviour.
 */
export function sourceWindow({
	file,
	trimStart,
	trimEnd,
	sourceDuration,
}: {
	file: Blob;
	trimStart: number;
	trimEnd: number;
	sourceDuration?: number;
}): SourceWindow | undefined {
	if (!isStreamed(file)) return undefined;
	const end = sourceDuration != null && sourceDuration > 0
		? Math.max(trimStart, sourceDuration - trimEnd) + WINDOW_PAD_SECONDS
		: undefined;
	return { start: Math.max(0, trimStart - WINDOW_PAD_SECONDS), end };
}

/**
 * Byte size of a URL from a one-byte range request: the engine answers
 * `Content-Range: bytes 0-0/<total>`. (Its routes don't answer HEAD.)
 */
export async function remoteSize(url: string): Promise<number> {
	const response = await fetch(url, { headers: { Range: "bytes=0-0" }, cache: "no-store" });
	const total = Number(response.headers.get("content-range")?.split("/")[1]);
	void response.body?.cancel();
	if (response.status !== 206 || !Number.isFinite(total) || total <= 0) {
		throw new Error("Recall couldn't stream this video: its source doesn't support seeking.");
	}
	return total;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. What the host app hands the editor about the clip open in
 * it. OpenCut code reads this through context and never talks to Recall's API
 * itself; the wrapper (../../OpenCutEditor.tsx) fills it in.
 */
import { createContext, useContext, type ReactNode } from "react";

/** One timed word, in seconds from the first frame of Recall's render. */
export interface RecallCaptionWord {
	word: string;
	start: number;
	end: number;
}

export interface RecallCaptionStyle {
	fontFamily: string;
	color: string;
	size: "small" | "medium" | "large";
	position: "top" | "middle" | "bottom";
}

export interface RecallCaptionsHost {
	/** File name of the clip as it sits in the media bin (Recall's render). */
	sourceName: string;
	/**
	 * Whether Recall's current render of this clip has captions burned in,
	 * as recorded when it was rendered.
	 */
	loadBurnedIn: () => Promise<boolean>;
	style: RecallCaptionStyle;
	/** The words on screen: the creator's correction if they made one. */
	loadWords: () => Promise<RecallCaptionWord[]>;
	/** The same clip rendered without captions. */
	loadCleanVideo: () => Promise<{ url: string; name: string }>;
}

/** Where a moment stands in the creator's review. "yours" = cut by the creator. */
export type RecallMomentState = "kept" | "maybe" | "passed" | "waiting" | "yours";

/** A moment Recall found (or the creator cut), in seconds of the source VOD. */
export interface RecallMoment {
	id: string;
	title: string;
	start: number;
	end: number;
	/** The reaction peak inside it, if known. */
	peak?: number;
	/** Recall's 0-100 hype score, if it has one. */
	score?: number;
	state: RecallMomentState;
}

/** What Recall knows about the streamed VOD open in the Cutting Room. */
export interface RecallMomentsHost {
	/** The streamed source these times belong to (see streamed-media.ts). */
	streamRef: string;
	moments: RecallMoment[];
	/** Chat's reaction over the stream: times (s) and values (any scale). */
	curve?: { t: number[]; r: number[] };
	/** What the tab says when there are no moments (e.g. still scanning). */
	emptyText?: string;
}

/** How Recall frames the clips the Cutting Room makes (vertical layout + facecam). */
export interface RecallLayoutHost {
	options: { value: string; label: string; note: string }[];
	value: string;
	onChange: (value: string) => void;
	/** The creator drew their own facecam box (else Recall detects it). */
	hasFacecam: boolean;
	/** The facecam box is being drawn right now. */
	drawing: boolean;
	/** Open (or close) the facecam drawing on the source frame at `sourceTime`. */
	onToggleDraw: (sourceTime?: number) => void;
	onClearFacecam: () => void;
	/** A note under the layout choices (e.g. the video is still being scanned). */
	hint?: string;
	/** Burned-in captions on the clips made here. */
	captions: boolean;
	onToggleCaptions: () => void;
}

/**
 * The Cutting Room's live 9:16 framing (see framing.ts). `lookup` answers from
 * the host's cache; `version` changes whenever new layers arrive, so the
 * editor re-applies them.
 */
export interface RecallFramingHost {
	canvas: { width: number; height: number };
	lookup: import("./framing").FramingLookup;
	version: number;
}

/**
 * The Cutting Room's caption track (caption-track.ts): the words each piece's
 * clip will burn in, read from the engine the way a render reads them.
 */
export interface RecallCaptionTrackHost {
	/** Captions are burned into the clips made here. */
	enabled: boolean;
	style: RecallCaptionStyle;
	/**
	 * The words a clip of [start, end] of this stream would show, clip-relative.
	 * `listen` decodes the audio when the scan has no words there (uses the GPU).
	 */
	loadWords: (streamRef: string, start: number, end: number, listen: boolean) => Promise<{ source: "scan" | "decode" | "none"; words: RecallCaptionWord[] }>;
}

/**
 * A clip opened for editing (from Export or Review): its piece is selected, or
 * cut if it isn't on the timeline, carrying the clip's title, fades and
 * caption fixes. "Make clip" then updates that clip.
 */
export interface RecallOpenClip {
	/** New per open, so opening the same clip again finds it again. */
	key: string;
	clipId: string;
	streamRef: string;
	/** Source seconds. */
	start: number;
	end: number;
	title: string;
	fades: { pictureIn: number; pictureOut: number; soundIn: number; soundOut: number };
	/** The clip's corrected caption words, clip-relative, when the creator fixed them. */
	captionWords?: RecallCaptionWord[];
}

export interface RecallEditorHost {
	captions?: RecallCaptionsHost;
	captionTrack?: RecallCaptionTrackHost;
	openClip?: RecallOpenClip;
	moments?: RecallMomentsHost;
	layout?: RecallLayoutHost;
	framing?: RecallFramingHost;
	/**
	 * Drawn over the preview, sized to the frame (not the letterboxing), so
	 * normalized coordinates in it are coordinates on the canvas.
	 */
	previewOverlay?: ReactNode;
}

export const RecallHostContext = createContext<RecallEditorHost>({});

export function useRecallHost(): RecallEditorHost {
	return useContext(RecallHostContext);
}

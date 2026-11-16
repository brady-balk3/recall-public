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
	/** The word being spoken. Absent: the engine's default yellow. */
	highlightColor?: string;
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
	/** The clip has a facecam band at the top (a top caption then sits mid-frame, as rendered). */
	topFacecam?: boolean;
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

/**
 * The full-frame pane beside the 9:16 preview (source-view.tsx): the whole
 * 16:9 frame with the regions the clip is cut from. Boxes are normalized
 * [x, y, w, h] on the source frame.
 */
export interface RecallSourceViewHost {
	/** The facecam box the clip's layout uses (drawn, or found by Recall), if any. */
	camBox: [number, number, number, number] | null;
	/** Move, resize or draw the facecam box. */
	onCamBox: (box: [number, number, number, number]) => void;
	/** Slide the clip's game window sideways: focus_x, 0 = left edge, 1 = right edge. */
	onClipFocus?: (focusX: number) => void;
	/**
	 * Gameplay layout: there's no facecam, the whole frame slides the window,
	 * and it's framed without the engine, so it follows the drag live.
	 * Otherwise a slide is sent once, on release.
	 */
	gameplayOnly?: boolean;
	/** One line under the frame (what dragging does here). */
	hint?: string;
	/** What the box is called on the frame ("Facecam", or "Avatar" for a VTuber cutout). */
	camLabel?: string;
}

/**
 * A VTuber cutout on the 9:16 clip. The editor can't key an alpha mask, so the
 * preview draws the avatar itself (cutout-overlay.tsx) the way the render
 * does: the source box, through the mask, faded at the bottom, on `target`.
 */
export interface RecallCutoutHost {
	/** Normalized [x, y, w, h] on the source frame. */
	source: [number, number, number, number];
	/** Canvas px on the clip (1080x1920). */
	target: [number, number, number, number];
	/** Share of the avatar's height that fades out at the bottom. */
	lowerFade: number;
	/** Changes when the mask does. */
	maskKey: string;
	loadMask: () => Promise<ImageBitmap>;
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
	/** Show the full frame beside the 9:16 preview (the Cutting Room). */
	sourceView?: RecallSourceViewHost;
	/** The avatar cutout the 9:16 preview draws over the clip. */
	cutout?: RecallCutoutHost;
}

export const RecallHostContext = createContext<RecallEditorHost>({});

export function useRecallHost(): RecallEditorHost {
	return useContext(RecallHostContext);
}

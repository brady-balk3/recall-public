// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. Opening a clip in the Cutting Room: find its piece on the
 * timeline (the one that made it), or cut it from the stream at exactly its
 * range, then give it the clip's title, fades and caption fixes, select it and
 * put the playhead on it.
 */
import type { EditorCore } from "@opencut/core";
import { mediaTimeFromSeconds } from "@opencut/wasm";
import { captionParam, CAPTION_PARAM, type PieceCaptions } from "./caption-track";
import { CLIP_PARAM, cutPiece, streamElements } from "./cutting";
import { fullLevelOf, readFades, withFades } from "./fades";
import type { RecallOpenClip } from "./host";
import { requestTimelineView } from "./timeline-view";

const SAME = 0.05;

/** Place the clip; false while the stream isn't on the timeline yet. */
export function placeOpenClip(editor: EditorCore, clip: RecallOpenClip): boolean {
	// The piece that made it (even if trimmed since), else one at its range.
	const find = () => {
		const pieces = streamElements(editor, clip.streamRef).filter((piece) => !piece.reel);
		return pieces.find((piece) => piece.element.params?.[CLIP_PARAM] === clip.clipId)
			?? pieces.find((piece) => Math.abs(piece.start - clip.start) < SAME && Math.abs(piece.end - clip.end) < SAME);
	};
	let piece = find();
	if (!piece) {
		if (!streamElements(editor, clip.streamRef).length) return false;
		if (!cutPiece(editor, clip.streamRef, { start: clip.start, end: clip.end }, { pad: 0, name: clip.title })) return false;
		piece = find();
		if (!piece) return false;
		// A fresh piece takes what the clip already has.
		const hasFades = Object.values(clip.fades).some((value) => value > 0);
		const element = hasFades ? withFades(piece.element, clip.fades, fullLevelOf(piece.element)) : piece.element;
		const captions: PieceCaptions | undefined = clip.captionWords?.length ? {
			from: clip.start,
			to: clip.end,
			source: "scan",
			edited: true,
			words: clip.captionWords.map((word) => ({ word: word.word, start: word.start + clip.start, end: word.end + clip.start })),
		} : undefined;
		editor.timeline.updateElements({
			updates: [{
				trackId: piece.trackId,
				elementId: piece.element.id,
				patch: {
					animations: element.animations,
					params: {
						...piece.element.params,
						[CLIP_PARAM]: clip.clipId,
						...(captions ? { [CAPTION_PARAM]: captionParam(captions) } : {}),
					},
				},
			}],
			pushHistory: false,
		});
	} else if (!Object.values(readFades(piece.element)).some((value) => value > 0) && Object.values(clip.fades).some((value) => value > 0)) {
		// Its piece is here but was made before fades lived on the timeline.
		const element = withFades(piece.element, clip.fades, fullLevelOf(piece.element));
		editor.timeline.updateElements({ updates: [{ trackId: piece.trackId, elementId: piece.element.id, patch: { animations: element.animations } }], pushHistory: false });
	}
	if (piece.element.params?.[CLIP_PARAM] !== clip.clipId) {
		const fresh = find();
		if (fresh) markPieceClip(editor, fresh, clip.clipId);
	}
	editor.selection.setSelectedElements({ elements: [{ trackId: piece.trackId, elementId: piece.element.id }] });
	editor.playback.seek({ time: mediaTimeFromSeconds({ seconds: piece.at }) });
	requestTimelineView({ time: piece.at + (piece.end - piece.start) / 2, span: Math.max(30, (piece.end - piece.start) * 1.6) });
	return true;
}

/** Remember which clip a piece made, so a trim later re-cuts that clip. */
export function markPieceClip(editor: EditorCore, piece: { trackId: string; element: { id: string; params: Record<string, unknown> } }, clipId: string): void {
	editor.timeline.updateElements({
		updates: [{ trackId: piece.trackId, elementId: piece.element.id, patch: { params: { ...(piece.element.params as Record<string, string | number | boolean>), [CLIP_PARAM]: clipId } } }],
		pushHistory: false,
	});
}

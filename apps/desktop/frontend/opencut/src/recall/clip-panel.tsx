// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. The Cutting Room's right-hand panel: the clip a piece of
 * the stream becomes. It takes over OpenCut's properties for stream pieces
 * (their position and crop belong to Recall's framing) and for an empty
 * selection, where it follows the piece under the playhead.
 *
 * - Title: the piece's name on the timeline; it names the clip.
 * - Layout and facecam: how every clip from this video is framed.
 * - Fades: keyframes on the piece (fades.ts), so the preview plays them.
 * - Captions: burned into every clip made here.
 */
import { useEffect, useState } from "react";
import type { EditorCore } from "@opencut/core";
import { useEditor } from "@opencut/editor/use-editor";
import { useElementSelection } from "@opencut/timeline/hooks/element/use-element-selection";
import type { VideoElement } from "@opencut/timeline";
import { TICKS_PER_SECOND, mediaTimeFromSeconds } from "@opencut/wasm";
import { customTitle, streamElements, streamRefsOnTimeline, type StreamElement } from "./cutting";
import { fadesInPlace, fullLevelOf, readFades, withFades, type PieceFades } from "./fades";
import { CAPTION_ELEMENT_SUFFIX, coversWindow, pieceChunks, readCaptions, replaceSpan, saveCaptions, storeCaptionWords } from "./caption-track";
import { useRecallHost, type RecallLayoutHost } from "./host";
import { clock } from "./moments-view";

const FADE_STEPS = [0, 0.5, 1, 2];
const LAYER_SUFFIX = "~recall-layer";

type Piece = StreamElement & { ref: string };

function allPieces(editor: EditorCore): Piece[] {
	return streamRefsOnTimeline(editor).flatMap((ref) => streamElements(editor, ref).map((item) => ({ ...item, ref })));
}

/**
 * The piece the panel is about: the selected one (a facecam copy stands for
 * its piece), else the piece under the playhead. `null` when the selection is
 * something else, so OpenCut's own properties show.
 */
export function clipPanelTarget(editor: EditorCore, selected: { elementId: string }[]): Piece | undefined | null {
	const pieces = allPieces(editor);
	if (selected.length > 1) return null;
	if (selected.length === 1) {
		// A facecam copy or a caption stands for its piece.
		const id = selected[0].elementId.split(LAYER_SUFFIX)[0].split(CAPTION_ELEMENT_SUFFIX)[0];
		return pieces.find((piece) => piece.element.id === id) ?? null;
	}
	const now = editor.playback.getCurrentTime() / TICKS_PER_SECOND;
	return pieces.find((piece) => !piece.reel && now >= piece.at && now < piece.at + (piece.end - piece.start));
}

export function RecallClipPanel() {
	const editor = useEditor();
	const recall = useRecallHost();
	useEditor((e) => e.scenes.getActiveSceneOrNull());
	// Follow the playhead piece by piece, not frame by frame.
	useEditor((e) => {
		const now = e.playback.getCurrentTime() / TICKS_PER_SECOND;
		return allPieces(e).find((piece) => !piece.reel && now >= piece.at && now < piece.at + (piece.end - piece.start))?.element.id ?? "";
	});
	const { selectedElements } = useElementSelection();
	const piece = clipPanelTarget(editor, selectedElements) ?? undefined;
	const clipPiece = piece && !piece.reel ? piece : undefined;

	return (
		<div className="panel bg-background flex h-full flex-col overflow-hidden rounded-sm border" data-recall-clip-panel="">
			<div className="border-b px-3 py-2.5">
				<p className="text-sm font-medium">{clipPiece ? "This clip" : "Your clips"}</p>
				<p className="text-muted-foreground text-xs tabular-nums">
					{clipPiece
						? `${clock(clipPiece.end - clipPiece.start)} long · from ${clock(clipPiece.start)} in the video`
						: piece?.reel
							? "This is the whole video. Split or trim it to make a clip."
							: recall.layout
								? "Select a piece on the timeline to name it and fade it."
								: "Open a video or load a session to start cutting."}
				</p>
			</div>
			<div className="flex-1 overflow-y-auto">
				{!recall.layout && (
					<ol className="text-muted-foreground flex flex-col gap-2 px-3 py-3 text-sm" data-recall-clip-intro="">
						<li><span className="text-foreground">1.</span> Open a video, load a session, or drop a recording on the timeline.</li>
						<li><span className="text-foreground">2.</span> Trim or split it, or cut a moment from the Moments tab.</li>
						<li><span className="text-foreground">3.</span> Title it, frame it, add fades and fix captions right here, then press Make clip.</li>
					</ol>
				)}
				{clipPiece && <TitleRow piece={clipPiece} />}
				{recall.layout && <LayoutRows layout={recall.layout} piece={piece} />}
				{clipPiece && <FadeRows piece={clipPiece} />}
				{recall.layout && <CaptionRows layout={recall.layout} piece={clipPiece} selected={selectedElements[0]?.elementId} />}
			</div>
		</div>
	);
}

function Section({ label, children, note }: { label: string; children: React.ReactNode; note?: string }) {
	return (
		<section className="border-b px-3 py-3">
			<p className="text-muted-foreground mb-2 text-xs">{label}</p>
			{children}
			{note && <p className="text-muted-foreground mt-2 text-xs">{note}</p>}
		</section>
	);
}

function TitleRow({ piece }: { piece: Piece }) {
	const editor = useEditor();
	const saved = customTitle(piece.element, piece.asset) ?? "";
	const [draft, setDraft] = useState(saved);
	useEffect(() => setDraft(saved), [saved, piece.element.id]);
	const commit = () => {
		const title = draft.trim();
		if (title === saved) return;
		editor.timeline.updateElements({
			updates: [{ trackId: piece.trackId, elementId: piece.element.id, patch: { name: title || piece.asset.name } }],
		});
	};
	return (
		<Section label="Title">
			<input
				className="bg-background focus-visible:ring-ring w-full rounded-md border px-2.5 py-1.5 text-sm outline-none focus-visible:ring-1"
				value={draft}
				placeholder={`Clip at ${clock(piece.start)}`}
				maxLength={200}
				aria-label="Clip title"
				onChange={(event) => setDraft(event.target.value)}
				onBlur={commit}
				onKeyDown={(event) => {
					if (event.key === "Enter") (event.target as HTMLInputElement).blur();
					if (event.key === "Escape") { setDraft(saved); (event.target as HTMLInputElement).blur(); }
					event.stopPropagation();
				}}
			/>
		</Section>
	);
}

function LayoutRows({ layout, piece }: { layout: RecallLayoutHost; piece?: Piece }) {
	const editor = useEditor();
	// Draw on the frame under the playhead, so it's the one you're looking at.
	const drawAt = () => {
		if (!piece) return undefined;
		const now = editor.playback.getCurrentTime() / TICKS_PER_SECOND;
		return Math.min(piece.end, Math.max(piece.start, piece.start + (now - piece.at)));
	};
	return (
		<Section label="Layout" note={layout.drawing ? undefined : layout.hint}>
			<div className="grid grid-cols-2 gap-1" role="radiogroup" aria-label="Clip layout" data-recall-layout="">
				{layout.options.map((option) => (
					<button
						key={option.value}
						type="button"
						role="radio"
						aria-checked={layout.value === option.value}
						onClick={() => layout.onChange(option.value)}
						title={option.note}
						className={`rounded-md border px-2 py-1.5 text-left text-xs ${layout.value === option.value ? "bg-accent border-foreground/40" : "hover:bg-accent/60"}`}
					>
						<span className="block text-sm">{option.label}</span>
						<span className="text-muted-foreground block truncate">{option.note}</span>
					</button>
				))}
			</div>
			<div className="mt-2 flex items-center gap-1.5">
				<button type="button" className="hover:bg-accent/60 rounded-md border px-2.5 py-1 text-xs" onClick={() => layout.onToggleDraw(drawAt())}>
					{layout.hasFacecam ? "Adjust facecam" : "Draw facecam"}
				</button>
				{layout.hasFacecam && (
					<button type="button" className="text-muted-foreground hover:text-foreground px-2 py-1 text-xs" onClick={layout.onClearFacecam}>Use Recall's</button>
				)}
			</div>
		</Section>
	);
}

function FadeRows({ piece }: { piece: Piece }) {
	const editor = useEditor();
	const fades = readFades(piece.element);
	const set = (patch: Partial<PieceFades>) => {
		const next = withFades(piece.element, { ...fades, ...patch }, fullLevelOf(piece.element));
		editor.timeline.updateElements({
			updates: [{ trackId: piece.trackId, elementId: piece.element.id, patch: { animations: next.animations } }],
		});
	};
	const row = (label: string, key: keyof PieceFades) => (
		<div className="flex items-center justify-between gap-2">
			<span className="text-sm">{label}</span>
			<div className="bg-accent/40 flex rounded-md p-0.5" role="radiogroup" aria-label={label}>
				{FADE_STEPS.map((step) => {
					const on = Math.abs(fades[key] - step) < 0.01;
					return (
						<button
							key={step}
							type="button"
							role="radio"
							aria-checked={on}
							onClick={() => set({ [key]: step })}
							className={`rounded px-2 py-0.5 text-xs tabular-nums ${on ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
						>
							{step === 0 ? "Off" : `${step}s`}
						</button>
					);
				})}
			</div>
		</div>
	);
	return (
		<Section label="Fades" note="Picture fades from and to black. They play in the preview.">
			<div className="flex flex-col gap-2">
				{row("Picture in", "pictureIn")}
				{row("Picture out", "pictureOut")}
				{row("Sound in", "soundIn")}
				{row("Sound out", "soundOut")}
			</div>
		</Section>
	);
}

/** Which caption cue a selected caption element is (its index), if one is. */
const selectedCue = (id: string | undefined) => {
	const at = id?.lastIndexOf(CAPTION_ELEMENT_SUFFIX) ?? -1;
	return id && at >= 0 ? Number(id.slice(at + CAPTION_ELEMENT_SUFFIX.length)) : undefined;
};

function CaptionRows({ layout, piece, selected }: { layout: RecallLayoutHost; piece?: Piece; selected?: string }) {
	const editor = useEditor();
	const { captionTrack } = useRecallHost();
	const [listening, setListening] = useState(false);
	const [failure, setFailure] = useState<string | null>(null);
	const toggle = (
		<label className="flex cursor-pointer items-center justify-between gap-3 text-sm">
			<span>
				Burn in captions
				<span className="text-muted-foreground block text-xs">Every clip made here, word by word.</span>
			</span>
			<input type="checkbox" role="switch" checked={layout.captions} onChange={layout.onToggleCaptions} aria-label="Captions" />
		</label>
	);
	if (!layout.captions || !piece || !captionTrack) return <Section label="Captions">{toggle}</Section>;

	const window = { start: piece.start, end: piece.end };
	const captions = readCaptions(piece.element);
	const read = coversWindow(captions, window) ? captions : undefined;
	const chunks = read ? pieceChunks(read, window) : [];
	const cue = selectedCue(selected);
	const load = async (listen: boolean, fresh: boolean) => {
		setListening(listen);
		setFailure(null);
		try {
			const { source, words } = await captionTrack.loadWords(piece.ref, piece.start, piece.end, listen);
			storeCaptionWords(editor, piece, source, words, { fresh });
		} catch (error) {
			setFailure(error instanceof Error ? error.message : "Recall couldn't read the words.");
		} finally {
			setListening(false);
		}
	};
	const seek = (from: number) => editor.playback.seek({ time: mediaTimeFromSeconds({ seconds: piece.at + Math.max(0, from - piece.start) }) });

	let body: React.ReactNode;
	if (piece.end - piece.start > 180) {
		body = <p className="text-muted-foreground text-xs">Trim this piece to 3 minutes or less to see its words.</p>;
	} else if (!read) {
		body = <p className="text-muted-foreground text-xs">Reading the words…</p>;
	} else if (!chunks.length) {
		body = (
			<div className="flex flex-col items-start gap-2">
				<p className="text-muted-foreground text-xs">
					{read.source === "none" ? "Your scan didn't catch words here, or this video hasn't been scanned." : "Recall didn't hear any words in this piece."}
				</p>
				{read.source === "none" && (
					<button type="button" className="hover:bg-accent/60 rounded-md border px-2.5 py-1 text-xs" disabled={listening} onClick={() => void load(true, false)}>
						{listening ? "Listening…" : "Listen for words"}
					</button>
				)}
				{read.source === "none" && <p className="text-muted-foreground text-xs">Uses the GPU for a few seconds.</p>}
			</div>
		);
	} else {
		body = (
			<>
				<ol className="flex flex-col gap-1" data-recall-caption-lines="">
					{chunks.map((chunk, index) => (
						<CaptionLine
							key={`${piece.element.id}:${index}:${chunk.text}`}
							at={clock(chunk.from - piece.start)}
							text={chunk.text}
							active={cue === index}
							onSeek={() => seek(chunk.from)}
							onCommit={(text) => saveCaptions(editor, piece, replaceSpan(read, chunk.from, chunk.to, text))}
						/>
					))}
				</ol>
				<div className="mt-2 flex items-center justify-between gap-2">
					<p className="text-muted-foreground text-xs">
						{read.edited ? "Your words. Recall burns these in." : read.source === "scan" ? "From your scan. Fix any word and it sticks." : "Recall listened to this piece."}
					</p>
					{read.edited && (
						<button type="button" className="text-muted-foreground hover:text-foreground shrink-0 text-xs" onClick={() => void load(read.source === "decode", true)}>Undo my changes</button>
					)}
				</div>
			</>
		);
	}
	return (
		<Section label="Captions">
			{toggle}
			<div className="mt-3">{body}</div>
			{failure && <p className="mt-2 text-xs text-red-400">{failure}</p>}
		</Section>
	);
}

function CaptionLine({ at, text, active, onSeek, onCommit }: { at: string; text: string; active: boolean; onSeek: () => void; onCommit: (text: string) => void }) {
	const [draft, setDraft] = useState(text);
	useEffect(() => setDraft(text), [text]);
	const commit = () => {
		if (draft.trim() && draft.trim() !== text) onCommit(draft.trim());
		else setDraft(text);
	};
	return (
		<li className={`flex items-center gap-2 rounded-md px-1 ${active ? "bg-accent" : ""}`}>
			<button type="button" className="text-muted-foreground hover:text-foreground w-10 shrink-0 text-left text-xs tabular-nums" onClick={onSeek} title="Jump here">
				{at}
			</button>
			<input
				className="focus-visible:bg-background min-w-0 flex-1 rounded border border-transparent bg-transparent px-1.5 py-1 text-sm outline-none focus-visible:border-current/20"
				value={draft}
				aria-label={`Caption at ${at}`}
				onFocus={onSeek}
				onChange={(event) => setDraft(event.target.value)}
				onBlur={commit}
				onKeyDown={(event) => {
					if (event.key === "Enter") (event.target as HTMLInputElement).blur();
					if (event.key === "Escape") { setDraft(text); (event.target as HTMLInputElement).blur(); }
					event.stopPropagation();
				}}
			/>
		</li>
	);
}

/**
 * Keeps each piece's fades on its ends: a trim or split moves the ends, and
 * the fade keyframes follow. Not an edit of its own, so it skips undo.
 */
export function syncFades(editor: EditorCore): void {
	const updates = allPieces(editor)
		.filter((piece) => !fadesInPlace(piece.element))
		.map((piece) => {
			const element: VideoElement = piece.element;
			const next = withFades(element, readFades(element), fullLevelOf(element));
			return { trackId: piece.trackId, elementId: element.id, patch: { animations: next.animations } };
		});
	if (updates.length) editor.timeline.updateElements({ updates, pushHistory: false });
}

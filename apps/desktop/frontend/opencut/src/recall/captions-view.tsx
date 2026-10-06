// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition: the Captions tab. Upstream transcribes in the browser with
 * a downloaded Whisper model; Recall already has the words (and the
 * creator's corrections), so this puts those on the timeline as text the
 * creator can restyle and retime.
 *
 * Recall burns captions into its renders. Adding them again as text would
 * stack two copies, so when they're burned in, the clip is first swapped
 * for the same render without captions (trims and moves are kept). SRT/ASS
 * import is upstream's and stays.
 */
import { useEffect, useRef, useState } from "react";
import { CloudUploadIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@opencut/components/ui/button";
import { Spinner } from "@opencut/components/ui/spinner";
import { PanelView } from "@opencut/components/editor/panels/assets/views/base-panel";
import { Section, SectionContent } from "@opencut/components/section";
import { useEditor } from "@opencut/editor/use-editor";
import type { EditorCore } from "@opencut/core";
import { processMediaAssets } from "@opencut/media/processing";
import { AddMediaAssetCommand } from "@opencut/commands/media";
import { UpdateElementsCommand } from "@opencut/commands/timeline";
import { BatchCommand } from "@opencut/commands";
import { insertCaptionChunksAsTextTrack } from "@opencut/subtitles/insert";
import { parseSubtitleFile } from "@opencut/subtitles/parse";
import type { SubtitleCue, SubtitleStyleOverrides } from "@opencut/subtitles/types";
import { mediaTimeToSeconds } from "@opencut/wasm";
import type { VideoElement } from "@opencut/timeline/types";
import { chunkCaptionWords } from "./caption-chunks";
import { useRecallHost, type RecallCaptionStyle, type RecallCaptionsHost } from "./host";

// Fractions of canvas height, matched by eye to Recall's burned-in sizes on a 1080x1920 clip.
const SIZE_RATIO: Record<RecallCaptionStyle["size"], number> = { small: 0.045, medium: 0.056, large: 0.07 };
const MARGIN_RATIO: Record<RecallCaptionStyle["position"], number> = { top: 0.12, middle: 0, bottom: 0.2 };

type Status = { phase: "idle" } | { phase: "busy"; step: string } | { phase: "done"; message: string } | { phase: "error"; message: string };

/** Recall's caption look, as OpenCut subtitle overrides. */
export function captionStyleOverrides(style: RecallCaptionStyle): SubtitleStyleOverrides {
	return {
		fontFamily: style.fontFamily,
		color: style.color,
		fontWeight: "bold",
		textAlign: "center",
		fontSizeRatioOfPlayHeight: SIZE_RATIO[style.size],
		// Burned-in captions carry an outline; this snapshot's text has no
		// stroke, so a soft backing keeps them readable over bright footage.
		background: { enabled: true, color: "rgba(0, 0, 0, 0.55)", cornerRadius: 12 },
		placement: { verticalAlign: style.position, marginVerticalRatio: MARGIN_RATIO[style.position] },
	};
}

/** Video elements on the timeline that play the given media asset(s). */
function elementsPlaying(editor: EditorCore, mediaIds: Set<string>) {
	const tracks = editor.scenes.getActiveScene().tracks;
	const found: Array<{ trackId: string; element: VideoElement }> = [];
	for (const track of [tracks.main, ...tracks.overlay]) {
		for (const element of track.elements) {
			if (element.type === "video" && mediaIds.has(element.mediaId)) found.push({ trackId: track.id, element });
		}
	}
	return found;
}

async function swapInCleanVideo(editor: EditorCore, host: RecallCaptionsHost, sourceIds: Set<string>) {
	const { url } = await host.loadCleanVideo();
	const name = cleanNameFor(host.sourceName);
	const response = await fetch(url);
	if (!response.ok) throw new Error(`Couldn't load the caption-free clip (${response.status}).`);
	const blob = await response.blob();
	const [asset] = await processMediaAssets({ files: [new File([blob], name, { type: blob.type || "video/mp4" })] });
	if (!asset) throw new Error("The caption-free clip couldn't be read.");
	const projectId = editor.project.getActive().metadata.id;
	const addMedia = new AddMediaAssetCommand({ projectId, asset });
	const updates = elementsPlaying(editor, sourceIds).map(({ trackId, element }) => ({
		trackId,
		elementId: element.id,
		patch: { mediaId: addMedia.getAssetId() },
	}));
	editor.command.execute({ command: new BatchCommand([addMedia, new UpdateElementsCommand({ updates })]) });
}

export function RecallCaptionsView() {
	const editor = useEditor();
	const { captions: host } = useRecallHost();
	const [status, setStatus] = useState<Status>({ phase: "idle" });
	const fileInputRef = useRef<HTMLInputElement>(null);
	const busy = status.phase === "busy";
	const assets = useEditor((e) => e.media.getAssets());
	// Asked, not assumed: the render records whether its captions are burned in.
	const [burnedIn, setBurnedIn] = useState<boolean | null>(null);
	useEffect(() => {
		if (!host) return;
		let live = true;
		host.loadBurnedIn().then(
			(value) => { if (live) setBurnedIn(value); },
			() => { if (live) setBurnedIn(true); },
		);
		return () => { live = false; };
	}, [host]);
	const needsCleanCopy = !!burnedIn && !!host && !assets.some((asset) => asset.name === cleanNameFor(host.sourceName));

	const insert = (cues: SubtitleCue[]) => insertCaptionChunksAsTextTrack({ editor, captions: cues }) !== null;

	const addRecallCaptions = async () => {
		if (!host || busy) return;
		try {
			const cleanName = cleanNameFor(host.sourceName);
			if (needsCleanCopy) {
				const sourceIds = new Set(editor.media.getAssets().filter((asset) => asset.name === host.sourceName).map((asset) => asset.id));
				if (sourceIds.size === 0) throw new Error("Recall's clip isn't in this project any more.");
				setStatus({ phase: "busy", step: "Rendering the clip without captions…" });
				await swapInCleanVideo(editor, host, sourceIds);
			}
			setStatus({ phase: "busy", step: "Reading the words…" });
			const words = await host.loadWords();
			// Words are timed from the clip's first frame; follow the clip if it was moved or trimmed.
			const playing = elementsPlaying(editor, new Set(editor.media.getAssets()
				.filter((asset) => asset.name === host.sourceName || asset.name === cleanName)
				.map((asset) => asset.id)))[0]?.element;
			const offset = playing
				? mediaTimeToSeconds({ time: playing.startTime }) - mediaTimeToSeconds({ time: playing.trimStart })
				: 0;
			const style = captionStyleOverrides(host.style);
			const cues = chunkCaptionWords({ words, offset })
				.filter((cue) => cue.startTime + cue.duration > 0)
				.map((cue) => ({ ...cue, style }));
			if (!cues.length || !insert(cues)) {
				setStatus({ phase: "error", message: "Recall didn't hear any words in this clip." });
				return;
			}
			setStatus({ phase: "done", message: `Added ${cues.length} captions on their own track. Restyle one, or select them all to restyle together.` });
		} catch (error) {
			console.error("Recall captions failed", error);
			setStatus({ phase: "error", message: error instanceof Error ? error.message : "Something went wrong adding the captions." });
		}
	};

	const importFile = async (file: File) => {
		setStatus({ phase: "busy", step: "Reading the subtitle file…" });
		try {
			const result = parseSubtitleFile({ fileName: file.name, input: await file.text() });
			if (!result.captions.length || !insert(result.captions)) {
				setStatus({ phase: "error", message: "No subtitles were found in that file." });
				return;
			}
			setStatus({ phase: "done", message: `Imported ${result.captions.length} subtitles.` });
		} catch (error) {
			setStatus({ phase: "error", message: error instanceof Error ? error.message : "That subtitle file couldn't be read." });
		}
	};

	return (
		<PanelView
			title="Captions"
			contentClassName="px-0 flex flex-col h-full"
			actions={
				<Button type="button" variant="outline" size="sm" onClick={() => fileInputRef.current?.click()} disabled={busy} className="items-center justify-center gap-1.5">
					<HugeiconsIcon icon={CloudUploadIcon} />
					Import
				</Button>
			}
		>
			<input
				ref={fileInputRef}
				type="file"
				accept=".srt,.ass"
				className="hidden"
				onChange={(event) => {
					const file = event.target.files?.[0];
					event.target.value = "";
					if (file) void importFile(file);
				}}
			/>
			<Section showTopBorder={false} showBottomBorder={false} className="flex-1">
				<SectionContent className="flex h-full flex-col gap-3 pt-1">
					{host ? (
						<p className="text-muted-foreground text-sm leading-relaxed">
							{needsCleanCopy
								? "Recall burned captions into this clip. Make them editable to restyle or retime them here: Recall swaps in the same clip without captions and adds the words as text."
								: "Add the words Recall heard in this clip as text you can restyle and retime."}
						</p>
					) : (
						<p className="text-muted-foreground text-sm leading-relaxed">Import an .srt or .ass file to add captions.</p>
					)}
					{status.phase === "done" && <p className="text-sm">{status.message}</p>}
					{status.phase === "error" && (
						<div className="bg-destructive/10 border-destructive/20 rounded-md border p-3">
							<p className="text-destructive text-sm">{status.message}</p>
						</div>
					)}
					{host && (
						<Button type="button" className="mt-auto w-full" onClick={() => void addRecallCaptions()} disabled={busy || burnedIn === null}>
							{busy && <Spinner className="mr-1" />}
							{busy ? status.step : needsCleanCopy ? "Make captions editable" : "Add Recall's captions"}
						</Button>
					)}
				</SectionContent>
			</Section>
		</PanelView>
	);
}

/** Media-bin name for the caption-free copy of a clip. */
export function cleanNameFor(sourceName: string): string {
	return sourceName.replace(/(\.mp4)?$/i, " (no captions).mp4");
}

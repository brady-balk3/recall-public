// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall's Moments tab (Cutting Room only): every moment Recall found in the
 * stream, and the ones the creator cut. Click one to jump there; "Cut" puts
 * its piece of the stream on the timeline, lined up above the reel, ready for
 * "Make clips".
 */
import { useMemo, useState } from "react";
import { Button } from "@opencut/components/ui/button";
import { PanelView } from "@opencut/components/editor/panels/assets/views/base-panel";
import { useEditor } from "@opencut/editor/use-editor";
import { cutPiece, isCut, timelineTimeForSource } from "./cutting";
import { useRecallHost, type RecallMoment, type RecallMomentState } from "./host";
import { requestTimelineView } from "./timeline-view";

/** Seconds of stream shown around a moment you jump to. */
const JUMP_SPAN_SECONDS = 90;

const STATE: Record<RecallMomentState, { label: string; color: string }> = {
	kept: { label: "Kept", color: "var(--keep, #3ecf8e)" },
	maybe: { label: "Maybe", color: "var(--maybe, #f5b544)" },
	passed: { label: "Passed", color: "var(--t3, #8a8a92)" },
	waiting: { label: "To review", color: "var(--heat, #ff6a3d)" },
	yours: { label: "Your cut", color: "var(--t1, #f4f4f5)" },
};

export function clock(seconds: number): string {
	const s = Math.max(0, Math.round(seconds));
	const h = Math.floor(s / 3600);
	const m = Math.floor((s % 3600) / 60);
	const r = String(s % 60).padStart(2, "0");
	return h ? `${h}:${String(m).padStart(2, "0")}:${r}` : `${m}:${r}`;
}

export function RecallMomentsView() {
	const recall = useRecallHost();
	const host = recall.moments;
	const editor = useEditor();
	// Re-render when the timeline changes, so "On timeline" stays true.
	useEditor((e) => e.scenes.getActiveSceneOrNull());
	const [order, setOrder] = useState<"time" | "loudest">("time");
	const [notice, setNotice] = useState<string | null>(null);

	const moments = useMemo(() => {
		const list = [...(host?.moments ?? [])];
		return order === "time"
			? list.sort((a, b) => a.start - b.start)
			: list.sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
	}, [host?.moments, order]);

	if (!host) {
		return (
			<PanelView title="Moments">
				<p className="text-muted-foreground p-2 text-sm">Moments show up when a stream is open in the Cutting Room.</p>
			</PanelView>
		);
	}

	const jump = (moment: RecallMoment) => {
		const at = timelineTimeForSource(editor, host.streamRef, moment.peak ?? moment.start);
		if (at == null) {
			setNotice("That part of the stream isn't on your timeline. Cut it to bring it in.");
			return;
		}
		setNotice(null);
		requestTimelineView({ time: at, span: JUMP_SPAN_SECONDS });
	};

	const cut = (moment: RecallMoment) => {
		if (!cutPiece(editor, host.streamRef, moment)) {
			setNotice("The stream isn't in this project anymore.");
			return;
		}
		setNotice(null);
		const at = timelineTimeForSource(editor, host.streamRef, moment.peak ?? moment.start);
		if (at != null) requestTimelineView({ time: at, span: JUMP_SPAN_SECONDS });
	};

	return (
		<PanelView
			title={`Moments · ${moments.length}`}
			contentClassName="px-1"
			actions={
				<Button type="button" variant="outline" size="sm" onClick={() => setOrder(order === "time" ? "loudest" : "time")}>
					{order === "time" ? "By time" : "Loudest first"}
				</Button>
			}
		>
			{notice && <p className="text-muted-foreground px-2 pb-2 pt-2 text-xs">{notice}</p>}
			{moments.length === 0 ? (
				<p className="text-muted-foreground p-2 text-sm">{host.emptyText ?? "Recall didn't find moments in this stream. Scrub the reel and cut your own."}</p>
			) : (
				<ul className="flex flex-col gap-1 pt-2 pb-3" data-recall-moments="">
					{moments.map((moment) => {
						const onTimeline = isCut(editor, host.streamRef, moment.start, moment.end);
						const state = STATE[moment.state];
						return (
							<li key={moment.id} className="hover:bg-accent/60 flex items-center gap-2 rounded-md px-2 py-1.5">
								<button
									type="button"
									className="flex min-w-0 flex-1 cursor-pointer items-start gap-2 text-left"
									onClick={() => jump(moment)}
									title={`Jump to ${clock(moment.start)}`}
								>
									<span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: state.color }} aria-label={state.label} />
									<span className="min-w-0 flex-1">
										<span className="line-clamp-2 text-sm leading-snug">{moment.title}</span>
										<span className="text-muted-foreground text-xs tabular-nums">
											{clock(moment.start)} · {Math.round(moment.end - moment.start)}s · {state.label}
											{moment.score != null ? ` · ${moment.score}` : ""}
										</span>
									</span>
								</button>
								{onTimeline ? (
									<span className="text-muted-foreground shrink-0 text-xs">On timeline</span>
								) : (
									<Button type="button" variant="outline" size="sm" className="shrink-0" onClick={() => cut(moment)}>
										Cut
									</Button>
								)}
							</li>
						);
					})}
				</ul>
			)}
		</PanelView>
	);
}

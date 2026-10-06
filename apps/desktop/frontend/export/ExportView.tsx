// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Export: your keepers, vertical and captioned, grouped by the stream they
 * came from. The "ready to post" card carries everything not exported yet;
 * one click exports it all, or pick clips from the groups below.
 */
import { useMemo, useRef, useState } from "react";
import { Poster, SessionPoster } from "../components/Posters";
import { fmtClock } from "../lib/format";
import { Check, ExportIcon, PlayCircle } from "../lib/icons";
import { durationOf } from "../lib/reviewDeck";
import type { Clip, ExportLayout, Session } from "../lib/store";
import { hypeOf } from "../home/homeModel";
import { KeepersReel } from "../ui/KeepersReel";
import { sessionDate } from "../ui/SessionCard";
import { ExportTabs } from "./ExportTabs";
import type { View } from "../shell/views";

export type ExportFilter = "all" | "todo" | "done";

export interface ExportItem { clip: Clip; session: Session }

export interface ExportViewProps {
  items: ExportItem[];
  selectedIds: Set<string>;
  exporting: boolean;
  layout: ExportLayout;
  captionsOn: boolean;
  onToggleSelect: (clipId: string) => void;
  onToggleSession: (sessionId: string) => void;
  onClearSelection: () => void;
  onExportSelected: () => void;
  onExportIds: (clipIds: string[]) => void;
  onEdit: (session: Session, clip: Clip) => void;
  onView: (view: View) => void;
  onChangeFormat: () => void;
  compilationsCount?: number;
}

const LAYOUT_LABEL: Record<ExportLayout, string> = { auto: "automatic framing", vertical_split: "stacked facecam", gameplay_pip: "gameplay with facecam inset" };

const isExported = ({ clip, session }: ExportItem) => !!clip.exported || session.exportedClipIds.includes(clip.id);

export function groupByStream(items: ExportItem[]) {
  const order: string[] = [];
  const groups = new Map<string, { session: Session; items: ExportItem[] }>();
  for (const item of items) {
    let group = groups.get(item.session.id);
    if (!group) {
      group = { session: item.session, items: [] };
      groups.set(item.session.id, group);
      order.push(item.session.id);
    }
    group.items.push(item);
  }
  return order.map((id) => groups.get(id)!);
}

export function ExportView(props: ExportViewProps) {
  const { items, selectedIds, exporting, layout, captionsOn, onToggleSelect, onToggleSession, onClearSelection, onExportSelected, onExportIds, onEdit, onView, onChangeFormat, compilationsCount } = props;
  const [filter, setFilter] = useState<ExportFilter>("all");
  const groupsRef = useRef<HTMLDivElement>(null);

  const todo = useMemo(() => items.filter((item) => !isExported(item)).sort((a, b) => hypeOf(b.clip) - hypeOf(a.clip)), [items]);
  const done = items.length - todo.length;
  const visible = filter === "all" ? items : filter === "todo" ? items.filter((i) => !isExported(i)) : items.filter(isExported);
  const groups = groupByStream(visible);
  const selectedCount = items.reduce((n, { clip }) => n + (selectedIds.has(clip.id) ? 1 : 0), 0);
  const todoSeconds = todo.reduce((sum, { clip }) => sum + durationOf(clip), 0);
  const format = `1080×1920 · ${captionsOn ? "captions burned in" : "no captions"} · ${LAYOUT_LABEL[layout]}`;

  return (
    <div className="page export">
      <ExportTabs active="clips" clipsCount={items.length} compilationsCount={compilationsCount} onView={onView} />

      {!items.length ? (
        <section className="ready glass is-empty">
          <div className="rd-copy">
            <h2 className="disp">Nothing to post yet.</h2>
            <p className="t2">Keep a few moments in Review and they land here, cut vertical and ready to go.</p>
            <div className="w-cta"><button type="button" className="btn heat lg" onClick={() => onView("review")}><PlayCircle aria-hidden="true" />Open Review</button></div>
          </div>
        </section>
      ) : todo.length ? (
        <section className="ready glass" aria-labelledby="ready-title">
          <div className="rd-copy">
            <h2 className="disp" id="ready-title">{todo.length} {todo.length === 1 ? "clip" : "clips"} ready to post</h2>
            <p className="t2"><span className="num">{fmtClock(todoSeconds)}</span> of keepers · {format}</p>
            <div className="w-cta">
              <button type="button" className="btn heat lg" disabled={exporting} onClick={() => onExportIds(todo.map(({ clip }) => clip.id))}>
                <ExportIcon aria-hidden="true" />{exporting ? "Preparing files…" : `Export all ${todo.length}`}
              </button>
              <button type="button" className="btn lg" onClick={() => { setFilter("todo"); groupsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }); }}>Pick some</button>
            </div>
          </div>
          <KeepersReel
            drift
            label="Clips ready to post"
            items={todo.slice(0, 24).map(({ clip, session }) => ({
              key: clip.id,
              label: `Edit ${clip.hookLine || clip.title}`,
              onOpen: () => onEdit(session, clip),
              content: (
                <>
                  <Poster clip={clip} className="rl-media" />
                  <span className="rl-hype num">{hypeOf(clip)}</span>
                  <span className="rl-b"><b>{clip.hookLine || clip.title}</b><small className="num">{fmtClock(durationOf(clip))}</small></span>
                </>
              ),
            }))}
          />
        </section>
      ) : (
        <section className="ready glass is-empty">
          <div className="rd-copy">
            <h2 className="disp">All {items.length} keepers are out.</h2>
            <p className="t2">Everything you kept has been exported. Review a new stream to fill this back up.</p>
          </div>
        </section>
      )}

      {items.length > 0 && (
        <>
          <div className="x-bar" ref={groupsRef}>
            <div className="seg" role="group" aria-label="Filter">
              <button type="button" aria-pressed={filter === "all"} onClick={() => setFilter("all")}>All kept <span className="c num">{items.length}</span></button>
              <button type="button" aria-pressed={filter === "todo"} onClick={() => setFilter("todo")}>Not exported <span className="c num">{todo.length}</span></button>
              <button type="button" aria-pressed={filter === "done"} onClick={() => setFilter("done")}>Exported <span className="c num">{done}</span></button>
            </div>
            <span className="sp" />
            <span className="t3 x-format">{format}</span>
            <button type="button" className="btn sm ghost" onClick={onChangeFormat}>Change</button>
          </div>

          <div className="xgroups">
            {groups.map(({ session, items: groupItems }) => {
              const picked = groupItems.filter(({ clip }) => selectedIds.has(clip.id)).length;
              const allPicked = picked === groupItems.length;
              return (
                <section key={session.id} className="xg" aria-label={`Kept clips from ${session.name}`}>
                  <div className="group-h">
                    <span className="gh-poster"><SessionPoster session={session} variant="card" /></span>
                    <div className="gh-copy">
                      <h3 className="disp">{session.name}</h3>
                      <span className="t3">{sessionDate(session)} · {groupItems.length} {groupItems.length === 1 ? "clip" : "clips"}{picked ? ` · ${picked} picked` : ""}</span>
                    </div>
                    <button type="button" className="btn sm ghost" aria-pressed={allPicked} onClick={() => onToggleSession(session.id)}>{allPicked ? "Clear" : "Select all"}</button>
                  </div>
                  <div className="x-clips">
                    {groupItems.map((item) => {
                      const { clip } = item;
                      const selected = selectedIds.has(clip.id);
                      return (
                        <article key={clip.id} className={`x-clip ${selected ? "is-selected" : ""}`}>
                          <button type="button" className="fr" onClick={() => onEdit(session, clip)} aria-label={`Edit ${clip.title} from ${session.name}`}>
                            <Poster clip={clip} className="rl-media" />
                            <span className="dur num">{fmtClock(durationOf(clip))}</span>
                            {isExported(item) && <span className="state badge keep"><Check aria-hidden="true" />Exported</span>}
                          </button>
                          <button type="button" className={`x-sel ${selected ? "on" : ""}`} role="checkbox" aria-checked={selected} aria-label={`${selected ? "Deselect" : "Select"} ${clip.title} for export`} onClick={() => onToggleSelect(clip.id)}>
                            {selected && <Check aria-hidden="true" />}
                          </button>
                          <h4>{clip.hookLine || clip.title}</h4>
                          <span className="m num t3">{clip.timestamp}</span>
                        </article>
                      );
                    })}
                  </div>
                </section>
              );
            })}
          </div>
        </>
      )}

      {selectedCount > 0 && (
        <div className="x-dock glass-float" role="toolbar" aria-label="Export selection">
          <b className="num">{selectedCount} picked</b>
          <button type="button" className="btn sm ghost" onClick={onClearSelection}>Clear</button>
          <button type="button" className="btn heat" disabled={exporting} onClick={onExportSelected}>
            <ExportIcon aria-hidden="true" />{exporting ? "Preparing files…" : `Export ${selectedCount}`}
          </button>
        </div>
      )}
    </div>
  );
}

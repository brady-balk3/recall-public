// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Library: every stream, grouped by month. Posters by default; the list view
 * trades artwork for density and shows a trailing preview on hover.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { SessionPoster, pickSessionPosterClip } from "../components/Posters";
import { fmtDurationHuman } from "../lib/format";
import { Check, Grid, Plus, Queue, Search, Trash2, X } from "../lib/icons";
import type { Session } from "../lib/store";
import { hypeOf } from "../home/homeModel";
import { MomentCurve } from "../ui/MomentCurve";
import { SessionCard, SessionStatusBadge, isFromTwitch, sessionDate } from "../ui/SessionCard";
import type { NewStreamKind } from "../shell/views";

export type LibrarySort = "newest" | "oldest" | "kept" | "name";
type Source = "all" | "twitch" | "local";
type Layout = "grid" | "list";

const LAYOUT_KEY = "recall-library-layout";
const SORTS: { id: LibrarySort; label: string }[] = [
  { id: "newest", label: "Newest" },
  { id: "oldest", label: "Oldest" },
  { id: "kept", label: "Most kept" },
  { id: "name", label: "A–Z" },
];

const when = (session: Session) => session.createdAt || session.updatedAt;

export function sortSessions(sessions: Session[], sort: LibrarySort) {
  const arr = [...sessions];
  if (sort === "name") arr.sort((a, b) => a.name.localeCompare(b.name));
  else if (sort === "kept") arr.sort((a, b) => b.savedClipIds.length - a.savedClipIds.length || when(b) - when(a));
  else if (sort === "oldest") arr.sort((a, b) => when(a) - when(b));
  else arr.sort((a, b) => when(b) - when(a));
  return arr;
}

/** Month groups for date sorts; one flat group otherwise. */
export function groupSessions(sessions: Session[], sort: LibrarySort) {
  if (sort === "kept" || sort === "name") return [{ key: "all", label: "", sessions }];
  const groups: { key: string; label: string; sessions: Session[] }[] = [];
  for (const session of sessions) {
    const date = new Date(when(session));
    const key = `${date.getFullYear()}-${date.getMonth()}`;
    let group = groups[groups.length - 1];
    if (!group || group.key !== key) {
      group = { key, label: date.toLocaleDateString(undefined, { month: "long", year: "numeric" }), sessions: [] };
      groups.push(group);
    }
    group.sessions.push(session);
  }
  return groups;
}

function readLayout(): Layout {
  try { return localStorage.getItem(LAYOUT_KEY) === "list" ? "list" : "grid"; } catch { return "grid"; }
}

export interface LibraryViewProps {
  sessions: Session[];
  onOpen: (session: Session) => void;
  onDelete: (session: Session) => void;
  onDeleteMany: (sessions: Session[]) => void;
  onNew: (kind: NewStreamKind) => void;
}

export function LibraryView({ sessions, onOpen, onDelete, onDeleteMany, onNew }: LibraryViewProps) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<LibrarySort>("newest");
  const [source, setSource] = useState<Source>("all");
  const [layout, setLayoutState] = useState<Layout>(readLayout);
  const [selecting, setSelecting] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());

  const setLayout = (next: Layout) => {
    setLayoutState(next);
    try { localStorage.setItem(LAYOUT_KEY, next); } catch {}
  };

  const hasBothSources = sessions.some(isFromTwitch) && sessions.some((s) => !isFromTwitch(s));
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return sortSessions(
      sessions.filter((s) => (!q || s.name.toLowerCase().includes(q)) && (source === "all" || (source === "twitch") === isFromTwitch(s))),
      sort,
    );
  }, [sessions, query, sort, source]);
  const groups = groupSessions(visible, sort);

  const visibleIds = useMemo(() => new Set(visible.map((s) => s.id)), [visible]);
  useEffect(() => {
    setSelectedIds((prev) => {
      const next = new Set([...prev].filter((id) => visibleIds.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [visibleIds]);
  const picked = visible.filter((s) => selectedIds.has(s.id));
  const allPicked = visible.length > 0 && picked.length === visible.length;

  const exitSelecting = useCallback(() => { setSelecting(false); setSelectedIds(new Set()); }, []);
  const toggle = useCallback((session: Session) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(session.id)) next.delete(session.id);
      else next.add(session.id);
      return next;
    });
  }, []);

  const hours = Math.round(sessions.reduce((sum, s) => sum + (s.vodDuration ?? 0), 0) / 3600);
  const keepers = sessions.reduce((sum, s) => sum + s.savedClipIds.length, 0);

  if (!sessions.length) {
    return (
      <div className="page library">
        <div className="ph"><div><h1 className="disp">Library</h1><div className="sub">Every stream you scan lands here.</div></div></div>
        <div className="lib-empty glass">
          <h2 className="disp">Nothing here yet.</h2>
          <p className="t3">Scan your first stream and it shows up here with its best moments.</p>
          <div className="w-cta">
            <button type="button" className="btn heat lg" onClick={() => onNew("local")}><Plus aria-hidden="true" />New stream</button>
            <button type="button" className="btn lg" onClick={() => onNew("batch")}><Queue aria-hidden="true" />Queue tonight's VODs</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page library">
      <div className="ph">
        <div>
          <h1 className="disp">Library</h1>
          <div className="sub num">{sessions.length} {sessions.length === 1 ? "session" : "sessions"} · {hours} {hours === 1 ? "hour" : "hours"} of stream · {keepers} {keepers === 1 ? "keeper" : "keepers"}</div>
        </div>
        <div className="actions">
          <div className="seg" role="group" aria-label="Layout">
            <button type="button" aria-pressed={layout === "grid"} onClick={() => setLayout("grid")}><Grid aria-hidden="true" />Grid</button>
            <button type="button" aria-pressed={layout === "list"} onClick={() => setLayout("list")}><Queue aria-hidden="true" />List</button>
          </div>
          <button type="button" className="btn" aria-pressed={selecting} onClick={() => (selecting ? exitSelecting() : setSelecting(true))}>
            {selecting ? <><X aria-hidden="true" />Done</> : "Select"}
          </button>
        </div>
      </div>

      <div className="lib-bar">
        <label className="search-f">
          <Search aria-hidden="true" />
          <input className="field" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Filter by title" aria-label="Filter sessions by title" />
        </label>
        {hasBothSources && (
          <div className="chips" role="group" aria-label="Source">
            {(["all", "twitch", "local"] as Source[]).map((id) => (
              <button key={id} type="button" className="chip" aria-pressed={source === id} onClick={() => setSource(id)}>
                {id === "all" ? "All sources" : id === "twitch" ? "Twitch" : "Recordings"}
              </button>
            ))}
          </div>
        )}
        <span className="sp" />
        <div className="seg" role="group" aria-label="Sort">
          {SORTS.map((s) => <button key={s.id} type="button" aria-pressed={sort === s.id} onClick={() => setSort(s.id)}>{s.label}</button>)}
        </div>
      </div>

      {selecting && (
        <div className="lib-select glass" role="toolbar" aria-label="Session selection">
          <button type="button" className="btn sm" onClick={() => setSelectedIds(allPicked ? new Set() : new Set(visible.map((s) => s.id)))}>
            {allPicked ? "Clear all" : `Select all ${visible.length}`}
          </button>
          <span className="t3" role="status">{picked.length} selected</span>
          <span className="sp" />
          <button type="button" className="btn sm danger" disabled={!picked.length} onClick={() => { onDeleteMany(picked); exitSelecting(); }}>
            <Trash2 aria-hidden="true" />Delete{picked.length ? ` ${picked.length}` : ""}
          </button>
        </div>
      )}

      {!visible.length ? (
        <div className="lib-none t3" role="status"><Search aria-hidden="true" />No sessions match “{query.trim()}”.</div>
      ) : (
        groups.map((group) => (
          <section key={group.key} className="lib-g" aria-label={group.label || "Sessions"}>
            {group.label && <div className="lib-gh"><b>{group.label}</b><span className="t3">{group.sessions.length} {group.sessions.length === 1 ? "session" : "sessions"}</span></div>}
            {layout === "grid" ? (
              <div className="sessions lib-grid">
                {group.sessions.map((session) => (
                  <SessionCard
                    key={session.id}
                    session={session}
                    onOpen={onOpen}
                    badge={<SessionStatusBadge session={session} />}
                    selecting={selecting}
                    selected={selectedIds.has(session.id)}
                    onToggle={toggle}
                    onDelete={onDelete}
                  />
                ))}
              </div>
            ) : (
              <SessionList sessions={group.sessions} onOpen={onOpen} onDelete={onDelete} selecting={selecting} selectedIds={selectedIds} onToggle={toggle} />
            )}
          </section>
        ))
      )}
    </div>
  );
}

/* ── list ── */

export function SessionList({ sessions, onOpen, onDelete, selecting, selectedIds, onToggle }: {
  sessions: Session[];
  onOpen: (session: Session) => void;
  onDelete: (session: Session) => void;
  selecting: boolean;
  selectedIds: Set<string>;
  onToggle: (session: Session) => void;
}) {
  const [hover, setHover] = useState<Session | null>(null);
  const preview = useRef<HTMLDivElement>(null);
  const frame = useRef(0);
  const pos = useRef({ x: 0, y: 0 });

  const follow = (event: React.PointerEvent) => {
    pos.current = { x: event.clientX, y: event.clientY };
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      const el = preview.current;
      if (!el) return;
      // Trail the cursor on the right, flipping left near the window edge.
      const w = el.offsetWidth;
      const x = pos.current.x + 24 + w > window.innerWidth ? pos.current.x - 24 - w : pos.current.x + 24;
      el.style.transform = `translate3d(${x}px, ${pos.current.y - el.offsetHeight / 2}px, 0)`;
    });
  };
  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  const hoverClip = hover ? pickSessionPosterClip(hover) : undefined;
  const peak = hover ? hover.clips.reduce((top, clip) => Math.max(top, hypeOf(clip)), 0) : 0;

  return (
    <div className="lrows glass" onPointerMove={follow} onPointerLeave={() => setHover(null)}>
      {sessions.map((session) => {
        const selected = selectedIds.has(session.id);
        return (
          <div key={session.id} className={`lrow ${selected ? "selected" : ""}`} onPointerEnter={() => setHover(session)}>
            <button
              type="button"
              className="lrow-main"
              aria-pressed={selecting ? selected : undefined}
              onClick={() => (selecting ? onToggle(session) : onOpen(session))}
            >
              {selecting && <span className={`chk ${selected ? "on" : ""}`} aria-hidden="true">{selected && <Check />}</span>}
              <span className="th"><SessionPoster session={session} variant="card" /></span>
              <span className="nm">
                <b>{session.name}</b>
                <small>{isFromTwitch(session) ? "Twitch" : "Recording"} · {sessionDate(session)}</small>
              </span>
              <span className="spark">{session.clips.length > 0 && <MomentCurve clips={session.clips} duration={session.vodDuration} className="spark-svg" />}</span>
              <span className="kp num">{session.vodDuration ? fmtDurationHuman(session.vodDuration) : ""}</span>
              <span className="st"><SessionStatusBadge session={session} /></span>
            </button>
            {!selecting && (
              <button type="button" className="icon-btn lrow-del" aria-label={`Delete ${session.name}`} title="Delete session" onClick={() => onDelete(session)}>
                <Trash2 aria-hidden="true" />
              </button>
            )}
          </div>
        );
      })}
      {/* In <body>: the list's glass (backdrop-filter) would trap a fixed child. */}
      {createPortal(<div ref={preview} className={`hovimg ${hover && hoverClip?.thumbUrl ? "on" : ""}`} aria-hidden="true">
        {hoverClip?.thumbUrl && <div className="hv-img" style={{ backgroundImage: `url("${hoverClip.thumbUrl}")` }} />}
        {hover && (
          <div className="hv-b">
            <span className="num">{hover.savedClipIds.length} {hover.savedClipIds.length === 1 ? "keeper" : "keepers"}</span>
            <span className="sp" />
            {peak > 0 && <><span className="t3">peak</span><b className="num">{peak}</b></>}
          </div>
        )}
      </div>, document.body)}
    </div>
  );
}

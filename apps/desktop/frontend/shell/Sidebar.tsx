// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The floating glass sidebar.
 *
 * Every page with its own view is here: Home and Library, then the workflow
 * in the order a stream moves through it (Scans, Review, Cutting Room,
 * Export), then the tools. Power tools that only make sense from a page
 * (compilations, highlight reels) live in that page and in Ctrl K instead.
 */
import { useEffect, useId, useLayoutEffect, useRef, useState, type ComponentType, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { createPortal } from "react-dom";
import {
  BarChart3,
  BookOpen,
  Broadcast,
  ChevronDown,
  ExportIcon,
  Folder,
  Grid,
  Home,
  PlayCircle,
  Plus,
  Pulse,
  Queue,
  Scissors,
  Search,
  Settings,
  Twitch,
  type IconProps,
} from "../lib/icons";
import { LetterAvatar } from "./LetterAvatar";
import { profileLine, type ShellCounts } from "./shellModel";
import type { NewStreamKind, View } from "./views";

type Icon = ComponentType<IconProps>;

export interface SidebarProps {
  view: View;
  counts: ShellCounts;
  displayName: string;
  settingsOpen?: boolean;
  liveActive?: boolean;
  onView: (view: View) => void;
  onNew: (kind: NewStreamKind) => void;
  onSearch: () => void;
  onSettings: () => void;
  /** Where the crew lives. Absent when pets are turned off. */
  den?: { roaming: boolean; slotOpen: boolean; onOpenCrew: () => void };
}

/** Views that light up a sidebar item other than their own. */
const OWNER: Partial<Record<View, View>> = {
  theater: "review",
  processing: "queue",
  "add-vod": "queue",
  compilations: "clips",
  reel: "clips",
};

export function Sidebar({ view, counts, displayName, settingsOpen = false, liveActive = false, onView, onNew, onSearch, onSettings, den }: SidebarProps) {
  const current = settingsOpen ? undefined : OWNER[view] ?? view;
  const item = (key: View, label: string, Glyph: Icon, extra?: React.ReactNode) => {
    const on = current === key;
    return (
      <button
        key={key}
        type="button"
        className={`sb-item ${on ? "on" : ""}`}
        aria-current={on ? "page" : undefined}
        onClick={() => onView(key)}
      >
        <Glyph className="sb-ic" aria-hidden="true" />
        <span className="sb-label">{label}</span>
        {extra}
      </button>
    );
  };
  const count = (n: number, heat = false, label?: string) =>
    n > 0 ? <span className={`sb-n num ${heat ? "heat" : ""}`} aria-label={label}>{n}</span> : null;

  const scanMeta = counts.scan
    ? <span className="sb-meta num"><ProgressRing value={counts.scan.progress} /><span>{Math.floor(counts.scan.progress)}%</span></span>
    : count(counts.queued, false, `${counts.queued} queued`);

  return (
    <aside className="sidebar glass-float" aria-label="Recall">
      <button type="button" className="sb-me" onClick={onSettings} title="Your profile and settings">
        <LetterAvatar name={displayName} size={38} />
        <span className="sb-who">
          <b>{displayName.trim() || "Creator"}</b>
          <small>{profileLine(counts)}</small>
        </span>
      </button>

      <NewStreamButton onNew={onNew} onLive={() => onView("live")} liveActive={liveActive} />

      <button type="button" className="sb-search" onClick={onSearch}>
        <Search className="sb-ic" aria-hidden="true" />
        <span>Search</span>
        <kbd>Ctrl K</kbd>
      </button>

      <nav className="sb-group" aria-label="Library">
        {item("home", "Home", Home)}
        {item("sessions", "Library", Grid, count(counts.sessions, false, `${counts.sessions} streams`))}
      </nav>

      <div className="sb-h" id="sb-h-workflow">Workflow</div>
      <nav className="sb-group" aria-labelledby="sb-h-workflow">
        {item("queue", "Scans", Pulse, scanMeta)}
        {item("review", "Review", PlayCircle, count(counts.toReview, true, `${counts.toReview} to review`))}
        {item("cutting", "Cutting Room", Scissors)}
        {item("clips", "Export", ExportIcon, count(counts.toExport, false, `${counts.toExport} to export`))}
      </nav>

      <div className="sb-h" id="sb-h-tools">Tools</div>
      <nav className="sb-group" aria-labelledby="sb-h-tools">
        {item("memory", "Memory", BookOpen)}
        {item("live", "Recall Live", Broadcast, liveActive ? <span className="sb-live-dot" aria-label="Listening" /> : null)}
        {item("insights", "Insights", BarChart3)}
      </nav>

      <span className="sb-spacer" />

      {den && (
        <div className={`den ${den.roaming ? "away" : ""}`} data-den aria-label="Your crew">
          {den.roaming && <small>Out watching the scan</small>}
          {den.slotOpen && !den.roaming && (
            <button type="button" className="den-open" onClick={den.onOpenCrew} title="Your second crew slot is open" aria-label="Your second crew slot is open">
              <Plus size={11} aria-hidden="true" />
            </button>
          )}
        </div>
      )}

      {counts.scan && (
        <button type="button" className="sb-scan" onClick={() => onView("queue")}>
          <span className="sb-scan-row">
            <span className="sb-scan-label">Scanning</span>
            <span className="num">{Math.floor(counts.scan.progress)}%</span>
          </span>
          <b>{counts.scan.title}</b>
          <span className="sb-bar" aria-hidden="true"><i style={{ transform: `scaleX(${counts.scan.progress / 100})` }} /></span>
        </button>
      )}

      <button
        type="button"
        className={`sb-item ${settingsOpen ? "on" : ""}`}
        aria-current={settingsOpen ? "page" : undefined}
        onClick={onSettings}
      >
        <Settings className="sb-ic" aria-hidden="true" />
        <span className="sb-label">Settings</span>
      </button>
    </aside>
  );
}

export function ProgressRing({ value, size = 16 }: { value: number; size?: number }) {
  const r = 6.5;
  const circumference = 2 * Math.PI * r;
  return (
    <svg className="sb-ring" viewBox="0 0 16 16" width={size} height={size} aria-hidden="true">
      <circle className="bg" cx="8" cy="8" r={r} />
      <circle
        className="fg"
        cx="8"
        cy="8"
        r={r}
        strokeDasharray={circumference}
        strokeDashoffset={circumference * (1 - Math.max(0, Math.min(100, value)) / 100)}
      />
    </svg>
  );
}

/**
 * New stream: the main part imports a recording (the common case), the arrow
 * opens every other way in. The menu is a real menu: arrow keys move, Escape
 * closes and returns focus to the arrow.
 */
function NewStreamButton({ onNew, onLive, liveActive }: { onNew: (kind: NewStreamKind) => void; onLive: () => void; liveActive: boolean }) {
  const [open, setOpen] = useState(false);
  const [anchor, setAnchor] = useState<{ left: number; top: number; originX: number }>();
  const menuId = useId();
  const wrapRef = useRef<HTMLDivElement>(null);
  const moreRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // The menu is wider than the sidebar, and the sidebar's glass (backdrop-filter)
  // would trap a fixed child, so it renders in <body> under the button.
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const r = wrapRef.current?.getBoundingClientRect();
      // Grow from the chevron at the right end of the split button.
      if (r) setAnchor({ left: r.left, top: r.bottom + 6, originX: r.width - 18 });
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, [open]);

  useEffect(() => {
    if (!open || !anchor) return;
    menuRef.current?.querySelector<HTMLButtonElement>("[role=menuitem]")?.focus();
    const onDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!menuRef.current?.contains(target) && !moreRef.current?.contains(target)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open, anchor]);

  const close = (refocus = true) => {
    setOpen(false);
    if (refocus) moreRef.current?.focus();
  };
  const choose = (run: () => void) => {
    close(false);
    run();
  };

  const onMenuKey = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const items = Array.from(menuRef.current?.querySelectorAll<HTMLButtonElement>("[role=menuitem]") ?? []);
    const index = items.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      items[(index + step + items.length) % items.length]?.focus();
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      items[event.key === "Home" ? 0 : items.length - 1]?.focus();
    } else if (event.key === "Tab") {
      close(false);
    }
  };

  const entries: { id: string; Glyph: Icon; title: string; sub: string; run: () => void }[] = [
    { id: "local", Glyph: Folder, title: "Import a recording", sub: "MP4, MKV, MOV, straight from OBS", run: () => onNew("local") },
    { id: "twitch", Glyph: Twitch, title: "Paste a Twitch link", sub: "Public VODs download once", run: () => onNew("twitch") },
    { id: "batch", Glyph: Queue, title: "Queue tonight's VODs", sub: "Scan several in order", run: () => onNew("batch") },
  ];

  return (
    <div ref={wrapRef} className="sb-new-wrap">
      <button type="button" className="sb-new" onClick={() => onNew("local")}>
        <span className="sb-plus" aria-hidden="true"><Plus size={13} /></span>
        <span>New stream</span>
      </button>
      <button
        ref={moreRef}
        type="button"
        className="sb-new-more"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label="More ways to add a stream"
        title="More ways to add a stream"
        onClick={() => setOpen((value) => !value)}
      >
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      {open && anchor && createPortal(
        <div ref={menuRef} id={menuId} className="sb-menu glass-read" role="menu" aria-label="Add a stream" onKeyDown={onMenuKey} style={{ left: anchor.left, top: anchor.top, transformOrigin: `${anchor.originX}px 0` }}>
          {entries.map(({ id, Glyph, title, sub, run }) => (
            <button key={id} type="button" role="menuitem" tabIndex={-1} onClick={() => choose(run)}>
              <span className="sb-menu-ic"><Glyph size={18} aria-hidden="true" /></span>
              <span><b>{title}</b><small>{sub}</small></span>
            </button>
          ))}
          <div className="sb-menu-sep" role="separator" />
          <button type="button" role="menuitem" tabIndex={-1} onClick={() => choose(onLive)}>
            <span className="sb-menu-ic"><Broadcast size={18} aria-hidden="true" /></span>
            <span>
              <b>{liveActive ? "Recall Live is listening" : "Start Recall Live"}</b>
              <small>Mark moments while you stream</small>
            </span>
          </button>
        </div>,
        document.body,
      )}
    </div>
  );
}

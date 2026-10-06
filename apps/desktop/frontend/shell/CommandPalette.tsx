// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Ctrl K: search your streams, paste a VOD link, or run a command.
 *
 * A combobox, not a menu: focus stays in the input, arrow keys move the
 * active option, Enter runs it. Thick reading glass, so the text stays crisp
 * over whatever is behind it.
 */
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ComponentType } from "react";
import {
  BarChart3,
  BookOpen,
  Broadcast,
  ExportIcon,
  Film,
  Folder,
  Grid,
  Home,
  Keyboard,
  Layers,
  LinkIcon,
  PlayCircle,
  Pulse,
  Queue,
  Scissors,
  Search,
  Settings,
  Twitch,
  type IconProps,
} from "../lib/icons";
import { isTwitchVodLink, paletteItems, type PaletteCommand, type PaletteIcon, type PaletteSession } from "./paletteItems";

const ICONS: Record<PaletteIcon, ComponentType<IconProps>> = {
  twitch: Twitch,
  memory: BookOpen,
  import: Folder,
  batch: Queue,
  live: Broadcast,
  review: PlayCircle,
  export: ExportIcon,
  scissors: Scissors,
  layers: Layers,
  reel: Film,
  settings: Settings,
  keyboard: Keyboard,
  insights: BarChart3,
  library: Grid,
  home: Home,
  scans: Pulse,
};

export interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  sessions: PaletteSession[];
  commands: PaletteCommand[];
  examples?: string[];
  scanLink: (url: string) => void;
  searchMemory: (query: string) => void;
}

export function CommandPalette({ open, onClose, ...rest }: CommandPaletteProps) {
  if (!open) return null;
  return <PaletteDialog onClose={onClose} {...rest} />;
}

function PaletteDialog({ onClose, sessions, commands, examples, scanLink, searchMemory }: Omit<CommandPaletteProps, "open">) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const listId = useId();
  const returnFocus = useRef<Element | null>(null);

  const items = useMemo(
    () => paletteItems({ query, sessions, commands, examples, scanLink, searchMemory }),
    [query, sessions, commands, examples, scanLink, searchMemory],
  );
  const safeActive = Math.min(active, Math.max(0, items.length - 1));
  const link = isTwitchVodLink(query);

  useLayoutEffect(() => {
    returnFocus.current = document.activeElement;
    inputRef.current?.focus();
    return () => {
      const el = returnFocus.current as HTMLElement | null;
      if (el && document.contains(el)) el.focus?.();
    };
  }, []);

  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${safeActive}"]`)?.scrollIntoView({ block: "nearest" });
  }, [safeActive]);

  const run = (index: number) => {
    const item = items[index];
    if (!item) return;
    onClose();
    item.run();
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!items.length) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((safeActive + step + items.length) % items.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      run(safeActive);
    } else if (event.key === "Tab") {
      // Focus lives in the input; the list is driven by the arrow keys.
      event.preventDefault();
    }
  };

  let section = "";
  return (
    <>
      <div className="pal-scrim glass-scrim" onPointerDown={onClose} aria-hidden="true" />
      <div className="palette glass-read" role="dialog" aria-modal="true" aria-label="Search and commands" onKeyDown={onKeyDown}>
        <div className="pal-in">
          {link ? <LinkIcon className="pal-in-ic" aria-hidden="true" /> : <Search className="pal-in-ic" aria-hidden="true" />}
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => { setQuery(event.target.value); setActive(0); }}
            placeholder="Search your streams, paste a VOD link, or type a command"
            role="combobox"
            aria-expanded="true"
            aria-controls={listId}
            aria-activedescendant={items[safeActive] ? `${listId}-${safeActive}` : undefined}
            aria-autocomplete="list"
            autoComplete="off"
            spellCheck={false}
          />
          <kbd>Esc</kbd>
        </div>
        <div ref={listRef} className="pal-list" id={listId} role="listbox" aria-label="Results">
          {items.length === 0 && <div className="pal-empty">Nothing matches.</div>}
          {items.map((item, index) => {
            const header = item.section !== section ? <div className="pal-sec" role="presentation">{item.section}</div> : null;
            section = item.section;
            const Glyph = item.icon ? ICONS[item.icon] : undefined;
            const on = index === safeActive;
            return (
              <div key={item.id} role="presentation">
                {header}
                <div
                  id={`${listId}-${index}`}
                  role="option"
                  aria-selected={on}
                  data-index={index}
                  className={`pal-item ${on ? "on" : ""}`}
                  onPointerMove={() => { if (!on) setActive(index); }}
                  onClick={() => run(index)}
                >
                  <span className={`pal-ic ${item.image ? "img" : ""}`} style={item.image ? { backgroundImage: `url("${item.image}")` } : undefined}>
                    {!item.image && Glyph && <Glyph size={18} aria-hidden="true" />}
                  </span>
                  <span className="pal-text">
                    <b>{item.title}</b>
                    {item.sub && <small>{item.sub}</small>}
                  </span>
                  {on && <kbd>Enter</kbd>}
                </div>
              </div>
            );
          })}
        </div>
        <div className="pal-foot" aria-hidden="true">
          <span><kbd>↑</kbd> <kbd>↓</kbd> move</span>
          <span><kbd>Enter</kbd> open</span>
          <span>Paste a twitch.tv/videos link to scan it</span>
        </div>
      </div>
    </>
  );
}

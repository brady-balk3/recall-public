// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * One page of the app shell. Home, Library, Scans and the rest stay mounted so
 * coming back to them keeps their scroll and state, but a hidden page doesn't
 * re-render: every switch (and every engine update) used to redraw all of them,
 * which made each switch cost the whole app. A hidden page keeps showing what
 * it last drew and catches up the moment it's shown again.
 *
 * Never for a page that must unmount when you leave (the editor, the theater:
 * anything holding video or audio). Its content would stay alive behind the
 * other pages.
 *
 * It also tells the page whether it is the one on screen (useViewActive), so
 * what a page reads from the store or polls on its own can wait while hidden.
 */
import { type ReactNode, memo, useRef } from "react";
import { ViewActiveContext } from "./viewActive";

function Panel({ id, active, lazy = false, children }: {
  id: string;
  active: boolean;
  /** Build nothing until the page is first shown (a heavy page nobody may open). */
  lazy?: boolean;
  children: ReactNode;
}) {
  // The children last drawn while active. Handing React the very same element
  // when the page hides lets it skip the whole subtree.
  const shown = useRef<ReactNode>(lazy && !active ? null : children);
  if (active) shown.current = children;
  return (
    <div className={`view-panel ${active ? "active" : ""}`} id={id}>
      <ViewActiveContext.Provider value={active}>{shown.current}</ViewActiveContext.Provider>
    </div>
  );
}

/** Skips the render entirely while the page stays hidden. */
export const ViewPanel = memo(Panel, (prev, next) => !prev.active && !next.active && prev.id === next.id);

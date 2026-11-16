// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * A block of cards that builds only when it nears the window. Long pages
 * (hundreds of keepers, every stream ever scanned) are thousands of elements,
 * so a block off screen holds one plain tile per card instead: same grid, same
 * size, so the page and its scrollbar are already the right length. Built
 * blocks stay built while they stay mounted.
 */
import { type ReactNode, useEffect, useRef, useState } from "react";

/** How far outside the window a block fills in, so it's there before you scroll to it. */
export const FILL_AHEAD = "1200px 0px";

export function LazyFill({ eager, count, className, placeholderClass, children }: {
  /** Build straight away (the first screen never waits). */
  eager: boolean;
  /** Cards in the block: one placeholder tile each until it builds. */
  count: number;
  className: string;
  placeholderClass: string;
  children: () => ReactNode;
}) {
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === "undefined");
  // A block that moves up to the first screen (a filter, a new sort) builds in
  // that same render, and stays built if it later moves back down.
  if (eager && !seen) setSeen(true);
  const built = eager || seen;
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (built || !el) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { setSeen(true); observer.disconnect(); }
    }, { rootMargin: FILL_AHEAD });
    observer.observe(el);
    return () => observer.disconnect();
  }, [built]);
  if (built) return <div className={className}>{children()}</div>;
  return (
    <div className={`${className} is-pending`} ref={ref} aria-busy="true">
      {Array.from({ length: count }, (_, i) => <div key={i} className={placeholderClass} />)}
    </div>
  );
}

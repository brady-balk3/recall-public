// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * A strip of vertical clip cards you can grab and fling.
 *
 * Drag moves the strip 1:1; letting go keeps the velocity and bleeds it off,
 * so a quick flick travels further than a slow drag. Past either end the strip
 * resists (rubber band) and settles back. A drag never counts as a click on
 * the card under the pointer. Wheel and arrow keys scroll it too. Reduced
 * motion drops the inertia and the settle animation.
 */
import { useCallback, useEffect, useRef, type ReactNode } from "react";

const FRICTION = 0.94;
const CLICK_SLOP = 5;
/** px per frame; a hard flick shouldn't send the strip off to the far end. */
const MAX_FLING = 48;

export interface ReelItem {
  key: string;
  content: ReactNode;
  onOpen: () => void;
  label: string;
}

const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/**
 * `drift` slowly carries the strip back and forth on its own (the Export
 * "ready to post" reel). It pauses while the pointer is over it, while it has
 * focus, during a drag, and entirely under reduced motion.
 */
export function KeepersReel({ items, label, drift = false }: { items: ReelItem[]; label: string; drift?: boolean }) {
  const viewport = useRef<HTMLDivElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const x = useRef(0);
  const frame = useRef(0);
  const drag = useRef<{ id: number; startX: number; startOffset: number; lastX: number; lastT: number; v: number; moved: boolean } | null>(null);
  const suppressClick = useRef(false);

  const bounds = useCallback(() => {
    const vw = viewport.current?.clientWidth ?? 0;
    const tw = track.current?.scrollWidth ?? 0;
    return { min: Math.min(0, vw - tw), max: 0 };
  }, []);

  const paint = () => {
    if (track.current) track.current.style.transform = `translate3d(${x.current}px, 0, 0)`;
  };

  const settle = useCallback((velocity = 0) => {
    cancelAnimationFrame(frame.current);
    const reduce = prefersReducedMotion();
    let v = reduce ? 0 : velocity;
    const step = () => {
      const { min, max } = bounds();
      x.current += v;
      v *= FRICTION;
      // Out of bounds: pull back toward the edge instead of stopping dead.
      const over = x.current > max ? x.current - max : x.current < min ? x.current - min : 0;
      if (over) {
        if (reduce) x.current -= over;
        else {
          x.current -= over * 0.18;
          v *= 0.6;
        }
      }
      paint();
      if (Math.abs(v) > 0.2 || Math.abs(over) > 0.5) frame.current = requestAnimationFrame(step);
      else if (over) {
        x.current -= over;
        paint();
      }
    };
    frame.current = requestAnimationFrame(step);
  }, [bounds]);

  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  const paused = useRef(false);
  useEffect(() => {
    if (!drift || prefersReducedMotion()) return;
    let dir = -1;
    let raf = 0;
    const tick = () => {
      if (!paused.current && !drag.current) {
        const { min, max } = bounds();
        if (min < 0) {
          x.current += dir * 0.35;
          if (x.current <= min) { x.current = min; dir = 1; }
          if (x.current >= max) { x.current = max; dir = -1; }
          paint();
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [drift, bounds]);
  useEffect(() => {
    // Content changed (fewer keepers, window resized): stay inside the strip.
    const { min } = bounds();
    if (x.current < min) {
      x.current = min;
      paint();
    }
  });

  const onPointerDown = (event: React.PointerEvent) => {
    if (event.button !== 0) return;
    cancelAnimationFrame(frame.current);
    drag.current = { id: event.pointerId, startX: event.clientX, startOffset: x.current, lastX: event.clientX, lastT: performance.now(), v: 0, moved: false };
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const d = drag.current;
    if (!d || d.id !== event.pointerId) return;
    const dx = event.clientX - d.startX;
    if (!d.moved && Math.abs(dx) > CLICK_SLOP) {
      d.moved = true;
      viewport.current?.setPointerCapture(event.pointerId);
      viewport.current?.classList.add("grab");
    }
    if (!d.moved) return;
    const { min, max } = bounds();
    let next = d.startOffset + dx;
    // Rubber band past the ends: the further you pull, the less it moves.
    if (next > max) next = max + (next - max) * 0.35;
    if (next < min) next = min + (next - min) * 0.35;
    x.current = next;
    const now = performance.now();
    const dt = Math.max(1, now - d.lastT);
    d.v = Math.max(-MAX_FLING, Math.min(MAX_FLING, ((event.clientX - d.lastX) / dt) * 16));
    d.lastX = event.clientX;
    d.lastT = now;
    paint();
  };

  const endDrag = (event: React.PointerEvent) => {
    const d = drag.current;
    if (!d || d.id !== event.pointerId) return;
    drag.current = null;
    viewport.current?.classList.remove("grab");
    if (d.moved) {
      suppressClick.current = true;
      settle(d.v);
    }
  };

  const onWheel = (event: React.WheelEvent) => {
    const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.shiftKey ? event.deltaY : 0;
    if (!delta) return;
    const { min, max } = bounds();
    x.current = Math.max(min, Math.min(max, x.current - delta));
    paint();
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    const card = track.current?.firstElementChild as HTMLElement | null;
    const step = (card?.offsetWidth ?? 180) + 14;
    const { min, max } = bounds();
    x.current = Math.max(min, Math.min(max, x.current + (event.key === "ArrowRight" ? -step : step)));
    paint();
  };

  return (
    <div
      ref={viewport}
      className="reel"
      role="region"
      aria-label={label}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onWheel={onWheel}
      onKeyDown={onKeyDown}
      onPointerEnter={() => { paused.current = true; }}
      onPointerLeave={() => { paused.current = false; }}
      onFocus={() => { paused.current = true; }}
      onBlur={() => { paused.current = false; }}
      onClickCapture={(event) => {
        if (suppressClick.current) {
          suppressClick.current = false;
          event.stopPropagation();
          event.preventDefault();
        }
      }}
    >
      <div ref={track} className="reel-track">
        {items.map((item) => (
          <button key={item.key} type="button" className="rl-card" aria-label={item.label} onClick={item.onOpen} onDragStart={(e) => e.preventDefault()}>
            {item.content}
          </button>
        ))}
      </div>
    </div>
  );
}

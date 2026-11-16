// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * One way to poll, and one clock for "now".
 *
 * Pages kept mounted behind others (ViewPanel) and a minimized window used to
 * keep every poller and 1s clock running: status fetches nobody would see and
 * re-renders of pages nobody was looking at. usePolling waits while the window
 * is hidden and while its page isn't the one on screen, catches up at once
 * when it's back (if a run is due), and never starts a run while the last one
 * is still waiting on the engine. useNow shares a single 1s timer between every
 * "elapsed" / "done around" display instead of one timer each.
 *
 * Anything that must keep running in the background (the job index refresh
 * behind desktop notifications and keep-awake, a scan's polling fallback)
 * does not use these.
 */
import { useEffect, useRef, useSyncExternalStore } from "react";
import { useViewActive } from "../shell/viewActive";

function subscribeVisibility(onChange: () => void) {
  document.addEventListener("visibilitychange", onChange);
  return () => document.removeEventListener("visibilitychange", onChange);
}
const documentVisible = () => document.visibilityState === "visible";

/** Is the window on screen (not minimized, not a hidden tab)? */
export function useDocumentVisible(): boolean {
  return useSyncExternalStore(subscribeVisibility, documentVisible, () => true);
}

export interface PollingOptions {
  /** Off: no timer at all. Turning it back on starts a fresh schedule. */
  enabled?: boolean;
  /** Run the first time as soon as it can, instead of one interval in. */
  leading?: boolean;
  /** Keep running while its page is hidden behind another (still waits while the window is hidden). */
  whileInactive?: boolean;
}

/**
 * Run `poll` every `intervalMs`, as setInterval would, but only while the
 * window and the page it sits on are shown. When either comes back, a run that
 * fell due meanwhile happens at once. A run that is still pending makes the
 * next tick skip. `poll` gets `isCurrent`, false once this schedule ended
 * (unmounted, disabled, or `poll` changed): check it before setting state from
 * a reply. A new `poll` (its useCallback deps changed) starts a new schedule,
 * the way an effect re-runs on its deps.
 */
export function usePolling(
  poll: (isCurrent: () => boolean) => unknown,
  intervalMs: number,
  { enabled = true, leading = false, whileInactive = false }: PollingOptions = {},
): void {
  const shown = useViewActive();
  const visible = useDocumentVisible();
  const runnable = enabled && visible && (shown || whileInactive);
  const schedule = useRef<{ alive: boolean; last: number; pending: boolean } | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const current = { alive: true, last: leading ? 0 : Date.now(), pending: false };
    schedule.current = current;
    return () => { current.alive = false; };
  }, [poll, intervalMs, enabled, leading]);

  useEffect(() => {
    const current = schedule.current;
    if (!runnable || !current?.alive) return;
    const tick = () => {
      if (current.pending || !current.alive) return;
      current.pending = true;
      current.last = Date.now();
      let result: unknown;
      try {
        result = poll(() => current.alive);
      } catch (error) {
        console.error("Polling error:", error);
      }
      if (result instanceof Promise) {
        void result
          .catch((error) => console.error("Polling error:", error))
          .finally(() => { current.pending = false; });
      } else {
        current.pending = false;
      }
    };
    if (Date.now() - current.last >= intervalMs) tick();
    const timer = window.setInterval(tick, intervalMs);
    return () => window.clearInterval(timer);
  }, [runnable, poll, intervalMs, enabled, leading]);
}

// --- One clock ---------------------------------------------------------------

const CLOCK_MS = 1000;
const clockListeners = new Set<() => void>();
let clockNow = Date.now();
let clockTimer: number | undefined;

function tickClock() {
  clockNow = Date.now();
  for (const listener of clockListeners) listener();
}

function syncClockTimer() {
  const run = clockListeners.size > 0 && documentVisible();
  if (run && clockTimer === undefined) {
    // Fresh the moment it (re)starts: the last tick may be long gone.
    tickClock();
    clockTimer = window.setInterval(tickClock, CLOCK_MS);
  } else if (!run && clockTimer !== undefined) {
    window.clearInterval(clockTimer);
    clockTimer = undefined;
  }
}

let visibilityHooked = false;
function subscribeClock(listener: () => void) {
  if (!visibilityHooked && typeof document !== "undefined") {
    visibilityHooked = true;
    document.addEventListener("visibilitychange", syncClockTimer);
  }
  clockListeners.add(listener);
  syncClockTimer();
  return () => {
    clockListeners.delete(listener);
    syncClockTimer();
  };
}
const readClock = () => clockNow;
const noSubscription = () => () => {};

/**
 * Date.now(), refreshed every second while `running` and the page and window
 * are shown, from one timer shared by every caller. Otherwise it holds the
 * last value it showed.
 */
export function useNow(running = true): number {
  const shown = useViewActive();
  const live = running && shown;
  const held = useRef<number | null>(null);
  if (held.current === null) held.current = Date.now();
  const now = useSyncExternalStore(live ? subscribeClock : noSubscription, live ? readClock : () => held.current as number);
  held.current = now;
  return now;
}

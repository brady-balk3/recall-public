// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Is a Recall Live session running, how many moments has it marked, and what
 * is the Remember hotkey? One poll for the whole app (sidebar dot, Home card).
 */
import { useCallback, useEffect, useState } from "react";
import { usePolling } from "./polling";
import { getActiveRecallSession } from "./recallSessions";

export interface RecallLiveState {
  active: boolean;
  marks: number;
  /** Display form, e.g. "Ctrl+Shift+R"; null when no hotkey is registered. */
  hotkey: string | null;
}

const POLL_MS = 15_000;

export function displayAccelerator(accelerator: string | null | undefined): string | null {
  if (!accelerator) return null;
  return accelerator
    .replace(/CommandOrControl|CmdOrCtrl|Control/g, "Ctrl")
    .replace(/Command|Cmd/g, "Cmd");
}

export function useRecallLive(): RecallLiveState {
  const [state, setState] = useState<RecallLiveState>({ active: false, marks: 0, hotkey: null });

  // Checked at once, then every 15s while the window is on screen: nothing
  // but the sidebar dot and the Home card reads it, so a minimized window
  // skips it and looks again as soon as it's back.
  const check = useCallback((isCurrent: () => boolean) => getActiveRecallSession()
    .then((session) => {
      if (!isCurrent()) return;
      const active = session?.status === "active";
      const marks = active ? session?.events?.filter((event) => event.kind === "remember").length ?? 0 : 0;
      setState((prev) => (prev.active === active && prev.marks === marks ? prev : { ...prev, active, marks }));
    })
    .catch(() => { if (isCurrent()) setState((prev) => (prev.active ? { ...prev, active: false, marks: 0 } : prev)); }), []);
  usePolling(check, POLL_MS, { leading: true });

  useEffect(() => {
    let mounted = true;
    void window.electronAPI?.getRememberHotkey?.()
      .then((hotkey) => {
        if (mounted) setState((prev) => ({ ...prev, hotkey: hotkey?.registered ? displayAccelerator(hotkey.accelerator) : null }));
      })
      .catch(() => {});
    const unsubscribe = window.electronAPI?.onRememberHotkeyState?.((hotkey) => {
      if (mounted) setState((prev) => ({ ...prev, hotkey: hotkey.registered ? displayAccelerator(hotkey.accelerator) : null }));
    });

    return () => {
      mounted = false;
      if (typeof unsubscribe === "function") unsubscribe();
    };
  }, []);

  return state;
}

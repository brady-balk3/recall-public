// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Is a Recall Live session running, how many moments has it marked, and what
 * is the Remember hotkey? One poll for the whole app (sidebar dot, Home card).
 */
import { useEffect, useState } from "react";
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

  useEffect(() => {
    let mounted = true;
    const check = () => {
      void getActiveRecallSession()
        .then((session) => {
          if (!mounted) return;
          const active = session?.status === "active";
          const marks = active ? session?.events?.filter((event) => event.kind === "remember").length ?? 0 : 0;
          setState((prev) => (prev.active === active && prev.marks === marks ? prev : { ...prev, active, marks }));
        })
        .catch(() => { if (mounted) setState((prev) => (prev.active ? { ...prev, active: false, marks: 0 } : prev)); });
    };
    check();
    const timer = window.setInterval(check, POLL_MS);

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
      window.clearInterval(timer);
      if (typeof unsubscribe === "function") unsubscribe();
    };
  }, []);

  return state;
}

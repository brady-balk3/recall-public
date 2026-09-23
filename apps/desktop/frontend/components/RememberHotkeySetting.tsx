// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Check, Keyboard, X } from "lucide-react";
import type { RememberHotkeyState } from "../electron";

const MODIFIER_KEYS = new Set(["Control", "Alt", "Shift", "Meta", "OS", "AltGraph"]);

export function displayHotkey(value: string | null | undefined): string {
  if (!value) return "Remember button";
  return value.replace("CommandOrControl", "Ctrl").replace(/\+/g, " + ");
}

/** Turn a browser key event into an Electron accelerator, or null if it is not
 *  yet a usable chord (a lone modifier, or a key with no modifier at all). */
export function acceleratorFromEvent(event: {
  key: string; code: string; ctrlKey: boolean; altKey: boolean;
  shiftKey: boolean; metaKey: boolean;
}): string | null {
  if (MODIFIER_KEYS.has(event.key)) return null;
  const parts: string[] = [];
  if (event.ctrlKey) parts.push("CommandOrControl");
  if (event.altKey) parts.push("Alt");
  if (event.shiftKey) parts.push("Shift");
  if (event.metaKey) parts.push("Super");
  if (!parts.length) return null;

  let key = "";
  if (/^Key[A-Z]$/.test(event.code)) key = event.code.slice(3);
  else if (/^Digit[0-9]$/.test(event.code)) key = event.code.slice(5);
  else if (/^F([1-9]|1[0-9]|2[0-4])$/.test(event.code)) key = event.code;
  else if (event.key.length === 1 && /[A-Za-z0-9]/.test(event.key)) key = event.key.toUpperCase();
  if (!key) return null;

  parts.push(key);
  return parts.join("+");
}

export default function RememberHotkeySetting() {
  const [state, setState] = useState<RememberHotkeyState | null>(null);
  const [listening, setListening] = useState(false);
  const [pendingError, setPendingError] = useState<string | null>(null);
  const listenButton = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    // `api?.getRememberHotkey?.()` short-circuits to undefined when the method
    // is missing -- and `.then()` on undefined throws, taking the renderer with
    // it. A preload built before this method existed is exactly that case, so
    // the call is guarded rather than chained.
    const api = window.electronAPI;
    const pending = api && typeof api.getRememberHotkey === "function"
      ? api.getRememberHotkey()
      : null;
    if (pending && typeof pending.then === "function") {
      pending.then(
        (next) => { if (!cancelled && next) setState(next); },
        () => { /* engine not up yet; the control simply stays hidden */ },
      );
    }
    const stop = typeof api?.onRememberHotkeyState === "function"
      ? api.onRememberHotkeyState((next) => setState(next))
      : undefined;
    return () => { cancelled = true; stop?.(); };
  }, []);

  const commit = useCallback(async (accelerator: string | null) => {
    const next = await window.electronAPI?.setRememberHotkey?.(accelerator);
    setListening(false);
    if (next) {
      setState(next);
      // A rejected change rolls back to the previous working shortcut, so the
      // reply can be `registered: true` and still carry the reason it failed.
      setPendingError(next.error);
    }
    listenButton.current?.focus();
  }, []);

  useEffect(() => {
    if (!listening) return;
    const onKeyDown = (event: KeyboardEvent) => {
      event.preventDefault();
      event.stopPropagation();
      if (event.key === "Escape") { setListening(false); return; }
      const accelerator = acceleratorFromEvent(event);
      if (accelerator) void commit(accelerator);
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [listening, commit]);

  if (!state) return null;

  const locked = state.source === "environment";
  const broken = Boolean(state.accelerator) && !state.registered;
  const error = pendingError || (broken ? state.error : null);

  return (
    <div className="hotkey-setting">
      <div className="hotkey-row">
        <Keyboard size={14} aria-hidden="true" />
        <span className="hotkey-label">Remember shortcut</span>
        {listening ? (
          <kbd className="hotkey-listening">Press a shortcut…</kbd>
        ) : (
          <kbd className={broken ? "is-broken" : undefined}>{displayHotkey(state.accelerator)}</kbd>
        )}
        {!state.accelerator && !listening && (
          <span className="hotkey-off">Off — use the Remember button</span>
        )}
        <button
          type="button" className="btn-secondary" ref={listenButton}
          disabled={locked}
          aria-pressed={listening}
          onClick={() => { setPendingError(null); setListening((value) => !value); }}
        >
          {listening ? <X size={13} aria-hidden="true" /> : <Check size={13} aria-hidden="true" />}
          {listening ? "Cancel" : state.accelerator ? "Change" : "Set a shortcut"}
        </button>
        {state.accelerator && !listening && !locked && (
          <button type="button" className="btn-secondary" onClick={() => void commit(null)}>
            Turn off
          </button>
        )}
      </div>

      {locked && (
        <p className="hotkey-note">
          Fixed by the RECALL_REMEMBER_HOTKEY environment variable.
        </p>
      )}
      {error && (
        <p className="hotkey-error" role="alert">
          <AlertTriangle size={13} aria-hidden="true" />
          {error}
        </p>
      )}
      {listening && (
        <p className="hotkey-note">
          Hold a modifier and press a key. Escape cancels.
        </p>
      )}
    </div>
  );
}

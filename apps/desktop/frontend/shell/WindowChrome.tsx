// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The window has no native title bar. A transparent strip along the top of the
 * content area is the drag handle, and the three controls sit in the corner.
 * Content keeps clear of the strip (main is padded by --shell-top), so the drag
 * region never swallows a click meant for a page.
 */
import { Maximize2, Minus, X } from "../lib/icons";

export function WindowChrome() {
  return (
    <>
      <div className="shell-drag" aria-hidden="true" />
      <div className="shell-win">
        <button type="button" aria-label="Minimize Recall" title="Minimize" onClick={() => window.electronAPI?.minimize?.()}>
          <Minus aria-hidden="true" />
        </button>
        <button type="button" aria-label="Maximize or restore Recall" title="Maximize or restore" onClick={() => window.electronAPI?.maximize?.()}>
          <Maximize2 aria-hidden="true" />
        </button>
        <button type="button" className="is-close" aria-label="Close Recall" title="Close" onClick={() => window.electronAPI?.close?.()}>
          <X aria-hidden="true" />
        </button>
      </div>
    </>
  );
}

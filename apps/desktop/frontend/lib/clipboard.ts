// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
const CLIPBOARD_TIMEOUT_MS = 1500;

function withTimeout<T>(promise: Promise<T>, timeoutMs = CLIPBOARD_TIMEOUT_MS): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_resolve, reject) => {
      window.setTimeout(() => reject(new Error("clipboard timeout")), timeoutMs);
    }),
  ]);
}

/**
 * Copy creator-facing text reliably in both the packaged Electron app and a
 * regular browser. Electron owns the system clipboard path because renderer
 * clipboard permissions are intentionally denied by the desktop shell.
 */
export async function writeClipboardText(text: string): Promise<void> {
  if (!text.trim()) throw new Error("nothing to copy");

  const desktopWriter = window.electronAPI?.writeClipboardText;
  if (desktopWriter) {
    try {
      const result = await withTimeout(desktopWriter(text));
      if (result.success) return;
    } catch {
      // Fall through so browser/dev builds still have two recovery paths.
    }
  }

  if (navigator.clipboard?.writeText) {
    try {
      await withTimeout(navigator.clipboard.writeText(text));
      return;
    } catch {
      // Permission-gated browsers may still support the selection fallback.
    }
  }

  const scratch = document.createElement("textarea");
  scratch.value = text;
  scratch.setAttribute("readonly", "");
  scratch.style.position = "fixed";
  scratch.style.left = "-9999px";
  scratch.style.opacity = "0";
  document.body.appendChild(scratch);
  scratch.focus();
  scratch.select();
  const copied = typeof document.execCommand === "function" && document.execCommand("copy");
  scratch.remove();
  if (!copied) throw new Error("clipboard rejected the copy request");
}

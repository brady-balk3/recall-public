// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
export {};

export type RememberHotkeyState = {
  accelerator: string | null;
  registered: boolean;
  source: "environment" | "setting" | "default";
  error: string | null;
};

declare global {
  interface Window {
    electronAPI?: {
      getApiConfig: () => { endpoint: string; token: string; rememberHotkey: string | null };
      openFileDialog: () => Promise<string | null>;
      openFolderDialog: () => Promise<string | null>;
      writeClipboardText: (text: string) => Promise<{ success: boolean; error?: string }>;
      openExternal: (url: string) => Promise<{ success: boolean; error?: string }>;
      openPath: (path: string) => Promise<{ success: boolean; error?: string }>;
      setWorkProtection: (active: boolean) => Promise<boolean>;
      showNotification: (payload: { title: string; body: string }) => Promise<boolean>;
      getDiagnosticsInfo: () => Promise<{ logFile: string; logDir: string } | null>;
      openDevTools: () => Promise<boolean>;
      getRememberHotkey: () => Promise<import("./electron").RememberHotkeyState | null>;
      setRememberHotkey: (
        accelerator: string | null,
      ) => Promise<import("./electron").RememberHotkeyState | null>;
      onRememberHotkeyState: (
        callback: (state: import("./electron").RememberHotkeyState) => void,
      ) => () => void;
      onRememberResult: (callback: (result: unknown) => void) => () => void;
      minimize: () => void;
      maximize: () => void;
      close: () => void;
      setTitleBarOverlay: (opts: { color: string; symbolColor: string; height?: number }) => void;
    };
  }
}

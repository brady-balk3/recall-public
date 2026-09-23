// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  getApiConfig: () => ipcRenderer.sendSync("app:getApiConfig"),
  openFileDialog: () => ipcRenderer.invoke("dialog:openFile"),
  openFolderDialog: () => ipcRenderer.invoke("dialog:openFolder"),
  writeClipboardText: (text: string) => ipcRenderer.invoke("clipboard:writeText", text),
  openExternal: (url: string) => ipcRenderer.invoke("shell:openExternal", url),
  openPath: (path: string) => ipcRenderer.invoke("shell:openPath", path),
  setWorkProtection: (active: boolean) => ipcRenderer.invoke("app:setWorkProtection", active),
  showNotification: (payload: { title: string; body: string }) =>
    ipcRenderer.invoke("app:showNotification", payload),
  getDiagnosticsInfo: () => ipcRenderer.invoke("diagnostics:info"),
  openDevTools: () => ipcRenderer.invoke("diagnostics:openDevTools"),
  getRememberHotkey: () => ipcRenderer.invoke("hotkey:get"),
  setRememberHotkey: (accelerator: string | null) =>
    ipcRenderer.invoke("hotkey:set", accelerator),
  onRememberHotkeyState: (callback: (state: unknown) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, state: unknown) => callback(state);
    ipcRenderer.on("recall:hotkey-state", listener);
    return () => ipcRenderer.removeListener("recall:hotkey-state", listener);
  },
  onRememberResult: (callback: (result: unknown) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, result: unknown) => callback(result);
    ipcRenderer.on("recall:remember-result", listener);
    return () => ipcRenderer.removeListener("recall:remember-result", listener);
  },
  minimize: () => ipcRenderer.send("window:minimize"),
  maximize: () => ipcRenderer.send("window:maximize"),
  close: () => ipcRenderer.send("window:close"),
  setTitleBarOverlay: (opts: { color: string; symbolColor: string; height?: number }) =>
    ipcRenderer.send("window:setTitleBarOverlay", opts),
});

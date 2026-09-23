// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { installedStorageMode, runtimeStorageEnvironment } from "./runtimeStorage";
import {
  app,
  BrowserWindow,
  clipboard,
  ipcMain,
  dialog,
  globalShortcut,
  Notification,
  powerSaveBlocker,
  shell,
  type IpcMainEvent,
  type IpcMainInvokeEvent,
} from "electron";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawn, ChildProcess } from "node:child_process";
import fs from "node:fs";
import { randomBytes } from "node:crypto";
import { createServer } from "node:net";
import { get as httpGet, request as httpRequest } from "node:http";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// The built directory structure
//
// ├─┬ dist-electron
// │ ├─┬ main.js
// │ └─┬ preload.mjs
// ├─┬ dist
// │ └── index.html
//
process.env.APP_ROOT = path.join(__dirname, "..");

export const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
export const MAIN_DIST = path.join(process.env.APP_ROOT, "dist-electron");
export const RENDERER_DIST = path.join(process.env.APP_ROOT, "dist");

export const VITE_PUBLIC = VITE_DEV_SERVER_URL
  ? path.join(process.env.APP_ROOT, "public")
  : RENDERER_DIST;
process.env.VITE_PUBLIC = VITE_PUBLIC;

let win: BrowserWindow | null = null;
let splash: BrowserWindow | null = null;
let apiProcess: ChildProcess | null = null;
let workProtectionId: number | null = null;
const DEFAULT_REMEMBER_HOTKEY = "CommandOrControl+Alt+R";
const rememberHotkeyOverride = process.env.RECALL_REMEMBER_HOTKEY?.trim();
// A hotkey another app already owns registers as `false`, not as a throw. That
// failure has to reach the creator: an unnoticed dead key costs a whole stream
// of markers, and the cost only shows up after the moments are gone.
type RememberHotkeyState = {
  accelerator: string | null;
  registered: boolean;
  source: "environment" | "setting" | "default";
  error: string | null;
};
let rememberHotkey: RememberHotkeyState = {
  accelerator: null, registered: false, source: "default", error: null,
};
// Held at module scope so re-registering after a settings change rebinds the
// callback without threading the loopback config back through every caller.
let rememberApiPort = 0;
let rememberApiToken = "";
// Guarantees the splash reads as intentional rather than a flicker when the
// backend happens to come up fast. The main window still loads in parallel.
const SPLASH_MIN_MS = 900;
// A renderer load can fail before Chromium emits ready-to-show. Never leave
// the always-on-top splash covering an otherwise recoverable app indefinitely.
const MAIN_WINDOW_REVEAL_TIMEOUT_MS = 15_000;
// Give antivirus/native-library cold starts room to settle, but do not leave
// the splash looking hung indefinitely. Process exits fail this wait instantly.
const BACKEND_READY_TIMEOUT_MS = 60_000;
let splashShownAt = 0;
const hasSingleInstanceLock = app.requestSingleInstanceLock();

// --- Diagnostics -----------------------------------------------------------
//
// There is no menu, so no Ctrl+Shift+I, so a renderer crash has until now been
// invisible: the only trace was the API logging a dropped connection, which
// reads like a server fault and is not one. Everything interesting goes to one
// lifecycle file and to stdout when a terminal is attached. Raw errors stay
// out of this file: they can contain media URLs, tokens and local paths.

const LOG_MAX_BYTES = 5 * 1024 * 1024;

function logFile(): string {
  const dir = path.join(app.getPath("userData"), "logs");
  try {
    fs.mkdirSync(dir, { recursive: true });
  } catch {
    // Nothing to do: logging must never be the reason the app fails to start.
  }
  // Keep legacy raw logs separate; do not append sanitized events to them.
  return path.join(dir, "recall-lifecycle.log");
}

function diag(scope: string, message: string): void {
  const line = `${new Date().toISOString()} [${scope}] ${message}\n`;
  process.stdout.write(line);
  try {
    const file = logFile();
    // Keep one previous file rather than growing without bound. A crash that
    // only reproduces after an hour of streaming must not be truncated away.
    try {
      if (fs.statSync(file).size > LOG_MAX_BYTES) {
        fs.renameSync(file, `${file}.1`);
      }
    } catch { /* first run, or the file vanished; either way just append */ }
    fs.appendFileSync(file, line, "utf-8");
  } catch { /* disk full or permissions: never throw from the logger */ }
}

/** Report what the renderer does when it dies, rather than leaving the API's
 *  dropped-connection message as the only evidence. */
function watchRenderer(target: BrowserWindow): void {
  target.webContents.on("render-process-gone", (_event, details) => {
    diag("renderer", `GONE reason=${details.reason} exitCode=${details.exitCode}`);
  });
  target.webContents.on("unresponsive", () => diag("renderer", "unresponsive"));
  target.webContents.on("responsive", () => diag("renderer", "responsive again"));
  target.webContents.on("preload-error", () => {
    diag("renderer", "preload failed");
  });
  target.webContents.on("console-message", (_event, level) => {
    // level 3 is error. Warnings and logs are noise for a crash report.
    if (level >= 3) diag("renderer", "console error reported");
  });
  target.webContents.on("did-fail-load", (_e, code) => {
    diag("renderer", `did-fail-load ${code}`);
  });
}

/** F12 and Ctrl+Shift+I open DevTools.
 *
 *  The window uses a custom title bar (`titleBarStyle: "hidden"`), so adding an
 *  Electron application menu just to get the default accelerator would put a
 *  menu bar back on screen. Intercepting the keystroke keeps the chrome intact.
 *
 *  Left enabled in packaged builds on purpose: Recall runs on the creator's own
 *  machine, and "press F12 and send me what it says" is the difference between
 *  a support conversation and a shrug.
 */
function enableDevToolsShortcut(target: BrowserWindow): void {
  target.webContents.on("before-input-event", (event, input) => {
    if (input.type !== "keyDown") return;
    const isF12 = input.key === "F12";
    const isInspect = (input.control || input.meta) && input.shift
      && input.key.toLowerCase() === "i";
    if (!isF12 && !isInspect) return;
    event.preventDefault();
    const contents = target.webContents;
    if (contents.isDevToolsOpened()) {
      contents.closeDevTools();
    } else {
      contents.openDevTools({ mode: "detach" });
      diag("renderer", "devtools opened by keyboard shortcut");
    }
  });
}

function rememberHotkeyFile(): string {
  return path.join(app.getPath("userData"), "remember-hotkey.json");
}

function readStoredHotkey(): string | null | undefined {
  try {
    const raw = fs.readFileSync(rememberHotkeyFile(), "utf-8");
    const parsed = JSON.parse(raw) as { accelerator?: unknown };
    if (parsed.accelerator === null) return null;       // deliberately disabled
    if (typeof parsed.accelerator === "string" && parsed.accelerator.trim()) {
      return parsed.accelerator.trim();
    }
  } catch {
    // No stored choice yet, or the file is unreadable. Fall back to the default
    // rather than leaving the creator with no hotkey at all.
  }
  return undefined;
}

function writeStoredHotkey(accelerator: string | null): void {
  fs.writeFileSync(
    rememberHotkeyFile(),
    JSON.stringify({ accelerator }, null, 1),
    "utf-8",
  );
}

/** A bare key would swallow normal typing everywhere; require a modifier. */
function isPlausibleAccelerator(value: string): boolean {
  if (!/^[A-Za-z0-9+]+$/.test(value.replace(/\s/g, ""))) return false;
  const parts = value.split("+").map((part) => part.trim()).filter(Boolean);
  if (parts.length < 2) return false;
  const modifiers = new Set([
    "command", "cmd", "control", "ctrl", "commandorcontrol", "cmdorctrl",
    "alt", "option", "altgr", "shift", "super", "meta",
  ]);
  const hasModifier = parts.some((part) => modifiers.has(part.toLowerCase()));
  const hasKey = parts.some((part) => !modifiers.has(part.toLowerCase()));
  return hasModifier && hasKey;
}

function broadcastHotkeyState(): void {
  if (win && !win.isDestroyed()) {
    win.webContents.send("recall:hotkey-state", rememberHotkey);
  }
}

function applyRememberHotkey(
  accelerator: string | null,
  source: RememberHotkeyState["source"],
): RememberHotkeyState {
  globalShortcut.unregisterAll();
  if (!accelerator) {
    rememberHotkey = { accelerator: null, registered: false, source, error: null };
    broadcastHotkeyState();
    return rememberHotkey;
  }
  let registered = false;
  let error: string | null = null;
  try {
    registered = globalShortcut.register(
      accelerator,
      () => { void captureRememberMoment(rememberApiPort, rememberApiToken); },
    );
    if (!registered) {
      error = "Another app is already using this shortcut. Pick a different one.";
    }
  } catch (exc) {
    error = `${accelerator} is not a shortcut Recall can register.`;
  }
  rememberHotkey = { accelerator, registered, source, error };
  if (!registered) {
    console.warn(`Could not register Recall Remember hotkey: ${accelerator}`);
  }
  broadcastHotkeyState();
  return rememberHotkey;
}

function initialRememberHotkey(): { accelerator: string | null; source: RememberHotkeyState["source"] } {
  if (rememberHotkeyOverride) {
    return rememberHotkeyOverride.toLowerCase() === "disabled"
      ? { accelerator: null, source: "environment" }
      : { accelerator: rememberHotkeyOverride, source: "environment" };
  }
  const stored = readStoredHotkey();
  if (stored !== undefined) return { accelerator: stored, source: "setting" };
  return { accelerator: DEFAULT_REMEMBER_HOTKEY, source: "default" };
}

function setWorkProtection(active: boolean): boolean {
  if (active) {
    if (workProtectionId == null || !powerSaveBlocker.isStarted(workProtectionId)) {
      // Keep long scans/exports alive while still letting the display turn off.
      workProtectionId = powerSaveBlocker.start("prevent-app-suspension");
    }
    return workProtectionId != null && powerSaveBlocker.isStarted(workProtectionId);
  }
  if (workProtectionId != null) {
    powerSaveBlocker.stop(workProtectionId);
    workProtectionId = null;
  }
  return false;
}

function showWorkNotification(payload: unknown): boolean {
  if (!Notification.isSupported() || win?.isFocused()) return false;
  if (!payload || typeof payload !== "object") return false;
  const candidate = payload as { title?: unknown; body?: unknown };
  if (typeof candidate.title !== "string" || typeof candidate.body !== "string") return false;
  const title = candidate.title.trim().slice(0, 96);
  const body = candidate.body.trim().slice(0, 240);
  if (!title || !body) return false;
  const notification = new Notification({ title, body });
  notification.on("click", () => {
    if (!win || win.isDestroyed()) return;
    if (win.isMinimized()) win.restore();
    win.show();
    win.focus();
  });
  notification.show();
  return true;
}

function postRecallJson(port: number, token: string, pathname: string): Promise<{
  ok: boolean;
  status: number;
  payload: unknown;
}> {
  return new Promise((resolve) => {
    const request = httpRequest({
      hostname: "127.0.0.1",
      port,
      path: pathname,
      method: "POST",
      headers: {
        "content-type": "application/json",
        "content-length": "0",
        "x-recall-token": token,
      },
      timeout: 3_000,
    }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
      response.on("end", () => {
        let payload: unknown = null;
        try {
          payload = JSON.parse(Buffer.concat(chunks).toString("utf8") || "null");
        } catch {
          payload = null;
        }
        const status = response.statusCode || 0;
        resolve({ ok: status >= 200 && status < 300, status, payload });
      });
    });
    request.once("timeout", () => request.destroy(new Error("Recall marker request timed out.")));
    request.once("error", (error) => resolve({
      ok: false,
      status: 0,
      payload: { detail: error.message },
    }));
    request.end();
  });
}

async function captureRememberMoment(port: number, token: string): Promise<void> {
  const result = await postRecallJson(port, token, "/recall-sessions/active/remember");
  if (win && !win.isDestroyed()) {
    win.webContents.send("recall:remember-result", result);
  }
  if (result.ok) {
    showWorkNotification({
      title: "Moment remembered",
      body: "Recall saved this point for the post-stream Deep Recall scan.",
    });
    return;
  }
  const detail = result.payload && typeof result.payload === "object"
    && "detail" in result.payload
    ? String((result.payload as { detail?: unknown }).detail || "")
    : "";
  showWorkNotification({
    title: "Moment not saved",
    body: detail || "Start a Recall Session before using the Remember hotkey.",
  });
}

if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!win) return;
    if (win.isMinimized()) win.restore();
    win.show();
    win.focus();
  });
}

function reserveLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.unref();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        reject(new Error("Could not reserve a loopback port for the Recall engine."));
        return;
      }
      const port = address.port;
      server.close((error) => error ? reject(error) : resolve(port));
    });
  });
}

type IpcSenderEvent = IpcMainEvent | IpcMainInvokeEvent;

function isTrustedRendererUrl(senderUrl: string): boolean {
  try {
    const candidate = new URL(senderUrl);
    if (VITE_DEV_SERVER_URL) {
      return candidate.origin === new URL(VITE_DEV_SERVER_URL).origin;
    }

    const expected = pathToFileURL(path.join(RENDERER_DIST, "index.html"));
    candidate.hash = "";
    candidate.search = "";
    return candidate.href === expected.href;
  } catch {
    return false;
  }
}

function isTrustedIpcSender(event: IpcSenderEvent, channel: string): boolean {
  const senderUrl = event.senderFrame?.url || event.sender.getURL();
  if (isTrustedRendererUrl(senderUrl)) return true;
  console.warn(`Blocked ${channel} IPC from an untrusted renderer`);
  return false;
}

function waitForBackend(
  port: number,
  token: string,
  timeoutMs = BACKEND_READY_TIMEOUT_MS,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  const url = `http://127.0.0.1:${port}/health?token=${encodeURIComponent(token)}`;
  return new Promise((resolve) => {
    const backend = apiProcess;
    let settled = false;
    const finish = (ready: boolean) => {
      if (settled) return;
      settled = true;
      backend?.removeListener("exit", onBackendExit);
      resolve(ready);
    };
    const onBackendExit = (code: number | null, signal: NodeJS.Signals | null) => {
      console.error(`Recall API exited before becoming ready (code=${code}, signal=${signal}).`);
      finish(false);
    };
    backend?.once("exit", onBackendExit);

    const probe = () => {
      if (settled) return;
      const request = httpGet(url, (response) => {
        response.resume();
        if (response.statusCode === 200) finish(true);
        else if (Date.now() >= deadline) finish(false);
        else setTimeout(probe, 150);
      });
      request.once("error", () => {
        if (Date.now() >= deadline) finish(false);
        else setTimeout(probe, 150);
      });
      request.setTimeout(1_000, () => request.destroy());
    };
    probe();
  });
}

function getPythonExecutable(apiPath?: string): string {
  const configuredPython = process.env.RECALL_PYTHON;
  if (configuredPython && fs.existsSync(configuredPython)) {
    return configuredPython;
  }
  // The repo venv carries Recall's native ML dependencies (including
  // llama-cpp-python). Prefer it for a source/dev backend so Electron cannot
  // silently launch an otherwise valid system Python without the semantic
  // judge. RECALL_PYTHON remains the explicit override above.
  if (apiPath) {
    const projectVenv = path.resolve(
      path.dirname(apiPath), "..", "..", "venv", "Scripts", "python.exe",
    );
    if (fs.existsSync(projectVenv)) {
      return projectVenv;
    }
  }
  // Fall back to known system interpreters for source trees without a venv.
  if (fs.existsSync("C:\\Python314\\python.exe")) {
    return "C:\\Python314\\python.exe";
  }
  const localAppData = process.env.LOCALAPPDATA || "";
  if (localAppData) {
    const py312 = path.join(localAppData, "Programs", "Python", "Python312", "python.exe");
    if (fs.existsSync(py312)) {
      return py312;
    }
  }
  // Final fallback may resolve through the Windows execution alias.
  return "python";
}

const configuredUserData = process.env.RECALL_USER_DATA_DIR;
if (configuredUserData) {
  app.setPath("userData", configuredUserData);
}

function backendEnvironment(port: number, token: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    RECALL_API_PORT: String(port),
    RECALL_API_TOKEN: token,
    // The batch launcher owns this process lifecycle. A nested Uvicorn reload
    // supervisor makes failures opaque and complicates reliable shutdown; a
    // developer who runs main.py directly still gets source hot reload.
    RECALL_API_RELOAD: "0",
    PYTHONUNBUFFERED: "1",
  };
  // This flag is an eval-only guardrail. A developer shell used for golden-set
  // measurement may carry it for days; inheriting it into the product silently
  // records creator feedback without ever activating the personal ranker.
  delete env.RECALL_FREEZE_PERSONALIZATION;
  const markerPath = path.join(process.resourcesPath, "recall-installation.json");
  const installed = app.isPackaged && fs.existsSync(markerPath)
    ? installedStorageMode(JSON.parse(fs.readFileSync(markerPath, "utf8")))
    : false;
  return runtimeStorageEnvironment(env, app.getPath("userData"), installed);
}

function startBundledBackend(port: number, token: string): boolean {
  // electron-builder's extraResources land under resourcesPath at runtime
  // (see apps/desktop/package.json build.extraResources).
  const exePath = path.join(process.resourcesPath, "recall-engine", "recall-engine.exe");
  if (!fs.existsSync(exePath)) {
    console.error("Bundled recall-engine.exe not found at:", exePath);
    return false;
  }

  console.log("Starting bundled Recall engine at:", exePath);
  try {
    // NOT detached. `detached` put the engine in its own process group so it
    // outlived the app, and taskkill on the tracked pid then reported "process
    // not found" while the real one kept running. On Windows a detached
    // console-subsystem child also gets its own console window, which is the
    // black box users were left staring at.
    apiProcess = spawn(exePath, [], {
      stdio: "ignore",
      windowsHide: true,
      env: backendEnvironment(port, token),
    });

    apiProcess.on("error", () => {
      diag("engine", "failed to spawn bundled engine");
    });
    apiProcess.on("exit", (code, signal) => {
      diag("engine", `bundled engine exited code=${code} signal=${signal}`);
    });
    return true;
  } catch (err) {
    console.error("Exception thrown when spawning bundled Recall engine:", err);
    return false;
  }
}

function startDevBackend(port: number, token: string) {
  const candidates = [
    // Relative to the built main.js (apps/desktop/dist-electron) — works
    // regardless of the process's working directory.
    path.join(__dirname, "..", "..", "api", "main.py"),
    // If running in development (npm run electron:dev)
    path.join(process.cwd(), "..", "api", "main.py"),
    // If running from packaged release folder (apps/desktop/release)
    path.join(process.cwd(), "..", "..", "..", "apps", "api", "main.py"),
  ];

  let apiPath = "";
  for (const c of candidates) {
    const resolved = path.resolve(c);
    if (fs.existsSync(resolved)) {
      apiPath = resolved;
      break;
    }
  }

  if (!apiPath) {
    console.error("Could not find Recall api/main.py server file.");
    return;
  }

  const pythonExec = getPythonExecutable(apiPath);
  console.log(`Starting Python Recall API server using [${pythonExec}] at:`, apiPath);

  try {
    // See the bundled path: `detached` is what leaked the process and spawned
    // the stray console window.
    apiProcess = spawn(pythonExec, [apiPath], {
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
      env: backendEnvironment(port, token),
    });

    apiProcess.stdout?.on("data", (chunk) => process.stdout.write(`[Recall API] ${chunk}`));
    apiProcess.stderr?.on("data", (chunk) => process.stderr.write(`[Recall API] ${chunk}`));

    apiProcess.on("error", () => {
      diag("engine", "failed to spawn dev API");
    });
    apiProcess.on("exit", (code, signal) => {
      diag("engine", `dev API exited code=${code} signal=${signal}`);
    });
  } catch (err) {
    console.error("Exception thrown when spawning Python API server:", err);
  }
}

function startBackend(port: number, token: string) {
  // Packaged app: use the self-contained PyInstaller-bundled engine, no
  // system Python required at all. Dev mode: fall back to shelling out to a
  // system Python running main.py from source, as before.
  if (app.isPackaged) {
    if (!startBundledBackend(port, token)) {
      console.error("Falling back to system Python — bundled engine failed to start.");
      startDevBackend(port, token);
    }
  } else {
    startDevBackend(port, token);
  }

  // Handle quitting. `before-quit` rather than `will-quit`, so the kill is
  // issued while the process tree is still intact, and the handle is captured
  // because `apiProcess` can be reassigned or nulled before the callback runs.
  app.on("before-quit", () => stopBackend());
  app.on("will-quit", () => stopBackend());
}

let backendStopped = false;

/** Kill the engine and everything it spawned. Safe to call more than once. */
function stopBackend(): void {
  const backend = apiProcess;
  if (!backend || backendStopped) return;
  backendStopped = true;
  const pid = backend.pid;
  diag("engine", `stopping backend pid=${pid}`);
  try {
    if (process.platform === "win32" && pid) {
      // /t takes the children too: uvicorn's workers and any ffmpeg it started
      // are the processes that actually held the console window open.
      spawn("taskkill", ["/pid", String(pid), "/f", "/t"], { windowsHide: true });
    } else {
      backend.kill();
    }
  } catch {
    diag("engine", "failed to stop backend");
  }
  apiProcess = null;
}

function createSplash() {
  splash = new BrowserWindow({
    width: 440,
    height: 320,
    frame: false,
    resizable: false,
    movable: true,
    center: true,
    show: false,
    transparent: true,
    backgroundColor: "#00000000",
    alwaysOnTop: true,
    skipTaskbar: true,
    hasShadow: true,
    title: "Recall",
    webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false },
  });
  splash.setMenu(null);
  // Both VITE_PUBLIC targets (public/ in dev, dist/ when packaged) carry
  // splash.html, the logo mark, and the bundled fonts as siblings.
  void splash.loadFile(path.join(VITE_PUBLIC, "splash.html")).catch((error) => {
    console.error("Recall splash failed to load:", error);
  });
  splash.once("ready-to-show", () => {
    splash?.show();
    splashShownAt = Date.now();
  });
}

function dismissSplash() {
  if (!splash || splash.isDestroyed()) return;
  const elapsed = Date.now() - splashShownAt;
  const wait = Math.max(0, SPLASH_MIN_MS - elapsed);
  setTimeout(() => {
    if (splash && !splash.isDestroyed()) {
      splash.close();
      splash = null;
    }
  }, wait);
}

function createWindow() {
  const iconPath = VITE_DEV_SERVER_URL
    ? path.join(VITE_PUBLIC, "icon.ico")
    : path.join(RENDERER_DIST, "icon.ico");

  win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 1024,
    minHeight: 700,
    title: "Recall",
    icon: iconPath,
    backgroundColor: "#131017",
    show: false,
    titleBarStyle: "hidden",
    titleBarOverlay: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.mjs"),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // A renderer crash is otherwise invisible: no menu means no DevTools.
  watchRenderer(win);
  enableDevToolsShortcut(win);

  win.setMenu(null);
  const createdWindow = win;
  let revealTimer: ReturnType<typeof setTimeout> | null = null;
  const revealWindow = () => {
    if (revealTimer) {
      clearTimeout(revealTimer);
      revealTimer = null;
    }
    if (createdWindow.isDestroyed()) return;
    createdWindow.show();
    createdWindow.focus();
    dismissSplash();
  };
  // Reveal only once the renderer has painted its first frame, so the splash
  // hands off to a ready UI instead of a flash of empty dark window.
  win.once("ready-to-show", () => {
    revealWindow();
  });
  win.webContents.on(
    "did-fail-load",
    (_event, errorCode, _errorDescription, _validatedURL, isMainFrame) => {
      if (!isMainFrame) return;
      console.error(
        `Recall renderer failed to load (code=${errorCode})`,
      );
      revealWindow();
    },
  );
  revealTimer = setTimeout(() => {
    console.error(
      `Recall renderer did not signal ready-to-show within ${MAIN_WINDOW_REVEAL_TIMEOUT_MS}ms; `
      + "revealing the main window and dismissing the splash.",
    );
    revealWindow();
  }, MAIN_WINDOW_REVEAL_TIMEOUT_MS);
  win.once("closed", () => {
    if (revealTimer) clearTimeout(revealTimer);
    revealTimer = null;
  });
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (event, url) => {
    if (!VITE_DEV_SERVER_URL) {
      event.preventDefault();
      return;
    }
    try {
      if (new URL(url).origin !== new URL(VITE_DEV_SERVER_URL).origin) event.preventDefault();
    } catch {
      event.preventDefault();
    }
  });
  win.webContents.session.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(RENDERER_DIST, "index.html"));
  }
}

// Quit when all windows are closed, except on macOS.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
    win = null;
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.whenReady().then(async () => {
  if (!hasSingleInstanceLock) return;
  app.setAppUserModelId("com.recall.app");
  // Instant visual feedback: the splash paints while the engine boots and the
  // renderer loads, replacing the old dead black window during startup.
  createSplash();
  const apiPort = await reserveLoopbackPort();
  const apiToken = randomBytes(32).toString("base64url");
  process.env.RECALL_API_PORT = String(apiPort);
  process.env.RECALL_API_TOKEN = apiToken;
  rememberApiPort = apiPort;
  rememberApiToken = apiToken;
  startBackend(apiPort, apiToken);
  ipcMain.on("app:getApiConfig", (event) => {
    if (!isTrustedIpcSender(event, "app:getApiConfig")) {
      event.returnValue = null;
      return;
    }
    event.returnValue = {
      endpoint: `http://127.0.0.1:${apiPort}`,
      token: apiToken,
      rememberHotkey: rememberHotkey.accelerator,
    };
  });
  ipcMain.handle("diagnostics:info", async (event) => {
    if (!isTrustedIpcSender(event, "diagnostics:info")) return null;
    return { logFile: logFile(), logDir: path.dirname(logFile()) };
  });

  ipcMain.handle("diagnostics:openDevTools", async (event) => {
    if (!isTrustedIpcSender(event, "diagnostics:openDevTools")) return false;
    if (!win || win.isDestroyed()) return false;
    win.webContents.openDevTools({ mode: "detach" });
    diag("renderer", "devtools opened from settings");
    return true;
  });

  ipcMain.handle("hotkey:get", async (event) => {
    if (!isTrustedIpcSender(event, "hotkey:get")) return null;
    return rememberHotkey;
  });

  ipcMain.handle("hotkey:set", async (event, value: unknown) => {
    if (!isTrustedIpcSender(event, "hotkey:set")) return null;
    // The environment override is the packaging/rollback escape hatch. If it is
    // set, it stays authoritative -- a UI change must not silently defeat it.
    if (rememberHotkeyOverride) {
      return {
        ...rememberHotkey,
        error: "This shortcut is fixed by RECALL_REMEMBER_HOTKEY and cannot be changed here.",
      };
    }
    if (value === null) {
      writeStoredHotkey(null);
      return applyRememberHotkey(null, "setting");
    }
    if (typeof value !== "string" || !isPlausibleAccelerator(value)) {
      return {
        ...rememberHotkey,
        error: "Use at least one modifier key plus another key, like Ctrl + Alt + R.",
      };
    }
    const previous = rememberHotkey;
    const next = applyRememberHotkey(value, "setting");
    if (!next.registered) {
      // Leave the creator with a working shortcut rather than a broken one.
      applyRememberHotkey(previous.accelerator, previous.source);
      return { ...rememberHotkey, error: next.error };
    }
    writeStoredHotkey(value);
    return next;
  });

  ipcMain.handle("dialog:openFile", async (event) => {
    if (!isTrustedIpcSender(event, "dialog:openFile")) return null;
    const { canceled, filePaths } = await dialog.showOpenDialog({
      properties: ["openFile"],
      filters: [
        {
          name: "Video",
          extensions: ["mp4", "mkv", "mov", "avi", "webm", "ts", "m2ts", "flv", "wmv"],
        },
      ],
    });
    if (canceled) {
      return null;
    } else {
      return filePaths[0];
    }
  });

  ipcMain.handle("dialog:openFolder", async (event) => {
    if (!isTrustedIpcSender(event, "dialog:openFolder")) return null;
    const { canceled, filePaths } = await dialog.showOpenDialog({
      properties: ["openDirectory"],
    });
    return canceled ? null : filePaths[0];
  });

  ipcMain.handle("clipboard:writeText", async (event, value: unknown) => {
    if (!isTrustedIpcSender(event, "clipboard:writeText")) {
      return { success: false, error: "Untrusted IPC sender." };
    }
    try {
      if (typeof value !== "string" || !value.trim()) {
        return { success: false, error: "Nothing to copy." };
      }
      // Diagnostics are bounded server-side. Keep the renderer bridge bounded
      // too so compromised UI content cannot turn it into an unbounded IPC sink.
      if (Buffer.byteLength(value, "utf8") > 4 * 1024 * 1024) {
        return { success: false, error: "Clipboard text is too large." };
      }
      clipboard.writeText(value);
      return { success: true };
    } catch (err: any) {
      return { success: false, error: err?.message || "Clipboard write failed." };
    }
  });

  ipcMain.handle("shell:openExternal", async (event, url: string) => {
    if (!isTrustedIpcSender(event, "shell:openExternal")) {
      return { success: false, error: "Untrusted IPC sender." };
    }
    // Only ever hand real web/mail URLs to the OS. Clip titles/descriptions are
    // derived from VOD transcript/OCR text, so refusing file:/other schemes here
    // keeps that content from ever becoming a "shell.openExternal anything" sink.
    try {
      const parsed = new URL(url);
      if (!["http:", "https:", "mailto:"].includes(parsed.protocol)) {
        return { success: false, error: `Blocked URL scheme: ${parsed.protocol}` };
      }
      await shell.openExternal(url);
      return { success: true };
    } catch (err: any) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle("shell:openPath", async (event, requestedPath: unknown) => {
    if (!isTrustedIpcSender(event, "shell:openPath")) {
      return { success: false, error: "Untrusted IPC sender." };
    }
    try {
      if (typeof requestedPath !== "string" || !path.isAbsolute(requestedPath)) {
        return { success: false, error: "Only absolute directory paths can be opened." };
      }
      const resolvedPath = path.resolve(requestedPath);
      const stats = await fs.promises.stat(resolvedPath);
      if (!stats.isDirectory()) {
        return { success: false, error: "Only existing directories can be opened." };
      }
      const error = await shell.openPath(resolvedPath);
      return { success: !error, error: error || undefined };
    } catch (err: unknown) {
      return { success: false, error: err instanceof Error ? err.message : "Directory could not be opened." };
    }
  });

  ipcMain.handle("app:setWorkProtection", async (event, active: unknown) => {
    if (!isTrustedIpcSender(event, "app:setWorkProtection")) return false;
    if (typeof active !== "boolean") return false;
    return setWorkProtection(active);
  });

  ipcMain.handle("app:showNotification", async (event, payload: unknown) => {
    if (!isTrustedIpcSender(event, "app:showNotification")) return false;
    return showWorkNotification(payload);
  });

  ipcMain.on("window:minimize", (event) => {
    if (!isTrustedIpcSender(event, "window:minimize")) return;
    win?.minimize();
  });

  ipcMain.on("window:maximize", (event) => {
    if (!isTrustedIpcSender(event, "window:maximize")) return;
    if (win?.isMaximized()) {
      win.unmaximize();
    } else {
      win?.maximize();
    }
  });

  ipcMain.on("window:close", (event) => {
    if (!isTrustedIpcSender(event, "window:close")) return;
    win?.close();
  });

  ipcMain.on("window:setTitleBarOverlay", (event, opts: { color: string; symbolColor: string; height?: number }) => {
    if (!isTrustedIpcSender(event, "window:setTitleBarOverlay")) return;
    try {
      if (process.platform === "win32") win?.setTitleBarOverlay(opts);
    } catch {}
  });

  const ready = await waitForBackend(apiPort, apiToken);
  if (!ready) console.error("Recall engine did not become ready before the startup deadline.");
  createWindow();
  if (ready) {
    const initial = initialRememberHotkey();
    applyRememberHotkey(initial.accelerator, initial.source);
    if (rememberHotkey.accelerator && !rememberHotkey.registered) {
      // Surface it immediately. The renderer also shows this, but a creator who
      // starts a stream without opening the Live panel still needs to know.
      showWorkNotification({
        title: "Remember shortcut unavailable",
        body: `${rememberHotkey.accelerator} is already taken by another app. Pick a new one in Recall.`,
      });
    }
  }
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
  setWorkProtection(false);
});

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Dev only: `?opencut-stream=<url>` opens that video in the OpenCut editor as
 * a streamed source, with nothing else mounted, so its memory and disk reads
 * can be measured in isolation. Its server must also answer /waveform and
 * /filmstrip the way the engine does (scripts/opencut_stream/range_server.py). `window.__ocStream` exposes the editor core
 * and the last export for scripted runs. Never bundled in production (main.tsx
 * gates the import on import.meta.env.DEV).
 */
import { useEffect } from "react";
import { EditorCore } from "@opencut/core";
import { OpenCutEditor } from "../opencut/OpenCutEditor";

interface StreamHarnessState {
  core?: EditorCore;
  exported?: number;
}

declare global {
  interface Window { __ocStream?: StreamHarnessState }
}

const REF_PREFIX = "url:";

export function OpenCutStreamHarness({ url }: { url: string }) {
  useEffect(() => {
    const state: StreamHarnessState = {};
    window.__ocStream = state;
    // Scripted runs read the console: say where a stray rejection came from.
    const onRejection = (event: PromiseRejectionEvent) => {
      const reason = event.reason as { stack?: string } | undefined;
      console.error("[harness] unhandled rejection:", reason?.stack ?? String(event.reason));
    };
    window.addEventListener("unhandledrejection", onRejection);
    const timer = window.setInterval(() => {
      try { state.core = EditorCore.getInstance(); } catch { /* not ready */ }
      if (state.core) window.clearInterval(timer);
    }, 100);
    return () => { window.clearInterval(timer); window.removeEventListener("unhandledrejection", onRejection); };
  }, []);

  return (
    <OpenCutEditor
      // A fresh project per page load, so every run starts from the seed.
      sessionKey={`stream-test:${performance.timeOrigin}`}
      title="Streamed VOD"
      source={{ streamRef: `${REF_PREFIX}${url}`, name: url.split("/").pop() || "vod.mp4" }}
      stream={{
        source: (ref) => ref.slice(REF_PREFIX.length),
        // The test server mirrors the engine's /waveform and /filmstrip routes.
        waveform: () => new URL("/waveform", url).toString(),
        filmstrip: (_ref, r) => {
          const q = new URLSearchParams({ start: String(r.start), end: String(r.end), frames: String(r.frames), width: String(r.width), height: String(r.height) });
          return new URL(`/filmstrip?${q}`, url).toString();
        },
      }}
      canvas={{ width: 1920, height: 1080 }}
      onSave={async (mp4) => { window.__ocStream!.exported = mp4.size; }}
      onClose={() => {}}
    />
  );
}

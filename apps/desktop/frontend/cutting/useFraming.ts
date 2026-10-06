// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The Cutting Room's live framing: for each piece on the timeline (and the
 * stretch of reel under the playhead), the engine's layout as rectangles
 * (POST /jobs/{id}/clips/manual/framing -> layers), cached, handed to the
 * editor through the host's `framing`. The editor turns them into its 9:16
 * preview (opencut/src/recall/framing.ts), so what you see is what renders.
 *
 * Until the engine answers, a piece keeps the last framing its video had, or a
 * plain centred Gameplay window, so the canvas is never a letterboxed 16:9.
 */
import { useCallback, useMemo, useRef, useState } from "react";
import type { FramingLayer, FramingLookup } from "@opencut/recall/framing";
import type { RecallFramingHost } from "@opencut/recall/host";
import { apiFetch } from "../lib/api";
import type { ManualClipLayout } from "../lib/store";
import type { Box } from "../components/FacecamCropper";

export const CLIP_CANVAS = { width: 1080, height: 1920 };
/** A 9:16 window's share of a 16:9 frame's width. */
const GAMEPLAY_SHARE = (9 / 16) / (16 / 9);
/** Longest range the engine frames at once (it frames clips, up to 3 min). */
const MAX_RANGE = 170;
/** How much reel to frame around the playhead. */
const REEL_SPAN = 120;

export interface VideoFraming {
  layout: ManualClipLayout;
  facecam: Box | null;
  focusX: number;
}

export function gameplayLayers(focusX: number): FramingLayer[] {
  return [{
    role: "gameplay",
    source: [(1 - GAMEPLAY_SHARE) * Math.min(1, Math.max(0, focusX)), 0, GAMEPLAY_SHARE, 1],
    target: [0, 0, CLIP_CANVAS.width, CLIP_CANVAS.height],
  }];
}

const jobIdOf = (ref: string) => ref.replace(/^job:/, "");

export function useFraming(apiEndpoint: string, framingFor: (jobId: string) => VideoFraming): RecallFramingHost {
  const [version, setVersion] = useState(0);
  const cache = useRef(new Map<string, FramingLayer[]>());
  const pending = useRef(new Set<string>());
  const lastFor = useRef(new Map<string, FramingLayer[]>());

  const lookup = useCallback<FramingLookup>((ref, piece, playhead) => {
    const jobId = jobIdOf(ref);
    const framing = framingFor(jobId);
    let start = piece.reel ? Math.max(0, playhead - REEL_SPAN / 2) : piece.start;
    let end = piece.reel ? start + REEL_SPAN : piece.end;
    if (end - start > MAX_RANGE) {
      const middle = (start + end) / 2;
      start = middle - MAX_RANGE / 2;
      end = middle + MAX_RANGE / 2;
    }
    if (end - start < 1) end = start + 1;
    const key = [
      jobId, start.toFixed(1), end.toFixed(1), framing.layout, framing.focusX.toFixed(3),
      framing.facecam?.map((value) => value.toFixed(4)).join(",") ?? "",
    ].join("|");
    const hit = cache.current.get(key);
    if (hit) {
      lastFor.current.set(jobId, hit);
      return hit;
    }
    if (!pending.current.has(key)) {
      pending.current.add(key);
      void apiFetch(`/jobs/${encodeURIComponent(jobId)}/clips/manual/framing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start,
          end,
          layout: framing.layout,
          focus_x: framing.focusX,
          facecam_override: framing.layout === "full_gameplay" ? null : framing.facecam,
        }),
      }, apiEndpoint)
        .then((response) => (response.ok ? response.json() : null))
        .then((body: { layers?: FramingLayer[] } | null) => {
          const layers = body?.layers?.length ? body.layers : gameplayLayers(framing.focusX);
          cache.current.set(key, layers);
          setVersion((value) => value + 1);
        })
        .catch(() => {
          cache.current.set(key, gameplayLayers(framing.focusX));
          setVersion((value) => value + 1);
        })
        .finally(() => pending.current.delete(key));
    }
    // Gameplay needs no engine to look right; anything else keeps its last
    // framing until the new one arrives, so switching doesn't flash.
    if (framing.layout === "full_gameplay") return gameplayLayers(framing.focusX);
    return lastFor.current.get(jobId) ?? gameplayLayers(framing.focusX);
  }, [apiEndpoint, framingFor]);

  return useMemo(() => ({ canvas: CLIP_CANVAS, lookup, version }), [lookup, version]);
}

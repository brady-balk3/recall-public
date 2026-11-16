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
import type { RecallCutoutHost, RecallFramingHost } from "@opencut/recall/host";
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

/** Everything about a video's framing that changes what the engine answers. */
const framingLook = (framing: VideoFraming) => [
  framing.layout, framing.focusX.toFixed(3), framing.facecam?.map((value) => value.toFixed(4)).join(",") ?? "",
].join("|");

const jobIdOf = (ref: string) => ref.replace(/^job:/, "");

/** The editor's framing, plus the facecam box the engine framed with (for the full-frame pane). */
export interface FramingState extends RecallFramingHost {
  /** The facecam box last framed for this video (drawn or detected), or null. */
  facecamFor: (jobId: string) => Box | null;
  /** The VTuber cutout last framed for this video, if its layout is one. */
  cutoutFor: (jobId: string) => RecallCutoutHost | null;
}

interface CutoutInfo { mask: string; lowerFade: number; source: FramingLayer["source"]; target: FramingLayer["target"] }

export function useFraming(apiEndpoint: string, framingFor: (jobId: string) => VideoFraming): FramingState {
  const [version, setVersion] = useState(0);
  const cache = useRef(new Map<string, FramingLayer[]>());
  const pending = useRef(new Set<string>());
  const lastFor = useRef(new Map<string, FramingLayer[]>());
  const boxes = useRef(new Map<string, Box | null>());
  const lastBox = useRef(new Map<string, Box | null>());
  const cutouts = useRef(new Map<string, CutoutInfo | null>());
  const lastCutout = useRef(new Map<string, CutoutInfo | null>());
  const masks = useRef(new Map<string, Promise<ImageBitmap>>());
  // The latest framing, for answers that land after it has changed.
  const currentFraming = useRef(framingFor);
  currentFraming.current = framingFor;

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
    const look = framingLook(framing);
    const key = [jobId, start.toFixed(1), end.toFixed(1), look].join("|");
    const hit = cache.current.get(key);
    if (hit) {
      lastFor.current.set(jobId, hit);
      lastBox.current.set(jobId, boxes.current.get(key) ?? null);
      lastCutout.current.set(jobId, cutouts.current.get(key) ?? null);
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
        .then((body: { layers?: FramingLayer[]; facecam?: number[] | null; cutout?: { mask: string; lower_fade: number } } | null) => {
          const layers = body?.layers?.length ? body.layers : gameplayLayers(framing.focusX);
          const box = body?.facecam?.length === 4 ? body.facecam.map(Number) as Box : null;
          boxes.current.set(key, box);
          const avatar = layers.find((layer) => layer.role === "avatar");
          const cutout = body?.cutout && avatar
            ? { mask: body.cutout.mask, lowerFade: Number(body.cutout.lower_fade) || 0, source: avatar.source, target: avatar.target }
            : null;
          cutouts.current.set(key, cutout);
          // Still the framing this video has: the pane and the cutout show it
          // now, not on the next lookup (which runs after this render).
          if (framingLook(currentFraming.current(jobId)) === look) {
            lastBox.current.set(jobId, box);
            lastCutout.current.set(jobId, cutout);
          }
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
    if (framing.layout === "full_gameplay") {
      lastCutout.current.set(jobId, null);
      return gameplayLayers(framing.focusX);
    }
    return lastFor.current.get(jobId) ?? gameplayLayers(framing.focusX);
  }, [apiEndpoint, framingFor]);

  const facecamFor = useCallback((jobId: string) => lastBox.current.get(jobId) ?? null, []);
  const cutoutFor = useCallback((jobId: string): RecallCutoutHost | null => {
    const info = lastCutout.current.get(jobId);
    if (!info) return null;
    const maskKey = `${jobId}/${info.mask}`;
    return {
      source: info.source,
      target: info.target,
      lowerFade: info.lowerFade,
      maskKey,
      loadMask: () => {
        let pendingMask = masks.current.get(maskKey);
        if (!pendingMask) {
          pendingMask = apiFetch(`/jobs/${encodeURIComponent(jobId)}/cutout-mask/${encodeURIComponent(info.mask)}`, undefined, apiEndpoint)
            .then((response) => {
              if (!response.ok) throw new Error("Recall couldn't load the avatar cutout.");
              return response.blob();
            })
            .then((blob) => createImageBitmap(blob));
          pendingMask.catch(() => masks.current.delete(maskKey));
          masks.current.set(maskKey, pendingMask);
        }
        return pendingMask;
      },
    };
  }, [apiEndpoint]);
  // `version` is in the deps so the box and cutout re-read when new framing lands.
  return useMemo(() => ({ canvas: CLIP_CANVAS, lookup, version, facecamFor, cutoutFor }), [lookup, version, facecamFor, cutoutFor]);
}

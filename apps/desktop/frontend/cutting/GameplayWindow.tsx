// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The vertical window a Gameplay clip is cut from, drawn on the Cutting Room's
 * preview and dragged left or right. It mirrors the renderer's cover crop
 * (engines/export/composition.py _cover_fill): the frame is scaled to fill
 * 9:16, and the window's left edge sits at focus_x * (frame - window).
 */
import { useRef } from "react";

/** Width of the 9:16 window as a share of a frame with this aspect (w/h). */
export function windowShare(frameAspect: number): number {
  return Math.min(1, (9 / 16) / frameAspect);
}

/** focus_x (0..1) for a window whose left edge is at `left` (share of width). */
export function focusForLeft(left: number, share: number): number {
  const travel = 1 - share;
  return travel <= 0 ? 0.5 : Math.min(1, Math.max(0, left / travel));
}

export function GameplayWindow({ focusX, frameAspect = 16 / 9, onChange }: {
  focusX: number;
  frameAspect?: number;
  onChange: (focusX: number) => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const drag = useRef<{ startX: number; startLeft: number } | null>(null);
  const share = windowShare(frameAspect);
  const left = focusX * (1 - share);

  const onDown = (event: React.PointerEvent) => {
    if (event.button !== 0) return;
    event.preventDefault();
    root.current?.setPointerCapture(event.pointerId);
    const width = root.current!.getBoundingClientRect().width;
    const x = (event.clientX - root.current!.getBoundingClientRect().left) / width;
    // Click outside the window: jump it there (centred on the click), then drag.
    const insideWindow = x >= left && x <= left + share;
    const startLeft = insideWindow ? left : Math.min(1 - share, Math.max(0, x - share / 2));
    if (!insideWindow) onChange(focusForLeft(startLeft, share));
    drag.current = { startX: event.clientX, startLeft };
  };
  const onMove = (event: React.PointerEvent) => {
    const d = drag.current;
    if (!d || !root.current) return;
    const width = root.current.getBoundingClientRect().width;
    const next = Math.min(1 - share, Math.max(0, d.startLeft + (event.clientX - d.startX) / width));
    onChange(focusForLeft(next, share));
  };
  const onUp = () => { drag.current = null; };
  const onKeyDown = (event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 0.1 : 0.02;
    const delta = event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0;
    if (!delta) return;
    event.preventDefault();
    onChange(Math.min(1, Math.max(0, focusX + delta)));
  };

  return (
    <div
      ref={root}
      className="gameplay-window"
      role="slider"
      aria-label="Where the vertical clip is cut from. Drag left or right, or use the arrow keys."
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(focusX * 100)}
      tabIndex={0}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onKeyDown={onKeyDown}
    >
      <div className="gw-frame" style={{ left: `${left * 100}%`, width: `${share * 100}%` }}>
        <span>Vertical clip</span>
      </div>
    </div>
  );
}

/**
 * Gameplay on the 9:16 preview: grab the picture and drag it left or right to
 * choose what the vertical clip shows. The preview is the window itself, so
 * dragging right reveals more of the left of the frame, like panning a photo.
 */
export function GameplayPan({ focusX, frameAspect = 16 / 9, onChange }: {
  focusX: number;
  frameAspect?: number;
  onChange: (focusX: number) => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const drag = useRef<{ startX: number; startFocus: number } | null>(null);
  const share = windowShare(frameAspect);
  const travel = 1 - share;

  const onDown = (event: React.PointerEvent) => {
    if (event.button !== 0 || travel <= 0) return;
    event.preventDefault();
    root.current?.setPointerCapture(event.pointerId);
    drag.current = { startX: event.clientX, startFocus: focusX };
  };
  const onMove = (event: React.PointerEvent) => {
    const d = drag.current;
    if (!d || !root.current) return;
    const width = root.current.getBoundingClientRect().width;
    // The preview shows `share` of the frame, so a full preview-width drag
    // moves the window by `share` of the frame.
    const moved = ((event.clientX - d.startX) / width) * share;
    onChange(Math.min(1, Math.max(0, d.startFocus - moved / travel)));
  };
  const onUp = () => { drag.current = null; };
  const onKeyDown = (event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 0.1 : 0.02;
    const delta = event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0;
    if (!delta) return;
    event.preventDefault();
    onChange(Math.min(1, Math.max(0, focusX + delta)));
  };

  return (
    <div
      ref={root}
      className="gameplay-pan"
      role="slider"
      aria-label="What the vertical clip shows. Drag the picture left or right, or use the arrow keys."
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(focusX * 100)}
      tabIndex={0}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onKeyDown={onKeyDown}
    >
      <span className="gameplay-pan-hint" aria-hidden="true">Drag to reframe</span>
    </div>
  );
}

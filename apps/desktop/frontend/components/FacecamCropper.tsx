// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Draw the facecam by hand when Recall's detection is off: drag on the source
 * frame to draw a box, drag inside it to move it, drag the corner to resize.
 * The box is normalized [x, y, w, h] on the source frame, the same space the
 * export engine crops from, so what you draw is exactly what gets cut.
 * Arrow keys nudge the box (Shift for bigger steps).
 */
import { useRef, useState } from "react";

export type Box = [number, number, number, number];

const MIN = 0.04;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** A box from two corner points, kept inside the frame and above a minimum size. */
export function boxFromPoints(ax: number, ay: number, bx: number, by: number): Box {
  const x1 = clamp01(Math.min(ax, bx));
  const y1 = clamp01(Math.min(ay, by));
  const x2 = clamp01(Math.max(ax, bx));
  const y2 = clamp01(Math.max(ay, by));
  const w = Math.max(MIN, x2 - x1);
  const h = Math.max(MIN, y2 - y1);
  return [Math.min(x1, 1 - w), Math.min(y1, 1 - h), w, h];
}

/** Move a box by (dx, dy) without letting it leave the frame. */
export function moveBox([x, y, w, h]: Box, dx: number, dy: number): Box {
  return [Math.min(1 - w, Math.max(0, x + dx)), Math.min(1 - h, Math.max(0, y + dy)), w, h];
}

type Drag =
  | { kind: "draw"; ax: number; ay: number }
  | { kind: "move"; sx: number; sy: number; start: Box }
  | { kind: "resize"; start: Box };

export function FacecamCropper({ value, onChange, style }: { value: Box | null; onChange: (box: Box) => void; style?: React.CSSProperties }) {
  const root = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const [draft, setDraft] = useState<Box | null>(null);
  const box = draft ?? value;

  const point = (event: React.PointerEvent) => {
    const r = root.current!.getBoundingClientRect();
    return [clamp01((event.clientX - r.left) / r.width), clamp01((event.clientY - r.top) / r.height)] as const;
  };

  const onDown = (event: React.PointerEvent) => {
    if (event.button !== 0) return;
    event.preventDefault();
    root.current?.setPointerCapture(event.pointerId);
    const [px, py] = point(event);
    const target = event.target as HTMLElement;
    if (value && target.dataset.handle === "resize") drag.current = { kind: "resize", start: value };
    else if (value && target.dataset.handle === "box") drag.current = { kind: "move", sx: px, sy: py, start: value };
    else drag.current = { kind: "draw", ax: px, ay: py };
  };
  const onMove = (event: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const [px, py] = point(event);
    if (d.kind === "draw") setDraft(boxFromPoints(d.ax, d.ay, px, py));
    else if (d.kind === "move") setDraft(moveBox(d.start, px - d.sx, py - d.sy));
    else setDraft(boxFromPoints(d.start[0], d.start[1], px, py));
  };
  const onUp = () => {
    drag.current = null;
    if (draft) onChange(draft);
    setDraft(null);
  };
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (!value) return;
    const step = event.shiftKey ? 0.02 : 0.005;
    const d = ({ ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] } as Record<string, number[]>)[event.key];
    if (!d) return;
    event.preventDefault();
    onChange(moveBox(value, d[0], d[1]));
  };

  return (
    <div
      ref={root}
      className="facecam-cropper"
      style={style}
      role="application"
      aria-label="Draw the facecam. Drag to draw a box, drag it to move, drag the corner to resize, arrow keys to nudge."
      tabIndex={0}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onKeyDown={onKeyDown}
    >
      {!box && <span className="fc-hint">Drag over the facecam</span>}
      {box && (
        <div
          className="fc-box"
          data-handle="box"
          style={{ left: `${box[0] * 100}%`, top: `${box[1] * 100}%`, width: `${box[2] * 100}%`, height: `${box[3] * 100}%` }}
        >
          <span className="fc-label">Cam</span>
          <span className="fc-resize" data-handle="resize" />
        </div>
      )}
    </div>
  );
}

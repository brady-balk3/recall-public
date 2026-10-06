// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Draw the facecam on the whole source frame. The Cutting Room previews the
 * finished 9:16 clip, where most of the frame is cropped away, so the box is
 * drawn here, on a still of the frame under the playhead (the engine's
 * filmstrip), and the preview reframes as soon as it's saved.
 */
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { FacecamCropper, type Box } from "../components/FacecamCropper";

export function FacecamDialog({ frameUrl, value, onSave, onClear, onClose }: {
  frameUrl: string | null;
  value: Box | null;
  onSave: (box: Box) => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const [box, setBox] = useState<Box | null>(value);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return createPortal(
    <div className="project-modal open facecam-dialog" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div className="project-dialog" role="dialog" aria-modal="true" aria-labelledby="facecam-dialog-title">
        <div>
          <h2 id="facecam-dialog-title" className="disp">Where's your facecam?</h2>
          <p>Drag a box around it on this frame. Drag the box to move it, its corner to resize.</p>
        </div>
        <div className="facecam-stage">
          {frameUrl ? <img src={frameUrl} alt="" draggable={false} /> : <span className="t3">Loading the frame…</span>}
          <FacecamCropper value={box} onChange={setBox} style={{ inset: 0 }} />
        </div>
        <div className="project-dialog-actions">
          {value && <button type="button" className="btn ghost" onClick={onClear}>Use Recall's</button>}
          <span className="sp" />
          <button type="button" className="btn ghost" onClick={onClose}>Cancel</button>
          <button type="button" className="btn heat" disabled={!box} onClick={() => box && onSave(box)}>Use this box</button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

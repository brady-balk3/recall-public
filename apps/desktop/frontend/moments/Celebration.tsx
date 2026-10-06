// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The milestone card: the product's one bigger beat. It lands in the middle
 * of the screen with the keepers that got you there, and steps aside on its
 * own after a few seconds.
 */
import { useEffect } from "react";
import { Poster } from "../components/Posters";
import { milestoneLine, milestoneUnlock, useMoments } from "./milestones";

const AUTO_DISMISS_MS = 6500;

export function Celebration({ onOpenCrew }: { onOpenCrew: () => void }) {
  const celebration = useMoments((state) => state.celebration);
  const dismiss = useMoments((state) => state.dismiss);
  const openRecap = useMoments((state) => state.openRecap);

  useEffect(() => {
    if (!celebration) return;
    const timer = window.setTimeout(dismiss, AUTO_DISMISS_MS);
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") dismiss(); };
    window.addEventListener("keydown", onKey);
    return () => { window.clearTimeout(timer); window.removeEventListener("keydown", onKey); };
  }, [celebration, dismiss]);

  if (!celebration) return null;
  const { n, recent } = celebration;
  const unlock = milestoneUnlock(n);
  return (
    <>
      <div className="cel-scrim" onPointerDown={dismiss} aria-hidden="true" />
      <div className="celebrate glass-read" role="status" aria-live="polite">
        <div className="cel-burst" aria-hidden="true" />
        <div className="cel-n disp num">{n}</div>
        <div className="cel-l disp">{n === 1 ? "keeper" : "keepers"}</div>
        <p>{milestoneLine(n)}</p>
        {recent.length > 0 && (
          <div className="cel-row" aria-hidden="true">
            {recent.slice(0, 5).map((clip, i) => (
              <span key={clip.id} style={{ ["--i" as string]: i }}><Poster clip={clip} className="rl-media" /></span>
            ))}
          </div>
        )}
        {unlock && <button type="button" className="cel-unlock" onClick={() => { dismiss(); onOpenCrew(); }}>{unlock}</button>}
        <div className="cel-cta">
          <button type="button" className="btn heat" onClick={openRecap}>Your week so far</button>
          <button type="button" className="btn ghost" onClick={dismiss}>Keep going</button>
        </div>
      </div>
    </>
  );
}

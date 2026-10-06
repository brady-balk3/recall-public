// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** The Export page head and its Clips / Reels / Compilations switch, shared by
 * all three so moving between them never changes the page around you. */
import { Film, Layers, PlayCircle } from "../lib/icons";
import type { View } from "../shell/views";

export type ExportTab = "clips" | "reel" | "compilations";

export function ExportTabs({ active, clipsCount, compilationsCount, onView }: {
  active: ExportTab;
  clipsCount?: number;
  compilationsCount?: number;
  onView: (view: View) => void;
}) {
  return (
    <>
      <div className="ph">
        <div>
          <h1 className="disp">Export</h1>
          <div className="sub">Your keepers, cut vertical and captioned. Post them while the moment's still hot.</div>
        </div>
      </div>
      <div className="x-tabs">
        <div className="seg" role="group" aria-label="What to export">
          <button type="button" aria-pressed={active === "clips"} onClick={() => active !== "clips" && onView("clips")}>
            <PlayCircle aria-hidden="true" />Clips{clipsCount ? <span className="c num">{clipsCount}</span> : null}
          </button>
          <button type="button" aria-pressed={active === "reel"} onClick={() => active !== "reel" && onView("reel")}>
            <Film aria-hidden="true" />Reels
          </button>
          <button type="button" aria-pressed={active === "compilations"} onClick={() => active !== "compilations" && onView("compilations")}>
            <Layers aria-hidden="true" />Compilations{compilationsCount ? <span className="c num">{compilationsCount}</span> : null}
          </button>
        </div>
      </div>
    </>
  );
}

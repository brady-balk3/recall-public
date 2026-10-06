// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The scan bar: paste a Twitch VOD link and press Enter, or choose a
 * recording. The bar lights up the moment it holds something it can scan.
 */
import { useState } from "react";
import { Download, Folder, Pulse } from "../lib/icons";
import { isTwitchVodLink } from "../shell/paletteItems";

export function ScanBar({ onScanLink, onChooseFile }: { onScanLink: (url: string) => void; onChooseFile: () => void }) {
  const [value, setValue] = useState("");
  const ready = isTwitchVodLink(value);
  const looksLikeLink = /^\s*(https?:\/\/|www\.|twitch\.tv)/i.test(value);

  const submit = () => {
    if (!ready) return;
    onScanLink(value.trim());
    setValue("");
  };

  return (
    <>
      <form className={`scanbar glass ${ready ? "ready" : ""}`} onSubmit={(event) => { event.preventDefault(); submit(); }}>
        <span className="sbar-ic" aria-hidden="true"><Download /></span>
        <input
          className="sbar-in"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Paste a Twitch VOD link, or choose a recording"
          aria-label="Twitch VOD link"
          spellCheck={false}
          autoComplete="off"
        />
        <button type="button" className="btn ghost" onClick={onChooseFile}><Folder aria-hidden="true" />Choose file</button>
        <button type="submit" className="btn heat" disabled={!ready}><Pulse aria-hidden="true" />Scan</button>
      </form>
      <div className="sbar-sub" aria-live="polite">
        <span className="sp" />
        <span className="t3 num">
          {looksLikeLink && !ready
            ? "That doesn't look like a VOD link yet. It should look like twitch.tv/videos/123456789"
            : "Starts right away · keep using Recall while it scans"}
        </span>
      </div>
    </>
  );
}

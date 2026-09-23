// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Share handoff — the step after an export finishes.
 *
 * Recall does not upload anything. This surface hands the creator off: it names
 * the folder the files landed in, opens it, and opens each destination's upload
 * page in the browser so the finished MP4 can be dragged straight in. The copy
 * has to stay honest about that split, because "TikTok" next to a logo reads as
 * "posts to TikTok" unless the surface says otherwise.
 *
 * Connected accounts are not wired; nothing here claims they are.
 */
import { type RefObject } from "react";
import { Check, FolderOpen } from "../lib/icons";
import { Instagram, TikTok, XPlatform, YouTube, type PlatformIconProps } from "../lib/platform-icons";
import { plural } from "../lib/copy";

type Destination = {
  id: string;
  name: string;
  /** Shown under the name so the creator knows what will open before it opens. */
  where: string;
  url: string;
  Mark: React.ComponentType<PlatformIconProps>;
};

/** Upload entry points, not marketing pages — each one lands on a composer. */
const DESTINATIONS: Destination[] = [
  { id: "tiktok", name: "TikTok", where: "tiktok.com/upload", url: "https://www.tiktok.com/upload", Mark: TikTok },
  { id: "youtube", name: "YouTube Shorts", where: "youtube.com/upload", url: "https://www.youtube.com/upload", Mark: YouTube },
  { id: "instagram", name: "Instagram Reels", where: "instagram.com", url: "https://www.instagram.com/", Mark: Instagram },
  { id: "x", name: "X", where: "x.com/compose/post", url: "https://x.com/compose/post", Mark: XPlatform },
];

export function ShareHandoffDialog({
  open,
  folder,
  count,
  dialogRef,
  onClose,
}: {
  open: boolean;
  folder: string;
  count: number;
  dialogRef: RefObject<HTMLDivElement | null>;
  onClose: () => void;
}) {
  const openFolder = () => {
    window.electronAPI?.openPath?.(folder);
  };

  // A full Windows path overruns the row and truncates to its drive and user
  // folder, which identifies nothing. The trailing folders are what the
  // creator recognizes; the whole path stays on the tooltip.
  const segments = folder.split(/[\\/]/).filter(Boolean);
  const folderLabel = segments.length > 2 ? `…\\${segments.slice(-2).join("\\")}` : folder;

  const openDestination = (url: string) => {
    window.electronAPI?.openExternal?.(url);
  };

  return (
    <div
      className={`project-modal share-modal ${open ? "open" : ""}`}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className="project-dialog share-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="share-handoff-title"
        aria-describedby="share-handoff-description"
        tabIndex={-1}
      >
        <div className="share-head">
          <span className="share-done" aria-hidden="true"><Check size={13} /></span>
          <div>
            <h2 id="share-handoff-title">{plural(count, "clip")} exported</h2>
            <p id="share-handoff-description">
              The files are on your machine. Recall opens the upload page — you drop the clip in and post it there.
            </p>
          </div>
        </div>

        <button type="button" className="share-folder" data-autofocus onClick={openFolder}>
          <FolderOpen size={16} aria-hidden="true" />
          <span className="share-folder-path" title={folder}>{folderLabel}</span>
          <span className="share-folder-action">Show files</span>
        </button>

        <ul className="share-destinations">
          {DESTINATIONS.map(({ id, name, where, url, Mark }) => (
            <li key={id}>
              <button
                type="button"
                className="share-destination"
                onClick={() => openDestination(url)}
                aria-label={`Open ${name} upload page at ${where}`}
              >
                <span className="share-destination-mark" aria-hidden="true"><Mark size={18} /></span>
                <span className="share-destination-copy">
                  <span className="share-destination-name">{name}</span>
                  <span className="share-destination-where">{where}</span>
                </span>
                <span className="share-destination-cue" aria-hidden="true">Open</span>
              </button>
            </li>
          ))}
        </ul>

        <div className="share-foot">
          <p className="share-note">Posting still happens in your browser. Recall never uploads your video.</p>
          <button type="button" className="btn-secondary" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
}

export default ShareHandoffDialog;

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The fades a clip was last rendered with. The engine applies fades at render
 * time and doesn't keep them on the clip row, so they're remembered here per
 * session (the key the old clip editor used, so its fades carry over).
 */
export interface ClipFades {
  pictureIn: number;
  pictureOut: number;
  soundIn: number;
  soundOut: number;
}

export const NO_CLIP_FADES: ClipFades = { pictureIn: 0, pictureOut: 0, soundIn: 0, soundOut: 0 };

const metaKey = (sessionId: string) => `recall-draft-meta-${sessionId}`;

function readMeta(sessionId: string): Record<string, { fades?: Record<string, unknown> } & Record<string, unknown>> {
  try {
    const raw = localStorage.getItem(metaKey(sessionId));
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function readClipFades(sessionId: string, clipId: string): ClipFades {
  const saved = readMeta(sessionId)[clipId]?.fades;
  if (!saved) return NO_CLIP_FADES;
  return {
    pictureIn: Number(saved.videoIn) || 0,
    pictureOut: Number(saved.videoOut) || 0,
    soundIn: Number(saved.audioIn) || 0,
    soundOut: Number(saved.audioOut) || 0,
  };
}

export function writeClipFades(sessionId: string, clipId: string, fades: ClipFades): void {
  try {
    const meta = readMeta(sessionId);
    meta[clipId] = {
      ...meta[clipId],
      fades: { videoIn: fades.pictureIn, videoOut: fades.pictureOut, audioIn: fades.soundIn, audioOut: fades.soundOut },
    };
    localStorage.setItem(metaKey(sessionId), JSON.stringify(meta));
  } catch {
    // Only a convenience: the clip itself already carries the fades.
  }
}

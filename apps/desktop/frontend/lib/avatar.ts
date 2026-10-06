// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The creator's optional profile photo. Their own file, picked on this PC:
 * center-cropped to a square, shrunk to AVATAR_PX and kept in local storage
 * like their name. Nothing is uploaded anywhere, and the first-letter avatar
 * remains the default.
 */
import { create } from "zustand";

const STORAGE_KEY = "recall-avatar";
export const AVATAR_PX = 256;
/** Refuse files big enough to stall decoding; any real photo fits. */
export const AVATAR_MAX_FILE_BYTES = 25 * 1024 * 1024;
export const AVATAR_ACCEPT = "image/png,image/jpeg,image/webp,image/gif,image/avif";

function readStoredAvatar(): string | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value && value.startsWith("data:image/") ? value : null;
  } catch {
    return null;
  }
}

interface AvatarState {
  photo: string | null;
  setPhoto: (photo: string | null) => void;
}

export const useAvatar = create<AvatarState>((set) => ({
  photo: readStoredAvatar(),
  setPhoto: (photo) => {
    try {
      if (photo) localStorage.setItem(STORAGE_KEY, photo);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* storage full or blocked: the photo lasts for this session only */
    }
    set({ photo });
  },
}));

/** The square crop of a width x height image, centered. */
export function centerSquare(width: number, height: number) {
  const side = Math.min(width, height);
  return { sx: Math.round((width - side) / 2), sy: Math.round((height - side) / 2), side };
}

/** Decode, crop and shrink a picked image into a small data URL. */
export async function avatarFromFile(file: File): Promise<string> {
  if (!file.type.startsWith("image/")) throw new Error("Pick an image file: PNG, JPEG, WebP or GIF.");
  if (file.size > AVATAR_MAX_FILE_BYTES) throw new Error("That image is too large. Pick one under 25 MB.");
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error("Recall couldn't read that image.");
  }
  try {
    const { sx, sy, side } = centerSquare(bitmap.width, bitmap.height);
    const size = Math.min(AVATAR_PX, side);
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Recall couldn't read that image.");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, size, size);
    return canvas.toDataURL("image/webp", 0.88);
  } finally {
    bitmap.close();
  }
}

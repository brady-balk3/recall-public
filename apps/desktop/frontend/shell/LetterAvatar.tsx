// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Identity is a name the creator types, and its first letter, or a photo
 * they pick from this PC (lib/avatar.ts). No handle, nothing from a
 * platform: nothing Recall shows about you comes from anywhere but you.
 */
import { useAvatar } from "../lib/avatar";
import { avatarLetter } from "./shellModel";

export function LetterAvatar({ name, size = 38, className = "" }: { name: string; size?: number; className?: string }) {
  const photo = useAvatar((state) => state.photo);
  return (
    <span
      className={`lav ${photo ? "has-photo" : ""} ${className}`}
      style={{ width: size, height: size, fontSize: Math.round(size * 0.45) }}
      aria-hidden="true"
    >
      {photo ? <img src={photo} alt="" draggable={false} /> : avatarLetter(name)}
    </span>
  );
}

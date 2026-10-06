// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Settings → You: the avatar, with an optional photo the creator picks (or
 * drops) from this PC. Without one, the first letter of their name.
 */
import { useRef, useState } from "react";
import { AVATAR_ACCEPT, avatarFromFile, useAvatar } from "../lib/avatar";
import { LetterAvatar } from "../shell/LetterAvatar";

export function AvatarPicker({ name, size = 56 }: { name: string; size?: number }) {
  const photo = useAvatar((state) => state.photo);
  const setPhoto = useAvatar((state) => state.setPhoto);
  const input = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);

  const take = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setPhoto(await avatarFromFile(file));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Recall couldn't use that image.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="avatar-pick">
      <button
        type="button"
        className={`avatar-pick-face ${dragging ? "is-over" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => { event.preventDefault(); setDragging(false); void take(event.dataTransfer.files?.[0]); }}
        aria-label={photo ? "Change your photo" : "Add a photo"}
        aria-busy={busy}
        disabled={busy}
      >
        <LetterAvatar name={name} size={size} />
      </button>
      <div className="avatar-pick-actions">
        <button type="button" className="link-btn" onClick={() => input.current?.click()} disabled={busy}>
          {photo ? "Change photo" : "Choose a photo"}
        </button>
        {photo && (
          <button type="button" className="link-btn" onClick={() => { setPhoto(null); setError(null); }} disabled={busy}>
            Remove
          </button>
        )}
      </div>
      {error && <small className="avatar-pick-error" role="alert">{error}</small>}
      <input
        ref={input}
        type="file"
        accept={AVATAR_ACCEPT}
        hidden
        onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; void take(file); }}
      />
    </div>
  );
}

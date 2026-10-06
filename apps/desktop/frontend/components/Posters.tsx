// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** Clip and session artwork, with honest fallbacks when media is gone. */
import { useEffect, useState } from "react";
import { Film } from "../lib/icons";
import { originalDeckClips, sortReviewClips } from "../lib/reviewDeck";
import type { Clip, Session } from "../lib/store";

export function ClipMediaPlaceholder({ clip, className }: { clip: Clip; className?: string }) {
  const rebuildable = clip.mediaState === "rebuildable";
  return (
    <span
      className={`clip-media-gone ${className ?? ""}`}
      data-state={clip.mediaState}
      title={rebuildable
        ? "Preview cleared to save space. Rebuild it from this session's source."
        : "The source recording and this preview are both gone. The moment's metadata and your decision are kept."}
    >
      <Film size={16} aria-hidden="true" />
      <span className="clip-media-gone-time t-num">{clip.timestamp}</span>
      <span className="clip-media-gone-label">{rebuildable ? "Preview cleared" : "Media unavailable"}</span>
    </span>
  );
}

export function Poster({ clip, className, eager = false }: { clip: Clip; className?: string; eager?: boolean }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [clip.id, clip.thumbUrl]);
  if (clip.thumbUrl && !failed) {
    return <img className={className} src={clip.thumbUrl} alt="" loading={eager ? "eager" : "lazy"} onError={() => setFailed(true)} />;
  }
  if (clip.videoUrl && !clip.sourceWindow) {
    return <video className={className} src={clip.videoUrl} muted preload="metadata" />;
  }
  if (clip.mediaState === "rebuildable" || clip.mediaState === "unavailable") {
    return <ClipMediaPlaceholder clip={clip} className={className} />;
  }
  return <div className={`draft-card-art ${className ?? ""}`} />;
}

export function pickSessionPosterClip(session: Session): Clip | undefined {
  const deck = originalDeckClips(session.clips);
  const saved = new Set(session.savedClipIds);
  // Prefer real rendered posters. Fall back to exported clip video (not
  // source-window seeks — those metadata frames are usually blank).
  const withThumb = sortReviewClips(
    deck.filter((clip) => !!clip.thumbUrl),
    "recommended",
  );
  const keptThumb = withThumb.find((clip) => saved.has(clip.id) || clip.kept);
  if (keptThumb) return keptThumb;
  if (withThumb[0]) return withThumb[0];
  const withVideo = sortReviewClips(
    deck.filter((clip) => !!clip.videoUrl && !clip.sourceWindow),
    "recommended",
  );
  return withVideo.find((clip) => saved.has(clip.id) || clip.kept) ?? withVideo[0];
}

export function SessionPoster({
  session,
  className,
  variant = "compact",
}: {
  session: Session;
  className?: string;
  /** card = fill the session-card media well; compact = sidebar/list cover */
  variant?: "card" | "compact";
}) {
  const fillClass = variant === "card" ? "session-card-poster" : className;
  const posterClip = pickSessionPosterClip(session);
  // Fetch the VOD still via fetch() (same auth as apiFetch). Only swap it in
  // after a real image blob arrives — never mount a bare <img src=posterUrl>
  // that can sit as a broken-image icon when the source file is gone.
  const [vodSrc, setVodSrc] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    let objectUrl: string | null = null;
    setVodSrc(null);
    if (!session.posterUrl) return () => { alive = false; };
    (async () => {
      try {
        const response = await fetch(session.posterUrl!);
        if (!alive || !response.ok) return;
        const blob = await response.blob();
        if (!alive || !blob.type.startsWith("image/")) return;
        const next = URL.createObjectURL(blob);
        if (!alive) {
          URL.revokeObjectURL(next);
          return;
        }
        objectUrl = next;
        setVodSrc(next);
      } catch {
        // Keep the clip-poster / empty fallback below.
      }
    })();
    return () => {
      alive = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [session.id, session.posterUrl]);

  if (vodSrc) {
    return <img className={fillClass} src={vodSrc} alt="" />;
  }
  if (posterClip) {
    return <Poster clip={posterClip} className={fillClass} />;
  }
  return (
    <span className={`session-poster-fallback ${variant === "card" ? "is-card" : ""} ${className ?? ""}`}>
      <Film aria-hidden="true" />
    </span>
  );
}

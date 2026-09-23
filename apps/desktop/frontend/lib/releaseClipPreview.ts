// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk

/** Close range readers before replacing an MP4 on Windows; restore on failure too. */
export function releaseClipPreview(videoUrl?: string): () => void {
  const target = videoUrl ? new URL(videoUrl, window.location.href) : null;
  const detached: { video: HTMLVideoElement; src: string }[] = [];
  for (const video of document.querySelectorAll("video")) {
    const src = video.getAttribute("src");
    if (!src || !target) continue;
    const url = new URL(src, window.location.href);
    if (url.origin !== target.origin || url.pathname !== target.pathname) continue;
    detached.push({ video, src });
    video.pause();
    video.removeAttribute("src");
    // pause alone leaves the HTTP range request and its Windows file handle open.
    video.load();
  }
  return () => {
    for (const { video, src } of detached) {
      // A successful save can replace the element or supply a refreshed URL.
      if (video.isConnected && !video.getAttribute("src")) {
        video.setAttribute("src", src);
        video.load();
      }
    }
  };
}

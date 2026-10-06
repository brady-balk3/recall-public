// SPDX-License-Identifier: AGPL-3.0-or-later
// Recall shim: OpenCut was written for Next.js; this stands in for it under Vite.
import type { ImgHTMLAttributes } from "react";
export default function Image({ src, fill, priority: _p, quality: _q, unoptimized: _u, style, ...rest }: Omit<ImgHTMLAttributes<HTMLImageElement>, "src"> & { src?: string | { src: string }; fill?: boolean; priority?: boolean; quality?: number; unoptimized?: boolean }) {
  const s = typeof src === "string" || src == null ? src : src.src;
  return <img src={s} style={fill ? { position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", ...style } : style} {...rest} />;
}

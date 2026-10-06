// SPDX-License-Identifier: AGPL-3.0-or-later
// Recall shim: OpenCut was written for Next.js; this stands in for it under Vite.
import type { AnchorHTMLAttributes } from "react";
export default function Link({ href, prefetch: _p, ...rest }: AnchorHTMLAttributes<HTMLAnchorElement> & { href: string; prefetch?: boolean }) {
  return <a href={typeof href === "string" ? href : "#"} {...rest} />;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Recall shim: OpenCut was written for Next.js; this stands in for it under Vite.
import { lazy, Suspense, type ComponentType } from "react";
export default function dynamic<P extends object>(load: () => Promise<{ default: ComponentType<P> } | ComponentType<P>>) {
  const C = lazy(async () => { const m = await load(); return "default" in (m as object) ? (m as { default: ComponentType<P> }) : { default: m as ComponentType<P> }; });
  return (props: P) => <Suspense fallback={null}><C {...props} /></Suspense>;
}

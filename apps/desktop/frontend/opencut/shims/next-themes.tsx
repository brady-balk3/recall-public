// SPDX-License-Identifier: AGPL-3.0-or-later
// Recall shim: OpenCut was written for Next.js; this stands in for it under Vite.
// Recall owns light/dark, so the editor's own toggle is a no-op.
import type { ReactNode } from "react";
export const ThemeProvider = ({ children }: { children: ReactNode; [prop: string]: unknown }) => <>{children}</>;
export const useTheme = () => ({ theme: "dark", resolvedTheme: "dark", setTheme: (_theme: string) => {}, themes: ["dark"] });

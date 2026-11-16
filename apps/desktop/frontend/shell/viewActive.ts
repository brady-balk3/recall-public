// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Is the page this component sits on the one on screen? ViewPanel says so for
 * the pages it keeps mounted behind the others. Anything outside a ViewPanel
 * (the sidebar, docks, pages that unmount when you leave) is on screen
 * whenever it is mounted, hence the default.
 */
import { createContext, useContext } from "react";

export const ViewActiveContext = createContext(true);

export function useViewActive(): boolean {
  return useContext(ViewActiveContext);
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. OpenCut's menus, dialogs and drag ghosts portal to
 * document.body, which sits outside the .oc-root scope its styles live in.
 * The Recall wrapper mounts a body-level `.oc-root.oc-portal-host` layer and
 * every portal targets it instead.
 */
export function getOpenCutPortalHost(): HTMLElement | undefined {
	if (typeof document === "undefined") return undefined;
	return document.querySelector<HTMLElement>(".oc-portal-host") ?? document.body;
}

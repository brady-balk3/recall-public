// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Recall addition. OpenCut offers ~1,900 Google Fonts and downloads each one
 * on use; Recall stays offline, so the editor offers only faces that will
 * actually render: the system fonts plus the ones Recall already ships.
 *
 * Recall's own CSS registers these as "Sora Variable" etc.; text on the
 * editor canvas asks for the plain family name, so each is registered again
 * under that name from the same bundled file.
 */
import soraUrl from "@fontsource-variable/sora/files/sora-latin-wght-normal.woff2?url";
import geistUrl from "@fontsource-variable/geist/files/geist-latin-wght-normal.woff2?url";
import geistMonoUrl from "@fontsource-variable/geist-mono/files/geist-mono-latin-wght-normal.woff2?url";
import archivoUrl from "@fontsource-variable/archivo/files/archivo-latin-wght-normal.woff2?url";

export const BUNDLED_FONTS: Readonly<Record<string, string>> = {
	Sora: soraUrl,
	Geist: geistUrl,
	"Geist Mono": geistMonoUrl,
	Archivo: archivoUrl,
};

const registered = new Map<string, Promise<void>>();

/** Make a bundled family available to canvas text. Unknown families are a no-op. */
export function loadBundledFont(family: string): Promise<void> {
	const url = BUNDLED_FONTS[family];
	if (!url || typeof document === "undefined") return Promise.resolve();
	let pending = registered.get(family);
	if (!pending) {
		const face = new FontFace(family, `url(${url}) format("woff2")`, { weight: "100 900", style: "normal" });
		pending = face.load().then(
			(loaded) => { document.fonts.add(loaded); },
			(error: unknown) => { console.warn(`Couldn't load bundled font ${family}`, error); },
		);
		registered.set(family, pending);
	}
	return pending;
}

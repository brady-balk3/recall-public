import { useState, useMemo, useCallback, useEffect } from "react";
import {
	getCachedFontAtlas,
	loadFontAtlas,
	clearFontAtlasCache,
} from "@opencut/fonts/google-fonts";
import type { FontAtlas } from "@opencut/fonts/types";
import { SYSTEM_FONTS } from "@opencut/fonts/system-fonts";
import { BUNDLED_FONTS } from "@opencut/recall/bundled-fonts";

type Status = "idle" | "loading" | "error";

export function useFontAtlas({ open }: { open: boolean }) {
	const [atlas, setAtlas] = useState<FontAtlas | null>(() =>
		getCachedFontAtlas(),
	);
	const [status, setStatus] = useState<Status>(() =>
		getCachedFontAtlas() ? "idle" : "loading",
	);

	useEffect(() => {
		if (!open || atlas) return;

		setStatus("loading");
		loadFontAtlas().then((data) => {
			if (data) {
				setAtlas(data);
				setStatus("idle");
			} else {
				setStatus("error");
			}
		});
	}, [open, atlas]);

	const retry = useCallback(() => {
		clearFontAtlasCache();
		setStatus("loading");
		loadFontAtlas().then((data) => {
			if (data) {
				setAtlas(data);
				setStatus("idle");
			} else {
				setStatus("error");
			}
		});
	}, []);

	const fontNames = useMemo(() => {
		if (!atlas) return [];
		// Recall: offline, so only faces that render (bundled + system), not the whole atlas.
		return [...Object.keys(BUNDLED_FONTS).filter((name) => name in atlas.fonts), ...SYSTEM_FONTS].sort();
	}, [atlas]);

	return { atlas, status, fontNames, retry };
}

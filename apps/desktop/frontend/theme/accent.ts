// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The accent: Recall is black and white until the creator picks one color.
 *
 * The color is stored in OKLCH terms so each control does exactly one thing:
 * the wheel sets hue, Intensity sets chroma (grey to vivid), Brightness sets
 * lightness. Everything that means "this matters" reads three stops derived
 * from it (`--h1`, `--h2`, `--h3`): one hue with a small lightness ramp, so a
 * gradient has life without ever turning into a second color.
 *
 * `null` is black and white, the default.
 *
 * Caption highlight color is a separate setting and never follows the app
 * accent unless the creator picks "Match accent".
 */

export interface Accent {
  /** 0-359, degrees on the OKLCH hue circle. */
  hue: number;
  /** 0-100, grey to vivid. */
  intensity: number;
  /** 0-100, deep to light. */
  brightness: number;
}

export type AccentChoice = Accent | null;

export const ACCENT_STORAGE_KEY = "recall-accent";
const LEGACY_ACCENT_KEY = "recall-accent-theme";

/** Chroma at Intensity 100. Past this most hues leave the sRGB gamut. */
const C_MAX = 0.26;
const lightnessOf = (brightness: number) => 0.38 + (brightness / 100) * 0.54;
const chromaOf = (intensity: number) => (intensity / 100) * C_MAX;
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const wrapHue = (hue: number) => ((hue % 360) + 360) % 360;

export interface AccentPreset {
  id: string;
  label: string;
  accent: AccentChoice;
}

export const ACCENT_PRESETS: AccentPreset[] = [
  { id: "mono", label: "Black & white", accent: null },
  { id: "heat", label: "Heat", accent: { hue: 38, intensity: 78, brightness: 68 } },
  { id: "ember", label: "Ember", accent: { hue: 32, intensity: 84, brightness: 64 } },
  { id: "prism", label: "Prism", accent: { hue: 278, intensity: 62, brightness: 70 } },
  { id: "voltage", label: "Voltage", accent: { hue: 312, intensity: 74, brightness: 62 } },
  { id: "citrus", label: "Citrus", accent: { hue: 92, intensity: 76, brightness: 84 } },
  { id: "acid", label: "Acid", accent: { hue: 148, intensity: 70, brightness: 80 } },
  { id: "ice", label: "Ice", accent: { hue: 232, intensity: 44, brightness: 84 } },
];

/** CSS `oklch()` for an accent, with optional lightness offset and chroma scale. */
export function accentOklch(accent: Accent, lightnessDelta = 0, chromaScale = 1): string {
  const l = clamp(lightnessOf(accent.brightness) + lightnessDelta, 0, 1);
  const c = chromaOf(accent.intensity) * chromaScale;
  return `oklch(${l.toFixed(3)} ${c.toFixed(3)} ${wrapHue(accent.hue).toFixed(1)})`;
}

/** sRGB hex for an accent (gamut-clipped), for readouts and non-CSS consumers. */
export function accentHex(accent: Accent): string {
  const L = lightnessOf(accent.brightness);
  const C = chromaOf(accent.intensity);
  const h = (wrapHue(accent.hue) * Math.PI) / 180;
  const a = C * Math.cos(h);
  const b = C * Math.sin(h);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3;
  const m = m_ ** 3;
  const s = s_ ** 3;
  const linear = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return (
    "#" +
    linear
      .map((v) => {
        const encoded = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(Math.max(v, 0), 1 / 2.4) - 0.055;
        return Math.round(clamp(encoded, 0, 1) * 255).toString(16).padStart(2, "0");
      })
      .join("")
      .toUpperCase()
  );
}

/** Text color that reads on an accent fill. */
export function accentInk(accent: Accent): string {
  return lightnessOf(accent.brightness) > 0.7 ? "#0B0B0C" : "#FFFFFF";
}

/** The CSS custom properties an accent sets on the root. Empty for black and white. */
export function accentVariables(accent: AccentChoice): Record<string, string> {
  if (!accent) return {};
  return {
    "--h1": accentOklch({ ...accent, hue: accent.hue - 6 }, -0.05),
    "--h2": accentOklch(accent),
    "--h3": accentOklch({ ...accent, hue: accent.hue + 6 }, 0.06, 0.88),
    "--heat-ink": accentInk(accent),
  };
}

const ACCENT_VARIABLE_NAMES = ["--h1", "--h2", "--h3", "--heat-ink"];

/** Apply an accent to the document. `data-palette` switches the mono token set on or off. */
export function applyAccent(accent: AccentChoice, root: HTMLElement = document.documentElement) {
  root.dataset.palette = accent ? "custom" : "mono";
  const vars = accentVariables(accent);
  for (const name of ACCENT_VARIABLE_NAMES) {
    if (vars[name]) root.style.setProperty(name, vars[name]);
    else root.style.removeProperty(name);
  }
}

function isAccent(value: unknown): value is Accent {
  if (!value || typeof value !== "object") return false;
  const { hue, intensity, brightness } = value as Record<string, unknown>;
  return [hue, intensity, brightness].every((n) => typeof n === "number" && Number.isFinite(n));
}

export function normalizeAccent(accent: Accent): Accent {
  return {
    hue: Math.round(wrapHue(accent.hue)),
    intensity: Math.round(clamp(accent.intensity, 0, 100)),
    brightness: Math.round(clamp(accent.brightness, 0, 100)),
  };
}

/**
 * The retired named accents, mapped to their nearest OKLCH color. "ember" was
 * the old default, so it can't tell a choice from a default: it becomes black
 * and white, the new default.
 */
const LEGACY_ACCENTS: Record<string, AccentChoice> = {
  ember: null,
  rose: { hue: 0, intensity: 42, brightness: 60 },
  violet: { hue: 295, intensity: 42, brightness: 58 },
  crimson: { hue: 24, intensity: 48, brightness: 58 },
  cobalt: { hue: 255, intensity: 38, brightness: 58 },
  sage: { hue: 155, intensity: 24, brightness: 60 },
};

export function readStoredAccent(storage: Pick<Storage, "getItem"> | undefined = safeLocalStorage()): AccentChoice {
  if (!storage) return null;
  try {
    const raw = storage.getItem(ACCENT_STORAGE_KEY);
    if (raw != null) {
      if (raw === "mono") return null;
      const parsed: unknown = JSON.parse(raw);
      return isAccent(parsed) ? normalizeAccent(parsed) : null;
    }
    const legacy = storage.getItem(LEGACY_ACCENT_KEY);
    if (legacy && legacy in LEGACY_ACCENTS) return LEGACY_ACCENTS[legacy];
  } catch {}
  return null;
}

export function writeStoredAccent(accent: AccentChoice, storage: Pick<Storage, "setItem" | "removeItem"> | undefined = safeLocalStorage()) {
  if (!storage) return;
  try {
    storage.setItem(ACCENT_STORAGE_KEY, accent ? JSON.stringify(normalizeAccent(accent)) : "mono");
    storage.removeItem(LEGACY_ACCENT_KEY);
  } catch {}
}

function safeLocalStorage(): Storage | undefined {
  try {
    return typeof localStorage !== "undefined" ? localStorage : undefined;
  } catch {
    return undefined;
  }
}

export function sameAccent(a: AccentChoice, b: AccentChoice): boolean {
  if (!a || !b) return a === b;
  return a.hue === b.hue && a.intensity === b.intensity && a.brightness === b.brightness;
}

/* ── Caption highlight color ── */

export type CaptionColor = "yellow" | "white" | "green" | "cyan" | "pink" | "orange" | "accent";

export const CAPTION_COLORS: { id: CaptionColor; label: string; hex: string | null }[] = [
  { id: "yellow", label: "Yellow", hex: "#FFFF00" },
  { id: "white", label: "White", hex: "#FFFFFF" },
  { id: "green", label: "Green", hex: "#35E37D" },
  { id: "cyan", label: "Cyan", hex: "#2FD8FF" },
  { id: "pink", label: "Pink", hex: "#FF4FA3" },
  { id: "orange", label: "Orange", hex: "#FF8A1F" },
  { id: "accent", label: "Match accent", hex: null },
];

export const CAPTION_COLOR_STORAGE_KEY = "recall-caption-color";

export function isCaptionColor(value: unknown): value is CaptionColor {
  return CAPTION_COLORS.some((c) => c.id === value);
}

/**
 * Concrete hex for a caption color. "Match accent" resolves to the accent's
 * hex, or white in black and white, so the renderer always gets a real color.
 */
export function captionColorHex(color: CaptionColor, accent: AccentChoice): string {
  if (color === "accent") return accent ? accentHex(accent) : "#FFFFFF";
  return CAPTION_COLORS.find((c) => c.id === color)?.hex ?? "#FFFF00";
}

export function applyCaptionColor(color: CaptionColor, accent: AccentChoice, root: HTMLElement = document.documentElement) {
  root.style.setProperty("--cap-hi", captionColorHex(color, accent));
}

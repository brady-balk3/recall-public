// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The dot field: Recall's dark background, drawn inside #ambient (theme/ambient.tsx).
 *
 * One ordered (Bayer) dither of light, in small square dots, running the full
 * length of the open view and moving with its scroll. The light comes from:
 * - the frame on screen, blurred the way the old footage glow was, at the top;
 * - every clip poster in the view, glowing in its own colour into the gutters;
 * - soft blooms in the accent, behind each section heading and down the page,
 *   settling to a faint floor so the texture never stops.
 *
 * It moves, always and slowly: soft waves of light roll across the dots as a loop
 * of pre-drawn states crossfades one into the next (see startLoop), so dots come
 * and go as the light passes. Compositor-only, stepped, off for reduced motion.
 *
 * It stays calm under the UI: dots thin out over a broad blur of wherever text
 * sits straight on the background (never per line, which reads as a shadow),
 * stay out from behind translucent controls and the edge-faded keepers reel,
 * glass gets half the light, and blooms ease off past a knee.
 *
 * Drawing is cheap on memory at any display scale: dots are one canvas pixel per
 * 4px cell, scaled up pixelated, with the gaps cut by a CSS mask (glass.css).
 * It redraws only when the view, its size, its content or the frame changes,
 * never while scrolling, and a view switch never waits on it (see DotField).
 */

export type Rgb = [number, number, number];

export interface DotFieldInput {
  /** The frame the top of the page is lit by (the session poster). */
  frameUrl?: string;
  /** Dot colour for the accent light (theme/accent.ts accentDotRgb). */
  dot: Rgb;
}

const STEP = 8; // light grid, css px
const PITCH = 4; // one dot cell, css px; the 2px square is the mask in glass.css
const WASH = 1.1; // how strongly the smooth light shows under the dots
const DOT_LO = 0.3; // dot brightness at no light...
const DOT_HI = 0.62; // ...plus this at full light
const STATES = 6; // drawn states in one loop of the always-on motion
const LOOP_MS = 14000; // one full loop of it
const SWELL = 0.5; // how strongly the travelling light lifts and dims the field

/** The opening blooms, in viewport units (x by width, y and r by height). */
const TOP_BLOOMS = [
  [0.2, 0.04, 0.42, 1.0],
  [0.56, -0.04, 0.26, 0.45],
  [0.86, 0.08, 0.36, 0.8],
  [0.03, 0.4, 0.28, 0.42],
  [0.44, 0.27, 0.22, 0.28],
  [0.99, 0.44, 0.24, 0.36],
];

/** 8×8 Bayer thresholds in (0, 1). */
export const BAYER: number[][] = (() => {
  let out = [[0]];
  for (let n = 1; n < 8; n *= 2) {
    const next: number[][] = [];
    for (let y = 0; y < n * 2; y++) {
      next.push([]);
      for (let x = 0; x < n * 2; x++) next[y].push(4 * out[y % n][x % n] + [0, 2, 3, 1][(y >= n ? 2 : 0) + (x >= n ? 1 : 0)]);
    }
    out = next;
  }
  return out.map((row) => row.map((v) => (v + 0.5) / 64));
})();

/** Low-frequency cloud so blooms break up instead of reading as gradients. */
const cloud = (x: number, y: number) => 0.5 * Math.sin(x * 5.1) * Math.sin(y * 6.3 - x * 2.2) + 0.25 * Math.sin(x * 11.7 - y * 8.1 + 1.3);
const hash = (n: number) => {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
};
/** Past this the blooms ease off rather than fill in. */
export const knee = (l: number) => (l > 0.45 ? 0.45 + (l - 0.45) * 0.4 : l);

interface Rect { l: number; r: number; t: number; b: number }
interface Clip { x: number; y: number; w: number; h: number; rgb: Rgb }
/** A rounded rect the dots stay out from behind, fading out over `f` px past its edge. */
interface Blocker extends Rect { f: number; rad: number }
export type Bloom = [number, number, number, number];

/** Blooms in page px: the opening group, then light behind each section heading. */
function bloomsFor(W: number, Hv: number, pageH: number, panel: HTMLElement): Bloom[] {
  const list: Bloom[] = TOP_BLOOMS.map(([x, y, r, a]) => [x * W, y * Hv, r * Hv, a]);
  const heads = [...panel.querySelectorAll(".sec-t")].map((el) => {
    const r = el.getBoundingClientRect();
    return [r.left + 40, r.top + panel.scrollTop + r.height / 2];
  });
  let lastY = Hv * 0.5;
  heads.forEach(([x, y], k) => {
    // fill long gaps so no stretch of page goes dark
    for (let gy = lastY + Hv * 0.9; gy < y - Hv * 0.5; gy += Hv * 0.9) list.push([W * (hash(gy) > 0.5 ? 0.86 : 0.5), gy, Hv * 0.34, 0.38]);
    list.push([x, y, Hv * 0.34, 0.78 - Math.min(0.25, k * 0.05)]);
    list.push([Math.min(W * 0.97, x + W * 0.55 + hash(k) * W * 0.2), y + Hv * 0.18, Hv * 0.26, 0.4]);
    lastY = y;
  });
  for (let gy = lastY + Hv * 0.9; gy < pageH + Hv * 0.2; gy += Hv * 0.9) list.push([W * (hash(gy) > 0.5 ? 0.84 : 0.16), gy, Hv * 0.34, 0.38]);
  return list;
}

/**
 * The blooms at point `t` (0-1) of the motion loop: each one circles its own
 * small path and swells and settles a little. t = 0 and t = 1 are the same, so
 * the loop is seamless.
 */
export function driftBlooms(blooms: Bloom[], Hv: number, t: number): Bloom[] {
  const w = 2 * Math.PI * t;
  return blooms.map(([x, y, r, a], i) => [
    x + (Math.cos(w + i * 1.7) - Math.cos(i * 1.7)) * 0.08 * Hv,
    y + (Math.sin(w + i * 2.3) - Math.sin(i * 2.3)) * 0.05 * Hv,
    r * (1 + 0.08 * Math.sin(w + i)),
    Math.min(1, a * (1 + 0.12 * Math.sin(w + i * 1.3))),
  ]);
}

/**
 * Light travelling across the page at point `t` of the loop, in [-1, 1]: two soft
 * diagonal waves about a screen across that roll along as t goes from 0 to 1. A
 * moving state is lit by `1 + SWELL * travel()` (after the knee), so all light
 * (frame, clips, accent) rises and falls as a wave passes.
 */
export function travel(x: number, y: number, Hv: number, t: number) {
  const u = x / Hv, v = y / Hv, w = 2 * Math.PI * t;
  const s = 0.6 * Math.sin(2.2 * u + 1.4 * v - w) + 0.4 * Math.sin(-1.3 * u + 2.6 * v - w + 1.1);
  return Math.max(-1, Math.min(1, s));
}

/** `cx`, `cy` move the cloud, so moving states differ everywhere, not just in the blooms. */
function accentLight(x: number, y: number, Hv: number, blooms: Bloom[], cx = 0, cy = 0) {
  let inv = 1;
  for (const [bx, by, r, a] of blooms) {
    const dy = (y - by) / r;
    if (dy > 3.2 || dy < -3.2) continue;
    const dx = (x - bx) / r;
    inv *= 1 - a * Math.exp(-(dx * dx + dy * dy));
  }
  const n = cloud(x / Hv + cx, y / Hv + cy);
  const f = Math.max(0, (1 - inv) * (0.7 + 0.5 * n));
  // a quiet floor: the texture never fully stops
  return Math.max(1.35 * Math.pow(f, 1.6), 0.045 * (0.7 + 0.6 * n));
}

/** Does this element paint its own surface? Text on one doesn't need the background quiet. */
function paints(el: Element) {
  const cs = getComputedStyle(el);
  return cs.backgroundImage !== "none" || !/rgba\(0, 0, 0, 0\)|transparent/.test(cs.backgroundColor) || cs.backdropFilter !== "none";
}

/** True when nothing between `el` (inclusive) and the view paints a surface. */
function bareChecker(panel: HTMLElement) {
  const painted = new Map<Element, boolean>();
  return (el: Element) => {
    for (let e: Element | null = el; e && e !== panel; e = e.parentElement) {
      if (!painted.has(e)) painted.set(e, paints(e));
      if (painted.get(e)) return false;
    }
    return true;
  };
}

/** Every text run in the view with nothing painted behind it, in page px. */
function bareText(panel: HTMLElement, bare: (el: Element) => boolean): Rect[] {
  const top = panel.scrollTop;
  const out: Rect[] = [];
  const walker = document.createTreeWalker(panel, NodeFilter.SHOW_TEXT);
  const range = document.createRange();
  for (let n = walker.nextNode() as Text | null; n; n = walker.nextNode() as Text | null) {
    if (!n.data.trim() || !n.parentElement || !bare(n.parentElement)) continue;
    range.selectNodeContents(n);
    for (const r of range.getClientRects()) {
      if (r.width > 1 && r.height > 1) out.push({ l: r.left - 6, r: r.right + 6, t: r.top + top - 4, b: r.bottom + top + 4 });
    }
  }
  return out;
}

const CONTROLS = "button, a[href], input, select, textarea, [role='button'], [role='tab'], .btn, .chip";

/**
 * What the dots must stay out from behind, in page px, with how far each fades
 * out past its edge:
 * - translucent controls on the bare background (glassy buttons, chips, inputs),
 *   which would otherwise show the dots right through;
 * - media strips that fade their own edges with a mask (the keepers reel), where
 *   dots would bleed into the fading cards.
 */
function blockers(panel: HTMLElement, bare: (el: Element) => boolean): Blocker[] {
  const top = panel.scrollTop;
  const out: Blocker[] = [];
  const add = (el: Element, f: number, rounded: boolean) => {
    const r = el.getBoundingClientRect();
    if (r.width <= 4 || r.height <= 4) return;
    // follow the control's own corners, so a pill clears a pill and not a box
    const rad = rounded ? Math.min(parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0, r.width / 2, r.height / 2) : 0;
    out.push({ l: r.left, r: r.right, t: r.top + top, b: r.bottom + top, f, rad });
  };
  for (const el of panel.querySelectorAll(CONTROLS)) if (el.parentElement && bare(el.parentElement)) add(el, 6, true);
  for (const el of panel.querySelectorAll(".reel")) add(el, 40, false);
  return out;
}

/** Glass that blurs the background. Outside the view it's fixed, so it spans every page y. */
function glassOver(panel: HTMLElement): Rect[] {
  const top = panel.scrollTop;
  const out: Rect[] = [];
  for (const el of document.querySelectorAll(".sidebar, aside, .glass, .glass-float, .glass-read, .scanbar")) {
    const r = el.getBoundingClientRect();
    if (r.width < 80 || r.height < 40 || getComputedStyle(el).backdropFilter === "none") continue;
    if (panel.contains(el)) out.push({ l: r.left, r: r.right, t: r.top + top, b: r.bottom + top });
    else out.push({ l: r.left, r: r.right, t: -1e6, b: 1e6 });
  }
  return out;
}

/** In-place box blur, three passes: close to a gaussian. */
function blur3(a: Float32Array, cols: number, rows: number, R: number) {
  const tmp = new Float32Array(a.length);
  const w = 2 * R + 1;
  for (let pass = 0; pass < 3; pass++) {
    for (let j = 0; j < rows; j++) {
      let s = 0;
      for (let i = -R; i <= R; i++) s += a[j * cols + Math.min(cols - 1, Math.max(0, i))];
      for (let i = 0; i < cols; i++) {
        tmp[j * cols + i] = s / w;
        s += a[j * cols + Math.min(cols - 1, i + R + 1)] - a[j * cols + Math.max(0, i - R)];
      }
    }
    for (let i = 0; i < cols; i++) {
      let s = 0;
      for (let j = -R; j <= R; j++) s += tmp[Math.min(rows - 1, Math.max(0, j)) * cols + i];
      for (let j = 0; j < rows; j++) {
        a[j * cols + i] = s / w;
        s += tmp[Math.min(rows - 1, j + R + 1) * cols + i] - tmp[Math.max(0, j - R) * cols + i];
      }
    }
  }
}

/**
 * Images load in CORS mode so their pixels can be read: the API echoes the
 * renderer's origin (apps/api/main.py). One that still taints yields no colour.
 */
function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function averageColour(img: HTMLImageElement): Rgb | null {
  try {
    const c = document.createElement("canvas");
    c.width = c.height = 6;
    const x = c.getContext("2d", { willReadFrequently: true });
    if (!x) return null;
    x.drawImage(img, 0, 0, 6, 6);
    const d = x.getImageData(0, 0, 6, 6).data;
    let r = 0, g = 0, b = 0;
    for (let k = 0; k < 36; k++) { r += d[k * 4]; g += d[k * 4 + 1]; b += d[k * 4 + 2]; }
    const val = Math.max(r, g, b) / 36 / 255;
    if (val <= 0.02) return null;
    const norm = 200 / (val * 255) / 36;
    return [r * norm, g * norm, b * norm];
  } catch {
    return null;
  }
}

/** The frame blurred into the light grid the way the old glow drew it: light and colour per cell. */
function frameLight(img: HTMLImageElement, cols: number, W: number, Hv: number) {
  const rows = Math.ceil(Hv / STEP) + 2;
  const c = document.createElement("canvas");
  c.width = cols;
  c.height = rows;
  const x = c.getContext("2d", { willReadFrequently: true });
  if (!x) return null;
  x.fillStyle = "#000";
  x.fillRect(0, 0, cols, rows);
  x.filter = `blur(${90 / STEP}px) saturate(1.6)`;
  const dw = cols * 1.2;
  const dh = Math.max((dw * img.naturalHeight) / img.naturalWidth, rows * 0.95);
  x.drawImage(img, -cols * 0.1, -rows * 0.3, dw, dh);
  let px: Uint8ClampedArray;
  try { px = x.getImageData(0, 0, cols, rows).data; } catch { return null; }
  const F = new Float32Array(cols * rows);
  const rgb = new Uint8ClampedArray(cols * rows * 3);
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const k = j * cols + i, r = px[k * 4], g = px[k * 4 + 1], b = px[k * 4 + 2];
    const u = ((i * STEP) / W - 0.5) / 0.6, v = (j * STEP) / Hv / 0.85, d = Math.sqrt(u * u + v * v);
    const mask = d < 0.38 ? 1 - (0.45 * d) / 0.38 : d < 0.72 ? 0.55 * (1 - (d - 0.38) / 0.34) : 0;
    const val = Math.max(r, g, b) / 255;
    F[k] = val * mask * 1.5;
    const norm = val > 0.02 ? 200 / (val * 255) : 0;
    rgb[k * 3] = r * norm; rgb[k * 3 + 1] = g * norm; rgb[k * 3 + 2] = b * norm;
  }
  return { F, rgb, rows };
}

/** A wash and a dot canvas, drawn as one state of the field. */
interface Layer { el: HTMLDivElement; wash: HTMLCanvasElement; dots: HTMLCanvasElement }

function makeLayer(className: string): Layer {
  const el = document.createElement("div");
  el.className = `dot-layer ${className}`;
  const wash = document.createElement("canvas");
  wash.className = "dot-wash";
  const dots = document.createElement("canvas");
  dots.className = "dot-cells";
  el.append(wash, dots);
  return { el, wash, dots };
}

/**
 * One drawn field for one view, shown as a unit. `rest` is the field as drawn,
 * shown at once. `motion` is the always-on movement: STATES copies of the field
 * with the light a step further along its loop, built in the background once the
 * view is up, then crossfaded one into the next (see startLoop).
 */
interface Sheet {
  el: HTMLDivElement;
  rest: Layer;
  motion: { el: HTMLDivElement; animations: Animation[] } | null;
  /** Builds `motion` in the background; null until the rest of the page is drawn. */
  buildMotion: (() => void) | null;
  building: boolean;
  /** What it was drawn for (see DotField.key); a mismatch means redraw. */
  key: string;
  /** Set when the sheet is discarded, so its remaining bands stop. */
  dead: boolean;
}

function makeSheet(): Sheet {
  const el = document.createElement("div");
  el.className = "dot-sheet";
  el.hidden = true;
  const rest = makeLayer("dot-rest");
  el.append(rest.el);
  return { el, rest, motion: null, buildMotion: null, building: false, key: "", dead: false };
}

/**
 * Keyframes for one moving layer: hidden until `a`, fades in on an ease-in-out
 * curve until `b`, then stays until the loop ends. The fade is stepped (three
 * changes between each of 8 points: ~10 updates a second) so the glass above,
 * which re-blurs whatever changes behind it, isn't made to re-blur every frame.
 */
function rampKeyframes(a: number, b: number): Keyframe[] {
  const frames: Keyframe[] = [{ offset: 0, opacity: 0 }];
  if (a > 0) frames.push({ offset: a, opacity: 0 });
  for (let s = 0; s <= 8; s++) {
    const p = s / 8;
    frames.push({ offset: a + (b - a) * p, opacity: p * p * (3 - 2 * p), easing: "steps(4, jump-none)" });
  }
  if (b < 1) frames.push({ offset: 1, opacity: 1 });
  return frames;
}

/**
 * The loop. Layers are stacked state 0, 1, ..., STATES-1, then a copy of state 0
 * on top. Each fades in over the one below in its own slot of the loop and stays;
 * when the loop ends all of them drop back at once, which shows state 0 again:
 * the same picture as the copy that was just on top, so the wrap is invisible.
 * Only opacity animates, on the compositor.
 */
function startLoop(layers: Layer[]): Animation[] {
  const n = layers.length - 1;
  return layers.slice(1).map((layer, i) =>
    layer.el.animate(rampKeyframes(i / n, (i + 1) / n), { duration: LOOP_MS, iterations: Infinity }));
}

/** Keep the motion of this many recently shown views; the rest is rebuilt when they're shown again. */
const KEEP_MOTION = 3;

const KEEP_VIEWS = 8;

/**
 * Switching views must never show a stale or late background. Each view keeps
 * its own drawn sheet, so going back to one swaps it in the same frame with no
 * drawing at all. A view seen for the first time is drawn before the frame
 * paints: only the visible window (about 10-15ms), the rest of the page in small
 * chunks right after. Changes within a view (posters finishing loading, the page
 * growing, the accent) draw a new sheet and crossfade to it.
 */
export class DotField {
  private readonly host: HTMLElement;
  private input: DotFieldInput;
  private panel: HTMLElement | null = null;
  /** Drawn sheets by view panel id, most recently shown last. */
  private readonly sheets = new Map<string, Sheet>();
  private shown: Sheet | null = null;
  private poll = 0;
  private soon = 0;
  /** Poster colour per URL; null when it couldn't be read. */
  private readonly colours = new Map<string, Rgb | null>();
  private readonly frames = new Map<string, HTMLImageElement | null>();
  /** The blurred frame light is the same for every view: compute it once per frame and size. */
  private footCache: { key: string; value: ReturnType<typeof frameLight> } | null = null;
  private readonly loading = new Set<string>();
  /** Sheets that hold built motion, most recently shown last. */
  private moving: Sheet[] = [];
  private readonly still = matchMedia("(prefers-reduced-motion: reduce)");

  constructor(host: HTMLElement, input: DotFieldInput) {
    this.host = host;
    this.input = input;
    this.still.addEventListener("change", this.onMotionPref);
    document.addEventListener("scroll", this.onScroll, { capture: true, passive: true });
    window.addEventListener("resize", this.redrawSoon);
    // Content can grow, or a view can change without telling us: check cheaply.
    this.poll = window.setInterval(() => {
      const panel = this.activePanel();
      if (panel && panel.id !== this.panel?.id) this.viewChanged();
      else if (this.shown && this.key() !== this.shown.key) this.redrawSoon();
    }, 500);
    this.viewChanged();
  }

  update(input: DotFieldInput) {
    this.input = input;
    this.redrawSoon();
  }

  /**
   * The view just changed. Call it after the new view is in the DOM but before
   * the browser paints (a layout effect): the right field is in place for that
   * very frame.
   */
  viewChanged() {
    window.clearTimeout(this.soon);
    const panel = (this.panel = this.activePanel());
    if (!panel) return;
    const cached = this.sheets.get(panel.id);
    if (cached) {
      this.swap(cached, false);
      // stale (resized, content grew, new frame)? redraw it in the background
      if (cached.key !== this.key()) this.redrawSoon();
      return;
    }
    this.guard(() => this.draw(false));
  }

  destroy() {
    window.clearInterval(this.poll);
    window.clearTimeout(this.soon);
    document.removeEventListener("scroll", this.onScroll, { capture: true });
    window.removeEventListener("resize", this.redrawSoon);
    this.still.removeEventListener("change", this.onMotionPref);
    for (const s of this.sheets.values()) { s.dead = true; this.stopMotion(s); s.el.remove(); }
    this.sheets.clear();
  }

  /** Reduced motion turned on: drop all motion. Turned off: start it for the view in front. */
  private readonly onMotionPref = () => {
    if (this.still.matches) for (const s of [...this.moving]) this.stopMotion(s);
    else if (this.shown) this.shown.buildMotion?.();
  };

  private stopMotion(sheet: Sheet) {
    sheet.el.classList.remove("is-alive");
    if (sheet.motion) {
      for (const a of sheet.motion.animations) a.cancel();
      sheet.motion.el.remove();
      sheet.motion = null;
    }
    this.moving = this.moving.filter((s) => s !== sheet);
  }

  /** Remember that `sheet` holds motion; drop it from the views shown longest ago. */
  private noteMotion(sheet: Sheet) {
    this.moving = this.moving.filter((s) => s !== sheet).concat(sheet);
    for (const s of this.moving.slice(0, Math.max(0, this.moving.length - KEEP_MOTION))) {
      if (s !== this.shown) this.stopMotion(s);
    }
  }

  /** A background must never take the app down: a failed draw keeps what's there. */
  private guard(fn: () => void) {
    try { fn(); } catch (err) { console.warn("[dotfield] draw failed", err); }
  }

  /** The tall layer moves with the view; nothing redraws while scrolling. */
  private follow() {
    this.host.style.transform = `translate3d(0, ${-(this.panel?.scrollTop ?? 0)}px, 0)`;
  }
  private readonly onScroll = (event: Event) => { if (event.target === this.panel) this.follow(); };
  private readonly redrawSoon = () => {
    window.clearTimeout(this.soon);
    this.soon = window.setTimeout(() => this.guard(() => this.draw(true)), 150);
  };

  private activePanel() {
    return document.querySelector<HTMLElement>(".view-panel.active");
  }

  private key() {
    const p = this.activePanel();
    const posters = p ? p.querySelectorAll("img").length : 0;
    const known = p ? [...p.querySelectorAll("img")].filter((img) => this.colours.has(img.currentSrc || img.src)).length : 0;
    return [innerWidth, innerHeight, devicePixelRatio, p?.id, p?.scrollHeight, posters, known, this.input.frameUrl, this.frames.has(this.input.frameUrl ?? ""), this.input.dot.join()].join("|");
  }

  /**
   * Put `sheet` in front. A view change cuts (same frame, no fade); an update in
   * the same view crossfades and then drops the sheet it replaced.
   */
  private swap(sheet: Sheet, fade: boolean) {
    const old = this.shown;
    if (old === sheet) { this.follow(); return; }
    sheet.el.hidden = false;
    if (fade && old) {
      sheet.el.classList.remove("is-cut");
      void sheet.el.offsetWidth; // start the fade from transparent
      sheet.el.classList.add("is-front");
      old.el.classList.remove("is-cut", "is-front");
      window.setTimeout(() => { if (this.shown !== old) old.el.hidden = true; }, 340);
    } else {
      sheet.el.classList.add("is-cut", "is-front");
      if (old) { old.el.classList.add("is-cut"); old.el.classList.remove("is-front"); old.el.hidden = true; }
    }
    this.shown = sheet;
    this.follow();
    if (sheet.motion) this.noteMotion(sheet);
    else sheet.buildMotion?.();
  }

  /** Remember `sheet` as this view's field, dropping whatever it replaces and the oldest views. */
  private keep(id: string, sheet: Sheet) {
    const prev = this.sheets.get(id);
    this.sheets.delete(id);
    this.sheets.set(id, sheet);
    if (prev && prev !== sheet) {
      prev.dead = true;
      window.setTimeout(() => { this.stopMotion(prev); prev.el.remove(); }, 400);
    }
    while (this.sheets.size > KEEP_VIEWS) {
      const [oldId, oldest] = this.sheets.entries().next().value as [string, Sheet];
      if (oldest === this.shown) break;
      oldest.dead = true;
      this.stopMotion(oldest);
      oldest.el.remove();
      this.sheets.delete(oldId);
    }
  }

  /** Read a poster's colour (or the frame) in the background, then redraw once for the batch. */
  private fetch(url: string, kind: "poster" | "frame") {
    if (this.loading.has(url)) return;
    this.loading.add(url);
    void loadImage(url).then((img) => {
      this.loading.delete(url);
      if (kind === "frame") this.frames.set(url, img);
      else this.colours.set(url, img ? averageColour(img) : null);
      if (this.colours.size > 600) this.colours.delete(this.colours.keys().next().value as string);
      if (this.frames.size > 20) this.frames.delete(this.frames.keys().next().value as string);
      if (!this.loading.size) this.redrawSoon();
    });
  }

  /** Posters in the view whose colour is known; the rest start loading. */
  private clipsOn(panel: HTMLElement): Clip[] {
    const top = panel.scrollTop;
    const out: Clip[] = [];
    for (const img of panel.querySelectorAll("img")) {
      const r = img.getBoundingClientRect();
      const url = img.currentSrc || img.src;
      // skip posters scrolled out of the window sideways (the rest of a reel)
      if (r.width < 40 || !url || r.right <= 0 || r.left >= innerWidth) continue;
      if (!this.colours.has(url)) { this.fetch(url, "poster"); continue; }
      const rgb = this.colours.get(url);
      if (rgb) out.push({ x: r.left + r.width / 2, y: r.top + top + r.height / 2, w: r.width, h: r.height, rgb });
    }
    return out;
  }

  /**
   * Draw the active view into a new sheet: the visible window now, then the rest
   * of the page in viewport-sized chunks. `fade` crossfades it in over the
   * current field; otherwise it cuts in for the frame being painted.
   */
  private draw(fade: boolean) {
    const panel = (this.panel = this.activePanel());
    if (!panel) return;
    const sheet = makeSheet();
    const restWx = sheet.rest.wash.getContext("2d"), restDx = sheet.rest.dots.getContext("2d");
    if (!restWx || !restDx) return; // no canvas (tests)
    sheet.key = this.key();
    this.host.append(sheet.el);
    const W = Math.max(1, innerWidth), Hv = Math.max(1, innerHeight);
    const pageH = Hv + Math.max(0, panel.scrollHeight - panel.clientHeight);
    const cols = Math.ceil(W / STEP) + 2, rows = Math.ceil(pageH / STEP) + 2;
    const dcols = Math.ceil(W / PITCH), drows = Math.ceil(pageH / PITCH);
    const { frameUrl, dot } = this.input;

    // everything read from the page, read once
    let frameImg: HTMLImageElement | null = null;
    if (frameUrl) {
      if (this.frames.has(frameUrl)) frameImg = this.frames.get(frameUrl) ?? null;
      else this.fetch(frameUrl, "frame");
    }
    const clips = this.clipsOn(panel);
    let foot: ReturnType<typeof frameLight> = null;
    if (frameImg) {
      const footKey = [frameUrl, cols, W, Hv].join("|");
      if (this.footCache?.key !== footKey) this.footCache = { key: footKey, value: frameLight(frameImg, cols, W, Hv) };
      foot = this.footCache.value;
    }
    const blooms = bloomsFor(W, Hv, pageH, panel);
    const glass = glassOver(panel);
    const bare = bareChecker(panel);
    const text = bareText(panel, bare);
    const blocks = blockers(panel, bare);
    const n = cols * rows;

    // Many posters and blooms on a long page: bucket them by the rows they can reach.
    const clipRows: Clip[][] = Array.from({ length: rows }, () => []);
    for (const c of clips) {
      const reach = 2.4 * (c.h / 2 + Hv * 0.16);
      for (let j = Math.max(0, Math.floor((c.y - reach) / STEP)); j <= Math.min(rows - 1, Math.ceil((c.y + reach) / STEP)); j++) clipRows[j].push(c);
    }
    const bucketBlooms = (list: Bloom[]) => {
      const out: Bloom[][] = Array.from({ length: rows }, () => []);
      for (const b of list) {
        const reach = 3.2 * b[2];
        for (let j = Math.max(0, Math.floor((b[1] - reach) / STEP)); j <= Math.min(rows - 1, Math.ceil((b[1] + reach) / STEP)); j++) out[j].push(b);
      }
      return out;
    };

    // calm: a broad blur of where text sits (cheap, so the whole page at once)
    const quiet = new Float32Array(n);
    if (text.length) {
      for (const r of text) {
        const i0 = Math.max(0, Math.floor(r.l / STEP)), i1 = Math.min(cols - 1, Math.ceil(r.r / STEP));
        const j0 = Math.max(0, Math.floor(r.t / STEP)), j1 = Math.min(rows - 1, Math.ceil(r.b / STEP));
        for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) quiet[j * cols + i] = 1;
      }
      blur3(quiet, cols, rows, 7);
      for (let k = 0; k < n; k++) quiet[k] = Math.min(1, quiet[k] * 4);
    }
    // no dots behind translucent controls or fading media strips, eased past their edges
    const block = new Float32Array(n);
    for (const r of blocks) {
      const i0 = Math.max(0, Math.floor((r.l - r.f) / STEP)), i1 = Math.min(cols - 1, Math.ceil((r.r + r.f) / STEP));
      const j0 = Math.max(0, Math.floor((r.t - r.f) / STEP)), j1 = Math.min(rows - 1, Math.ceil((r.b + r.f) / STEP));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const x = i * STEP, y = j * STEP;
        // distance to the rounded rect: shrink by the radius, then measure past it
        const d = Math.hypot(Math.max(r.l + r.rad - x, 0, x - r.r + r.rad), Math.max(r.t + r.rad - y, 0, y - r.b + r.rad)) - r.rad;
        const s = d >= r.f ? 0 : 1 - d / r.f;
        const k = j * cols + i;
        block[k] = Math.max(block[k], s * s * (3 - 2 * s));
      }
    }
    const glassAt = (x: number, y: number) => {
      let g = 0;
      for (const r of glass) {
        const d = Math.hypot(Math.max(r.l - x, 0, x - r.r), Math.max(r.t - y, 0, y - r.b));
        if (d < 24) g = Math.max(g, 1 - d / 24);
      }
      return g;
    };

    /**
     * One state of the light: dot light L, wash light LW, footage share Sh, footage
     * colour FC. `t` is its point in the motion loop; the still field has no travel.
     */
    interface Light { bloomRows: Bloom[][]; t: number; travel: number; cx: number; cy: number; L: Float32Array; LW: Float32Array; Sh: Float32Array; FC: Uint8ClampedArray; lit: Uint8Array }
    const makeLight = (t: number, amount: number): Light => ({
      bloomRows: bucketBlooms(amount ? driftBlooms(blooms, Hv, t) : blooms), t, travel: amount,
      cx: amount ? 0.35 * (Math.cos(2 * Math.PI * t) - 1) : 0, cy: amount ? 0.35 * Math.sin(2 * Math.PI * t) : 0,
      L: new Float32Array(n), LW: new Float32Array(n), Sh: new Float32Array(n), FC: new Uint8ClampedArray(n * 3), lit: new Uint8Array(rows),
    });
    const restLight = makeLight(0, 0);

    const lightRows = (F: Light, ja: number, jb: number) => {
      const { L, LW, Sh, FC, lit } = F;
      for (let j = Math.max(0, ja); j <= Math.min(rows - 1, jb); j++) {
        if (lit[j]) continue;
        lit[j] = 1;
        const rowClips = clipRows[j], rowBlooms = F.bloomRows[j];
        for (let i = 0; i < cols; i++) {
          const k = j * cols + i, px = i * STEP, py = j * STEP;
          const fa = accentLight(px, py, Hv, rowBlooms, F.cx, F.cy) * (clips.length ? 0.7 : 1);
          let ff = 0, cr = 0, cg = 0, cb = 0;
          if (foot && j < foot.rows) {
            ff = foot.F[k];
            cr = foot.rgb[k * 3] * ff; cg = foot.rgb[k * 3 + 1] * ff; cb = foot.rgb[k * 3 + 2] * ff;
          }
          // each clip glows just past its own edges, in its own colour
          let fm = 0;
          for (const c of rowClips) {
            const ddx = (px - c.x) / (c.w / 2 + Hv * 0.16);
            if (ddx > 2.4 || ddx < -2.4) continue;
            const ddy = (py - c.y) / (c.h / 2 + Hv * 0.16);
            const w = 0.3 * Math.exp(-(ddx * ddx + ddy * ddy) * 1.1);
            fm = 1 - (1 - fm) * (1 - w);
            cr += c.rgb[0] * w; cg += c.rgb[1] * w; cb += c.rgb[2] * w;
          }
          const fs = ff + fm;
          if (fs > 0) { FC[k * 3] = cr / fs; FC[k * 3 + 1] = cg / fs; FC[k * 3 + 2] = cb / fs; }
          const light = 1 - (1 - Math.min(1, fa)) * (1 - Math.min(1, ff)) * (1 - Math.min(1, fm * 1.2));
          // the travelling light goes on after the knee, or the knee would flatten it in the bright places
          const lift = F.travel ? 1 + F.travel * travel(px, py, Hv, F.t) : 1;
          const base = knee(Math.min(1.2, light)) * lift * (1 - 0.5 * glassAt(px, py));
          // calm: dots thin out near text and vanish behind controls; the wash barely moves
          L[k] = base * (1 - 0.6 * quiet[k]) * (1 - 0.95 * block[k]);
          LW[k] = base * (1 - 0.15 * quiet[k]) * (1 - 0.1 * block[k]);
          Sh[k] = fs + fa > 0 ? fs / (fs + fa) : 0;
        }
      }
    };

    const mix = (F: Light, k: number, out: Rgb) => {
      const s = F.Sh[k], FC = F.FC;
      out[0] = dot[0] + (FC[k * 3] - dot[0]) * s;
      out[1] = dot[1] + (FC[k * 3 + 1] - dot[1]) * s;
      out[2] = dot[2] + (FC[k * 3 + 2] - dot[2]) * s;
      return out;
    };
    const c: Rgb = [0, 0, 0];

    /** A layer's canvases, sized for the page, and the image buffers bands are written into. */
    interface Paint { wx: CanvasRenderingContext2D; dx: CanvasRenderingContext2D; washImg: ImageData; dotImg: ImageData }
    const makePaint = (layer: Layer, wx: CanvasRenderingContext2D, dx: CanvasRenderingContext2D): Paint => {
      const { wash, dots } = layer;
      wash.width = cols;
      wash.height = rows;
      wash.style.width = `${cols * STEP}px`;
      wash.style.height = `${rows * STEP}px`;
      dots.width = dcols;
      dots.height = drows;
      dots.style.width = `${dcols * PITCH}px`;
      dots.style.height = `${drows * PITCH}px`;
      return { wx, dx, washImg: wx.createImageData(cols, rows), dotImg: dx.createImageData(dcols, drows) };
    };
    const restPaint = makePaint(sheet.rest, restWx, restDx);

    /** Light, wash and dots of one state for the page between y0 and y1 (css px). */
    const paintBand = (F: Light, P: Paint, y0: number, y1: number) => {
      const ja = Math.max(0, Math.floor(y0 / STEP) - 1), jb = Math.min(rows - 1, Math.ceil(y1 / STEP) + 1);
      lightRows(F, ja, jb);
      const wd = P.washImg.data, LW = F.LW, L = F.L;
      for (let k = ja * cols; k < (jb + 1) * cols; k++) {
        const f = Math.min(1, LW[k]) * WASH;
        mix(F, k, c);
        wd[k * 4] = (c[0] * 0.4 + 26) * f; wd[k * 4 + 1] = (c[1] * 0.4 + 26) * f; wd[k * 4 + 2] = (c[2] * 0.4 + 29) * f; wd[k * 4 + 3] = 255;
      }
      P.wx.putImageData(P.washImg, 0, 0, 0, ja, cols, jb - ja + 1);
      // dots: one pixel per 4px cell; glass.css scales it pixelated and masks the gaps
      const da = Math.max(0, Math.floor(y0 / PITCH)), db = Math.min(drows - 1, Math.ceil(y1 / PITCH));
      const d = P.dotImg.data;
      for (let j = da; j <= db; j++) for (let i = 0; i < dcols; i++) {
        // bilinear light at the cell centre
        const fx = Math.min(cols - 1.001, (i * PITCH + PITCH / 2) / STEP), fy = Math.min(rows - 1.001, (j * PITCH + PITCH / 2) / STEP);
        const i0 = fx | 0, j0 = fy | 0, ax = fx - i0, ay = fy - j0, k0 = j0 * cols + i0;
        const l = (L[k0] * (1 - ax) + L[k0 + 1] * ax) * (1 - ay) + (L[k0 + cols] * (1 - ax) + L[k0 + cols + 1] * ax) * ay;
        if (l <= BAYER[j & 7][i & 7]) continue;
        const kk = DOT_LO + DOT_HI * Math.min(1, l);
        mix(F, Math.round(fy) * cols + Math.round(fx), c);
        const o = (j * dcols + i) * 4;
        d[o] = c[0] * kk; d[o + 1] = c[1] * kk; d[o + 2] = c[2] * kk; d[o + 3] = 255;
      }
      P.dx.putImageData(P.dotImg, 0, 0, 0, da, dcols, db - da + 1);
    };

    // first the window you're looking at, then show it
    const top = panel.scrollTop;
    const seen: [number, number] = [Math.max(0, top), Math.min(pageH, top + Hv)];
    paintBand(restLight, restPaint, seen[0], seen[1]);
    this.keep(panel.id, sheet);
    if (this.panel === panel) this.swap(sheet, fade);

    // then the rest of the page, nearest first, a viewport at a time; it keeps
    // going after a view change, so a remembered view is never half drawn
    const bands: [number, number][] = [];
    for (let y = seen[1]; y < pageH; y += Hv) bands.push([y, Math.min(pageH, y + Hv)]);
    for (let y = seen[0]; y > 0; y -= Hv) bands.push([Math.max(0, y - Hv), y]);
    bands.sort((a, b) => Math.min(Math.abs(a[0] - top), Math.abs(a[1] - top)) - Math.min(Math.abs(b[0] - top), Math.abs(b[1] - top)));
    /** Run steps one per task, so none of this ever holds up the app; stops if the sheet is dropped. */
    const runSteps = (steps: (() => void)[], onStop?: () => void) => {
      const next = () => {
        if (sheet.dead) { onStop?.(); return; }
        const step = steps.shift();
        if (!step) return;
        try { step(); } catch (err) { console.warn("[dotfield] band failed", err); onStop?.(); return; }
        window.setTimeout(next, 0);
      };
      window.setTimeout(next, 0);
    };

    /**
     * The always-on motion: STATES copies of the field a step further along the loop,
     * plus a copy of the first on top so the loop wraps cleanly. Built only for a view
     * that's showing, in the background, and it only starts once every state is
     * complete, so a half-drawn state can never fade in.
     */
    sheet.buildMotion = () => {
      if (sheet.dead || sheet.motion || sheet.building || this.still.matches) return;
      sheet.building = true;
      const el = document.createElement("div");
      el.className = "dot-motion";
      const layers = Array.from({ length: STATES + 1 }, () => makeLayer("dot-state"));
      const steps: (() => void)[] = [];
      for (let k = 0; k < STATES; k++) {
        let F: Light | null = null, P: Paint | null = null;
        steps.push(() => {
          const wx = layers[k].wash.getContext("2d"), dx = layers[k].dots.getContext("2d");
          if (!wx || !dx) throw new Error("no canvas");
          F = makeLight(k / STATES, SWELL);
          P = makePaint(layers[k], wx, dx);
        });
        for (const b of [seen, ...bands]) steps.push(() => { if (F && P) paintBand(F, P, b[0], b[1]); });
      }
      steps.push(() => {
        // the copy of state 0 on top
        const first = layers[0], top = layers[STATES];
        const pairs: [HTMLCanvasElement, HTMLCanvasElement][] = [[first.wash, top.wash], [first.dots, top.dots]];
        for (const [from, to] of pairs) {
          to.width = from.width;
          to.height = from.height;
          to.style.width = from.style.width;
          to.style.height = from.style.height;
          to.getContext("2d")?.drawImage(from, 0, 0);
        }
        el.append(...layers.map((l) => l.el));
        sheet.el.append(el);
        sheet.motion = { el, animations: startLoop(layers) };
        sheet.building = false;
        if (this.still.matches) { this.stopMotion(sheet); return; }
        void el.offsetWidth; // let the motion fade in over the still field
        sheet.el.classList.add("is-alive");
        this.noteMotion(sheet);
      });
      runSteps(steps, () => { sheet.building = false; });
    };

    runSteps(bands.map((b) => () => paintBand(restLight, restPaint, b[0], b[1])).concat(() => {
      if (this.shown === sheet) sheet.buildMotion?.();
    }));
  }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The dot field: Recall's dark background, drawn inside #ambient (theme/ambient.tsx).
 *
 * One ordered (Bayer) dither of light, in small square dots, filling the window
 * and fixed behind everything. It is the bottom layer and nothing more: it never
 * reads the page, never follows the scroll and never changes with the view, so
 * cards, glass and text simply sit on top of a field that stays put.
 *
 * It is lit evenly: the same level everywhere, broken up only by a soft cloud so
 * it reads as a texture rather than a grid. The dots take their colour from the
 * active session's frame, blurred across the whole window, so the colour changes
 * gently from side to side; with no frame they take the accent.
 *
 * It moves, always and slowly: soft waves of light roll across the dots as a loop
 * of pre-drawn states crossfades one into the next (see startLoop), so dots come
 * and go as the light passes. Compositor-only, stepped, off for reduced motion.
 *
 * It is drawn once and redrawn only when the window size, the accent or the frame
 * changes, so switching views and scrolling cost it nothing. Dots are one canvas
 * pixel per 4px cell, scaled up pixelated, with the gaps cut by a CSS mask (glass.css).
 */

export type Rgb = [number, number, number];

export interface DotFieldInput {
  /** The frame the top of the window is lit by (the active session's poster). */
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
const SWELL = 0.35; // how strongly the travelling light lifts and dims the field
const LEVEL = 0.3; // how lit the field is, everywhere
const GRAIN = 0.35; // how much the cloud lifts or dims it, either way
const TINT = 0.75; // how much of the frame's colour the dots take

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

/** Low-frequency cloud, about [-0.75, 0.75], so the even light reads as a texture rather than a flat grid. */
const cloud = (x: number, y: number) => 0.5 * Math.sin(x * 5.1) * Math.sin(y * 6.3 - x * 2.2) + 0.25 * Math.sin(x * 11.7 - y * 8.1 + 1.3);

/**
 * How lit the field is at (x, y), in page px, at point `t` of the motion loop.
 * The same level everywhere, varied only by the cloud (which drifts round a
 * small circle as t goes 0 → 1) and, while moving, the travelling waves. `t` = 0
 * and 1 are the same field, so the loop is seamless; `swell` 0 is the still field.
 */
export function evenLight(x: number, y: number, Hv: number, t: number, swell: number) {
  const w = 2 * Math.PI * t;
  const cx = swell ? 0.35 * (Math.cos(w) - 1) : 0, cy = swell ? 0.35 * Math.sin(w) : 0;
  const lift = swell ? 1 + swell * travel(x, y, Hv, t) : 1;
  return LEVEL * (1 + GRAIN * cloud(x / Hv + cx, y / Hv + cy)) * lift;
}

/**
 * Light travelling across the window at point `t` of the loop, in [-1, 1]: two
 * soft diagonal waves about a screen across that roll along as t goes from 0 to
 * 1. A moving state is lit by `1 + SWELL * travel()`, so the field rises and
 * falls gently as a wave passes.
 */
export function travel(x: number, y: number, Hv: number, t: number) {
  const u = x / Hv, v = y / Hv, w = 2 * Math.PI * t;
  const s = 0.6 * Math.sin(2.2 * u + 1.4 * v - w) + 0.4 * Math.sin(-1.3 * u + 2.6 * v - w + 1.1);
  return Math.max(-1, Math.min(1, s));
}

/**
 * Images load in CORS mode so their pixels can be read: the API echoes the
 * renderer's origin (apps/api/main.py). One that still taints yields no light.
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

/**
 * The frame's colour across the whole window: the image covers the light grid,
 * blurred far enough that only broad colour is left, then each cell is brought
 * to the same brightness so the frame tints the dots without lighting some parts
 * more than others. `share` is how much of that colour a cell takes: near-black
 * parts of the frame have no colour to give, so they fall back to the accent.
 */
function frameColour(img: HTMLImageElement, cols: number, rows: number) {
  const c = document.createElement("canvas");
  c.width = cols;
  c.height = rows;
  const x = c.getContext("2d", { willReadFrequently: true });
  if (!x) return null;
  x.fillStyle = "#000";
  x.fillRect(0, 0, cols, rows);
  x.filter = `blur(${Math.max(cols, rows) / 8}px) saturate(1.5)`;
  // cover, with room to spare so the blur doesn't pull black in from the edges
  const scale = 1.3 * Math.max(cols / img.naturalWidth, rows / img.naturalHeight);
  const dw = img.naturalWidth * scale, dh = img.naturalHeight * scale;
  x.drawImage(img, (cols - dw) / 2, (rows - dh) / 2, dw, dh);
  let px: Uint8ClampedArray;
  try { px = x.getImageData(0, 0, cols, rows).data; } catch { return null; }
  const share = new Float32Array(cols * rows);
  const rgb = new Uint8ClampedArray(cols * rows * 3);
  for (let k = 0; k < cols * rows; k++) {
    const r = px[k * 4], g = px[k * 4 + 1], b = px[k * 4 + 2];
    const val = Math.max(r, g, b) / 255;
    share[k] = TINT * Math.min(1, val / 0.12);
    const norm = val > 0.02 ? 200 / (val * 255) : 0;
    rgb[k * 3] = r * norm; rgb[k * 3 + 1] = g * norm; rgb[k * 3 + 2] = b * norm;
  }
  return { share, rgb };
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
 * One drawn field, shown as a unit. `rest` is the field as drawn, shown at once.
 * `motion` is the always-on movement: STATES copies of the field with the light a
 * step further along its loop, built in the background, then crossfaded one into
 * the next (see startLoop).
 */
interface Sheet {
  el: HTMLDivElement;
  rest: Layer;
  motion: { el: HTMLDivElement; animations: Animation[] } | null;
  /** Builds `motion` in the background. */
  buildMotion: (() => void) | null;
  building: boolean;
  /** What it was drawn for (see DotField.key); a mismatch means redraw. */
  key: string;
  /** Set when the sheet is replaced, so its remaining steps stop. */
  dead: boolean;
}

function makeSheet(): Sheet {
  const el = document.createElement("div");
  el.className = "dot-sheet";
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

/** Run steps one per task, so none of this ever holds up the app; stops once `sheet` is replaced. */
function runSteps(sheet: Sheet, steps: (() => void)[], onStop?: () => void) {
  const next = () => {
    if (sheet.dead) { onStop?.(); return; }
    const step = steps.shift();
    if (!step) return;
    try { step(); } catch (err) { console.warn("[dotfield] step failed", err); onStop?.(); return; }
    window.setTimeout(next, 0);
  };
  window.setTimeout(next, 0);
}

/**
 * The field behind the whole app. One sheet the size of the window: drawn on
 * mount, redrawn (and crossfaded in) only when the window size, the accent or
 * the frame changes. Views come and go on top of it without touching it.
 */
export class DotField {
  private readonly host: HTMLElement;
  private input: DotFieldInput;
  private shown: Sheet | null = null;
  private soon = 0;
  private readonly frames = new Map<string, HTMLImageElement | null>();
  private readonly loading = new Set<string>();
  private readonly still = matchMedia("(prefers-reduced-motion: reduce)");

  constructor(host: HTMLElement, input: DotFieldInput) {
    this.host = host;
    this.input = input;
    this.still.addEventListener("change", this.onMotionPref);
    window.addEventListener("resize", this.redrawSoon);
    this.guard(() => this.draw(false));
  }

  update(input: DotFieldInput) {
    this.input = input;
    if (this.shown?.key !== this.key()) this.redrawSoon();
  }

  destroy() {
    window.clearTimeout(this.soon);
    window.removeEventListener("resize", this.redrawSoon);
    this.still.removeEventListener("change", this.onMotionPref);
    if (this.shown) { this.drop(this.shown); this.shown = null; }
    this.host.replaceChildren();
  }

  /** Reduced motion turned on: drop the motion. Turned off: build it again. */
  private readonly onMotionPref = () => {
    if (!this.shown) return;
    if (this.still.matches) this.stopMotion(this.shown);
    else this.shown.buildMotion?.();
  };

  private stopMotion(sheet: Sheet) {
    sheet.el.classList.remove("is-alive");
    if (sheet.motion) {
      for (const a of sheet.motion.animations) a.cancel();
      sheet.motion.el.remove();
      sheet.motion = null;
    }
  }

  private drop(sheet: Sheet) {
    sheet.dead = true;
    this.stopMotion(sheet);
    sheet.el.remove();
  }

  /** A background must never take the app down: a failed draw keeps what's there. */
  private guard(fn: () => void) {
    try { fn(); } catch (err) { console.warn("[dotfield] draw failed", err); }
  }

  private readonly redrawSoon = () => {
    window.clearTimeout(this.soon);
    this.soon = window.setTimeout(() => {
      if (this.shown?.key !== this.key()) this.guard(() => this.draw(true));
    }, 150);
  };

  private key() {
    const { frameUrl, dot } = this.input;
    return [innerWidth, innerHeight, frameUrl, this.frames.has(frameUrl ?? ""), dot.join()].join("|");
  }

  /** Load the frame in the background, then redraw once it's in. */
  private fetchFrame(url: string) {
    if (this.loading.has(url)) return;
    this.loading.add(url);
    void loadImage(url).then((img) => {
      this.loading.delete(url);
      this.frames.set(url, img);
      if (this.frames.size > 20) this.frames.delete(this.frames.keys().next().value as string);
      if (this.input.frameUrl === url) this.redrawSoon();
    });
  }

  /** Put `sheet` in front: cut in on the first draw, crossfade over the old field after that. */
  private swap(sheet: Sheet, fade: boolean) {
    const old = this.shown;
    this.shown = sheet;
    if (fade && old) {
      void sheet.el.offsetWidth; // start the fade from transparent
      sheet.el.classList.add("is-front");
      window.setTimeout(() => this.drop(old), 400);
    } else {
      sheet.el.classList.add("is-cut", "is-front");
      if (old) this.drop(old);
    }
  }

  private draw(fade: boolean) {
    const sheet = makeSheet();
    const restWx = sheet.rest.wash.getContext("2d"), restDx = sheet.rest.dots.getContext("2d");
    if (!restWx || !restDx) return; // no canvas (tests)
    sheet.key = this.key();
    const W = Math.max(1, innerWidth), Hv = Math.max(1, innerHeight);
    const cols = Math.ceil(W / STEP) + 2, rows = Math.ceil(Hv / STEP) + 2;
    const dcols = Math.ceil(W / PITCH), drows = Math.ceil(Hv / PITCH);
    const n = cols * rows;
    const { frameUrl, dot } = this.input;

    let tint: ReturnType<typeof frameColour> = null;
    if (frameUrl) {
      const img = this.frames.get(frameUrl);
      if (img) tint = frameColour(img, cols, rows);
      else if (img === undefined) this.fetchFrame(frameUrl);
    }

    /** One state of the light (dots and wash alike), `t` along the loop. */
    interface Light { L: Float32Array }
    const light = (t: number, swell: number): Light => {
      const L = new Float32Array(n);
      for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) L[j * cols + i] = evenLight(i * STEP, j * STEP, Hv, t, swell);
      return { L };
    };

    // colour per light cell is the same in every state: the frame's, else the accent
    const colour = new Uint8ClampedArray(n * 3);
    for (let k = 0; k < n; k++) {
      const s = tint ? tint.share[k] : 0;
      for (let ch = 0; ch < 3; ch++) colour[k * 3 + ch] = dot[ch] + ((tint ? tint.rgb[k * 3 + ch] : dot[ch]) - dot[ch]) * s;
    }
    const c: Rgb = [0, 0, 0];
    const mix = (k: number) => {
      c[0] = colour[k * 3]; c[1] = colour[k * 3 + 1]; c[2] = colour[k * 3 + 2];
      return c;
    };

    /** Size `layer` to the window and paint state `F` into it. */
    const paint = (F: Light, layer: Layer, wx: CanvasRenderingContext2D, dx: CanvasRenderingContext2D) => {
      const { wash, dots } = layer;
      wash.width = cols; wash.height = rows;
      wash.style.width = `${cols * STEP}px`; wash.style.height = `${rows * STEP}px`;
      dots.width = dcols; dots.height = drows;
      dots.style.width = `${dcols * PITCH}px`; dots.style.height = `${drows * PITCH}px`;
      const washImg = wx.createImageData(cols, rows), wd = washImg.data;
      for (let k = 0; k < n; k++) {
        const f = Math.min(1, F.L[k]) * WASH;
        mix(k);
        wd[k * 4] = (c[0] * 0.4 + 26) * f; wd[k * 4 + 1] = (c[1] * 0.4 + 26) * f; wd[k * 4 + 2] = (c[2] * 0.4 + 29) * f; wd[k * 4 + 3] = 255;
      }
      wx.putImageData(washImg, 0, 0);
      // dots: one pixel per 4px cell; glass.css scales it pixelated and masks the gaps
      const dotImg = dx.createImageData(dcols, drows), d = dotImg.data, L = F.L;
      for (let j = 0; j < drows; j++) for (let i = 0; i < dcols; i++) {
        // bilinear light at the cell centre
        const fx = Math.min(cols - 1.001, (i * PITCH + PITCH / 2) / STEP), fy = Math.min(rows - 1.001, (j * PITCH + PITCH / 2) / STEP);
        const i0 = fx | 0, j0 = fy | 0, ax = fx - i0, ay = fy - j0, k0 = j0 * cols + i0;
        const l = (L[k0] * (1 - ax) + L[k0 + 1] * ax) * (1 - ay) + (L[k0 + cols] * (1 - ax) + L[k0 + cols + 1] * ax) * ay;
        if (l <= BAYER[j & 7][i & 7]) continue;
        const kk = DOT_LO + DOT_HI * Math.min(1, l);
        mix(Math.round(fy) * cols + Math.round(fx));
        const o = (j * dcols + i) * 4;
        d[o] = c[0] * kk; d[o + 1] = c[1] * kk; d[o + 2] = c[2] * kk; d[o + 3] = 255;
      }
      dx.putImageData(dotImg, 0, 0);
    };

    paint(light(0, 0), sheet.rest, restWx, restDx);
    this.host.append(sheet.el);
    this.swap(sheet, fade);

    /**
     * The always-on motion: STATES copies of the field a step further along the
     * loop, plus a copy of the first on top so the loop wraps cleanly. Built in
     * the background, one state per task, and started only once every state is
     * complete, so a half-drawn state can never fade in.
     */
    sheet.buildMotion = () => {
      if (sheet.dead || sheet.motion || sheet.building || this.still.matches) return;
      sheet.building = true;
      const el = document.createElement("div");
      el.className = "dot-motion";
      const layers = Array.from({ length: STATES + 1 }, () => makeLayer("dot-state"));
      const steps: (() => void)[] = layers.slice(0, STATES).map((layer, k) => () => {
        const wx = layer.wash.getContext("2d"), dx = layer.dots.getContext("2d");
        if (!wx || !dx) throw new Error("no canvas");
        paint(light(k / STATES, SWELL), layer, wx, dx);
      });
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
      });
      runSteps(sheet, steps, () => { sheet.building = false; });
    };
    sheet.buildMotion();
  }
}

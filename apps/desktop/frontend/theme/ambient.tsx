// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The background and the glass optics, as three pieces mounted once at the
 * app root:
 *
 * - <GlassDefs/>   the SVG edge-lens filter (#lg-lens) that .glass-float and
 *                  .glass-read refract through: neutral in the middle, bending
 *                  toward the rim. R carries horizontal shift, G vertical.
 * - <AmbientGlow/> the background. Dark: the dot field (theme/dotfield.ts), a
 *                  fixed layer lit by `image` and the accent. Light: black
 *                  ink on paper keeps the blurred wash of `image`, cross-fading
 *                  between two layers when the image changes, plus grain.
 * - useGlassHighlight() one passive pointer listener that feeds --mx / --my to
 *                  whichever glass surface the cursor is over.
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { accentDotRgb, type AccentChoice } from "./accent";
import { DotField } from "./dotfield";

const GLASS_SELECTOR = ".glass, .glass-float, .glass-read, .project-dialog, .settings-dialog, .hardware-setup-dialog, .queue-rail, .queue-detail";

function lensMapUri() {
  const ramp = (id: string, x2: number, y2: number, lo: string, mid: string, hi: string) =>
    `<linearGradient id='${id}' x1='0' y1='0' x2='${x2}' y2='${y2}'><stop offset='0' stop-color='${lo}'/><stop offset='.16' stop-color='${mid}'/><stop offset='.84' stop-color='${mid}'/><stop offset='1' stop-color='${hi}'/></linearGradient>`;
  const svg =
    `<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200' preserveAspectRatio='none'><defs>` +
    ramp("r", 1, 0, "rgb(0,0,0)", "rgb(128,0,0)", "rgb(255,0,0)") +
    ramp("g", 0, 1, "rgb(0,0,0)", "rgb(0,128,0)", "rgb(0,255,0)") +
    `</defs><rect width='200' height='200' fill='url(%23r)'/><rect width='200' height='200' fill='url(%23g)' style='mix-blend-mode:screen'/></svg>`;
  return "data:image/svg+xml;utf8," + svg.replace(/</g, "%3C").replace(/>/g, "%3E");
}

const LENS_MAP = lensMapUri();

export function GlassDefs() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true" focusable="false">
      <defs>
        <filter id="lg-lens" x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
          <feImage href={LENS_MAP} x="0" y="0" width="100%" height="100%" preserveAspectRatio="none" result="map" />
          <feGaussianBlur in="map" stdDeviation="6" result="soft" />
          <feDisplacementMap in="SourceGraphic" in2="soft" scale="42" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </defs>
    </svg>
  );
}

/**
 * The background. `image` is any URL the renderer can load (a clip poster, a
 * session frame); undefined leaves the last one in place rather than flashing
 * to black between views.
 */
export function AmbientGlow({ image, accent, theme }: { image?: string; accent: AccentChoice; theme: string }) {
  const [held, setHeld] = useState(image);
  useEffect(() => { if (image) setHeld(image); }, [image]);
  useEffect(() => {
    if (held) document.documentElement.style.setProperty("--amb-url", `url("${held}")`);
  }, [held]);
  return theme === "light" ? <FootageGlow image={image} /> : <DotGround image={image ?? held} accent={accent} />;
}

/**
 * The fixed field under everything. It doesn't know or care which view is open:
 * only the frame and the accent reach it. The first draw is a layout effect so
 * the field is there in the app's very first frame.
 */
function DotGround({ image, accent }: { image?: string; accent: AccentChoice }) {
  const host = useRef<HTMLDivElement>(null);
  const field = useRef<DotField | null>(null);
  const dot = accentDotRgb(accent);
  const dotKey = dot.join();

  useLayoutEffect(() => {
    if (!host.current) return;
    field.current = new DotField(host.current, { frameUrl: image, dot });
    return () => { field.current?.destroy(); field.current = null; };
    // created once; later changes go through update() below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => { field.current?.update({ frameUrl: image, dot }); }, [image, dotKey]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div id="ambient" aria-hidden="true">
      <div className="amb-dots" ref={host} />
    </div>
  );
}

function FootageGlow({ image }: { image?: string }) {
  const [layers, setLayers] = useState<[string | undefined, string | undefined]>([image, undefined]);
  const [front, setFront] = useState(0);
  const current = layers[front];

  useEffect(() => {
    if (!image || image === current) return;
    const back = front === 0 ? 1 : 0;
    setLayers((prev) => {
      const next: [string | undefined, string | undefined] = [prev[0], prev[1]];
      next[back] = image;
      return next;
    });
    setFront(back);
  }, [image, current, front]);

  return (
    <div id="ambient" aria-hidden="true">
      {layers.map((url, index) => (
        <div
          key={index}
          className={`amb-layer ${index === front && url ? "is-front" : ""}`}
          style={url ? { backgroundImage: `url("${url}")` } : undefined}
        />
      ))}
      <div className="amb-grain" />
    </div>
  );
}

export function useGlassHighlight() {
  const frame = useRef(0);
  useEffect(() => {
    let pending: PointerEvent | null = null;
    const flush = () => {
      frame.current = 0;
      const ev = pending;
      pending = null;
      if (!ev) return;
      const target = ev.target as Element | null;
      const el = target?.closest?.(GLASS_SELECTOR) as HTMLElement | null;
      if (!el) return;
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${ev.clientX - r.left}px`);
      el.style.setProperty("--my", `${ev.clientY - r.top}px`);
    };
    const onMove = (ev: PointerEvent) => {
      pending = ev;
      if (!frame.current) frame.current = requestAnimationFrame(flush);
    };
    document.addEventListener("pointermove", onMove, { passive: true });
    return () => {
      document.removeEventListener("pointermove", onMove);
      if (frame.current) cancelAnimationFrame(frame.current);
    };
  }, []);
}

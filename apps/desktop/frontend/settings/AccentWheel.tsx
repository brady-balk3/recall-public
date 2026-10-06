// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The one color choice Recall offers. The wheel is hue; Intensity is chroma
 * (grey to vivid); Brightness is lightness. In black and white the sliders are
 * disabled so they can't invent a color; touching the wheel or a preset turns
 * the accent on. Arrow keys turn the wheel (Shift for bigger steps).
 */
import { useRef, type CSSProperties } from "react";
import { Play } from "../lib/icons";
import { ACCENT_PRESETS, accentHex, accentInk, accentOklch, sameAccent, type Accent, type AccentChoice } from "../theme/accent";

const DEFAULT_ON: Accent = { hue: 32, intensity: 70, brightness: 64 };
const RING = Array.from({ length: 13 }, (_, k) => `oklch(.74 .17 ${k * 30})`).join(", ");

export function AccentWheel({ value, onChange }: { value: AccentChoice; onChange: (next: AccentChoice) => void }) {
  const wheel = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const on = value !== null;
  const accent = value ?? DEFAULT_ON;

  const hueAt = (event: React.PointerEvent) => {
    const r = wheel.current!.getBoundingClientRect();
    const x = event.clientX - (r.left + r.width / 2);
    const y = event.clientY - (r.top + r.height / 2);
    // The ring starts at 3 o'clock (conic "from 90deg"), so atan2 lines up directly.
    return Math.round((Math.atan2(y, x) * 180 / Math.PI + 360) % 360);
  };
  const setHue = (hue: number) => {
    const base = value ?? { ...DEFAULT_ON, intensity: Math.max(DEFAULT_ON.intensity, 30) };
    onChange({ ...base, hue });
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    const d = ({ ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 } as Record<string, number>)[event.key];
    if (!d) return;
    event.preventDefault();
    setHue((accent.hue + d * (event.shiftKey ? 15 : 3) + 360) % 360);
  };

  const color = on ? accentOklch(accent) : "#fff";
  return (
    <div className={`accent ${on ? "" : "is-mono"}`}>
      <div
        ref={wheel}
        className="wheel"
        role="slider"
        tabIndex={0}
        aria-label="Accent hue"
        aria-valuemin={0}
        aria-valuemax={359}
        aria-valuenow={Math.round(accent.hue)}
        aria-valuetext={on ? `${Math.round(accent.hue)} degrees` : "Black and white"}
        style={{ "--ring": `conic-gradient(from 90deg, ${RING})` } as CSSProperties}
        onPointerDown={(event) => { dragging.current = true; wheel.current?.setPointerCapture(event.pointerId); setHue(hueAt(event)); }}
        onPointerMove={(event) => { if (dragging.current) setHue(hueAt(event)); }}
        onPointerUp={() => { dragging.current = false; }}
        onPointerCancel={() => { dragging.current = false; }}
        onKeyDown={onKeyDown}
      >
        <span className="wheel-knob" style={{ "--a": `${accent.hue}deg`, background: color } as CSSProperties} />
        <div className="wheel-core" style={{ background: color }}>
          <b className="num" style={{ color: on ? accentInk(accent) : "#000" }}>{on ? accentHex(accent) : "B&W"}</b>
        </div>
      </div>
      <div className="accent-side">
        <label className="acc-row">
          <span>Intensity<small>grey to vivid</small></span>
          <input
            type="range"
            min={0}
            max={100}
            value={accent.intensity}
            disabled={!on}
            onChange={(event) => onChange({ ...accent, intensity: Number(event.target.value) })}
            style={{ "--track": `linear-gradient(90deg, ${accentOklch({ ...accent, intensity: 0 })}, ${accentOklch({ ...accent, intensity: 100 })})` } as CSSProperties}
          />
        </label>
        <label className="acc-row">
          <span>Brightness<small>deep to light</small></span>
          <input
            type="range"
            min={0}
            max={100}
            value={accent.brightness}
            disabled={!on}
            onChange={(event) => onChange({ ...accent, brightness: Number(event.target.value) })}
            style={{ "--track": `linear-gradient(90deg, ${accentOklch({ ...accent, brightness: 0 })}, ${accentOklch({ ...accent, brightness: 100 })})` } as CSSProperties}
          />
        </label>
        {!on && <small className="t3 acc-hint">Pick a color on the wheel or a preset to tune it.</small>}
        <div className="acc-presets" role="radiogroup" aria-label="Accent presets">
          {ACCENT_PRESETS.map((preset) => {
            const selected = sameAccent(value, preset.accent);
            return (
              <button
                key={preset.id}
                type="button"
                role="radio"
                aria-checked={selected}
                aria-label={preset.label}
                title={preset.label}
                className={`acc-dot ${selected ? "on" : ""}`}
                style={{ background: preset.accent ? accentOklch(preset.accent) : "linear-gradient(135deg, #fff 50%, #111 50%)" }}
                onClick={() => onChange(preset.accent)}
              />
            );
          })}
        </div>
        <div className="acc-prev" aria-hidden="true">
          <span className="btn heat sm"><Play />Review</span>
          <span className="badge heat">9 to review</span>
        </div>
      </div>
    </div>
  );
}

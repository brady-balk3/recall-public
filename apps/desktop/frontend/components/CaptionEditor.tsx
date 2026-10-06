// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useEffect, useState, type CSSProperties, type ReactNode } from "react";
import { Check } from "../lib/icons";
import type { CaptionFont, CaptionPosition, CaptionSize, CaptionStyle } from "../lib/store";
import { StudioSwitch } from "./StudioControls";
import { CAPTION_COLORS } from "../theme/accent";

const CAPTION_FONTS: CaptionFont[] = ["Arial Black", "Impact", "Arial", "Verdana", "Georgia", "Trebuchet MS", "Segoe UI Black"];
const CAPTION_SIZES: { value: CaptionSize; label: string }[] = [
  { value: "small", label: "Small" },
  { value: "medium", label: "Medium" },
  { value: "large", label: "Large" },
];
const CAPTION_POSITIONS: { value: CaptionPosition; label: string }[] = [
  { value: "top", label: "Top" },
  { value: "middle", label: "Middle" },
  { value: "bottom", label: "Bottom" },
];
const PREVIEW_SIZE_PX: Record<CaptionSize, number> = { small: 12, medium: 15, large: 19 };
const PREVIEW_JUSTIFY: Record<CaptionPosition, "flex-start" | "center" | "flex-end"> = { top: "flex-start", middle: "center", bottom: "flex-end" };
const CAPTION_TEXT_SWATCHES = ["#FFFFFF", "#F2F2F2", "#FFFF00", "#35E37D"];
const CAPTION_HIGHLIGHT_SWATCHES = CAPTION_COLORS.flatMap((color) => (color.hex ? [color.hex] : []));
const PREVIEW_WORDS = ["that", "was", "actually", "insane"];

/** Burned-in caption style editor with a live 9:16 preview. */
export function CaptionEditor({
  caption,
  onChange,
  animateHighlight = false,
  highlightControl,
}: {
  caption: CaptionStyle;
  onChange: (patch: Partial<CaptionStyle>) => void;
  animateHighlight?: boolean;
  /** Replaces the built-in highlight swatches (Settings uses the named caption colors). */
  highlightControl?: ReactNode;
}) {
  const [hot, setHot] = useState(2);
  useEffect(() => {
    if (!animateHighlight || !caption.enabled) return;
    const id = window.setInterval(() => setHot((index) => (index + 1) % PREVIEW_WORDS.length), 560);
    return () => window.clearInterval(id);
  }, [animateHighlight, caption.enabled]);
  const highlightIndex = animateHighlight ? hot : 2;

  return (
    <div className="mock-v1-caption">
      <div className="mock-v1-caption-editor">
        <div className="setrow">
          <div className="st"><b>Burn in captions</b><small>Word-by-word subtitles on every exported clip. Turn off for clean video.</small></div>
          <StudioSwitch checked={caption.enabled} onChange={(enabled) => onChange({ enabled })} label="Burn in captions" />
        </div>
        <div className="mock-v1-caption-knobs" style={{ opacity: caption.enabled ? 1 : 0.45, pointerEvents: caption.enabled ? "auto" : "none" }}>
          <label className="form-group"><span className="form-label">Font</span>
            <select className="field" value={caption.font} onChange={(event) => onChange({ font: event.target.value as CaptionFont })}>{CAPTION_FONTS.map((font) => <option key={font} value={font}>{font}</option>)}</select>
          </label>
          <div className="form-group"><span className="form-label">Size</span>
            <div className="seg" role="group" aria-label="Caption size">{CAPTION_SIZES.map((option) => <button type="button" key={option.value} className={caption.size === option.value ? "on" : ""} aria-pressed={caption.size === option.value} onClick={() => onChange({ size: option.value })}>{option.label}</button>)}</div>
          </div>
          <div className="mock-v1-caption-colors">
            <CaptionColorField label="Text color" value={caption.textColor} swatches={CAPTION_TEXT_SWATCHES} onChange={(textColor) => onChange({ textColor })} />
            {highlightControl ?? <CaptionColorField label="Highlight color" value={caption.highlightColor} swatches={CAPTION_HIGHLIGHT_SWATCHES} onChange={(highlightColor) => onChange({ highlightColor })} />}
          </div>
          <div className="form-group"><span className="form-label">Position</span>
            <div className="seg" role="group" aria-label="Caption position">{CAPTION_POSITIONS.map((option) => <button type="button" key={option.value} className={caption.position === option.value ? "on" : ""} aria-pressed={caption.position === option.value} onClick={() => onChange({ position: option.value })}>{option.label}</button>)}</div>
          </div>
        </div>
      </div>
      <div className="mock-v1-caption-preview-wrap">
        <div className="mock-v1-caption-preview" style={{ justifyContent: PREVIEW_JUSTIFY[caption.position] }}>
          {caption.enabled ? (
            <div className="mock-v1-caption-preview-text" style={{ fontFamily: `"${caption.font}", system-ui, sans-serif`, fontSize: PREVIEW_SIZE_PX[caption.size] }}>
              {PREVIEW_WORDS.map((word, index) => <span key={index} style={{ color: index === highlightIndex ? caption.highlightColor : caption.textColor }}>{word}{index < PREVIEW_WORDS.length - 1 ? " " : ""}</span>)}
            </div>
          ) : (
            <span className="mock-v1-caption-off">Captions off</span>
          )}
        </div>
        <span className="form-description">Live preview</span>
      </div>
    </div>
  );
}

function CaptionColorField({ label, value, swatches, onChange }: { label: string; value: string; swatches: string[]; onChange: (value: string) => void }) {
  const normalized = value.toUpperCase();
  return <div className="form-group settings-color-field">
    <span className="form-label">{label}</span>
    <div className="settings-color-swatches" role="group" aria-label={`${label} presets`}>
      {swatches.map((color) => <button type="button" key={color} className={normalized === color ? "is-selected" : ""} aria-pressed={normalized === color} aria-label={`${label} ${color}`} style={{ "--swatch": color } as CSSProperties} onClick={() => onChange(color)}>{normalized === color && <Check size={12} />}</button>)}
    </div>
    <label className="settings-custom-color"><span>{normalized}</span><input type="color" value={value} aria-label={`Custom ${label.toLowerCase()}`} onChange={(event) => onChange(event.target.value)} /></label>
  </div>;
}

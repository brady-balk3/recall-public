// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useEffect, useState, type ButtonHTMLAttributes, type CSSProperties, type ReactNode, type RefObject } from "react";
import { Check, Volume2, VolumeX } from "../lib/icons";
import { useJobStore, type ExportLayout } from "../lib/store";
import { ACCENT_PRESETS, accentHex, sameAccent, type AccentChoice } from "../theme/accent";

type ButtonTone = "primary" | "secondary" | "ghost" | "success" | "danger";

export function StudioButton({
  tone = "secondary",
  loading = false,
  success = false,
  icon,
  children,
  className = "",
  disabled,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  tone?: ButtonTone;
  loading?: boolean;
  success?: boolean;
  icon?: ReactNode;
}) {
  // The page kit's pill button: primary is the heat pill, danger is solid red.
  const look = tone === "primary" ? "heat" : tone === "ghost" ? "ghost" : tone === "danger" ? "danger solid" : tone === "success" ? "is-keep" : "";
  return (
    <button
      type="button"
      className={`btn ${look} ${loading ? "is-loading" : ""} ${className}`.replace(/\s+/g, " ").trim()}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...props}
    >
      {loading ? <span className="btn-spinner" aria-hidden="true" /> : success ? <Check aria-hidden="true" /> : icon}
      <span>{children}</span>
    </button>
  );
}

export function StateIcon({ icon, tone = "accent", animate = true, className = "" }: { icon: ReactNode; tone?: "accent" | "success" | "warning" | "danger"; animate?: boolean; className?: string }) {
  return <span className={`studio-state-icon tone-${tone} ${animate ? "is-animated" : ""} ${className}`.trim()} aria-hidden="true">{icon}</span>;
}

export function StudioSwitch({ checked, onChange, label, disabled = false }: { checked: boolean; onChange: (checked: boolean) => void; label: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      className="studio-switch"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    ><span aria-hidden="true" /></button>
  );
}

export function StudioSlider({
  value,
  min,
  max,
  step,
  onChange,
  label,
  className = "",
  ticks,
  output,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  label: string;
  className?: string;
  ticks?: string[];
  output?: string;
}) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <div className={`studio-slider ${className}`.trim()} style={{ "--studio-slider-fill": `${pct}%` } as CSSProperties}>
      <div className="studio-slider-line">
        <input type="range" min={min} max={max} step={step} value={value} aria-label={label} onChange={(event) => onChange(Number(event.currentTarget.value))} />
        <span className="studio-slider-track" aria-hidden="true"><i /></span>
        <span className="studio-slider-thumb" aria-hidden="true" />
      </div>
      {(ticks || output) && <div className="studio-slider-meta" aria-hidden="true">{ticks?.map((tick, index) => <span key={`${tick}-${index}`}>{tick}</span>)}{output && <output>{output}</output>}</div>}
    </div>
  );
}

export function useMediaVolume(videoRefs: RefObject<HTMLVideoElement | null>[], syncKey?: unknown) {
  const volume = useJobStore((state) => state.settings.mediaVolume);
  const muted = useJobStore((state) => state.settings.mediaMuted);
  const setSettings = useJobStore((state) => state.setSettings);

  useEffect(() => {
    videoRefs.forEach((ref) => {
      if (!ref.current) return;
      ref.current.volume = volume;
      ref.current.muted = muted;
    });
  }, [volume, muted, syncKey]);

  const setVolume = (next: number) => {
    const clamped = Math.max(0, Math.min(1, next));
    setSettings({ mediaVolume: clamped, mediaMuted: clamped === 0 });
  };

  return {
    volume,
    muted,
    setVolume,
    toggleMuted: () => setSettings({ mediaMuted: !muted }),
  };
}

export function MediaVolumeControl({
  volume,
  muted,
  onVolumeChange,
  onToggleMuted,
  className = "",
  orientation = "horizontal",
  collapsible = false,
}: {
  volume: number;
  muted: boolean;
  onVolumeChange: (value: number) => void;
  onToggleMuted: () => void;
  className?: string;
  orientation?: "horizontal" | "vertical";
  collapsible?: boolean;
}) {
  const [expanded, setExpanded] = useState(!collapsible);
  const audibleVolume = muted ? 0 : volume;
  const showSlider = !collapsible || expanded;
  const vertical = orientation === "vertical";

  return (
    <div
      className={[
        "studio-volume",
        vertical ? "is-vertical" : "",
        collapsible ? "is-collapsible" : "",
        showSlider ? "is-expanded" : "",
        className,
      ].filter(Boolean).join(" ")}
      onMouseEnter={() => { if (collapsible) setExpanded(true); }}
      onMouseLeave={() => { if (collapsible) setExpanded(false); }}
      onFocusCapture={() => { if (collapsible) setExpanded(true); }}
      onBlurCapture={(event) => {
        if (!collapsible) return;
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setExpanded(false);
      }}
    >
      <button
        type="button"
        onClick={onToggleMuted}
        aria-label={muted ? "Unmute preview" : "Mute preview"}
        aria-expanded={collapsible ? showSlider : undefined}
      >
        {muted || volume === 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
      </button>
      {showSlider && (
        <>
          <StudioSlider
            value={audibleVolume}
            min={0}
            max={1}
            step={0.01}
            onChange={onVolumeChange}
            label="Preview volume"
            className={vertical ? "is-vertical" : ""}
          />
          <output>{Math.round(audibleVolume * 100)}</output>
        </>
      )}
    </div>
  );
}

export type ChoiceOption<T extends string> = { value: T; label: string; description: string; badge?: string; icon?: ReactNode };

export function ChoiceCards<T extends string>({ value, options, onChange, label, compact = false, className = "" }: { value: T; options: ChoiceOption<T>[]; onChange: (value: T) => void; label: string; compact?: boolean; className?: string }) {
  return (
    <div className={`optgrid choice-cards ${options.length > 2 ? "is-3" : ""} ${compact ? "is-compact" : ""} ${className}`.replace(/\s+/g, " ").trim()} role="radiogroup" aria-label={label}>
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button type="button" role="radio" aria-checked={selected} key={option.value} className={`opt ${selected ? "on" : ""}`} onClick={() => onChange(option.value)}>
            <b>
              {option.icon && <span className="opt-icon" aria-hidden="true">{option.icon}</span>}
              {option.label}
              {option.badge && <span className={`badge ${option.badge === "Recommended" ? "heat" : ""}`}>{option.badge}</span>}
            </b>
            <small>{option.description}</small>
          </button>
        );
      })}
    </div>
  );
}

/** Facecam composition picker. Shared by Settings and first-run onboarding. */
export function ExportLayoutChoices({ value, onChange }: { value: ExportLayout; onChange: (value: ExportLayout) => void }) {
  const options: { value: ExportLayout; label: string; copy: string; preview: string }[] = [
    { value: "auto", label: "Automatic", copy: "Recall uses the detected scene geometry.", preview: "auto" },
    { value: "vertical_split", label: "Stacked", copy: "Facecam above gameplay in a clean split.", preview: "stacked" },
    { value: "gameplay_pip", label: "Gameplay PiP", copy: "Gameplay fills the frame with bounded facecam.", preview: "pip" },
  ];
  return <div className="settings-layout-choices" role="radiogroup" aria-label="Facecam layout">
    {options.map((option) => <button type="button" role="radio" aria-checked={value === option.value} key={option.value} className={value === option.value ? "is-selected" : ""} onClick={() => onChange(option.value)}>
      <span className={`settings-layout-preview is-${option.preview}`} aria-hidden="true"><i className="is-game" /><i className="is-face" /></span>
      <span><strong>{option.label}</strong><small>{option.copy}</small></span>
      <i className="settings-layout-check" aria-hidden="true"><Check size={10} /></i>
    </button>)}
  </div>;
}

/** Theme picker. Shared by Settings and first-run onboarding. */
export function ThemeChoices({ value, onChange }: { value: "dark" | "light"; onChange: (value: "dark" | "light") => void }) {
  return <div className="settings-theme-choices" role="radiogroup" aria-label="Theme">
    {(["dark", "light"] as const).map((theme) => <button type="button" role="radio" aria-checked={value === theme} key={theme} className={value === theme ? "is-selected" : ""} onClick={() => onChange(theme)}>
      <span className={`settings-theme-preview is-${theme}`} aria-hidden="true"><i /><b /><em /></span>
      <span><strong>{theme === "dark" ? "Dark studio" : "Light studio"}</strong><small>{theme === "dark" ? "For dim editing rooms" : "For brighter workspaces"}</small></span>
      <i className="settings-layout-check" aria-hidden="true"><Check size={10} /></i>
    </button>)}
  </div>;
}

/** Accent presets. The full color wheel lives in Settings > Look. Meaning colors never change. */
export function AccentChoices({ value, onChange }: { value: AccentChoice; onChange: (value: AccentChoice) => void }) {
  return <div className="settings-accent-choices" role="radiogroup" aria-label="Accent color">
    {ACCENT_PRESETS.map((preset) => {
      const selected = sameAccent(value, preset.accent);
      const swatch = preset.accent ? accentHex(preset.accent) : "linear-gradient(135deg, #fff 50%, #111 50%)";
      return <button type="button" role="radio" aria-checked={selected} aria-label={preset.label} title={preset.label} key={preset.id} className={selected ? "is-selected" : ""} style={{ "--swatch": swatch } as CSSProperties} onClick={() => onChange(preset.accent)}>
        <span className="settings-accent-swatch" aria-hidden="true"><i /></span>
        <strong>{preset.label}</strong>
        <i className="settings-accent-check" aria-hidden="true"><Check size={10} /></i>
      </button>;
    })}
  </div>;
}

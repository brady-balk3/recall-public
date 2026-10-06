// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * First run, in six steps: welcome, your name (the only required answer), a
 * real check of this PC, your look, how clips come out, and a scan bar to
 * start the first stream. Everything here also lives in Settings.
 *
 * Full-window and opaque, with its own footage glow, like Theater.
 */
import { useEffect, useRef, useState, type RefObject } from "react";
import { HardwareProbeCard, useHardwareCheck } from "../components/HardwareSetup";
import { ExportLayoutChoices, StudioSwitch } from "../components/StudioControls";
import { ChevronRight, Folder } from "../lib/icons";
import { markHardwareSetupComplete } from "../lib/hardware";
import { markOnboardingComplete } from "../lib/onboarding";
import { DEFAULT_CAPTION_STYLE, useJobStore, type CaptionStyle } from "../lib/store";
import { ScanBar } from "../home/ScanBar";
import { AccentWheel } from "../settings/AccentWheel";
import { AvatarPicker } from "../settings/AvatarPicker";
import { RecallMark } from "../components/RecallMark";

const STEPS = ["Welcome", "Your name", "Your PC", "Your look", "Your clips", "Ready"] as const;
const PERF = [
  { value: "background", label: "Responsive" },
  { value: "balanced", label: "Balanced" },
  { value: "max", label: "Full speed" },
] as const;

const persistLocal = (key: string, value: string) => { try { localStorage.setItem(key, value); } catch {} };

export function OnboardingFlow({
  open,
  apiEndpoint,
  dialogRef,
  onComplete,
  onScanLink,
  onChooseFile,
}: {
  open: boolean;
  apiEndpoint: string;
  dialogRef: RefObject<HTMLDivElement | null>;
  onComplete: () => void;
  onScanLink?: (url: string) => void;
  onChooseFile?: () => void;
}) {
  const settings = useJobStore((state) => state.settings);
  const setSettings = useJobStore((state) => state.setSettings);
  const [step, setStep] = useState(0);
  const hardwareMarked = useRef(false);
  const nameRef = useRef<HTMLInputElement>(null);
  const { state: hwState, capabilities, error: hwError, slow: hwSlow, check: recheck } = useHardwareCheck(apiEndpoint, open);
  const caption: CaptionStyle = settings.captionStyle ?? DEFAULT_CAPTION_STYLE;
  const name = settings.displayName;
  const nameOk = name.trim().length > 0;
  const checking = hwState === "checking";

  useEffect(() => {
    if (step === 1) nameRef.current?.focus();
  }, [step]);

  const markHardware = () => {
    if (hardwareMarked.current) return;
    hardwareMarked.current = true;
    markHardwareSetupComplete(capabilities);
  };

  const setName = (value: string) => {
    setSettings({ displayName: value });
    persistLocal("recall-display-name", value);
  };

  const finish = (skippedFrom?: number) => {
    // The name is the one thing Recall needs; skipping without it lands on the name step.
    if (!nameOk) { setStep(1); return; }
    markHardware();
    markOnboardingComplete(skippedFrom == null ? [] : STEPS.slice(skippedFrom).map((label) => label.toLowerCase().replace(/\s+/g, "-")));
    onComplete();
  };

  const canContinue = step === 1 ? nameOk : step === 2 ? !checking || hwSlow : true;
  const next = () => {
    if (!canContinue) return;
    if (step === 2) markHardware();
    if (step >= STEPS.length - 1) finish();
    else setStep(step + 1);
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (event.key !== "Enter" || target?.tagName === "BUTTON" || target?.closest(".scanbar")) return;
      event.preventDefault();
      next();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!open) return null;

  const cta = step === 0 ? "Get started" : step === 2 && checking ? (hwSlow ? "Continue without waiting" : "Checking…") : "Continue";
  const title = [
    <>Hours of stream in.<br />The best bits out.</>,
    "What should we call you?",
    checking ? "Checking your PC…" : hwState === "ready" ? "You're all set up." : "Let's get this PC ready.",
    "Black and white, or bring a color.",
    "How your clips come out.",
    `Let's find your best moments, ${name.trim() || "friend"}.`,
  ][step];
  const lede = [
    "Recall watches your VOD the way your chat did, finds where the room popped off, and hands you vertical, captioned clips to keep or skip. It all runs on this PC.",
    "Just a name for the greeting and your avatar. It stays on this PC.",
    "Recall picks the fastest safe route for this machine. Choose how hard it can push while you're gaming or streaming.",
    "One accent marks everything that matters: buttons, curves, the keep. Change it any time in Settings.",
    "Recall frames your facecam and gameplay on its own. Pick a default layout and whether captions are burned in.",
    "Paste a Twitch VOD link or pick a recording, and your first deck will be waiting when you're back.",
  ][step];

  return (
    <div className="onb glow-opaque" ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="onb-title" aria-describedby="onb-lede" tabIndex={-1}>
      <div className="amb-grain" aria-hidden="true" />
      <div className="onb-brand"><RecallMark size={26} /><span className="disp">Recall</span></div>
      <div className="onb-main">
        <div className="onb-l">
          <span className="o-kick">{["Welcome to Recall", "First things first", "Your PC", "Your look", "Your clips", "You're ready"][step]}</span>
          <h1 className="disp" id="onb-title">{title}</h1>
          <p id="onb-lede">{lede}</p>
          {step === 1 && (
            <>
              <input
                ref={nameRef}
                className="field o-name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Your name or stream name"
                maxLength={32}
                aria-label="Your name"
              />
              {!nameOk && <small className="t3">Needed to continue</small>}
            </>
          )}
          {step === 2 && (
            <div className="seg" role="radiogroup" aria-label="While scanning">
              {PERF.map((option) => (
                <button key={option.value} type="button" role="radio" aria-checked={settings.performanceProfile === option.value} onClick={() => setSettings({ performanceProfile: option.value })}>
                  {option.label}
                </button>
              ))}
            </div>
          )}
          {step === 3 && (
            <div className="seg" role="radiogroup" aria-label="Theme">
              {(["dark", "light"] as const).map((theme) => (
                <button key={theme} type="button" role="radio" aria-checked={settings.theme === theme} onClick={() => setSettings({ theme })}>{theme === "dark" ? "Dark" : "Light"}</button>
              ))}
            </div>
          )}
          {step === 4 && (
            <div className="o-clip-opts">
              <ExportLayoutChoices value={settings.exportLayout} onChange={(exportLayout) => { setSettings({ exportLayout }); persistLocal("recall-export-layout", exportLayout); }} />
              <div className="o-toggle">
                <b>Burn in captions</b>
                <StudioSwitch
                  checked={caption.enabled}
                  label="Burn in captions"
                  onChange={(enabled) => {
                    const nextStyle = { ...caption, enabled };
                    setSettings({ captionStyle: nextStyle });
                    persistLocal("recall-caption-style", JSON.stringify(nextStyle));
                  }}
                />
              </div>
            </div>
          )}
        </div>

        <div className="onb-r">
          {step === 0 && <div className="o-hero" aria-hidden="true"><RecallMark size={220} /></div>}
          {step === 1 && (
            <div className="o-av">
              <AvatarPicker name={name || "?"} size={150} />
              <b>{name.trim() || "Your name"}</b>
              <small className="t3">This is how you'll show up in Recall. A photo is optional, and it stays on this PC.</small>
            </div>
          )}
          {step === 2 && <div className="o-hw"><HardwareProbeCard state={hwState} capabilities={capabilities} error={hwError} onRecheck={() => recheck()} /></div>}
          {step === 3 && <div className="o-look"><AccentWheel value={settings.accent} onChange={(accent) => setSettings({ accent })} /></div>}
          {step === 4 && (
            <div className="o-clip" aria-hidden="true">
              <div className={`o-frame is-${settings.exportLayout}`}><i className="cam" /><i className="game" /></div>
              {caption.enabled && <div className="cap cap-face">that was not empty</div>}
            </div>
          )}
          {step === 5 && (
            <div className="o-go">
              <ScanBar
                onScanLink={(url) => { finish(); onScanLink?.(url); }}
                onChooseFile={() => { finish(); onChooseFile?.(); }}
              />
              <button type="button" className="btn ghost" onClick={() => { finish(); onChooseFile?.(); }}><Folder aria-hidden="true" />Choose a recording instead</button>
            </div>
          )}
        </div>
      </div>

      <div className="onb-foot">
        <div className="onb-steps" aria-hidden="true">
          {STEPS.map((label, i) => <span key={label} className={i === step ? "on" : i < step ? "done" : ""} />)}
        </div>
        <span className="t3 onb-count">{step + 1} of {STEPS.length} · {STEPS[step]}</span>
        <span className="sp" />
        {step < STEPS.length - 1 && (
          <button type="button" className="btn ghost" onClick={() => finish(step)} disabled={step === 1 && !nameOk} title={step === 1 && !nameOk ? "Add your name first" : undefined}>
            Skip the rest
          </button>
        )}
        {step > 0 && <button type="button" className="btn" onClick={() => setStep(step - 1)}>Back</button>}
        {step < STEPS.length - 1 ? (
          <button type="button" className="btn heat" onClick={next} disabled={!canContinue}>{cta}<ChevronRight aria-hidden="true" /></button>
        ) : (
          <button type="button" className="btn heat" onClick={() => finish()}>Go to Home<ChevronRight aria-hidden="true" /></button>
        )}
      </div>
    </div>
  );
}

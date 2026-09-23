// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useEffect, useRef, useState, type RefObject } from "react";
import { CaptionEditor } from "./CaptionEditor";
import { HardwareProbeCard, useHardwareCheck } from "./HardwareSetup";
import { AccentChoices, ChoiceCards, StudioButton, ThemeChoices, type ChoiceOption } from "./StudioControls";
import { ArrowLeft, Check, ChevronRight, Download, Info } from "../lib/icons";
import { markHardwareSetupComplete } from "../lib/hardware";
import { markOnboardingComplete } from "../lib/onboarding";
import { DEFAULT_CAPTION_STYLE, useJobStore, type CaptionStyle, type ExportLayout } from "../lib/store";

type OnboardingStep = "welcome" | "how" | "machine" | "you" | "clips";

const STEPS: { id: OnboardingStep; label: string; hint: string; skippable: boolean; cta: string }[] = [
  { id: "welcome", label: "Welcome", hint: "What Recall does", skippable: false, cta: "Show me how it works" },
  { id: "how", label: "How it works", hint: "The four-step loop", skippable: false, cta: "Got it — set me up" },
  { id: "machine", label: "Your machine", hint: "System check and scans", skippable: false, cta: "Continue" },
  { id: "you", label: "You", hint: "Name, theme, color", skippable: true, cta: "Continue" },
  { id: "clips", label: "Clips", hint: "Layout and captions", skippable: true, cta: "Finish setup" },
];

const PROCESSING_CHOICES: ChoiceOption<"fast" | "quality">[] = [
  { value: "fast", label: "Smart scan", description: "Focuses deep analysis on the moments most likely to matter.", badge: "Faster" },
  { value: "quality", label: "Best quality", description: "Scans the full recording closely and takes longer.", badge: "Thorough" },
];
const PERFORMANCE_CHOICES: ChoiceOption<"background" | "balanced" | "max">[] = [
  { value: "background", label: "Keep PC responsive", description: "Quieter resource use while you work, play, or stream.", badge: "Quietest" },
  { value: "balanced", label: "Balanced", description: "Good scan speed without taking over the machine.", badge: "Recommended" },
  { value: "max", label: "Full speed", description: "Finishes sooner with heavier CPU and GPU use.", badge: "Fastest" },
];
const SPEED_MULT = { fast: 1, quality: 2.2 };
const PERF_MULT = { background: 1.7, balanced: 1, max: 0.72 };
const PERF_LOAD = { background: 2, balanced: 3, max: 5 };
const SPEED_NAME = { fast: "Smart scan", quality: "Best quality" };
const PERF_NAME = { background: "Keep PC responsive", balanced: "Balanced", max: "Full speed" };
const LAYOUT_NAME: Record<ExportLayout, string> = { auto: "Automatic", vertical_split: "Stacked", gameplay_pip: "Gameplay PiP" };

const BEATS: { title: string; copy: string; mini: "point" | "scan" | "review" | "export" }[] = [
  { title: "Point it at a recording", copy: "Drop in a local file, paste a Twitch VOD link, or queue a batch to run overnight.", mini: "point" },
  { title: "It scans while you do something else", copy: "Audio, chat, faces, and on-screen events become one reaction curve. You set how fast it runs on the next screen.", mini: "scan" },
  { title: "Review in the Theater", copy: "Every candidate plays full-screen with its evidence. Keep, pass, or nudge the in/out points a second at a time.", mini: "review" },
  { title: "Export the keepers", copy: "Selected clips render vertical with your facecam layout and captions, straight into a folder.", mini: "export" },
];

function persistLocal(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* non-fatal */ }
}

function shouldAnimateHandoff() {
  if (typeof window.matchMedia !== "function") return false;
  return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function formatEta(minutes: number) {
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const rest = String(minutes % 60).padStart(2, "0");
    return { value: `${hours}`, unit: `h ${rest}m` };
  }
  return { value: String(minutes), unit: "min" };
}

export function OnboardingFlow({
  open,
  apiEndpoint,
  dialogRef,
  onComplete,
}: {
  open: boolean;
  apiEndpoint: string;
  dialogRef: RefObject<HTMLDivElement | null>;
  onComplete: () => void;
}) {
  const { settings, setSettings } = useJobStore();
  const [step, setStep] = useState<OnboardingStep>("welcome");
  const [furthest, setFurthest] = useState(0);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [beat, setBeat] = useState(0);
  const [phase, setPhase] = useState<"flow" | "leaving" | "handoff">("flow");
  const hardwareMarked = useRef(false);
  const { state: hwState, capabilities, error: hwError, slow: hwSlow, check: recheck } = useHardwareCheck(apiEndpoint, open);
  const caption: CaptionStyle = settings.captionStyle ?? DEFAULT_CAPTION_STYLE;
  const accentTheme = settings.accentTheme || "ember";

  const index = STEPS.findIndex((entry) => entry.id === step);
  const current = STEPS[index];
  const machineBlocked = step === "machine" && hwState === "checking" && !hwSlow;
  const gpu = capabilities?.route === "nvidia_cuda";

  const ensureHardwareMarked = () => {
    if (hardwareMarked.current) return;
    hardwareMarked.current = true;
    markHardwareSetupComplete(capabilities);
  };

  const patchCaption = (patch: Partial<CaptionStyle>) => {
    const next = { ...caption, ...patch };
    setSettings({ captionStyle: next });
    persistLocal("recall-caption-style", JSON.stringify(next));
  };

  const go = (nextIndex: number) => {
    const clamped = Math.max(0, Math.min(STEPS.length - 1, nextIndex));
    setStep(STEPS[clamped].id);
    setFurthest((value) => Math.max(value, clamped));
  };

  const finish = (alsoSkipped: string[] = []) => {
    ensureHardwareMarked();
    markOnboardingComplete([...skipped, ...alsoSkipped]);
    if (!shouldAnimateHandoff()) {
      onComplete();
      return;
    }
    setPhase("leaving");
    window.setTimeout(() => setPhase("handoff"), 380);
    window.setTimeout(onComplete, 2500);
  };

  const advance = () => {
    if (step === "machine") ensureHardwareMarked();
    if (index === STEPS.length - 1) finish();
    else go(index + 1);
  };

  const skip = () => {
    const rest = STEPS.slice(index).map((entry) => entry.id);
    setSkipped((currentSkipped) => [...currentSkipped, ...rest]);
    finish(rest);
  };

  useEffect(() => {
    if (!open || phase !== "flow") return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName);
      if (event.key === "Enter" && !machineBlocked && !typing && target?.tagName !== "BUTTON") {
        event.preventDefault();
        if (step === "machine") ensureHardwareMarked();
        if (index === STEPS.length - 1) finish();
        else go(index + 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, phase, machineBlocked, step, index]);

  if (!open) return null;

  const etaMinutes = hwState === "checking"
    ? null
    : Math.round((gpu ? 20 : 68) * SPEED_MULT[settings.processingMode] * PERF_MULT[settings.performanceProfile]);
  const eta = etaMinutes == null ? null : formatEta(etaMinutes);
  const loadBars = PERF_LOAD[settings.performanceProfile];
  const name = settings.displayName.trim() || "Creator";
  const handoffSkipped = skipped.length > 0;

  return (
    <div className="hardware-setup-modal open">
      {phase !== "handoff" && (
        <div
          ref={dialogRef}
          className={`onboarding-stage ${phase === "leaving" ? "is-leaving" : ""}`}
          role="dialog"
          aria-modal="true"
          aria-labelledby="onboarding-step-title"
          aria-describedby="onboarding-step-description"
          tabIndex={-1}
        >
          <aside className="onboarding-rail-col">
            <div className="onboarding-brand">
              <img className="brand-mark" src={`${import.meta.env.BASE_URL}recall_logo_mark_${accentTheme}.png`} alt="" draggable={false} />
              <span>RECALL</span>
              <span className="onboarding-brand-badge">SETUP</span>
            </div>
            <ol className="onboarding-steps" aria-label={`Setup step ${index + 1} of ${STEPS.length}`}>
              {STEPS.map((entry, position) => (
                <li
                  key={entry.id}
                  className={[
                    position < index ? "is-done" : "",
                    position === index ? "is-current" : "",
                    position <= furthest ? "is-clickable" : "",
                  ].filter(Boolean).join(" ")}
                  aria-current={position === index ? "step" : undefined}
                >
                  <button type="button" disabled={position > furthest} onClick={() => go(position)}>
                    <span className="onboarding-dot" aria-hidden="true">{position < index ? <Check size={10} /> : position + 1}</span>
                    <span className="onboarding-lbl"><b>{entry.label}</b><small>{entry.hint}</small></span>
                  </button>
                </li>
              ))}
            </ol>
            <div className="onboarding-rail-foot">
              <div className="onboarding-rail-progress" aria-hidden="true"><i style={{ width: `${(index / (STEPS.length - 1)) * 100}%` }} /></div>
              <p>Under two minutes. Every choice here also lives in Settings.</p>
            </div>
          </aside>

          <main className="onboarding-pane">
            <div className="onboarding-pane-body" key={step}>
              {step === "welcome" && <WelcomeScreen />}
              {step === "how" && <HowScreen beat={beat} onBeat={setBeat} />}
              {step === "machine" && (
                <MachineScreen
                  hwState={hwState}
                  capabilities={capabilities}
                  hwError={hwError}
                  hwSlow={hwSlow}
                  onRecheck={recheck}
                  eta={eta}
                  gpu={gpu}
                  loadBars={loadBars}
                />
              )}
              {step === "you" && <YouScreen />}
              {step === "clips" && (
                <ClipsScreen
                  caption={caption}
                  onCaption={patchCaption}
                  layout={settings.exportLayout}
                  onLayout={(exportLayout) => {
                    setSettings({ exportLayout });
                    persistLocal("recall-export-layout", exportLayout);
                  }}
                />
              )}
            </div>
            <footer className="onboarding-pane-foot">
              <StudioButton tone="ghost" icon={<ArrowLeft size={14} />} onClick={() => go(index - 1)} style={{ visibility: index === 0 ? "hidden" : "visible" }}>Back</StudioButton>
              <span className="onboarding-spacer" />
              <StudioButton tone="ghost" onClick={skip}>{index >= 2 ? "Skip the rest" : "Skip setup"}</StudioButton>
              <StudioButton data-autofocus={step === "clips" || undefined} tone="primary" disabled={machineBlocked} icon={step === "clips" ? <Check size={14} /> : <ChevronRight size={14} />} onClick={advance}>
                {current.cta}
              </StudioButton>
            </footer>
          </main>
        </div>
      )}

      {phase === "handoff" && (
        <div className="onboarding-handoff" role="status" aria-live="polite">
          <div className="onboarding-seal" aria-hidden="true">
            <svg viewBox="0 0 92 92"><circle className="onboarding-seal-ring" cx="46" cy="46" r="42" /></svg>
            <span className="onboarding-seal-tick"><Check size={34} /></span>
          </div>
          <h2>{handoffSkipped ? `Ready when you are, ${name}` : `You're set, ${name}`}</h2>
          <p>{handoffSkipped ? "Defaults applied — change anything in Settings." : "Opening your studio…"}</p>
          <div className="onboarding-handoff-sum">
            <span>{hwState === "checking" ? "System check pending" : gpu ? "NVIDIA acceleration" : "CPU processing"}</span>
            <span>{SPEED_NAME[settings.processingMode]}</span>
            <span>{PERF_NAME[settings.performanceProfile]}</span>
            <span>{LAYOUT_NAME[settings.exportLayout]}</span>
            <span>{caption.enabled ? `Captions · ${caption.font}` : "Captions off"}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function WelcomeScreen() {
  return (
    <div className="onboarding-hero">
      <div>
        <p className="onboarding-eyebrow">Welcome to Recall</p>
        <h1 className="onboarding-title" id="onboarding-step-title">Six hours of stream.<br />Twelve clips worth posting.</h1>
        <p className="onboarding-lede" id="onboarding-step-description">Recall watches your VOD the way your chat did — it tracks the reaction curve, finds where the room actually popped off, and hands you cut, captioned, vertical clips to approve or bin.</p>
        <p className="onboarding-lede">It runs entirely on your machine. Nothing uploads.</p>
      </div>
      <div className="onboarding-hero-art" aria-hidden="true">
        <svg viewBox="0 0 560 130" preserveAspectRatio="none">
          <defs>
            <linearGradient id="onboarding-hg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity=".28" />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path className="onboarding-grid-l" d="M0 34H560M0 65H560M0 96H560" />
          <path className="onboarding-curve-fill" d="M0 103C40 98 62 84 96 89C132 94 148 59 182 66C214 73 224 26 258 39C288 49 300 84 336 72C372 59 386 91 420 80C452 70 470 29 500 40C528 50 540 75 560 64V130H0Z" />
          <path className="onboarding-curve" d="M0 103C40 98 62 84 96 89C132 94 148 59 182 66C214 73 224 26 258 39C288 49 300 84 336 72C372 59 386 91 420 80C452 70 470 29 500 40C528 50 540 75 560 64" />
          <g className="onboarding-pick"><circle cx="258" cy="39" r="5" fill="var(--accent)" /><rect x="248" y="14" width="20" height="100" fill="none" stroke="var(--accent)" strokeOpacity=".4" strokeDasharray="3 3" /></g>
          <g className="onboarding-pick"><circle cx="500" cy="40" r="5" fill="var(--accent)" /><rect x="490" y="14" width="20" height="100" fill="none" stroke="var(--accent)" strokeOpacity=".4" strokeDasharray="3 3" /></g>
          <g className="onboarding-pick"><circle cx="182" cy="66" r="4" fill="var(--accent-2)" /></g>
        </svg>
      </div>
      <div className="onboarding-value-row">
        <div className="onboarding-value">
          <span className="onboarding-value-ico" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 15c3 0 3-8 6-8s3 10 6 10 3-6 6-6" /></svg></span>
          <b>Reaction, not volume</b>
          <span>Audio spikes, chat velocity, your face, and what's on screen — scored together.</span>
        </div>
        <div className="onboarding-value">
          <span className="onboarding-value-ico" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="3" /><path d="M9 12l2 2 4-4" /></svg></span>
          <b>You stay the editor</b>
          <span>Recall proposes. You keep, cut, and trim. It learns your taste as you go.</span>
        </div>
        <div className="onboarding-value">
          <span className="onboarding-value-ico" aria-hidden="true"><Download size={14} /></span>
          <b>Post-ready output</b>
          <span>Vertical framing, facecam layout, word-by-word captions burned in.</span>
        </div>
      </div>
    </div>
  );
}

function HowScreen({ beat, onBeat }: { beat: number; onBeat: (index: number) => void }) {
  return (
    <>
      <p className="onboarding-eyebrow">How Recall works</p>
      <h2 className="onboarding-title is-h2" id="onboarding-step-title">Four steps, every time</h2>
      <p className="onboarding-lede" id="onboarding-step-description">This is the whole loop. Once you know it you never have to think about it again.</p>
      <div className="onboarding-flow">
        {BEATS.map((entry, position) => (
          <button type="button" key={entry.title} className={`onboarding-beat ${beat === position ? "on" : ""}`} onClick={() => onBeat(position)}>
            <span className="onboarding-beat-n">{String(position + 1).padStart(2, "0")}</span>
            <span><h3>{entry.title}</h3><p>{entry.copy}</p></span>
            <span className="onboarding-mini" aria-hidden="true"><BeatMini kind={entry.mini} /></span>
          </button>
        ))}
      </div>
      <div className="onboarding-tip"><Info size={15} aria-hidden="true" /><span>Keeping and passing clips isn't just filing — it's the training signal. After a couple of sessions Recall's picks start looking like yours.</span></div>
    </>
  );
}

function MachineScreen({
  hwState,
  capabilities,
  hwError,
  hwSlow,
  onRecheck,
  eta,
  gpu,
  loadBars,
}: {
  hwState: "checking" | "ready" | "error";
  capabilities: ReturnType<typeof useHardwareCheck>["capabilities"];
  hwError: string;
  hwSlow: boolean;
  onRecheck: () => void;
  eta: { value: string; unit: string } | null;
  gpu: boolean;
  loadBars: number;
}) {
  const { settings, setSettings } = useJobStore();
  return (
    <>
      <p className="onboarding-eyebrow">Step 1 · System check required</p>
      <h2 className="onboarding-title is-h2" id="onboarding-step-title">Your machine, and how scans run on it</h2>
      <p className="onboarding-lede" id="onboarding-step-description">Recall checks what this PC can do, then you set how deeply it looks and how much of the machine it may take.</p>
      <HardwareProbeCard state={hwState} capabilities={capabilities} error={hwError} onRecheck={onRecheck} />
      <div className="onboarding-eta">
        <div className="onboarding-eta-big">{eta ? <>{eta.value}<small>{eta.unit}</small></> : "—"}</div>
        <div className="onboarding-eta-copy">
          <b>Estimated scan time for a 2-hour VOD</b>
          <span>{hwState === "checking" ? (hwSlow ? "Still checking. The first check can take a few minutes; you can continue and Recall finishes it in the background." : "Waiting on the system check…") : `${SPEED_NAME[settings.processingMode]} · ${PERF_NAME[settings.performanceProfile]} · ${gpu ? "NVIDIA acceleration" : "CPU processing"}`}</span>
          <div className="onboarding-load-meter" aria-hidden="true">{[0, 1, 2, 3, 4].map((bar) => <i key={bar} className={bar < loadBars ? "on" : ""} />)}</div>
        </div>
      </div>
      <div className="onboarding-section">
        <div className="onboarding-sec-head"><b>Scan speed</b><span>Smart scan looks closely at the moments that matter. Best quality scans the whole VOD and takes longer.</span></div>
        <ChoiceCards className="onboarding-choices-2" value={settings.processingMode} options={PROCESSING_CHOICES} onChange={(processingMode) => setSettings({ processingMode })} label="Scan speed" />
      </div>
      <div className="onboarding-section">
        <div className="onboarding-sec-head"><b>While scanning</b><span>How much of this PC a scan may use. Keep PC responsive is for scanning while you stream or play; Full speed is for when you step away.</span></div>
        <ChoiceCards className="onboarding-choices-3" value={settings.performanceProfile} options={PERFORMANCE_CHOICES} onChange={(performanceProfile) => setSettings({ performanceProfile })} label="While scanning" />
      </div>
      <div className="onboarding-tip"><Info size={15} aria-hidden="true" /><span>Scans survive a closed window. Queue a batch on Full speed before bed and review them in the morning.</span></div>
    </>
  );
}

function YouScreen() {
  const { settings, setSettings } = useJobStore();
  return (
    <>
      <p className="onboarding-eyebrow">Step 2 · Optional</p>
      <h2 className="onboarding-title is-h2" id="onboarding-step-title">Make it yours</h2>
      <p className="onboarding-lede" id="onboarding-step-description">Cosmetic only. All of it lives in Settings → Appearance.</p>
      <div className="onboarding-fields">
        <label className="form-group">
          <span className="form-label">What should we call you?</span>
          <input
            className="input-text"
            data-autofocus
            value={settings.displayName}
            placeholder="Creator"
            maxLength={32}
            onChange={(event) => {
              setSettings({ displayName: event.target.value });
              persistLocal("recall-display-name", event.target.value);
            }}
          />
          <span className="form-description">Used for the greeting on your Start screen. Leave it blank to be “Creator”.</span>
        </label>
        <div className="form-group">
          <span className="form-label">Theme</span>
          <ThemeChoices value={settings.theme} onChange={(theme) => setSettings({ theme })} />
        </div>
        <div className="form-group">
          <span className="form-label">Studio color</span>
          <AccentChoices value={settings.accentTheme} onChange={(accentTheme) => setSettings({ accentTheme })} />
          <span className="form-description">A warm, low-chroma accent for the parts of Recall that help you act and navigate. Status colors stay consistent — try one, the whole studio recolors live.</span>
        </div>
      </div>
    </>
  );
}

function ClipsScreen({
  caption,
  onCaption,
  layout,
  onLayout,
}: {
  caption: CaptionStyle;
  onCaption: (patch: Partial<CaptionStyle>) => void;
  layout: ExportLayout;
  onLayout: (value: ExportLayout) => void;
}) {
  return (
    <>
      <p className="onboarding-eyebrow">Step 3 · Optional · Last one</p>
      <h2 className="onboarding-title is-h2" id="onboarding-step-title">How your clips come out</h2>
      <p className="onboarding-lede" id="onboarding-step-description">The default shape of every clip Recall renders, and the captions burned into it. All of it is also in Settings → Export &amp; Captions.</p>
      <div className="onboarding-section">
        <div className="onboarding-sec-head"><b>Facecam layout</b><span>Automatic keeps one composition across the VOD: stacked for context-heavy games, PiP for action-forward gameplay. You can still lock either style.</span></div>
        <OnboardingLayoutChoices value={layout} onChange={onLayout} />
      </div>
      <div className="onboarding-section">
        <CaptionEditor caption={caption} onChange={onCaption} animateHighlight />
      </div>
    </>
  );
}

function OnboardingLayoutChoices({ value, onChange }: { value: ExportLayout; onChange: (value: ExportLayout) => void }) {
  const options: { value: ExportLayout; label: string; copy: string; preview: "auto" | "stacked" | "pip" }[] = [
    { value: "auto", label: "Automatic", copy: "Recall uses the detected scene geometry.", preview: "auto" },
    { value: "vertical_split", label: "Stacked", copy: "Facecam above gameplay in a clean split.", preview: "stacked" },
    { value: "gameplay_pip", label: "Gameplay PiP", copy: "Gameplay fills the frame with bounded facecam.", preview: "pip" },
  ];
  return (
    <div className="onboarding-layouts" role="radiogroup" aria-label="Facecam layout">
      {options.map((option) => (
        <button type="button" role="radio" aria-checked={value === option.value} key={option.value} className={value === option.value ? "is-selected" : ""} onClick={() => onChange(option.value)}>
          <span className="onboarding-layout-frame" aria-hidden="true"><LayoutPreview kind={option.preview} /></span>
          <span><b>{option.label}</b><small>{option.copy}</small></span>
        </button>
      ))}
    </div>
  );
}

function LayoutPreview({ kind }: { kind: "auto" | "stacked" | "pip" }) {
  if (kind === "stacked") {
    return (
      <svg viewBox="0 0 90 130">
        <rect width="90" height="130" fill="var(--elevated)" />
        <rect x="4" y="4" width="82" height="50" rx="3" fill="var(--accent-soft)" stroke="var(--accent-border)" />
        <circle cx="45" cy="29" r="12" fill="var(--accent)" opacity=".5" />
        <rect x="4" y="58" width="82" height="68" rx="3" fill="var(--track)" stroke="var(--border)" />
        <rect x="20" y="112" width="50" height="6" rx="3" fill="var(--accent-2)" opacity=".75" />
      </svg>
    );
  }
  if (kind === "pip") {
    return (
      <svg viewBox="0 0 90 130">
        <rect width="90" height="130" fill="var(--elevated)" />
        <rect x="4" y="4" width="82" height="122" rx="3" fill="var(--track)" stroke="var(--border)" />
        <rect x="52" y="10" width="30" height="26" rx="4" fill="var(--accent-soft)" stroke="var(--accent)" />
        <circle cx="67" cy="23" r="7" fill="var(--accent)" opacity=".5" />
        <rect x="20" y="112" width="50" height="6" rx="3" fill="var(--accent-2)" opacity=".75" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 90 130">
      <rect width="90" height="130" fill="var(--elevated)" />
      <rect x="4" y="4" width="82" height="44" rx="3" fill="var(--accent-soft)" stroke="var(--accent-border)" />
      <circle cx="45" cy="26" r="10" fill="var(--accent)" opacity=".5" />
      <rect x="4" y="52" width="82" height="74" rx="3" fill="var(--track)" stroke="var(--border)" />
      <rect x="56" y="58" width="26" height="20" rx="3" fill="var(--accent-soft)" stroke="var(--accent-border)" strokeDasharray="2.5 2.5" />
      <path d="M45 90h-8m0 0l3-3m-3 3l3 3M45 100h8m0 0l-3-3m3 3l-3 3" stroke="var(--accent)" strokeWidth="1.6" fill="none" strokeLinecap="round" />
      <rect x="20" y="112" width="50" height="6" rx="3" fill="var(--accent-2)" opacity=".75" />
    </svg>
  );
}

function BeatMini({ kind }: { kind: "point" | "scan" | "review" | "export" }) {
  if (kind === "scan") {
    return <svg viewBox="0 0 156 52"><rect x="8" y="7" width="140" height="38" fill="none" stroke="var(--border)" /><path d="M8 40C22 36 28 24 42 27C56 30 62 11 78 16C95 21 99 36 115 31C131 26 137 19 148 22" fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" /><circle cx="78" cy="16" r="3.5" fill="var(--accent)" /></svg>;
  }
  if (kind === "review") {
    return <svg viewBox="0 0 156 52"><rect x="46" y="7" width="40" height="38" rx="4" fill="var(--accent-soft)" stroke="var(--accent)" /><rect x="12" y="13" width="26" height="26" rx="3" fill="none" stroke="var(--border-strong)" /><rect x="94" y="13" width="26" height="26" rx="3" fill="none" stroke="var(--border-strong)" /><path d="M61 20l12 6-12 6z" fill="var(--accent)" /></svg>;
  }
  if (kind === "export") {
    return <svg viewBox="0 0 156 52"><rect x="18" y="8" width="22" height="34" rx="3" fill="var(--accent-soft)" stroke="var(--accent)" /><rect x="48" y="8" width="22" height="34" rx="3" fill="var(--accent-soft)" stroke="var(--accent)" /><rect x="78" y="8" width="22" height="34" rx="3" fill="none" stroke="var(--border-strong)" /><path d="M118 17v13M112 24l6 6 6-6" stroke="var(--success)" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" /><path d="M108 37h20" stroke="var(--success)" strokeWidth="1.8" strokeLinecap="round" /></svg>;
  }
  return <svg viewBox="0 0 156 52"><rect x="12" y="12" width="54" height="26" rx="4" fill="none" stroke="var(--accent)" strokeWidth="1.5" /><path d="M26 25h26M39 18v14" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" /><path d="M76 25h26" stroke="var(--text-3)" strokeWidth="1.5" strokeDasharray="3 3" /><path d="M100 21l6 4-6 4z" fill="var(--text-3)" /><rect x="114" y="15" width="30" height="20" rx="3" fill="var(--accent-soft)" stroke="var(--accent-border)" /></svg>;
}

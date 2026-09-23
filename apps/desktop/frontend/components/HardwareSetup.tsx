// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useEffect, useMemo, useState, type RefObject } from "react";
import { Check, HardDrive, Info, RefreshCw, Sparkles } from "../lib/icons";
import {
  loadHardwareCapabilities,
  markHardwareSetupComplete,
  type HardwareCapabilities,
} from "../lib/hardware";
import { StateIcon, StudioButton } from "./StudioControls";

export type CheckState = "checking" | "ready" | "error";

function gpuLabel(capabilities: HardwareCapabilities): string {
  return capabilities.gpu?.name || "No supported GPU found";
}

function routeLabel(capabilities: HardwareCapabilities): string {
  return capabilities.route === "nvidia_cuda" ? "NVIDIA acceleration" : "CPU processing";
}

function resultCopy(capabilities: HardwareCapabilities): string {
  if (capabilities.route === "nvidia_cuda") {
    return "Recall will use your GPU for supported analysis stages and automatically fall back to CPU if an accelerator cannot initialize.";
  }
  if (capabilities.reason === "nvidia_cuda_unavailable") {
    return "Recall found an NVIDIA GPU but could not initialize CUDA. CPU mode is active, so scans can still run safely.";
  }
  if (capabilities.reason === "unsupported_gpu" && capabilities.gpu) {
    const vendor = capabilities.gpu.vendor === "amd" ? "AMD" : capabilities.gpu.vendor === "intel" ? "Intel" : "This";
    return `${vendor} GPU acceleration is not supported in this alpha. Recall will process recordings with the CPU and limit workers to keep the PC responsive.`;
  }
  return "Recall did not find a supported NVIDIA accelerator. CPU mode is active and long recordings may take considerably longer.";
}

/**
 * A first check can legitimately take minutes: the engine loads its GPU
 * libraries cold, often under antivirus scanning, and a VM's virtual GPU can
 * answer slowly. So the check is never cut off, but after this long setup
 * offers to continue while it finishes. The engine picks GPU or CPU per scan
 * on its own, so continuing early does not lock anyone into CPU mode.
 */
export const HARDWARE_CHECK_SLOW_MS = 60_000;

export function useSlowAfter(running: boolean, ms = HARDWARE_CHECK_SLOW_MS): boolean {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    setSlow(false);
    if (!running) return;
    const timer = window.setTimeout(() => setSlow(true), ms);
    return () => window.clearTimeout(timer);
  }, [running, ms]);
  return slow;
}

function checkedAtLabel(value: string): string {
  const time = new Date(value);
  if (Number.isNaN(time.getTime())) return "Checked recently";
  return `Checked ${time.toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}`;
}

function HardwareFacts({ capabilities }: { capabilities: HardwareCapabilities }) {
  const workers = capabilities.recommended.worker_mode === "gpu"
    ? `${capabilities.recommended.gpu_workers} GPU workers`
    : `${capabilities.recommended.cpu_workers} CPU workers`;
  return (
    <dl className="hardware-facts">
      <div><dt>Processor</dt><dd>{capabilities.cpu.physical_cores} physical cores</dd></div>
      <div><dt>Available memory</dt><dd>{capabilities.memory.available_gb == null ? "Checked per scan" : `${capabilities.memory.available_gb.toFixed(1)} GB`}</dd></div>
      <div><dt>Processing route</dt><dd>{routeLabel(capabilities)}</dd></div>
      <div><dt>Balanced setup</dt><dd>Up to {workers}</dd></div>
    </dl>
  );
}

function HardwareDetails({ capabilities }: { capabilities: HardwareCapabilities }) {
  return (
    <details className="hardware-details">
      <summary>Technical details</summary>
      <dl>
        <div><dt>Graphics adapter</dt><dd>{gpuLabel(capabilities)}</dd></div>
        <div><dt>CUDA</dt><dd>{capabilities.acceleration.cuda_ready ? "Ready" : "Unavailable"}</dd></div>
        <div><dt>ONNX providers</dt><dd>{capabilities.acceleration.onnx_providers.join(", ")}</dd></div>
        <div><dt>Local judges</dt><dd>{capabilities.acceleration.llama_gpu === true ? "GPU offload" : capabilities.acceleration.llama_gpu === false ? "CPU" : "Checked when available"}</dd></div>
        {capabilities.acceleration.free_vram_gb != null && <div><dt>Free VRAM</dt><dd>{capabilities.acceleration.free_vram_gb.toFixed(1)} GB</dd></div>}
      </dl>
    </details>
  );
}

/**
 * The system check itself, without modal chrome. Rendered as the first step of
 * the first-run flow (see Onboarding.tsx) and by the standalone dialog below,
 * so the check reads identically wherever it runs.
 */
export function HardwareCheckStep({
  active,
  apiEndpoint,
  dialogRef,
  onComplete,
}: {
  active: boolean;
  apiEndpoint: string;
  dialogRef: RefObject<HTMLDivElement | null>;
  onComplete: () => void;
}) {
  const open = active;
  const [state, setState] = useState<CheckState>("checking");
  const [capabilities, setCapabilities] = useState<HardwareCapabilities | null>(null);
  const [error, setError] = useState("");
  const slow = useSlowAfter(open && state === "checking");

  const check = () => {
    setState("checking");
    setError("");
    loadHardwareCapabilities(apiEndpoint)
      .then((result) => {
        setCapabilities(result);
        setState("ready");
      })
      .catch((reason) => {
        setError(reason instanceof Error ? reason.message : "Recall could not check this PC right now.");
        setState("error");
      });
  };

  useEffect(() => {
    if (!open) return;
    let active = true;
    setState("checking");
    setError("");
    loadHardwareCapabilities(apiEndpoint)
      .then((result) => {
        if (!active) return;
        setCapabilities(result);
        setState("ready");
      })
      .catch((reason) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "Recall could not check this PC right now.");
        setState("error");
      });
    return () => {
      active = false;
    };
  }, [open, apiEndpoint]);

  useEffect(() => {
    if (state !== "checking") {
      window.requestAnimationFrame(() => {
        dialogRef.current?.querySelector<HTMLElement>("[data-hardware-primary]")?.focus();
      });
    }
  }, [state, dialogRef]);

  if (!open) return null;
  const accelerated = capabilities?.route === "nvidia_cuda";
  const secondaryWarning = capabilities?.warnings.find((warning) => warning.code !== capabilities.reason);
  const complete = () => {
    markHardwareSetupComplete(capabilities);
    onComplete();
  };

  return (
    <>
        {state === "checking" && (
          <div className="hardware-checking" role="status" aria-live="polite">
            <span className="hardware-check-spinner" aria-hidden="true" />
            <div>
              <h2 id="hardware-setup-title">Preparing Recall for this PC</h2>
              <p id="hardware-setup-description">Checking the processor, available memory, graphics hardware, and acceleration libraries.</p>
            </div>
            <small>Nothing will be installed or downloaded.</small>
          </div>
        )}
        {state === "checking" && slow && (
          <footer className="hardware-setup-actions">
            <span>This is taking longer than usual. The first check can take a few minutes; Recall still uses your GPU for scans if it has one.</span>
            <StudioButton data-hardware-primary tone="primary" onClick={complete}>Continue without waiting</StudioButton>
          </footer>
        )}

        {state === "ready" && capabilities && (
          <>
            <header className="hardware-result-heading">
              <StateIcon
                tone={accelerated ? "success" : "warning"}
                icon={accelerated ? <Check size={18} /> : <HardDrive size={18} />}
              />
              <div>
                <h2 id="hardware-setup-title">{accelerated ? "Recall is ready" : "Recall is ready in CPU mode"}</h2>
                <p id="hardware-setup-description">{gpuLabel(capabilities)}</p>
              </div>
            </header>
            <p className="hardware-result-copy">{resultCopy(capabilities)}</p>
            <HardwareFacts capabilities={capabilities} />
            {secondaryWarning && (
              <div className="hardware-notice"><Info size={15} aria-hidden="true" /><span>{secondaryWarning.message}</span></div>
            )}
            <HardwareDetails capabilities={capabilities} />
            <footer className="hardware-setup-actions">
              <span>Worker limits are recalculated before every scan.</span>
              <StudioButton data-hardware-primary tone="primary" onClick={complete}>
                {accelerated ? "Use recommended setup" : "Continue with CPU mode"}
              </StudioButton>
            </footer>
          </>
        )}

        {state === "error" && (
          <>
            <header className="hardware-result-heading is-warning">
              <StateIcon tone="warning" icon={<Info size={18} />} />
              <div>
                <h2 id="hardware-setup-title">Recall will use the safe CPU setup</h2>
                <p id="hardware-setup-description">Hardware acceleration could not be verified.</p>
              </div>
            </header>
            <p className="hardware-result-copy">{error} You can check again now or rescan later in Settings.</p>
            <footer className="hardware-setup-actions">
              <StudioButton tone="ghost" icon={<RefreshCw size={14} />} onClick={check}>Check again</StudioButton>
              <StudioButton data-hardware-primary tone="primary" onClick={complete}>Continue with CPU mode</StudioButton>
            </footer>
          </>
        )}
    </>
  );
}

export function useHardwareCheck(apiEndpoint: string, active: boolean) {
  const [state, setState] = useState<CheckState>("checking");
  const [capabilities, setCapabilities] = useState<HardwareCapabilities | null>(null);
  const [error, setError] = useState("");
  const slow = useSlowAfter(active && state === "checking");

  const check = (force = false) => {
    setState("checking");
    setError("");
    loadHardwareCapabilities(apiEndpoint, force)
      .then((result) => {
        setCapabilities(result);
        setState("ready");
      })
      .catch((reason) => {
        setError(reason instanceof Error ? reason.message : "Recall could not check this PC right now.");
        setState("error");
      });
  };

  useEffect(() => {
    if (!active) return;
    let mounted = true;
    setState("checking");
    setError("");
    loadHardwareCapabilities(apiEndpoint)
      .then((result) => {
        if (!mounted) return;
        setCapabilities(result);
        setState("ready");
      })
      .catch((reason) => {
        if (!mounted) return;
        setError(reason instanceof Error ? reason.message : "Recall could not check this PC right now.");
        setState("error");
      });
    return () => {
      mounted = false;
    };
  }, [active, apiEndpoint]);

  return { state, capabilities, error, slow, check: () => check(true) };
}

/** Compact system-check card used inside first-run onboarding. */
export function HardwareProbeCard({
  state,
  capabilities,
  error,
  onRecheck,
}: {
  state: CheckState;
  capabilities: HardwareCapabilities | null;
  error: string;
  onRecheck: () => void;
}) {
  const accelerated = capabilities?.route === "nvidia_cuda";
  const workers = capabilities
    ? capabilities.recommended.worker_mode === "gpu"
      ? `${capabilities.recommended.gpu_workers} GPU`
      : `${capabilities.recommended.cpu_workers} CPU`
    : "—";
  const title = state === "checking"
    ? "Reading your hardware…"
    : state === "error"
      ? "Could not verify acceleration"
      : accelerated
        ? gpuLabel(capabilities!)
        : capabilities?.gpu?.name || "No supported NVIDIA accelerator";
  const subtitle = state === "checking"
    ? "Looking for a supported accelerator"
    : state === "error"
      ? error
      : accelerated
        ? "CUDA ready · Recall will use your GPU for the heavy stages"
        : resultCopy(capabilities!);

  return (
    <div className="onboarding-check-card">
      <div className="onboarding-check-head">
        <span className={`onboarding-pulse ${state === "checking" ? "is-scanning" : accelerated ? "is-ok" : "is-warn"}`} aria-hidden="true">
          {state === "checking" ? <HardDrive size={18} /> : accelerated ? <Check size={18} /> : <HardDrive size={18} />}
        </span>
        <span>
          <b>{title}</b>
          <span>{subtitle}</span>
        </span>
        {state !== "checking" && (
          <button className="onboarding-redo" type="button" onClick={onRecheck}>Re-check</button>
        )}
      </div>
      <dl className="onboarding-facts" style={{ opacity: state === "ready" && capabilities ? 1 : 0.35 }}>
        <div><dt>Processor</dt><dd className="t-num">{capabilities ? `${capabilities.cpu.physical_cores} cores` : "—"}</dd></div>
        <div><dt>Memory</dt><dd className="t-num">{capabilities?.memory.available_gb == null ? "—" : `${capabilities.memory.available_gb.toFixed(1)} GB`}</dd></div>
        <div><dt>Route</dt><dd>{state === "ready" && capabilities ? (accelerated ? <span className="ok">NVIDIA</span> : "CPU") : "—"}</dd></div>
        <div><dt>Workers</dt><dd className="t-num">{state === "ready" ? workers : "—"}</dd></div>
      </dl>
    </div>
  );
}

export function HardwareSetupDialog({
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
  if (!open) return null;
  return (
    <div className="hardware-setup-modal open">
      <div
        ref={dialogRef}
        className="hardware-setup-dialog elev-2"
        role="dialog"
        aria-modal="true"
        aria-labelledby="hardware-setup-title"
        aria-describedby="hardware-setup-description"
        tabIndex={-1}
      >
        <div className="hardware-setup-brand"><Sparkles size={16} aria-hidden="true" /><span>Recall</span></div>
        <HardwareCheckStep active={open} apiEndpoint={apiEndpoint} dialogRef={dialogRef} onComplete={onComplete} />
      </div>
    </div>
  );
}

export function HardwareSettingsPanel({
  active,
  apiEndpoint,
}: {
  active: boolean;
  apiEndpoint: string;
}) {
  const [capabilities, setCapabilities] = useState<HardwareCapabilities | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!active || capabilities) return;
    let mounted = true;
    setChecking(true);
    loadHardwareCapabilities(apiEndpoint)
      .then((result) => {
        if (!mounted) return;
        setCapabilities(result);
        setError("");
      })
      .catch((reason) => {
        if (!mounted) return;
        setError(reason instanceof Error ? reason.message : "Recall could not check this PC right now.");
      })
      .finally(() => {
        if (mounted) setChecking(false);
      });
    return () => {
      mounted = false;
    };
  }, [active, apiEndpoint, capabilities]);

  const summary = useMemo(() => {
    if (!capabilities) return null;
    return {
      title: capabilities.route === "nvidia_cuda" ? "GPU acceleration ready" : "CPU processing mode",
      copy: resultCopy(capabilities),
    };
  }, [capabilities]);

  const rescan = () => {
    setChecking(true);
    setError("");
    loadHardwareCapabilities(apiEndpoint, true)
      .then((result) => setCapabilities(result))
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Recall could not check this PC right now."))
      .finally(() => setChecking(false));
  };

  return (
    <section className="hardware-settings-panel" aria-labelledby="hardware-settings-title">
      <div className="hardware-settings-heading">
        <div>
          <h3 id="hardware-settings-title">This PC</h3>
          <p>{capabilities ? checkedAtLabel(capabilities.checked_at) : "Recall chooses a safe processing route for this computer."}</p>
        </div>
        <StudioButton tone="ghost" loading={checking} icon={<RefreshCw size={14} />} onClick={rescan}>Run system check again</StudioButton>
      </div>
      {summary && capabilities && (
        <>
          <div className={`hardware-settings-status ${capabilities.route === "nvidia_cuda" ? "is-ready" : "is-cpu"}`}>
            <StateIcon tone={capabilities.route === "nvidia_cuda" ? "success" : "warning"} icon={capabilities.route === "nvidia_cuda" ? <Check size={15} /> : <HardDrive size={15} />} />
            <div><strong>{summary.title}</strong><span>{gpuLabel(capabilities)}</span></div>
          </div>
          <p className="hardware-settings-copy">{summary.copy}</p>
          <HardwareDetails capabilities={capabilities} />
        </>
      )}
      {error && <div className="hardware-settings-error" role="alert"><Info size={14} aria-hidden="true" /><span>{error}</span></div>}
    </section>
  );
}

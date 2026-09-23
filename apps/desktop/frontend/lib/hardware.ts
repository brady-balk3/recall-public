// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { apiFetch } from "./api";

export type HardwareRoute = "nvidia_cuda" | "cpu";
export type HardwareReason =
  | "accelerated"
  | "nvidia_cuda_unavailable"
  | "unsupported_gpu"
  | "no_supported_gpu";

export interface HardwareAdapter {
  name: string;
  vendor: "nvidia" | "amd" | "intel" | "unknown";
  driver_version: string;
}

export interface HardwareCapabilities {
  schema_version: number;
  checked_at: string;
  route: HardwareRoute;
  reason: HardwareReason;
  cpu: {
    physical_cores: number;
    logical_cores: number;
  };
  memory: {
    available_gb: number | null;
  };
  gpu: HardwareAdapter | null;
  adapters: HardwareAdapter[];
  acceleration: {
    cuda_ready: boolean;
    torch_device: "cuda" | "cpu";
    onnx_providers: string[];
    onnx_cuda_ready: boolean;
    llama_gpu: boolean | null;
    free_vram_gb: number | null;
    gpu_perception_ready: boolean;
  };
  recommended: {
    performance_profile: "balanced";
    worker_mode: "gpu" | "cpu";
    cpu_workers: number;
    gpu_workers: number;
  };
  warnings: Array<{
    code: string;
    message: string;
  }>;
}

export const HARDWARE_SETUP_STORAGE_KEY = "recall-hardware-setup-complete";

let cachedCapabilities: HardwareCapabilities | null = null;
let capabilityRequest: Promise<HardwareCapabilities> | null = null;

export function hasCompletedHardwareSetup(): boolean {
  try {
    return !!localStorage.getItem(HARDWARE_SETUP_STORAGE_KEY);
  } catch {
    return false;
  }
}

export function markHardwareSetupComplete(capabilities: HardwareCapabilities | null): void {
  try {
    localStorage.setItem(HARDWARE_SETUP_STORAGE_KEY, JSON.stringify({
      completedAt: new Date().toISOString(),
      route: capabilities?.route ?? "cpu",
      schemaVersion: capabilities?.schema_version ?? 1,
    }));
  } catch {
    // A locked-down storage context should not trap the creator in onboarding.
  }
}

async function responseError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string" && body.detail.trim()) return body.detail;
  } catch {
    // Fall through to the stable creator-facing message.
  }
  return "Recall could not check this PC right now.";
}

export async function loadHardwareCapabilities(
  apiEndpoint: string,
  refresh = false,
): Promise<HardwareCapabilities> {
  if (!refresh && cachedCapabilities) return cachedCapabilities;
  if (capabilityRequest) return capabilityRequest;

  const path = refresh ? "/system/capabilities?refresh=true" : "/system/capabilities";
  capabilityRequest = apiFetch(path, undefined, apiEndpoint)
    .then(async (response) => {
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as HardwareCapabilities;
      cachedCapabilities = result;
      return result;
    })
    .finally(() => {
      capabilityRequest = null;
    });
  return capabilityRequest;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import path from "node:path";

/** Installed builds opt in through a packaging marker; portable defaults stay put. */
export function runtimeStorageEnvironment(
  inherited: NodeJS.ProcessEnv,
  userData: string,
  installed: boolean,
): NodeJS.ProcessEnv {
  const env = { ...inherited };
  const configured = env.RECALL_STORAGE_ROOT?.trim();
  const root = configured || (installed ? path.join(userData, "runtime") : undefined);
  if (!root) return env;
  if (!path.isAbsolute(root) || (process.platform === "win32" && !path.parse(root).root.replaceAll(/[\\/]/g, ""))) {
    throw new Error("Recall storage root must be a fully qualified absolute path");
  }
  env.RECALL_STORAGE_ROOT = path.normalize(root);
  env.HF_HOME ||= path.join(root, "model-cache");
  env.HF_HUB_CACHE ||= path.join(env.HF_HOME, "hub");
  return env;
}

export function installedStorageMode(marker: unknown): boolean {
  if (!marker || typeof marker !== "object") throw new Error("Invalid Recall installation marker");
  const value = marker as Record<string, unknown>;
  if (value.version !== 1 || value.storageMode !== "installed") {
    throw new Error("Unsupported Recall installation marker");
  }
  return true;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { hasCompletedHardwareSetup } from "./hardware";

export const ONBOARDING_STORAGE_KEY = "recall-onboarding";

/**
 * Bump when a later release adds a step worth re-running for existing installs.
 * A stored record below this version replays the flow; at or above it, skip.
 */
export const ONBOARDING_VERSION = 1;

export interface OnboardingRecord {
  version: number;
  completedAt: string;
  /** Steps the creator passed on rather than answering. */
  skipped: string[];
}

export function hasCompletedOnboarding(): boolean {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(ONBOARDING_STORAGE_KEY);
  } catch {
    // A locked-down storage context can never record completion, so replaying
    // onboarding every launch would trap the creator. Treat it as done.
    return true;
  }

  if (raw) {
    try {
      const parsed = JSON.parse(raw) as Partial<OnboardingRecord>;
      if (typeof parsed?.version === "number" && parsed.version >= ONBOARDING_VERSION) return true;
    } catch {
      // A corrupted record falls through to the pre-onboarding signal below.
    }
  }

  // Installs that predate this flow already cleared the standalone system
  // check. Dragging them back through onboarding would be a regression, so the
  // old key still counts as a completed first run.
  return hasCompletedHardwareSetup();
}

export function markOnboardingComplete(skipped: string[] = []): void {
  try {
    localStorage.setItem(ONBOARDING_STORAGE_KEY, JSON.stringify({
      version: ONBOARDING_VERSION,
      completedAt: new Date().toISOString(),
      skipped,
    } satisfies OnboardingRecord));
  } catch {
    // Same reasoning as above: storage failure must not block the studio.
  }
}

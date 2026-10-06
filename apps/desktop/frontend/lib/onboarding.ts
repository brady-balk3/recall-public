// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
export const ONBOARDING_STORAGE_KEY = "recall-onboarding";

/**
 * Bump when a later release adds a step worth re-running for existing installs.
 * A stored record below this version replays the flow; at or above it, skip.
 */
// 2: the redesigned first run asks for the creator's name (identity is a typed
// name and its letter now), so every install goes through it once.
export const ONBOARDING_VERSION = 2;

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
      // A corrupted record replays the flow.
    }
  }

  // Older records, older installs (the standalone system check counted as
  // "onboarded" before version 2) and corrupt records all replay the flow.
  return false;
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

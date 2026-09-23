// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** Count-aware label: `plural(1, "clip")` -> "1 clip", `plural(3, "clip")` ->
 * "3 clips". Pass `plural(2, "match", "matches")` for irregular nouns. */
export const plural = (count: number, singular: string, pluralForm?: string): string =>
  `${count} ${count === 1 ? singular : pluralForm ?? `${singular}s`}`;

const CLUE_LABELS: Record<string, string> = {
  match_win: "Match win",
  elimination: "Elimination",
  knock: "Knockdown",
  rank_progress: "Rank progress",
  voice_reaction: "Strong reaction",
  facecam_reaction: "Facecam moment",
  speech_hype: "Excited callout",
  laughter_burst: "Laughter or scream",
  chat_spike: "Chat reacted",
  multi_signal: "Several clues lined up",
  low_dead_air: "Gets to the moment quickly",
  generic_highlight: "Possible highlight",
};

export const STAGE_LABELS: Record<string, string> = {
  Init: "Getting the recording ready",
  "Resolving Input": "Getting the recording ready",
  Perception: "Looking through the recording",
  Reaction: "Finding exciting moments",
  Event: "Finding exciting moments",
  Story: "Finding exciting moments",
  Clip: "Choosing the strongest clips",
  Caption: "Preparing captions",
  Export: "Preparing clips",
  Complete: "Ready to review",
};

export function stageLabel(raw?: string | null): string {
  if (!raw) return "Looking through the recording";
  return STAGE_LABELS[raw] ?? raw;
}

export function clueLabel(raw: string): string {
  if (!raw) return "Clip clue";
  const key = raw.trim().toLowerCase();
  if (CLUE_LABELS[key]) return CLUE_LABELS[key];
  const cleaned = key.replace(/[_-]+/g, " ");
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

export function learningStateLabel(state?: string): string {
  if (state === "personal_active") return "Personal picks active";
  if (state === "ready_to_personalize") return "Ready to personalize";
  if (state === "learning") return "Learning from your picks";
  return "Ready to learn";
}

export function learningStateDescription(state?: string): string {
  if (state === "personal_active") return "Recall is using your past decisions to sort future clips.";
  if (state === "ready_to_personalize") return "Recall has enough decisions to tune future review sets.";
  if (state === "learning") return "Saved, exported, and rejected clips help Recall understand your taste.";
  return "Start saving, exporting, or rejecting clips and Recall will learn what you like.";
}

/** Creator-facing scan health copy. Returns null when there's nothing worth
 * surfacing (healthy judges on GPU, or no health data yet). */
export interface ScanHealthInfo {
  tone: "warning" | "info";
  title: string;
  detail: string;
}

export function describeScanHealth(health?: {
  degraded?: boolean;
  device?: string | null;
  semantic?: { status?: string; degraded?: boolean; fallback_candidates?: number; judged_candidates?: number } | null;
  visual?: {
    status?: string;
    degraded?: boolean;
    reason?: string;
    judged_candidates?: number;
    failures?: number;
    recent_failures?: Array<{ error?: string }>;
  } | null;
} | null): ScanHealthInfo | null {
  if (!health) return null;
  const visualStatus = health.visual?.status;
  const semanticBad = Boolean(
    health.semantic?.degraded || health.semantic?.status === "degraded",
  );
  const onCpu = health.device === "cpu";

  if (visualStatus === "skipped") {
    return {
      tone: "warning",
      title: "Vision review unavailable",
      detail: "This scan used reaction signals without the local vision model. Wins and on-screen moments may be thinner.",
    };
  }
  if (visualStatus === "degraded" || semanticBad) {
    const parts: string[] = [];
    if (visualStatus === "degraded") parts.push("vision review");
    if (semanticBad) parts.push("AI moment review");
    const failCount = Number(health.visual?.failures || 0);
    let detail = `${parts.join(" and ")} had to fall back mid-scan. The deck still built, but quality may be uneven.`;
    if (visualStatus === "degraded" && failCount > 0) {
      const judgedCount = Number(health.visual?.judged_candidates || 0);
      const totalCount = judgedCount + failCount;
      const mostlyCompleted = judgedCount > 0
        && failCount <= Math.max(2, Math.ceil(totalCount * 0.05));
      if (mostlyCompleted && !semanticBad) {
        return {
          tone: "warning",
          title: "Vision review mostly completed",
          detail: `Recall reviewed ${judgedCount} moment${judgedCount === 1 ? "" : "s"} normally and used reaction and transcript fallback for ${failCount}. Your deck is ready; no rerun is needed.`,
        };
      }
      detail = `Recall used reaction and transcript fallback for ${failCount} moment${failCount === 1 ? "" : "s"}. Review the deck normally; rerun only if it feels unusually thin.`;
    }
    return {
      tone: "warning",
      title: "Some AI review ran degraded",
      detail,
    };
  }
  if (onCpu && (health.semantic || health.visual || health.degraded)) {
    return {
      tone: "info",
      title: "Running on CPU",
      detail: "No GPU acceleration detected for this scan. Expect longer run times.",
    };
  }
  return null;
}

/** Failed-scan diagnosis (plan 22 §1.5): translate a raw engine error into
 * creator language + the concrete next step. The raw text stays available
 * for the diagnostic report; this is what a person should read first. */
export interface ScanFailureInfo {
  /** Creator-facing stage the failure hit, when the backend recorded one. */
  stage?: string;
  title: string;
  hint: string;
  raw: string;
}

export function describeScanFailure(errorMessage?: string): ScanFailureInfo {
  const raw = (errorMessage || "").trim();
  // mark_job_failed prefixes "[<stage>] " when it knows the running stage.
  const stageMatch = raw.match(/^\[([^\]]+)\]\s*/);
  const stage = stageMatch ? stageLabel(stageMatch[1]) : undefined;
  const text = (stageMatch ? raw.slice(stageMatch[0].length) : raw).toLowerCase();

  const info = (title: string, hint: string): ScanFailureInfo => ({ stage, title, hint, raw });

  if (!text) {
    return info("The scan stopped unexpectedly.", "Try the scan again — if it keeps happening, copy the report below.");
  }
  if (/(yt-dlp|403|404|http error|unable to download|urlopen|getaddrinfo|network|connection)/.test(text)) {
    return info("The link couldn't be downloaded.", "Check that the VOD is public and still online, then retry. Sub-only VODs need a downloaded file instead.");
  }
  if (/(no such file|cannot find the file|winerror 2|not found|filenotfound)/.test(text)) {
    return info("The recording file couldn't be found.", "It may have been moved or renamed since you picked it. Browse for it again.");
  }
  if (/(invalid data|moov atom|could not find codec|invalid argument|corrupt|unsupported|decode)/.test(text)) {
    return info("The recording couldn't be read.", "The file may be incomplete or in a format the scan can't open. Remuxing it to MP4 usually fixes this.");
  }
  if (/(no space|disk full|errno 28)/.test(text)) {
    return info("This machine ran out of disk space.", "Free some space (Settings has one-click cache cleanup), then retry.");
  }
  if (/(cuda|out of memory|memoryerror|allocat)/.test(text)) {
    return info("The scan ran out of memory.", "Close other heavy apps, or switch Scan settings to Fast mode and retry.");
  }
  if (/(permission|access is denied|errno 13)/.test(text)) {
    return info("Recall wasn't allowed to open the file.", "Another program may have it locked, or it's in a protected folder. Move it and retry.");
  }
  return info("Something inside the scan engine failed.", "Retry the scan — if it fails the same way, copy the report below.");
}

/** Ready-to-paste post text per platform (plan 22 §3.3) — deterministic
 * composition of the clip's title, post-ready description, and game-aware
 * hashtags. Length-capped per platform. */
export function postText(
  clip: { title?: string; description?: string; postTags?: string[] },
  platform: "tiktok" | "shorts" | "x" | "none",
): string {
  const title = (clip.title || "").trim();
  const desc = (clip.description || "").trim();
  const tags = (clip.postTags ?? []).map((t) => `#${t}`).join(" ");
  if (platform === "tiktok") {
    // TikTok captions cut off around 150 visible chars — keep it tight.
    const base = tags ? `${title} ${tags}` : title;
    return base.length > 150 ? `${title.slice(0, 150 - tags.length - 2)} ${tags}` : base;
  }
  if (platform === "shorts") {
    return [title, desc, tags].filter(Boolean).join("\n\n");
  }
  if (platform === "x") {
    // 280-char budget: title + as many tags as fit.
    let out = title;
    for (const tag of clip.postTags ?? []) {
      if (out.length + tag.length + 2 > 278) break;
      out += ` #${tag}`;
    }
    return out;
  }
  return [title, desc].filter(Boolean).join("\n\n");
}

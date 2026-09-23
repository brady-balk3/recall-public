// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { apiFetch } from "./api";

export type CompilationProjectStatus = "draft" | "approved";

export interface CompilationProjectItem {
  id: string;
  clip_id: string | null;
  job_id: string | null;
  source_group: string | null;
  position: number;
  included: boolean;
  selection_score: number | null;
  selection_reason: string;
  title: string;
  session_name: string;
  source_date: string | null;
  start_time: number;
  end_time: number;
  duration: number;
  media_available: boolean;
  media_state: "rendered" | "rebuildable" | "source_unavailable" | "clip_removed" | "not_cut_yet";
  // "event" moments are detected in the match but not cut into a clip yet.
  source_kind: "clip" | "event";
  moment_kind: "elimination" | "win" | null;
  /** Kills render clean; the closing win keeps its captions. */
  captions_enabled: boolean | null;
  /** A muted moment plays silent in the montage; its clip keeps its audio. */
  muted: boolean;
}

export interface CompilationProjectSummary {
  schema_version: number;
  id: string;
  title: string;
  template: "best_of_range" | "single_match" | "theme_montage";
  query: string;
  game: string | null;
  date_from: string | null;
  date_to: string | null;
  target_duration_seconds: number | null;
  max_clips: number;
  max_per_session: number;
  status: CompilationProjectStatus;
  generation_version: number;
  selection_summary: {
    candidates_considered?: number;
    duplicates_suppressed?: number;
    sessions_represented?: number;
    selected_duration_seconds?: number;
    selection_policy?: string;
    match_kill_estimate?: number;
    match_outcome?: string;
    match_confidence?: string;
    uncut_moment_count?: number;
    offline_moment_count?: number;
    oversized_clip_count?: number;
    finale_reason?: string;
  };
  created_at: string;
  updated_at: string;
  approved_at: string | null;
  item_count: number;
  included_count: number;
  available_count: number;
  session_count: number;
  total_duration_seconds: number;
  // Set only on a project scoped to one match inside one session.
  source_job_id: string | null;
  match_index: number | null;
  match_label: string | null;
  match_start: number | null;
  match_end: number | null;
  uncut_count: number;
  muted_count: number;
  /** The rendered montage, once one has been built. */
  reel_path: string | null;
  reel_url: string | null;
  reel_duration_seconds: number | null;
  reel_built_at: string | null;
  /** True when the sequence changed after the montage was rendered. */
  reel_stale: boolean;
}

export interface CompilationProject extends CompilationProjectSummary {
  items: CompilationProjectItem[];
}

export type CompilationPreparationStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "completed"
  | "partial"
  | "cancelled"
  | "interrupted";

export interface CompilationPreparationSource {
  source_key: string;
  label: string;
  kind: "twitch" | "local" | "clip_removed";
  state: "restorable" | "restoring" | "manual_required" | "unavailable" | "clip_removed";
  state_label: string;
  restorable: boolean;
  clip_count: number;
  session_names: string[];
  estimated_bytes: number;
  last_error: string | null;
}

export interface CompilationPreparationItem {
  source_key: string;
  label: string;
  position: number;
  status: "pending" | "restoring" | "prepared" | "failed" | "cancelled" | "interrupted";
  estimated_bytes: number;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  updated_at: string;
}

export interface CompilationPreparation {
  id: string;
  project_id: string;
  status: CompilationPreparationStatus;
  total_sources: number;
  prepared_sources: number;
  failed_sources: number;
  completed_sources: number;
  estimated_bytes: number;
  current_source_key: string | null;
  cancel_requested: boolean;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  updated_at: string;
  items: CompilationPreparationItem[];
}

export interface CompilationPreparationPlan {
  schema_version: number;
  project_id: string;
  project_status: CompilationProjectStatus;
  included_count: number;
  available_count: number;
  missing_clip_count: number;
  required_source_count: number;
  restorable_source_count: number;
  manual_source_count: number;
  unavailable_source_count: number;
  estimated_download_bytes: number;
  free_bytes: number;
  reserve_bytes: number;
  disk_allowed: boolean;
  active_scan: boolean;
  can_start: boolean;
  blockers: string[];
  safe_sources: Array<{ source_key: string; state_label: string; file_size_bytes: number }>;
  sources: CompilationPreparationSource[];
  preparation: CompilationPreparation | null;
}

async function jsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof body?.detail === "string"
      ? body.detail
      : typeof body?.detail?.message === "string"
        ? body.detail.message
        : "Compilation request failed.";
    throw new Error(detail);
  }
  return body as T;
}

export async function listCompilationProjects(baseUrl?: string): Promise<CompilationProjectSummary[]> {
  const response = await jsonOrThrow<{ projects: CompilationProjectSummary[] }>(
    await apiFetch("/compilation-projects", undefined, baseUrl),
  );
  return response.projects;
}

export async function getCompilationProject(id: string, baseUrl?: string): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(
    await apiFetch(`/compilation-projects/${encodeURIComponent(id)}`, undefined, baseUrl),
  );
}

export type CompilationRecipe = {
  id: string;
  label: string;
  description: string;
  needs_game: boolean;
  tags: string[];
  games: string[];
};

export async function listCompilationRecipes(baseUrl?: string): Promise<CompilationRecipe[]> {
  const body = await jsonOrThrow<{ recipes: CompilationRecipe[] }>(
    await apiFetch("/compilation-projects/recipes", {}, baseUrl),
  );
  return body.recipes || [];
}

export interface SessionMatch {
  job_id: string;
  match_index: number;
  label: string;
  outcome: "win" | "placed" | "eliminated" | "unknown";
  placement: number | null;
  confidence: "high" | "medium" | "low";
  start_time: number;
  end_time: number;
  duration_seconds: number;
  kill_estimate: number;
  detected_kills: number;
  milestone_kills: number;
  has_win: boolean;
  moment_count: number;
  already_cut_count: number;
  boundary_reasons: string[];
}

export interface SessionMatches {
  job_id: string;
  session_name: string;
  source_date: string | null;
  match_count: number;
  matches: SessionMatch[];
}

/** The individual games Recall detected inside one scanned session. */
export async function listSessionMatches(
  jobId: string,
  baseUrl?: string,
): Promise<SessionMatches> {
  return jsonOrThrow<SessionMatches>(
    await apiFetch(`/jobs/${encodeURIComponent(jobId)}/matches`, {}, baseUrl),
  );
}

/** Build a compilation from one match, with its win placed last. */
export async function generateCompilationProjectFromMatch(input: {
  jobId: string;
  matchIndex: number;
  title?: string;
  /** The montage length to aim for. This is the primary control. */
  targetDurationSeconds: number;
  baseUrl?: string;
}): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(await apiFetch("/compilation-projects/generate-match", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_id: input.jobId,
      match_index: input.matchIndex,
      title: input.title || undefined,
      target_duration_seconds: input.targetDurationSeconds,
      max_clips: 40,
    }),
  }, input.baseUrl));
}

/** Copy the built montage into a folder the creator picked. */
export async function saveCompilationReel(
  projectId: string,
  destFolder: string,
  baseUrl?: string,
): Promise<{ path: string }> {
  return jsonOrThrow<{ path: string }>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(projectId)}/save-reel`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dest_folder: destFolder }),
    },
    baseUrl,
  ));
}

/** Build one montage from one theme, across every scanned session. */
export async function generateCompilationProjectFromTheme(input: {
  title: string;
  query?: string;
  dateFrom?: string;
  dateTo?: string;
  targetDurationSeconds: number;
  baseUrl?: string;
}): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(await apiFetch("/compilation-projects/generate-theme", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: input.title,
      query: input.query || "",
      date_from: input.dateFrom || null,
      date_to: input.dateTo || null,
      target_duration_seconds: input.targetDurationSeconds,
    }),
  }, input.baseUrl));
}

export interface BuiltReel {
  project: CompilationProject;
  clips: number;
  errors: number;
}

/** Stitch this project's cut moments into one montage Recall keeps and plays. */
export async function buildCompilationReel(
  projectId: string,
  baseUrl?: string,
): Promise<BuiltReel> {
  return jsonOrThrow<BuiltReel>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(projectId)}/build-reel`,
    { method: "POST" },
    baseUrl,
  ));
}

export interface CutMomentResult {
  done: boolean;
  cut_clip_id: string | null;
  project: CompilationProject;
}

/**
 * Cut one detected moment into a real clip. One per call: each cut is a real
 * render, so the caller drives the loop and can stop after any of them.
 */
export async function cutNextCompilationMoment(
  projectId: string,
  baseUrl?: string,
): Promise<CutMomentResult> {
  return jsonOrThrow<CutMomentResult>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(projectId)}/cut-next`,
    { method: "POST" },
    baseUrl,
  ));
}

export async function generateCompilationProjectFromRecipe(input: {
  recipeId: string;
  game?: string;
  title?: string;
  baseUrl?: string;
}): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(await apiFetch("/compilation-projects/generate-recipe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      recipe_id: input.recipeId,
      game: input.game || undefined,
      title: input.title || undefined,
    }),
  }, input.baseUrl));
}

export async function generateCompilationProject(input: {
  title: string;
  query?: string;
  game?: string;
  dateFrom?: string;
  dateTo?: string;
  maxClips?: number;
  maxPerSession?: number;
  targetDurationSeconds?: number;
  baseUrl?: string;
}): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(await apiFetch("/compilation-projects/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: input.title,
      query: input.query ?? "",
      game: input.game || undefined,
      date_from: input.dateFrom || undefined,
      date_to: input.dateTo || undefined,
      max_clips: input.maxClips ?? 12,
      max_per_session: input.maxPerSession ?? 3,
      target_duration_seconds: input.targetDurationSeconds,
    }),
  }, input.baseUrl));
}

export async function renameCompilationProject(
  id: string,
  title: string,
  baseUrl?: string,
): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(id)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    },
    baseUrl,
  ));
}

/** Silence the montage in one action, keeping the moment it ends on audible. */
export async function setCompilationAudio(
  projectId: string,
  input: { muted: boolean; keepFinale?: boolean },
  baseUrl?: string,
): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(projectId)}/audio`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ muted: input.muted, keep_finale: input.keepFinale ?? true }),
    },
    baseUrl,
  ));
}

export async function saveCompilationProjectItems(
  id: string,
  items: Array<Pick<CompilationProjectItem, "id" | "included">>,
  baseUrl?: string,
): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(id)}/items`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    },
    baseUrl,
  ));
}

export async function addClipToCompilationProject(input: {
  projectId: string;
  clipId: string;
  memoryEntryId?: string;
  memoryQuery?: string;
  baseUrl?: string;
}): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(input.projectId)}/clips`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clip_id: input.clipId,
        memory_entry_id: input.memoryEntryId,
        memory_query: input.memoryQuery,
      }),
    },
    input.baseUrl,
  ));
}

export async function approveCompilationProject(id: string, baseUrl?: string): Promise<CompilationProject> {
  return jsonOrThrow<CompilationProject>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(id)}/approve`,
    { method: "POST" },
    baseUrl,
  ));
}

export async function deleteCompilationProject(id: string, baseUrl?: string): Promise<void> {
  await jsonOrThrow(await apiFetch(
    `/compilation-projects/${encodeURIComponent(id)}`,
    { method: "DELETE" },
    baseUrl,
  ));
}

export async function getCompilationPreparation(
  id: string,
  baseUrl?: string,
): Promise<CompilationPreparationPlan> {
  return jsonOrThrow<CompilationPreparationPlan>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(id)}/preparation`,
    undefined,
    baseUrl,
  ));
}

export async function startCompilationPreparation(
  id: string,
  sourceKeys: string[],
  baseUrl?: string,
): Promise<CompilationPreparationPlan> {
  return jsonOrThrow<CompilationPreparationPlan>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(id)}/preparation`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_keys: sourceKeys }),
    },
    baseUrl,
  ));
}

export async function cancelCompilationPreparation(
  id: string,
  baseUrl?: string,
): Promise<CompilationPreparationPlan> {
  return jsonOrThrow<CompilationPreparationPlan>(await apiFetch(
    `/compilation-projects/${encodeURIComponent(id)}/preparation/cancel`,
    { method: "POST" },
    baseUrl,
  ));
}

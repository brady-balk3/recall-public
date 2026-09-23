// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { apiFetch } from "./api";

export type MemoryKind = "all" | "transcript" | "clip" | "evidence";
export type MemoryDecision = "all" | "kept" | "passed" | "maybe" | "unreviewed";
export type MemorySearchMode = "hybrid" | "keyword";
export type MemoryExportFilter = "all" | "exported" | "not_exported";
export type MemoryOriginFilter = "all" | "creator_marker";

export interface MemoryResolvedFilters {
  kind: MemoryKind;
  decision: MemoryDecision;
  exported: MemoryExportFilter;
  origin: MemoryOriginFilter;
  game: string | null;
  date_from: string | null;
  date_to: string | null;
}
export type MemoryMatchSource =
  | "creator_speech"
  | "clip_details"
  | "creator_marker"
  | "gameplay_event"
  | "chat_activity"
  | "visual_description"
  | "on_screen_text"
  | "related_meaning";

export interface MemoryStats {
  schema_version: number;
  completed_jobs: number;
  indexed_jobs: number;
  transcript_entries: number;
  clip_entries: number;
  evidence_entries?: number;
  evidence_source_bytes?: number;
  evidence_version?: number;
  last_indexed_at: string | null;
  semantic_status: "not_built" | "building" | "ready" | "stale" | "error";
  semantic_available: boolean;
  semantic_model_id: string | null;
  semantic_dimensions: number;
  semantic_entries: number;
  semantic_index_bytes: number;
  semantic_model_bytes: number;
  semantic_indexed_at: string | null;
  semantic_last_error: string | null;
  alias_count: number;
  feedback_count: number;
}

export interface MemoryAlias {
  id: string;
  canonical_term: string;
  alias: string;
  created_at: string;
  updated_at: string;
}

export interface MemoryAliasExpansion {
  canonical_term: string;
  matched_term: string;
  terms: string[];
}

export interface MemoryCompoundClause {
  id: string;
  label: string;
}

export interface MemoryMatchedEvidence {
  detail: string;
  source: "reaction_signal" | "gameplay_event" | "chat_activity" | "creator_marker"
    | "visual_description" | "on_screen_text" | string;
  clause_id: string;
  label: string;
}

export interface MemoryResult {
  id: string;
  job_id: string;
  clip_id: string | null;
  kind: "transcript" | "clip" | "evidence";
  start_time: number;
  end_time: number;
  title: string;
  text: string;
  match_context: string;
  session_name: string;
  game: string;
  source_date: string | null;
  source_type: string | null;
  source_path: string | null;
  duration: number | null;
  kept: boolean;
  passed: boolean;
  maybe: boolean;
  exported?: boolean;
  exported_at?: string | null;
  review_state?: "kept" | "passed" | "maybe" | "unreviewed" | null;
  selection_score?: number | null;
  evidence_kind?: "clip_pack" | "creator_marker" | "detected_event" | "chat_activity" | null;
  evidence?: {
    version?: number;
    anchor?: "clip" | "creator_marker" | "detected_event" | "chat_activity";
    search_text?: string;
    labels?: string[];
    gameplay_events?: string[];
    reactions?: string[];
    ocr_phrases?: string[];
    visual_scenes?: Array<{
      description?: string;
      support?: string | null;
      outcome?: string | null;
      confidence?: number;
      source?: "candidate_trace" | "visual_judge_cache" | string;
    }>;
    chat_activity?: {
      message_count?: number;
      emote_count?: number;
      clip_intent_count?: number;
      repeated_phrases?: Array<{ text?: string; count?: number }>;
      source_scope?: "full_vod" | "smart_regions" | string | null;
    } | null;
    layout?: {
      facecam_present?: boolean;
      windowed_gameplay?: boolean;
    };
    creator_marker?: {
      event_ids?: string[];
      timestamp?: number | null;
      source?: string;
      ordinal?: number;
      alignment?: string | null;
      uncertainty_seconds?: number | null;
    } | null;
    provenance?: Array<{ artifact?: string }>;
  };
  origin?: "transcript" | "clip" | "scan_evidence" | "creator_marker";
  transcript_coverage: "full" | "partial" | "none" | "unknown" | null;
  source_available: boolean;
  match_type: "keyword" | "semantic" | "hybrid" | "compound";
  match_source?: MemoryMatchSource;
  match_reason: string;
  semantic_score: number | null;
  creator_feedback: "relevant" | "not_relevant" | null;
  matched_evidence?: MemoryMatchedEvidence[];
  compound_coverage?: {
    matched: number;
    required: number;
    ratio: number;
  } | null;
}

export interface MemorySearchResponse {
  schema_version: number;
  search_id?: string | null;
  query: string;
  effective_query?: string;
  count: number;
  mode_requested: MemorySearchMode;
  mode_used: MemorySearchMode;
  semantic_status: MemoryStats["semantic_status"];
  alias_expansions: MemoryAliasExpansion[];
  interpreted_filters?: Array<{ field: string; value: string }>;
  resolved_filters?: MemoryResolvedFilters;
  compound_query?: {
    active: boolean;
    clauses: MemoryCompoundClause[];
    residual_query: string;
  };
  results: MemoryResult[];
}

export type MemoryInteractionType = "impression" | "open";

export interface MemoryFeedbackResponse {
  query: string;
  entry_id: string;
  verdict: "relevant" | "not_relevant" | null;
}

export interface MemoryEvaluationCase {
  id: string;
  query: string;
  expected_description: string;
  status: "labeled" | "missed";
  relevant_count: number;
  negative_count: number;
  resolved_relevant_count: number;
  stale_target_count: number;
  ready_to_score: boolean;
  created_at: string;
  updated_at: string;
}

export interface MemoryEvaluationMetrics {
  queries_scored: number;
  queries_with_targets: number;
  queries_unresolved: number;
  top_1_rate: number;
  top_3_rate: number;
  top_5_rate: number;
  mean_reciprocal_rank_at_10: number;
  negative_hits_top_5: number;
  average_latency_ms: number;
  p95_latency_ms: number;
  truth_revision: string;
  alias_revision: string;
  system_revision: string;
  cases: Array<{
    case_id: string;
    query: string;
    top_rank: number | null;
    result_count: number;
    negative_hits_top_5: number;
    latency_ms: number | null;
    status?: "unresolved_miss";
  }>;
}

export interface MemoryEvaluationRun {
  id: string;
  system_id: string;
  case_count: number;
  metrics: MemoryEvaluationMetrics;
  created_at: string;
  stale: boolean;
}

export interface MemoryEvaluationSummary {
  schema_version: number;
  target_cases: number;
  case_count: number;
  scorable_cases: number;
  unresolved_cases: number;
  cases: MemoryEvaluationCase[];
  latest_run: MemoryEvaluationRun | null;
}

export interface MemoryLearningMetrics {
  queries_scored: number;
  queries_with_targets: number;
  queries_unresolved: number;
  baseline: Pick<MemoryEvaluationMetrics, "top_1_rate" | "top_3_rate" | "top_5_rate" | "mean_reciprocal_rank_at_10" | "negative_hits_top_5">;
  shadow: Pick<MemoryEvaluationMetrics, "top_1_rate" | "top_3_rate" | "top_5_rate" | "mean_reciprocal_rank_at_10" | "negative_hits_top_5">;
  deltas: Pick<MemoryEvaluationMetrics, "top_1_rate" | "top_3_rate" | "top_5_rate" | "mean_reciprocal_rank_at_10" | "negative_hits_top_5">;
  measurement_floor_met: boolean;
  eligible_for_promotion: false;
  live_weight: 0;
}

export interface MemoryLearningRun {
  id: string;
  system_id: string;
  event_revision: string;
  case_count: number;
  metrics: MemoryLearningMetrics;
  created_at: string;
  stale: boolean;
}

export interface MemoryLearningSummary {
  schema_version: number;
  learning_version: number;
  status: "collecting" | "measurable";
  event_counts: Record<"impression" | "open" | "clip_created" | "compilation_add", number>;
  positive_action_count: number;
  reviewed_clip_count: number;
  exported_clip_count: number;
  compilation_clip_count: number;
  measurement_floor_met: boolean;
  live_weight: 0;
  eligible_for_promotion: false;
  promotion_blockers: string[];
  latest_run: MemoryLearningRun | null;
}

export interface MemorySemanticReindexResponse {
  status: "ready" | "stale";
  entries_indexed: number;
  dimensions: number;
  index_bytes: number;
  model_id: string;
}

export interface MemoryReindexResponse {
  indexed_jobs: number;
  skipped_jobs: number;
  failed_jobs: number;
  transcript_jobs: number;
  evidence_jobs?: number;
  evidence_entries?: number;
  entries_written: number;
  errors: Array<{ job_id: string; error: string }>;
}

async function jsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : "Stream Memory request failed.";
    throw new Error(detail);
  }
  return body as T;
}

export async function getMemoryStats(baseUrl?: string): Promise<MemoryStats> {
  return jsonOrThrow<MemoryStats>(
    await apiFetch("/memory/stats", undefined, baseUrl),
  );
}

export async function searchStreamMemory(input: {
  query: string;
  kind?: MemoryKind;
  decision?: MemoryDecision;
  exported?: MemoryExportFilter;
  origin?: MemoryOriginFilter;
  game?: string;
  dateFrom?: string;
  dateTo?: string;
  limit?: number;
  mode?: MemorySearchMode;
  baseUrl?: string;
}): Promise<MemorySearchResponse> {
  const params = new URLSearchParams({
    q: input.query,
    kind: input.kind ?? "all",
    decision: input.decision ?? "all",
    exported: input.exported ?? "all",
    origin: input.origin ?? "all",
    limit: String(input.limit ?? 40),
    mode: input.mode ?? "hybrid",
  });
  if (input.game) params.set("game", input.game);
  if (input.dateFrom) params.set("date_from", input.dateFrom);
  if (input.dateTo) params.set("date_to", input.dateTo);
  return jsonOrThrow<MemorySearchResponse>(
    await apiFetch(`/memory/search?${params.toString()}`, undefined, input.baseUrl),
  );
}

export async function recordMemoryInteractions(input: {
  searchId: string;
  eventType: MemoryInteractionType;
  entryIds: string[];
  baseUrl?: string;
}): Promise<{ search_id: string; event_type: MemoryInteractionType; recorded: number }> {
  return jsonOrThrow(await apiFetch("/memory/interactions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      search_id: input.searchId,
      event_type: input.eventType,
      entries: input.entryIds.map((entryId) => ({ entry_id: entryId })),
    }),
  }, input.baseUrl));
}

export async function getMemoryAliases(baseUrl?: string): Promise<MemoryAlias[]> {
  const response = await jsonOrThrow<{ aliases: MemoryAlias[] }>(
    await apiFetch("/memory/aliases", undefined, baseUrl),
  );
  return response.aliases;
}

export async function addMemoryAlias(
  canonicalTerm: string,
  alias: string,
  baseUrl?: string,
): Promise<MemoryAlias> {
  return jsonOrThrow<MemoryAlias>(await apiFetch("/memory/aliases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ canonical_term: canonicalTerm, alias }),
  }, baseUrl));
}

export async function deleteMemoryAlias(id: string, baseUrl?: string): Promise<void> {
  await jsonOrThrow<{ deleted: boolean }>(
    await apiFetch(`/memory/aliases/${encodeURIComponent(id)}`, { method: "DELETE" }, baseUrl),
  );
}

export async function setMemoryFeedback(
  query: string,
  entryId: string,
  verdict: "relevant" | "not_relevant" | "clear",
  baseUrl?: string,
): Promise<MemoryFeedbackResponse> {
  return jsonOrThrow<MemoryFeedbackResponse>(await apiFetch("/memory/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, entry_id: entryId, verdict }),
  }, baseUrl));
}

export async function getMemoryEvaluation(baseUrl?: string): Promise<MemoryEvaluationSummary> {
  return jsonOrThrow<MemoryEvaluationSummary>(
    await apiFetch("/memory/evaluation", undefined, baseUrl),
  );
}

export async function getMemoryLearning(baseUrl?: string): Promise<MemoryLearningSummary> {
  return jsonOrThrow<MemoryLearningSummary>(
    await apiFetch("/memory/learning", undefined, baseUrl),
  );
}

export async function runMemoryLearningEvaluation(baseUrl?: string): Promise<MemoryLearningRun> {
  return jsonOrThrow<MemoryLearningRun>(
    await apiFetch("/memory/learning/evaluate", { method: "POST" }, baseUrl),
  );
}

export async function clearMemoryLearning(baseUrl?: string): Promise<{
  cleared: true;
  searches: number;
  interactions: number;
  shadow_runs: number;
}> {
  return jsonOrThrow(
    await apiFetch("/memory/learning", { method: "DELETE" }, baseUrl),
  );
}

export async function saveMemoryMissedQuery(
  query: string,
  expectedDescription: string,
  baseUrl?: string,
): Promise<MemoryEvaluationCase> {
  return jsonOrThrow<MemoryEvaluationCase>(await apiFetch("/memory/evaluation/misses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, expected_description: expectedDescription }),
  }, baseUrl));
}

export async function deleteMemoryEvaluationCase(id: string, baseUrl?: string): Promise<void> {
  await jsonOrThrow<{ deleted: boolean }>(
    await apiFetch(`/memory/evaluation/cases/${encodeURIComponent(id)}`, { method: "DELETE" }, baseUrl),
  );
}

export async function resolveMemoryEvaluationCase(
  id: string,
  entryId: string,
  baseUrl?: string,
): Promise<MemoryEvaluationCase> {
  return jsonOrThrow<MemoryEvaluationCase>(await apiFetch(
    `/memory/evaluation/cases/${encodeURIComponent(id)}/target`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entry_id: entryId }),
    },
    baseUrl,
  ));
}

export async function runMemoryEvaluation(baseUrl?: string): Promise<MemoryEvaluationRun> {
  return jsonOrThrow<MemoryEvaluationRun>(
    await apiFetch("/memory/evaluation/run", { method: "POST" }, baseUrl),
  );
}

export async function reindexStreamMemory(
  force = false,
  baseUrl?: string,
): Promise<MemoryReindexResponse> {
  return jsonOrThrow<MemoryReindexResponse>(await apiFetch("/memory/reindex", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force }),
  }, baseUrl));
}

export async function reindexStreamMemorySemantics(
  baseUrl?: string,
): Promise<MemorySemanticReindexResponse> {
  return jsonOrThrow<MemorySemanticReindexResponse>(
    await apiFetch("/memory/semantic/reindex", { method: "POST" }, baseUrl),
  );
}

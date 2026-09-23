// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { apiFetch } from "./api";

export type RecallSessionStatus = "active" | "ended";

export interface RecallSessionEvent {
  id: string;
  session_id: string;
  kind: string;
  source: string;
  occurred_at_utc: string;
  stream_offset_seconds: number | null;
  confidence: number;
  payload: Record<string, unknown>;
  schema_version: number;
}

export interface RecallClockAnchor {
  id: string;
  session_id: string;
  anchor_type: string;
  wall_time_utc: string;
  stream_time_seconds: number;
  uncertainty_seconds: number;
  source: string;
  schema_version: number;
}

export interface RecallSession {
  id: string;
  status: RecallSessionStatus;
  source_platform: string;
  source_ref: string | null;
  title: string | null;
  started_at_utc: string;
  ended_at_utc: string | null;
  metadata: Record<string, unknown>;
  schema_version: number;
  events?: RecallSessionEvent[];
  clock_anchors?: RecallClockAnchor[];
}

export interface RecallRegionMarker {
  event_id: string;
  timestamp: number;
  search_start: number;
  search_end: number;
  alignment: string;
  anchor_id: string | null;
  uncertainty_seconds: number;
  confidence: number;
  source: string;
}

export interface RecallRegionPlan {
  schema_version: number;
  session_id: string;
  strategy: "creator_markers_then_deep_recall";
  alignment_status: "ready" | "partial" | "unmapped";
  vod_started_at_utc: string | null;
  vod_duration_seconds: number | null;
  pre_roll_seconds: number;
  post_roll_seconds: number;
  markers: RecallRegionMarker[];
  regions: Array<{
    start: number;
    end: number;
    priority: "creator";
    event_ids: string[];
    marker_times: number[];
  }>;
  mapped_event_count: number;
  unmapped_event_ids: string[];
  omitted_event_count: number;
  total_region_seconds: number;
}

async function jsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : "Recall Session request failed.";
    throw new Error(detail);
  }
  return body as T;
}

export async function startRecallSession(input: {
  sourcePlatform?: string;
  sourceRef?: string;
  title?: string;
  startedAtUtc?: string;
  initialStreamTimeSeconds?: number;
  metadata?: Record<string, unknown>;
} = {}): Promise<RecallSession> {
  const response = await apiFetch("/recall-sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_platform: input.sourcePlatform ?? "unknown",
      source_ref: input.sourceRef,
      title: input.title,
      started_at_utc: input.startedAtUtc,
      initial_stream_time_seconds: input.initialStreamTimeSeconds,
      metadata: input.metadata ?? {},
    }),
  });
  return jsonOrThrow<RecallSession>(response);
}

/** Every session ever recorded, newest first.
 *
 * The route has always existed; nothing called it, so an ended session with
 * marks on it left the interface for good the moment a new one started. The
 * marks were never lost — there was simply no way back to them. */
export async function listRecallSessions(limit = 50): Promise<RecallSession[]> {
  const response = await apiFetch(`/recall-sessions?limit=${encodeURIComponent(String(limit))}`);
  const body = await jsonOrThrow<RecallSession[] | { sessions?: RecallSession[] }>(response);
  return Array.isArray(body) ? body : (body.sessions ?? []);
}

export async function getActiveRecallSession(): Promise<RecallSession | null> {
  const response = await apiFetch("/recall-sessions/active");
  const body = await jsonOrThrow<{ session: RecallSession | null }>(response);
  return body.session;
}

export async function getRecallSession(sessionId: string): Promise<RecallSession> {
  const response = await apiFetch(`/recall-sessions/${encodeURIComponent(sessionId)}`);
  return jsonOrThrow<RecallSession>(response);
}

export async function stopRecallSession(
  sessionId: string,
  endedAtUtc?: string,
): Promise<RecallSession> {
  const response = await apiFetch(`/recall-sessions/${encodeURIComponent(sessionId)}/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ended_at_utc: endedAtUtc }),
  });
  return jsonOrThrow<RecallSession>(response);
}

export async function rememberMoment(input: {
  sessionId: string;
  occurredAtUtc?: string;
  streamOffsetSeconds?: number;
  source?: string;
  confidence?: number;
  payload?: Record<string, unknown>;
}): Promise<RecallSessionEvent> {
  const response = await apiFetch(
    `/recall-sessions/${encodeURIComponent(input.sessionId)}/events`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kind: "remember",
        source: input.source ?? "manual",
        occurred_at_utc: input.occurredAtUtc,
        stream_offset_seconds: input.streamOffsetSeconds,
        confidence: input.confidence ?? 1,
        payload: input.payload ?? {},
      }),
    },
  );
  return jsonOrThrow<RecallSessionEvent>(response);
}

/** Drop one mark. A misfire costs a two-minute search window in the scan, so
 * it has to be removable — on an ended session too, which is when the creator
 * is actually reading back what they marked. */
export async function deleteRecallEvent(input: {
  sessionId: string;
  eventId: string;
}): Promise<void> {
  const response = await apiFetch(
    `/recall-sessions/${encodeURIComponent(input.sessionId)}`
    + `/events/${encodeURIComponent(input.eventId)}`,
    { method: "DELETE" },
  );
  await jsonOrThrow<{ deleted: string }>(response);
}

export async function addRecallClockAnchor(input: {
  sessionId: string;
  wallTimeUtc: string;
  streamTimeSeconds: number;
  anchorType?: string;
  uncertaintySeconds?: number;
  source?: string;
}): Promise<RecallClockAnchor> {
  const response = await apiFetch(
    `/recall-sessions/${encodeURIComponent(input.sessionId)}/clock-anchors`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        wall_time_utc: input.wallTimeUtc,
        stream_time_seconds: input.streamTimeSeconds,
        anchor_type: input.anchorType ?? "player_position",
        uncertainty_seconds: input.uncertaintySeconds ?? 1,
        source: input.source ?? "player",
      }),
    },
  );
  return jsonOrThrow<RecallClockAnchor>(response);
}

export async function buildRecallRegionPlan(input: {
  sessionId: string;
  vodStartedAtUtc?: string;
  vodDurationSeconds?: number;
  preRollSeconds?: number;
  postRollSeconds?: number;
}): Promise<RecallRegionPlan> {
  const response = await apiFetch(
    `/recall-sessions/${encodeURIComponent(input.sessionId)}/region-plan`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vod_started_at_utc: input.vodStartedAtUtc,
        vod_duration_seconds: input.vodDurationSeconds,
        pre_roll_seconds: input.preRollSeconds ?? 90,
        post_roll_seconds: input.postRollSeconds ?? 30,
      }),
    },
  );
  return jsonOrThrow<RecallRegionPlan>(response);
}

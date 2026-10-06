// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The scan log: the running feed the creator watched live, kept with the
 * session. Fetched the first time it's opened (or on mount when `open`), so a
 * review stays cheap.
 */
import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "../lib/api";
import { Check, ChevronDown, Layers } from "../lib/icons";
import { mapEvent, useJobStore, type JobEvent } from "../lib/store";
import { activityFromEvents, type ActivityItem } from "./scanModel";

export function useScanLog(jobId: string | undefined) {
  const [items, setItems] = useState<ActivityItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => { setItems(null); setFailed(false); }, [jobId]);
  // One try per job: a missing log stays missing rather than being asked
  // for again on every render.
  const load = useCallback(() => {
    if (!jobId || items !== null || loading || failed) return;
    setLoading(true);
    setFailed(false);
    apiFetch(`/jobs/${jobId}`, undefined, useJobStore.getState().settings.apiEndpoint)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error("scan log unavailable"))))
      .then((data) => {
        const raw = Array.isArray(data.recent_events) ? data.recent_events : [];
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const events = raw.map((ev: any) => mapEvent(jobId, ev)).sort((a: JobEvent, b: JobEvent) => b.at - a.at);
        setItems(activityFromEvents(events, jobId, 60));
      })
      .catch(() => setFailed(true))
      .finally(() => setLoading(false));
  }, [jobId, items, loading, failed]);
  return { items, loading, failed, load };
}

export function ActivityList({ items }: { items: ActivityItem[] }) {
  return (
    <ol className="feed">
      {items.map((item) => (
        <li key={item.key} className={item.ok ? "ok" : ""}>
          <span className="feed-mark" aria-hidden="true">{item.ok && <Check />}</span>
          <time className="num">{item.timestamp}</time>
          <span>{item.text}</span>
        </li>
      ))}
    </ol>
  );
}

export function ScanLog({ jobId }: { jobId: string }) {
  const { items, loading, failed, load } = useScanLog(jobId);
  return (
    <details className="scanlog glass" onToggle={(event) => { if (event.currentTarget.open) load(); }}>
      <summary>
        <Layers aria-hidden="true" />
        <b>Scan log</b>
        <span className="t3">What Recall did on this scan</span>
        <ChevronDown className="scanlog-chev" aria-hidden="true" />
      </summary>
      <div className="scanlog-body">
        {loading && <p className="t3">Loading the scan history…</p>}
        {failed && <p className="t3">The scan history isn't available for this session.</p>}
        {items && items.length > 0 && <ActivityList items={items} />}
        {items && items.length === 0 && !loading && <p className="t3">No scan activity was recorded for this session.</p>}
      </div>
    </details>
  );
}

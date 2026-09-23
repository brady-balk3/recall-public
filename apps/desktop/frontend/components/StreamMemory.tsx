// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { apiUrl } from "../lib/api";
import { AlertTriangle, BarChart3, Check, Clock, Film, HardDrive, Play, Plus, RefreshCw, RotateCcw, Scissors, Search, Settings, Sparkles, Target, Trash2, Video, X } from "../lib/icons";
import {
  addClipToCompilationProject,
  listCompilationProjects,
  type CompilationProjectSummary,
} from "../lib/compilationProjects";
import {
  addMemoryAlias,
  deleteMemoryEvaluationCase,
  deleteMemoryAlias,
  getMemoryAliases,
  getMemoryEvaluation,
  getMemoryStats,
  reindexStreamMemory,
  reindexStreamMemorySemantics,
  recordMemoryInteractions,
  resolveMemoryEvaluationCase,
  runMemoryEvaluation,
  saveMemoryMissedQuery,
  searchStreamMemory,
  setMemoryFeedback,
  type MemoryAlias,
  type MemoryAliasExpansion,
  type MemoryCompoundClause,
  type MemoryDecision,
  type MemoryExportFilter,
  type MemoryEvaluationSummary,
  type MemoryKind,
  type MemoryOriginFilter,
  type MemoryResult,
  type MemorySearchResponse,
  type MemorySearchMode,
  type MemoryStats,
} from "../lib/streamMemory";
import {
  parseMemoryRefinement,
  type MemorySearchContext,
} from "../lib/memoryRefinement";
import { fmtClock } from "../lib/format";

const STARTER_QUERIES = [
  "funny reactions with friends",
  "close calls I somehow survived",
  "scary moments that made me scream",
];

type MemorySort = "relevance" | "clips" | "newest" | "oldest";
const RESULT_BATCH_SIZE = 12;

function sortMemoryResults(results: MemoryResult[], sort: MemorySort): MemoryResult[] {
  return results
    .map((result, index) => ({ result, index }))
    .sort((left, right) => {
      if (sort === "clips") {
        const kindOrder = Number(right.result.kind === "clip") - Number(left.result.kind === "clip");
        return kindOrder || left.index - right.index;
      }
      if (sort === "newest" || sort === "oldest") {
        const leftDate = left.result.source_date ?? "";
        const rightDate = right.result.source_date ?? "";
        const dateOrder = leftDate.localeCompare(rightDate);
        return (sort === "newest" ? -dateOrder : dateOrder) || left.index - right.index;
      }
      return left.index - right.index;
    })
    .map(({ result }) => result);
}

function resultDecision(result: MemoryResult): string | null {
  if (result.kept) return "Kept";
  if (result.passed) return "Passed";
  if (result.maybe) return "Maybe";
  return result.kind === "clip" ? "Unreviewed" : null;
}

function resultDate(value: string | null): string {
  if (!value) return "Date unavailable";
  const date = new Date(`${value}T12:00:00`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function resultContext(result: MemoryResult): string {
  const context = result.match_context || (result.kind === "clip" ? result.text || result.title : result.text);
  const evidenceText = result.evidence?.search_text?.trim();
  if (result.kind === "clip" && evidenceText && context.endsWith(evidenceText)) {
    return context.slice(0, -evidenceText.length).trim() || result.title;
  }
  return context;
}

function resultType(result: MemoryResult): string {
  if (result.origin === "creator_marker") return "Marked live";
  if (result.kind === "evidence") return "Scan evidence";
  return result.kind === "clip" ? "Saved clip" : "Conversation";
}

function matchSourceLabel(result: MemoryResult): string {
  if (result.match_reason) return result.match_reason;
  switch (result.match_source) {
    case "creator_speech": return "Creator speech";
    case "clip_details": return "Clip details";
    case "creator_marker": return "Marked live";
    case "gameplay_event": return "Gameplay or reaction signal";
    case "chat_activity": return "Twitch chat activity";
    case "visual_description": return "Visual scene description";
    case "on_screen_text": return "On-screen text";
    default: return "Related meaning";
  }
}

function matchSourceClass(result: MemoryResult): string {
  return `is-${result.match_source ?? "legacy"}`;
}

function evidenceClues(result: MemoryResult): string[] {
  const evidence = result.evidence;
  if (!evidence) return [];
  const visualClues = (evidence.visual_scenes ?? [])
    .map(({ description }) => description?.trim() ?? "")
    .filter(Boolean);
  const chat = evidence.chat_activity;
  const chatClues = [
    ...(chat?.repeated_phrases ?? []).map(({ text, count }) => {
      const phrase = text?.trim();
      if (!phrase) return "";
      return count && count > 1 ? `${phrase} ×${count}` : phrase;
    }),
    chat?.message_count ? `${chat.message_count} chat messages` : "",
    chat?.clip_intent_count ? `${chat.clip_intent_count} viewer clip request${chat.clip_intent_count === 1 ? "" : "s"}` : "",
  ];
  return Array.from(new Set([
    ...visualClues,
    ...chatClues,
    ...(evidence.gameplay_events ?? []),
    ...(evidence.reactions ?? []),
    ...(evidence.labels ?? []),
    ...(evidence.ocr_phrases ?? []),
  ].map((value) => value.trim()).filter(Boolean))).slice(0, 8);
}

function interpretedFilterLabel(field: string, value: string): string {
  if (field === "kind") return "Clips";
  if (field === "origin") return "Marked live";
  if (field === "exported") return value === "exported" ? "Exported" : "Not exported";
  if (field === "decision") return value.charAt(0).toUpperCase() + value.slice(1);
  if (field === "sort") return "Best first";
  if (field === "date") return "Date understood";
  return value;
}

function formatBytes(bytes = 0): string {
  if (bytes < 1024 * 1024) return `${Math.max(0, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatBuildElapsed(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

function semanticLabel(stats: MemoryStats | null): string {
  switch (stats?.semantic_status) {
    case "ready": return "Concept search ready";
    case "stale": return "New memories to learn";
    case "building": return "Learning your archive";
    case "error": return "Concept search needs attention";
    default: return "Concept search not built";
  }
}

interface RejectedCorrection {
  result: MemoryResult;
  index: number;
  query: string;
}

interface MemorySearchSnapshot {
  results: MemoryResult[];
  selectedId?: string;
  searchedQuery: string;
  context: MemorySearchContext;
  modeUsed: MemorySearchMode;
  aliasExpansions: MemoryAliasExpansion[];
  interpretedFilters: Array<{ field: string; value: string }>;
  compoundClauses: MemoryCompoundClause[];
  refinementHistory: string[];
  searchId?: string;
}

function responseContext(
  response: MemorySearchResponse,
  requested: MemorySearchContext,
): MemorySearchContext {
  const resolved = response.resolved_filters;
  return {
    query: response.effective_query?.trim() || requested.query,
    kind: resolved?.kind ?? requested.kind,
    decision: resolved?.decision ?? requested.decision,
    exported: resolved?.exported ?? requested.exported,
    origin: resolved?.origin ?? requested.origin,
    game: resolved?.game ?? requested.game,
    dateFrom: resolved?.date_from ?? requested.dateFrom,
    dateTo: resolved?.date_to ?? requested.dateTo,
    mode: requested.mode,
  };
}

export default function StreamMemory({
  apiEndpoint,
  onOpenClip,
  onOpenMoment,
  onOpenCompilations,
}: {
  apiEndpoint?: string;
  onOpenClip: (jobId: string, clipId: string) => void | Promise<void>;
  onOpenMoment?: (result: MemoryResult, query: string) => void | Promise<void>;
  onOpenCompilations?: () => void;
}) {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [query, setQuery] = useState("");
  const [searchedQuery, setSearchedQuery] = useState("");
  const [searchId, setSearchId] = useState<string>();
  const [kind, setKind] = useState<MemoryKind>("all");
  const [decision, setDecision] = useState<MemoryDecision>("all");
  const [mode, setMode] = useState<MemorySearchMode>("hybrid");
  const [modeUsed, setModeUsed] = useState<MemorySearchMode>("keyword");
  const [searchContext, setSearchContext] = useState<MemorySearchContext | null>(null);
  const [refinement, setRefinement] = useState("");
  const [refining, setRefining] = useState(false);
  const [refinementError, setRefinementError] = useState<string>();
  const [refinementNotice, setRefinementNotice] = useState<string>();
  const [refinementHistory, setRefinementHistory] = useState<string[]>([]);
  const [refinementStack, setRefinementStack] = useState<MemorySearchSnapshot[]>([]);
  const [sort, setSort] = useState<MemorySort>("relevance");
  const [visibleCount, setVisibleCount] = useState(RESULT_BATCH_SIZE);
  const [results, setResults] = useState<MemoryResult[]>([]);
  const [aliasExpansions, setAliasExpansions] = useState<MemoryAliasExpansion[]>([]);
  const [interpretedFilters, setInterpretedFilters] = useState<Array<{ field: string; value: string }>>([]);
  const [compoundClauses, setCompoundClauses] = useState<MemoryCompoundClause[]>([]);
  const [aliases, setAliases] = useState<MemoryAlias[]>([]);
  const [evaluation, setEvaluation] = useState<MemoryEvaluationSummary | null>(null);
  const [selectedId, setSelectedId] = useState<string>();
  const [searching, setSearching] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [buildingConcepts, setBuildingConcepts] = useState(false);
  const [buildElapsedSeconds, setBuildElapsedSeconds] = useState(0);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [toolsTab, setToolsTab] = useState<"archive" | "vocabulary" | "evaluation">("archive");
  const [canonicalTerm, setCanonicalTerm] = useState("");
  const [aliasTerm, setAliasTerm] = useState("");
  const [savingAlias, setSavingAlias] = useState(false);
  const [expectedMemory, setExpectedMemory] = useState("");
  const [savingMiss, setSavingMiss] = useState(false);
  const [runningEvaluation, setRunningEvaluation] = useState(false);
  const [teachingResultId, setTeachingResultId] = useState<string>();
  const [compilationPickerOpen, setCompilationPickerOpen] = useState(false);
  const [compilationProjects, setCompilationProjects] = useState<CompilationProjectSummary[]>([]);
  const [compilationProjectId, setCompilationProjectId] = useState("");
  const [compilationBusy, setCompilationBusy] = useState(false);
  const [compilationError, setCompilationError] = useState<string>();
  const [rejectedCorrection, setRejectedCorrection] = useState<RejectedCorrection>();
  const [message, setMessage] = useState<string>();
  // Each background loader owns its own key. They used to share `message`, so
  // three failures on mount left whichever resolved last as the only visible
  // one -- and a later success never cleared it.
  const [unavailable, setUnavailable] = useState<Partial<Record<"archive" | "vocabulary" | "searchTest", true>>>({});
  const videoRef = useRef<HTMLVideoElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const recordedImpressionsRef = useRef(new Set<string>());
  const selected = useMemo(
    () => results.find((result) => result.id === selectedId) ?? results[0],
    [results, selectedId],
  );
  const sortedResults = useMemo(() => sortMemoryResults(results, sort), [results, sort]);
  const degradedTools = [
    unavailable.vocabulary ? "Creator vocabulary" : "",
    unavailable.searchTest ? "the search test" : "",
  ].filter(Boolean);
  const visibleResults = sortedResults.slice(0, visibleCount);

  useEffect(() => {
    if (!searchId || !visibleResults.length) return;
    const entryIds = visibleResults
      .map((result) => result.id)
      .filter((entryId) => !recordedImpressionsRef.current.has(`${searchId}:${entryId}`));
    if (!entryIds.length) return;
    entryIds.forEach((entryId) => recordedImpressionsRef.current.add(`${searchId}:${entryId}`));
    void recordMemoryInteractions({
      searchId,
      eventType: "impression",
      entryIds,
      baseUrl: apiEndpoint,
    }).catch(() => undefined);
  }, [apiEndpoint, searchId, visibleResults]);

  const recordOpen = (result: MemoryResult) => {
    if (!searchId) return;
    void recordMemoryInteractions({
      searchId,
      eventType: "open",
      entryIds: [result.id],
      baseUrl: apiEndpoint,
    }).catch(() => undefined);
  };

  const selectResult = (result: MemoryResult) => {
    setSelectedId(result.id);
    setCompilationPickerOpen(false);
    setCompilationError(undefined);
  };

  const navigateResults = (
    event: KeyboardEvent<HTMLButtonElement>,
    currentIndex: number,
  ) => {
    let nextIndex: number | undefined;
    if (event.key === "ArrowDown") nextIndex = Math.min(currentIndex + 1, visibleResults.length - 1);
    if (event.key === "ArrowUp") nextIndex = Math.max(currentIndex - 1, 0);
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = visibleResults.length - 1;
    if (nextIndex === undefined || nextIndex === currentIndex) return;
    event.preventDefault();
    selectResult(visibleResults[nextIndex]);
    window.requestAnimationFrame(() => {
      document.getElementById(`memory-result-${nextIndex}`)?.focus();
    });
  };

  const markAvailability = (key: "archive" | "vocabulary" | "searchTest", ok: boolean) => {
    setUnavailable((current) => {
      if (ok === !current[key]) return current;
      const next = { ...current };
      if (ok) delete next[key];
      else next[key] = true;
      return next;
    });
  };

  const TOOL_TABS = ["archive", "vocabulary", "evaluation"] as const;

  const navigateToolTabs = (event: KeyboardEvent<HTMLDivElement>) => {
    const current = TOOL_TABS.indexOf(toolsTab);
    let next: number | undefined;
    if (event.key === "ArrowRight") next = (current + 1) % TOOL_TABS.length;
    if (event.key === "ArrowLeft") next = (current - 1 + TOOL_TABS.length) % TOOL_TABS.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = TOOL_TABS.length - 1;
    if (next === undefined || next === current) return;
    event.preventDefault();
    setToolsTab(TOOL_TABS[next]);
    window.requestAnimationFrame(() => {
      document.getElementById(`memory-tools-tab-${TOOL_TABS[next!]}`)?.focus();
    });
  };

  const refreshStats = async () => {
    try {
      setStats(await getMemoryStats(apiEndpoint));
      markAvailability("archive", true);
    } catch {
      markAvailability("archive", false);
    }
  };

  const refreshAliases = async () => {
    try {
      setAliases(await getMemoryAliases(apiEndpoint));
      markAvailability("vocabulary", true);
    } catch {
      markAvailability("vocabulary", false);
    }
  };

  const refreshEvaluation = async () => {
    try {
      setEvaluation(await getMemoryEvaluation(apiEndpoint));
      markAvailability("searchTest", true);
    } catch {
      markAvailability("searchTest", false);
    }
  };

  const retryUnavailable = () => {
    void refreshStats();
    void refreshAliases();
    void refreshEvaluation();
  };

  useEffect(() => {
    void refreshStats();
    void refreshAliases();
    void refreshEvaluation();
  }, [apiEndpoint]);

  useEffect(() => {
    if (!buildingConcepts) return undefined;
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setBuildElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [buildingConcepts]);

  const runSearch = async (searchQuery: string) => {
    const trimmed = searchQuery.trim();
    if (!trimmed) {
      setMessage("Search for a name, phrase, game, or moment.");
      return;
    }
    setSearching(true);
    setMessage(undefined);
    setRejectedCorrection(undefined);
    setRefinementError(undefined);
    setRefinementNotice(undefined);
    const requestedContext: MemorySearchContext = {
      query: trimmed,
      kind,
      decision,
      exported: "all" as MemoryExportFilter,
      origin: "all" as MemoryOriginFilter,
      game: null,
      dateFrom: null,
      dateTo: null,
      mode,
    };
    try {
      const response = await searchStreamMemory({
        query: requestedContext.query,
        kind: requestedContext.kind,
        decision: requestedContext.decision,
        exported: requestedContext.exported,
        origin: requestedContext.origin,
        game: requestedContext.game ?? undefined,
        dateFrom: requestedContext.dateFrom ?? undefined,
        dateTo: requestedContext.dateTo ?? undefined,
        mode: requestedContext.mode,
        baseUrl: apiEndpoint,
      });
      setResults(response.results);
      recordedImpressionsRef.current.clear();
      setSearchId(response.search_id ?? undefined);
      setSelectedId(response.results[0]?.id);
      setSearchedQuery(trimmed);
      setModeUsed(response.mode_used);
      setSearchContext(responseContext(response, requestedContext));
      setAliasExpansions(response.alias_expansions ?? []);
      setInterpretedFilters(response.interpreted_filters ?? []);
      setCompoundClauses(response.compound_query?.active ? response.compound_query.clauses : []);
      setVisibleCount(RESULT_BATCH_SIZE);
      setRefinement("");
      setRefinementHistory([]);
      setRefinementStack([]);
      if (mode === "hybrid" && response.mode_used === "keyword") {
        setMessage("Exact-word search was used. Build the concept index to search by meaning too.");
      }
    } catch {
      setMessage(
        results.length
          ? "Search couldn’t finish. Check that Recall is running, then try again. Your previous results are unchanged."
          : "Search couldn’t finish. Check that Recall is running, then try again.",
      );
    } finally {
      setSearching(false);
    }
  };

  const search = async (event?: FormEvent) => {
    event?.preventDefault();
    await runSearch(query);
  };

  const useStarterQuery = (value: string) => {
    setQuery(value);
    void runSearch(value);
  };

  const refineResults = async (event: FormEvent) => {
    event.preventDefault();
    if (!searchContext || refining) return;
    const plan = parseMemoryRefinement({
      instruction: refinement,
      context: searchContext,
      selected,
      resultCount: sortedResults.length,
    });
    setRefinementError(undefined);
    setRefinementNotice(undefined);
    if (plan.type === "error") {
      setRefinementError(plan.message);
      return;
    }
    if (plan.type === "select") {
      const result = sortedResults[plan.index];
      if (!result) {
        setRefinementError("That result is no longer in the current list.");
        return;
      }
      setSelectedId(result.id);
      setCompilationPickerOpen(false);
      setCompilationError(undefined);
      setVisibleCount((count) => Math.max(count, plan.index + 1));
      setRefinement("");
      setRefinementNotice(`${plan.label} of ${sortedResults.length}.`);
      return;
    }

    const snapshot: MemorySearchSnapshot = {
      results,
      selectedId: selected?.id,
      searchedQuery,
      context: searchContext,
      modeUsed,
      aliasExpansions,
      interpretedFilters,
      compoundClauses,
      refinementHistory,
      searchId,
    };
    setRefining(true);
    try {
      const response = await searchStreamMemory({
        query: plan.context.query,
        kind: plan.context.kind,
        decision: plan.context.decision,
        exported: plan.context.exported,
        origin: plan.context.origin,
        game: plan.context.game ?? undefined,
        dateFrom: plan.context.dateFrom ?? undefined,
        dateTo: plan.context.dateTo ?? undefined,
        mode: plan.context.mode,
        baseUrl: apiEndpoint,
      });
      const nextResults = response.results.filter((result) => result.id !== plan.excludeEntryId);
      if (!nextResults.length) {
        setRefinementError(`No results matched “${plan.label}”. Your previous results are unchanged.`);
        return;
      }
      const nextContext = responseContext(response, plan.context);
      const nextHistory = [...refinementHistory, plan.label].slice(-5);
      setRefinementStack((current) => [...current, snapshot].slice(-5));
      setResults(nextResults);
      recordedImpressionsRef.current.clear();
      setSearchId(response.search_id ?? undefined);
      setSelectedId(nextResults[0].id);
      setCompilationPickerOpen(false);
      setCompilationError(undefined);
      setSearchedQuery(`${searchedQuery} · ${plan.label}`.slice(0, 240));
      setSearchContext(nextContext);
      setMode(nextContext.mode);
      setKind(nextContext.kind);
      setDecision(nextContext.decision);
      setModeUsed(response.mode_used);
      setAliasExpansions(response.alias_expansions ?? []);
      setInterpretedFilters(response.interpreted_filters ?? []);
      setCompoundClauses(response.compound_query?.active ? response.compound_query.clauses : []);
      setVisibleCount(RESULT_BATCH_SIZE);
      setRefinementHistory(nextHistory);
      setRefinement("");
      setRefinementNotice(`${nextResults.length} result${nextResults.length === 1 ? "" : "s"} after ${plan.label.toLowerCase()}.`);
      setRejectedCorrection(undefined);
      setMessage(undefined);
    } catch (error) {
      setRefinementError(
        error instanceof Error
          ? `${error.message} Your previous results are unchanged.`
          : "Recall could not refine these results. Your previous results are unchanged.",
      );
    } finally {
      setRefining(false);
    }
  };

  const undoRefinement = () => {
    const snapshot = refinementStack[refinementStack.length - 1];
    if (!snapshot) return;
    setResults(snapshot.results);
    recordedImpressionsRef.current.clear();
    setSearchId(snapshot.searchId);
    setSelectedId(snapshot.selectedId);
    setSearchedQuery(snapshot.searchedQuery);
    setSearchContext(snapshot.context);
    setMode(snapshot.context.mode);
    setKind(snapshot.context.kind);
    setDecision(snapshot.context.decision);
    setModeUsed(snapshot.modeUsed);
    setAliasExpansions(snapshot.aliasExpansions);
    setInterpretedFilters(snapshot.interpretedFilters);
    setCompoundClauses(snapshot.compoundClauses);
    setRefinementHistory(snapshot.refinementHistory);
    setRefinementStack((current) => current.slice(0, -1));
    setRefinementError(undefined);
    setRefinementNotice("Last refinement undone.");
    setVisibleCount(RESULT_BATCH_SIZE);
  };

  const indexLibrary = async () => {
    setIndexing(true);
    setMessage(undefined);
    try {
      const result = await reindexStreamMemory(false, apiEndpoint);
      await Promise.all([refreshStats(), refreshEvaluation()]);
      setMessage(
        result.indexed_jobs > 0
          ? `Added ${result.indexed_jobs} session${result.indexed_jobs === 1 ? "" : "s"} and ${result.entries_written} searchable moments.`
          : "Every completed session is already in your archive.",
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Library indexing failed.");
    } finally {
      setIndexing(false);
    }
  };

  const buildConcepts = async () => {
    setBuildElapsedSeconds(0);
    setBuildingConcepts(true);
    setMessage("Updating concept search locally. This usually takes 3–6 minutes; you can keep searching while it runs.");
    try {
      const result = await reindexStreamMemorySemantics(apiEndpoint);
      await Promise.all([refreshStats(), refreshEvaluation()]);
      setMessage(
        `Concept search learned ${result.entries_indexed.toLocaleString()} moments in ${formatBytes(result.index_bytes)}.`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Concept indexing failed.");
    } finally {
      setBuildingConcepts(false);
    }
  };

  const saveFailedSearch = async (event: FormEvent) => {
    event.preventDefault();
    const expected = expectedMemory.trim();
    if (!searchedQuery || !expected) {
      setMessage("Describe the moment Recall should have found.");
      return;
    }
    setSavingMiss(true);
    setMessage(undefined);
    try {
      await saveMemoryMissedQuery(searchedQuery, expected, apiEndpoint);
      await refreshEvaluation();
      setExpectedMemory("");
      setMessage("Failed search saved. It will count honestly against the current baseline.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "That failed search could not be saved.");
    } finally {
      setSavingMiss(false);
    }
  };

  const runBaseline = async () => {
    setRunningEvaluation(true);
    setMessage("Measuring the current search without using your relevance boosts…");
    try {
      const run = await runMemoryEvaluation(apiEndpoint);
      await refreshEvaluation();
      setMessage(
        `Baseline measured ${run.case_count} search${run.case_count === 1 ? "" : "es"}: ${Math.round(run.metrics.top_5_rate * 100)}% found a relevant memory in the first five.`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "The Memory baseline could not be measured.");
    } finally {
      setRunningEvaluation(false);
    }
  };

  const removeEvaluationCase = async (caseId: string) => {
    setMessage(undefined);
    try {
      await deleteMemoryEvaluationCase(caseId, apiEndpoint);
      await refreshEvaluation();
      setMessage("Removed that search from the test set. Your archive was not changed.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "That test search could not be removed.");
    }
  };

  const resolveEvaluationCase = async (caseId: string) => {
    if (!selected) return;
    setMessage(undefined);
    try {
      const resolved = await resolveMemoryEvaluationCase(caseId, selected.id, apiEndpoint);
      await refreshEvaluation();
      setMessage(`“${selected.title}” is now the expected memory for “${resolved.query}”.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "That expected memory could not be attached.");
    }
  };

  const saveAlias = async (event: FormEvent) => {
    event.preventDefault();
    const canonical = canonicalTerm.trim();
    const alternate = aliasTerm.trim();
    if (!canonical || !alternate) {
      setMessage("Enter the remembered term and the alternate name Recall should connect.");
      return;
    }
    setSavingAlias(true);
    setMessage(undefined);
    try {
      const saved = await addMemoryAlias(canonical, alternate, apiEndpoint);
      await Promise.all([refreshAliases(), refreshStats()]);
      setCanonicalTerm("");
      setAliasTerm("");
      setMessage(`Recall now connects “${saved.alias}” with “${saved.canonical_term}”.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "That Memory connection could not be saved.");
    } finally {
      setSavingAlias(false);
    }
  };

  const removeAlias = async (item: MemoryAlias) => {
    setMessage(undefined);
    try {
      await deleteMemoryAlias(item.id, apiEndpoint);
      await Promise.all([refreshAliases(), refreshStats()]);
      setMessage(`Removed the connection between “${item.alias}” and “${item.canonical_term}”.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "That Memory connection could not be removed.");
    }
  };

  const teachResult = async (result: MemoryResult, verdict: "relevant" | "not_relevant") => {
    if (!searchedQuery) return;
    const nextVerdict = result.creator_feedback === verdict ? "clear" : verdict;
    setTeachingResultId(result.id);
    setMessage(undefined);
    try {
      const saved = await setMemoryFeedback(searchedQuery, result.id, nextVerdict, apiEndpoint);
      if (saved.verdict === "not_relevant") {
        const removedIndex = results.findIndex((item) => item.id === result.id);
        const remaining = results.filter((item) => item.id !== result.id);
        setResults(remaining);
        setSelectedId(remaining[0]?.id);
        setRejectedCorrection({ result, index: Math.max(0, removedIndex), query: searchedQuery });
        setMessage("Removed from this search. Recall will remember that correction.");
        requestAnimationFrame(() => searchInputRef.current?.focus());
      } else {
        setRejectedCorrection(undefined);
        setResults((current) => current.map((item) => (
          item.id === result.id ? { ...item, creator_feedback: saved.verdict } : item
        )));
        setMessage(saved.verdict === "relevant"
          ? "Marked relevant. Recall will prioritize this memory for the same search."
          : "Search feedback cleared.");
      }
      await Promise.all([refreshStats(), refreshEvaluation()]);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not save that correction.");
    } finally {
      setTeachingResultId(undefined);
    }
  };

  const undoRejectedCorrection = async () => {
    if (!rejectedCorrection) return;
    setTeachingResultId(rejectedCorrection.result.id);
    try {
      await setMemoryFeedback(
        rejectedCorrection.query,
        rejectedCorrection.result.id,
        "clear",
        apiEndpoint,
      );
      setResults((current) => {
        const next = [...current];
        next.splice(
          Math.min(rejectedCorrection.index, next.length),
          0,
          { ...rejectedCorrection.result, creator_feedback: null },
        );
        return next;
      });
      setSelectedId(rejectedCorrection.result.id);
      setRejectedCorrection(undefined);
      setMessage("Correction undone. The memory is back in this search.");
      await Promise.all([refreshStats(), refreshEvaluation()]);
      requestAnimationFrame(() => searchInputRef.current?.focus());
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not undo that correction.");
    } finally {
      setTeachingResultId(undefined);
    }
  };

  const openCompilationPicker = async () => {
    if (!selected?.clip_id || !selected.kept || compilationBusy) return;
    setCompilationPickerOpen(true);
    setCompilationError(undefined);
    if (compilationProjects.length) return;
    setCompilationBusy(true);
    try {
      const projects = await listCompilationProjects(apiEndpoint);
      setCompilationProjects(projects);
      setCompilationProjectId(projects[0]?.id ?? "");
    } catch (error) {
      setCompilationError(error instanceof Error ? error.message : "Recall could not load compilation drafts.");
    } finally {
      setCompilationBusy(false);
    }
  };

  const addSelectedToCompilation = async () => {
    if (!selected?.clip_id || !compilationProjectId || compilationBusy) return;
    setCompilationBusy(true);
    setCompilationError(undefined);
    try {
      const project = await addClipToCompilationProject({
        projectId: compilationProjectId,
        clipId: selected.clip_id,
        memoryEntryId: selected.id,
        memoryQuery: searchedQuery,
        baseUrl: apiEndpoint,
      });
      setCompilationProjects((current) => current.map((item) => (
        item.id === project.id ? project : item
      )));
      setCompilationPickerOpen(false);
      setMessage(`Added to “${project.title}”. The draft is ready in Compilations.`);
    } catch (error) {
      setCompilationError(error instanceof Error ? error.message : "Recall could not add that clip to the draft.");
    } finally {
      setCompilationBusy(false);
    }
  };

  const sourceUrl = selected?.source_available
    ? `${apiUrl(`/jobs/${encodeURIComponent(selected.job_id)}/source`, apiEndpoint)}#t=${Math.max(0, selected.start_time)}`
    : null;
  const missingSessions = Math.max(0, (stats?.completed_jobs ?? 0) - (stats?.indexed_jobs ?? 0));

  return (
    <section className={`memory-workspace ${toolsOpen ? "is-tools-open" : ""}`} aria-labelledby="memory-title">
      <header className="memory-command">
        <div className="memory-command-copy">
          <h1 id="memory-title">Stream Memory</h1>
          <p>Ask for the moment you remember—even when you cannot remember the exact words.</p>
        </div>
        <div className={`memory-health is-${buildingConcepts ? "building" : stats?.semantic_status ?? "not-built"}`}>
          <span className="memory-health-dot" aria-hidden="true" />
          <span>
            <strong>{buildingConcepts ? "Learning your archive" : semanticLabel(stats)}</strong>
            <small>
              {buildingConcepts
                ? `${((stats?.transcript_entries ?? 0) + (stats?.clip_entries ?? 0) + (stats?.evidence_entries ?? 0)).toLocaleString()} moments · usually 3–6 minutes · search stays available`
                : stats?.semantic_available
                ? `${stats.semantic_entries.toLocaleString()} moments · ${formatBytes(stats.semantic_index_bytes + stats.semantic_model_bytes)} model + index · stored locally`
                : "A compact local index; no recordings are copied"}
            </small>
          </span>
          <button type="button" onClick={buildConcepts} disabled={buildingConcepts || !stats?.indexed_jobs} aria-label={buildingConcepts ? `Learning archive, ${formatBuildElapsed(buildElapsedSeconds)} elapsed` : undefined}>
            {buildingConcepts ? <Clock size={14} aria-hidden="true" /> : <Sparkles size={14} aria-hidden="true" />}
            {buildingConcepts ? `Learning ${formatBuildElapsed(buildElapsedSeconds)}` : stats?.semantic_available ? "Update" : "Build"}
          </button>
        </div>

        <form className="memory-search" onSubmit={search}>
          <label className="memory-search-field">
            <Search size={21} aria-hidden="true" />
            <input
              ref={searchInputRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder='Try “the close call where everyone started yelling”'
              aria-label="Search Stream Memory"
            />
            <kbd>Enter</kbd>
          </label>
          <button className="cta-accent" type="submit" disabled={searching}>
            {searching ? "Searching…" : "Search memory"}
          </button>
        </form>

        <div className="memory-starters" aria-label="Example memory searches">
          <span>Try a memory</span>
          {STARTER_QUERIES.map((starter) => (
            <button type="button" key={starter} onClick={() => useStarterQuery(starter)}>
              {starter}
            </button>
          ))}
        </div>

        <div className="memory-ledger" aria-label="Memory index summary">
          <span><strong>{stats?.indexed_jobs ?? 0}</strong> sessions remembered</span>
          <span><strong>{(stats?.transcript_entries ?? 0).toLocaleString()}</strong> spoken moments</span>
          <span><strong>{(stats?.clip_entries ?? 0).toLocaleString()}</strong> saved clips</span>
          <span><strong>{(stats?.evidence_entries ?? 0).toLocaleString()}</strong> evidence packs</span>
          <span><HardDrive size={13} aria-hidden="true" /> Derived locally, no media copies</span>
        </div>
      </header>

      <div className="memory-toolbar">
        <div className="memory-filter-group" aria-label="Search mode">
          {(["hybrid", "keyword"] as MemorySearchMode[]).map((value) => (
            <button type="button" key={value} className={mode === value ? "is-active" : ""} aria-pressed={mode === value} onClick={() => setMode(value)}>
              {value === "hybrid" ? "Meaning + words" : "Exact words"}
            </button>
          ))}
        </div>
        <div className="memory-filter-group" aria-label="Result type">
          {(["all", "transcript", "clip", "evidence"] as MemoryKind[]).map((value) => (
            <button type="button" key={value} className={kind === value ? "is-active" : ""} aria-pressed={kind === value} onClick={() => setKind(value)}>
              {value === "all" ? "Everything" : value === "transcript" ? "Conversation" : value === "clip" ? "Clips" : "Evidence"}
            </button>
          ))}
        </div>
        <label className="memory-decision-filter">
          <span>Review state</span>
          <select value={decision} onChange={(event) => setDecision(event.target.value as MemoryDecision)}>
            <option value="all">All</option>
            <option value="kept">Kept</option>
            <option value="maybe">Maybe</option>
            <option value="passed">Passed</option>
            <option value="unreviewed">Unreviewed</option>
          </select>
        </label>
        <button
          className={`btn-secondary memory-tools-button ${toolsOpen ? "is-active" : ""}`}
          type="button"
          aria-expanded={toolsOpen}
          aria-controls="memory-tools"
          onClick={() => setToolsOpen((open) => !open)}
        >
          <Settings size={14} aria-hidden="true" />
          Archive tools
          {missingSessions > 0 && <span aria-label={`${missingSessions} sessions not yet indexed`}>{missingSessions}</span>}
        </button>
      </div>

      {toolsOpen && (
        <div className="memory-tools" id="memory-tools">
          <div
            className="memory-tools-tabs"
            role="tablist"
            aria-label="Archive tools"
            onKeyDown={navigateToolTabs}
          >
            <button
              type="button"
              role="tab"
              id="memory-tools-tab-archive"
              aria-selected={toolsTab === "archive"}
              aria-controls="memory-tools-panel-archive"
              tabIndex={toolsTab === "archive" ? 0 : -1}
              className={toolsTab === "archive" ? "is-active" : ""}
              onClick={() => setToolsTab("archive")}
            >
              <RefreshCw size={13} aria-hidden="true" /> Archive
            </button>
            <button
              type="button"
              role="tab"
              id="memory-tools-tab-vocabulary"
              aria-selected={toolsTab === "vocabulary"}
              aria-controls="memory-tools-panel-vocabulary"
              tabIndex={toolsTab === "vocabulary" ? 0 : -1}
              className={toolsTab === "vocabulary" ? "is-active" : ""}
              onClick={() => setToolsTab("vocabulary")}
            >
              <Sparkles size={13} aria-hidden="true" /> Creator vocabulary
              <span>{aliases.length}</span>
            </button>
            <button
              type="button"
              role="tab"
              id="memory-tools-tab-evaluation"
              aria-selected={toolsTab === "evaluation"}
              aria-controls="memory-tools-panel-evaluation"
              tabIndex={toolsTab === "evaluation" ? 0 : -1}
              className={toolsTab === "evaluation" ? "is-active" : ""}
              onClick={() => setToolsTab("evaluation")}
            >
              <Target size={13} aria-hidden="true" /> Search test
              <span>{evaluation?.case_count ?? 0}/20</span>
            </button>
          </div>

          {toolsTab === "archive" && (
            <section
              className="memory-archive-tools"
              id="memory-tools-panel-archive"
              role="tabpanel"
              aria-labelledby="memory-tools-tab-archive"
            >
              <div className="memory-archive-copy">
                <strong>Keep the searchable archive current</strong>
                <span>
                  Indexing reads finished scans already on this computer. It never downloads a
                  recording and never starts a scan.
                </span>
              </div>
              <button className="btn-secondary memory-index-button" type="button" onClick={indexLibrary} disabled={indexing}>
                <RefreshCw size={14} aria-hidden="true" />
                {indexing ? "Updating archive…" : missingSessions ? `Add ${missingSessions} session${missingSessions === 1 ? "" : "s"}` : "Update archive"}
              </button>
              <div className="memory-archive-facts">
                <span><strong>{stats?.indexed_jobs ?? 0}</strong> sessions indexed</span>
                <span>
                  <strong>{missingSessions}</strong> waiting to be added
                </span>
                <span>{semanticLabel(stats)}</span>
              </div>
            </section>
          )}

          {toolsTab === "vocabulary" && (
        <section
          className="memory-vocabulary"
          id="memory-tools-panel-vocabulary"
          role="tabpanel"
          aria-labelledby="memory-tools-tab-vocabulary"
          aria-label="Creator vocabulary"
        >
          <div className="memory-vocabulary-copy">
            <strong>Teach Recall your language</strong>
            <span>Connect usernames, nicknames, recurring jokes, and phrases. Search will use both names without rebuilding the archive.</span>
          </div>
          <form className="memory-vocabulary-form" onSubmit={saveAlias}>
            <label>
              <span>Remembered as</span>
              <input value={canonicalTerm} onChange={(event) => setCanonicalTerm(event.target.value)} maxLength={80} placeholder="Kevin" />
            </label>
            <label>
              <span>Also called</span>
              <input value={aliasTerm} onChange={(event) => setAliasTerm(event.target.value)} maxLength={80} placeholder="KevPlays" />
            </label>
            <button className="btn-secondary" type="submit" disabled={savingAlias}>
              {savingAlias ? "Saving…" : "Connect terms"}
            </button>
          </form>
          {aliases.length > 0 && (
            <div className="memory-alias-list" aria-label="Saved vocabulary connections">
              {aliases.map((item) => (
                <span key={item.id}>
                  <b>{item.alias}</b>
                  <i>also means</i>
                  {item.canonical_term}
                  <button type="button" aria-label={`Remove ${item.alias} connection`} onClick={() => void removeAlias(item)}>
                    <Trash2 size={12} aria-hidden="true" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </section>
          )}

          {toolsTab === "evaluation" && (
        <section
          className="memory-evaluation"
          id="memory-tools-panel-evaluation"
          role="tabpanel"
          aria-labelledby="memory-tools-tab-evaluation"
          aria-label="Memory search test"
        >
          <div className="memory-evaluation-copy">
            <strong>Measure Recall against memories you actually want</strong>
            <span>Relevant and Not this build the truth set. A failed search records what Recall should have found. Baselines ignore your relevance boost.</span>
          </div>
          <div className="memory-evaluation-progress">
            <span><strong>{evaluation?.case_count ?? 0}</strong> of {evaluation?.target_cases ?? 20} searches captured</span>
            <progress
              max={evaluation?.target_cases ?? 20}
              value={evaluation?.case_count ?? 0}
              aria-label={`Search test progress: ${evaluation?.case_count ?? 0} of ${evaluation?.target_cases ?? 20} searches captured`}
            />
          </div>
          <div className="memory-evaluation-facts">
            <span><strong>{evaluation?.scorable_cases ?? 0}</strong> ready to score</span>
            <span><strong>{evaluation?.unresolved_cases ?? 0}</strong> missing an expected result</span>
            {evaluation?.latest_run ? (
              <span>
                <strong>{Math.round(evaluation.latest_run.metrics.top_5_rate * 100)}%</strong>{" "}
                {evaluation.latest_run.stale ? "last run outdated" : "last run Top 5"}
              </span>
            ) : <span>No baseline saved yet</span>}
          </div>
          <button
            className="btn-secondary memory-evaluation-run"
            type="button"
            onClick={() => void runBaseline()}
            disabled={runningEvaluation || !(evaluation?.scorable_cases)}
          >
            <BarChart3 size={14} aria-hidden="true" />
            {runningEvaluation ? "Measuring…" : "Run current baseline"}
          </button>
          {Boolean(evaluation?.cases.length) && (
            <div className="memory-evaluation-cases" aria-label="Captured search tests">
              {evaluation!.cases.map((item) => (
                <span key={item.id}>
                  <i className={item.ready_to_score ? "is-ready" : "is-unresolved"} aria-hidden="true" />
                  <b>{item.query}</b>
                  <small>{item.ready_to_score ? `${item.resolved_relevant_count} relevant` : "Needs the right memory"}</small>
                  {!item.ready_to_score && selected && (
                    <button
                      className="memory-evaluation-resolve"
                      type="button"
                      aria-label={`Use selected result for “${item.query}”`}
                      onClick={() => void resolveEvaluationCase(item.id)}
                    >
                      Use selected
                    </button>
                  )}
                  <button type="button" aria-label={`Remove ${item.query} from search test`} onClick={() => void removeEvaluationCase(item.id)}>
                    <Trash2 size={12} aria-hidden="true" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </section>
          )}
        </div>
      )}

      {(unavailable.archive || degradedTools.length > 0) && (
        <div className="memory-degraded" role="status">
          <AlertTriangle size={15} aria-hidden="true" />
          <span>
            <strong>
              {unavailable.archive
                ? "Recall’s engine is not responding"
                : `${degradedTools.join(" and ")} did not load`}
            </strong>
            <small>
              {unavailable.archive
                ? "Searching, indexing, and the archive summary all need the local engine. Nothing in your archive was lost — start Recall’s engine, then try again."
                : "Search still works. Reopen Archive tools after retrying."}
            </small>
          </span>
          <button type="button" className="btn-secondary" onClick={retryUnavailable}>
            <RefreshCw size={13} aria-hidden="true" /> Try again
          </button>
        </div>
      )}

      {message && (
        <div className="memory-message" role="status">
          <span>{message}</span>
          {rejectedCorrection && (
            <button type="button" onClick={() => void undoRejectedCorrection()} disabled={Boolean(teachingResultId)}>
              Undo
            </button>
          )}
        </div>
      )}

      {!searchedQuery && !results.length ? (
        <div className="memory-empty">
          <div className="memory-empty-signal" aria-hidden="true"><i /><i /><Sparkles size={27} /><i /><i /></div>
          <h2>Your archive is listening</h2>
          <p>Search names and quotes exactly, or build the concept index to find moments by what happened and how they felt.</p>
          {missingSessions > 0 && (
            <button className="btn-secondary" type="button" onClick={indexLibrary} disabled={indexing}>
              Add {missingSessions} existing session{missingSessions === 1 ? "" : "s"}
            </button>
          )}
        </div>
      ) : !results.length ? (
        <div className="memory-empty memory-empty-search">
          <h2>No trail for “{searchedQuery}”</h2>
          <p>Try fewer details, switch to Meaning + words, or update the archive after a new scan.</p>
          <form className="memory-miss-form" onSubmit={saveFailedSearch}>
            <label htmlFor="memory-expected-result">What should Recall have found?</label>
            <div>
              <input
                id="memory-expected-result"
                value={expectedMemory}
                onChange={(event) => setExpectedMemory(event.target.value)}
                maxLength={500}
                placeholder="The sniper shot near the end of the match"
              />
              <button className="btn-secondary" type="submit" disabled={savingMiss || !expectedMemory.trim()}>
                {savingMiss ? "Saving…" : "Save failed search"}
              </button>
            </div>
            <small>This stores text and a search label only—never another copy of the recording.</small>
          </form>
        </div>
      ) : (
        <div className="memory-results-layout">
          <div className="memory-results" aria-label={`${results.length} search results`}>
            <div className="memory-results-count">
              <span><strong>{results.length}</strong> memories for “{searchedQuery}”</span>
              <div className="memory-results-tools">
                {(interpretedFilters.length > 0 || compoundClauses.length > 0) && (
                  <span className="memory-understood" aria-label="Requirements understood from this search">
                    {interpretedFilters.slice(0, 3).map(({ field, value }) => (
                      <i key={`${field}-${value}`}>{interpretedFilterLabel(field, value)}</i>
                    ))}
                    {compoundClauses.slice(0, Math.max(0, 4 - interpretedFilters.length)).map(({ id, label }) => (
                      <i className="is-compound" key={id}><Check size={10} aria-hidden="true" />{label}</i>
                    ))}
                  </span>
                )}
                <span>
                  <Sparkles size={12} />
                  {aliasExpansions.length
                    ? `${aliasExpansions[0].matched_term} → ${aliasExpansions[0].canonical_term}`
                    : modeUsed === "hybrid" ? "Meaning + words" : "Exact words"}
                </span>
                <label>
                  <span>Sort</span>
                  <select
                    value={sort}
                    onChange={(event) => {
                      const nextSort = event.target.value as MemorySort;
                      const nextResults = sortMemoryResults(results, nextSort);
                      setSort(nextSort);
                      setVisibleCount(RESULT_BATCH_SIZE);
                      setSelectedId(nextResults[0]?.id);
                    }}
                  >
                    <option value="relevance">Best match</option>
                    <option value="clips">Clips first</option>
                    <option value="newest">Newest first</option>
                    <option value="oldest">Oldest first</option>
                  </select>
                </label>
              </div>
            </div>
            <form className="memory-refine" onSubmit={refineResults} aria-label="Refine current Memory results">
              <div className="memory-refine-copy">
                <strong id="memory-refine-title">Refine these results</strong>
                <small>Uses the current search and selected result.</small>
              </div>
              <label>
                <Search size={13} aria-hidden="true" />
                <input
                  value={refinement}
                  onChange={(event) => {
                    setRefinement(event.target.value);
                    setRefinementError(undefined);
                    setRefinementNotice(undefined);
                  }}
                  maxLength={120}
                  placeholder='Try “only last month” or “the second one”'
                  aria-labelledby="memory-refine-title"
                />
              </label>
              <button type="submit" className="btn-secondary" disabled={refining || !refinement.trim()}>
                {refining ? "Refining…" : "Refine results"}
              </button>
              {refinementStack.length > 0 && (
                <button type="button" className="memory-refine-undo" onClick={undoRefinement} disabled={refining}>
                  <RotateCcw size={12} aria-hidden="true" /> Undo
                </button>
              )}
              <div className="memory-refine-guidance">
                {refinementHistory.length > 0 ? (
                  <span aria-label="Applied refinements">
                    {refinementHistory.map((label, index) => <i key={`${label}-${index}`}>{label}</i>)}
                  </span>
                ) : (
                  <small>Also supports “unexported” and “more like this.”</small>
                )}
                {(refinementError || refinementNotice) && (
                  <small className={refinementError ? "is-error" : "is-notice"} role={refinementError ? "alert" : "status"}>
                    {refinementError ?? refinementNotice}
                  </small>
                )}
              </div>
            </form>
            <div
              className="memory-result-stack"
              role="listbox"
              aria-label={`Memories for “${searchedQuery}”`}
            >
              {visibleResults.map((result, index) => {
                const state = resultDecision(result);
                const isSelected = selected?.id === result.id;
                return (
                  <button
                    type="button"
                    key={result.id}
                    id={`memory-result-${index}`}
                    role="option"
                    aria-selected={isSelected}
                    tabIndex={isSelected ? 0 : -1}
                    className={`memory-result ${isSelected ? "is-selected" : ""}`}
                    onKeyDown={(event) => navigateResults(event, index)}
                    onClick={() => selectResult(result)}
                  >
                    <span className="memory-result-rail"><i /><time>{fmtClock(result.start_time)}</time></span>
                    <span className="memory-result-copy">
                      <span className="memory-result-meta">
                        <span>{result.session_name}</span>
                        {result.game && <span>{result.game}</span>}
                        <span>{resultDate(result.source_date)}</span>
                      </span>
                      <strong>{result.kind === "clip" ? result.title : resultContext(result)}</strong>
                      {result.kind === "clip" && resultContext(result) !== result.title && <small>{resultContext(result)}</small>}
                      <span
                        className={`memory-match ${matchSourceClass(result)}`}
                        aria-label={`Why this result matched: ${matchSourceLabel(result)}`}
                      >
                        <Sparkles size={11} aria-hidden="true" />{matchSourceLabel(result)}
                      </span>
                    </span>
                    <span className="memory-result-state">
                      {result.kind === "clip" ? <Film size={14} /> : result.kind === "evidence" ? <Sparkles size={14} /> : <Video size={14} />}
                      {state && <span className={`memory-decision is-${state.toLowerCase()}`}>{state}</span>}
                      {result.exported && <span className="memory-decision is-exported">Exported</span>}
                    </span>
                  </button>
                );
              })}
              {visibleCount < sortedResults.length && (
                <div className="memory-results-more">
                  <button
                    type="button"
                    onClick={() => setVisibleCount((count) => Math.min(count + RESULT_BATCH_SIZE, sortedResults.length))}
                  >
                    Show {Math.min(RESULT_BATCH_SIZE, sortedResults.length - visibleCount)} more
                    <span>{sortedResults.length - visibleCount} remaining</span>
                  </button>
                </div>
              )}
            </div>
          </div>

          {selected && (
            <aside className="memory-detail">
              <div className="memory-player">
                {sourceUrl ? (
                  <video key={sourceUrl} ref={videoRef} controls preload="metadata" src={sourceUrl} onLoadedMetadata={() => { if (videoRef.current) videoRef.current.currentTime = selected.start_time; }} />
                ) : (
                  <div className="memory-player-unavailable">
                    <Video size={30} aria-hidden="true" />
                    <strong>Source recording unavailable</strong>
                    <span>The searchable memory remains, but the original VOD is no longer on this computer.</span>
                  </div>
                )}
              </div>
              <div className="memory-actions" aria-label="Use this memory">
                {selected.clip_id ? (
                  <button type="button" className="cta-accent" onClick={() => {
                    recordOpen(selected);
                    void onOpenClip(selected.job_id, selected.clip_id!);
                  }}>
                    <Play size={13} aria-hidden="true" /> Open in Theater
                  </button>
                ) : onOpenMoment ? (
                  <button type="button" className="cta-accent" onClick={() => {
                    recordOpen(selected);
                    void onOpenMoment(selected, searchedQuery);
                  }}>
                    <Scissors size={13} aria-hidden="true" />
                    {selected.source_available
                      ? "Cut this moment"
                      : selected.source_type === "twitch" || selected.source_type === "url"
                        ? "Restore source to cut"
                        : "Check source to cut"}
                  </button>
                ) : null}
                {selected.clip_id && selected.kept && (
                  <button type="button" className="btn-secondary" onClick={() => void openCompilationPicker()} disabled={compilationBusy}>
                    <Plus size={13} aria-hidden="true" /> Add to compilation
                  </button>
                )}
                {!selected.source_available && !selected.clip_id && (
                  <small>The VOD Editor will show source status and Twitch restoration when available. Your searchable evidence stays available either way.</small>
                )}
                {compilationPickerOpen && (
                  <div className="memory-compilation-picker">
                    {compilationBusy && !compilationProjects.length ? (
                      <span><span className="studio-spinner" aria-hidden="true" />Loading compilation drafts…</span>
                    ) : compilationProjects.length ? (
                      <>
                        <label>
                          <span>Compilation draft</span>
                          <select value={compilationProjectId} onChange={(event) => setCompilationProjectId(event.target.value)}>
                            {compilationProjects.map((project) => (
                              <option key={project.id} value={project.id}>
                                {project.title}{project.status === "approved" ? " — reopens as draft" : ""}
                              </option>
                            ))}
                          </select>
                        </label>
                        <button type="button" className="btn-secondary" disabled={!compilationProjectId || compilationBusy} onClick={() => void addSelectedToCompilation()}>
                          {compilationBusy ? "Adding…" : "Add to draft"}
                        </button>
                      </>
                    ) : !compilationError ? (
                      <span>No compilation drafts yet. Create one, then return to add this clip.</span>
                    ) : null}
                    {!compilationProjects.length && onOpenCompilations && !compilationBusy && (
                      <button type="button" className="btn-secondary" onClick={onOpenCompilations}>Open Compilations</button>
                    )}
                  </div>
                )}
                {compilationError && <small className="is-error" role="alert">{compilationError} Try again, or open Compilations to check the draft.</small>}
              </div>
              <div className="memory-detail-body">
                <div className="memory-detail-label">
                  <span>{resultType(selected)}</span>
                  <span
                    className={`memory-match ${matchSourceClass(selected)}`}
                    aria-label={`Why this result matched: ${matchSourceLabel(selected)}`}
                  >
                    <Sparkles size={11} aria-hidden="true" />{matchSourceLabel(selected)}
                  </span>
                </div>
                <h2>{selected.title}</h2>
                <p>{resultContext(selected)}</p>
                <div className="memory-detail-meta">
                  <span>{selected.session_name}</span>
                  <span>{resultDate(selected.source_date)}</span>
                  <span><Clock size={11} />{fmtClock(selected.start_time)}–{fmtClock(selected.end_time)}</span>
                  {selected.transcript_coverage === "partial" && <span>Partial transcript</span>}
                  {selected.exported && <span><Check size={11} />Exported</span>}
                </div>
                {((selected.matched_evidence?.length ?? 0) > 0 || evidenceClues(selected).length > 0) && (
                  <section className="memory-why" aria-labelledby="memory-why-title">
                    <h3 id="memory-why-title">Why this matched</h3>
                    {(selected.matched_evidence?.length ?? 0) > 0 && (
                      <div className="memory-compound-evidence" aria-label="Clues matched together">
                        <small>
                          {selected.matched_evidence!.length} clues matched together. Each required
                          signal overlaps this same moment.
                        </small>
                        <div>
                          {selected.matched_evidence!.map((match) => (
                            <span key={match.clause_id}>
                              <Check size={12} aria-hidden="true" />
                              <span><strong>{match.label}</strong><small>{match.detail}</small></span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {evidenceClues(selected).length > 0 && (
                      <div className="memory-evidence" aria-label="Evidence on hand">
                        <small>
                          <b>Evidence on hand</b> — compact clues retained from this scan, not
                          another video copy.
                        </small>
                        <div>
                          {evidenceClues(selected).map((clue) => <span key={clue}>{clue}</span>)}
                        </div>
                      </div>
                    )}
                  </section>
                )}
                <div className="memory-feedback" aria-label="Teach this search">
                  <span>
                    <strong>Teach this search</strong>
                    <small>Changes only “{searchedQuery}” and adds evidence to the search test.</small>
                  </span>
                  <button
                    type="button"
                    className={selected.creator_feedback === "relevant" ? "is-relevant" : ""}
                    aria-pressed={selected.creator_feedback === "relevant"}
                    disabled={teachingResultId === selected.id}
                    onClick={() => void teachResult(selected, "relevant")}
                  >
                    <Check size={12} aria-hidden="true" /> Relevant
                  </button>
                  <button
                    type="button"
                    aria-pressed={selected.creator_feedback === "not_relevant"}
                    disabled={teachingResultId === selected.id}
                    onClick={() => void teachResult(selected, "not_relevant")}
                  >
                    <X size={12} aria-hidden="true" /> Not this
                  </button>
                </div>
              </div>
            </aside>
          )}
        </div>
      )}
    </section>
  );
}

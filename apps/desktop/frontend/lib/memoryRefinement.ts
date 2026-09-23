// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import type {
  MemoryDecision,
  MemoryExportFilter,
  MemoryKind,
  MemoryOriginFilter,
  MemoryResult,
  MemorySearchMode,
} from "./streamMemory";

export interface MemorySearchContext {
  query: string;
  kind: MemoryKind;
  decision: MemoryDecision;
  exported: MemoryExportFilter;
  origin: MemoryOriginFilter;
  game: string | null;
  dateFrom: string | null;
  dateTo: string | null;
  mode: MemorySearchMode;
}

export type MemoryRefinementPlan =
  | { type: "select"; index: number; label: string }
  | { type: "search"; context: MemorySearchContext; label: string; excludeEntryId?: string }
  | { type: "error"; message: string };

const ORDINALS: Record<string, number> = {
  first: 1,
  second: 2,
  third: 3,
  fourth: 4,
  fifth: 5,
  sixth: 6,
  seventh: 7,
  eighth: 8,
  ninth: 9,
  tenth: 10,
};

function isoDate(year: number, month: number, day: number) {
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function monthBounds(reference: Date, monthDelta: number): [string, string] {
  const start = new Date(reference.getFullYear(), reference.getMonth() + monthDelta, 1);
  const next = new Date(start.getFullYear(), start.getMonth() + 1, 1);
  const end = new Date(next.getFullYear(), next.getMonth(), 0);
  return [
    isoDate(start.getFullYear(), start.getMonth() + 1, 1),
    isoDate(end.getFullYear(), end.getMonth() + 1, end.getDate()),
  ];
}

function ordinalIndex(instruction: string, resultCount: number): MemoryRefinementPlan | null {
  const normalized = instruction.trim().toLowerCase();
  const lastMatch = normalized.match(/^(?:(?:show|pick|select|open)\s+)?(?:the\s+)?last(?:\s+(?:one|result|moment|clip))?$/);
  if (lastMatch) {
    return resultCount > 0
      ? { type: "select", index: resultCount - 1, label: `Selected result ${resultCount}` }
      : { type: "error", message: "There are no results to select." };
  }
  const match = normalized.match(
    /^(?:(?:show|pick|select|open)\s+)?(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d{1,3})(?:st|nd|rd|th)?(?:\s+(?:one|result|moment|clip))?$/,
  );
  if (!match) return null;
  const oneBased = ORDINALS[match[1]] ?? Number(match[1]);
  if (!Number.isInteger(oneBased) || oneBased < 1) {
    return { type: "error", message: "Choose a result number starting at 1." };
  }
  if (oneBased > resultCount) {
    return {
      type: "error",
      message: `This search has ${resultCount} result${resultCount === 1 ? "" : "s"}; there is no result ${oneBased}.`,
    };
  }
  return { type: "select", index: oneBased - 1, label: `Selected result ${oneBased}` };
}

function similarSeed(result: MemoryResult): string {
  const values = [
    result.title,
    result.match_context,
    result.evidence?.search_text,
  ].map((value) => (value ?? "").trim()).filter(Boolean);
  return [...new Set(values)].join(" ").slice(0, 240);
}

export function parseMemoryRefinement(input: {
  instruction: string;
  context: MemorySearchContext;
  selected?: MemoryResult;
  resultCount: number;
  now?: Date;
}): MemoryRefinementPlan {
  const instruction = input.instruction.trim();
  if (!instruction) {
    return { type: "error", message: "Describe how to narrow these results." };
  }
  const ordinal = ordinalIndex(instruction, input.resultCount);
  if (ordinal) return ordinal;

  if (/^(?:show\s+)?more\s+like\s+(?:this|that|the\s+selected\s+(?:one|result))$/i.test(instruction)) {
    if (!input.selected) {
      return { type: "error", message: "Select a result before asking for more like it." };
    }
    const query = similarSeed(input.selected);
    if (!query) {
      return { type: "error", message: "That result does not retain enough text for a similarity search." };
    }
    return {
      type: "search",
      context: { ...input.context, query, mode: "hybrid" },
      label: `More like “${input.selected.title.slice(0, 72)}”`,
      excludeEntryId: input.selected.id,
    };
  }

  const next = { ...input.context };
  const labels: string[] = [];
  const touched = new Map<string, string>();
  let remaining = instruction.toLowerCase();
  let conflict: string | null = null;

  const setField = <K extends keyof MemorySearchContext>(key: K, value: MemorySearchContext[K], label: string) => {
    const serialized = String(value ?? "");
    if (touched.has(key) && touched.get(key) !== serialized) conflict = `Choose one ${label.toLowerCase()} refinement at a time.`;
    touched.set(key, serialized);
    next[key] = value;
    labels.push(label);
  };
  const consume = (pattern: RegExp, apply: () => void) => {
    if (!pattern.test(remaining)) return;
    remaining = remaining.replace(pattern, " ");
    apply();
  };

  consume(/\b(?:only\s+)?last\s+month\b/i, () => {
    const [dateFrom, dateTo] = monthBounds(input.now ?? new Date(), -1);
    next.dateFrom = dateFrom;
    next.dateTo = dateTo;
    labels.push("Last month");
  });
  consume(/\b(?:only\s+)?this\s+month\b/i, () => {
    const [dateFrom, dateTo] = monthBounds(input.now ?? new Date(), 0);
    next.dateFrom = dateFrom;
    next.dateTo = dateTo;
    labels.push("This month");
  });
  consume(/\b(?:all\s+time|any\s+(?:time|date)|clear\s+date)\b/i, () => {
    next.dateFrom = null;
    next.dateTo = null;
    labels.push("All dates");
  });

  consume(/\b(?:unexported|not\s+exported|never\s+exported)\b/i, () => setField("exported", "not_exported", "Not exported"));
  consume(/\b(?:already\s+)?exported\b/i, () => setField("exported", "exported", "Exported"));
  consume(/\b(?:any\s+export\s+state|all\s+exports?)\b/i, () => setField("exported", "all", "Any export state"));

  consume(/\b(?:only\s+)?kept(?:\s+clips?)?\b/i, () => setField("decision", "kept", "Kept"));
  consume(/\b(?:only\s+)?passed(?:\s+clips?)?\b/i, () => setField("decision", "passed", "Passed"));
  consume(/\b(?:only\s+)?maybe(?:\s+clips?)?\b/i, () => setField("decision", "maybe", "Maybe"));
  consume(/\b(?:only\s+)?(?:unreviewed|not\s+reviewed)(?:\s+clips?)?\b/i, () => setField("decision", "unreviewed", "Unreviewed"));
  consume(/\b(?:any\s+review\s+state|all\s+review\s+states?)\b/i, () => setField("decision", "all", "Any review state"));

  consume(/\b(?:only\s+)?clips?(?:\s+only)?\b/i, () => setField("kind", "clip", "Clips only"));
  consume(/\b(?:only\s+)?(?:conversation|transcript|speech)(?:\s+only)?\b/i, () => setField("kind", "transcript", "Conversation only"));
  consume(/\b(?:only\s+)?evidence(?:\s+only)?\b/i, () => setField("kind", "evidence", "Evidence only"));
  consume(/\b(?:everything|all\s+result\s+types?)\b/i, () => setField("kind", "all", "Every result type"));

  if (conflict) return { type: "error", message: conflict };
  const residue = remaining
    .replace(/\b(?:only|just|show|me|the|results?|moments?|ones?|please|and)\b/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (residue || labels.length === 0) {
    return {
      type: "error",
      message: "Try a bounded refinement such as “only last month,” “unexported,” “the second one,” or “more like this.”",
    };
  }
  return { type: "search", context: next, label: labels.join(" · ") };
}

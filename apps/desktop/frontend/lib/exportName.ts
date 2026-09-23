// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import type { ExportPreset } from "./store";

const TOKEN_PATTERN = /\{(title|index|date|platform|original)\}/g;
const UNSAFE_PATTERN = /[<>:"/\\|?*\u0000-\u001f]/g;
export const EXPORT_TITLE_MAX_LENGTH = 64;

function characters(value: string): string[] {
  return Array.from(value);
}

function takeCharacters(value: string, limit: number): string {
  return characters(value).slice(0, limit).join("");
}

export function sanitizeExportStem(value: string, fallback = "clip"): string {
  const safe = takeCharacters(
    (value || "").replace(UNSAFE_PATTERN, " ").replace(/\s+/g, " ").trim().replace(/^\.+|\.+$/g, ""),
    120,
  ).trim();
  return safe || fallback;
}

export function truncateExportTitle(
  title: string,
  fallback = "Untitled",
  maxLength = EXPORT_TITLE_MAX_LENGTH,
): string {
  const safe = sanitizeExportStem(title, fallback);
  if (characters(safe).length <= maxLength) return safe;
  let clipped = takeCharacters(safe, maxLength).trimEnd();
  const boundary = clipped.lastIndexOf(" ");
  if (boundary >= Math.floor(maxLength * 0.6)) clipped = clipped.slice(0, boundary).trimEnd();
  return clipped || takeCharacters(sanitizeExportStem(fallback, "Untitled"), maxLength);
}

export function creatorExportFilename({
  title,
  clipNumber,
  maxClipNumber,
  sourceDate,
  preset,
  template,
  originalStem,
}: {
  title: string;
  clipNumber: number;
  maxClipNumber?: number;
  sourceDate?: string;
  preset?: ExportPreset;
  template?: string;
  originalStem?: string;
}): string {
  const safeNumber = Math.max(1, Math.trunc(clipNumber || 1));
  const width = Math.max(2, String(Math.max(safeNumber, maxClipNumber || safeNumber)).length);
  const index = String(safeNumber).padStart(width, "0");
  const fallbackTitle = sanitizeExportStem(originalStem || "Untitled", "Untitled");
  const safeTitle = truncateExportTitle(title, fallbackTitle);
  const date = /^\d{4}-\d{2}-\d{2}/.test(sourceDate || "") ? (sourceDate || "").slice(0, 10) : "undated";
  const tokens: Record<string, string> = {
    title: safeTitle,
    index,
    date,
    platform: (preset || "clip").toLowerCase(),
    original: originalStem || "clip",
  };
  const rendered = template?.trim()
    ? template.replace(TOKEN_PATTERN, (_match, token: string) => tokens[token])
    : `${date}_clip-${index}_${safeTitle}`;
  return `${sanitizeExportStem(rendered, originalStem || `clip-${safeNumber}`)}.mp4`;
}

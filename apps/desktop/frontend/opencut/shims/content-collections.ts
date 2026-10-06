// SPDX-License-Identifier: AGPL-3.0-or-later
// Recall shim: OpenCut's website content (changelog, blog) isn't shipped in Recall.
export interface ChangelogEntry { version: string; title: string; summary?: string; published?: boolean; date?: string; [field: string]: unknown }
export const allChangelogs: ChangelogEntry[] = [];
export const allPosts: Record<string, unknown>[] = [];
export const allGuides: Record<string, unknown>[] = [];

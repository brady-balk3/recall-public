// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Import aliases for the vendored OpenCut editor (frontend/opencut, MIT).
 * `@opencut/*` is its own source root; Next.js and its online-only packages
 * are replaced by the shims next to it.
 */
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../frontend/opencut");
const shim = (file: string) => path.join(root, "shims", file);

export const openCutAliases = [
  { find: /^@opencut\//, replacement: `${path.join(root, "src").split(path.sep).join("/")}/` },
  { find: /^next\/link$/, replacement: shim("next-link.tsx") },
  { find: /^next\/image$/, replacement: shim("next-image.tsx") },
  { find: /^next\/navigation$/, replacement: shim("next-navigation.ts") },
  { find: /^next\/script$/, replacement: shim("empty.tsx") },
  { find: /^next\/dynamic$/, replacement: shim("next-dynamic.tsx") },
  { find: /^next-themes$/, replacement: shim("next-themes.tsx") },
  { find: /^content-collections$/, replacement: shim("content-collections.ts") },
  { find: /^@huggingface\/transformers$/, replacement: shim("transformers.ts") },
];

// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import electron from "vite-plugin-electron/simple";
import { thirdPartyNotices } from "./build/third-party-notices";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    thirdPartyNotices(),
    electron({
      // Terser, not the default esbuild minifier.
      //
      // app.asar is a plain archive anyone can extract, and the main process
      // shipped as ~286 readable lines with its original comments intact.
      // `minify: true` does NOT fix that: vite-plugin-electron builds these
      // through `build.lib` with format "es", and Vite deliberately skips
      // WHITESPACE minification for ES lib builds because stripping it can
      // break `/* @__PURE__ */` annotations. esbuild therefore mangles the
      // identifiers (it already did, by production default) and leaves every
      // newline and comment in place — setting `minify: true` here measured
      // byte-identical to setting nothing at all.
      //
      // Terser has no such carve-out, so it is the only way to actually
      // collapse the file. This is obfuscation, not protection — see
      // core/model_vault.py — but it removes the free read.
      main: {
        // Shortcut of `build.lib.entry`
        entry: "electron/main.ts",
        vite: {
          plugins: [thirdPartyNotices("THIRD-PARTY-NOTICES-main.txt")],
          build: {
            minify: "terser",
            sourcemap: false,
            terserOptions: { format: { comments: false } },
          },
        },
      },
      preload: {
        // Shortcut of `build.rollupOptions.input`
        input: "electron/preload.ts",
        vite: {
          plugins: [thirdPartyNotices("THIRD-PARTY-NOTICES-preload.txt")],
          build: {
            minify: "terser",
            sourcemap: false,
            terserOptions: { format: { comments: false } },
          },
        },
      },
    }),
  ],
});

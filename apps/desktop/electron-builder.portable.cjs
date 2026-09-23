// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
const base = require("./package.json").build;
const output = process.env.RECALL_PORTABLE_OUTPUT;
const engine = process.env.RECALL_PORTABLE_ENGINE_DIR;
if (!output || !engine) throw new Error("Use scripts/package_portable.ps1 to select fresh portable build paths");
module.exports = {
  ...base,
  directories: { ...base.directories, output },
  extraResources: [{ from: engine, to: "recall-engine" }],
  publish: null,
};

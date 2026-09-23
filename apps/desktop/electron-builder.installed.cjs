// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
const base = require("./package.json").build;

const supplied = process.env.RECALL_INSTALLER_PACKAGE_URL || "";
// The wrapper appends .partNNN to this logical archive URL. Only those chunks
// are uploaded; large models are downloaded from pinned Hugging Face revisions.
let url;
try { url = new URL(supplied); } catch { throw new Error("Set RECALL_INSTALLER_PACKAGE_URL to the HTTPS release URL ending in the runtime archive name"); }
if (url.protocol !== "https:" || url.username || url.password || url.search || url.hash
    || /[\s$"\\]/.test(supplied) || url.pathname.endsWith("/")) {
  throw new Error("Installer payload needs a permanent HTTPS file URL without credentials, query strings or fragments");
}

module.exports = {
  ...base,
  directories: { ...base.directories, output: process.env.RECALL_INSTALLER_OUTPUT || "release-installed" },
  extraResources: [
    { from: process.env.RECALL_INSTALLER_ENGINE_DIR || "../api/dist-installed/recall-engine", to: "recall-engine" },
    { from: "build/recall-installation.json", to: "recall-installation.json" },
  ],
  win: { ...base.win, target: [{ target: "nsis-web", arch: ["x64"] }] },
  publish: null,
  nsisWeb: {
    include: "build/recall-installer-integrity.nsh",
    appPackageUrl: url.href,
    oneClick: false,
    perMachine: false,
    allowToChangeInstallationDirectory: true,
    deleteAppDataOnUninstall: false,
    artifactName: "Recall-Setup-${version}.${ext}",
  },
};

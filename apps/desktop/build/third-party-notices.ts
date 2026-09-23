// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import fs from "node:fs";
import path from "node:path";
import type { Plugin } from "vite";

/** Preserve notices for packages whose code actually survives bundling. */
export function thirdPartyNotices(fileName = "THIRD-PARTY-NOTICES.txt"): Plugin {
  return {
    name: "recall-third-party-notices",
    generateBundle(_options, bundle) {
      const packages = new Map<string, string>();
      for (const output of Object.values(bundle)) {
        if (output.type !== "chunk") continue;
        for (const moduleId of Object.keys(output.modules)) {
          if (moduleId.startsWith("\0") || !moduleId.replaceAll("\\", "/").includes("/node_modules/")) continue;
          let directory = path.dirname(moduleId.split("?")[0]);
          let found = false;
          while (directory !== path.dirname(directory)) {
            const metadataPath = path.join(directory, "package.json");
            if (fs.existsSync(metadataPath)) {
              const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
              // Some packages have nested package.json files containing only type.
              if (metadata.name && metadata.version) {
                const notices = fs.readdirSync(directory).filter(name =>
                  /^(licen[cs]e|copying|notice|copyright)([.-].*)?$/i.test(name)
                  && fs.statSync(path.join(directory, name)).isFile()).sort();
                if (!notices.length) this.error(`Missing bundled dependency notice: ${metadata.name}@${metadata.version}`);
                packages.set(directory, `${metadata.name}@${metadata.version}\n` + notices.map(name =>
                  `\n--- ${name} ---\n${fs.readFileSync(path.join(directory, name), "utf8")}`).join("\n"));
                found = true;
                break;
              }
            }
            directory = path.dirname(directory);
          }
          if (!found) this.error("Cannot identify the package for a bundled dependency");
        }
      }
      if (packages.size) this.emitFile({
        type: "asset",
        fileName,
        source: "Third-party notices for code included in this bundle.\n\n" +
          [...packages.values()].sort().join("\n\n========================================\n\n") + "\n",
      });
    },
  };
}

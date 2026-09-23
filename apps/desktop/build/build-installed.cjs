// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
// Split the runtime package before NSIS compilation so every release asset fits GitHub.
const fs = require("node:fs/promises");
const path = require("node:path");
const crypto = require("node:crypto");

const CHUNK_BYTES = 1900 * 1024 * 1024;

async function splitArchive(archive, expectedHash, chunkBytes = CHUNK_BYTES) {
  if (!Number.isSafeInteger(chunkBytes) || chunkBytes < 1 || chunkBytes >= 2 ** 31) throw new Error("Invalid chunk size");
  const input = await fs.open(archive, "r");
  const wholeHash = crypto.createHash("sha512");
  const chunks = [];
  const buffer = Buffer.alloc(Math.min(1024 * 1024, chunkBytes));
  try {
    const before = await input.stat();
    if (!before.isFile() || before.size === 0 || Math.ceil(before.size / chunkBytes) > 999) throw new Error("Invalid runtime archive");
    let consumed = 0;
    while (consumed < before.size) {
      const suffix = `.part${String(chunks.length + 1).padStart(3, "0")}`;
      const destination = archive + suffix;
      const output = await fs.open(destination, "wx");
      const hash = crypto.createHash("sha512");
      let size = 0;
      try {
        while (size < chunkBytes && consumed < before.size) {
          const length = Math.min(buffer.length, chunkBytes - size, before.size - consumed);
          const { bytesRead } = await input.read(buffer, 0, length, null);
          if (!bytesRead) throw new Error("Runtime archive truncated while splitting");
          const data = buffer.subarray(0, bytesRead);
          hash.update(data);
          wholeHash.update(data);
          let written = 0;
          while (written < bytesRead) {
            const result = await output.write(data, written, bytesRead - written, null);
            if (!result.bytesWritten) throw new Error("Could not write runtime chunk");
            written += result.bytesWritten;
          }
          size += bytesRead;
          consumed += bytesRead;
        }
      } finally { await output.close(); }
      chunks.push({ file: path.basename(destination), suffix, bytes: size, sha512: hash.digest("hex").toUpperCase() });
    }
    const after = await input.stat();
    if (before.size !== after.size || before.mtimeMs !== after.mtimeMs || wholeHash.digest("hex").toUpperCase() !== expectedHash.toUpperCase()) {
      throw new Error("Runtime archive changed or did not match the builder checksum");
    }
  } finally { await input.close(); }
  return chunks;
}

function downloadInclude(chunks) {
  const lines = ["; Generated from the reviewed runtime archive; do not edit.", "!define RECALL_CHUNK_TRANSPORT 1", "!macro downloadApplicationFiles"];
  chunks.forEach((chunk, index) => {
    if (!Number.isSafeInteger(chunk.bytes) || chunk.bytes < 1 || chunk.bytes >= 2 ** 31) throw new Error("Invalid chunk byte count");
    const number = String(index + 1).padStart(3, "0");
    // All variable text is a numeric suffix or digest; the URL has already passed configuration validation.
    lines.push(`  recall_download_${number}:`,
      `    DetailPrint "Downloading application part ${index + 1} of ${chunks.length}..."`,
      `    inetc::get /USERAGENT "Recall Installer" /RESUME "" "\${APP_PACKAGE_URL}${chunk.suffix}" "$PLUGINSDIR\\package.part${number}" /END`,
      "    Pop $0",
      `    StrCmp $0 "OK" recall_check_${number}`,
      `    StrCmp $0 "Cancelled" recall_chunks_cancel`,
      `    Goto recall_retry_${number}`,
      `  recall_check_${number}:`,
      "    ClearErrors",
      `    \${StdUtils.HashFile} $3 "SHA2-512" "$PLUGINSDIR\\package.part${number}"`,
      `    IfErrors recall_bad_${number}`,
      `    StrCmp $3 "${chunk.sha512}" recall_next_${number}`,
      `  recall_bad_${number}:`,
      `    Delete "$PLUGINSDIR\\package.part${number}"`,
      `  recall_retry_${number}:`,
      "    IfSilent recall_chunks_cancel",
      `    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "Application part ${index + 1} could not be downloaded or verified. Check your connection and disk space, then retry." IDRETRY recall_download_${number}`,
      "    Goto recall_chunks_cancel",
      `  recall_next_${number}:`);
  });
  lines.push(
    '  DetailPrint "Preparing the verified application files..."',
    '  ClearErrors',
    '  FileOpen $0 "$PLUGINSDIR\\package.7z" w',
    '  IfErrors recall_chunks_join_failed',
    '  FileClose $0');
  chunks.forEach((chunk, index) => {
    const number = String(index + 1).padStart(3, "0");
    lines.push(`  \${StdUtils.AppendToFile} $0 "$PLUGINSDIR\\package.part${number}" "$PLUGINSDIR\\package.7z" 0 0`,
      `  StrCmp $0 "${chunk.bytes}" recall_joined_${number} recall_chunks_join_failed`,
      `  recall_joined_${number}:`);
  });
  lines.push('  Goto recall_chunks_joined',
    '  recall_chunks_join_failed:',
    '  MessageBox MB_OK|MB_ICONSTOP "The application package could not be prepared. Check available disk space and run setup again." /SD IDOK',
    '  Goto recall_chunks_cancel',
    '  recall_chunks_joined:');
  chunks.forEach((_, index) => lines.push(`    Delete "$PLUGINSDIR\\package.part${String(index + 1).padStart(3, "0")}"`));
  lines.push('    StrCpy $packageFile "$PLUGINSDIR\\package.7z"', '    Goto recall_chunks_done', '  recall_chunks_cancel:', '    SetErrorLevel 2', '    Quit', '  recall_chunks_done:', '!macroend', '');
  return lines.join("\n");
}

async function prepareTransport(defines) {
  if (!defines.APP_64 || !defines.APP_64_HASH || defines.APP_32 || defines.APP_ARM64) throw new Error("Expected one x64 runtime archive");
  const payloadUrl = new URL(process.env.RECALL_INSTALLER_PACKAGE_URL);
  if (decodeURIComponent(payloadUrl.pathname.split("/").pop()) !== path.basename(defines.APP_64)) {
    throw new Error(`The release URL must end in ${path.basename(defines.APP_64)} so uploaded chunk names match their download URLs`);
  }
  const directory = process.env.RECALL_NSIS_INTEGRITY_DIR;
  if (!directory) throw new Error("Prepare NSIS integrity files first");
  const chunks = await splitArchive(defines.APP_64, defines.APP_64_HASH);
  await fs.writeFile(path.join(directory, "webPackage.nsh"), downloadInclude(chunks), { flag: "wx" });
  await fs.writeFile(path.join(path.dirname(defines.APP_64), "runtime-chunks.json"), JSON.stringify({ version: 1, archive: path.basename(defines.APP_64), sha512: defines.APP_64_HASH, chunks }, null, 2), { flag: "wx" });
  return false;
}

if (require.main === module) {
  const { build, Platform, Arch } = require("electron-builder");
  build({
    projectDir: path.resolve(__dirname, ".."),
    config: path.resolve(__dirname, "../electron-builder.installed.cjs"),
    targets: Platform.WINDOWS.createTarget(["nsis-web"], Arch.x64),
    publish: "never",
    effectiveOptionComputed: async ([defines]) => prepareTransport(defines),
  }).catch(error => { console.error(error); process.exitCode = 1; });
}
module.exports = { splitArchive, downloadInclude, prepareTransport };


// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * New stream: hand Recall a recording or a Twitch VOD and start a scan.
 *
 * One stream opens on a single drop zone that also takes a pasted link. Once a
 * source is picked, its poster card, its name and avatar cutout, and how long
 * it will take on this PC take the zone's place. Several VODs is a paste box
 * that checks every link and scans them in order. Scan mode lives in Settings;
 * this page only says which one will run.
 */
import { useEffect, useState } from "react";
import { StudioSwitch } from "../components/StudioControls";
import { fmtDurationHuman, sourceName } from "../lib/format";
import { AlertTriangle, Broadcast, Check, ChevronRight, Download, Folder, FolderOpen, Play, Plus, Queue, RefreshCw, Twitch, X } from "../lib/icons";
import { parseTwitchVodUrls } from "../lib/twitchBatch";
import { clockIn, estimateScanSeconds, fmtEstimate, type ScanRates } from "./estimate";
import { batchRowReady, titleSuggestsVtuberMode, TWITCH_VOD_URL, type BatchRow, type ImportMode, type ProbeMeta } from "./importModel";

type ScanMode = "fast" | "quality";

export interface NewScanViewProps {
  importMode: ImportMode;
  onImportMode: (mode: ImportMode) => void;
  source: string;
  sourceKind: "local" | "twitch";
  twitchUrl: string;
  setTwitchUrl: (value: string) => void;
  projectName: string;
  setProjectName: (value: string) => void;
  /** The scan mode from Settings, shown but not changed here. */
  processingMode: ScanMode;
  /** Scans running or waiting now; a new one joins behind them. */
  scansAhead: number;
  /** This PC's measured scan speed; null until it has finished a scan. */
  rates: ScanRates | null;
  /** Seconds until a scan added now would start; null when unknown. */
  queueWait: number | null;
  probeMeta: ProbeMeta | null;
  probing: boolean;
  probeError: boolean;
  vtuberConfirmed: boolean;
  onVtuberConfirmed: (confirmed: boolean) => void;
  onBrowse: () => void;
  /** Fetch the typed link, or the one given (a paste lands before state does). */
  onFetch: (url?: string) => void;
  onClear: () => void;
  onRetry: () => void;
  onStart: () => void;
  onLive?: () => void;
  onScanSettings?: () => void;
  batchRows: BatchRow[];
  batchPaste: string;
  setBatchPaste: (value: string) => void;
  batchStarting: boolean;
  onAddBatchUrls: (text: string) => void;
  onBatchTitle: (id: string, title: string) => void;
  onBatchVtuber: (id: string, confirmed: boolean) => void;
  onRemoveBatchRow: (id: string) => void;
  onRetryBatchRow: (row: BatchRow) => void;
  onStartBatch: () => void;
}

const modeLabel = (mode: ScanMode) => (mode === "quality" ? "Best quality" : "Smart scan");

function queueNote(ahead: number) {
  if (!ahead) return "Starts right away. You can leave this screen, the scan keeps going.";
  return `Joins the queue behind ${ahead} ${ahead === 1 ? "scan" : "scans"}. Scans keep going with the window closed.`;
}

function startsLabel(ahead: number, wait: number | null) {
  if (!ahead) return "Right away";
  if (wait == null) return `After ${ahead} ${ahead === 1 ? "scan" : "scans"}`;
  return `About ${clockIn(wait)}`;
}

const LEARNING = "Learns this PC's speed on the first scan";

/** True while a file is dragged over the window, for the drop zone's glow. The
 * app shell owns the actual drop. */
function useDragOver() {
  const [over, setOver] = useState(false);
  useEffect(() => {
    let depth = 0;
    const enter = (event: DragEvent) => {
      if (!event.dataTransfer?.types.includes("Files")) return;
      depth += 1;
      setOver(true);
    };
    const leave = () => { depth = Math.max(0, depth - 1); if (!depth) setOver(false); };
    const done = () => { depth = 0; setOver(false); };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", done);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", done);
    };
  }, []);
  return over;
}

export function NewScanView(props: NewScanViewProps) {
  const { importMode, onImportMode } = props;
  return (
    <div className="page imp2">
      <div className="ph">
        <div>
          <h1 className="disp">New stream</h1>
          <div className="sub">Hand Recall a stream and it finds the moments chat lost it over. Everything stays on this PC.</div>
        </div>
        <div className="actions">
          <div className="seg" role="radiogroup" aria-label="How many streams">
            <button type="button" role="radio" aria-checked={importMode === "single"} onClick={() => onImportMode("single")}><Folder aria-hidden="true" />One stream</button>
            <button type="button" role="radio" aria-checked={importMode === "batch"} onClick={() => onImportMode("batch")}><Queue aria-hidden="true" />Several VODs</button>
          </div>
        </div>
      </div>
      {importMode === "batch" ? <BatchImport {...props} /> : <SingleImport {...props} />}
    </div>
  );
}

function SingleImport(props: NewScanViewProps) {
  const { source, twitchUrl, setTwitchUrl, onFetch, onBrowse, onImportMode, onLive, processingMode, scansAhead, queueWait, rates, onScanSettings } = props;
  const fourHours = estimateScanSeconds(rates, 4 * 3600, true);
  const over = useDragOver();
  const twitchValid = TWITCH_VOD_URL.test(twitchUrl.trim());

  if (source) return <SourceChosen {...props} />;

  return (
    <>
      <section className={`composer ${over ? "over" : ""}`} aria-label="Add a stream">
        <div className="cmp-orb" aria-hidden="true"><Download /></div>
        <h2 className="disp">{over ? "Let go to add it" : "Drop a recording, or paste a link"}</h2>
        <form
          className="cmp-field"
          onSubmit={(event) => { event.preventDefault(); if (twitchValid) onFetch(); }}
        >
          <Twitch aria-hidden="true" />
          <input
            className="field"
            value={twitchUrl}
            onChange={(event) => setTwitchUrl(event.target.value)}
            onPaste={(event) => {
              // A pasted link is the whole intent: fetch it without a second click.
              const pasted = event.clipboardData.getData("text").trim();
              if (TWITCH_VOD_URL.test(pasted)) {
                event.preventDefault();
                setTwitchUrl(pasted);
                onFetch(pasted);
              }
            }}
            placeholder="https://twitch.tv/videos/…"
            aria-label="Twitch VOD link"
            aria-invalid={twitchUrl.trim().length > 0 && !twitchValid}
          />
          <button type="submit" className="btn heat" disabled={!twitchValid}>Fetch</button>
        </form>
        {twitchUrl.trim().length > 0 && !twitchValid && (
          <p className="cmp-err" role="alert">That isn't a VOD link. Copy it from a twitch.tv/videos page.</p>
        )}
        <div className="cmp-alt">
          <button type="button" className="btn ghost" onClick={onBrowse}><FolderOpen aria-hidden="true" />Choose a file</button>
          <button type="button" className="btn ghost" onClick={() => onImportMode("batch")}><Queue aria-hidden="true" />Queue several VODs</button>
          {onLive && <button type="button" className="btn ghost" onClick={onLive}><Broadcast aria-hidden="true" />Mark a live stream instead</button>}
        </div>
        <small className="t3">MP4, MKV, MOV, WebM, AVI, M4V · public Twitch VODs · nothing uploads</small>
      </section>

      <section className="lands2 glass" aria-label="How this scan runs">
        <div className="l2-item"><span className="t3">Scan mode</span><b>{modeLabel(processingMode)}</b></div>
        <div className="l2-item"><span className="t3">A 4h Twitch VOD takes</span><b>{fourHours != null ? `${fmtEstimate(fourHours)} on this PC` : LEARNING}</b></div>
        <div className="l2-item"><span className="t3">{scansAhead ? `Starts after ${scansAhead} ${scansAhead === 1 ? "scan" : "scans"}` : "Starts"}</span><b>{startsLabel(scansAhead, queueWait)}</b></div>
        <span className="sp" />
        {onScanSettings && <button type="button" className="btn sm ghost" onClick={onScanSettings}>Scan settings<ChevronRight aria-hidden="true" /></button>}
      </section>
    </>
  );
}

function SourceChosen({
  source, sourceKind, probeMeta, probing, probeError, onClear, onRetry, onStart, onBrowse,
  projectName, setProjectName, processingMode, vtuberConfirmed, onVtuberConfirmed, scansAhead, rates, queueWait, onScanSettings,
}: NewScanViewProps) {
  const ready = !!(probeMeta?.thumbnail && Number(probeMeta.duration) > 0 && !probing && !probeError);
  const twitch = sourceKind === "twitch";
  const title = probeMeta?.title || (twitch ? "Twitch VOD" : sourceName(source, 80));
  const suggestV = titleSuggestsVtuberMode(probeMeta?.title);
  const duration = Number(probeMeta?.duration) || 0;
  const takes = ready ? estimateScanSeconds(rates, duration, twitch) : null;
  const wait = scansAhead ? queueWait : 0;
  const streamed = probeMeta?.source_date ? new Date(probeMeta.source_date) : null;
  const meta = [
    twitch ? probeMeta?.creator : null,
    duration ? fmtDurationHuman(duration) : null,
    streamed && !Number.isNaN(streamed.getTime()) ? `Streamed ${streamed.toLocaleDateString(undefined, { month: "short", day: "numeric" })}` : null,
    twitch ? probeMeta?.game : null,
  ].filter(Boolean) as string[];

  return (
    <div className="imp-stack">
      <section className={`src glass ${probeError ? "is-bad" : ""}`} aria-label="Chosen source" aria-busy={probing}>
        <div className={`src-poster ${probing && !probeMeta?.thumbnail ? "skeleton" : ""}`}>
          {probeMeta?.thumbnail && <img src={probeMeta.thumbnail} alt="" />}
          {probeError && <AlertTriangle aria-hidden="true" />}
        </div>
        <div className="src-copy">
          {probing ? (
            <>
              <span className="eyebrow">{twitch ? "Twitch VOD" : "Local recording"}</span>
              <div className="skeleton" style={{ height: 30, width: "70%" }} />
              <div className="skeleton" style={{ height: 14, width: "45%" }} />
              <span className="t3 src-note">Reading the title, creator, and runtime…</span>
            </>
          ) : probeError ? (
            <>
              <span className="eyebrow is-bad">Couldn't read this {twitch ? "VOD" : "file"}</span>
              <h3 className="disp">{title}</h3>
              <p className="t3 src-note">
                {twitch
                  ? "Check the VOD is public and still online. Sub-only VODs can't be fetched, download it and choose the file instead."
                  : "The file may be damaged or still being written. Try it again, or pick another recording."}
              </p>
              <div className="row">
                <button type="button" className="btn sm" onClick={onRetry}><RefreshCw aria-hidden="true" />Try again</button>
                {twitch && <button type="button" className="btn sm ghost" onClick={onBrowse}><FolderOpen aria-hidden="true" />Use a downloaded file</button>}
              </div>
            </>
          ) : (
            <>
              <span className="eyebrow is-ready">{twitch ? "Twitch VOD · ready" : "Local recording · ready"}</span>
              <h3 className="disp">{title}</h3>
              {meta.length > 0 && <div className="meta">{meta.map((item) => <span key={item}>{item}</span>)}</div>}
              {twitch && <p className="t3 src-note">Recall downloads it once, keeps it while you work on it, then cleans it up.</p>}
            </>
          )}
        </div>
        <button type="button" className="btn sm ghost src-change" onClick={onClear}><X aria-hidden="true" />Change</button>
      </section>

      <section className="panel glass imp-opts" aria-label="Scan options">
        <label className="lbl" htmlFor="imp-name">Session name</label>
        <input
          id="imp-name"
          className="field"
          value={projectName}
          onChange={(event) => setProjectName(event.target.value)}
          placeholder={probeMeta?.title || "Name this session"}
          maxLength={200}
        />
        <div className="setrow">
          <div className="st">
            <b>VTuber / PNGtuber cutout{suggestV && <span className="badge heat">Suggested from the title</span>}</b>
            <small>{vtuberConfirmed ? "Uses Auto framing for this scan only and moves the avatar to the top. Your saved layout stays unchanged." : suggestV ? "The title suggests a virtual avatar. Turn this on only if the stream uses one." : "Moves a virtual avatar to the top of vertical clips. Leave it off for a real facecam."}</small>
          </div>
          <StudioSwitch checked={vtuberConfirmed} onChange={onVtuberConfirmed} label="VTuber / PNGtuber cutout" disabled={!ready} />
        </div>
      </section>

      {ready && (
        <section className="plan" aria-label="When it's ready">
          <div><span>Starts</span><b className="num">{startsLabel(scansAhead, queueWait)}</b></div>
          <div><span>Scan takes</span><b className="num">{takes != null ? fmtEstimate(takes) : "Learning"}</b><small>{takes != null ? `${modeLabel(processingMode)}${twitch ? ", download included" : ""}` : LEARNING}</small></div>
          <div><span>Ready to review</span><b className="num">{takes == null ? "–" : wait != null ? `About ${clockIn(wait + takes)}` : "After the queue"}</b>{takes != null && wait == null && <small>{fmtEstimate(takes)} once it starts</small>}{onScanSettings && <button type="button" className="linkish" onClick={onScanSettings}>Change scan mode in Settings</button>}</div>
        </section>
      )}

      <div className="row imp-go">
        <button type="button" className="btn heat lg" disabled={!ready} onClick={onStart}>
          <Play aria-hidden="true" />{probing ? "Checking the source…" : "Start scan"}
        </button>
        <span className="t3">{queueNote(scansAhead)}</span>
      </div>
    </div>
  );
}

function BatchImport({
  batchRows, batchPaste, setBatchPaste, batchStarting, onAddBatchUrls, onBatchTitle, onBatchVtuber, onRemoveBatchRow, onRetryBatchRow, onStartBatch,
  processingMode, scansAhead, rates, queueWait, onScanSettings,
}: NewScanViewProps) {
  const ready = batchRows.filter(batchRowReady);
  const checking = batchRows.some((row) => row.probing);
  const draftCount = parseTwitchVodUrls(batchPaste).length;
  const total = ready.reduce((sum, row) => sum + (Number(row.probeMeta?.duration) || 0), 0);
  const takes = estimateScanSeconds(rates, total, true);
  const wait = scansAhead ? queueWait : 0;

  return (
    <div className="imp-stack">
      <section className="panel glass" aria-label="Add VOD links">
        <label className="lbl" htmlFor="batch-vod-urls">Twitch VOD links, one per line, in the order to scan</label>
        <textarea
          id="batch-vod-urls"
          className="field imp-batch-text"
          rows={4}
          value={batchPaste}
          onChange={(event) => setBatchPaste(event.target.value)}
          placeholder={"https://twitch.tv/videos/…\nhttps://twitch.tv/videos/…"}
        />
        <div className="row imp-batch-foot">
          <button type="button" className="btn" onClick={() => onAddBatchUrls(batchPaste)} disabled={!draftCount}>
            <Plus aria-hidden="true" />{draftCount ? `Check ${draftCount} ${draftCount === 1 ? "link" : "links"}` : "Check links"}
          </button>
          <span className="t3">{batchPaste.trim() && !draftCount ? "No VOD links in there yet. They look like twitch.tv/videos/123456789." : "Sub-only VODs need a downloaded file."}</span>
        </div>
      </section>

      {batchRows.length ? (
        <>
          <div className="imp-batch-list" aria-label="VODs in this batch">
            {batchRows.map((row, index) => {
              const rowReady = batchRowReady(row);
              return (
                <article key={row.id} className={`brow glass ${row.probeError ? "is-bad" : ""}`}>
                  <span className="n num">{String(index + 1).padStart(2, "0")}</span>
                  <span className={`brow-poster ${row.probing ? "skeleton" : ""}`}>
                    {row.probeMeta?.thumbnail && <img src={row.probeMeta.thumbnail} alt="" />}
                  </span>
                  <div className="brow-copy">
                    <input className="brow-name" value={row.title} onChange={(event) => onBatchTitle(row.id, event.target.value)} aria-label={`Session name for VOD ${index + 1}`} maxLength={200} />
                    <small>
                      {row.probing ? "Reading the title and runtime…"
                        : row.probeError ? "Couldn't read this VOD. It may be sub-only or deleted."
                          : [row.probeMeta?.creator, row.probeMeta?.duration ? fmtDurationHuman(Number(row.probeMeta.duration)) : null].filter(Boolean).join(" · ")}
                    </small>
                  </div>
                  <div className="row brow-actions">
                    {rowReady && (
                      <button
                        type="button"
                        className={`chip ${row.vtuberConfirmed ? "is-on" : ""}`}
                        aria-pressed={row.vtuberConfirmed}
                        title={row.vtuberConfirmed ? "Auto framing and avatar cutout for this VOD only. Your saved layout stays unchanged." : titleSuggestsVtuberMode(row.probeMeta?.title) ? "The title suggests a virtual avatar. Turn this on only if the stream uses one." : "Uses Auto framing and avatar cutout for this VOD only."}
                        onClick={() => onBatchVtuber(row.id, !row.vtuberConfirmed)}
                      >
                        {row.vtuberConfirmed && <Check aria-hidden="true" />}VTuber
                      </button>
                    )}
                    {row.probing ? <span className="badge">Checking</span>
                      : row.probeError ? <button type="button" className="btn sm" onClick={() => onRetryBatchRow(row)}><RefreshCw aria-hidden="true" />Try again</button>
                        : <span className="badge keep">Ready</span>}
                    <button type="button" className="icon-btn" aria-label={`Remove ${row.title}`} onClick={() => onRemoveBatchRow(row.id)}><X aria-hidden="true" /></button>
                  </div>
                </article>
              );
            })}
          </div>

          <section className="runsum glass" aria-label="Batch summary">
            <div><b className="disp num">{ready.length}</b><span>{ready.length === 1 ? "VOD ready" : "VODs ready"}</span></div>
            <div><b className="disp num">{total ? fmtDurationHuman(total) : "–"}</b><span>of stream</span></div>
            <div><b className="disp num">{takes != null ? fmtEstimate(takes) : "–"}</b><span>{takes != null ? "to scan them all" : LEARNING}</span></div>
            {takes != null && wait != null && <div><b className="disp num">{clockIn(wait + takes)}</b><span>ready to review by</span></div>}
            <span className="sp" />
            <div className="runsum-go">
              {onScanSettings
                ? <button type="button" className="btn sm ghost" onClick={onScanSettings}>{modeLabel(processingMode)}<ChevronRight aria-hidden="true" /></button>
                : <span className="t3">{modeLabel(processingMode)}</span>}
              <button type="button" className="btn heat lg" disabled={!ready.length || checking || batchStarting} onClick={onStartBatch}>
                <Play aria-hidden="true" />
                {batchStarting ? "Starting…" : checking ? "Checking links…" : `Scan ${ready.length} in order`}
              </button>
            </div>
          </section>
          <p className="t3 imp-batch-note">One VOD uses the machine at a time. Finished streams wait in your Library, ready for one review pass.</p>
        </>
      ) : (
        <section className="imp-batch-empty glass">
          <h3 className="disp">Line up tonight's VODs</h3>
          <p className="t3">Recall checks every link now, then scans them one after another while you're away.</p>
        </section>
      )}
    </div>
  );
}

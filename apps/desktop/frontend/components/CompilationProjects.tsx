// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Compilations — the cutting bench.
 *
 * Two surfaces, never both at once:
 *
 *   Shelf   Your compilations as 9:16 posters. Opening one gives it the whole
 *           workspace. There is no second rail: a saved-drafts list beside the
 *           shell's own project rail put 508px of near-identical vertical lists
 *           in front of the work, which is what made the page hard to navigate.
 *
 *   Bench   One montage. The player takes real scale on the left, the moments
 *           that make it run beside it as a storyboard you take in at a glance,
 *           and a ribbon under the player draws the reel to scale so the shape
 *           of it — 24 short kills, then a finale worth 40% of the runtime — is
 *           visible without reading anything.
 *
 * A moment is a poster, not a database row. Twenty-six rows at 140px each cost
 * 3,640px of scrolling for a 75-second video and repeated one sentence twenty
 * four times; the tile carries only what the frame cannot say — its place in
 * the reel and how long it runs — and the moment you select fills ONE inspector
 * row that owns the actions. Twenty-six moments cost three buttons, not 78.
 *
 * Creation is a step you enter and leave, not a permanent 166px header. It asks
 * what the montage is made of and how long it should run, in the creator's
 * words; `One match / Recipes / Date range` was a taxonomy of input methods,
 * which is how the generator thinks and not how anyone edits.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  Clock,
  Download,
  Film,
  HardDrive,
  Play,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
  Trophy,
  Volume2,
  VolumeX,
  X,
} from "../lib/icons";
import { apiFetch, apiUrl } from "../lib/api";
import { fmtClock, fmtDurationHuman } from "../lib/format";
import { plural } from "../lib/copy";
import {
  approveCompilationProject,
  cancelCompilationPreparation,
  buildCompilationReel,
  saveCompilationReel,
  setCompilationAudio,
  cutNextCompilationMoment,
  deleteCompilationProject,
  generateCompilationProjectFromMatch,
  generateCompilationProjectFromRecipe,
  generateCompilationProjectFromTheme,
  listCompilationRecipes,
  listSessionMatches,
  type CompilationRecipe,
  type SessionMatch,
  getCompilationPreparation,
  getCompilationProject,
  listCompilationProjects,
  renameCompilationProject,
  saveCompilationProjectItems,
  startCompilationPreparation,
  type CompilationPreparationPlan,
  type CompilationProject,
  type CompilationProjectItem,
  type CompilationProjectSummary,
} from "../lib/compilationProjects";

function inputDate(value: Date): string {
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, "0");
  const day = String(value.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function projectDate(value: string | null): string {
  if (!value) return "All dates";
  const parsed = new Date(`${value}T12:00:00`);
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function runtime(seconds: number): string {
  return seconds < 60 ? `${Math.round(seconds)}s` : fmtDurationHuman(seconds);
}

function includedItems(project: CompilationProject): CompilationProjectItem[] {
  return project.items.filter((item) => item.included && item.clip_id);
}

function storageSize(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.max(0, bytes / 1024 ** 2).toFixed(0)} MB`;
}

const ACTIVE_PREPARATION = new Set(["queued", "running", "cancelling"]);

/** What the montage is cut from. The creator's two words for it, not three. */
type SourceKind = "match" | "theme";

/**
 * Montage lengths, in seconds. Length is the creator's control — the generator
 * solves for the runtime it is asked for — so these are stated as runtimes.
 *
 * There is deliberately no "every moment" option: the packer derives the
 * closing win's length from the requested runtime, so a very large target asks
 * for a fifteen-minute victory screen. Cutting every moment needs a mode of its
 * own in `core/compilation_projects.py`, not a bigger number here.
 */
const REEL_LENGTHS: Array<[number, string, string]> = [
  [30, "0:30", "Punchy"],
  [45, "0:45", "Tight"],
  [60, "1:00", "Standard"],
  [90, "1:30", "Room to breathe"],
  [120, "2:00", "Long form"],
  [180, "3:00", "Everything that fits"],
];

interface ScannedSession {
  id: string;
  session_name: string;
  source_date: string | null;
  created_at: string;
}

/** Completed scans, newest first — the sessions a match reel can come from. */
async function listScannedSessions(baseUrl?: string): Promise<ScannedSession[]> {
  const response = await apiFetch("/jobs", undefined, baseUrl);
  if (!response.ok) throw new Error("Recall could not list your scanned sessions.");
  const rows = (await response.json()) as Array<Record<string, unknown>>;
  return (Array.isArray(rows) ? rows : [])
    .filter((row) => String(row.status || "") === "completed")
    .map((row) => ({
      id: String(row.id || ""),
      session_name: String(row.session_name || "Untitled session"),
      source_date: (row.source_date as string) || null,
      created_at: String(row.created_at || ""),
    }))
    .filter((row) => row.id);
}

function matchClock(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}`
    : `${minutes}m`;
}

/** A match states its outcome as a word. Colour alone is not a status. */
function matchOutcome(match: SessionMatch): string {
  if (match.has_win) return "Victory";
  if (match.placement) return `#${match.placement}`;
  return match.outcome === "eliminated" ? "Eliminated" : "Unknown";
}

/** The bare noun, for the places a count is already rendered on its own. */
function noun(count: number, singular: string, pluralForm?: string): string {
  return count === 1 ? singular : pluralForm ?? `${singular}s`;
}

function matchKills(match: SessionMatch): string {
  if (match.kill_estimate <= 0) return "Match";
  // A milestone above what Recall counted means the real total is at least
  // this: "25+ kills", never a number the match did not reach.
  const floor = match.milestone_kills > match.detected_kills ? "+" : "";
  return `${match.kill_estimate}${floor} ${noun(match.kill_estimate, "kill")}`;
}

/* ============================================================
   Creation — the step you enter and leave
   ============================================================ */

interface NewCompilationRequest {
  kind: SourceKind;
  match?: SessionMatch;
  recipe?: CompilationRecipe;
  recipeGame?: string;
  query: string;
  dateFrom: string;
  dateTo: string;
  title: string;
  seconds: number;
}

/**
 * Two questions in the creator's words: what the montage is made of, then how
 * long it runs. It is a dialog because the answer determines everything
 * downstream and deserves protected focus — not because a form needed a box.
 */
function NewCompilationSheet({
  open,
  busy,
  sessions,
  matches,
  matchesLoading,
  recipes,
  sessionId,
  onSessionChange,
  initialMatchIndex,
  onCancel,
  onCreate,
}: {
  open: boolean;
  busy: boolean;
  sessions: ScannedSession[];
  matches: SessionMatch[];
  matchesLoading: boolean;
  recipes: CompilationRecipe[];
  sessionId: string;
  onSessionChange: (id: string) => void;
  initialMatchIndex?: number;
  onCancel: () => void;
  onCreate: (request: NewCompilationRequest) => void;
}) {
  const now = useMemo(() => new Date(), []);
  const [step, setStep] = useState<1 | 2>(1);
  const [kind, setKind] = useState<SourceKind>("match");
  const [matchIndex, setMatchIndex] = useState<number>();
  const [recipeId, setRecipeId] = useState<string>();
  const [recipeGame, setRecipeGame] = useState<Record<string, string>>({});
  const [query, setQuery] = useState("");
  const [dateFrom, setDateFrom] = useState(inputDate(new Date(now.getFullYear(), now.getMonth(), 1)));
  const [dateTo, setDateTo] = useState(inputDate(now));
  const [title, setTitle] = useState("");
  const [seconds, setSeconds] = useState(60);
  const dialogRef = useRef<HTMLDivElement>(null);
  const chosenMatchRef = useRef<HTMLButtonElement>(null);

  // Opening resets to the first question; a half-finished answer from last time
  // is worse than no answer.
  useEffect(() => {
    if (!open) return;
    setStep(1);
    setMatchIndex(initialMatchIndex);
    setTitle("");
    dialogRef.current?.focus();
  }, [open, initialMatchIndex]);

  // A match is pre-selected, and a dozen games in a session put it below the
  // fold: a choice the creator cannot see reads as no choice at all.
  useEffect(() => {
    if (!open || matchIndex === undefined) return;
    // Optional call: scrolling is a nicety, and an environment without it
    // must still open the sheet.
    chosenMatchRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [open, matchIndex, matches.length]);

  useEffect(() => {
    if (matchIndex === undefined && matches.length) {
      // The game the creator came for reads first: a win, then the most kills.
      const best = [...matches].sort((a, b) => (
        Number(b.has_win) - Number(a.has_win) || b.kill_estimate - a.kill_estimate
      ))[0];
      setMatchIndex(best?.match_index);
    }
  }, [matches, matchIndex]);

  if (!open) return null;

  const match = matches.find((entry) => entry.match_index === matchIndex);
  const recipe = recipes.find((entry) => entry.id === recipeId);
  const canAdvance = kind === "match" ? !!match : true;

  const suggestedTitle = kind === "match"
    ? match ? `${matchKills(match)} — ${matchOutcome(match)}` : "Match montage"
    : recipe
      ? recipe.label.replace("{game}", recipeGame[recipe.id] || recipe.games[0] || "").trim()
      : query.trim() || `Best of ${now.toLocaleDateString(undefined, { month: "long" })}`;

  const submit = () => {
    if (busy) return;
    onCreate({
      kind,
      match,
      recipe,
      recipeGame: recipe?.needs_game ? (recipeGame[recipe.id] || recipe.games[0]) : undefined,
      query,
      dateFrom,
      dateTo,
      title: title.trim() || suggestedTitle,
      seconds,
    });
  };

  return (
    <div
      className="project-modal open compilation-sheet-modal"
      onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onCancel(); }}
    >
      <div
        className="project-dialog compilation-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="compilation-sheet-title"
        tabIndex={-1}
        ref={dialogRef}
        onKeyDown={(event) => { if (event.key === "Escape" && !busy) onCancel(); }}
      >
        <div className="compilation-sheet-head">
          <h2 id="compilation-sheet-title">
            {step === 1 ? "New compilation" : "How long?"}
          </h2>
          <p>
            {step === 1
              ? "Two questions, then Recall cuts it."
              : "Length is your control, not clip count — Recall solves for the runtime you ask for."}
          </p>
        </div>

        {step === 1 ? (
          <div className="compilation-sheet-body">
            <div className="compilation-field">
              <span className="compilation-field-label" id="compilation-source-label">
                What&rsquo;s it made of?
              </span>
              <div className="compilation-sources" role="radiogroup" aria-labelledby="compilation-source-label">
                <button
                  type="button"
                  role="radio"
                  aria-checked={kind === "match"}
                  className={`compilation-source ${kind === "match" ? "is-on" : ""}`}
                  onClick={() => setKind("match")}
                >
                  <span className="compilation-source-mark" aria-hidden="true"><Trophy size={17} /></span>
                  <span className="compilation-source-copy">
                    <strong>One game</strong>
                    <small>Every kill from a single match, ending on the win</small>
                  </span>
                  <span className="compilation-radio" aria-hidden="true" />
                </button>
                <button
                  type="button"
                  role="radio"
                  aria-checked={kind === "theme"}
                  className={`compilation-source ${kind === "theme" ? "is-on" : ""}`}
                  onClick={() => setKind("theme")}
                >
                  <span className="compilation-source-mark" aria-hidden="true"><Search size={17} /></span>
                  <span className="compilation-source-copy">
                    <strong>A theme, across sessions</strong>
                    <small>Best snipes, jump scares, clutches — from every VOD you have scanned</small>
                  </span>
                  <span className="compilation-radio" aria-hidden="true" />
                </button>
              </div>
            </div>

            {kind === "match" ? (
              <>
                <label className="compilation-field">
                  <span className="compilation-field-label">Which session?</span>
                  <select value={sessionId} onChange={(event) => {
                    setMatchIndex(undefined);
                    onSessionChange(event.currentTarget.value);
                  }}>
                    {sessions.map((session) => (
                      <option key={session.id} value={session.id}>
                        {session.source_date ? `${session.source_date} — ` : ""}{session.session_name}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="compilation-field">
                  <span className="compilation-field-label" id="compilation-match-label">Which game in it?</span>
                  <div className="compilation-matches" role="radiogroup" aria-labelledby="compilation-match-label">
                    {matchesLoading ? (
                      <p className="compilation-note">Reading detected events…</p>
                    ) : !matches.length ? (
                      <p className="compilation-note">
                        Recall found no separable matches in this session. Detected events are
                        what split a stream into games, so older scans may not have them.
                      </p>
                    ) : matches.map((entry) => (
                      <button
                        type="button"
                        key={entry.match_index}
                        role="radio"
                        ref={matchIndex === entry.match_index ? chosenMatchRef : undefined}
                        aria-checked={matchIndex === entry.match_index}
                        className={`compilation-match ${matchIndex === entry.match_index ? "is-on" : ""} ${entry.has_win ? "is-win" : ""}`}
                        onClick={() => setMatchIndex(entry.match_index)}
                      >
                        <span className="compilation-match-mark" aria-hidden="true">
                          {entry.has_win ? <Trophy size={16} /> : <Clock size={15} />}
                        </span>
                        <span className="compilation-match-copy">
                          <strong>{matchKills(entry)}</strong>
                          <small>
                            {matchClock(entry.start_time)}–{matchClock(entry.end_time)}
                            {" · "}
                            {plural(entry.moment_count, "moment")}
                            {entry.confidence !== "high" ? " · approximate" : ""}
                          </small>
                        </span>
                        <span className={`compilation-pill ${entry.has_win ? "is-ready" : "is-draft"}`}>
                          {matchOutcome(entry)}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <>
                {recipes.length > 0 && (
                  <div className="compilation-field">
                    <span className="compilation-field-label" id="compilation-recipe-label">Start from a suggestion</span>
                    <div className="compilation-recipes" role="radiogroup" aria-labelledby="compilation-recipe-label">
                      {recipes.map((entry) => {
                        const game = recipeGame[entry.id] || entry.games[0] || "";
                        const label = entry.label.replace(
                          "{game}", game ? game[0].toUpperCase() + game.slice(1) : "",
                        ).trim();
                        return (
                          <button
                            type="button"
                            key={entry.id}
                            role="radio"
                            aria-checked={recipeId === entry.id}
                            className={`compilation-recipe ${recipeId === entry.id ? "is-on" : ""}`}
                            onClick={() => setRecipeId(recipeId === entry.id ? undefined : entry.id)}
                            title={entry.description}
                          >
                            <Sparkles size={13} aria-hidden="true" />
                            {label}
                          </button>
                        );
                      })}
                    </div>
                    {recipe?.needs_game && (
                      <label className="compilation-subfield">
                        <span>Which game?</span>
                        <select
                          aria-label={`Game for ${recipe.label.replace("{game}", "").trim()}`}
                          value={recipeGame[recipe.id] || recipe.games[0] || ""}
                          onChange={(event) => {
                            const next = event.currentTarget.value;
                            setRecipeGame((prev) => ({ ...prev, [recipe.id]: next }));
                          }}
                        >
                          {recipe.games.map((option) => (
                            <option key={option} value={option}>{option}</option>
                          ))}
                        </select>
                      </label>
                    )}
                  </div>
                )}

                {!recipe && (
                  <>
                    <label className="compilation-field">
                      <span className="compilation-field-label">
                        Or describe it <small>Optional</small>
                      </span>
                      <input
                        value={query}
                        onChange={(event) => setQuery(event.currentTarget.value)}
                        placeholder="funny reactions, crazy snipes…"
                        maxLength={240}
                      />
                    </label>
                    <div className="compilation-daterow">
                      <label className="compilation-field">
                        <span className="compilation-field-label">From</span>
                        <input type="date" value={dateFrom} onChange={(event) => setDateFrom(event.currentTarget.value)} />
                      </label>
                      <label className="compilation-field">
                        <span className="compilation-field-label">Through</span>
                        <input type="date" value={dateTo} onChange={(event) => setDateTo(event.currentTarget.value)} />
                      </label>
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        ) : (
          <div className="compilation-sheet-body">
            <div className="compilation-field">
              <span className="compilation-field-label" id="compilation-length-label">Target runtime</span>
              <div className="compilation-lengths" role="radiogroup" aria-labelledby="compilation-length-label">
                {REEL_LENGTHS.map(([value, clock, hint]) => (
                  <button
                    type="button"
                    key={value}
                    role="radio"
                    aria-checked={seconds === value}
                    className={`compilation-length ${seconds === value ? "is-on" : ""}`}
                    onClick={() => setSeconds(value)}
                  >
                    <b className="t-num">{clock}</b>
                    <small>{hint}</small>
                  </button>
                ))}
              </div>
            </div>

            <label className="compilation-field">
              <span className="compilation-field-label">Name it</span>
              <input
                value={title}
                onChange={(event) => setTitle(event.currentTarget.value)}
                placeholder={suggestedTitle}
                maxLength={160}
              />
            </label>

            {kind === "match" && match && (
              <div className="compilation-notice is-info">
                <span className="compilation-notice-mark" aria-hidden="true"><Film size={15} /></span>
                <div>
                  <strong>
                    {plural(match.moment_count, "moment")} detected in this match
                  </strong>
                  <p>
                    Recall keeps the strongest that fit the runtime you asked for
                    {match.has_win ? " and closes on the win." : "."}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="compilation-sheet-foot">
          <button
            type="button"
            className="compilation-btn"
            onClick={() => (step === 1 ? onCancel() : setStep(1))}
            disabled={busy}
          >
            {step === 1 ? "Cancel" : "Back"}
          </button>
          <span className="compilation-spacer" />
          {step === 1 ? (
            <button
              type="button"
              className="cta-accent"
              onClick={() => setStep(2)}
              disabled={!canAdvance}
            >
              Next
            </button>
          ) : (
            <button type="button" className="cta-accent" onClick={submit} disabled={busy}>
              <Sparkles size={15} aria-hidden="true" />
              {busy ? "Cutting…" : "Cut the montage"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   The page
   ============================================================ */

export default function CompilationProjects({
  apiEndpoint,
  compiling,
  onCompile,
  onOpenClip,
}: {
  apiEndpoint?: string;
  compiling: boolean;
  onCompile: (clipIds: string[], title: string) => void | Promise<void>;
  onOpenClip: (jobId: string, clipId: string) => void | Promise<void>;
}) {
  const [projects, setProjects] = useState<CompilationProjectSummary[]>([]);
  const [project, setProject] = useState<CompilationProject>();
  const [titleDraft, setTitleDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string>();
  const [deleteArmed, setDeleteArmed] = useState(false);
  const [preparation, setPreparation] = useState<CompilationPreparationPlan>();
  const [prepareArmed, setPrepareArmed] = useState(false);
  const [preparationAction, setPreparationAction] = useState(false);
  const [building, setBuilding] = useState(false);
  const [buildStep, setBuildStep] = useState("");
  const [sheetOpen, setSheetOpen] = useState(false);
  const [selectedItemId, setSelectedItemId] = useState<string>();
  const cutStop = useRef(false);

  const loadPreparation = useCallback(async (projectId: string) => {
    const plan = await getCompilationPreparation(projectId, apiEndpoint);
    setPreparation(plan);
    return plan;
  }, [apiEndpoint]);

  /** The shelf is the landing surface, so nothing opens itself on first load. */
  useEffect(() => {
    let active = true;
    setLoading(true);
    listCompilationProjects(apiEndpoint)
      .then((listed) => { if (active) setProjects(listed); })
      .catch((error) => active && setMessage(
        error instanceof Error ? error.message : "Compilation projects are unavailable.",
      ))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, [apiEndpoint]);

  const preparationStatus = preparation?.preparation?.status;
  useEffect(() => {
    const projectId = project?.id;
    if (!projectId || !preparationStatus || !ACTIVE_PREPARATION.has(preparationStatus)) return;
    let active = true;
    const refresh = async () => {
      try {
        const [plan, refreshed] = await Promise.all([
          getCompilationPreparation(projectId, apiEndpoint),
          getCompilationProject(projectId, apiEndpoint),
        ]);
        if (!active) return;
        setPreparation(plan);
        setProject(refreshed);
        setProjects((current) => current.map((entry) => entry.id === refreshed.id ? refreshed : entry));
      } catch (error) {
        if (active) setMessage(error instanceof Error ? error.message : "Source preparation status is unavailable.");
      }
    };
    const timer = window.setInterval(() => void refresh(), 1200);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [apiEndpoint, preparationStatus, project?.id]);

  const [recipes, setRecipes] = useState<CompilationRecipe[]>([]);
  const [sessions, setSessions] = useState<ScannedSession[]>([]);
  const [matchSessionId, setMatchSessionId] = useState("");
  const [matches, setMatches] = useState<SessionMatch[]>([]);
  const [matchesLoading, setMatchesLoading] = useState(false);

  useEffect(() => {
    let active = true;
    void listScannedSessions(apiEndpoint)
      .then((rows) => {
        if (!active) return;
        setSessions(rows);
        setMatchSessionId((current) => current || rows[0]?.id || "");
      })
      .catch(() => { if (active) setSessions([]); });
    return () => { active = false; };
  }, [apiEndpoint]);

  useEffect(() => {
    let active = true;
    void listCompilationRecipes(apiEndpoint)
      .then((next) => { if (active) setRecipes(next); })
      .catch(() => { if (active) setRecipes([]); });
    return () => { active = false; };
  }, [apiEndpoint]);

  useEffect(() => {
    if (!matchSessionId) {
      setMatches([]);
      return;
    }
    let active = true;
    setMatchesLoading(true);
    void listSessionMatches(matchSessionId, apiEndpoint)
      .then((found) => { if (active) setMatches(found.matches); })
      .catch(() => { if (active) setMatches([]); })
      .finally(() => { if (active) setMatchesLoading(false); });
    return () => { active = false; };
  }, [apiEndpoint, matchSessionId]);

  const openProject = async (id: string) => {
    setLoading(true);
    setDeleteArmed(false);
    setPrepareArmed(false);
    setMessage(undefined);
    try {
      const [selected, plan] = await Promise.all([
        getCompilationProject(id, apiEndpoint),
        getCompilationPreparation(id, apiEndpoint),
      ]);
      setProject(selected);
      setTitleDraft(selected.title);
      setPreparation(plan);
      setSelectedItemId(selected.items[0]?.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not open that draft.");
    } finally {
      setLoading(false);
    }
  };

  /** Back to the shelf. The bench belongs to one montage at a time. */
  const closeProject = () => {
    setProject(undefined);
    setPreparation(undefined);
    setSelectedItemId(undefined);
    setDeleteArmed(false);
    setMessage(undefined);
  };

  const refreshList = async (preferredId?: string) => {
    const listed = await listCompilationProjects(apiEndpoint);
    setProjects(listed);
    if (preferredId) {
      const [selected] = await Promise.all([
        getCompilationProject(preferredId, apiEndpoint),
        loadPreparation(preferredId),
      ]);
      setProject(selected);
      setTitleDraft(selected.title);
      setSelectedItemId((current) => (
        selected.items.some((item) => item.id === current) ? current : selected.items[0]?.id
      ));
    }
  };

  /** Cut whatever the draft still needs, then stitch. One action, three phases. */
  const finishMontage = async (draft: CompilationProject) => {
    let current = draft;
    setProject(current);
    setTitleDraft(current.title);
    setSelectedItemId(current.items[0]?.id);
    const total = current.uncut_count;
    for (let done = 0; done < total; done += 1) {
      if (cutStop.current) break;
      setBuildStep(`Cutting moment ${done + 1} of ${total}…`);
      const cut = await cutNextCompilationMoment(current.id, apiEndpoint);
      current = cut.project;
      setProject(current);
      if (cut.done) break;
    }
    if (cutStop.current) {
      setMessage(`Stopped after ${total - current.uncut_count} of ${total} moments. Nothing was stitched.`);
      return;
    }
    setBuildStep("Stitching the montage…");
    const built = await buildCompilationReel(current.id, apiEndpoint);
    await refreshList(built.project.id);
    const summary = built.project.selection_summary;
    const skipped = (summary.offline_moment_count || 0) + (summary.oversized_clip_count || 0);
    setMessage(
      `Montage ready — ${plural(built.clips, "moment")}, ${runtime(built.project.reel_duration_seconds || 0)}.`
      + (skipped ? ` ${skipped} more matched but could not be used.` : ""),
    );
  };

  const createFromSheet = async (request: NewCompilationRequest) => {
    if (generating) return;
    setGenerating(true);
    setSheetOpen(false);
    cutStop.current = false;
    setMessage(undefined);
    try {
      if (request.kind === "match" && request.match) {
        setBuildStep(`Choosing the best moments of ${request.match.label}…`);
        await finishMontage(await generateCompilationProjectFromMatch({
          jobId: request.match.job_id,
          matchIndex: request.match.match_index,
          targetDurationSeconds: request.seconds,
          baseUrl: apiEndpoint,
        }));
      } else if (request.recipe) {
        setBuildStep(`Building ${request.title}…`);
        await finishMontage(await generateCompilationProjectFromRecipe({
          recipeId: request.recipe.id,
          game: request.recipeGame,
          baseUrl: apiEndpoint,
        }));
      } else {
        setBuildStep(`Searching every stream for ${request.query.trim() || "kept moments"}…`);
        await finishMontage(await generateCompilationProjectFromTheme({
          title: request.title,
          query: request.query,
          dateFrom: request.dateFrom,
          dateTo: request.dateTo,
          targetDurationSeconds: request.seconds,
          baseUrl: apiEndpoint,
        }));
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not build that montage.");
    } finally {
      setGenerating(false);
      setBuildStep("");
    }
  };

  /** Re-stitch after the creator drops a moment. No re-cutting is needed. */
  const rebuildReel = async () => {
    if (!project || building) return;
    setBuilding(true);
    setBuildStep("Stitching the montage…");
    try {
      const built = await buildCompilationReel(project.id, apiEndpoint);
      setProject(built.project);
      setProjects((current) => current.map((entry) => (
        entry.id === built.project.id ? built.project : entry
      )));
      setMessage(`Montage rebuilt — ${plural(built.clips, "moment")}, ${runtime(built.project.reel_duration_seconds || 0)}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not rebuild that montage.");
    } finally {
      setBuilding(false);
      setBuildStep("");
    }
  };

  // Cuts run one moment at a time so the creator sees real progress and can
  // stop the batch at any point. Nothing is rendered without this action.
  const cutMoments = async () => {
    if (!project || building) return;
    cutStop.current = false;
    setBuilding(true);
    const total = project.uncut_count;
    let done = 0;
    try {
      for (;;) {
        if (cutStop.current) {
          setMessage(`Stopped after cutting ${done} of ${total} moments. The rest stay in the draft.`);
          break;
        }
        setBuildStep(`Cutting moment ${Math.min(done + 1, total)} of ${total}…`);
        const result = await cutNextCompilationMoment(project.id, apiEndpoint);
        setProject(result.project);
        setProjects((current) => current.map((entry) => (
          entry.id === result.project.id ? result.project : entry
        )));
        if (result.done) {
          setMessage(`All ${plural(total, "moment")} are cut. Approve the sequence to export it.`);
          break;
        }
        done += 1;
      }
      await loadPreparation(project.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not cut that moment.");
    } finally {
      setBuilding(false);
      setBuildStep("");
    }
  };

  const saveItems = async (items: CompilationProjectItem[]) => {
    if (!project || saving) return;
    setSaving(true);
    setMessage(undefined);
    try {
      const updated = await saveCompilationProjectItems(
        project.id,
        items.map((item) => ({ id: item.id, included: item.included, muted: item.muted })),
        apiEndpoint,
      );
      setProject(updated);
      setProjects((current) => current.map((entry) => entry.id === updated.id ? updated : entry));
      setPrepareArmed(false);
      await loadPreparation(updated.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not save that project change.");
    } finally {
      setSaving(false);
    }
  };

  const toggleMuted = (id: string) => {
    if (!project) return;
    void saveItems(project.items.map((item) => (
      item.id === id ? { ...item, muted: !item.muted } : item
    )));
  };

  const toggleItem = (id: string) => {
    if (!project) return;
    void saveItems(project.items.map((item) => (
      item.id === id ? { ...item, included: !item.included } : item
    )));
  };

  /** The posted-montage shape: a track over silent gameplay, reaction kept. */
  const setAudio = async (muted: boolean) => {
    if (!project || saving) return;
    setSaving(true);
    try {
      const updated = await setCompilationAudio(project.id, { muted }, apiEndpoint);
      setProject(updated);
      setProjects((current) => current.map((entry) => (
        entry.id === updated.id ? updated : entry
      )));
      setMessage(muted
        ? "Kills muted for a sound; the last moment keeps its audio. Rebuild to hear it."
        : "Every moment plays its own audio again. Rebuild to hear it.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not change the audio.");
    } finally {
      setSaving(false);
    }
  };

  const rename = async () => {
    if (!project || !titleDraft.trim() || titleDraft.trim() === project.title) return;
    setSaving(true);
    try {
      const updated = await renameCompilationProject(project.id, titleDraft.trim(), apiEndpoint);
      setProject(updated);
      setProjects((current) => current.map((entry) => entry.id === updated.id ? updated : entry));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not rename that project.");
    } finally {
      setSaving(false);
    }
  };

  const approve = async () => {
    if (!project) return;
    setSaving(true);
    try {
      const updated = await approveCompilationProject(project.id, apiEndpoint);
      setProject(updated);
      setProjects((current) => current.map((entry) => entry.id === updated.id ? updated : entry));
      setMessage(updated.available_count >= 2
        ? `Sequence approved. ${updated.available_count} included clips can be prepared locally.`
        : "Sequence approved. Restore source media for at least two included clips before export.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not approve that draft.");
    } finally {
      setSaving(false);
    }
  };

  const removeProject = async () => {
    if (!project) return;
    if (!deleteArmed) {
      setDeleteArmed(true);
      return;
    }
    try {
      await deleteCompilationProject(project.id, apiEndpoint);
      setDeleteArmed(false);
      closeProject();
      setProjects(await listCompilationProjects(apiEndpoint));
      setMessage("Compilation draft deleted. Source clips were untouched.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not delete that draft.");
    }
  };

  /** Saving the montage copies the file Recall already built. */
  const saveReel = async () => {
    if (!project?.reel_url || saving) return;
    const folder = await window.electronAPI?.openFolderDialog?.();
    if (!folder) return;
    setSaving(true);
    try {
      const saved = await saveCompilationReel(project.id, folder, apiEndpoint);
      setMessage(`Saved to ${saved.path}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not save that montage.");
    } finally {
      setSaving(false);
    }
  };

  const exportProject = async () => {
    if (!project || project.status !== "approved") return;
    const items = includedItems(project);
    await onCompile(items.map((item) => item.clip_id!), project.title);
  };

  const refreshPreparation = async () => {
    if (!project || preparationAction) return;
    setPreparationAction(true);
    try {
      const [plan, refreshed] = await Promise.all([
        getCompilationPreparation(project.id, apiEndpoint),
        getCompilationProject(project.id, apiEndpoint),
      ]);
      setPreparation(plan);
      setProject(refreshed);
      setProjects((current) => current.map((entry) => entry.id === refreshed.id ? refreshed : entry));
      setMessage("Source readiness refreshed.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not refresh source readiness.");
    } finally {
      setPreparationAction(false);
    }
  };

  const prepareSources = async () => {
    if (!project || !preparation || preparationAction) return;
    if (!prepareArmed) {
      setPrepareArmed(true);
      return;
    }
    setPreparationAction(true);
    try {
      const sourceKeys = preparation.sources
        .filter((source) => source.restorable)
        .map((source) => source.source_key);
      const plan = await startCompilationPreparation(project.id, sourceKeys, apiEndpoint);
      setPreparation(plan);
      setPrepareArmed(false);
      setMessage("Source preparation started. Recall will restore one Twitch VOD at a time.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not start source preparation.");
    } finally {
      setPreparationAction(false);
    }
  };

  const cancelPreparation = async () => {
    if (!project || preparationAction) return;
    setPreparationAction(true);
    try {
      setPreparation(await cancelCompilationPreparation(project.id, apiEndpoint));
      setMessage("Stopping source preparation. The current download will close safely.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recall could not stop source preparation.");
    } finally {
      setPreparationAction(false);
    }
  };

  const busy = saving || generating || building;
  const hasExportablePair = !!project && project.available_count >= 2;
  const canApprove = !!project && project.included_count >= 2 && !saving;
  const canExport = !!project && project.status === "approved" && hasExportablePair && !compiling;
  const preparationRun = preparation?.preparation;
  const preparationActive = !!preparationRun && ACTIVE_PREPARATION.has(preparationRun.status);
  const currentPreparationSource = preparationRun?.items.find(
    (item) => item.source_key === preparationRun.current_source_key,
  );
  const preparationProgress = preparationRun?.total_sources
    ? Math.round(preparationRun.completed_sources / preparationRun.total_sources * 100)
    : 0;

  // The ribbon draws THE REEL to scale, so only what is in the reel takes width.
  const reelItems = project?.items.filter((item) => item.included) ?? [];
  const reelSeconds = reelItems.reduce((total, item) => total + item.duration, 0);
  const selected = project?.items.find((item) => item.id === selectedItemId) ?? project?.items[0];
  // The player earns its place only when there is something in it.
  const hasStage = !!project?.reel_url || generating || building;

  const sheet = (
    <NewCompilationSheet
      open={sheetOpen}
      busy={generating}
      sessions={sessions}
      matches={matches}
      matchesLoading={matchesLoading}
      recipes={recipes}
      sessionId={matchSessionId}
      onSessionChange={setMatchSessionId}
      initialMatchIndex={project?.match_index ?? undefined}
      onCancel={() => setSheetOpen(false)}
      onCreate={(request) => void createFromSheet(request)}
    />
  );

  /* ---------------- Shelf ---------------- */
  if (!project) {
    return (
      <section className="compilation-workspace" aria-labelledby="compilation-title">
        <div className="compilation-shelf">
          <header className="compilation-shelf-head">
            <div className="compilation-shelf-copy">
              <h1 id="compilation-title">Compilations</h1>
              <p>
                Vertical montages cut from a single match, or from a theme across every
                session you have scanned.
              </p>
            </div>
            <span className="compilation-spacer" />
            <button
              type="button"
              className="cta-accent"
              onClick={() => setSheetOpen(true)}
              disabled={generating}
            >
              <Plus size={15} aria-hidden="true" />
              {generating ? "Cutting…" : "New compilation"}
            </button>
          </header>

          {message && <div className="compilation-message" role="status">{message}</div>}

          {generating && (
            <div className="compilation-working-strip" role="status">
              <Sparkles size={16} aria-hidden="true" />
              <strong>{buildStep || "Working…"}</strong>
              <span>Each moment is a real render. You can stop at any point.</span>
              <span className="compilation-spacer" />
              <button type="button" className="compilation-btn is-sm" onClick={() => { cutStop.current = true; }}>
                <X size={13} aria-hidden="true" />Stop
              </button>
            </div>
          )}

          {loading ? (
            <p className="compilation-note">Opening your compilations…</p>
          ) : !projects.length ? (
            <div className="compilation-empty">
              <svg width="64" height="40" viewBox="0 0 64 40" fill="none" stroke="currentColor"
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M2 30c6 0 8-16 14-16s7 10 12 10 7-18 13-18 8 22 14 22" opacity=".75" />
                <circle cx="28" cy="24" r="2.6" fill="currentColor" stroke="none" />
                <circle cx="45" cy="12" r="2.6" fill="currentColor" stroke="none" />
              </svg>
              <h2>No compilations yet</h2>
              <p>Scan a session, then cut every kill from one game into a vertical montage.</p>
              <button type="button" className="cta-accent" onClick={() => setSheetOpen(true)}>
                <Plus size={15} aria-hidden="true" />New compilation
              </button>
            </div>
          ) : (
            <div className="compilation-shelf-grid">
              {projects.map((entry) => (
                <button
                  type="button"
                  key={entry.id}
                  className="compilation-card"
                  onClick={() => void openProject(entry.id)}
                >
                  <span className={`compilation-card-thumb ${entry.reel_url ? "" : "is-empty"}`}>
                    {entry.reel_url ? (
                      <video
                        // A still of the montage's own second: the card shows
                        // the reel, not a stand-in for it.
                        src={`${apiUrl(entry.reel_url, apiEndpoint)}#t=0.5`}
                        preload="metadata"
                        muted
                        playsInline
                        tabIndex={-1}
                        aria-hidden="true"
                      />
                    ) : (
                      <Film size={26} aria-hidden="true" />
                    )}
                    <span className={`compilation-pill ${entry.reel_url ? "is-ready" : "is-draft"}`}>
                      {entry.reel_url
                        ? (entry.reel_stale ? "Needs a rebuild" : "Built")
                        : "Not built"}
                    </span>
                    {entry.reel_url && (
                      <span className="compilation-card-len t-num">
                        {runtime(entry.reel_duration_seconds || entry.total_duration_seconds)}
                      </span>
                    )}
                  </span>
                  <span className="compilation-card-meta">
                    <strong>{entry.title}</strong>
                    <small className="t-num">
                      {entry.included_count
                        ? `${plural(entry.included_count, "moment")} · ${runtime(entry.total_duration_seconds)}`
                        : "No moments yet"}
                    </small>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
        {sheet}
      </section>
    );
  }

  /* ---------------- Bench ---------------- */
  return (
    <section className="compilation-workspace" aria-labelledby="compilation-title">
      <header className="compilation-bench-top">
        <button type="button" className="compilation-back" onClick={closeProject}>
          <ChevronLeft size={14} aria-hidden="true" />All compilations
        </button>
        <input
          className="compilation-docname"
          value={titleDraft}
          id="compilation-title"
          aria-label="Compilation title"
          onChange={(event) => setTitleDraft(event.currentTarget.value)}
          onKeyDown={(event) => { if (event.key === "Enter") void rename(); }}
          onBlur={() => void rename()}
        />
        <span className={`compilation-pill ${project.reel_stale ? "is-stale" : project.reel_url ? "is-ready" : "is-draft"}`}>
          {project.reel_stale
            ? <><RefreshCw size={11} aria-hidden="true" />Needs a rebuild</>
            : project.reel_url
              ? <><Check size={11} aria-hidden="true" />Built</>
              : <><Clock size={11} aria-hidden="true" />Not built yet</>}
        </span>
        <div className="compilation-topfacts t-num">
          <strong>{runtime(project.reel_duration_seconds || project.total_duration_seconds)}</strong>
          <i aria-hidden="true" />
          <span>{plural(project.included_count, "moment")}</span>
          <i aria-hidden="true" />
          <span>
            {project.match_label || `${projectDate(project.date_from)}–${projectDate(project.date_to)}`}
          </span>
        </div>
        <span className="compilation-spacer" />
        <button
          type="button"
          className="compilation-btn is-sm"
          onClick={() => setSheetOpen(true)}
          disabled={generating}
        >
          <Plus size={13} aria-hidden="true" />New from this match
        </button>
      </header>

      {message && <div className="compilation-message" role="status">{message}</div>}

      {preparation && preparation.required_source_count > 0 && (
        <section
          className={`compilation-readiness ${prepareArmed ? "is-confirming" : ""} ${preparationActive ? "is-active" : ""} ${!preparation.disk_allowed ? "is-blocked" : ""}`}
          aria-label="Compilation source readiness"
          aria-live="polite"
        >
          <span className="compilation-readiness-mark" aria-hidden="true">
            {preparationActive ? <Download size={17} /> : preparation.disk_allowed ? <HardDrive size={17} /> : <AlertTriangle size={17} />}
          </span>
          <div className="compilation-readiness-copy">
            {preparationActive ? (
              <>
                <strong>{preparationRun?.status === "cancelling" ? "Stopping source preparation" : `Restoring source ${Math.min((preparationRun?.completed_sources || 0) + 1, preparationRun?.total_sources || 1)} of ${preparationRun?.total_sources}`}</strong>
                <span>{currentPreparationSource?.label || "Preparing the next Twitch VOD"}. Scans wait until this source queue finishes.</span>
              </>
            ) : prepareArmed ? (
              <>
                <strong>Restore {plural(preparation.restorable_source_count, "Twitch source")}?</strong>
                <span>Recall downloads them one at a time and keeps {storageSize(preparation.reserve_bytes)} free as a safety reserve. Existing clips and review choices stay untouched.</span>
              </>
            ) : (
              <>
                <strong>{preparation.available_count} of {preparation.included_count} included clips are ready</strong>
                <span>
                  {preparation.restorable_source_count > 0
                    ? `${preparation.missing_clip_count} clips need ${plural(preparation.restorable_source_count, "Twitch source")}.`
                    : "The remaining clips need their original recording and cannot be restored automatically."}
                  {preparation.manual_source_count > 0 ? ` ${preparation.manual_source_count} local ${preparation.manual_source_count === 1 ? "recording must" : "recordings must"} be located manually.` : ""}
                </span>
              </>
            )}
          </div>
          {preparationActive ? (
            <div className="compilation-readiness-progress">
              <span><i style={{ transform: `scaleX(${preparationProgress / 100})` }} /></span>
              <b className="t-num">{preparationProgress}%</b>
            </div>
          ) : preparation.restorable_source_count > 0 ? (
            <div className="compilation-readiness-space">
              <span>Estimated download</span>
              <strong className="t-num">{storageSize(preparation.estimated_download_bytes)}</strong>
              <small className="t-num">{storageSize(preparation.free_bytes)} free</small>
            </div>
          ) : null}
          <div className="compilation-readiness-actions">
            {preparationActive ? (
              <button type="button" onClick={() => void cancelPreparation()} disabled={preparationAction || preparationRun?.status === "cancelling"}>
                <X size={13} />{preparationRun?.status === "cancelling" ? "Stopping…" : "Cancel"}
              </button>
            ) : prepareArmed ? (
              <>
                <button type="button" onClick={() => setPrepareArmed(false)} disabled={preparationAction}>Not now</button>
                <button type="button" className="is-primary" onClick={() => void prepareSources()} disabled={preparationAction || !preparation.can_start}>
                  <Download size={13} />{preparationAction ? "Starting…" : "Restore sources"}
                </button>
              </>
            ) : preparation.restorable_source_count > 0 ? (
              <button type="button" className="is-primary" onClick={() => void prepareSources()} disabled={preparationAction || !preparation.can_start}>
                <Download size={13} />{preparation.active_scan ? "Waiting for scan" : preparation.disk_allowed ? "Prepare sources" : "Not enough space"}
              </button>
            ) : (
              <button type="button" onClick={() => void refreshPreparation()} disabled={preparationAction}>
                <RefreshCw size={13} />{preparationAction ? "Checking…" : "Check again"}
              </button>
            )}
          </div>
          {!preparationActive && !preparation.disk_allowed && (
            <p className="compilation-readiness-note">
              Free {storageSize(preparation.estimated_download_bytes + preparation.reserve_bytes - preparation.free_bytes)} or restore fewer sources. Settings → Storage lists safe downloads you can remove.
            </p>
          )}
          {!preparationActive && preparationRun?.status === "interrupted" && (
            <p className="compilation-readiness-note">Preparation paused when Recall closed. Finished downloads remain safe; prepare again to continue with what is still missing.</p>
          )}
          {!preparationActive && preparationRun?.status === "partial" && (
            <p className="compilation-readiness-note">Some sources could not be restored. Available clips remain usable; prepare again to retry sources that are still available.</p>
          )}
        </section>
      )}

      <div className={`compilation-bench ${hasStage ? "" : "is-list-only"}`}>
        {hasStage && (
          <div className="compilation-stage">
            <div className="compilation-screen">
              {generating || building ? (
                <div className="compilation-working" role="status">
                  <Sparkles size={26} aria-hidden="true" />
                  <strong>{buildStep || "Working…"}</strong>
                  <span>Each moment is a real render. You can stop at any point.</span>
                  <button type="button" onClick={() => { cutStop.current = true; }}>
                    <X size={13} aria-hidden="true" />Stop
                  </button>
                </div>
              ) : project.reel_url ? (
                <video
                  key={`${project.id}:${project.reel_built_at ?? ""}`}
                  // #t=0.001 makes the browser decode and show the first frame,
                  // so a freshly built montage looks like a video rather than a
                  // black box until someone presses play.
                  src={`${apiUrl(project.reel_url, apiEndpoint)}#t=0.001`}
                  controls
                  playsInline
                  preload="metadata"
                />
              ) : null}
            </div>

            {reelItems.length > 1 && reelSeconds > 0 && (
              <div className="compilation-ribbon-wrap">
                <div className="compilation-ribbon" role="list" aria-label="The reel, to scale">
                  {reelItems.map((item) => (
                    <button
                      type="button"
                      role="listitem"
                      key={item.id}
                      style={{ flexGrow: item.duration }}
                      className={`${item.moment_kind === "win" ? "is-win" : ""} ${item.id === selected?.id ? "is-selected" : ""}`}
                      title={`${item.title} · ${runtime(item.duration)}`}
                      aria-label={`${item.title}, ${runtime(item.duration)}`}
                      onClick={() => setSelectedItemId(item.id)}
                    />
                  ))}
                </div>
                <div className="compilation-ribbon-legend">
                  <span className="compilation-ribbon-keys">
                    <em><i className="is-kill" aria-hidden="true" />{plural(reelItems.filter((i) => i.moment_kind !== "win").length, "moment")}</em>
                    {reelItems.some((item) => item.moment_kind === "win") && (
                      <em><i className="is-win" aria-hidden="true" />finale</em>
                    )}
                  </span>
                  <span>the reel, to scale</span>
                </div>
              </div>
            )}

            <div className="compilation-stagefacts t-num">
              <span><strong>{runtime(project.reel_duration_seconds || project.total_duration_seconds)}</strong> runtime</span>
              <span><strong>{project.included_count}</strong> {noun(project.included_count, "moment")}</span>
              {project.reel_url && <span><strong>1080×1920</strong></span>}
            </div>
          </div>
        )}

        <div className="compilation-board">
          <div className="compilation-board-head">
            <h2>Moments in the reel</h2>
            <span className="compilation-count t-num">{project.items.length}</span>
            <span className="compilation-spacer" />
            <div className="compilation-seg" role="group" aria-label="Audio in the montage">
              <button
                type="button"
                className={project.muted_count > 0 ? "is-on" : ""}
                aria-pressed={project.muted_count > 0}
                onClick={() => void setAudio(true)}
                disabled={busy}
              >
                <VolumeX size={13} aria-hidden="true" />Muted, except the finale
              </button>
              <button
                type="button"
                className={project.muted_count === 0 ? "is-on" : ""}
                aria-pressed={project.muted_count === 0}
                onClick={() => void setAudio(false)}
                disabled={busy}
              >
                <Volume2 size={13} aria-hidden="true" />Keep all audio
              </button>
            </div>
          </div>

          {project.reel_stale && (
            <div className="compilation-stale" role="status">
              <AlertTriangle size={14} aria-hidden="true" />
              <span>
                The video is the previous version. Rebuild to make it match what you see here.
              </span>
              <button type="button" onClick={() => void rebuildReel()} disabled={building || generating}>
                <RefreshCw size={13} aria-hidden="true" />Rebuild
              </button>
            </div>
          )}

          {!project.items.length ? (
            /* A draft can outlive its cuts: removing the clips they were made
               from empties the sequence while the last rendered reel stays on
               disk. Say which one is which rather than showing a blank grid. */
            <div className="compilation-board-empty">
              <Film size={22} aria-hidden="true" />
              <strong>No moments in this draft</strong>
              <p>
                Recall has no moments recorded for it any more.
                {project.reel_url ? " The video is the last one it built." : ""}
                {" "}Cut it again from this match, or delete the draft.
              </p>
            </div>
          ) : (
          <div className="compilation-tiles" role="listbox" aria-label="Moments in this montage">
            {project.items.map((item, index) => (
              <button
                type="button"
                role="option"
                key={item.id}
                aria-selected={item.id === selected?.id}
                className={`compilation-tile ${item.included ? "" : "is-out"} ${item.moment_kind === "win" ? "is-win" : ""} ${item.id === selected?.id ? "is-selected" : ""}`}
                onClick={() => setSelectedItemId(item.id)}
              >
                {item.clip_id
                  ? <img src={apiUrl(`/clips/${encodeURIComponent(item.clip_id)}/thumb`, apiEndpoint)} alt="" />
                  : <span className="compilation-tile-uncut" aria-hidden="true"><Film size={16} /></span>}
                <span className="compilation-tile-ord t-num">{String(index + 1).padStart(2, "0")}</span>
                {item.muted && item.included && (
                  <span className="compilation-tile-mark" aria-hidden="true"><VolumeX size={10} /></span>
                )}
                <span className="compilation-tile-len t-num">{runtime(item.duration)}</span>
                <span className="sr-only">
                  {item.title}
                  {item.included ? "" : ", dropped"}
                  {item.muted ? ", muted" : ""}
                  {item.media_available ? "" : ", source unavailable"}
                </span>
              </button>
            ))}
          </div>
          )}

          {selected && (
            <div className="compilation-inspector" role="group" aria-label="Selected moment">
              <span className="compilation-inspector-poster">
                {selected.clip_id
                  ? <img src={apiUrl(`/clips/${encodeURIComponent(selected.clip_id)}/thumb`, apiEndpoint)} alt="" />
                  : <Film size={16} aria-hidden="true" />}
              </span>
              <div className="compilation-inspector-copy">
                <div className="compilation-inspector-row">
                  <strong>{selected.title}</strong>
                  {selected.moment_kind === "win" && (
                    <span className="compilation-pill is-ready">
                      <Trophy size={11} aria-hidden="true" />Finale
                    </span>
                  )}
                  {!selected.media_available && (
                    <span className="compilation-pill is-draft">
                      {selected.media_state === "clip_removed"
                        ? "Clip removed"
                        : selected.media_state === "not_cut_yet"
                          ? "Not cut yet"
                          : "Source offline"}
                    </span>
                  )}
                </div>
                <div className="compilation-inspector-where t-num">
                  {project.match_label
                    ? `In the VOD ${fmtClock(selected.start_time)}`
                    : `${selected.session_name} · ${projectDate(selected.source_date)} · ${fmtClock(selected.start_time)}`}
                  {" · "}{runtime(selected.duration)}
                  {" · "}{selected.included ? (selected.muted ? "muted" : "audio kept") : "not in the video"}
                </div>
              </div>
              <div className="compilation-inspector-actions">
                {selected.clip_id && selected.job_id && (
                  <button
                    type="button"
                    className="compilation-btn is-sm"
                    onClick={() => void onOpenClip(selected.job_id!, selected.clip_id!)}
                  >
                    <Play size={13} aria-hidden="true" />Review
                  </button>
                )}
                <button
                  type="button"
                  className="compilation-btn is-sm"
                  onClick={() => toggleMuted(selected.id)}
                  disabled={busy || !selected.included}
                  aria-pressed={selected.muted}
                >
                  {selected.muted
                    ? <><VolumeX size={13} aria-hidden="true" />Muted</>
                    : <><Volume2 size={13} aria-hidden="true" />Audio on</>}
                </button>
                <button
                  type="button"
                  className={`compilation-btn is-sm ${selected.included ? "is-danger" : ""}`}
                  onClick={() => toggleItem(selected.id)}
                  disabled={busy}
                  aria-pressed={!selected.included}
                >
                  {selected.included
                    ? <><X size={13} aria-hidden="true" />Drop from reel</>
                    : <><Plus size={13} aria-hidden="true" />Bring back</>}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <footer className="compilation-dock">
        <div className="compilation-dock-copy">
          <strong>{project.reel_stale
            ? "Rebuild the video to see your changes"
            : project.reel_url
              ? "Montage ready"
              : project.uncut_count > 0
                ? `${plural(project.uncut_count, "moment")} still to cut`
                : project.status === "approved"
                  ? (hasExportablePair ? "Approved and ready to export" : "Sequence approved — source needed")
                  : "Review the generated sequence"}</strong>
          <small>
            {project.reel_stale
              ? "Stitching again is quick — the moments are already rendered."
              : project.reel_url
                ? `${runtime(project.reel_duration_seconds || 0)} · ${plural(project.included_count, "moment")} · saved to your compilations`
                : project.uncut_count > 0
                  ? "Recall found these moments in the match but has not rendered them yet."
                  : project.status === "approved"
                    ? (hasExportablePair
                      ? `${project.available_count} of ${project.included_count} included clips can be prepared locally.${project.available_count < project.included_count ? " Restore source media to recover the rest." : ""}`
                      : `Restore source media for at least ${2 - project.available_count} more included ${noun(2 - project.available_count, "clip")} to export.`)
                    : "Reordering or changing the included set always returns this project to draft status."}
          </small>
        </div>
        <span className="compilation-spacer" />
        <button
          type="button"
          className={`compilation-btn is-danger ${deleteArmed ? "is-armed" : ""}`}
          onClick={() => void removeProject()}
        >
          {deleteArmed ? <X size={13} aria-hidden="true" /> : <Trash2 size={13} aria-hidden="true" />}
          {deleteArmed ? "Confirm delete" : "Delete draft"}
        </button>
        {project.reel_stale ? (
          <button type="button" className="cta-accent" onClick={() => void rebuildReel()} disabled={building || generating}>
            <RefreshCw size={14} aria-hidden="true" />{building ? "Rebuilding…" : "Rebuild video"}
          </button>
        ) : project.reel_url ? (
          <button type="button" className="cta-accent" onClick={() => void saveReel()} disabled={busy}>
            <Download size={14} aria-hidden="true" />{saving ? "Saving…" : "Save video…"}
          </button>
        ) : project.uncut_count > 0 ? (
          <button type="button" className="cta-accent" onClick={() => void cutMoments()} disabled={building || generating}>
            <Film size={14} aria-hidden="true" />Cut {plural(project.uncut_count, "moment")}
          </button>
        ) : project.status === "approved" ? (
          <button type="button" className="cta-accent" onClick={() => void exportProject()} disabled={!canExport}>
            <Film size={14} aria-hidden="true" />{compiling ? "Preparing reel…" : (hasExportablePair ? "Export approved reel" : "Restore source to export")}
          </button>
        ) : (
          <button type="button" className="cta-accent" onClick={() => void approve()} disabled={!canApprove}>
            <Check size={14} aria-hidden="true" />Approve sequence
          </button>
        )}
      </footer>
      {sheet}
    </section>
  );
}

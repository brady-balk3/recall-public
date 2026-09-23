# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable

Migration = tuple[str, Callable[[sqlite3.Connection], None]]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if column not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _applied_versions(conn: sqlite3.Connection) -> set[str]:
    _ensure_migrations_table(conn)
    return {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}


def _migration_001_initial(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT,
            session_name TEXT,
            source_type TEXT,
            source_path TEXT,
            duration REAL,
            progress REAL DEFAULT 0.0,
            current_stage TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            start_time REAL,
            end_time REAL,
            score REAL,
            title TEXT,
            export_path TEXT,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS exports (
            id TEXT PRIMARY KEY,
            clip_id TEXT,
            platform TEXT,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(clip_id) REFERENCES clips(id)
        )
        """
    )
    _add_column(conn, "jobs", "session_name", "TEXT")
    _add_column(conn, "jobs", "created_at", "TIMESTAMP")
    _add_column(conn, "jobs", "source_type", "TEXT")
    _add_column(conn, "jobs", "source_path", "TEXT")
    _add_column(conn, "jobs", "duration", "REAL")
    _add_column(conn, "jobs", "progress", "REAL DEFAULT 0.0")
    _add_column(conn, "jobs", "current_stage", "TEXT")


def _migration_002_job_telemetry(conn: sqlite3.Connection) -> None:
    for col, ddl in (
        ("message", "TEXT"),
        ("started_at", "TIMESTAMP"),
        ("updated_at", "TIMESTAMP"),
        ("stage_progress", "REAL"),
        ("scanned_seconds", "REAL"),
        ("clips_found", "INTEGER"),
        ("stage_timings", "TEXT"),
        ("scan_profile", "TEXT"),
    ):
        _add_column(conn, "jobs", col, ddl)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            phase TEXT,
            message TEXT,
            progress REAL,
            payload TEXT,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, id)")


def _migration_003_clip_evidence(conn: sqlite3.Connection) -> None:
    for col, ddl in (
        ("description", "TEXT"),
        ("signals", "TEXT"),
        ("story_label", "TEXT"),
        ("features", "TEXT"),
        ("reason", "TEXT"),
        ("peak_timestamp", "REAL"),
        ("modality_breakdown", "TEXT"),
        ("reaction_auc", "REAL"),
    ):
        _add_column(conn, "clips", col, ddl)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_job ON clips(job_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exports_clip ON exports(clip_id)")


def _migration_004_reaction_learning(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reaction_timelines (
            job_id TEXT PRIMARY KEY,
            timeline TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clip_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id TEXT,
            features TEXT,
            label INTEGER,
            event TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE TABLE IF NOT EXISTS ranker_metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")


def _migration_005_clip_thumbnails(conn: sqlite3.Connection) -> None:
    # Poster JPEG per clip (grabbed at the reaction peak) for grid cards and
    # filmstrips; NULL for clips rendered before this migration (backfilled
    # lazily by GET /clips/{id}/thumb).
    _add_column(conn, "clips", "thumb_path", "TEXT")


def _migration_006_studio_columns(conn: sqlite3.Connection) -> None:
    # Columns that previously lived only in DatabaseManager._init_db's inline
    # DDL. Folded in here so migrations are the single source of truth for the
    # schema. Idempotent: existing installs already have these (added by the old
    # inline block), so _add_column no-ops and only records the version.
    _add_column(conn, "jobs", "asset_path", "TEXT")
    for col, ddl in (
        ("scene", "TEXT"),           # non-gameplay scene code (lobby/intermission)
        ("hook_line", "TEXT"),       # semantic judge subtitle for review
        ("moment_type", "TEXT"),     # funny/clutch/fail/rage/story/wholesome/filler
        ("deck_score", "REAL"),      # selection-score percentile within the review deck
        ("kept", "INTEGER"),         # 1 = creator kept this clip
        ("passed", "INTEGER"),       # 1 = reviewed and passed on it
        ("exported_at", "TIMESTAMP"),
        ("tags", "TEXT"),            # JSON posting hashtags
        # 1 = export_path points at a cheap low-res review proxy, not the final
        # export; ensure_rendered() re-renders in place on first real export.
        ("preview_only", "INTEGER"),
    ):
        _add_column(conn, "clips", col, ddl)
    # Presentation rank at label time (position bias for the learned ranker).
    _add_column(conn, "clip_labels", "rank", "INTEGER")


def _migration_007_learning_context(conn: sqlite3.Connection) -> None:
    """Retain the session and confidence of every feedback decision.

    Pairwise ranking must compare clips from the same review deck.  Older
    installs only stored the clip id, which loses that context once a clip is
    deleted.  Weight distinguishes explicit creator choices from future weak
    supervision without pretending they have equal authority.
    """
    _add_column(conn, "clip_labels", "job_id", "TEXT")
    _add_column(conn, "clip_labels", "weight", "REAL DEFAULT 1.0")
    conn.execute(
        """UPDATE clip_labels
           SET job_id = (SELECT clips.job_id FROM clips WHERE clips.id = clip_labels.clip_id)
           WHERE job_id IS NULL"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_clip_labels_job ON clip_labels(job_id, id)"
    )


def _migration_008_clip_diagnostics(conn: sqlite3.Connection) -> None:
    """Persist the evidence needed to audit bad framing and model decisions."""
    _add_column(conn, "clips", "layout", "TEXT")
    _add_column(conn, "clips", "semantic_verdict", "TEXT")


def _migration_009_boundary_edits(conn: sqlite3.Connection) -> None:
    """Capture creator re-cuts of a clip window (HUMAN_CLIPS Package 1).

    A trim says "the window was wrong", not "keep/reject this moment", so it
    lives in its own table and is deliberately NEVER mirrored into clip_labels:
    feeding boundary corrections to the pairwise ranker would corrupt
    preference training. The founder-fixture exporter reads this table to turn
    edited windows into truth-v2 "boundary_fix" decisions.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clip_boundary_edits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id TEXT,
            job_id TEXT,
            old_start REAL,
            old_end REAL,
            new_start REAL,
            new_end REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_boundary_edits_job ON clip_boundary_edits(job_id, id)"
    )


def _migration_010_job_framing(conn: sqlite3.Connection) -> None:
    """Persist the scan's compact framing model for later manual cuts.

    The automatic clip path resolves facecam and authored-gameplay geometry
    while the full perception result is in memory.  The VOD Editor may be
    opened days later, so keep the small, JSON-encoded layout timeline with
    the job instead of guessing a manual clip's crop from a generic default.
    """
    _add_column(conn, "jobs", "framing", "TEXT")


def _migration_011_job_poster(conn: sqlite3.Connection) -> None:
    """Landscape still from the source VOD for session gallery cards.

    Clip posters are 9:16 composed exports; session cards want a real VOD
    frame. Generated lazily by GET /jobs/{id}/thumb when missing.
    """
    _add_column(conn, "jobs", "poster_path", "TEXT")


def _migration_012_maybe_verdict(conn: sqlite3.Connection) -> None:
    """Persist uncertainty without collapsing it to a Pass label."""
    _add_column(conn, "clips", "maybe", "INTEGER")
    _add_column(conn, "clip_labels", "label_value", "REAL")
    conn.execute(
        "UPDATE clip_labels SET label_value = CAST(label AS REAL) "
        "WHERE label_value IS NULL AND label IS NOT NULL"
    )


def _migration_013_storage_lifecycle(conn: sqlite3.Connection) -> None:
    """Track shared Recall-managed Twitch sources independently from sessions."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_assets (
            source_key TEXT PRIMARY KEY,
            source_type TEXT NOT NULL DEFAULT 'twitch',
            twitch_url TEXT,
            vod_id TEXT,
            local_path TEXT,
            file_size_bytes INTEGER NOT NULL DEFAULT 0,
            downloaded_at TIMESTAMP,
            last_used_at TIMESTAMP,
            retention_expires_at TIMESTAMP,
            pinned INTEGER NOT NULL DEFAULT 0,
            lifecycle_state TEXT NOT NULL DEFAULT 'source_removed',
            scan_completed INTEGER NOT NULL DEFAULT 0,
            review_proxies_verified INTEGER NOT NULL DEFAULT 0,
            restore_available INTEGER NOT NULL DEFAULT 1,
            last_error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_asset_sessions (
            source_key TEXT NOT NULL,
            job_id TEXT NOT NULL UNIQUE,
            linked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_key, job_id),
            FOREIGN KEY(source_key) REFERENCES source_assets(source_key) ON DELETE CASCADE,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_assets_expiry "
        "ON source_assets(retention_expires_at, pinned, lifecycle_state)"
    )

    # The old hidden cache cap is the closest predecessor to Recall's unified
    # storage target. Preserve it when present, then retire every conflicting
    # key so only one policy can be active after this migration.
    row = conn.execute("SELECT value FROM settings WHERE key = 'cacheCapGb'").fetchone()
    if row is None:
        row = conn.execute("SELECT value FROM settings WHERE key = 'assetRetentionGb'").fetchone()
    previous_cap = row["value"] if row is not None else "25"
    conn.execute(
        "INSERT OR IGNORE INTO settings(key, value) VALUES ('recallStorageCapGb', ?)",
        (previous_cap,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO settings(key, value) VALUES ('sourceRetentionDays', '7')"
    )
    conn.execute(
        "DELETE FROM settings WHERE key IN ('cacheCapGb', 'keepDownloadedVods', 'assetRetentionGb')"
    )

    # Backfill canonical Twitch identities without touching local creator files.
    from core.source_identity import canonical_source_key

    rows = conn.execute(
        "SELECT id, source_path, asset_path, status, created_at, updated_at FROM jobs"
    ).fetchall()
    for job in rows:
        source_key = canonical_source_key(job["source_path"] or "", job["asset_path"] or "")
        if not source_key.startswith("twitch:"):
            continue
        vod_id = source_key.split(":", 1)[1]
        completed = 1 if job["status"] == "completed" else 0
        last_used = job["updated_at"] or job["created_at"]
        conn.execute(
            """
            INSERT INTO source_assets(
                source_key, twitch_url, vod_id, local_path, last_used_at,
                lifecycle_state, scan_completed
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                twitch_url = COALESCE(source_assets.twitch_url, excluded.twitch_url),
                local_path = COALESCE(source_assets.local_path, excluded.local_path),
                last_used_at = CASE
                    WHEN source_assets.last_used_at IS NULL THEN excluded.last_used_at
                    WHEN excluded.last_used_at IS NULL THEN source_assets.last_used_at
                    ELSE MAX(source_assets.last_used_at, excluded.last_used_at)
                END,
                scan_completed = MAX(source_assets.scan_completed, excluded.scan_completed),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                source_key,
                job["source_path"],
                vod_id,
                job["asset_path"],
                last_used,
                "available" if job["asset_path"] else "source_removed",
                completed,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO source_asset_sessions(source_key, job_id) VALUES (?, ?)",
            (source_key, job["id"]),
        )


def _migration_014_explicit_clip_scores(conn: sqlite3.Connection) -> None:
    """Separate selection authority from creator-facing opening evidence.

    ``clips.score`` historically became the 0..1 hook/opening score while
    ``deck_score`` was derived from the selector's unbounded score.  Keeping
    both meanings behind the generic name made a stored deck impossible to
    audit.  Preserve ``score`` for older clients, but give every new write an
    explicit field for each quantity.

    Existing modern rows can safely inherit their hook score because their
    non-null deck score proves they were written by the split presentation
    path.  The raw selection score cannot be reconstructed from a min/max
    percentile, so it intentionally remains NULL for historical rows; their
    immutable candidates.v1.json artifact remains the source of truth.
    """
    _add_column(conn, "clips", "selection_score", "REAL")
    _add_column(conn, "clips", "hook_score", "REAL")
    conn.execute(
        """UPDATE clips
           SET hook_score = score
           WHERE hook_score IS NULL
             AND deck_score IS NOT NULL
             AND COALESCE(story_label, '') != 'manual'"""
    )


def _migration_015_job_source_date(conn: sqlite3.Connection) -> None:
    """Store the recording/VOD date separately from the scan creation time.

    Historical jobs intentionally remain NULL. Backfilling from ``created_at``
    would give exports the processing date, which is precisely the ambiguity
    this column removes.
    """
    _add_column(conn, "jobs", "source_date", "TEXT")


def _migration_016_stable_clip_numbers(conn: sqlite3.Connection) -> None:
    """Give every clip an append-only per-VOD creator-facing number."""
    _add_column(conn, "clips", "clip_number", "INTEGER")
    jobs = conn.execute(
        "SELECT DISTINCT job_id FROM clips WHERE job_id IS NOT NULL ORDER BY job_id"
    ).fetchall()
    for job in jobs:
        rows = conn.execute(
            """SELECT id FROM clips
               WHERE job_id = ?
               ORDER BY COALESCE(start_time, 0), COALESCE(end_time, 0), id""",
            (job[0],),
        ).fetchall()
        for number, row in enumerate(rows, start=1):
            conn.execute(
                "UPDATE clips SET clip_number = ? WHERE id = ?",
                (number, row[0]),
            )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_clips_job_clip_number
           ON clips(job_id, clip_number)
           WHERE job_id IS NOT NULL AND clip_number IS NOT NULL"""
    )


def _migration_017_recall_sessions(conn: sqlite3.Connection) -> None:
    """Crash-safe live evidence, independent from operational job telemetry."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recall_sessions (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            source_platform TEXT NOT NULL,
            source_ref TEXT,
            title TEXT,
            started_at_utc TEXT NOT NULL,
            ended_at_utc TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            schema_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_recall_sessions_one_active
           ON recall_sessions(status) WHERE status = 'active'"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_recall_sessions_started
           ON recall_sessions(started_at_utc DESC)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recall_session_events (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            source TEXT NOT NULL,
            occurred_at_utc TEXT NOT NULL,
            stream_offset_seconds REAL,
            confidence REAL NOT NULL DEFAULT 1.0,
            payload TEXT NOT NULL DEFAULT '{}',
            schema_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES recall_sessions(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_recall_session_events_time
           ON recall_session_events(session_id, occurred_at_utc, created_at)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recall_session_clock_anchors (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            anchor_type TEXT NOT NULL,
            wall_time_utc TEXT NOT NULL,
            stream_time_seconds REAL NOT NULL,
            uncertainty_seconds REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES recall_sessions(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_recall_session_anchors_time
           ON recall_session_clock_anchors(session_id, wall_time_utc, created_at)"""
    )
    _add_column(conn, "jobs", "recall_session_id", "TEXT")
    _add_column(conn, "jobs", "region_plan", "TEXT")
    _add_column(conn, "clips", "recall_provenance", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_recall_session ON jobs(recall_session_id)"
    )


def _migration_018_stream_memory(conn: sqlite3.Connection) -> None:
    """Rebuildable local search index over transcripts and clip metadata."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_entries (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            clip_id TEXT,
            kind TEXT NOT NULL,
            entry_index INTEGER NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL DEFAULT '',
            session_name TEXT NOT NULL DEFAULT '',
            game TEXT NOT NULL DEFAULT '',
            source_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(job_id, kind, entry_index),
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(clip_id) REFERENCES clips(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_job_time
           ON stream_memory_entries(job_id, start_time)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_kind_date
           ON stream_memory_entries(kind, source_date)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_index_state (
            job_id TEXT PRIMARY KEY,
            transcript_status TEXT NOT NULL,
            transcript_coverage TEXT NOT NULL DEFAULT 'unknown',
            transcript_entries INTEGER NOT NULL DEFAULT 0,
            clip_entries INTEGER NOT NULL DEFAULT 0,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS stream_memory_fts USING fts5(
            title,
            text,
            session_name,
            game,
            content='stream_memory_entries',
            content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )
    conn.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS stream_memory_entries_ai AFTER INSERT ON stream_memory_entries BEGIN
          INSERT INTO stream_memory_fts(rowid, title, text, session_name, game)
          VALUES (new.rowid, new.title, new.text, new.session_name, new.game);
        END;
        CREATE TRIGGER IF NOT EXISTS stream_memory_entries_ad AFTER DELETE ON stream_memory_entries BEGIN
          INSERT INTO stream_memory_fts(stream_memory_fts, rowid, title, text, session_name, game)
          VALUES ('delete', old.rowid, old.title, old.text, old.session_name, old.game);
        END;
        CREATE TRIGGER IF NOT EXISTS stream_memory_entries_au AFTER UPDATE ON stream_memory_entries BEGIN
          INSERT INTO stream_memory_fts(stream_memory_fts, rowid, title, text, session_name, game)
          VALUES ('delete', old.rowid, old.title, old.text, old.session_name, old.game);
          INSERT INTO stream_memory_fts(rowid, title, text, session_name, game)
          VALUES (new.rowid, new.title, new.text, new.session_name, new.game);
        END;
        """
    )


def _migration_019_stream_memory_semantic_state(conn: sqlite3.Connection) -> None:
    """Version and invalidate the rebuildable Stream Memory concept index."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_semantic_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            source_revision INTEGER NOT NULL DEFAULT 0,
            indexed_revision INTEGER NOT NULL DEFAULT -1,
            status TEXT NOT NULL DEFAULT 'not_built',
            model_id TEXT,
            dimensions INTEGER NOT NULL DEFAULT 0,
            indexed_entries INTEGER NOT NULL DEFAULT 0,
            index_bytes INTEGER NOT NULL DEFAULT 0,
            indexed_at TIMESTAMP,
            last_error TEXT
        )
        """
    )
    conn.execute(
        """INSERT OR IGNORE INTO stream_memory_semantic_state
           (id, source_revision, indexed_revision, status)
           VALUES (
             1,
             CASE WHEN EXISTS (SELECT 1 FROM stream_memory_entries LIMIT 1) THEN 1 ELSE 0 END,
             -1,
             'not_built'
           )"""
    )


def _migration_020_compilation_projects(conn: sqlite3.Connection) -> None:
    """Persist editable cross-session compilation drafts and their provenance."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compilation_projects (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            template TEXT NOT NULL DEFAULT 'best_of_range',
            query TEXT NOT NULL DEFAULT '',
            game TEXT,
            date_from TEXT,
            date_to TEXT,
            target_duration_seconds REAL,
            max_clips INTEGER NOT NULL DEFAULT 12,
            max_per_session INTEGER NOT NULL DEFAULT 3,
            status TEXT NOT NULL DEFAULT 'draft',
            generation_version INTEGER NOT NULL DEFAULT 1,
            selection_summary TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            approved_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compilation_project_items (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            clip_id TEXT,
            source_job_id TEXT,
            position INTEGER NOT NULL,
            included INTEGER NOT NULL DEFAULT 1,
            selection_score REAL,
            selection_reason TEXT NOT NULL DEFAULT '',
            title_snapshot TEXT NOT NULL DEFAULT '',
            session_name_snapshot TEXT NOT NULL DEFAULT '',
            source_date_snapshot TEXT,
            start_time_snapshot REAL NOT NULL DEFAULT 0,
            end_time_snapshot REAL NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES compilation_projects(id) ON DELETE CASCADE,
            FOREIGN KEY(clip_id) REFERENCES clips(id) ON DELETE SET NULL,
            FOREIGN KEY(source_job_id) REFERENCES jobs(id) ON DELETE SET NULL,
            UNIQUE(project_id, clip_id)
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_compilation_projects_updated
           ON compilation_projects(updated_at DESC)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_compilation_project_items_order
           ON compilation_project_items(project_id, position)"""
    )


def _migration_021_compilation_source_groups(conn: sqlite3.Connection) -> None:
    """Keep rescans of one underlying VOD in the same diversity group."""
    _add_column(conn, "compilation_project_items", "source_group_snapshot", "TEXT")


def _migration_022_compilation_preparations(conn: sqlite3.Connection) -> None:
    """Persist project-level source restoration queues and their outcomes."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compilation_preparations (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            total_sources INTEGER NOT NULL DEFAULT 0,
            prepared_sources INTEGER NOT NULL DEFAULT 0,
            failed_sources INTEGER NOT NULL DEFAULT 0,
            estimated_bytes INTEGER NOT NULL DEFAULT 0,
            current_source_key TEXT,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES compilation_projects(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compilation_preparation_items (
            id TEXT PRIMARY KEY,
            preparation_id TEXT NOT NULL,
            source_key TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT '',
            position INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            estimated_bytes INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(preparation_id) REFERENCES compilation_preparations(id) ON DELETE CASCADE,
            UNIQUE(preparation_id, source_key)
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_compilation_preparations_project
           ON compilation_preparations(project_id, created_at DESC)"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_compilation_preparations_active
           ON compilation_preparations(project_id)
           WHERE status IN ('queued', 'running', 'cancelling')"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_compilation_preparation_items_order
           ON compilation_preparation_items(preparation_id, position)"""
    )


def _migration_023_stream_memory_creator_learning(conn: sqlite3.Connection) -> None:
    """Persist creator vocabulary and query-specific relevance corrections."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_aliases (
            id TEXT PRIMARY KEY,
            canonical_term TEXT NOT NULL COLLATE NOCASE,
            alias TEXT NOT NULL COLLATE NOCASE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(alias)
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_aliases_canonical
           ON stream_memory_aliases(canonical_term, alias)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_feedback (
            id TEXT PRIMARY KEY,
            normalized_query TEXT NOT NULL,
            job_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            entry_index INTEGER NOT NULL,
            verdict TEXT NOT NULL CHECK (verdict IN ('relevant', 'not_relevant')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(normalized_query, job_id, kind, entry_index),
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        )
        """
    )


def _migration_024_stream_memory_stable_feedback(conn: sqlite3.Connection) -> None:
    """Move early entry-id feedback onto stable session/timeline identity."""
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(stream_memory_feedback)")
    }
    if "entry_id" in columns:
        conn.execute("ALTER TABLE stream_memory_feedback RENAME TO stream_memory_feedback_legacy")
        conn.execute(
            """
            CREATE TABLE stream_memory_feedback (
                id TEXT PRIMARY KEY,
                normalized_query TEXT NOT NULL,
                job_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                entry_index INTEGER NOT NULL,
                verdict TEXT NOT NULL CHECK (verdict IN ('relevant', 'not_relevant')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(normalized_query, job_id, kind, entry_index),
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """INSERT OR IGNORE INTO stream_memory_feedback
               (id, normalized_query, job_id, kind, entry_index, verdict,
                created_at, updated_at)
               SELECT f.id, f.normalized_query, m.job_id, m.kind, m.entry_index,
                      f.verdict, f.created_at, f.updated_at
               FROM stream_memory_feedback_legacy AS f
               JOIN stream_memory_entries AS m ON m.id = f.entry_id"""
        )
        conn.execute("DROP TABLE stream_memory_feedback_legacy")
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_feedback_query
           ON stream_memory_feedback(normalized_query, verdict, updated_at DESC)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_feedback_query
           ON stream_memory_feedback(normalized_query, verdict, updated_at DESC)"""
    )


def _migration_025_stream_memory_evaluation(conn: sqlite3.Connection) -> None:
    """Persist creator-authored search truth and comparable local eval runs."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_evaluation_cases (
            id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            normalized_query TEXT NOT NULL UNIQUE,
            expected_description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL CHECK (status IN ('labeled', 'missed')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_evaluation_targets (
            id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            entry_index INTEGER NOT NULL,
            verdict TEXT NOT NULL CHECK (verdict IN ('relevant', 'not_relevant')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(case_id, job_id, kind, entry_index),
            FOREIGN KEY(case_id) REFERENCES stream_memory_evaluation_cases(id) ON DELETE CASCADE,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_evaluation_targets_case
           ON stream_memory_evaluation_targets(case_id, verdict, updated_at DESC)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_evaluation_runs (
            id TEXT PRIMARY KEY,
            system_id TEXT NOT NULL,
            case_count INTEGER NOT NULL,
            metrics TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_evaluation_runs_created
           ON stream_memory_evaluation_runs(created_at DESC)"""
    )
    # Feedback collected before this migration remains useful evaluation truth.
    # Query casing is unavailable for those rows, so preserve the normalized text.
    conn.execute(
        """INSERT OR IGNORE INTO stream_memory_evaluation_cases
           (id, query_text, normalized_query, status, created_at, updated_at)
           SELECT 'memory_eval_' || LOWER(HEX(RANDOMBLOB(16))),
                  normalized_query, normalized_query, 'labeled',
                  MIN(created_at), MAX(updated_at)
           FROM stream_memory_feedback
           GROUP BY normalized_query"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO stream_memory_evaluation_targets
           (id, case_id, job_id, kind, entry_index, verdict, created_at, updated_at)
           SELECT 'memory_target_' || LOWER(HEX(RANDOMBLOB(16))),
                  c.id, f.job_id, f.kind, f.entry_index, f.verdict,
                  f.created_at, f.updated_at
           FROM stream_memory_feedback AS f
           JOIN stream_memory_evaluation_cases AS c
             ON c.normalized_query = f.normalized_query"""
    )


def _memory_identity_fingerprint(row: sqlite3.Row) -> str:
    kind = str(row["kind"] or "")
    clip_id = str(row["clip_id"] or "")
    start_ms = int(round(float(row["start_time"] or 0.0) * 1000.0))
    end_ms = int(round(float(row["end_time"] or 0.0) * 1000.0))
    title = " ".join(str(row["title"] or "").split()).casefold()
    text = " ".join(str(row["text"] or "").split()).casefold()
    payload = f"{kind}\n{clip_id}\n{start_ms}\n{end_ms}\n{title}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _migration_026_stream_memory_durable_identity(conn: sqlite3.Connection) -> None:
    """Fail closed when rebuilt entries no longer represent creator-labeled truth."""
    for table in ("stream_memory_feedback", "stream_memory_evaluation_targets"):
        _add_column(conn, table, "clip_id", "TEXT")
        _add_column(conn, table, "entry_fingerprint", "TEXT")
        rows = conn.execute(
            f"""SELECT t.id AS target_id, m.kind, m.clip_id, m.start_time,
                       m.end_time, m.title, m.text
                FROM {table} AS t
                LEFT JOIN stream_memory_entries AS m
                  ON m.job_id = t.job_id
                 AND m.kind = t.kind
                 AND m.entry_index = t.entry_index
                WHERE t.entry_fingerprint IS NULL"""
        ).fetchall()
        for row in rows:
            if row["kind"] is None:
                continue
            conn.execute(
                f"""UPDATE {table}
                    SET clip_id = ?, entry_fingerprint = ? WHERE id = ?""",
                (row["clip_id"], _memory_identity_fingerprint(row), row["target_id"]),
            )
        scope_column = "normalized_query" if table == "stream_memory_feedback" else "case_id"
        conn.execute(
            f"""DELETE FROM {table}
                WHERE clip_id IS NOT NULL AND rowid NOT IN (
                  SELECT MAX(rowid) FROM {table}
                  WHERE clip_id IS NOT NULL GROUP BY {scope_column}, clip_id
                )"""
        )
        conn.execute(
            f"""DELETE FROM {table}
                WHERE entry_fingerprint IS NOT NULL AND rowid NOT IN (
                  SELECT MAX(rowid) FROM {table}
                  WHERE entry_fingerprint IS NOT NULL
                  GROUP BY {scope_column}, job_id, kind, entry_fingerprint
                )"""
        )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_stream_memory_feedback_clip_identity
           ON stream_memory_feedback(normalized_query, clip_id)
           WHERE clip_id IS NOT NULL"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_stream_memory_feedback_entry_identity
           ON stream_memory_feedback(normalized_query, job_id, kind, entry_fingerprint)
           WHERE entry_fingerprint IS NOT NULL"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_stream_memory_eval_clip_identity
           ON stream_memory_evaluation_targets(case_id, clip_id)
           WHERE clip_id IS NOT NULL"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_stream_memory_eval_entry_identity
           ON stream_memory_evaluation_targets(case_id, job_id, kind, entry_fingerprint)
           WHERE entry_fingerprint IS NOT NULL"""
    )


def _migration_027_stream_memory_evidence_packs(conn: sqlite3.Connection) -> None:
    """Attach rebuildable, explainable scan evidence to Memory entries."""
    _add_column(conn, "stream_memory_entries", "evidence_kind", "TEXT")
    _add_column(
        conn,
        "stream_memory_entries",
        "evidence_json",
        "TEXT NOT NULL DEFAULT '{}'",
    )
    _add_column(
        conn,
        "stream_memory_index_state",
        "evidence_status",
        "TEXT NOT NULL DEFAULT 'pending'",
    )
    _add_column(
        conn,
        "stream_memory_index_state",
        "evidence_entries",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _add_column(
        conn,
        "stream_memory_index_state",
        "evidence_version",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _add_column(
        conn,
        "stream_memory_index_state",
        "evidence_source_bytes",
        "INTEGER NOT NULL DEFAULT 0",
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_evidence_kind
           ON stream_memory_entries(evidence_kind, source_date)"""
    )


def _migration_028_stream_memory_passive_learning(conn: sqlite3.Connection) -> None:
    """Persist bounded Memory interactions and shadow-only comparisons."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_search_sessions (
            id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            normalized_query TEXT NOT NULL,
            filters_json TEXT NOT NULL DEFAULT '{}',
            results_json TEXT NOT NULL DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_search_sessions_query
           ON stream_memory_search_sessions(normalized_query, created_at DESC)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_interactions (
            id TEXT PRIMARY KEY,
            search_id TEXT,
            event_type TEXT NOT NULL CHECK (
                event_type IN ('impression', 'open', 'clip_created', 'compilation_add')
            ),
            query_text TEXT NOT NULL DEFAULT '',
            normalized_query TEXT NOT NULL DEFAULT '',
            job_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            entry_index INTEGER NOT NULL,
            clip_id TEXT,
            entry_fingerprint TEXT NOT NULL,
            stable_key TEXT NOT NULL,
            rank INTEGER,
            context_json TEXT NOT NULL DEFAULT '{}',
            dedupe_key TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(search_id) REFERENCES stream_memory_search_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_interactions_entry
           ON stream_memory_interactions(stable_key, event_type, created_at DESC)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_interactions_query
           ON stream_memory_interactions(normalized_query, event_type, created_at DESC)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_memory_shadow_runs (
            id TEXT PRIMARY KEY,
            system_id TEXT NOT NULL,
            event_revision TEXT NOT NULL,
            case_count INTEGER NOT NULL,
            metrics TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_stream_memory_shadow_runs_created
           ON stream_memory_shadow_runs(created_at DESC)"""
    )


def _migration_029_stream_memory_continuous_indexing(conn: sqlite3.Connection) -> None:
    """Persist automatic semantic-update scheduling and bounded retry state."""
    _add_column(conn, "stream_memory_semantic_state", "pending_since", "TIMESTAMP")
    _add_column(conn, "stream_memory_semantic_state", "last_attempt_at", "TIMESTAMP")
    _add_column(conn, "stream_memory_semantic_state", "next_attempt_at", "TIMESTAMP")
    _add_column(
        conn,
        "stream_memory_semantic_state",
        "retry_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _add_column(conn, "stream_memory_semantic_state", "last_build_kind", "TEXT")
    _add_column(
        conn,
        "stream_memory_semantic_state",
        "last_encoded_entries",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _add_column(
        conn,
        "stream_memory_semantic_state",
        "last_reused_entries",
        "INTEGER NOT NULL DEFAULT 0",
    )
    conn.execute(
        """UPDATE stream_memory_semantic_state
           SET pending_since = COALESCE(pending_since, CURRENT_TIMESTAMP)
           WHERE source_revision != indexed_revision
             AND (SELECT COUNT(*) FROM stream_memory_entries) >= 2"""
    )


def _migration_030_stream_memory_evidence_coverage(conn: sqlite3.Connection) -> None:
    """Persist compact per-session evidence-coverage measurements."""
    _add_column(
        conn,
        "stream_memory_index_state",
        "coverage_json",
        "TEXT NOT NULL DEFAULT '{}'",
    )


def _migration_031_clip_caption_edits(conn: sqlite3.Connection) -> None:
    """Creator corrections to burned-in caption text.

    Captions are the deliverable, not an intermediate, and the per-clip ASR
    decode is fallible in two distinct ways (engines/caption/slang_words.py):
    a stable wrong word, which a lexicon rule can pin, and an unstable one on
    an out-of-vocabulary span, which nothing deterministic can reach. This
    table is the floor under both -- what the creator says the words are.

    ``words_json`` is the whole corrected word list (the flattened
    word/start/end stream generate_tiktok_ass renders), not a diff. A diff
    would not survive: the same audio decodes differently at different clip
    boundaries, so there is no stable machine text for a patch to anchor to.

    ``clip_start``/``clip_end`` record the window the edit was made against.
    A re-render inside the same window replays these words verbatim; after a
    boundary change the timings no longer describe the audio, so the edit is
    held (not silently applied) and the caller re-decodes.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clip_caption_edits (
            clip_id TEXT PRIMARY KEY,
            job_id TEXT,
            clip_start REAL NOT NULL,
            clip_end REAL NOT NULL,
            words_json TEXT NOT NULL,
            machine_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_caption_edits_job "
        "ON clip_caption_edits(job_id)"
    )


def _migration_032_compilation_match_scope(conn: sqlite3.Connection) -> None:
    """Let a compilation be scoped to one match inside one session.

    Cross-session drafts stay exactly as they were: every column is nullable
    and unset for them. A match-scoped project additionally remembers which
    game it came from, and its items may point at a detected moment that has
    no clip cut for it yet.
    """
    for column, ddl in (
        ("source_job_id", "TEXT"),
        ("match_index", "INTEGER"),
        ("match_label", "TEXT"),
        ("match_start", "REAL"),
        ("match_end", "REAL"),
    ):
        _add_column(conn, "compilation_projects", column, ddl)
    # 'clip' points at a canonical clip row; 'event' is a detected moment the
    # creator has not cut yet, carried by its time bounds alone.
    _add_column(conn, "compilation_project_items", "source_kind", "TEXT DEFAULT 'clip'")
    _add_column(conn, "compilation_project_items", "moment_kind", "TEXT")
    conn.execute(
        "UPDATE compilation_project_items SET source_kind = 'clip' WHERE source_kind IS NULL"
    )


def _migration_033_compilation_reels(conn: sqlite3.Connection) -> None:
    """Remember the rendered montage a compilation produced.

    Compilations deliver a video, not a list of clips, so a project owns the
    reel built from it: where it is, how long it runs, and whether an edit has
    made it stale. Every column is nullable, so drafts that never built one are
    unaffected.
    """
    for column, ddl in (
        ("reel_path", "TEXT"),
        ("reel_duration_seconds", "REAL"),
        ("reel_built_at", "TIMESTAMP"),
        ("reel_stale", "INTEGER NOT NULL DEFAULT 0"),
    ):
        _add_column(conn, "compilation_projects", column, ddl)
    # Kills render clean so a gameplay montage is not covered in talking-head
    # subtitles; the closing win keeps its captions.
    _add_column(conn, "compilation_project_items", "captions_enabled", "INTEGER")


def _migration_034_compilation_muted_moments(conn: sqlite3.Connection) -> None:
    """Let a montage carry a sound instead of the moments' own audio.

    A kill montage is usually posted over a track, with the creator's voice
    kept only where the reaction is the payoff. Mute is per moment and lives
    on the item, so it survives a rebuild and can differ inside one reel.
    """
    _add_column(conn, "compilation_project_items", "muted", "INTEGER NOT NULL DEFAULT 0")


def _migration_035_clip_origin(conn: sqlite3.Connection) -> None:
    """Separate machine-made compilation cuts from the creator's own clips.

    A montage cut is a means to a video, not a clip the creator kept. Writing
    them through the manual-clip path put them in Clip Library as kept clips
    and recorded each as a positive training label, which buried real keeps and
    taught the ranker on cuts nobody chose. NULL is every existing clip.
    """
    _add_column(conn, "clips", "origin", "TEXT")


MIGRATIONS: list[Migration] = [
    ("001_initial_schema", _migration_001_initial),
    ("002_job_telemetry", _migration_002_job_telemetry),
    ("003_clip_evidence", _migration_003_clip_evidence),
    ("004_reaction_learning", _migration_004_reaction_learning),
    ("005_clip_thumbnails", _migration_005_clip_thumbnails),
    ("006_studio_columns", _migration_006_studio_columns),
    ("007_learning_context", _migration_007_learning_context),
    ("008_clip_diagnostics", _migration_008_clip_diagnostics),
    ("009_boundary_edits", _migration_009_boundary_edits),
    ("010_job_framing", _migration_010_job_framing),
    ("011_job_poster", _migration_011_job_poster),
    ("012_maybe_verdict", _migration_012_maybe_verdict),
    ("013_storage_lifecycle", _migration_013_storage_lifecycle),
    ("014_explicit_clip_scores", _migration_014_explicit_clip_scores),
    ("015_job_source_date", _migration_015_job_source_date),
    ("016_stable_clip_numbers", _migration_016_stable_clip_numbers),
    ("017_recall_sessions", _migration_017_recall_sessions),
    ("018_stream_memory", _migration_018_stream_memory),
    ("019_stream_memory_semantic_state", _migration_019_stream_memory_semantic_state),
    ("020_compilation_projects", _migration_020_compilation_projects),
    ("021_compilation_source_groups", _migration_021_compilation_source_groups),
    ("022_compilation_preparations", _migration_022_compilation_preparations),
    ("023_stream_memory_creator_learning", _migration_023_stream_memory_creator_learning),
    ("024_stream_memory_stable_feedback", _migration_024_stream_memory_stable_feedback),
    ("025_stream_memory_evaluation", _migration_025_stream_memory_evaluation),
    ("026_stream_memory_durable_identity", _migration_026_stream_memory_durable_identity),
    ("027_stream_memory_evidence_packs", _migration_027_stream_memory_evidence_packs),
    ("028_stream_memory_passive_learning", _migration_028_stream_memory_passive_learning),
    ("029_stream_memory_continuous_indexing", _migration_029_stream_memory_continuous_indexing),
    ("030_stream_memory_evidence_coverage", _migration_030_stream_memory_evidence_coverage),
    ("031_clip_caption_edits", _migration_031_clip_caption_edits),
    ("032_compilation_match_scope", _migration_032_compilation_match_scope),
    ("033_compilation_reels", _migration_033_compilation_reels),
    ("034_compilation_muted_moments", _migration_034_compilation_muted_moments),
    ("035_clip_origin", _migration_035_clip_origin),
]


def run_migrations(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    _ensure_migrations_table(conn)
    applied = _applied_versions(conn)
    conn.commit()
    for version, migrate in MIGRATIONS:
        if version in applied:
            continue
        try:
            conn.execute("BEGIN")
            migrate(conn)
            conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

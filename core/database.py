# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import sqlite3
import os
import json
from contextlib import contextmanager

from core.bundle_paths import get_data_dir
from core.migrations import run_migrations

DATABASE_FILENAME = "recall.db"
LEGACY_DATABASE_FILENAME = "g" + "ie.db"
DB_PATH = os.path.join(get_data_dir(), DATABASE_FILENAME)


def _legacy_db_path() -> str:
    return os.path.join(get_data_dir(), LEGACY_DATABASE_FILENAME)

class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        if db_path == DB_PATH and not os.path.exists(db_path) and os.path.exists(_legacy_db_path()):
            # Migrate the legacy-named DB to recall.db instead of keeping the
            # old name alive forever. If the file is locked (another process
            # still has it open), fall back to using it in place as before.
            if self._migrate_legacy_db():
                db_path = DB_PATH
            else:
                db_path = _legacy_db_path()
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()

    @staticmethod
    def _migrate_legacy_db() -> bool:
        legacy = _legacy_db_path()
        try:
            os.replace(legacy, DB_PATH)
        except OSError:
            return False
        # WAL sidecars only exist while a connection is open; if the main file
        # renamed, nothing holds them, so these should follow.
        for suffix in ("-wal", "-shm"):
            try:
                if os.path.exists(legacy + suffix):
                    os.replace(legacy + suffix, DB_PATH + suffix)
            except OSError:
                pass
        print(f"Migrated database: {legacy} -> {DB_PATH}")
        return True

    def _ensure_db_dir(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        # ``synchronous`` is connection-local, unlike WAL mode. Apply NORMAL
        # here so every short-lived connection gets the intended durability /
        # write-throughput balance rather than silently falling back to FULL.
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self.get_connection() as conn:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.OperationalError:
                pass
            # Schema is defined and versioned entirely in core/migrations —
            # each migration is idempotent, applied once, and tracked in the
            # schema_migrations table. No inline DDL here: one source of truth.
            run_migrations(conn)

    # --- Reaction timeline: per-job R(t) curve for the review UI -------------
    def set_reaction_timeline(self, job_id: str, timeline: dict):
        with self.get_connection() as conn:
            conn.cursor().execute(
                'INSERT OR REPLACE INTO reaction_timelines (job_id, timeline) VALUES (?, ?)',
                (job_id, json.dumps(timeline))
            )

    def get_reaction_timeline(self, job_id: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT timeline FROM reaction_timelines WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()
        if not row or not row['timeline']:
            return None
        try:
            return json.loads(row['timeline'])
        except (ValueError, TypeError):
            return None

    def delete_reaction_timeline(self, job_id: str):
        with self.get_connection() as conn:
            conn.cursor().execute('DELETE FROM reaction_timelines WHERE job_id = ?', (job_id,))

    # --- Learned ranker: feature snapshots + keep/reject labels (plan §5.6) ---
    def set_clip_features(self, clip_id: str, features: dict):
        """Snapshot a clip's reaction-engine feature dict so it can be labeled later."""
        with self.get_connection() as conn:
            conn.cursor().execute(
                'UPDATE clips SET features = ? WHERE id = ?',
                (json.dumps(features), clip_id)
            )

    def record_label(self, clip_id: str, label: float, event: str, weight: float = 1.0):
        """Log a keep(1), maybe(.5), or reject(0) signal with its features.

        Also snapshots the clip's presentation rank (its position in the
        session's confidence-descending review order — the order the Theater
        deals cards). Keep decisions are position-biased: early cards get more
        attention. The trainer feeds rank as a train-only feature so the model
        can attribute some of the keep signal to position instead of content
        (plan 22 §4.3).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT features, job_id FROM clips WHERE id = ?', (clip_id,))
            row = cursor.fetchone()
            features = row['features'] if row and row['features'] else None
            rank = None
            if row and row['job_id']:
                # Mirror the Theater's deal order: confidence desc, then time.
                cursor.execute(
                    "SELECT id FROM clips WHERE job_id = ? "
                    "ORDER BY CASE WHEN story_label = 'recall_more_candidate' THEN 1 ELSE 0 END, "
                    "COALESCE(deck_score, score) DESC, start_time ASC",
                    (row['job_id'],),
                )
                ordered = [r['id'] for r in cursor.fetchall()]
                if clip_id in ordered:
                    rank = ordered.index(clip_id) + 1
            cursor.execute(
                '''INSERT INTO clip_labels
                   (clip_id, job_id, features, label, label_value, event, rank, weight)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    clip_id,
                    row['job_id'] if row else None,
                    features,
                    float(label),
                    float(label),
                    event,
                    rank,
                    max(0.0, float(weight)),
                )
            )

    def retract_label(self, clip_id: str, event: str):
        """Return one feedback action to neutral without inventing its inverse.

        For example, undoing Pass is not a Keep.  Deleting the matching pass
        events lets an earlier explicit Keep/Export become authoritative again,
        or leaves the clip unlabeled when no other decision exists.
        """
        with self.get_connection() as conn:
            conn.cursor().execute(
                'DELETE FROM clip_labels WHERE clip_id = ? AND event = ?',
                (clip_id, event),
            )

    def record_labels_batch(self, clip_ids, label: float, event: str):
        """Log keep/reject labels for many clips in one connection.

        Equivalent to calling record_label per clip, but the per-job
        presentation ordering (used for position-bias rank) is computed once
        per distinct job instead of re-querying and re-sorting the whole job
        for every clip — O(jobs) sorts instead of O(clips) sorts.
        """
        clip_ids = list(clip_ids)
        if not clip_ids:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(clip_ids))
            cursor.execute(
                f"SELECT id, features, job_id FROM clips WHERE id IN ({placeholders})",  # nosec B608
                clip_ids,
            )
            rows = {r["id"]: r for r in cursor.fetchall()}

            # One ordering per distinct job (mirrors the Theater's deal order).
            rank_by_clip: dict = {}
            for job_id in {r["job_id"] for r in rows.values() if r["job_id"]}:
                cursor.execute(
                    'SELECT id FROM clips WHERE job_id = ? '
                    "ORDER BY CASE WHEN story_label = 'recall_more_candidate' THEN 1 ELSE 0 END, "
                    "COALESCE(deck_score, score) DESC, start_time ASC",
                    (job_id,),
                )
                for position, r in enumerate(cursor.fetchall(), start=1):
                    rank_by_clip[r["id"]] = position

            for clip_id in clip_ids:
                row = rows.get(clip_id)
                features = row["features"] if row and row["features"] else None
                cursor.execute(
                    '''INSERT INTO clip_labels
                       (clip_id, job_id, features, label, label_value, event, rank, weight)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        clip_id,
                        row["job_id"] if row else None,
                        features,
                        float(label),
                        float(label),
                        event,
                        rank_by_clip.get(clip_id),
                        1.0,
                    ),
                )

    # --- Boundary edits: creator trims as boundary supervision (HUMAN_CLIPS pkg 1)
    def record_boundary_edit(self, clip_id: str, job_id, old_start: float,
                             old_end: float, new_start: float, new_end: float):
        """Log a creator's manual re-cut of a clip window.

        Deliberately NOT a clip_labels row: a trim corrects the CUT, not the
        keep/reject verdict, so it must never enter ranker preference training.
        Consumed by scripts/export_founder_fixture.py, which turns the final
        edited window into a truth-v2 "boundary_fix" decision.
        """
        with self.get_connection() as conn:
            conn.cursor().execute(
                '''INSERT INTO clip_boundary_edits
                   (clip_id, job_id, old_start, old_end, new_start, new_end)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (
                    clip_id,
                    job_id,
                    float(old_start),
                    float(old_end),
                    float(new_start),
                    float(new_end),
                ),
            )

    def get_boundary_edits(self, job_id=None):
        """Boundary-edit rows (oldest first), optionally scoped to one job."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if job_id is None:
                cursor.execute('SELECT * FROM clip_boundary_edits ORDER BY id ASC')
            else:
                cursor.execute(
                    'SELECT * FROM clip_boundary_edits WHERE job_id = ? ORDER BY id ASC',
                    (job_id,),
                )
            return [dict(row) for row in cursor.fetchall()]

    # -- caption edits ----------------------------------------------------
    #
    # Word timings are stored CLIP-RELATIVE, matching what
    # transcribe_clip_window returns and what generate_tiktok_ass consumes, so
    # a stored edit drops into the render path in place of a fresh decode.

    def record_caption_edit(self, clip_id: str, job_id, clip_start: float,
                            clip_end: float, words, machine_text: str = ""):
        """Store the creator's corrected caption words for one clip.

        One row per clip: a correction replaces the previous correction rather
        than stacking, because the row IS the caption -- there is no meaningful
        history of "what they meant before they were done".
        """
        with self.get_connection() as conn:
            conn.cursor().execute(
                '''INSERT INTO clip_caption_edits
                   (clip_id, job_id, clip_start, clip_end, words_json,
                    machine_text, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(clip_id) DO UPDATE SET
                     job_id = excluded.job_id,
                     clip_start = excluded.clip_start,
                     clip_end = excluded.clip_end,
                     words_json = excluded.words_json,
                     machine_text = excluded.machine_text,
                     updated_at = CURRENT_TIMESTAMP''',
                (
                    clip_id,
                    job_id,
                    float(clip_start),
                    float(clip_end),
                    json.dumps(list(words)),
                    machine_text or "",
                ),
            )

    def get_caption_edit(self, clip_id: str):
        """The stored correction for one clip, or None.

        ``words`` comes back parsed. A row whose JSON no longer parses is
        treated as absent rather than raised: a corrupt edit must degrade to
        the machine decode, never block the render of a clip.
        """
        with self.get_connection() as conn:
            row = conn.cursor().execute(
                "SELECT * FROM clip_caption_edits WHERE clip_id = ?", (clip_id,),
            ).fetchone()
        if not row:
            return None
        edit = dict(row)
        try:
            edit["words"] = json.loads(edit.pop("words_json"))
        except (TypeError, ValueError):
            return None
        if not isinstance(edit["words"], list) or not edit["words"]:
            return None
        return edit

    def clear_caption_edit(self, clip_id: str) -> bool:
        """Drop a correction so the clip falls back to the machine decode."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM clip_caption_edits WHERE clip_id = ?", (clip_id,))
            return cursor.rowcount > 0

    def set_ranker_metadata(self, key: str, value):
        with self.get_connection() as conn:
            conn.cursor().execute(
                'INSERT OR REPLACE INTO ranker_metadata (key, value) VALUES (?, ?)',
                (key, json.dumps(value))
            )

    def get_ranker_metadata(self, key: str, default=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('SELECT value FROM ranker_metadata WHERE key = ?', (key,))
            except sqlite3.OperationalError:
                return default
            row = cursor.fetchone()
            return json.loads(row['value']) if row else default

    def get_training_data(self, events=None):
        """Return rows with a real-valued Pass/Maybe/Keep target.

        If the same clip has both a keep and a reject signal, the latest wins.
        When ``events`` is supplied, retain only those events *before*
        collapsing to the latest decision per clip.  This lets a personal
        trainer use explicit Keep/Pass feedback without a later export,
        manual, or Maybe action masking that supervision.
        """
        if events is not None:
            events = tuple(events)
            # An explicit empty filter must never quietly fall back to the
            # legacy all-events query.
            if not events:
                return []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if events is None:
                # Preserve the legacy query and result shape for callers that
                # do not opt into the explicit-supervision policy.
                query = '''
                    SELECT id, clip_id, job_id, features,
                           COALESCE(label_value, CAST(label AS REAL)) AS label,
                           event, rank, COALESCE(weight, 1.0) AS weight
                    FROM clip_labels
                    WHERE features IS NOT NULL
                    ORDER BY id ASC
                '''
                params = ()
            else:
                placeholders = ",".join("?" * len(events))
                query = f'''
                    SELECT clip_labels.id, clip_labels.clip_id,
                           clip_labels.job_id, clip_labels.features,
                           COALESCE(label_value, CAST(label AS REAL)) AS label,
                           clip_labels.event, clip_labels.rank,
                           COALESCE(clip_labels.weight, 1.0) AS weight,
                           jobs.source_path, jobs.asset_path
                    FROM clip_labels
                    LEFT JOIN jobs ON jobs.id = clip_labels.job_id
                    WHERE clip_labels.features IS NOT NULL
                      AND clip_labels.event IN ({placeholders})
                    ORDER BY clip_labels.id ASC
                '''
                params = events
            cursor.execute(query, params)
            latest = {}
            for row in cursor.fetchall():
                latest[row['clip_id']] = dict(row)
        return [
            {
                "clip_id": row["clip_id"],
                "job_id": row["job_id"],
                "features": json.loads(row["features"]),
                "label": float(row["label"]),
                "event": row["event"],
                "rank": row["rank"],
                "weight": float(row["weight"]),
                # Provenance is needed only by explicit-policy consumers. Keep
                # the historical no-argument result shape byte-for-byte
                # compatible for legacy training/eval callers.
                **({
                    "id": int(row["id"]),
                    "source_path": row["source_path"],
                    "asset_path": row["asset_path"],
                } if events is not None else {}),
            }
            for row in latest.values()
        ]

    def clear_learning_data(self):
        """Clear creator feedback used for personalization without touching clips."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM clip_labels")
            cursor.execute("DELETE FROM ranker_metadata")

    # Settings CRUD
    def set_setting(self, key: str, value):
        with self.get_connection() as conn:
            conn.cursor().execute(
                'INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)',
                (key, json.dumps(value))
            )
            
    def get_setting(self, key: str, default=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row['value'])
            return default

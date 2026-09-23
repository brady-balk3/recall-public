# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Learned-ranker service (plan §5.6) — model resolution + per-install training.

Ties together the DB labels, the canonical feature spec, and the numpy ranker:
  * ``get_active_ranker()`` -> bundled base quality plus an optional validated
    personal adapter, else None (caller falls back to the signal path).
  * ``train_user_ranker(db)`` -> fine-tune the base on captured keep/reject labels
    once there are enough of them, and save the per-install model.
"""

import hashlib
import functools
import json
import os
import re
import threading
from datetime import datetime, timezone

import numpy as np

from core.bundle_paths import get_models_dir, get_data_dir
from core.model_vault import protected_json_exists, sealed_path_for
from core.source_identity import canonical_source_key
import engines.reaction.features as feat
import engines.reaction.ranker as rk

BASE_PATH = os.path.join(get_models_dir(), "ranker_base.json")
PROFILE_ENV = "RECALL_PERSONALIZATION_PROFILE"


def _profile_slug(value: str | None) -> str:
    """Stable, filesystem-safe personalization namespace.

    Packaged installs retain the historical ``default`` paths. Development can
    opt into an explicit founder profile without letting that adapter masquerade
    as a future product user's taste model.
    """
    raw = str(value or "default").strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-._")
    return slug[:64] or "default"


PERSONALIZATION_PROFILE = _profile_slug(os.environ.get(PROFILE_ENV))
if PERSONALIZATION_PROFILE == "default":
    USER_PATH = os.path.join(get_data_dir(), "ranker_user.json")
    USER_GATE_PATH = os.path.join(get_data_dir(), "ranker_user.gate.json")
    USER_CHALLENGER_PATH = os.path.join(get_data_dir(), "ranker_user.challenger.json")
    USER_CHALLENGER_META_PATH = os.path.join(get_data_dir(), "ranker_user.challenger.meta.json")
    USER_CHALLENGER_GENERATIONS_DIR = os.path.join(get_data_dir(), "ranker_user.challenger.generations")
    USER_CHAMPION_BACKUP_PATH = os.path.join(get_data_dir(), "ranker_user.previous.json")
    USER_GATE_BACKUP_PATH = os.path.join(get_data_dir(), "ranker_user.gate.previous.json")
    USER_PROMOTION_JOURNAL_PATH = os.path.join(get_data_dir(), "ranker_user.promotion.pending.json")
else:
    _PROFILE_DIR = os.path.join(
        get_data_dir(), "personalization", PERSONALIZATION_PROFILE,
    )
    USER_PATH = os.path.join(_PROFILE_DIR, "ranker_user.json")
    USER_GATE_PATH = os.path.join(_PROFILE_DIR, "ranker_user.gate.json")
    USER_CHALLENGER_PATH = os.path.join(_PROFILE_DIR, "ranker_user.challenger.json")
    USER_CHALLENGER_META_PATH = os.path.join(_PROFILE_DIR, "ranker_user.challenger.meta.json")
    USER_CHALLENGER_GENERATIONS_DIR = os.path.join(_PROFILE_DIR, "ranker_user.challenger.generations")
    USER_CHAMPION_BACKUP_PATH = os.path.join(_PROFILE_DIR, "ranker_user.previous.json")
    USER_GATE_BACKUP_PATH = os.path.join(_PROFILE_DIR, "ranker_user.gate.previous.json")
    USER_PROMOTION_JOURNAL_PATH = os.path.join(_PROFILE_DIR, "ranker_user.promotion.pending.json")


def user_model_path_for_root(data_root: str) -> str:
    """Champion path under an arbitrary data root, using this profile's layout.

    The module constants above bind to ``get_data_dir()`` at import. A caller
    that owns a *different* root -- a test with a temporary directory, or any
    future multi-root tool -- must not reach through them to the real install:
    ``wipe()`` did exactly that and destroyed a real founder champion despite
    the caller having redirected its own data root.
    """
    if PERSONALIZATION_PROFILE == "default":
        return os.path.join(data_root, "ranker_user.json")
    return os.path.join(
        data_root, "personalization", PERSONALIZATION_PROFILE, "ranker_user.json",
    )


# Require multiple explicit clips and within-session preference evidence before
# training a personal candidate. Activation additionally requires the separate
# deck-quality gate below. Training is pointwise so all-Pass sessions remain
# valuable negative supervision instead of contributing zero pairs.
#
# Package 4 note (2026-07-16): the planned raise to 24 labels / 3 sessions /
# base-model-required is DEFERRED. The founder-set LOVO run (scripts/
# train_base_ranker.py) showed even a 310-label model failing the deck-level
# ship gate, so no base exists to require — and requiring one would disable
# personalization outright. Revisit both together once a base model earns its
# gate.
MIN_LABELS = 12
MIN_PAIRS = 8

# Personalization freeze (HUMAN_CLIPS Package 1 guardrail): while the golden
# set is being measured/labeled, a mid-window retrain or personal-model swap
# would silently change every subsequent eval run.
FREEZE_ENV = "RECALL_FREEZE_PERSONALIZATION"
_TRAINING_LOCK = threading.RLock()


def _serialized_training(operation):
    """Keep concurrent feedback tasks from racing one model artifact."""
    @functools.wraps(operation)
    def wrapped(*args, **kwargs):
        with _TRAINING_LOCK:
            return operation(*args, **kwargs)
    return wrapped


class RankerStack:
    """Universal base scorer plus an optional validated taste adapter.

    ``predict`` deliberately means BASE quality so existing selection callers
    cannot accidentally turn one creator's taste into universal clipability.
    Personal scores are available only through ``predict_personal`` and are
    consumed as a separately capped ordering signal.
    """

    def __init__(self, base, personal=None, challenger=None):
        self.base = base
        self.personal = personal
        self.challenger = challenger
        self.feature_names = list(base.feature_names)
        self.version = int(base.version)
        self.n_labels = int(
            personal.n_labels if personal is not None else base.n_labels
        )

    @property
    def has_personal(self) -> bool:
        return self.personal is not None

    @property
    def has_challenger(self) -> bool:
        return self.challenger is not None

    def predict(self, X):
        return self.base.predict(X)

    def predict_calibrated(self, X):
        return self.base.predict_calibrated(X)

    def predict_personal(self, X):
        if self.personal is None:
            raise ValueError("no validated personal adapter is active")
        return self.personal.predict(X)

    def predict_challenger(self, X):
        if self.challenger is None:
            raise ValueError("no personal challenger is available")
        return self.challenger.predict(X)


def _personalization_frozen() -> bool:
    """True when RECALL_FREEZE_PERSONALIZATION is set to 1/true.

    Freezing pins ranking to the shipped base model (the user model is
    skipped, training/sync refuse) without touching any file on disk, so
    unfreezing restores the exact pre-freeze state. Read from the environment
    at CALL time, never cached at import, so eval scripts and tests can toggle
    it per-run.
    """
    return os.environ.get(FREEZE_ENV, "").strip().lower() in ("1", "true")


def _compatible_rows(data):
    compatible = []
    for row in data:
        if not row.get("job_id"):
            continue
        features = feat.ensure_feature_layout(row.get("features") or {})
        if features is None:
            continue
        compatible.append({**row, "features": features})
    return compatible


def _training_rows(db):
    """Only deliberate Keep/Pass decisions may change personal artifacts."""
    return _compatible_rows(db.get_training_data(events=("saved", "passed")))


def _load_compatible(path):
    if not os.path.exists(path):
        return None
    try:
        model = rk.LogisticRanker.load(path)
    except Exception:
        return None
    if model.version != feat.FEATURE_VERSION or model.feature_names != feat.FEATURE_NAMES:
        return None
    return model


def _atomic_save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    try:
        model.save(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_write_json(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_write_bytes(payload: bytes, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _promotion_transaction_paths() -> dict[str, str]:
    return {
        "old_model": USER_PROMOTION_JOURNAL_PATH + ".old.model",
        "old_gate": USER_PROMOTION_JOURNAL_PATH + ".old.gate",
        "new_model": USER_PROMOTION_JOURNAL_PATH + ".new.model",
        "new_gate": USER_PROMOTION_JOURNAL_PATH + ".new.gate",
    }


def _cleanup_promotion_transaction() -> None:
    for path in (*_promotion_transaction_paths().values(), USER_PROMOTION_JOURNAL_PATH):
        _remove_file(path)


def _promotion_file_matches(path: str, expected_sha256: str | None) -> bool:
    if expected_sha256 is None:
        return not os.path.exists(path)
    return os.path.isfile(path) and _model_sha256(path) == expected_sha256


def _recover_pending_promotion() -> bool:
    """Recover a process-interrupted champion/gate swap.

    The journal is written only after durable old/new snapshots exist. If both
    active files are already the new pair, finalize the backup and commit. Any
    partial state rolls back to the exact old pair. A malformed or incomplete
    journal is retained and makes `_user_gate_status` fail closed.
    """
    if not os.path.exists(USER_PROMOTION_JOURNAL_PATH):
        return True
    paths = _promotion_transaction_paths()
    try:
        journal = _read_json(USER_PROMOTION_JOURNAL_PATH)
        if (
            int(journal.get("version", -1)) != 1
            or journal.get("personalization_profile") != PERSONALIZATION_PROFILE
        ):
            return False
        old_model_sha = journal.get("old_model_sha256")
        old_gate_sha = journal.get("old_gate_sha256")
        new_model_sha = journal.get("new_model_sha256")
        new_gate_sha = journal.get("new_gate_sha256")
        hashes = (old_model_sha, old_gate_sha, new_model_sha, new_gate_sha)
        if any(
            value is not None
            and not re.fullmatch(r"[a-f0-9]{64}", str(value))
            for value in hashes
        ):
            return False
        if new_model_sha is None or new_gate_sha is None:
            return False

        new_pair_active = (
            _promotion_file_matches(USER_PATH, new_model_sha)
            and _promotion_file_matches(USER_GATE_PATH, new_gate_sha)
        )
        if new_pair_active:
            # The old pair becomes the recoverable user-facing backup only
            # after the new pair is known complete.
            if old_model_sha is not None:
                if not _promotion_file_matches(paths["old_model"], old_model_sha):
                    return False
                _atomic_write_bytes(
                    open(paths["old_model"], "rb").read(),
                    USER_CHAMPION_BACKUP_PATH,
                )
            else:
                _remove_file(USER_CHAMPION_BACKUP_PATH)
            if old_gate_sha is not None:
                if not _promotion_file_matches(paths["old_gate"], old_gate_sha):
                    return False
                _atomic_write_bytes(
                    open(paths["old_gate"], "rb").read(),
                    USER_GATE_BACKUP_PATH,
                )
            else:
                _remove_file(USER_GATE_BACKUP_PATH)
            _cleanup_promotion_transaction()
            return True

        # A partial swap never becomes active. Validate the durable rollback
        # snapshots before touching either active path.
        if not _promotion_file_matches(paths["old_model"], old_model_sha):
            return False
        if not _promotion_file_matches(paths["old_gate"], old_gate_sha):
            return False
        # Gate first keeps the pair mismatched/Base-only until the corresponding
        # old model is restored. Missing old files are restored as missing.
        if old_gate_sha is None:
            _remove_file(USER_GATE_PATH)
        else:
            _atomic_write_bytes(open(paths["old_gate"], "rb").read(), USER_GATE_PATH)
        if old_model_sha is None:
            _remove_file(USER_PATH)
        else:
            _atomic_write_bytes(open(paths["old_model"], "rb").read(), USER_PATH)
        if not _promotion_file_matches(USER_PATH, old_model_sha):
            return False
        if not _promotion_file_matches(USER_GATE_PATH, old_gate_sha):
            return False
        _cleanup_promotion_transaction()
        return True
    except Exception:  # noqa: BLE001 - recovery must fail closed, never guess
        return False


def _feature_schema_sha256() -> str:
    return hashlib.sha256(
        json.dumps(list(feat.FEATURE_NAMES), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _challenger_manifest_status(model=None, model_path=None, meta_path=None) -> tuple[bool, str, dict | None]:
    """Validate provenance without ever granting a challenger runtime authority."""
    model_path = model_path or USER_CHALLENGER_PATH
    meta_path = meta_path or USER_CHALLENGER_META_PATH
    if not os.path.exists(meta_path):
        return False, "missing_manifest", None
    try:
        manifest = _read_json(meta_path)
        if int(manifest.get("version", -1)) != 1 or manifest.get("role") != "challenger":
            return False, "invalid_manifest_contract", manifest
        if not os.path.exists(model_path):
            return False, "missing_model", manifest
        model_sha = manifest.get("model_sha256")
        if (
            not isinstance(model_sha, str)
            or not re.fullmatch(r"[a-f0-9]{64}", model_sha)
            or model_sha != _model_sha256(model_path)
        ):
            return False, "model_hash_mismatch", manifest
        if manifest.get("base_sha256") != _base_sha256():
            return False, "base_hash_mismatch", manifest
        if int(manifest.get("feature_version", -1)) != int(feat.FEATURE_VERSION):
            return False, "feature_version_mismatch", manifest
        if manifest.get("feature_schema_sha256") != _feature_schema_sha256():
            return False, "feature_schema_mismatch", manifest
        if manifest.get("personalization_profile") != PERSONALIZATION_PROFILE:
            return False, "profile_mismatch", manifest
        if model is None:
            model = _load_compatible(model_path)
        if model is None:
            return False, "incompatible_model", manifest
        if int(manifest.get("label_count", -1)) != int(model.n_labels):
            return False, "label_count_mismatch", manifest
        if (
            not isinstance(manifest.get("max_label_id"), int)
            or int(manifest["max_label_id"]) < 0
        ):
            return False, "missing_label_cutoff", manifest
        if (
            not isinstance(manifest.get("training_job_ids"), list)
            or not manifest["training_job_ids"]
        ):
            return False, "missing_training_jobs", manifest
        if (
            not isinstance(manifest.get("training_source_keys"), list)
            or not manifest["training_source_keys"]
        ):
            return False, "missing_training_sources", manifest
        if manifest.get("training_events") != ["saved", "passed"]:
            return False, "invalid_training_events", manifest
        training_decisions = manifest.get("training_decisions")
        if (
            not isinstance(training_decisions, list)
            or len(training_decisions) != int(manifest.get("label_count", -1))
            or any(
                not isinstance(row, dict)
                or not isinstance(row.get("id"), int)
                or not row.get("clip_id")
                or not row.get("job_id")
                or row.get("event") not in ("saved", "passed")
                or float(row.get("label", -1)) not in (0.0, 1.0)
                for row in training_decisions
            )
        ):
            return False, "invalid_training_decisions", manifest
    except Exception:  # malformed provenance must fail closed
        return False, "invalid_manifest", None
    return True, "valid", manifest


def _generation_paths(model_sha256: str) -> tuple[str, str]:
    if not re.fullmatch(r"[a-f0-9]{64}", str(model_sha256)):
        raise ValueError("invalid challenger generation hash")
    root = os.path.join(USER_CHALLENGER_GENERATIONS_DIR, model_sha256)
    return root + ".json", root + ".meta.json"


def _archive_current_challenger() -> bool:
    """Preserve only a complete valid generation before a newer shadow replaces it."""
    model = _load_compatible(USER_CHALLENGER_PATH)
    valid, _reason, manifest = _challenger_manifest_status(model)
    if not valid or model is None or manifest is None:
        return False
    model_sha = str(manifest["model_sha256"])
    archived_model, archived_meta = _generation_paths(model_sha)
    if os.path.exists(archived_model) and os.path.exists(archived_meta):
        return True
    os.makedirs(USER_CHALLENGER_GENERATIONS_DIR, exist_ok=True)
    model_tmp, meta_tmp = archived_model + ".tmp", archived_meta + ".tmp"
    try:
        with open(USER_CHALLENGER_PATH, "rb") as source, open(model_tmp, "wb") as target:
            target.write(source.read())
        with open(USER_CHALLENGER_META_PATH, "rb") as source, open(meta_tmp, "wb") as target:
            target.write(source.read())
        os.replace(model_tmp, archived_model)
        os.replace(meta_tmp, archived_meta)
    finally:
        for path in (model_tmp, meta_tmp):
            if os.path.exists(path):
                os.remove(path)
    return True


def _resolve_challenger_generation(model_sha256: str):
    """Return the requested current or archived artifact, never a merely latest model."""
    candidates = [(USER_CHALLENGER_PATH, USER_CHALLENGER_META_PATH)]
    try:
        candidates.append(_generation_paths(model_sha256))
    except ValueError:
        return None, None, None
    for model_path, meta_path in candidates:
        if not os.path.exists(model_path) or _model_sha256(model_path) != model_sha256:
            continue
        model = _load_compatible(model_path)
        valid, _reason, manifest = _challenger_manifest_status(model, model_path, meta_path)
        if valid:
            return model_path, meta_path, manifest
    return None, None, None


def _atomic_save_challenger(model, manifest: dict) -> None:
    """Install a matching shadow model/manifest pair, or leave the old pair intact.

    Manifest serialization happens before replacing the model; the common failed
    write path therefore cannot strand a new model beside old provenance.
    """
    os.makedirs(os.path.dirname(USER_CHALLENGER_PATH), exist_ok=True)
    model_tmp = USER_CHALLENGER_PATH + ".tmp"
    meta_tmp = USER_CHALLENGER_META_PATH + ".tmp"
    old_model = open(USER_CHALLENGER_PATH, "rb").read() if os.path.exists(USER_CHALLENGER_PATH) else None
    old_meta = open(USER_CHALLENGER_META_PATH, "rb").read() if os.path.exists(USER_CHALLENGER_META_PATH) else None
    try:
        model.save(model_tmp)
        # bind the manifest to the exact serialized bytes, not to in-memory weights
        manifest = {**manifest, "model_sha256": _model_sha256(model_tmp)}
        with open(meta_tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.replace(model_tmp, USER_CHALLENGER_PATH)
        os.replace(meta_tmp, USER_CHALLENGER_META_PATH)
    except Exception:
        # Restore the prior complete pair atomically enough for in-process I/O
        # faults; a crash between replacements remains intentionally inert.
        if old_model is not None:
            _atomic_write_bytes(old_model, USER_CHALLENGER_PATH)
        else:
            _remove_file(USER_CHALLENGER_PATH)
        if old_meta is not None:
            _atomic_write_bytes(old_meta, USER_CHALLENGER_META_PATH)
        else:
            _remove_file(USER_CHALLENGER_META_PATH)
        raise
    finally:
        for temporary in (model_tmp, meta_tmp):
            if os.path.exists(temporary):
                os.remove(temporary)


def _pair_count(data) -> int:
    by_job = {}
    for row in data:
        by_job.setdefault(row["job_id"], []).append(float(row["label"]))
    return sum(
        sum(1 for i, left in enumerate(labels) for right in labels[i + 1:] if left != right)
        for labels in by_job.values()
    )


def _job_count(data) -> int:
    return len({row["job_id"] for row in data})


def _compatible_base():
    # Sealed-aware: only the encrypted form exists in a packaged build, so a
    # plain os.path.exists here would silently disable the bundled base ranker
    # on every shipped install while looking perfectly healthy.
    if not protected_json_exists(BASE_PATH):
        return None
    try:
        model = rk.LogisticRanker.load(BASE_PATH)
    except Exception:
        return None
    if model.version != feat.FEATURE_VERSION:
        return None
    if model.feature_names != feat.FEATURE_NAMES:
        return None
    return model


def _model_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranker_identity(model) -> str:
    """Sealed-aware stable identity of the effective base parameters."""
    payload = {
        "version": int(model.version), "feature_names": list(model.feature_names),
        "mean": np.asarray(model.mean, dtype=np.float64).tolist(),
        "std": np.asarray(model.std, dtype=np.float64).tolist(),
        "weights": np.asarray(model.w, dtype=np.float64).tolist(),
        "bias": float(model.b), "n_labels": int(model.n_labels),
        "calibration": model.calibration,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _base_sha256() -> str:
    base = _compatible_base()
    if base is None:
        raise ValueError("compatible base model is required")
    return _ranker_identity(base)


def _user_gate_status(model) -> tuple[bool, str]:
    """Whether the exact personal model earned the deck-level release gate."""
    if os.path.exists(USER_PROMOTION_JOURNAL_PATH):
        return False, "promotion_recovery_pending"
    if not os.path.exists(USER_GATE_PATH):
        return False, "missing_gate"
    try:
        with open(USER_GATE_PATH, "r", encoding="utf-8") as handle:
            gate = json.load(handle)
        if int(gate.get("version", -1)) != 2:
            return False, "legacy_gate_retired"
        if not gate.get("passed"):
            return False, "gate_failed"
        if gate.get("model_sha256") != _model_sha256(USER_PATH):
            return False, "model_changed_after_gate"
        if int(gate.get("feature_version", -1)) != int(model.version):
            return False, "feature_version_changed_after_gate"
        if int(gate.get("model_label_count", -1)) != int(model.n_labels):
            return False, "label_count_changed_after_gate"
        if gate.get("personalization_profile") != PERSONALIZATION_PROFILE:
            return False, "profile_changed_after_gate"
        if gate.get("base_model_sha256") != _base_sha256():
            return False, "base_changed_after_gate"
        if float(gate.get("recommended_weight", -1.0)) not in (0.10, 0.20, 0.25):
            return False, "invalid_personal_weight"
        report_path = str(gate.get("report_path") or "")
        if not os.path.isfile(report_path):
            return False, "promotion_report_missing"
        if gate.get("report_sha256") != _model_sha256(report_path):
            return False, "promotion_report_changed"
        manifest_sha = str(gate.get("challenger_manifest_sha256") or "")
        if not re.fullmatch(r"[a-f0-9]{64}", manifest_sha):
            return False, "invalid_challenger_manifest_hash"
    except Exception:  # noqa: BLE001 - malformed approval must fail closed
        return False, "invalid_gate"
    return True, "passed"


def invalidate_user_ranker_gate() -> bool:
    """Remove a stale approval whenever feedback produces a new model."""
    if not os.path.exists(USER_GATE_PATH):
        return False
    os.remove(USER_GATE_PATH)
    return True


def approve_user_ranker(report_path: str, gate_summary: dict) -> dict:
    """Legacy entry point deliberately disabled in favour of proven promotion.

    A caller-provided ``{passed: true}`` is not evidence.  A report must be
    generated from prospective VODs and bound to the current shadow artifact.
    """
    raise RuntimeError(
        "approve_user_ranker is retired; use promote_user_challenger(report_path)"
    )


def _report_field(report: dict, name: str):
    """Accept the compact v2 report fields or their explicit artifacts block."""
    return report.get(name, (report.get("artifacts") or {}).get(name))


def _verify_prospective_report(
    report: dict,
    *,
    database: str | None = None,
    data_dir: str | None = None,
) -> dict:
    """Reproduce every promotion-relevant report field from current evidence."""
    # Promotion is a developer operation, not a reason to distribute private
    # evaluation tooling with the application. Missing tooling must never
    # downgrade verification to trusting a caller-supplied report.
    try:
        from scripts.eval_personal_adapter_prospective import evaluate
    except ModuleNotFoundError as exc:
        if exc.name not in {"scripts", "scripts.eval_personal_adapter_prospective"}:
            raise
        raise RuntimeError(
            "Ranker promotion requires the separate evaluation tooling; "
            "this application distribution cannot verify promotion reports."
        ) from exc

    verified_database = os.path.abspath(
        database or os.path.join(get_data_dir(), "recall.db")
    )
    verified_data_dir = os.path.abspath(data_dir or get_data_dir())
    if os.path.abspath(str(report.get("database") or "")) != verified_database:
        raise ValueError("prospective report database does not match this install")
    if os.path.abspath(str(report.get("data_dir") or "")) != verified_data_dir:
        raise ValueError("prospective report data directory does not match this install")
    targets = report.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("prospective report has no target jobs")
    reproduced = evaluate(
        [str(job_id) for job_id in targets],
        verified_database,
        data_dir=verified_data_dir,
    )
    contract_fields = (
        "version", "measurement_only", "database", "data_dir",
        "base_model_sha256", "challenger_model_sha256", "targets",
        "feature_version", "model_label_count", "profile",
        "challenger_manifest_sha256", "lanes", "gate",
    )
    for field in contract_fields:
        if report.get(field) != reproduced.get(field):
            raise ValueError(f"prospective report is stale or unverifiable: {field}")
    return reproduced


def _training_decisions_still_current(manifest: dict, database: str) -> bool:
    """An archived model is invalid if any supervision it learned was undone."""
    from core.database import DatabaseManager

    db = DatabaseManager(database)
    current = {
        str(row["clip_id"]): row
        for row in db.get_training_data(events=("saved", "passed"))
    }
    for expected in manifest.get("training_decisions") or []:
        row = current.get(str(expected["clip_id"]))
        if row is None:
            return False
        if (
            int(row.get("id", -1)) != int(expected["id"])
            or str(row.get("job_id") or "") != str(expected["job_id"])
            or str(row.get("event") or "") != str(expected["event"])
            or float(row.get("label", -1)) != float(expected["label"])
        ):
            return False
    return True


@_serialized_training
def promote_user_challenger(
    report_path: str,
    *,
    database: str | None = None,
    data_dir: str | None = None,
) -> dict:
    """Explicitly promote one measured challenger; this function is never automatic."""
    if _personalization_frozen():
        raise ValueError("personalization is frozen")
    if not _recover_pending_promotion():
        raise RuntimeError("an interrupted personal-ranker promotion needs recovery")
    if not os.path.isfile(report_path):
        raise FileNotFoundError(report_path)
    report_bytes = open(report_path, "rb").read()
    try:
        report = json.loads(report_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError("invalid prospective report") from exc
    report = _verify_prospective_report(
        report, database=database, data_dir=data_dir,
    )
    if int(report.get("version", -1)) != 2 or report.get("measurement_only") is not True:
        raise ValueError("report must be a measurement-only v2 report")
    gate = report.get("gate") or {}
    reasons = gate.get("reasons") or []
    if gate.get("passed") is not True or reasons:
        raise ValueError("report gate did not pass cleanly")
    targets = report.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("report requires prospective targets")
    source_keys = set()
    target_payloads = []
    for job_id, target in targets.items():
        if not isinstance(target, dict) or not target.get("source_key") or not target.get("candidate_artifact") or not target.get("candidate_artifact_sha256"):
            raise ValueError("report target provenance is incomplete")
        if int(target.get("decision_count", 0) or 0) <= 0:
            raise ValueError("report target has no explicit Keep/Pass decisions")
        path = str(target["candidate_artifact"])
        if not os.path.isfile(path) or _model_sha256(path) != target["candidate_artifact_sha256"]:
            raise ValueError("report target artifact changed or missing")
        payload = _read_json(path)
        if str(payload.get("job_id") or "") != str(job_id):
            raise ValueError("report target artifact job mismatch")
        target_payloads.append(payload)
        source_keys.add(str(target["source_key"]))
    if len(source_keys) < 2 or int(gate.get("evaluated_source_vods", -1)) != len(source_keys):
        raise ValueError("report requires at least two matching prospective VODs")
    requested_sha = report.get("challenger_model_sha256")
    if not isinstance(requested_sha, str):
        raise ValueError("report is missing challenger_sha256")
    challenger_path, challenger_meta_path, manifest = _resolve_challenger_generation(requested_sha)
    if challenger_path is None or manifest is None:
        raise ValueError("unknown or invalid challenger generation")
    challenger = _load_compatible(challenger_path)
    if challenger is None:
        raise ValueError("challenger generation is incompatible")
    verified_database = os.path.abspath(
        database or os.path.join(get_data_dir(), "recall.db")
    )
    if not _training_decisions_still_current(manifest, verified_database):
        raise ValueError("challenger training feedback changed after training")
    expected = {
        "challenger_model_sha256": _model_sha256(challenger_path),
        "base_model_sha256": _base_sha256(),
        "feature_version": int(feat.FEATURE_VERSION),
        "model_label_count": int(challenger.n_labels),
    }
    for name, value in expected.items():
        if _report_field(report, name) != value:
            raise ValueError(f"stale or mismatched report: {name}")
    if report.get("profile") != PERSONALIZATION_PROFILE:
        raise ValueError("stale or mismatched report: profile")
    # If the evaluator supplies the provenance hash, bind it too. A missing
    # value remains backwards-compatible with the first v2 evaluator, while
    # artifact/version/model fields above remain mandatory.
    manifest_sha = hashlib.sha256(
        open(challenger_meta_path, "rb").read()
    ).hexdigest()
    supplied_manifest_sha = report.get("challenger_manifest_sha256")
    if supplied_manifest_sha != manifest_sha:
        raise ValueError("stale or mismatched report: challenger_manifest_sha256")
    weights = gate.get("eligible_weights") or []
    recommended_weight = gate.get("recommended_weight")
    if (
        not isinstance(weights, list)
        or not weights
        or any(float(weight) not in (0.10, 0.20, 0.25) for weight in weights)
        or recommended_weight not in weights
        or float(recommended_weight) != min(float(weight) for weight in weights)
    ):
        raise ValueError("report recommended weight is not eligible")

    expected_metadata = {
        "profile": PERSONALIZATION_PROFILE,
        "max_label_id": manifest.get("max_label_id"),
        "training_job_ids": list(manifest.get("training_job_ids") or []),
        "training_source_keys": list(manifest.get("training_source_keys") or []),
        "feature_version": int(manifest.get("feature_version", -1)),
        "model_label_count": int(manifest.get("label_count", -1)),
        "manifest_sha256": manifest_sha,
        "base_model_sha256": manifest.get("base_sha256"),
        "challenger_model_sha256": manifest.get("model_sha256"),
    }
    for payload in target_payloads:
        diagnostics = payload.get("diagnostics") or {}
        if (
            diagnostics.get("base_model_sha256") != expected["base_model_sha256"]
            or diagnostics.get("challenger_model_sha256")
            != expected["challenger_model_sha256"]
            or diagnostics.get("challenger_manifest_sha256") != manifest_sha
            or diagnostics.get("challenger_metadata") != expected_metadata
        ):
            raise ValueError("report target challenger provenance mismatch")
        settings = diagnostics.get("selection_settings") or {}
        if (
            settings.get("active_ranker_mode") != "base"
            or settings.get("ranker_policy") != "blend"
            or float(settings.get("ranker_blend_weight", -1)) != 0.35
            or float(settings.get("personal_ranker_blend_weight", -1)) != 0.0
        ):
            raise ValueError("report target was not measured from the Base blend lane")
    quality_reasons = gate.get("quality_reasons")
    if not isinstance(quality_reasons, list) or quality_reasons:
        raise ValueError("report quality gate did not pass cleanly")
    baseline = (report.get("lanes") or {}).get("baseline")
    lane_name = f"challenger_w{int(float(recommended_weight) * 100):02d}"
    chosen_lane = (report.get("lanes") or {}).get(lane_name)
    deltas = (gate.get("metric_deltas") or {}).get(lane_name)
    if not isinstance(baseline, dict) or not isinstance(chosen_lane, dict):
        raise ValueError("report is missing the measured baseline/challenger lanes")
    if not isinstance(deltas, dict):
        raise ValueError("report is missing recommended-lane metric deltas")
    chosen_aggregate = chosen_lane.get("aggregate") or {}
    if int(chosen_aggregate.get("unknown_selected_count", -1)) != 0:
        raise ValueError("recommended lane contains unreviewed Primary clips")
    metric_names = (
        "known_precision", "top5_known_precision", "reviewed_keep_recall",
    )
    metric_values = [deltas.get(name) for name in metric_names]
    if any(value is None or float(value) < 0 for value in metric_values):
        raise ValueError("recommended lane regresses a required quality metric")
    strict_metric_gain = any(float(value) > 0 for value in metric_values)
    strict_membership_gain = (
        int(deltas.get("added_kept_membership", 0))
        > int(deltas.get("removed_kept_membership", 0))
        and int(deltas.get("added_pass_membership", 0)) == 0
    )
    if not strict_metric_gain and not strict_membership_gain:
        raise ValueError("recommended lane has no strict reviewed quality gain")

    challenger_bytes = open(challenger_path, "rb").read()
    # Validate copied bytes before any champion replacement.
    challenger_sha = hashlib.sha256(challenger_bytes).hexdigest()
    if challenger_sha != expected["challenger_model_sha256"]:
        raise ValueError("challenger changed during promotion")
    old_champion = open(USER_PATH, "rb").read() if os.path.exists(USER_PATH) else None
    old_gate = open(USER_GATE_PATH, "rb").read() if os.path.exists(USER_GATE_PATH) else None
    gate_payload = {
        "version": 2,
        "passed": True,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": challenger_sha,
        "feature_version": int(feat.FEATURE_VERSION),
        "model_label_count": int(challenger.n_labels),
        "report_path": os.path.abspath(report_path),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "challenger_manifest_sha256": manifest_sha,
        "personalization_profile": PERSONALIZATION_PROFILE,
        "base_model_sha256": _base_sha256(),
        "recommended_weight": recommended_weight,
        "gate": gate,
    }
    gate_bytes = json.dumps(
        gate_payload, indent=2, sort_keys=True
    ).encode("utf-8")
    transaction_paths = _promotion_transaction_paths()
    try:
        # Durable old/new snapshots precede the journal. A crash before the
        # journal leaves active state untouched; a crash after it is recoverable.
        for name, payload in (
            ("old_model", old_champion),
            ("old_gate", old_gate),
            ("new_model", challenger_bytes),
            ("new_gate", gate_bytes),
        ):
            if payload is None:
                _remove_file(transaction_paths[name])
            else:
                _atomic_write_bytes(payload, transaction_paths[name])
        journal = {
            "version": 1,
            "personalization_profile": PERSONALIZATION_PROFILE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "old_model_sha256": (
                hashlib.sha256(old_champion).hexdigest()
                if old_champion is not None else None
            ),
            "old_gate_sha256": (
                hashlib.sha256(old_gate).hexdigest()
                if old_gate is not None else None
            ),
            "new_model_sha256": challenger_sha,
            "new_gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        }
        _atomic_write_json(journal, USER_PROMOTION_JOURNAL_PATH)

        # Gate first: until the model switch, the hash mismatch forces Base.
        os.replace(transaction_paths["new_gate"], USER_GATE_PATH)
        os.replace(transaction_paths["new_model"], USER_PATH)
        if not _recover_pending_promotion():
            raise RuntimeError("promoted pair could not be finalized")
    except Exception:
        if os.path.exists(USER_PROMOTION_JOURNAL_PATH):
            _recover_pending_promotion()
        raise
    return {"promoted": True, "role": "champion", "model_sha256": challenger_sha,
            "report_path": os.path.abspath(report_path), "report_sha256": gate_payload["report_sha256"],
            "backup_created": os.path.exists(USER_CHAMPION_BACKUP_PATH)}


def resolve_active_ranker():
    """Return ``(model, diagnostics)`` for the exact runtime resolution.

    A plain ``None`` used to make three materially different states look the
    same in a completed scan: no model existed, a model was incompatible, or
    an eval freeze deliberately suppressed the personal model.  The selector
    still receives the same model/None contract, while the diagnostics ride in
    the private candidate artifact so a bad deck can be audited after the run.
    """
    with _TRAINING_LOCK:
        promotion_recovered = _recover_pending_promotion()
    frozen = _personalization_frozen()
    candidates = (("base", BASE_PATH),) if frozen else (
        ("champion", USER_PATH),
        ("challenger", USER_CHALLENGER_PATH),
        ("base", BASE_PATH),
    )
    attempts = []
    personal_model = None
    challenger_model = None
    for mode, path in candidates:
        # Sealed-aware existence: in a packaged build the bundled base exists
        # only as its `.sealed` sibling, and a bare os.path.exists here would
        # record "missing" and fall through to the signal path on every shipped
        # install. `sealed` rides along so the artifact says which form loaded.
        sealed_only = not os.path.exists(path) and os.path.exists(sealed_path_for(path))
        attempt = {
            "mode": mode,
            "path": os.path.abspath(path),
            "exists": protected_json_exists(path),
            "sealed": sealed_only,
        }
        attempts.append(attempt)
        if not attempt["exists"]:
            attempt["result"] = "missing"
            continue
        try:
            model = rk.LogisticRanker.load(path)
        except Exception as exc:  # noqa: BLE001 - resolution must fall back
            attempt["result"] = "load_error"
            attempt["error"] = str(exc)[:300]
            continue
        if model.version != feat.FEATURE_VERSION:
            attempt["result"] = "feature_version_mismatch"
            attempt["model_feature_version"] = model.version
            continue
        if model.feature_names != feat.FEATURE_NAMES:
            attempt["result"] = "feature_schema_mismatch"
            continue
        attempt["model_feature_version"] = model.version
        attempt["model_label_count"] = model.n_labels
        attempt["model_sha256"] = (
            _ranker_identity(model) if mode == "base" else _model_sha256(path)
        )
        if mode == "champion":
            approved, gate_reason = _user_gate_status(model)
            attempt["gate"] = gate_reason
            if not approved:
                attempt["result"] = "shadow_unvalidated"
                continue
            # A personal model is a preference ADAPTER, never the universal
            # scorer. Hold it until the compatible base is resolved below.
            personal_model = model
            attempt["result"] = "validated_adapter"
            continue
        if mode == "challenger":
            valid, manifest_reason, _manifest = _challenger_manifest_status(model)
            attempt["manifest"] = manifest_reason
            if not valid:
                attempt["result"] = "shadow_invalid_provenance"
                continue
            challenger_model = model
            attempt["result"] = "shadow_challenger"
            continue
        attempt["result"] = "active"
        stack = RankerStack(model, personal_model, challenger_model)
        active_mode = "personal" if personal_model is not None else "base"
        challenger_manifest = None
        challenger_manifest_sha = None
        if challenger_model is not None:
            _valid, _reason, challenger_manifest = _challenger_manifest_status(
                challenger_model
            )
            challenger_manifest_sha = hashlib.sha256(
                open(USER_CHALLENGER_META_PATH, "rb").read()
            ).hexdigest()
        return stack, {
            "active_mode": active_mode,
            "model_loaded": True,
            "model_path": os.path.abspath(path),
            "model_feature_version": model.version,
            "model_label_count": stack.n_labels,
            "base_model_label_count": model.n_labels,
            "base_model_sha256": _ranker_identity(model),
            "personal_model_loaded": personal_model is not None,
            "personal_model_path": (
                os.path.abspath(USER_PATH) if personal_model is not None else None
            ),
            "personal_model_label_count": (
                personal_model.n_labels if personal_model is not None else 0
            ),
            "challenger_model_loaded": challenger_model is not None,
            "challenger_model_path": (
                os.path.abspath(USER_CHALLENGER_PATH)
                if challenger_model is not None else None
            ),
            "challenger_model_label_count": (
                challenger_model.n_labels if challenger_model is not None else 0
            ),
            "challenger_model_sha256": (
                _model_sha256(USER_CHALLENGER_PATH)
                if challenger_model is not None else None
            ),
            "challenger_manifest_sha256": (
                challenger_manifest_sha
            ),
            "challenger_metadata": (
                {
                    "profile": challenger_manifest.get("personalization_profile"),
                    "max_label_id": challenger_manifest.get("max_label_id"),
                    "training_job_ids": list(
                        challenger_manifest.get("training_job_ids") or []
                    ),
                    "training_source_keys": list(
                        challenger_manifest.get("training_source_keys") or []
                    ),
                    "feature_version": challenger_manifest.get("feature_version"),
                    "model_label_count": challenger_manifest.get("label_count"),
                    "manifest_sha256": challenger_manifest_sha,
                    "base_model_sha256": challenger_manifest.get("base_sha256"),
                    "challenger_model_sha256": challenger_manifest.get(
                        "model_sha256"
                    ),
                }
                if challenger_manifest is not None else None
            ),
            "ranker_contract": "base_plus_personal_adapter",
            "personalization_profile": PERSONALIZATION_PROFILE,
            "personalization_frozen": frozen,
            "promotion_recovery_pending": not promotion_recovered,
            "attempts": attempts,
        }

    shadow_attempt = next((
        attempt for attempt in attempts
        if attempt.get("result") == "shadow_unvalidated"
    ), None)
    return None, {
        "active_mode": "signals",
        "model_loaded": False,
        "model_path": None,
        "model_feature_version": feat.FEATURE_VERSION,
        "model_label_count": int(
            (shadow_attempt or {}).get("model_label_count", 0) or 0
        ),
        "personalization_frozen": frozen,
        "personalization_profile": PERSONALIZATION_PROFILE,
        "promotion_recovery_pending": not promotion_recovered,
        "ranker_contract": "base_plus_personal_adapter",
        "reason": (
            "personalization_frozen" if frozen
            else "personal_adapter_requires_base" if personal_model is not None
            else "personal_challenger_requires_base" if challenger_model is not None
            else "personal_model_shadow_only" if shadow_attempt
            else "no_compatible_model"
        ),
        "attempts": attempts,
    }


def get_active_ranker():
    """Base-quality stack with an optional validated personal adapter."""
    model, _diagnostics = resolve_active_ranker()
    return model


def ranker_status(db):
    with _TRAINING_LOCK:
        promotion_recovered = _recover_pending_promotion()
    data = _training_rows(db)
    pair_count = _pair_count(data)
    has_user = False
    user_validated = False
    user_gate_reason = "missing_model"
    champion_model = _load_compatible(USER_PATH)
    if champion_model is not None:
        has_user = True
        user_validated, user_gate_reason = _user_gate_status(champion_model)
    elif os.path.exists(USER_PATH):
        user_gate_reason = "incompatible_model"
    challenger_model = _load_compatible(USER_CHALLENGER_PATH)
    challenger_valid, challenger_manifest_reason, challenger_manifest = _challenger_manifest_status(challenger_model)

    has_base = _compatible_base() is not None

    frozen = _personalization_frozen()
    _active_model, resolution = resolve_active_ranker()
    active_mode = str(resolution.get("active_mode", "signals"))
    shadow_only = bool(
        (has_user and not user_validated)
        or (challenger_model is not None and challenger_valid)
    )
    last_trained_at = None
    last_trained_path = (
        USER_CHALLENGER_PATH
        if challenger_model is not None and challenger_valid
        else USER_PATH
    )
    if (challenger_model is not None and challenger_valid) or has_user:
        try:
            last_trained_at = datetime.fromtimestamp(
                os.path.getmtime(last_trained_path), timezone.utc
            ).isoformat()
        except Exception:
            last_trained_at = None
    return {
        "labels": len(data),
        "label_count": len(data),
        "min_labels": MIN_LABELS,
        "min_pairs": MIN_PAIRS,
        "preference_pairs": pair_count,
        "job_count": _job_count(data),
        "positives": sum(float(row["label"]) == 1.0 for row in data),
        "maybes": sum(float(row["label"]) == 0.5 for row in data),
        "negatives": sum(float(row["label"]) == 0.0 for row in data),
        "can_train": len(data) >= MIN_LABELS and pair_count >= MIN_PAIRS,
        "active_mode": active_mode,
        "state": _learning_state(
            len(data), pair_count, active_mode, shadow_only=shadow_only,
        ),
        "last_trained_at": last_trained_at,
        "has_user_model": has_user,
        "champion": {
            "path": os.path.abspath(USER_PATH),
            "present": os.path.exists(USER_PATH),
            "compatible": has_user,
            "validated": user_validated,
            "gate_reason": user_gate_reason,
            "label_count": champion_model.n_labels if champion_model else 0,
            "state": (
                "validated" if user_validated
                else "shadow" if has_user
                else "invalid" if os.path.exists(USER_PATH)
                else "missing"
            ),
        },
        "challenger": {
            "path": os.path.abspath(USER_CHALLENGER_PATH),
            "present": os.path.exists(USER_CHALLENGER_PATH),
            "compatible": challenger_model is not None,
            "manifest_valid": challenger_valid,
            "manifest_reason": challenger_manifest_reason,
            "model_sha256": (
                _model_sha256(USER_CHALLENGER_PATH)
                if challenger_model is not None and challenger_valid else None
            ),
            "base_sha256": (
                (challenger_manifest or {}).get("base_sha256")
                if challenger_valid else None
            ),
            "generation_count": (
                len(os.listdir(USER_CHALLENGER_GENERATIONS_DIR)) // 2
                if os.path.isdir(USER_CHALLENGER_GENERATIONS_DIR) else 0
            ),
            "label_count": challenger_model.n_labels if challenger_model else 0,
            "state": (
                "shadow" if challenger_model is not None and challenger_valid
                else "invalid" if challenger_model is not None
                else "invalid" if os.path.exists(USER_CHALLENGER_PATH)
                else "missing"
            ),
        },
        "has_challenger": challenger_model is not None and challenger_valid,
        "personal_model_validated": user_validated,
        "personal_model_shadow_only": shadow_only,
        "personal_model_gate_reason": user_gate_reason,
        "has_base_model": has_base,
        "feature_version": feat.FEATURE_VERSION,
        "personalization_profile": PERSONALIZATION_PROFILE,
        "creator_message": _creator_message(
            len(data), pair_count, active_mode, shadow_only=shadow_only,
        ),
        "frozen": frozen,
        "promotion_recovery_pending": (
            not promotion_recovered
            or os.path.exists(USER_PROMOTION_JOURNAL_PATH)
        ),
        "resolution": resolution,
    }


def _creator_message(
    label_count: int,
    pair_count: int,
    active_mode: str,
    *,
    shadow_only: bool = False,
) -> str:
    if active_mode == "personal":
        return f"Recall is using what it learned from {label_count} decisions."
    if shadow_only:
        return (
            f"Recall has learned from {label_count} decisions and is validating "
            "them before changing your clips."
        )
    remaining = max(0, MIN_LABELS - label_count)
    if remaining:
        return f"Recall is learning from your picks. {remaining} more decisions unlock personal ranking."
    if pair_count < MIN_PAIRS:
        return "Recall needs a mix of Keeps and Passes from the same session to personalize ranking."
    return "Recall has enough decisions to tune personal ranking."


def _learning_state(
    label_count: int,
    pair_count: int,
    active_mode: str,
    *,
    shadow_only: bool = False,
) -> str:
    if active_mode == "personal":
        return "personal_active"
    if shadow_only:
        # Reuse the existing creator-facing state/badge while the richer
        # creator_message explains that deck validation is still pending.
        return "ready_to_personalize"
    if label_count >= MIN_LABELS and pair_count >= MIN_PAIRS:
        return "ready_to_personalize"
    if label_count > 0:
        return "learning"
    return "not_started"


@_serialized_training
def reset_learning(db):
    """Remove per-install personalization while keeping clips and sessions."""
    removed_model = False
    if os.path.exists(USER_PATH):
        os.remove(USER_PATH)
        removed_model = True
    removed_challenger = False
    if os.path.exists(USER_CHALLENGER_PATH):
        os.remove(USER_CHALLENGER_PATH)
        removed_challenger = True
    for path in (
        USER_CHALLENGER_META_PATH, USER_CHAMPION_BACKUP_PATH, USER_GATE_BACKUP_PATH,
        USER_PROMOTION_JOURNAL_PATH, *_promotion_transaction_paths().values(),
    ):
        if os.path.exists(path):
            os.remove(path)
    if os.path.isdir(USER_CHALLENGER_GENERATIONS_DIR):
        for name in os.listdir(USER_CHALLENGER_GENERATIONS_DIR):
            path = os.path.join(USER_CHALLENGER_GENERATIONS_DIR, name)
            if os.path.isfile(path):
                os.remove(path)
        try:
            os.rmdir(USER_CHALLENGER_GENERATIONS_DIR)
        except OSError:
            pass
    invalidate_user_ranker_gate()
    db.clear_learning_data()
    status = ranker_status(db)
    return {
        "reset": True,
        "removed_personal_picks": removed_model,
        "removed_challenger": removed_challenger,
        **status,
    }


@_serialized_training
def sync_user_ranker(db):
    """Make the personal model exactly reflect the current usable feedback.

    Feedback can be retracted or clips can be deleted.  Keeping the previously
    trained file in those cases would silently rank with decisions the creator
    has undone.  Invalidate it first, then either rebuild from the current
    review set or fall back to the deterministic signal/base policy.
    """
    # Refuse BEFORE the stale-model removal: a freeze must leave every file
    # exactly as it was, or unfreezing would not restore the prior behavior.
    if _personalization_frozen():
        return {
            "trained": False,
            "reason": "personalization_frozen",
            "removed_stale_model": False,
            **ranker_status(db),
        }
    status = ranker_status(db)
    if status["can_train"]:
        # train_user_ranker writes a temporary file and atomically replaces the
        # existing challenger. Keep the previous complete challenger readable
        # until that replacement succeeds; a failed background train must not
        # leave subsequent scans with a missing/partial shadow model.
        return {
            **train_user_ranker(db),
            "removed_stale_model": False,
        }
    removed_stale_model = False
    if os.path.exists(USER_CHALLENGER_PATH):
        os.remove(USER_CHALLENGER_PATH)
        removed_stale_model = True
    if os.path.exists(USER_CHALLENGER_META_PATH):
        os.remove(USER_CHALLENGER_META_PATH)
    return {
        "trained": False,
        "reason": "not_enough_current_feedback",
        "removed_stale_model": removed_stale_model,
        **ranker_status(db),
    }


@_serialized_training
def train_user_ranker(db, min_labels: int = MIN_LABELS, anchor: float = 2.0):
    """Train absolute Keep/Pass ranking from every compatible decision.

    The old pairwise-only objective ignored an entire VOD when every surfaced
    clip was Passed. Those are the strongest possible "this deck was junk"
    sessions, so feature-v5's VOD-context inputs now support pointwise training
    across sessions while the activation gate still requires preference pairs.
    """
    if _personalization_frozen():
        return {
            "trained": False,
            "reason": "personalization_frozen",
            **ranker_status(db),
        }
    data = _training_rows(db)
    pair_count = _pair_count(data)
    if len(data) < min_labels:
        return {"trained": False, "reason": "not_enough_labels",
                "labels": len(data), "min_labels": min_labels}

    if pair_count < MIN_PAIRS:
        return {"trained": False, "reason": "not_enough_preferences",
                "labels": len(data), "pairs": pair_count, "min_pairs": MIN_PAIRS}

    X, y, weights = [], [], []
    rank_idx = feat.FEATURE_NAMES.index("presentation_rank")
    for row in data:
        fd = row["features"]
        vec = [float(fd.get(name, 0.0)) for name in feat.FEATURE_NAMES]
        # Position-bias debiasing (plan 22 §4.3): the review deck deals cards
        # confidence-first, so early positions soak up keep decisions
        # regardless of content. Feed the recorded rank at TRAIN time only —
        # inference always passes 0 — so the rank weight absorbs the position
        # effect instead of the content features taking credit for it.
        rank = row.get("rank")
        if rank:
            vec[rank_idx] = min(1.0, float(rank) / 20.0)
        X.append(vec)
        y.append(float(row["label"]))
        weights.append(float(row.get("weight", 1.0)))
    X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

    if y.min() == y.max():
        return {"trained": False, "reason": "single_class",
                "labels": len(data), "kept": int((y == 1.0).sum())}

    base = _compatible_base()
    if base is None:
        return {
            "trained": False,
            "reason": "base_model_required",
            "labels": len(data),
            "pairs": pair_count,
        }
    model = rk.train(
        X, y, feat.FEATURE_NAMES, feat.FEATURE_VERSION,
        sample_weight=np.asarray(weights, dtype=np.float64),
        base=base, anchor=anchor,
    )
    trained_at = datetime.now(timezone.utc).isoformat()
    # Archive the prior complete generation before replacing current shadow.
    # Invalid/orphaned pairs are intentionally never retained as evidence.
    _archive_current_challenger()
    manifest = {
        "version": 1,
        "role": "challenger",
        "base_sha256": _base_sha256(),
        "feature_version": int(feat.FEATURE_VERSION),
        "feature_schema_sha256": _feature_schema_sha256(),
        "label_count": int(len(data)),
        "max_label_id": max(int(row.get("id", 0)) for row in data),
        "training_job_ids": sorted({str(row["job_id"]) for row in data}),
        "training_job_count": _job_count(data),
        "training_source_keys": sorted({
            canonical_source_key(
                str(row.get("source_path") or ""),
                str(row.get("asset_path") or ""),
            )
            for row in data
        }),
        "training_decisions": [
            {
                "id": int(row.get("id", 0)),
                "clip_id": str(row["clip_id"]),
                "job_id": str(row["job_id"]),
                "label": float(row["label"]),
                "event": str(row["event"]),
            }
            for row in sorted(data, key=lambda item: int(item.get("id", 0)))
        ],
        "training_events": ["saved", "passed"],
        "trained_at": trained_at,
        "personalization_profile": PERSONALIZATION_PROFILE,
    }
    # A candidate is never allowed to overwrite the approved champion. It is
    # written atomically so shadow scoring can never observe a partial JSON.
    _atomic_save_challenger(model, manifest)
    try:
        db.set_ranker_metadata("last_trained_at", trained_at)
    except Exception:
        pass
    return {"trained": True, "role": "challenger", "artifact_state": "shadow",
            "challenger_path": os.path.abspath(USER_CHALLENGER_PATH),
            "labels": len(data), "label_count": len(data),
            "kept": int((y == 1.0).sum()),
            "maybe": int((y == 0.5).sum()),
            "rejected": int((y == 0.0).sum()),
            "preference_pairs": pair_count,
            "last_trained_at": trained_at}

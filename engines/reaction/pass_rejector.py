# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""High-precision negative classifier primitives for clip publishability.

This module is intentionally independent from the production selector.  It
supports shadow evaluation of a pointwise Keep-probability model and chooses a
rejection threshold under an explicit Keep-loss budget.  A model that does not
clear the ship gate remains incapable of changing creator-facing decks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from core.bundle_paths import get_models_dir
from core.model_vault import load_protected_json, protected_json_exists
from engines.reaction import features as features_mod


logger = logging.getLogger(__name__)


SHIP_MIN_PASS_REMOVAL_RATE = 0.30
SHIP_MAX_KEEP_LOSS_RATE = 0.02
SHIP_MIN_SOURCE_VODS = 12
# Second Look gate, retuned 2026-08-04 against the measured OOF frontier on 387
# rows / 33 source VODs.  The previous 0.20 removal bar was inherited from the
# pre-1afaee5 (61-feature, visual_judged) layout and is not reachable on the
# current feature set: at a 0.03 Keep-loss budget the best attainable removal is
# ~0.105, and clearing 0.20 costs ~0.05 Keep loss.  Rather than pay that, the
# Keep-loss cap stays at 0.03 and the removal bar drops to what the data can
# actually support.  The 0.04 -> 0.05 budget region is a cliff (removal jumps
# 12.6% -> 24.1% for one Keep), so the operating point is deliberately kept
# below it.
SECOND_LOOK_MIN_PASS_REMOVAL_RATE = 0.10
SECOND_LOOK_MAX_KEEP_LOSS_RATE = 0.03
SECOND_LOOK_MIN_SOURCE_VODS = 12
SECOND_LOOK_MODEL_FILENAME = "second_look_rejector.json"
SECOND_LOOK_ENSEMBLE_SEEDS = (
    11,
    101,
    1001,
    10001,
    20260725,
    314159,
    271828,
    161803,
)
TRAJECTORY_FEATURE_NAMES = [
    f"{channel}_{name}"
    for channel in ("audio_spike", "audio_amplitude", "motion")
    for name in (
        "mean",
        "stddev",
        "maximum",
        "opening_third_mean",
        "middle_third_mean",
        "closing_third_mean",
        "closing_minus_opening",
        "closing_to_max_third",
        "peak_position",
        "top_quintile_fraction",
    )
]
SECOND_LOOK_FEATURE_NAMES = (
    list(features_mod.FEATURE_NAMES)
    + TRAJECTORY_FEATURE_NAMES
)


@dataclass(frozen=True)
class PointwiseModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def predict_keep_probability(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        normalized = (values - self.mean) / self.scale
        logits = normalized @ self.weights + self.bias
        return _sigmoid(logits)


@dataclass(frozen=True)
class ShallowModel:
    mean: np.ndarray
    scale: np.ndarray
    input_weights: np.ndarray
    input_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float

    def predict_keep_probability(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        normalized = (values - self.mean) / self.scale
        hidden = np.tanh(normalized @ self.input_weights + self.input_bias)
        logits = hidden @ self.output_weights + self.output_bias
        return _sigmoid(logits)


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    pass_total: int
    passes_removed: int
    pass_removal_rate: float
    keep_total: int
    keeps_lost: int
    keep_loss_rate: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class SecondLookRejectorModel:
    pointwise: PointwiseModel | None
    ensemble: Tuple[ShallowModel, ...]
    threshold: float
    calibration_scale: float
    calibration_intercept: float
    metadata: Dict

    def predict_keep_probability(self, candidate: dict) -> float | None:
        trajectory = candidate.get("second_look_trajectory_features")
        if not isinstance(trajectory, (list, tuple)):
            return None
        if len(trajectory) != len(TRAJECTORY_FEATURE_NAMES):
            return None
        vector = np.concatenate(
            [
                features_mod.feature_vector(candidate),
                np.asarray(trajectory, dtype=np.float64),
            ]
        )
        if self.ensemble:
            member_probabilities = np.asarray(
                [
                    model.predict_keep_probability(vector[None, :])[0]
                    for model in self.ensemble
                ],
                dtype=np.float64,
            )
            member_probabilities = np.clip(
                member_probabilities, 1e-9, 1.0 - 1e-9,
            )
            member_logits = np.log(
                member_probabilities / (1.0 - member_probabilities)
            )
            raw_probability = float(
                _sigmoid(np.asarray([member_logits.mean()]))[0]
            )
        elif self.pointwise is not None:
            raw_probability = float(
                self.pointwise.predict_keep_probability(vector[None, :])[0]
            )
        else:
            return None
        raw_probability = min(1.0 - 1e-9, max(1e-9, raw_probability))
        raw_logit = np.log(raw_probability / (1.0 - raw_probability))
        calibrated_logit = (
            self.calibration_scale * raw_logit + self.calibration_intercept
        )
        return float(_sigmoid(np.asarray([calibrated_logit]))[0])

    def rejects(self, candidate: dict) -> bool:
        probability = self.predict_keep_probability(candidate)
        if probability is None:
            return False
        candidate["second_look_keep_probability"] = probability
        candidate["second_look_reject_threshold"] = float(self.threshold)
        return probability < float(self.threshold)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def fit_pointwise_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    steps: int = 900,
    learning_rate: float = 0.08,
    l2: float = 0.10,
) -> PointwiseModel:
    """Fit a deterministic weighted logistic Keep-probability model."""
    values = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(targets) or len(values) == 0:
        raise ValueError("matrix and labels must contain the same non-zero rows")
    weights = (
        np.ones(len(targets), dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if len(weights) != len(targets):
        raise ValueError("sample_weights must match labels")

    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (values - mean) / scale

    coefficients = np.zeros(values.shape[1], dtype=np.float64)
    positive_rate = float(np.average(targets, weights=weights))
    positive_rate = min(1.0 - 1e-4, max(1e-4, positive_rate))
    bias = float(np.log(positive_rate / (1.0 - positive_rate)))
    weight_sum = max(1e-9, float(weights.sum()))

    for _ in range(max(1, int(steps))):
        probabilities = _sigmoid(normalized @ coefficients + bias)
        errors = (probabilities - targets) * weights
        gradient = normalized.T @ errors / weight_sum + float(l2) * coefficients
        bias_gradient = float(errors.sum() / weight_sum)
        coefficients -= float(learning_rate) * gradient
        bias -= float(learning_rate) * bias_gradient

    return PointwiseModel(
        mean=mean,
        scale=scale,
        weights=coefficients,
        bias=bias,
    )


def fit_shallow_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    seed: int,
    hidden_units: int = 16,
    steps: int = 500,
    learning_rate: float = 0.01,
    weight_decay: float = 0.10,
) -> ShallowModel:
    """Fit the tiny tanh challenger used by the Second Look ensemble."""
    import torch

    values = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    weights = (
        np.ones(len(targets), dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if values.ndim != 2 or len(values) != len(targets) or len(values) == 0:
        raise ValueError("matrix and labels must contain the same non-zero rows")
    if len(weights) != len(targets):
        raise ValueError("sample_weights must match labels")

    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (values - mean) / scale

    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(values.shape[1], int(hidden_units)),
        torch.nn.Tanh(),
        torch.nn.Linear(int(hidden_units), 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    inputs = torch.tensor(normalized, dtype=torch.float32)
    expected = torch.tensor(targets[:, None], dtype=torch.float32)
    row_weights = torch.tensor(weights[:, None], dtype=torch.float32)
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                expected,
                reduction="none",
            )
            * row_weights
        ).sum() / row_weights.sum()
        loss.backward()
        optimizer.step()

    first = model[0]
    output = model[2]
    return ShallowModel(
        mean=mean,
        scale=scale,
        input_weights=first.weight.detach().cpu().numpy().T.astype(np.float64),
        input_bias=first.bias.detach().cpu().numpy().astype(np.float64),
        output_weights=output.weight.detach().cpu().numpy().reshape(-1).astype(
            np.float64
        ),
        output_bias=float(output.bias.detach().cpu().numpy()[0]),
    )


def ensemble_keep_probability(
    models: Sequence[ShallowModel],
    matrix: np.ndarray,
) -> np.ndarray:
    """Combine members by mean log-odds so one noisy seed cannot dominate."""
    if not models:
        raise ValueError("ensemble must contain at least one model")
    probabilities = np.vstack(
        [model.predict_keep_probability(matrix) for model in models]
    )
    probabilities = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    logits = np.log(probabilities / (1.0 - probabilities))
    return _sigmoid(logits.mean(axis=0))


def source_group_oof_ensemble_probabilities(
    matrix: np.ndarray,
    labels: np.ndarray,
    source_groups: Sequence[str],
    *,
    sample_weights: np.ndarray | None = None,
    seeds: Sequence[int] = SECOND_LOOK_ENSEMBLE_SEEDS,
) -> Tuple[np.ndarray, List[Dict], Dict[str, Tuple[ShallowModel, ...]]]:
    """Leave one source VOD out for the deterministic shallow ensemble."""
    values = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    groups = np.asarray([str(group) for group in source_groups])
    weights = (
        np.ones(len(targets), dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if len(values) != len(targets) or len(groups) != len(targets):
        raise ValueError("matrix, labels, and source_groups must align")
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2:
        raise ValueError("source-grouped validation needs at least two sources")

    predictions = np.zeros(len(targets), dtype=np.float64)
    folds: List[Dict] = []
    model_bank: Dict[str, Tuple[ShallowModel, ...]] = {}
    for fold_index, group in enumerate(unique_groups):
        held_out = groups == group
        training = ~held_out
        members = [
            fit_shallow_model(
                values[training],
                targets[training],
                sample_weights=weights[training],
                seed=int(seed) + fold_index,
            )
            for seed in seeds
        ]
        predictions[held_out] = ensemble_keep_probability(
            members,
            values[held_out],
        )
        model_bank[group] = tuple(members)
        folds.append(
            {
                "source_group": group,
                "training_rows": int(training.sum()),
                "held_out_rows": int(held_out.sum()),
                "held_out_keeps": int(((targets == 1) & held_out).sum()),
                "held_out_passes": int(((targets == 0) & held_out).sum()),
            }
        )
    return predictions, folds, model_bank


def source_group_oof_probabilities(
    matrix: np.ndarray,
    labels: np.ndarray,
    source_groups: Sequence[str],
    *,
    sample_weights: np.ndarray | None = None,
    fit_kwargs: Dict | None = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Leave one source VOD out so rescans cannot leak into validation."""
    values = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    groups = np.asarray([str(group) for group in source_groups])
    weights = (
        np.ones(len(targets), dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if len(values) != len(targets) or len(groups) != len(targets):
        raise ValueError("matrix, labels, and source_groups must align")
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2:
        raise ValueError("source-grouped validation needs at least two sources")

    predictions = np.zeros(len(targets), dtype=np.float64)
    folds: List[Dict] = []
    for group in unique_groups:
        held_out = groups == group
        training = ~held_out
        model = fit_pointwise_model(
            values[training],
            targets[training],
            sample_weights=weights[training],
            **(fit_kwargs or {}),
        )
        predictions[held_out] = model.predict_keep_probability(values[held_out])
        folds.append(
            {
                "source_group": group,
                "training_rows": int(training.sum()),
                "held_out_rows": int(held_out.sum()),
                "held_out_keeps": int(((targets == 1) & held_out).sum()),
                "held_out_passes": int(((targets == 0) & held_out).sum()),
            }
        )
    return predictions, folds


def operating_point(
    keep_probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> OperatingPoint:
    probabilities = np.asarray(keep_probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    rejected = probabilities < float(threshold)
    pass_mask = targets == 0
    keep_mask = targets == 1
    passes_removed = int((pass_mask & rejected).sum())
    keeps_lost = int((keep_mask & rejected).sum())
    pass_total = int(pass_mask.sum())
    keep_total = int(keep_mask.sum())
    return OperatingPoint(
        threshold=float(threshold),
        pass_total=pass_total,
        passes_removed=passes_removed,
        pass_removal_rate=passes_removed / pass_total if pass_total else 0.0,
        keep_total=keep_total,
        keeps_lost=keeps_lost,
        keep_loss_rate=keeps_lost / keep_total if keep_total else 0.0,
    )


def best_safe_operating_point(
    keep_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    max_keep_loss_rate: float = SHIP_MAX_KEEP_LOSS_RATE,
) -> OperatingPoint:
    """Maximize Pass removal while respecting the Keep-loss budget."""
    probabilities = np.asarray(keep_probabilities, dtype=np.float64)
    candidates: Iterable[float] = sorted(
        {0.0, 1.0, *(float(value) + 1e-12 for value in probabilities)}
    )
    safe = [
        operating_point(probabilities, labels, threshold)
        for threshold in candidates
    ]
    safe = [
        point
        for point in safe
        if point.keep_loss_rate <= float(max_keep_loss_rate) + 1e-12
    ]
    return max(
        safe,
        key=lambda point: (
            point.pass_removal_rate,
            -point.keep_loss_rate,
            -point.threshold,
        ),
    )


def ship_gate(
    point: OperatingPoint,
    *,
    source_vods: int | None = None,
) -> Dict:
    pass_ok = point.pass_removal_rate >= SHIP_MIN_PASS_REMOVAL_RATE
    keep_ok = point.keep_loss_rate <= SHIP_MAX_KEEP_LOSS_RATE
    coverage_ok = (
        True
        if source_vods is None
        else int(source_vods) >= SHIP_MIN_SOURCE_VODS
    )
    return {
        "passed": bool(pass_ok and keep_ok and coverage_ok),
        "minimum_pass_removal_rate": SHIP_MIN_PASS_REMOVAL_RATE,
        "maximum_keep_loss_rate": SHIP_MAX_KEEP_LOSS_RATE,
        "minimum_source_vods": SHIP_MIN_SOURCE_VODS,
        "pass_removal_passed": bool(pass_ok),
        "keep_loss_passed": bool(keep_ok),
        "source_coverage_passed": bool(coverage_ok),
        "reason": (
            "passed"
            if pass_ok and keep_ok and coverage_ok
            else "insufficient_source_coverage"
            if not coverage_ok
            else "insufficient_pass_removal"
            if not pass_ok
            else "excessive_keep_loss"
        ),
    }


def second_look_ship_gate(point: OperatingPoint, *, source_vods: int) -> Dict:
    pass_ok = point.pass_removal_rate >= SECOND_LOOK_MIN_PASS_REMOVAL_RATE
    keep_ok = point.keep_loss_rate <= SECOND_LOOK_MAX_KEEP_LOSS_RATE
    coverage_ok = int(source_vods) >= SECOND_LOOK_MIN_SOURCE_VODS
    return {
        "passed": bool(pass_ok and keep_ok and coverage_ok),
        "minimum_pass_removal_rate": SECOND_LOOK_MIN_PASS_REMOVAL_RATE,
        "maximum_keep_loss_rate": SECOND_LOOK_MAX_KEEP_LOSS_RATE,
        "minimum_source_vods": SECOND_LOOK_MIN_SOURCE_VODS,
        "pass_removal_passed": bool(pass_ok),
        "keep_loss_passed": bool(keep_ok),
        "source_coverage_passed": bool(coverage_ok),
        "reason": (
            "passed"
            if pass_ok and keep_ok and coverage_ok
            else "insufficient_source_coverage"
            if not coverage_ok
            else "insufficient_pass_removal"
            if not pass_ok
            else "excessive_keep_loss"
        ),
    }


def trajectory_features(
    timestamps: np.ndarray,
    channels: Sequence[np.ndarray],
    start: float,
    end: float,
) -> np.ndarray:
    """Summarize how audio/motion build and settle inside one candidate."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    mask = (timestamps >= float(start)) & (timestamps <= float(end))
    result: List[float] = []
    for raw_channel in channels:
        values = np.asarray(raw_channel, dtype=np.float64)[mask]
        if len(values) == 0:
            result.extend([0.0] * 10)
            continue
        thirds = np.array_split(values, 3)
        means = [
            float(part.mean()) if len(part) else 0.0
            for part in thirds
        ]
        peak_position = int(np.argmax(values)) / max(1, len(values) - 1)
        result.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(values.max()),
                *means,
                means[2] - means[0],
                means[2] / (max(means) + 1e-6),
                peak_position,
                float(np.mean(values > np.quantile(values, 0.80))),
            ]
        )
    return np.asarray(result, dtype=np.float64)


def model_payload(
    model: PointwiseModel,
    *,
    threshold: float,
    metadata: Dict,
    calibration_scale: float = 1.0,
    calibration_intercept: float = 0.0,
) -> Dict:
    return {
        "version": 1,
        "kind": "second_look_pass_rejector",
        "feature_names": SECOND_LOOK_FEATURE_NAMES,
        "threshold": float(threshold),
        "calibration_scale": float(calibration_scale),
        "calibration_intercept": float(calibration_intercept),
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "weights": model.weights.tolist(),
        "bias": float(model.bias),
        "metadata": dict(metadata),
    }


def source_key_digest(source_key: str) -> str:
    """Stable local lookup for selecting a source's held-out model.

    This digest is pseudonymous, not anonymous: guessed source IDs can be
    hashed and compared. Public packaging removes these keys with
    publish/export_public_models.py while local evaluation retains its lookup.
    """
    import hashlib

    return hashlib.sha256(str(source_key or "").encode("utf-8")).hexdigest()[:16]


def ensemble_model_payload(
    source_models: Dict[str, Sequence[ShallowModel]],
    *,
    threshold: float,
    metadata: Dict,
    calibration_scale: float = 1.0,
    calibration_intercept: float = 0.0,
) -> Dict:
    def compact(values: np.ndarray) -> List:
        return np.asarray(values, dtype=np.float64).round(8).tolist()

    return {
        "version": 2,
        "kind": "second_look_pass_rejector",
        "architecture": "shallow_tanh_logit_ensemble",
        "feature_names": SECOND_LOOK_FEATURE_NAMES,
        "threshold": float(threshold),
        "calibration_scale": float(calibration_scale),
        "calibration_intercept": float(calibration_intercept),
        "source_models": {
            source_key_digest(str(source)): [
                {
                    "mean": compact(model.mean),
                    "scale": compact(model.scale),
                    "input_weights": compact(model.input_weights),
                    "input_bias": compact(model.input_bias),
                    "output_weights": compact(model.output_weights),
                    "output_bias": float(model.output_bias),
                }
                for model in models
            ]
            for source, models in sorted(source_models.items())
        },
        "source_key_scheme": "sha256-16",
        "metadata": dict(metadata),
    }


def load_second_look_model(
    path: str | None = None,
    *,
    source_key: str | None = None,
) -> SecondLookRejectorModel | None:
    """Load only a validated model; malformed/unapproved artifacts fail closed.

    Failing closed is deliberate, but it must never be silent: an artifact that
    goes stale (most easily by a SECOND_LOOK_FEATURE_NAMES change landing
    without a retrain) otherwise disables Second Look rejection entirely with
    no operator-visible signal.  Every rejection path below logs why.
    """
    resolved = path or os.path.join(get_models_dir(), SECOND_LOOK_MODEL_FILENAME)

    def reject(reason: str, *args) -> None:
        logger.warning(
            "second-look rejector DISABLED (%s): " + reason,
            resolved,
            *args,
        )
        return None

    # Sealed-aware (core/model_vault.py): the packaged build ships only the
    # encrypted form, so existence and read both have to go through the vault.
    # An unreadable or tampered blob raises, which the except below turns into
    # the same fail-closed None as a malformed model.
    if not protected_json_exists(resolved):
        return reject("no model artifact at this path")
    try:
        payload = load_protected_json(resolved)
        if payload.get("kind") != "second_look_pass_rejector":
            return reject(
                "wrong artifact kind %r", payload.get("kind"),
            )
        model_features = payload.get("feature_names")
        if model_features != SECOND_LOOK_FEATURE_NAMES:
            if isinstance(model_features, list):
                added = [
                    f for f in SECOND_LOOK_FEATURE_NAMES
                    if f not in model_features
                ]
                removed = [
                    f for f in model_features
                    if f not in SECOND_LOOK_FEATURE_NAMES
                ]
                return reject(
                    "feature layout is stale -- model has %d feature(s), code "
                    "expects %d (code added %s; model still has %s). Retrain "
                    "with scripts/train_second_look_rejector.py",
                    len(model_features),
                    len(SECOND_LOOK_FEATURE_NAMES),
                    added or "nothing",
                    removed or "nothing",
                )
            return reject("feature_names missing or malformed")
        metadata = dict(payload.get("metadata") or {})
        gate = metadata.get("ship_gate") or {}
        if not bool(gate.get("passed")):
            return reject(
                "artifact never cleared its ship gate (reason=%r)",
                gate.get("reason"),
            )
        pointwise = None
        ensemble: Tuple[ShallowModel, ...] = ()
        if payload.get("architecture") == "shallow_tanh_logit_ensemble":
            raw_source_models = payload.get("source_models") or {}
            # Digest first; a plain key still resolves so an artifact trained
            # before the scheme changed keeps its held-out selection.
            selected_raw_models = raw_source_models.get(
                source_key_digest(str(source_key or ""))
            ) or raw_source_models.get(str(source_key or ""))
            if not selected_raw_models:
                selected_raw_models = [
                    raw
                    for models in raw_source_models.values()
                    for raw in (models or [])
                ]
            parsed_models = []
            for raw in selected_raw_models:
                model = ShallowModel(
                    mean=np.asarray(raw["mean"], dtype=np.float64),
                    scale=np.asarray(raw["scale"], dtype=np.float64),
                    input_weights=np.asarray(
                        raw["input_weights"], dtype=np.float64,
                    ),
                    input_bias=np.asarray(raw["input_bias"], dtype=np.float64),
                    output_weights=np.asarray(
                        raw["output_weights"], dtype=np.float64,
                    ),
                    output_bias=float(raw["output_bias"]),
                )
                feature_count = len(SECOND_LOOK_FEATURE_NAMES)
                hidden_units = len(model.input_bias)
                if (
                    len(model.mean) != feature_count
                    or len(model.scale) != feature_count
                    or model.input_weights.shape != (
                        feature_count,
                        hidden_units,
                    )
                    or len(model.output_weights) != hidden_units
                ):
                    return reject(
                        "ensemble member has malformed shapes for a "
                        "%d-feature layout",
                        feature_count,
                    )
                parsed_models.append(model)
            if not parsed_models:
                return reject("artifact carries no ensemble members")
            ensemble = tuple(parsed_models)
        else:
            pointwise = PointwiseModel(
                mean=np.asarray(payload["mean"], dtype=np.float64),
                scale=np.asarray(payload["scale"], dtype=np.float64),
                weights=np.asarray(payload["weights"], dtype=np.float64),
                bias=float(payload["bias"]),
            )
            if (
                len(pointwise.mean) != len(SECOND_LOOK_FEATURE_NAMES)
                or len(pointwise.scale) != len(SECOND_LOOK_FEATURE_NAMES)
                or len(pointwise.weights) != len(SECOND_LOOK_FEATURE_NAMES)
            ):
                return reject(
                    "pointwise model has malformed shapes for a "
                    "%d-feature layout",
                    len(SECOND_LOOK_FEATURE_NAMES),
                )
        return SecondLookRejectorModel(
            pointwise=pointwise,
            ensemble=ensemble,
            threshold=float(payload["threshold"]),
            calibration_scale=float(payload.get("calibration_scale", 1.0)),
            calibration_intercept=float(
                payload.get("calibration_intercept", 0.0)
            ),
            metadata=metadata,
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return reject("artifact unreadable or malformed (%s: %s)",
                      type(exc).__name__, exc)

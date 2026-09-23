# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Learned ranker for clip selection (plan §5.6) — pure numpy logistic model.

Why numpy logistic regression (not LightGBM/torch):
  * zero new dependencies and tiny (<5 KB JSON), fits the frozen-app packaging;
  * deterministic train + inference (handbook §9);
  * trivial to fine-tune with a warm start, which is how the HYBRID works:
    fine-tuning anchors the weights to the bundled base via an L2 penalty, so a
    fresh install behaves like the base and only drifts toward personal taste as
    real keep/reject labels accumulate.

Operates on raw feature arrays; the caller turns clips into vectors via
features.feature_vector so this module stays decoupled.
"""

import json
import os
from typing import Optional

import numpy as np

from core.model_vault import load_protected_json


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticRanker:
    def __init__(self, weights, bias, mean, std, feature_names, version, n_labels=0,
                 calibration=None):
        self.w = np.asarray(weights, dtype=np.float64)
        self.b = float(bias)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.feature_names = list(feature_names)
        self.version = int(version)
        self.n_labels = int(n_labels)  # how many labels this model was trained on
        # Optional Platt-scaling block {"a": float, "b": float} fit on this
        # model's raw scores vs pointwise keep/pass labels (see
        # predict_calibrated). Per-install user models never carry it, so it
        # must stay optional for load/save backward compatibility.
        self.calibration = dict(calibration) if calibration else None

    # --- io ---
    @classmethod
    def load(cls, path: str) -> "LogisticRanker":
        # Sealed-aware: the bundled base ranker ships encrypted (core/
        # model_vault.py), the per-install user model stays plain JSON under
        # data/. One loader serves both — see load_protected_json.
        d = load_protected_json(path)
        return cls(d["weights"], d["bias"], d["mean"], d["std"],
                   d["feature_names"], d["version"], d.get("n_labels", 0),
                   calibration=d.get("calibration"))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "version": self.version,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.w.tolist(),
            "bias": self.b,
            "n_labels": self.n_labels,
        }
        if self.calibration:
            payload["calibration"] = {
                "a": float(self.calibration["a"]),
                "b": float(self.calibration["b"]),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # --- inference ---
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        """Pre-sigmoid linear score — the quantity pairwise training orders."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return self._standardize(X) @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return _sigmoid(self._standardize(X) @ self.w + self.b)

    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Absolute keep-probability via the Platt calibration block.

        A pairwise-trained model only promises a monotonic ORDERING within a
        deck; within-session deltas cancel the per-VOD context features, so
        the raw score carries no absolute publishability threshold. This maps
        the raw pairwise score s through sigmoid(a*s + b), where (a, b) were
        fit pointwise on keep/pass labels across all training VODs.

        NOT wired into selection yet: it is the substrate for a future
        adaptive deck threshold ("only surface clips with calibrated
        P(keep) >= t"), not a live behavior change. Raises ValueError when the
        model has no calibration block (e.g. per-install user models).
        """
        if not self.calibration:
            raise ValueError("this model has no calibration block")
        a = float(self.calibration["a"])
        b = float(self.calibration["b"])
        return _sigmoid(a * self.raw_score(X) + b)


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names,
    version: int,
    base: Optional[LogisticRanker] = None,
    sample_weight: Optional[np.ndarray] = None,
    l2: float = 1e-2,
    anchor: float = 2.0,
    lr: float = 0.2,
    epochs: int = 800,
) -> LogisticRanker:
    """Fit a logistic ranker. With ``base`` set, anchor to it (HYBRID fine-tune).

    ``anchor`` is the strength of the L2 pull toward the base weights: high anchor
    + few labels => stays near base; as labels grow, the data term dominates and
    the model personalizes. Standardization is inherited from the base so the
    feature space is identical across base and fine-tuned models.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    weights = (
        np.ones(n, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).ravel()
    )
    if len(weights) != n:
        raise ValueError("pointwise training arrays must have equal lengths")
    normalizer = max(float(weights.sum()), 1.0)

    if base is not None:
        mean, std = base.mean.copy(), base.std.copy()
        w = base.w.copy()
        b = base.b
        w_anchor = base.w.copy()
        b_anchor = base.b
    else:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
        w = np.zeros(d)
        b = 0.0
        w_anchor = np.zeros(d)
        b_anchor = 0.0
        anchor = 0.0  # no base to anchor to

    Xs = (X - mean) / std
    for _ in range(epochs):
        p = _sigmoid(Xs @ w + b)
        err = (p - y) * weights
        grad_w = Xs.T @ err / normalizer + l2 * w + anchor * (w - w_anchor)
        grad_b = float(err.sum() / normalizer) + anchor * (b - b_anchor)
        w -= lr * grad_w
        b -= lr * grad_b

    return LogisticRanker(w, b, mean, std, feature_names, version, n_labels=n)


def train_pairwise(
    X: np.ndarray,
    y: np.ndarray,
    groups,
    feature_names,
    version: int,
    *,
    sample_weight: Optional[np.ndarray] = None,
    base: Optional[LogisticRanker] = None,
    l2: float = 1e-2,
    anchor: float = 1.0,
    lr: float = 0.12,
    epochs: int = 1000,
) -> LogisticRanker:
    """Fit a linear RankNet-style model from within-session preferences.

    A Keep in one VOD is only compared with a Pass from that same review deck.
    That avoids teaching the model that a quiet horror highlight must have the
    same raw feature scale as a loud shooter highlight.  The returned object is
    still ``LogisticRanker`` because inference only needs a stable monotonic
    score; the intercept is intentionally zero since it cancels in pairwise
    ranking.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).ravel()
    groups = np.asarray(groups, dtype=object).ravel()
    weights = (
        np.ones(len(y), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).ravel()
    )
    if not (len(X) == len(y) == len(groups) == len(weights)):
        raise ValueError("pairwise training arrays must have equal lengths")

    if base is not None:
        mean, std = base.mean.copy(), base.std.copy()
        w = base.w.copy()
        w_anchor = base.w.copy()
    else:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
        w = np.zeros(X.shape[1], dtype=np.float64)
        w_anchor = np.zeros(X.shape[1], dtype=np.float64)
        anchor = 0.0

    deltas = []
    targets = []
    pair_weights = []
    # Stable group ordering and row ordering keep training deterministic.
    for group in dict.fromkeys(groups.tolist()):
        indexes = np.flatnonzero(groups == group)
        for offset, first in enumerate(indexes):
            for second in indexes[offset + 1:]:
                if y[first] == y[second]:
                    continue
                high, low = (
                    (first, second)
                    if y[first] > y[second]
                    else (second, first)
                )
                delta = (X[high] - X[low]) / std
                # Keep/Maybe and Maybe/Pass are deliberately half-strength
                # preferences; Keep/Pass retains the historical full weight.
                ordinal_distance = float(y[high] - y[low])
                weight = max(
                    1e-6,
                    float(np.sqrt(weights[high] * weights[low]))
                    * ordinal_distance,
                )
                # Symmetric examples make the preference direction explicit
                # and prevent a one-sided optimizer from learning an offset.
                deltas.extend((delta, -delta))
                targets.extend((1.0, 0.0))
                pair_weights.extend((weight, weight))

    if not deltas:
        raise ValueError("pairwise training needs a keep and a pass in the same session")

    D = np.asarray(deltas, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    pair_weights = np.asarray(pair_weights, dtype=np.float64)
    normalizer = max(float(pair_weights.sum()), 1.0)
    for _ in range(epochs):
        probabilities = _sigmoid(D @ w)
        weighted_error = (probabilities - targets) * pair_weights
        grad_w = D.T @ weighted_error / normalizer + l2 * w + anchor * (w - w_anchor)
        w -= lr * grad_w

    return LogisticRanker(
        w, 0.0, mean, std, feature_names, version, n_labels=len(y)
    )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Export runtime-only learned models without source lookup identifiers.

Run with --source models --out models/public-release (a new directory).
Review the resulting bytes and notices before adding them to the model manifest.
Local training artifacts remain unchanged. This removes serialized identifiers;
it does not provide a statistical privacy guarantee for learned weights.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engines.reaction.features import FEATURE_NAMES
from engines.reaction.pass_rejector import SECOND_LOOK_FEATURE_NAMES

MODEL_NAMES = ("ranker_base.json", "second_look_rejector.json")


def _numeric(value, shape=()):
    if shape:
        if not isinstance(value, list) or len(value) != shape[0]:
            raise ValueError("Malformed learned model dimensions")
        for item in value:
            _numeric(item, shape[1:])
    elif type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Learned model contains a non-finite or non-numeric parameter")


def public_payload(name, payload):
    """Project known runtime fields; retain ensemble order for new-source scores."""
    if name not in MODEL_NAMES or not isinstance(payload, dict):
        raise ValueError("Unsupported learned model")
    payload = copy.deepcopy(payload)
    if name == "ranker_base.json":
        fields = {"version", "feature_names", "mean", "std", "weights", "bias", "n_labels", "calibration"}
        if set(payload) - fields or payload.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Unreviewed base model schema or feature layout")
        for key in ("version", "n_labels", "bias"):
            _numeric(payload[key])
        for key in ("mean", "std", "weights"):
            _numeric(payload[key], (len(FEATURE_NAMES),))
        if any(x <= 0 for x in payload["std"]):
            raise ValueError("Invalid base model normalization")
        if "calibration" in payload:
            if not isinstance(payload["calibration"], dict) or set(payload["calibration"]) != {"a", "b"}:
                raise ValueError("Unreviewed base calibration schema")
            for value in payload["calibration"].values():
                _numeric(value)
        return payload

    fields = {"version", "kind", "architecture", "feature_names", "threshold",
              "calibration_scale", "calibration_intercept", "source_models", "metadata", "source_key_scheme"}
    if set(payload) - fields:
        raise ValueError("Unreviewed Second Look schema")
    if (payload.get("kind") != "second_look_pass_rejector"
            or payload.get("architecture") != "shallow_tanh_logit_ensemble"
            or payload.get("feature_names") != SECOND_LOOK_FEATURE_NAMES):
        raise ValueError("Unsupported Second Look architecture or features")
    for key in ("version", "threshold", "calibration_scale", "calibration_intercept"):
        _numeric(payload[key])
    if not 0 <= payload["threshold"] <= 1:
        raise ValueError("Invalid rejection threshold")
    if payload.get("metadata", {}).get("ship_gate", {}).get("passed") is not True:
        raise ValueError("Second Look has no passing runtime gate")
    groups = payload["source_models"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Second Look has no ensemble")
    members = []
    member_fields = {"mean", "scale", "input_weights", "input_bias", "output_weights", "output_bias"}
    for group in groups.values():
        if not isinstance(group, list) or not group:
            raise ValueError("Invalid ensemble group")
        for member in group:
            if not isinstance(member, dict) or set(member) != member_fields:
                raise ValueError("Unreviewed ensemble member schema")
            hidden = member["input_bias"]
            if not isinstance(hidden, list) or not hidden:
                raise ValueError("Empty ensemble hidden layer")
            features, units = len(SECOND_LOOK_FEATURE_NAMES), len(hidden)
            for key in ("mean", "scale"):
                _numeric(member[key], (features,))
            for key in ("input_bias", "output_weights"):
                _numeric(member[key], (units,))
            _numeric(member["input_weights"], (features, units))
            _numeric(member["output_bias"])
            if any(x <= 0 for x in member["scale"]):
                raise ValueError("Invalid ensemble normalization")
            members.append(member)
    # Empty lookup key selects the same full ensemble for every input source.
    payload["source_models"] = {"": members}
    payload.pop("source_key_scheme", None)
    payload["metadata"] = {"ship_gate": {"passed": True}}
    return payload


def validate_public_model(path):
    """Reject private artifacts even if someone approved their file hash."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if public_payload(path.name, payload) != payload:
        raise ValueError("Learned model contains private lookup keys or training metadata; export a public copy")


def export_models(source, out):
    source, out = Path(source).absolute(), Path(out).absolute()
    if source.resolve() != source or out.resolve() != out:
        raise ValueError("Model export paths must not be redirected")
    outputs = {}
    for name in MODEL_NAMES:
        path = source / name
        if not path.exists():
            continue
        if path.resolve() != path or not path.is_file():
            raise ValueError("Model source must be an unredirected file")
        payload = public_payload(name, json.loads(path.read_text(encoding="utf-8")))
        outputs[name] = json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    if not outputs:
        raise ValueError("No learned models to export")
    out.mkdir(parents=True, exist_ok=False)
    for name, contents in outputs.items():
        (out / name).write_text(contents, encoding="utf-8", newline="\n")
    return sorted(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for name in export_models(args.source, args.out):
        print(f"Exported public candidate: {name}")

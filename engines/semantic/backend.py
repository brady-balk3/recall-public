# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""llama.cpp backend for the semantic judge (HUMAN_CLIPS plan, Package C).

Wraps llama-cpp-python behind a tiny generate(prompt) -> dict interface so the
judge (and its tests) never import llama_cpp directly. Generation is
JSON-schema-constrained — llama.cpp compiles the schema to a grammar, so the
output ALWAYS parses; verdict.SemanticVerdict.from_raw handles the remaining
semantic validation (enum membership, ranges).

Model discovery is resolve_model(): a GGUF placed under
``get_models_dir()/llm/`` enables the judge; nothing there disables it. No
auto-download in this package — the download UX is a follow-up, and this app
is offline-first, so absence must degrade silently (same contract as the
yamnet burst model in engines/audio/audio_events.py).
"""

import json
import os
import re
from typing import Optional

from core.bundle_paths import get_models_dir
from core.device import get_torch_device


class MalformedCompletionError(ValueError):
    """The model produced a completion, but its text is not usable JSON even
    after salvage. Distinct from runtime failures (load/OOM/crash) so the
    judge can retry/split instead of counting it toward the abort budget —
    a model that answers sloppily is not a broken runtime."""

    def __init__(self, message: str, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text

# Constrained output shape. min/max on numbers are advisory to the model
# (llama.cpp grammars can't express numeric ranges); verdict.from_raw clamps.
VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "hook_strength": {"type": "number", "minimum": 0, "maximum": 1},
        "self_contained": {"type": "number", "minimum": 0, "maximum": 1},
        "payoff": {"type": "number", "minimum": 0, "maximum": 1},
        "moment_type": {
            "type": "string",
            "enum": [
                "funny", "clutch", "fail", "rage", "scare", "story",
                "wholesome", "filler",
            ],
        },
        "title": {"type": "string", "maxLength": 64},
        "hook_line": {"type": "string", "maxLength": 48},
        "verdict": {"type": "string", "enum": ["post", "maybe", "skip"]},
    },
    "required": [
        "hook_strength", "self_contained", "payoff",
        "moment_type", "title", "hook_line", "verdict",
    ],
}


TRIAGE_VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "h": {"type": "number", "minimum": 0, "maximum": 1},
        "s": {"type": "number", "minimum": 0, "maximum": 1},
        "p": {"type": "number", "minimum": 0, "maximum": 1},
        "m": VERDICT_JSON_SCHEMA["properties"]["moment_type"],
        "v": VERDICT_JSON_SCHEMA["properties"]["verdict"],
    },
    "required": ["h", "s", "p", "m", "v"],
}


def map_triage_verdicts(raw) -> list:
    """Expand a compact ``{"verdicts": [{h,s,p,m,v}, ...]}`` payload.

    Module-level so every backend that can serve the text-judge role shares one
    decoder — the visual lane's shared-instance adapter included. A second copy
    would drift the moment the triage schema changes.
    """
    verdicts = raw.get("verdicts") if isinstance(raw, dict) else None
    if not isinstance(verdicts, list):
        return []
    return [
        {
            "hook_strength": item.get("h") if isinstance(item, dict) else None,
            "self_contained": item.get("s") if isinstance(item, dict) else None,
            "payoff": item.get("p") if isinstance(item, dict) else None,
            "moment_type": item.get("m") if isinstance(item, dict) else None,
            "verdict": item.get("v") if isinstance(item, dict) else None,
            # Broad triage decides selection only. Caption generation keeps
            # its existing transcript/scene fallback for selected clips.
            "title": "",
            "hook_line": "",
        }
        for item in verdicts
    ]


def _batch_verdict_schema(count: int) -> dict:
    """Compact fixed-size triage schema for the broad challenger pass."""
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": TRIAGE_VERDICT_JSON_SCHEMA,
                "minItems": count,
                "maxItems": count,
            },
        },
        "required": ["verdicts"],
    }

# ~400 input tokens per candidate prompt; 160 output tokens is comfortably
# above what the JSON verdict needs, so the grammar closes before the cap.
MAX_TOKENS = 160
TEMPERATURE = 0.2
# Enough context for the system text + transcript slice + chat lines; small
# keeps the KV cache cheap on CPU.
N_CTX = 2048


# --- Completion-text salvage -------------------------------------------------
#
# llama.cpp grammar constraints make malformed JSON rare but not impossible:
# truncation at max_tokens, grammar gaps on user-swapped models, and chat
# templates that wrap the body in fences/prose have all been observed (real
# scan log: "Semantic judge batch failed ... Expecting ',' delimiter"). The
# salvage path repairs the TRANSPORT only — SemanticVerdict.from_raw still
# performs all semantic validation on whatever parses out.

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*")
# Bound the repair work; a verdict batch is a few hundred chars.
_MAX_SALVAGE_CHARS = 32768


def _strip_wrappers(text: str) -> str:
    """Drop markdown fences and leading/trailing prose around the JSON body."""
    t = _FENCE_RE.sub("", text).strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        return t
    start = min(starts)
    end = max(t.rfind("}"), t.rfind("]"))
    return t[start:end + 1] if end >= start else t[start:]


def _trim_dangling_tail(out: list) -> None:
    """Remove a trailing partial token (whitespace, comma, dangling key/colon,
    or an incomplete number) from the char list, in place."""
    while out and (out[-1].isspace() or out[-1] == ","):
        out.pop()
    # Incomplete number: {"h": 0.   /   "p": 1e
    while out and out[-1] in ".+-eE" and len(out) > 1 and (out[-2].isdigit() or out[-2] in ".+-eE"):
        out.pop()
    if out and out[-1] == ":":
        # Dangling key with no value: drop ':' and the key string before it.
        out.pop()
        while out and out[-1].isspace():
            out.pop()
        if out and out[-1] == '"':
            out.pop()
            while out:
                ch = out.pop()
                if ch == '"':
                    break
        while out and (out[-1].isspace() or out[-1] == ","):
            out.pop()


def _balance_close(text: str) -> str:
    """Bounded bracket-balance repair: drop trailing commas before closers,
    insert the comma a `}{` element boundary is missing, close an unterminated
    string, trim a dangling partial token, and close any open brackets."""
    out: list = []
    stack: list = []
    in_str = False
    escaped = False
    for ch in text:
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            continue
        if ch in "{[":
            # `}{` / `]{` between array elements: the comma the model forgot.
            idx = len(out) - 1
            while idx >= 0 and out[idx].isspace():
                idx -= 1
            if idx >= 0 and out[idx] in "}]":
                out.append(",")
            stack.append(ch)
            out.append(ch)
            continue
        if ch in "}]":
            while out and out[-1].isspace():
                out.pop()
            if out and out[-1] == ",":
                out.pop()
            if stack and stack[-1] == ("{" if ch == "}" else "["):
                stack.pop()
                out.append(ch)
            # Unmatched closer: drop it rather than corrupting the stack.
            continue
        out.append(ch)
    if in_str:
        out.append('"')
    _trim_dangling_tail(out)
    for opener in reversed(stack):
        out.append("}" if opener == "{" else "]")
    return "".join(out)


def parse_completion(text: str):
    """Parse a completion's text to JSON, salvaging cheaply before giving up.

    Order: direct parse -> wrapper strip (fences/prose) -> bracket-balance
    repair -> repair of the text truncated at its last complete object.
    Raises MalformedCompletionError when none of it parses; the caller decides
    whether to regenerate, split, or drop (engines/semantic/judge.py).
    """
    try:
        return json.loads(text)
    except (TypeError, ValueError) as exc:
        first_error = exc
    if not isinstance(text, str) or not text.strip():
        raise MalformedCompletionError(
            f"empty or non-text completion: {first_error}", raw_text=str(text or ""),
        )
    body = _strip_wrappers(text[:_MAX_SALVAGE_CHARS])
    attempts = [body, _balance_close(body)]
    last_close = body.rfind("}")
    if last_close > 0:
        attempts.append(_balance_close(body[:last_close + 1]))
    for attempt in attempts:
        if not attempt:
            continue
        try:
            return json.loads(attempt)
        except ValueError:
            continue
    raise MalformedCompletionError(
        f"unsalvageable completion: {first_error}", raw_text=text,
    )


# Newest generation first. Discovery cannot rely on a plain filename sort:
# "qwen3-4b" sorts BEFORE "qwen3.5-4b" ('-' is 0x2d, '.' is 0x2e), so dropping
# a newer lane beside an older one would silently keep loading the old model.
# Order matters within the tuple too — "qwen3.5-4b" also contains "qwen3", so
# the newest family has to be tested first.
_MODEL_FAMILY_ORDER = (("qwen3.5", "qwen35"), ("qwen3",), ("qwen2.5", "qwen25"))


def family_rank(label: str) -> int:
    """Preference index for a model path/filename; lower is newer.

    Shared by both judge lanes (text GGUF here, model+mmproj pair in
    visual_judge.resolve_models) so a dropped-in upgrade wins in one place.
    Unknown families rank last but stay selectable — a user-supplied model is
    still better than no judge.
    """
    text = str(label).lower()
    for index, aliases in enumerate(_MODEL_FAMILY_ORDER):
        if any(alias in text for alias in aliases):
            return index
    return len(_MODEL_FAMILY_ORDER)


def resolve_model() -> Optional[str]:
    """Path to the judge's GGUF under models/llm/, or None (judge disabled).

    Newest known family wins; the filename sort only breaks ties, so the pick
    stays deterministic when several models are present. The Settings model
    picker is a follow-up.
    """
    llm_dir = os.path.join(get_models_dir(), "llm")
    if not os.path.isdir(llm_dir):
        return None
    ggufs = sorted(
        name for name in os.listdir(llm_dir)
        if name.lower().endswith(".gguf")
    )
    if not ggufs:
        return None
    return os.path.join(llm_dir, min(ggufs, key=lambda name: (family_rank(name), name)))


def _cpu_threads() -> int:
    """Physical-core-ish thread count: llama.cpp scales badly past the real
    cores, and hogging every logical CPU starves the rest of the scan."""
    logical = os.cpu_count() or 4
    return max(2, min(8, logical // 2))


class LlamaBackend:
    """Lazy-loading llama.cpp chat backend. Import and model load both happen
    on first generate() so constructing the backend can never fail a scan —
    any problem surfaces as an exception the judge/engine swallow."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return self._llm
        # Import guarded: llama-cpp-python is optional at runtime (it needs a
        # native wheel that may not build on every install); when it's absent
        # the ImportError propagates to the judge's failure path and the scan
        # continues without semantic fields.
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=N_CTX,
            # Full GPU offload when torch already found CUDA; CPU otherwise.
            n_gpu_layers=-1 if get_torch_device() == "cuda" else 0,
            n_threads=_cpu_threads(),
            verbose=False,
        )
        # llama-cpp ships an optional disk-backed prompt cache through
        # ``diskcache``. Recall never enables it: cached model state is not worth
        # a pickle-deserialization surface in a local app that processes
        # attacker-influenced media. Keep the invariant explicit for future edits.
        self._llm.set_cache(None)
        return self._llm

    def generate(self, prompt: str) -> dict:
        """One schema-constrained completion. Returns the parsed JSON dict."""
        llm = self._load()
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_object",
                "schema": VERDICT_JSON_SCHEMA,
            },
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = result["choices"][0]["message"]["content"]
        return parse_completion(text)

    def generate_batch(self, prompt: str, count: int) -> list:
        """Judge several candidates in one constrained completion.

        The root stays a JSON object because llama.cpp's ``json_object`` mode
        is most reliable in that shape; callers receive only the ordered list.
        """
        llm = self._load()
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_object",
                "schema": _batch_verdict_schema(count),
            },
            temperature=TEMPERATURE,
            max_tokens=72 * max(1, count),
        )
        text = result["choices"][0]["message"]["content"]
        return map_triage_verdicts(parse_completion(text))

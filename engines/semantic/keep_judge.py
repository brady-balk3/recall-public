# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Keep judge: how likely the creator is to KEEP each finalist, as a 0-1 score.

A second, narrower pass over the finalists the text judge already read. It
differs from judge.build_prompt in four measured ways:

* wide context -- 20 s of transcript before the clip and 8 s after, so a
  setup and a payoff that lands just past the cut are both visible;
* no loudness numbers -- the signal summary anchored the judge on exactly the
  reaction magnitude the deck was already over-weighting;
* the visual judge's one-line summary of what is on screen;
* a few of THIS creator's own past Keep/Pass decisions as calibration.

Measured 2026-10-12/13 on 379 reviewed clips from 25 scans (within-scan
pairwise, keeps above passes): the stored production payoff scored 0.615 and
this prompt 0.674 on the same 4B model; prompt-only variants without the
creator examples stayed at ~0.62. Larger models (9B, 35B-A3B MoE), whole-scan
ranking, 20 examples and audio/video descriptions all failed to beat it.

The score is read from the model's own probabilities over the ten score
digits (expected value / 9) rather than its single sampled digit: a 4B model
rates nearly everything 7-8, and the tied integers cannot order a deck.
"""

from typing import Callable, Dict, List, Optional, Sequence

from engines.semantic.judge import _transcript_words, _words_in_window

BEFORE_SEC = 20.0
AFTER_SEC = 8.0
EXAMPLE_TEXT_CHARS = 280
MAX_TOKENS = 220
# Two consecutive failures = the runtime is broken for this scan.
MAX_CONSECUTIVE_FAILURES = 2

RUBRIC = """You help a Twitch streamer choose which moments from their stream to post as short clips (TikTok/Shorts).
Read one candidate clip and judge how likely the streamer is to KEEP it for posting.

Clips that get kept usually have at least one of:
- a quotable line, punchline, bit or running joke that lands inside the clip
- a big reaction with a clear cause (a win, clutch, scare, fail, surprise)
- a passionate or funny rant or story that makes a point
- something a stranger would enjoy without knowing the stream

Clips that get passed are usually:
- lobby, menu or shop chatter; greeting or thanking chatters; reading game text aloud
- routine gameplay callouts with no payoff
- mundane small talk, or a fragment of only a few words
- a setup whose payoff happens after the clip ends"""

KEEP_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "maxLength": 400},
        "keep_score": {"type": "integer", "enum": list(range(10))},
    },
    "required": ["reason", "keep_score"],
}


def build_keep_prompt(
    before: Sequence[str],
    clip: Sequence[str],
    after: Sequence[str],
    duration: float,
    *,
    title: Optional[str] = None,
    vision: Optional[str] = None,
    examples: Optional[Sequence[dict]] = None,
) -> str:
    parts = [RUBRIC, ""]
    if examples:
        parts.append("Past decisions by this streamer on other streams (for calibration):")
        for example in examples:
            label = "KEPT" if example.get("label") else "PASSED"
            text = str(example.get("text", ""))[:EXAMPLE_TEXT_CHARS]
            parts.append(f'- {label} ({float(example.get("dur", 0.0)):.0f}s): "{text}"')
        parts.append("")
    if title:
        parts.append(f'Stream title: "{title}"')
    parts += [
        f"Context BEFORE the clip (not posted): {' '.join(before) or '(silence)'}",
        f"CLIP ({duration:.0f}s, this is what would be posted): {' '.join(clip) or '(no speech)'}",
        f"Right AFTER the clip (not posted): {' '.join(after) or '(silence)'}",
    ]
    if vision:
        parts.append(f"On screen during the clip (from a vision model): {vision}")
    parts += [
        "",
        "First give a one- or two-sentence reason, then keep_score from 0 "
        "(certainly passed) to 9 (certainly kept).",
        "Respond with JSON only.",
    ]
    return "\n".join(parts)


def expected_keep_score(llm, prompt: str, content: str, fallback_digit: int) -> float:
    """0-1 keep score from the model's probabilities at the score digit.

    Re-reads the model's own answer up to ``"keep_score":`` and takes the
    softmax over the ten digit tokens. Any failure (unusual chat template,
    tokenizer, binding change) falls back to the sampled digit, so a score is
    always produced.
    """
    try:
        import jinja2
        import llama_cpp
        import numpy as np

        cut = content.index('"keep_score"')
        prefix = content[: content.index(":", cut) + 1] + " "
        template = llm.metadata["tokenizer.chat_template"]
        env = jinja2.Environment()

        def _raise(message):
            raise ValueError(message)

        env.globals["raise_exception"] = _raise
        bos = llm._model.token_get_text(llm.token_bos()) if llm.token_bos() >= 0 else ""
        eos = llm._model.token_get_text(llm.token_eos()) if llm.token_eos() >= 0 else ""
        text = env.from_string(template).render(
            messages=[{"role": "user", "content": prompt}],
            add_generation_prompt=True, bos_token=bos, eos_token=eos,
        )
        tokens = llm.tokenize((text + prefix).encode(), add_bos=False, special=True)
        llm.reset()
        llm.eval(tokens)
        logits = np.ctypeslib.as_array(
            llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1), shape=(llm.n_vocab(),),
        )
        digits = [llm.tokenize(str(d).encode(), add_bos=False)[0] for d in range(10)]
        values = logits[digits].astype(np.float64)
        probs = np.exp(values - values.max())
        probs /= probs.sum()
        return float((np.arange(10) * probs).sum()) / 9.0
    except Exception:  # noqa: BLE001 - the sampled digit is always a valid score
        return max(0, min(9, int(fallback_digit))) / 9.0


def judge_keep(
    candidates: List[dict],
    transcript: Optional[dict],
    *,
    backend,
    title: Optional[str] = None,
    examples: Optional[Sequence[dict]] = None,
    cancel_check: Optional[Callable[[], None]] = None,
    health: Optional[dict] = None,
) -> Dict[int, float]:
    """Keep score (0-1) per candidate index. Candidates with no speech and no
    on-screen summary are skipped: the prompt would carry no evidence."""
    words = _transcript_words(transcript)
    scores: Dict[int, float] = {}
    attempts = failures = consecutive = skipped = 0
    aborted = False
    for idx, candidate in enumerate(candidates):
        if cancel_check:
            cancel_check()
        start = float(candidate.get("judge_start", candidate.get("start", 0.0)))
        end = float(candidate.get("judge_end", candidate.get("end", start)))
        clip_words = _words_in_window(words, start, end)
        vision = str(candidate.get("visual_semantic_summary") or "").strip() or None
        if not clip_words and not vision:
            skipped += 1
            continue
        prompt = build_keep_prompt(
            _words_in_window(words, start - BEFORE_SEC, start),
            clip_words,
            _words_in_window(words, end, end + AFTER_SEC),
            end - start,
            title=title, vision=vision, examples=examples,
        )
        attempts += 1
        try:
            scores[idx] = float(backend.keep_score(prompt))
            consecutive = 0
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad generation must not kill the pass
            failures += 1
            consecutive += 1
            print(f"Keep judge failed (candidate {idx}): {exc}")
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                print("Keep judge aborted: backend failing consistently.")
                aborted = True
                break
    if health is not None:
        health.update({
            "candidates_received": len(candidates),
            "scored_candidates": len(scores),
            "skipped_no_evidence": skipped,
            "attempts": attempts,
            "failures": failures,
            "aborted": aborted,
            "examples": len(examples or []),
        })
    return scores


__all__ = ["KEEP_JSON_SCHEMA", "build_keep_prompt", "expected_keep_score", "judge_keep"]

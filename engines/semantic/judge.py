# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Judge candidate clips with a local LLM (HUMAN_CLIPS plan, Package C).

judge_candidates() builds compact evidence prompts from transcript, visible
chat/OCR, and measured signals. The bundled backend batches several candidates
per schema-constrained completion; test backends can keep using one at a time.

Failure posture matches the other optional channels: no usable text or
nonverbal evidence produces no verdict for that candidate. Backend errors skip
the candidate, and repeated failures abort early rather than blocking a scan.
"""

from typing import Dict, List, Optional

from engines.semantic.backend import MalformedCompletionError
from engines.semantic.verdict import SemanticVerdict

# Transcript slice extends past the clip so the judge sees the sentence the
# clip opens into and the one it settles on (mirrors the boundary-snapping
# windows in Package B).
WINDOW_PAD_SEC = 3.0
# Cap on chat lines per prompt: enough to convey crowd reaction, small enough
# to keep the prompt near the ~400-token budget.
MAX_CHAT_LINES = 5
# Two consecutive backend failures = the model/runtime is broken for this run.
MAX_CONSECUTIVE_FAILURES = 2
# Four real candidate evidence blocks plus the constrained response fit the
# bundled model's 2k context. Longer batches can exceed it on transcript-heavy
# clips; the batch loop below also splits and retries context overflows so one
# unusually verbose group cannot disable the judge for the rest of the scan.
BATCH_SIZE = 4

_CROWD_REACTION_TERMS = (
    "lmao", "lol", "funny", "wtf", "no way", "so loud", "scream",
    "scared", "jumped", "my ears", "headphone", "heart", "hilarious",
    "clip it", "holy", "bro", "what was that",
)


def _get(obj, key, default=None):
    """Field access for dataclass frames and plain dicts alike."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _transcript_words(transcript: Optional[dict]) -> List[tuple]:
    """Flatten to (raw_text, start) sorted by start. Raw casing/punctuation is
    kept — the judge reads the words, unlike the hype lexicon which matches
    them (engines/caption/vod_transcribe.iter_words lowercases for that)."""
    words = []
    for segment in (transcript or {}).get("segments", []) or []:
        for word in segment.get("words", []) or []:
            text = (word.get("word") or "").strip()
            if not text:
                continue
            words.append((text, float(word.get("start", 0.0))))
    words.sort(key=lambda w: w[1])
    return words


def _words_in_window(words: List[tuple], lo: float, hi: float) -> List[str]:
    return [text for (text, start) in words if lo <= start <= hi]


def _chat_lines(chat_frames, lo: float, hi: float, limit: int = MAX_CHAT_LINES) -> List[str]:
    """Render in-window chat for the prompt.

    Two shapes arrive here: raw messages with text (ChatMessage-like), which
    become quoted lines ranked by emote weight, and the pipeline's per-second
    velocity frames (ChatFrame-like, no text), which collapse to a single
    intensity summary — the judge can still weigh "chat erupted" without the
    message bodies.
    """
    in_window = []
    for frame in chat_frames or []:
        ts = _get(frame, "timestamp")
        if ts is None or not (lo <= float(ts) <= hi):
            continue
        in_window.append(frame)
    if not in_window:
        return []

    with_text = [f for f in in_window if str(_get(f, "text", "") or "").strip()]
    if with_text:
        # "Top" = emote-heaviest first; emote spam is the strongest single-line
        # hype marker chat gives us.
        with_text.sort(key=lambda f: float(_get(f, "emotes", 0) or 0), reverse=True)
        return [str(_get(f, "text", "")).strip() for f in with_text[:limit]]

    peak = max(float(_get(f, "chat", 0.0) or 0.0) for f in in_window)
    messages = max(int(_get(f, "messages", 0) or 0) for f in in_window)
    if peak <= 0.0 and messages <= 0:
        return []
    return [f"(chat velocity peaked at {messages} msg/s, intensity {peak:.1f})"]


def _signal_summary(candidate: dict) -> str:
    breakdown = candidate.get("modality_breakdown") or {}
    summary = (
        f"game_label={candidate.get('game_label') or 'none'} "
        f"scene={candidate.get('scene_label') or 'none'} "
        f"reaction_peak={float(candidate.get('peak_value', 0.0) or 0.0):.2f} "
        f"face={float(breakdown.get('face', 0.0) or 0.0):.2f} "
        f"voice={float(breakdown.get('voice', 0.0) or 0.0):.2f} "
        f"burst_peak={float(breakdown.get('burst', 0.0) or 0.0):.2f} "
        f"loud_onset={float(breakdown.get('loud_onset', 0.0) or 0.0):.2f} "
        f"startle={float(breakdown.get('startle', 0.0) or 0.0):.2f} "
        f"game_audio={float(breakdown.get('game_audio', 0.0) or 0.0):.2f} "
        f"chat={float(breakdown.get('chat', 0.0) or 0.0):.2f} "
        f"visible_crowd={float(candidate.get('visible_crowd_reaction', 0.0) or 0.0):.2f} "
        f"signal_agreement={float(candidate.get('signal_agreement', 0.0) or 0.0):.2f} "
        f"event_density={float(candidate.get('event_density', 1.0) or 1.0):.2f}"
    )
    ocr = [str(text).strip() for text in candidate.get("ocr_context", []) if str(text).strip()]
    if ocr:
        snippets = []
        for text in ocr:
            lowered = text.lower()
            for term in _CROWD_REACTION_TERMS:
                cursor = 0
                while len(snippets) < 8:
                    found = lowered.find(term, cursor)
                    if found < 0:
                        break
                    lo = max(0, found - 28)
                    hi = min(len(text), found + len(term) + 42)
                    snippet = " ".join(text[lo:hi].split())
                    if snippet and snippet not in snippets:
                        snippets.append(snippet)
                    cursor = found + len(term)
                if len(snippets) >= 8:
                    break
            if len(snippets) >= 8:
                break
        if snippets:
            summary += " crowd_reaction_snippets=" + " | ".join(snippets)
        else:
            summary += " onscreen_text=" + " | ".join(text[:90] for text in ocr[:3])
    return summary


def _has_nonverbal_context(candidate: dict, chat_lines: List[str]) -> bool:
    """Whether a no-speech moment still has enough evidence to judge."""
    breakdown = candidate.get("modality_breakdown") or {}
    return bool(
        candidate.get("game_label")
        or int(candidate.get("crowd_clip", 0) or 0) > 0
        or candidate.get("ocr_context")
        or chat_lines
        or float(breakdown.get("startle", 0.0) or 0.0) >= 0.35
        or float(breakdown.get("loud_onset", 0.0) or 0.0) >= 0.55
        or float(breakdown.get("burst", 0.0) or 0.0) >= 0.45
    )


def build_prompt(candidate: dict, window_words: List[str],
                 chat_lines: List[str], game_context: Optional[str] = None) -> str:
    lines = [
        "You are reviewing a candidate clip from a live gaming stream for a",
        "highlights editor. Judge from the transcript, on-screen text, chat,",
        "and measured reaction signals below whether this works",
        "as a standalone clip for a viewer with zero stream context.",
        "On-screen OCR may contain viewer chat; repeated laughter, startle,",
        "or 'clip it' comments are direct crowd evidence. A scream, funny noise,",
        "or scare can have a payoff without spoken transcript when signals and",
        "crowd reactions corroborate it.",
        "",
    ]
    if game_context:
        lines.append(f"Game: {game_context}")
    lines.append(f"Signals: {_signal_summary(candidate)}")
    duration = float(candidate.get("end", 0.0)) - float(candidate.get("start", 0.0))
    lines += [
        "",
        f"Streamer transcript ({duration:.0f}s clip window):",
        " ".join(window_words) if window_words else "(no clear speech in this moment)",
    ]
    if chat_lines:
        lines += ["", "Chat during the clip:"]
        lines += [f"- {c}" for c in chat_lines]
    lines += [
        "",
        "Rate the clip:",
        "- hook_strength (0-1): do the first few seconds grab a cold viewer?",
        "- self_contained (0-1): comprehensible with no outside context?",
        "- payoff (0-1): does something actually land (punchline, play, scare)?",
        "- moment_type: funny, clutch, fail, rage, scare, story, wholesome, or filler.",
        "- title: max 8 words, in the streamer's voice.",
        "- hook_line: max 7 words, for an on-screen caption.",
        '- verdict: "post" (clearly worth posting), "maybe", or "skip".',
        'Casual conversation with no punchline or payoff is "filler" + "skip".',
        "Respond with JSON only.",
    ]
    return "\n".join(lines)


def build_batch_prompt(items, game_context: Optional[str] = None) -> str:
    """One compact prompt for several independent candidate verdicts."""
    lines = [
        "You are reviewing candidate clips from one live gaming stream.",
        "Judge each independently for a cold viewer using transcript, on-screen",
        "text, chat, and measured reaction signals. Do not let one strong clip",
        "raise another clip's score.",
        "On-screen OCR may contain viewer chat. Repeated laughter/startle terms",
        "are direct crowd evidence. Nonverbal scares and funny noises can have a",
        "payoff without spoken transcript when multiple signals corroborate it.",
    ]
    if game_context:
        lines.append(f"Game: {game_context}")
    for ordinal, (candidate, window_words, chat_lines) in enumerate(items):
        duration = float(candidate.get("end", 0.0)) - float(candidate.get("start", 0.0))
        lines += [
            "",
            f"CANDIDATE {ordinal}",
            f"Signals: {_signal_summary(candidate)}",
            f"Transcript ({duration:.0f}s): "
            + (" ".join(window_words) if window_words else "(no clear speech)"),
        ]
        if chat_lines:
            lines.append("Chat: " + " | ".join(chat_lines))
    lines += [
        "",
        "For every candidate return h=hook strength, s=self-contained, p=payoff",
        "(all 0-1), m=moment type (funny, clutch, fail, rage, scare, story, wholesome,",
        'or filler), and v=("post", "maybe", or "skip"). Casual conversation',
        'without a payoff is "filler" + "skip".',
        "Return verdicts in candidate order.",
        "Respond with JSON only.",
    ]
    return "\n".join(lines)


def _is_context_overflow(exc: Exception) -> bool:
    """Whether llama.cpp rejected a request before generation for context size."""
    message = str(exc).lower()
    return (
        isinstance(exc, ValueError)
        and "context window" in message
        and ("exceed" in message or "too many" in message)
    )


def _write_health(
    health: Optional[dict],
    *,
    candidates_received: int,
    eligible_candidates: int,
    judged_candidates: int,
    batch_attempts: int = 0,
    batch_failures: int = 0,
    generation_attempts: int = 0,
    generation_failures: int = 0,
    context_overflows: int = 0,
    invalid_outputs: int = 0,
    aborted: bool = False,
) -> None:
    """Populate the caller-owned per-scan health snapshot.

    ``fallback_candidates`` deliberately counts only candidates that had enough
    evidence to ask the model but received no valid verdict. Candidates with no
    text or concrete nonverbal evidence were never judge work and must not make
    a healthy model look degraded.
    """
    if health is None:
        return
    fallback_candidates = max(0, eligible_candidates - judged_candidates)
    health.clear()
    health.update({
        "candidates_received": int(candidates_received),
        "eligible_candidates": int(eligible_candidates),
        "judged_candidates": int(judged_candidates),
        "fallback_candidates": int(fallback_candidates),
        "batch_attempts": int(batch_attempts),
        "batch_failures": int(batch_failures),
        "generation_attempts": int(generation_attempts),
        "generation_failures": int(generation_failures),
        "context_overflows": int(context_overflows),
        "invalid_outputs": int(invalid_outputs),
        "aborted": bool(aborted),
        "degraded": bool(aborted or fallback_candidates > 0),
    })


_HEALTH_COUNTERS = (
    "candidates_received", "eligible_candidates", "judged_candidates",
    "fallback_candidates", "batch_attempts", "batch_failures",
    "generation_attempts", "generation_failures", "context_overflows",
    "invalid_outputs",
)


def accumulate_health(total: dict, current: dict) -> dict:
    """Merge one judge invocation into its scan-level health snapshot."""
    for key in _HEALTH_COUNTERS:
        total[key] = int(total.get(key, 0) or 0) + int(current.get(key, 0) or 0)
    total["aborted"] = bool(total.get("aborted") or current.get("aborted"))
    total["degraded"] = bool(total.get("degraded") or current.get("degraded"))
    if current.get("error"):
        total["error"] = current["error"]
    return total


def judge_candidates(
    candidates: List[dict],
    transcript: Optional[dict],
    chat_frames=None,
    game_context: Optional[str] = None,
    backend=None,
    max_candidates: Optional[int] = None,
    cancel_check=None,
    health: Optional[dict] = None,
) -> Dict[int, SemanticVerdict]:
    """Judge each candidate's transcript window. Returns {candidate_index:
    SemanticVerdict} — indices into ``candidates`` as passed. No-speech
    candidates are judged only when concrete nonverbal evidence is available.
    """
    if backend is None:
        _write_health(
            health,
            candidates_received=len(candidates), eligible_candidates=0,
            judged_candidates=0,
        )
        return {}

    words = _transcript_words(transcript)

    generate_batch = getattr(backend, "generate_batch", None)
    if callable(generate_batch):
        prepared = []
        for idx, candidate in enumerate(candidates):
            if max_candidates is not None and len(prepared) >= max_candidates:
                break
            if cancel_check:
                cancel_check()
            # Match the VLM's prepared causal/editorial window. Scoring fields
            # still describe the compact candidate arc; only the evidence shown
            # to both judges expands, preventing one judge from reading the
            # payoff while the other sees only its setup.
            lo = float(
                candidate.get("judge_start", candidate.get("start", 0.0))
            ) - WINDOW_PAD_SEC
            hi = float(
                candidate.get("judge_end", candidate.get("end", 0.0))
            ) + WINDOW_PAD_SEC
            window_words = _words_in_window(words, lo, hi)
            chat_lines = _chat_lines(chat_frames, lo, hi)
            if not window_words and not _has_nonverbal_context(candidate, chat_lines):
                continue
            prepared.append((idx, candidate, window_words, chat_lines))

        verdicts: Dict[int, SemanticVerdict] = {}
        consecutive_failures = 0
        abort = False
        batch_attempts = 0
        batch_failures = 0
        context_overflows = 0
        invalid_outputs = 0
        for offset in range(0, len(prepared), BATCH_SIZE):
            if cancel_check:
                cancel_check()
            # (batch, regenerated): a malformed-JSON completion earns exactly ONE
            # regeneration of the same batch before falling back to splitting it
            # (the same mechanism context overflows use). Halves are new prompts,
            # so they get a fresh single-retry budget; the sequence is bounded at
            # 2 attempts per node of the split tree.
            pending = [(prepared[offset:offset + BATCH_SIZE], False)]
            while pending:
                batch, regenerated = pending.pop(0)
                prompt = build_batch_prompt(
                    [(candidate, window_words, chat_lines)
                     for _, candidate, window_words, chat_lines in batch],
                    game_context=game_context,
                )
                try:
                    batch_attempts += 1
                    raw_batch = generate_batch(prompt, len(batch))
                except InterruptedError:
                    raise
                except MalformedCompletionError as exc:
                    # The model answered, but the JSON transport broke (usually
                    # truncation) and backend-side salvage could not repair it.
                    # A responding-but-sloppy model is not a broken runtime, so
                    # like context overflows this never feeds the abort counter.
                    batch_failures += 1
                    consecutive_failures = 0
                    if not regenerated:
                        print(
                            "Semantic judge batch returned malformed JSON "
                            f"(offset {offset}, size {len(batch)}); regenerating once."
                        )
                        pending.insert(0, (batch, True))
                        continue
                    if len(batch) > 1:
                        midpoint = len(batch) // 2
                        print(
                            "Semantic judge batch malformed after retry "
                            f"(offset {offset}, size {len(batch)}); splitting batch."
                        )
                        pending[0:0] = [(batch[:midpoint], False), (batch[midpoint:], False)]
                        continue
                    invalid_outputs += 1
                    print(
                        "Semantic judge dropped candidate "
                        f"(index {batch[0][0]}) after salvage, regeneration, "
                        f"and split all failed: {exc}"
                    )
                    continue
                except Exception as exc:  # noqa: BLE001 - optional local model
                    batch_failures += 1
                    if _is_context_overflow(exc):
                        context_overflows += 1
                        if len(batch) > 1:
                            midpoint = len(batch) // 2
                            print(
                                "Semantic judge batch exceeded model context "
                                f"(offset {offset}, size {len(batch)}); retrying smaller batches."
                            )
                            pending[0:0] = [(batch[:midpoint], False), (batch[midpoint:], False)]
                        else:
                            # This candidate's own evidence is too large. Skip it,
                            # but do not treat content length as a broken runtime.
                            print(
                                "Semantic judge candidate exceeded model context "
                                f"(index {batch[0][0]}); skipping verdict."
                            )
                        consecutive_failures = 0
                        continue
                    consecutive_failures += 1
                    print(f"Semantic judge batch failed (offset {offset}): {exc}")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print("Semantic judge aborted: backend failing consistently.")
                        abort = True
                        break
                    continue
                consecutive_failures = 0
                if not isinstance(raw_batch, list):
                    invalid_outputs += len(batch)
                    continue
                for (idx, _candidate, _words, _chat), raw in zip(batch, raw_batch):
                    verdict = SemanticVerdict.from_raw(raw)
                    if verdict is not None:
                        verdicts[idx] = verdict
                    else:
                        invalid_outputs += 1
                if len(raw_batch) < len(batch):
                    invalid_outputs += len(batch) - len(raw_batch)
            if abort:
                break
        _write_health(
            health,
            candidates_received=len(candidates),
            eligible_candidates=len(prepared),
            judged_candidates=len(verdicts),
            batch_attempts=batch_attempts,
            batch_failures=batch_failures,
            context_overflows=context_overflows,
            invalid_outputs=invalid_outputs,
            aborted=abort,
        )
        return verdicts

    verdicts: Dict[int, SemanticVerdict] = {}
    consecutive_failures = 0
    generations = 0
    eligible_candidates = 0
    generation_failures = 0
    invalid_outputs = 0
    aborted = False
    for idx, candidate in enumerate(candidates):
        if max_candidates is not None and generations >= max_candidates:
            break
        if cancel_check:
            cancel_check()

        lo = float(
            candidate.get("judge_start", candidate.get("start", 0.0))
        ) - WINDOW_PAD_SEC
        hi = float(
            candidate.get("judge_end", candidate.get("end", 0.0))
        ) + WINDOW_PAD_SEC
        window_words = _words_in_window(words, lo, hi)
        chat_lines = _chat_lines(chat_frames, lo, hi)
        if not window_words and not _has_nonverbal_context(candidate, chat_lines):
            continue
        eligible_candidates += 1

        prompt = build_prompt(
            candidate, window_words,
            chat_lines,
            game_context=game_context,
        )
        generations += 1
        try:
            raw = backend.generate(prompt)
        except InterruptedError:
            raise
        except MalformedCompletionError as exc:
            # Responding-but-sloppy model: count the lost output, never abort.
            generation_failures += 1
            invalid_outputs += 1
            consecutive_failures = 0
            print(f"Semantic judge generation malformed (candidate {idx}): {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - one bad generation must not kill the batch
            generation_failures += 1
            consecutive_failures += 1
            print(f"Semantic judge generation failed (candidate {idx}): {exc}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("Semantic judge aborted: backend failing consistently.")
                aborted = True
                break
            continue
        consecutive_failures = 0

        verdict = SemanticVerdict.from_raw(raw)
        if verdict is not None:
            verdicts[idx] = verdict
        else:
            invalid_outputs += 1
    _write_health(
        health,
        candidates_received=len(candidates),
        eligible_candidates=eligible_candidates,
        judged_candidates=len(verdicts),
        generation_attempts=generations,
        generation_failures=generation_failures,
        invalid_outputs=invalid_outputs,
        aborted=aborted,
    )
    return verdicts

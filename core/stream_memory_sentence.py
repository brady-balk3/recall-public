# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import os
import hashlib
import tempfile
from typing import Callable, Optional, Sequence

import numpy as np

from core.bundle_paths import get_models_dir
from core.stream_memory_semantic import (
    SemanticBuildCancelled,
    SemanticBuildResult,
    SemanticDocument,
)


SENTENCE_INDEX_VERSION = 2
SENTENCE_MODEL_ID = "all-MiniLM-L6-v2-qint8-avx2"
SENTENCE_DIMENSIONS = 384
SENTENCE_INDEX_FILENAME = "stream_memory_sentences.v1.npz"
SENTENCE_MODEL_DIRNAME = "all-minilm-l6-v2"
SENTENCE_MODEL_FILENAME = "model_quint8_avx2.onnx"
SENTENCE_TOKENIZER_FILENAME = "tokenizer.json"
MAX_SEQUENCE_LENGTH = 256


def sentence_model_dir() -> str:
    return os.path.join(get_models_dir(), "memory", SENTENCE_MODEL_DIRNAME)


def sentence_model_paths() -> tuple[str, str]:
    root = sentence_model_dir()
    return (
        os.path.join(root, SENTENCE_MODEL_FILENAME),
        os.path.join(root, SENTENCE_TOKENIZER_FILENAME),
    )


def sentence_model_available() -> bool:
    return all(os.path.isfile(path) for path in sentence_model_paths())


def sentence_model_bytes() -> int:
    return sum(
        os.path.getsize(path) for path in sentence_model_paths()
        if os.path.isfile(path)
    )


def sentence_index_path(data_root: str) -> str:
    return os.path.join(data_root, "memory", SENTENCE_INDEX_FILENAME)


class SentenceEncoder:
    """Small, offline ONNX sentence encoder with no Transformers dependency."""

    def __init__(self):
        self._session = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._session is not None and self._tokenizer is not None:
            return
        model_path, tokenizer_path = sentence_model_paths()
        if not os.path.isfile(model_path) or not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(
                "The local Stream Memory sentence model is not installed."
            )
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self._session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = tokenizer

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        self._load()
        assert self._session is not None
        assert self._tokenizer is not None
        vectors: list[np.ndarray] = []
        input_names = {item.name for item in self._session.get_inputs()}
        for offset in range(0, len(texts), max(1, batch_size)):
            if cancel_check is not None and cancel_check():
                raise SemanticBuildCancelled("Stream Memory sentence encoding was preempted.")
            batch = [str(text or "") for text in texts[offset:offset + batch_size]]
            encodings = self._tokenizer.encode_batch(batch)
            input_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
            attention_mask = np.asarray(
                [item.attention_mask for item in encodings], dtype=np.int64,
            )
            token_type_ids = np.asarray(
                [item.type_ids for item in encodings], dtype=np.int64,
            )
            feeds = {}
            if "input_ids" in input_names:
                feeds["input_ids"] = input_ids
            if "attention_mask" in input_names:
                feeds["attention_mask"] = attention_mask
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = token_type_ids
            outputs = self._session.run(None, feeds)
            sentence_output = next(
                (
                    np.asarray(value, dtype=np.float32)
                    for value in outputs
                    if np.asarray(value).ndim == 2
                    and np.asarray(value).shape[0] == len(batch)
                    and np.asarray(value).shape[1] == SENTENCE_DIMENSIONS
                ),
                None,
            )
            if sentence_output is None:
                token_output = next(
                    (
                        np.asarray(value, dtype=np.float32)
                        for value in outputs
                        if np.asarray(value).ndim == 3
                        and np.asarray(value).shape[0] == len(batch)
                        and np.asarray(value).shape[2] == SENTENCE_DIMENSIONS
                    ),
                    None,
                )
                if token_output is None:
                    raise RuntimeError("The sentence model returned an unsupported output shape.")
                mask = attention_mask[:, :, None].astype(np.float32)
                sentence_output = (token_output * mask).sum(axis=1) / np.clip(
                    mask.sum(axis=1), 1e-9, None,
                )
            norms = np.linalg.norm(sentence_output, axis=1, keepdims=True)
            sentence_output = np.divide(
                sentence_output,
                norms,
                out=np.zeros_like(sentence_output),
                where=norms > 1e-9,
            )
            vectors.append(sentence_output)
        if cancel_check is not None and cancel_check():
            raise SemanticBuildCancelled("Stream Memory sentence encoding was preempted.")
        if not vectors:
            return np.empty((0, SENTENCE_DIMENSIONS), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


def build_sentence_index(
    documents: Sequence[SemanticDocument],
    path: str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> SemanticBuildResult:
    """Atomically update the sentence index, encoding only changed documents."""
    if len(documents) < 2:
        raise ValueError("At least two searchable moments are needed to build the concept index.")
    if cancel_check is not None and cancel_check():
        raise SemanticBuildCancelled("Stream Memory sentence encoding was preempted.")

    document_hashes = [
        hashlib.sha256(str(document.text or "").encode("utf-8")).hexdigest()
        for document in documents
    ]
    reusable_by_hash: dict[str, np.ndarray] = {}
    reusable_by_id: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as payload:
            version = int(payload["version"][0])
            model_id = str(payload["model_id"][0])
            if version in {1, SENTENCE_INDEX_VERSION} and model_id == SENTENCE_MODEL_ID:
                previous_ids = payload["ids"].astype(str)
                previous_vectors = payload["vectors"].astype(np.float32)
                if version >= 2 and "text_hashes" in payload.files:
                    previous_hashes = payload["text_hashes"].astype(str)
                    reusable_by_hash = {
                        value: previous_vectors[index]
                        for index, value in enumerate(previous_hashes.tolist())
                    }
                else:
                    reusable_by_id = {
                        value: previous_vectors[index]
                        for index, value in enumerate(previous_ids.tolist())
                    }
    except (OSError, KeyError, ValueError):
        # A missing, old, or damaged sidecar is rebuildable from SQLite truth.
        reusable_by_hash = {}
        reusable_by_id = {}

    vectors = np.empty((len(documents), SENTENCE_DIMENSIONS), dtype=np.float32)
    missing_positions: list[int] = []
    for index, document in enumerate(documents):
        vector = reusable_by_hash.get(document_hashes[index])
        if vector is None:
            vector = reusable_by_id.get(document.entry_id)
        if vector is None:
            missing_positions.append(index)
        else:
            vectors[index] = vector

    encoder = SentenceEncoder()
    if missing_positions:
        encoded = encoder.encode(
            [documents[index].text for index in missing_positions],
            cancel_check=cancel_check,
        )
        for offset, position in enumerate(missing_positions):
            vectors[position] = encoded[offset]
    if cancel_check is not None and cancel_check():
        raise SemanticBuildCancelled("Stream Memory sentence encoding was preempted.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temp_path = tempfile.mkstemp(
        prefix="stream-memory-sentences-", suffix=".npz", dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(handle, "wb") as destination:
            max_id_length = max(1, max(len(document.entry_id) for document in documents))
            np.savez_compressed(
                destination,
                version=np.asarray([SENTENCE_INDEX_VERSION], dtype=np.int16),
                model_id=np.asarray([SENTENCE_MODEL_ID]),
                ids=np.asarray(
                    [document.entry_id for document in documents],
                    dtype=f"U{max_id_length}",
                ),
                text_hashes=np.asarray(document_hashes, dtype="U64"),
                vectors=vectors.astype(np.float16),
            )
        if cancel_check is not None and cancel_check():
            raise SemanticBuildCancelled("Stream Memory sentence encoding was preempted.")
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return SemanticBuildResult(
        entries=len(documents),
        dimensions=SENTENCE_DIMENSIONS,
        bytes_written=os.path.getsize(path),
        path=path,
        encoded_entries=len(missing_positions),
        reused_entries=len(documents) - len(missing_positions),
    )


class SentenceIndex:
    """Lazy local cosine index backed by float16 vectors on disk."""

    def __init__(self, path: str):
        self.path = path
        self._signature: Optional[tuple[int, int]] = None
        self._ids: Optional[np.ndarray] = None
        self._positions: Optional[dict[str, int]] = None
        self._vectors: Optional[np.ndarray] = None
        self._encoder = SentenceEncoder()
        self._last_query: Optional[str] = None
        self._last_query_vector: Optional[np.ndarray] = None

    def clear(self) -> None:
        self._signature = None
        self._ids = None
        self._positions = None
        self._vectors = None

    def _query_vector(self, query: str) -> np.ndarray:
        if self._last_query == query and self._last_query_vector is not None:
            return self._last_query_vector
        vector = self._encoder.encode([query], batch_size=1)[0]
        self._last_query = query
        self._last_query_vector = vector
        return vector

    def _load(self) -> bool:
        try:
            stat = os.stat(self.path)
        except OSError:
            self.clear()
            return False
        signature = (stat.st_mtime_ns, stat.st_size)
        if self._signature == signature and self._vectors is not None:
            return True
        with np.load(self.path, allow_pickle=False) as payload:
            if int(payload["version"][0]) not in {1, SENTENCE_INDEX_VERSION}:
                self.clear()
                return False
            ids = payload["ids"].astype(str)
            vectors = payload["vectors"].astype(np.float32)
        self._signature = signature
        self._ids = ids
        self._positions = {entry_id: index for index, entry_id in enumerate(ids.tolist())}
        self._vectors = vectors
        return True

    def search(
        self,
        query: str,
        *,
        allowed_ids: Optional[set[str]] = None,
        limit: int = 200,
    ) -> list[tuple[str, float]]:
        if not self._load():
            return []
        assert self._ids is not None
        assert self._positions is not None
        assert self._vectors is not None
        query_vector = self._query_vector(query)
        if allowed_ids is not None:
            allowed_positions = [
                self._positions[entry_id]
                for entry_id in allowed_ids
                if entry_id in self._positions
            ]
            if not allowed_positions:
                return []
            positions = np.asarray(allowed_positions, dtype=np.int64)
            candidate_scores = self._vectors[positions] @ query_vector
        else:
            positions = np.arange(len(self._vectors), dtype=np.int64)
            candidate_scores = self._vectors @ query_vector
        take = min(max(1, int(limit)), len(positions))
        if take < len(positions):
            local = np.argpartition(candidate_scores, -take)[-take:]
            positions = positions[local]
            candidate_scores = candidate_scores[local]
        order = np.argsort(candidate_scores)[::-1]
        return [
            (str(self._ids[int(positions[index])]), float(candidate_scores[index]))
            for index in order
            if float(candidate_scores[index]) >= 0.18
        ]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import os
import re
import tempfile
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy import sparse


SEMANTIC_INDEX_VERSION = 3
SEMANTIC_MODEL_ID = "recall-lsi-hash-v3"
SEMANTIC_DIMENSIONS = 192
SEMANTIC_FEATURES = 32768
SEMANTIC_FILENAME = "stream_memory_semantic.v3.npz"
_OVERSAMPLE = 12
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "i", "if", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "there", "they", "this", "to", "up", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "with",
    "you", "your", "every", "find", "moment", "moments", "search", "show",
    "time", "times",
}
_TERM_NORMALIZATION = {
    "sniped": "snipe",
    "snipes": "snipe",
    "sniping": "snipe",
    "snipers": "sniper",
    "headshots": "headshot",
    "noscope": "no-scope",
    "noscopes": "no-scope",
}


@dataclass(frozen=True)
class SemanticDocument:
    entry_id: str
    text: str


@dataclass(frozen=True)
class SemanticBuildResult:
    entries: int
    dimensions: int
    bytes_written: int
    path: str
    encoded_entries: int = 0
    reused_entries: int = 0


class SemanticBuildCancelled(InterruptedError):
    """Raised when foreground work preempts a rebuildable Memory index update."""


def _raise_if_cancelled(cancel_check: Optional[Callable[[], bool]]) -> None:
    if cancel_check is not None and cancel_check():
        raise SemanticBuildCancelled("Stream Memory concept-index update was preempted.")


def semantic_index_path(data_root: str) -> str:
    return os.path.join(data_root, "memory", SEMANTIC_FILENAME)


def _terms(text: str) -> list[str]:
    tokens = [
        _TERM_NORMALIZATION.get(token.lower(), token.lower())
        for token in _TOKEN_RE.findall(str(text or ""))
        if len(token) > 1 and token.lower() not in _STOP_WORDS
    ]
    return tokens + [
        f"{tokens[index]}::{tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    ]


def _feature(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) % SEMANTIC_FEATURES


def _fingerprint(term: str) -> np.uint64:
    """Use two independent checksums to reject unseen hashed query terms."""
    crc = zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF
    adler = zlib.adler32(term.encode("utf-8")) & 0xFFFFFFFF
    return np.uint64((crc << 32) | adler)


def _contains_fingerprint(fingerprints: np.ndarray, value: np.uint64) -> bool:
    position = int(np.searchsorted(fingerprints, value))
    return position < len(fingerprints) and fingerprints[position] == value


def _hashed_counts(
    texts: Iterable[str],
    *,
    allowed_fingerprints: Optional[np.ndarray] = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    indices: list[int] = []
    values: list[float] = []
    indptr = [0]
    document_frequency = np.zeros(SEMANTIC_FEATURES, dtype=np.int32)
    term_fingerprints: set[int] = set()
    rows = 0
    for text in texts:
        counts: dict[int, int] = {}
        for term in _terms(text):
            fingerprint = _fingerprint(term)
            if (
                allowed_fingerprints is not None
                and not _contains_fingerprint(allowed_fingerprints, fingerprint)
            ):
                continue
            term_fingerprints.add(int(fingerprint))
            feature = _feature(term)
            counts[feature] = counts.get(feature, 0) + 1
        for feature, count in sorted(counts.items()):
            indices.append(feature)
            values.append(1.0 + np.log(float(count)))
            document_frequency[feature] += 1
        indptr.append(len(indices))
        rows += 1
    matrix = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(rows, SEMANTIC_FEATURES),
        dtype=np.float32,
    )
    return (
        matrix,
        document_frequency,
        np.asarray(sorted(term_fingerprints), dtype=np.uint64),
    )


def _apply_tfidf(
    matrix: sparse.csr_matrix,
    document_frequency: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    count = max(1, matrix.shape[0])
    idf = (np.log((1.0 + count) / (1.0 + document_frequency)) + 1.0).astype(np.float32)
    matrix = matrix.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
    inverse = np.zeros_like(norms, dtype=np.float32)
    np.divide(1.0, norms, out=inverse, where=norms > 0)
    return sparse.diags(inverse, dtype=np.float32).dot(matrix).tocsr(), idf


def _randomized_lsi(
    matrix: sparse.csr_matrix,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    rank = max(1, min(dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1))
    sample_size = min(rank + _OVERSAMPLE, matrix.shape[0], matrix.shape[1])
    generator = np.random.default_rng(20260825)
    omega = generator.standard_normal(
        (matrix.shape[1], sample_size), dtype=np.float32,
    )
    projected = np.asarray(matrix @ omega, dtype=np.float32)
    # One power iteration sharpens the concept basis without the dependency and
    # packaging cost of a separate machine-learning runtime.
    projected = np.asarray(matrix @ (matrix.T @ projected), dtype=np.float32)
    basis, _ = np.linalg.qr(projected, mode="reduced")
    compressed = np.asarray(basis.T @ matrix, dtype=np.float32)
    _u, _singular, components = np.linalg.svd(compressed, full_matrices=False)
    components = np.asarray(components[:rank], dtype=np.float32)
    vectors = np.asarray(matrix @ components.T, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors, components


def build_semantic_index(
    documents: Sequence[SemanticDocument],
    path: str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> SemanticBuildResult:
    _raise_if_cancelled(cancel_check)
    if len(documents) < 2:
        raise ValueError("At least two searchable moments are needed to build the concept index.")
    matrix, document_frequency, term_fingerprints = _hashed_counts(
        document.text for document in documents
    )
    _raise_if_cancelled(cancel_check)
    if matrix.nnz == 0:
        raise ValueError("The searchable moments do not contain enough words to build the concept index.")
    matrix, idf = _apply_tfidf(matrix, document_frequency)
    _raise_if_cancelled(cancel_check)
    vectors, components = _randomized_lsi(matrix, SEMANTIC_DIMENSIONS)
    _raise_if_cancelled(cancel_check)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temp_path = tempfile.mkstemp(
        prefix="stream-memory-", suffix=".npz", dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(handle, "wb") as destination:
            max_id_length = max(1, max(len(document.entry_id) for document in documents))
            np.savez_compressed(
                destination,
                version=np.asarray([SEMANTIC_INDEX_VERSION], dtype=np.int16),
                model_id=np.asarray([SEMANTIC_MODEL_ID]),
                ids=np.asarray(
                    [document.entry_id for document in documents],
                    dtype=f"U{max_id_length}",
                ),
                vectors=vectors.astype(np.float16),
                components=components.astype(np.float16),
                idf=idf.astype(np.float32),
                term_fingerprints=term_fingerprints,
            )
        _raise_if_cancelled(cancel_check)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return SemanticBuildResult(
        entries=len(documents),
        dimensions=int(vectors.shape[1]),
        bytes_written=os.path.getsize(path),
        path=path,
        encoded_entries=len(documents),
    )


class SemanticIndex:
    """Lazy, read-only view of a rebuildable local concept index."""

    def __init__(self, path: str):
        self.path = path
        self._signature: Optional[tuple[int, int]] = None
        self._ids: Optional[np.ndarray] = None
        self._positions: Optional[dict[str, int]] = None
        self._vectors: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._idf: Optional[np.ndarray] = None
        self._term_fingerprints: Optional[np.ndarray] = None

    def clear(self) -> None:
        self._signature = None
        self._ids = None
        self._positions = None
        self._vectors = None
        self._components = None
        self._idf = None
        self._term_fingerprints = None

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
            version = int(payload["version"][0])
            if version != SEMANTIC_INDEX_VERSION:
                self.clear()
                return False
            ids = payload["ids"].astype(str)
            vectors = payload["vectors"].astype(np.float32)
            components = payload["components"].astype(np.float32)
            idf = payload["idf"].astype(np.float32)
            term_fingerprints = payload["term_fingerprints"].astype(np.uint64)
        self._signature = signature
        self._ids = ids
        self._positions = {entry_id: index for index, entry_id in enumerate(ids.tolist())}
        self._vectors = vectors
        self._components = components
        self._idf = idf
        self._term_fingerprints = term_fingerprints
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
        assert self._components is not None
        assert self._idf is not None
        assert self._term_fingerprints is not None
        query_matrix, _, _ = _hashed_counts(
            [query], allowed_fingerprints=self._term_fingerprints,
        )
        if query_matrix.nnz == 0:
            return []
        query_matrix = query_matrix.multiply(self._idf).tocsr()
        query_vector = np.asarray(query_matrix @ self._components.T, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query_vector))
        if norm <= 1e-9:
            return []
        query_vector /= norm
        scores = self._vectors @ query_vector
        if allowed_ids is not None:
            allowed_positions = [
                self._positions[entry_id]
                for entry_id in allowed_ids
                if entry_id in self._positions
            ]
            if not allowed_positions:
                return []
            positions = np.asarray(allowed_positions, dtype=np.int64)
            candidate_scores = scores[positions]
        else:
            positions = np.arange(len(scores), dtype=np.int64)
            candidate_scores = scores
        take = min(max(1, int(limit)), len(positions))
        if take < len(positions):
            local = np.argpartition(candidate_scores, -take)[-take:]
            positions = positions[local]
            candidate_scores = candidate_scores[local]
        order = np.argsort(candidate_scores)[::-1]
        results = []
        for index in order:
            score = float(candidate_scores[index])
            # Near-zero latent similarity is noise rather than a useful memory.
            if score <= 0.02:
                continue
            position = int(positions[index])
            results.append((str(self._ids[position]), score))
        return results

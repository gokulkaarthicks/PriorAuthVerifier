"""Internal retrieval domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.models.chunk import Chunk
from app.models.section import Section


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A ranked retrieval hit for a chunk."""

    chunk: Chunk
    rank: int
    score: float
    retrieval_method: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


@dataclass(slots=True)
class HybridIndex:
    """In-memory hybrid BM25 + dense index over a narrowed chunk set."""

    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: Any
    retrieval_texts: list[str]


@dataclass(slots=True)
class SectionBm25Index:
    """Request-scoped BM25 index over compact section texts."""

    sections: list[Section]
    bm25: Any
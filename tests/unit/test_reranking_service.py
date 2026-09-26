"""Unit tests for RerankingService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import RerankingError
from app.models.chunk import Chunk
from app.models.retrieval import RetrievalResult
from app.services.reranking_service import RerankingService


def _result(
    chunk_id: str,
    text: str,
    *,
    rank: int,
    heading: str | None = "Heading",
) -> RetrievalResult:
    chunk = Chunk(
        id=chunk_id,
        document_id="policy-doc",
        text=text,
        page_number=2,
        end_page_number=3,
        heading=heading,
        parent_heading="Parent",
        token_count=12,
        section_id=f"sec-{chunk_id}",
        chunk_index=0,
    )
    return RetrievalResult(
        chunk=chunk,
        rank=rank,
        score=0.5,
        retrieval_method="rrf",
        dense_rank=rank,
        bm25_rank=rank,
        rrf_score=0.02,
    )


def test_reranks_only_supplied_candidates_and_returns_top_k() -> None:
    candidates = [
        _result("c1", "alpha text", rank=1),
        _result("c2", "beta text", rank=2),
        _result("c3", "gamma text", rank=3),
        _result("c4", "delta text", rank=4),
        _result("c5", "epsilon text", rank=5),
        _result("c6", "zeta text", rank=6),
    ]
    service = RerankingService(settings=Settings(RERANK_TOP_K=5))
    fake_model = MagicMock()
    fake_model.compute_score.return_value = [0.1, 0.2, 0.9, 0.4, 0.3, 0.8]
    service._model = fake_model

    reranked = service.rerank("query", candidates, top_k=5)

    assert len(reranked) == 5
    assert [item.chunk.id for item in reranked] == ["c3", "c6", "c4", "c5", "c2"]
    assert all(item.retrieval_method == "rerank" for item in reranked)
    pairs = fake_model.compute_score.call_args.args[0]
    assert len(pairs) == 6
    assert pairs[0][0] == "query"
    assert "Section:" in pairs[0][1]
    assert "Parent:" in pairs[0][1]
    assert "Evidence:" in pairs[0][1]


def test_preserves_chunk_metadata() -> None:
    candidate = _result("c1", "policy body", rank=1, heading="Coverage")
    service = RerankingService(settings=Settings(RERANK_TOP_K=5))
    service._model = MagicMock(compute_score=MagicMock(return_value=[0.77]))

    reranked = service.rerank("query", [candidate])

    assert len(reranked) == 1
    assert reranked[0].chunk.heading == "Coverage"
    assert reranked[0].chunk.page_number == 2
    assert reranked[0].chunk.end_page_number == 3
    assert reranked[0].dense_rank == 1
    assert reranked[0].bm25_rank == 1
    assert reranked[0].rrf_score == 0.02
    assert reranked[0].reranker_score == pytest.approx(0.77)


def test_stable_sorting_on_tied_scores() -> None:
    candidates = [
        _result("c1", "one", rank=1),
        _result("c2", "two", rank=2),
        _result("c3", "three", rank=3),
    ]
    service = RerankingService(settings=Settings(RERANK_TOP_K=3))
    service._model = MagicMock(compute_score=MagicMock(return_value=[0.5, 0.5, 0.5]))

    first = service.rerank("query", candidates)
    second = service.rerank("query", candidates)

    assert [item.chunk.id for item in first] == ["c1", "c2", "c3"]
    assert [item.chunk.id for item in second] == ["c1", "c2", "c3"]


def test_empty_candidates_return_empty() -> None:
    service = RerankingService(settings=Settings())
    assert service.rerank("query", []) == []


def test_model_loads_lazily_once() -> None:
    service = RerankingService(settings=Settings(RERANKER_MODEL="fake-reranker"))
    fake_model = MagicMock(compute_score=MagicMock(return_value=[0.5]))

    with patch(
        "FlagEmbedding.FlagReranker",
        return_value=fake_model,
    ) as ctor:
        first = service.rerank("query", [_result("c1", "one", rank=1)])
        second = service.rerank("query", [_result("c2", "two", rank=1)])

    assert first[0].chunk.id == "c1"
    assert second[0].chunk.id == "c2"
    ctor.assert_called_once_with("fake-reranker", use_fp16=False)
    assert service.init_count == 1


def test_model_failure_raises_application_error() -> None:
    service = RerankingService(settings=Settings(RERANKER_MODEL="fake-reranker"))
    with (
        patch(
            "FlagEmbedding.FlagReranker",
            side_effect=RuntimeError("load failed"),
        ),
        pytest.raises(RerankingError, match="Failed to load reranker"),
    ):
        service.rerank("query", [_result("c1", "text", rank=1)])


def test_compute_score_failure_raises_application_error() -> None:
    service = RerankingService(settings=Settings())
    broken = MagicMock()
    broken.compute_score.side_effect = RuntimeError("score failed")
    service._model = broken

    with pytest.raises(RerankingError, match="Failed to rerank"):
        service.rerank("query", [_result("c1", "text", rank=1)])

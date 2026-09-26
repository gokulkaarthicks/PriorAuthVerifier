"""Integration tests for canonical policy governing-criteria retrieval."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.config import Settings, get_settings
from app.models.document import DocumentType
from app.schemas.procedure import ProcedureExtraction
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.heading_service import HeadingService
from app.services.pdf_service import PDFService
from app.services.policy_service import PolicyService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
PATIENT_PDF = SAMPLE_DIR / "patient.pdf"
POLICY_PDF = SAMPLE_DIR / "bcbs_policy.pdf"


def _fake_embedding_service(dim: int = 8) -> MagicMock:
    service = MagicMock()

    def embed(texts: list[str], **_kwargs) -> np.ndarray:
        seed = sum(len(text) for text in texts) + 17 * len(texts)
        rng = np.random.default_rng(seed % (2**32))
        matrix = rng.normal(size=(len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    def similarity(
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        docs = np.asarray(document_embeddings, dtype=np.float32)
        return docs @ query

    service.embed.side_effect = embed
    service.similarity.side_effect = similarity
    return service


@pytest.mark.skipif(
    not POLICY_PDF.exists() or not PATIENT_PDF.exists(),
    reason="sample PDFs missing",
)
def test_canonical_policy_retrieval_on_sample_pdfs() -> None:
    settings = Settings(
        BM25_TOP_K=10,
        DENSE_TOP_K=10,
        RRF_TOP_K=10,
        RERANK_TOP_K=5,
        RRF_K=60,
        POLICY_BM25_TOP_K=8,
        POLICY_DENSE_TOP_K=4,
        POLICY_RERANK_TOP_K=3,
        SECTION_NEIGHBOR_WINDOW=1,
    )
    pdf_service = PDFService(settings=settings)
    heading_service = HeadingService()
    chunking_service = ChunkingService(settings=settings)

    policy_doc = pdf_service.parse(POLICY_PDF, DocumentType.POLICY)
    policy_sections = heading_service.build_sections(policy_doc)
    assert policy_sections

    procedure = ProcedureExtraction(
        procedure_name="Permanent Cardiac Pacemaker Implantation",
        cpt_code="33207",
        status="Pending",
        is_future_procedure=True,
        confidence=0.9,
    )

    embedding_service = _fake_embedding_service()
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        settings=settings,
    )
    reranker = RerankingService(settings=settings)

    def _mock_scores(pairs: list[list[str]]) -> list[float]:
        return [float(len(pairs) - index) for index in range(len(pairs))]

    reranker._model = MagicMock(compute_score=_mock_scores)

    dense_sizes: list[int] = []
    original_dense = retrieval_service._rank_texts_dense
    original_bm25_select = retrieval_service.bm25_select_sections
    bm25_sizes: list[int] = []

    def spy_dense(query: str, texts: list[str], **kwargs) -> list[int]:
        dense_sizes.append(len(texts))
        return original_dense(query, texts, **kwargs)

    def spy_bm25_select(sections, *, query, top_k=None, **kwargs):
        bm25_sizes.append(len(sections))
        return original_bm25_select(
            sections, query=query, top_k=top_k, **kwargs
        )

    retrieval_service._rank_texts_dense = spy_dense  # type: ignore[method-assign]
    retrieval_service.bm25_select_sections = spy_bm25_select  # type: ignore[method-assign]
    retrieval_service.build_hybrid_index = MagicMock(  # type: ignore[method-assign]
        wraps=retrieval_service.build_hybrid_index
    )

    policy_service = PolicyService(
        llm_service=MagicMock(),
        retrieval_service=retrieval_service,
        chunking_service=chunking_service,
        reranking_service=reranker,
        settings=settings,
    )
    candidates = policy_service.retrieve_policy_candidates(
        procedure,
        policy_sections,
    )

    assert candidates
    assert all(item.retrieval_method == "policy_governing" for item in candidates)
    for result in candidates:
        assert result.chunk.page_number >= 1
        assert result.chunk.end_page_number >= result.chunk.page_number
        assert result.chunk.text.strip()

    assert bm25_sizes
    assert bm25_sizes[0] == len(
        [section for section in policy_sections if section.text.strip()]
    )
    assert dense_sizes
    # Dense ranks only the CPT+BM25 union, never a separate full-corpus pass.
    assert all(size <= settings.POLICY_BM25_TOP_K for size in dense_sizes)
    assert all(size < len(policy_sections) for size in dense_sizes)
    retrieval_service.build_hybrid_index.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("RUN_RETRIEVAL_INTEGRATION"),
    reason="Set RUN_RETRIEVAL_INTEGRATION=1 to run heavy model integration",
)
@pytest.mark.skipif(
    not POLICY_PDF.exists(),
    reason="sample bcbs_policy.pdf missing",
)
async def test_live_embedding_and_reranker_models() -> None:
    settings = get_settings()
    policy_doc = PDFService(settings=settings).parse(
        POLICY_PDF,
        DocumentType.POLICY,
    )
    policy_sections = HeadingService().build_sections(policy_doc)
    embedding_service = EmbeddingService(settings=settings)
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        settings=settings,
    )
    procedure = ProcedureExtraction(
        procedure_name="Permanent Cardiac Pacemaker Implantation",
        cpt_code="33207",
        is_future_procedure=True,
        confidence=0.9,
    )
    candidates = PolicyService(
        llm_service=MagicMock(),
        retrieval_service=retrieval_service,
        chunking_service=ChunkingService(settings=settings),
        reranking_service=RerankingService(settings=settings),
        settings=settings,
    ).retrieve_policy_candidates(procedure, policy_sections)
    assert 1 <= len(candidates) <= settings.POLICY_RERANK_TOP_K * 5

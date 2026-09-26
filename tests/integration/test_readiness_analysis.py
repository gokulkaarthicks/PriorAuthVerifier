"""Integration coverage for NOT_APPLICABLE readiness on sample documents."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.core.config import Settings
from app.models.document import DocumentType
from app.schemas.analysis import ReadinessStatus
from app.schemas.policy import PolicyCriteriaResult
from app.schemas.procedure import ProcedureExtraction
from app.services.chunking_service import ChunkingService
from app.services.evidence_service import EvidenceService
from app.services.heading_service import HeadingService
from app.services.pdf_service import PDFService
from app.services.policy_service import PolicyService
from app.services.readiness_service import ReadinessService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
POLICY_PDF = SAMPLE_DIR / "bcbs_policy.pdf"
PATIENT_PDF = SAMPLE_DIR / "patient.pdf"


def _fake_embedding_service(dim: int = 8) -> MagicMock:
    service = MagicMock()

    def embed(texts: list[str], **_kwargs) -> np.ndarray:
        seed = sum(len(text) for text in texts) + 29 * len(texts)
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
@pytest.mark.asyncio
async def test_sample_documents_readiness_not_applicable() -> None:
    settings = Settings(
        BM25_TOP_K=10,
        DENSE_TOP_K=10,
        RRF_TOP_K=10,
        RERANK_TOP_K=5,
        POLICY_BM25_TOP_K=40,
        POLICY_DENSE_TOP_K=15,
        POLICY_RERANK_TOP_K=5,
        SECTION_NEIGHBOR_WINDOW=1,
        POLICY_REASONING_MODEL="gpt-6-sol",
    )
    pdf_service = PDFService(settings=settings)
    heading_service = HeadingService()
    chunking_service = ChunkingService(settings=settings)

    policy_sections = heading_service.build_sections(
        pdf_service.parse(POLICY_PDF, DocumentType.POLICY)
    )
    patient_sections = heading_service.build_sections(
        pdf_service.parse(PATIENT_PDF, DocumentType.PATIENT_CHART)
    )

    procedure = ProcedureExtraction(
        procedure_name="Permanent Cardiac Pacemaker Implantation",
        cpt_code="33207",
        is_future_procedure=True,
        confidence=0.95,
    )

    embedding = _fake_embedding_service()
    retrieval = RetrievalService(embedding, settings)
    reranker = RerankingService(settings=settings)
    reranker._model = MagicMock(
        compute_score=MagicMock(
            side_effect=lambda pairs: [float(len(pairs) - i) for i in range(len(pairs))]
        )
    )

    llm = MagicMock()
    policy_service = PolicyService(
        llm_service=llm,
        retrieval_service=retrieval,
        chunking_service=chunking_service,
        reranking_service=reranker,
        settings=settings,
    )
    candidates = policy_service.retrieve_policy_candidates(procedure, policy_sections)
    llm.extract_structured = AsyncMock(
        return_value=PolicyCriteriaResult(
            matching_policy_criteria_found=False,
            explanation=(
                "Supplied policy excerpts address emergency medical "
                "transport coverage, not the detected cardiac procedure."
            ),
            criteria_tree=None,
        )
    )
    policy_analysis = await policy_service._analyze_candidates(procedure, candidates)
    evidence = await EvidenceService(
        retrieval_service=retrieval,
        chunking_service=chunking_service,
        reranking_service=reranker,
        llm_service=MagicMock(extract_structured=AsyncMock()),
        settings=settings,
    ).evaluate_evidence(policy_analysis, patient_sections)

    readiness = ReadinessService().evaluate(
        procedure=procedure,
        policy_analysis=policy_analysis,
        evidence_analysis=evidence,
    )

    assert readiness.overall_status == ReadinessStatus.NOT_APPLICABLE
    assert readiness.matching_policy_criteria_found is False
    assert readiness.missing_items == []
    assert readiness.review_items == []
    assert evidence.skipped is True
    assert evidence.llm_call_count == 0

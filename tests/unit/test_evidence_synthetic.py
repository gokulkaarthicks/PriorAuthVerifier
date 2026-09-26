"""Synthetic applicable patient fixture covering MET / MISSING / UNCLEAR."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.config import Settings
from app.models.chunk import Chunk
from app.models.policy_analysis import PolicyAnalysisResult
from app.models.section import Section
from app.schemas.evaluation import (
    CriterionEvaluation,
    CriterionStatus,
    EvidenceEvaluationBatch,
    ExclusionEvaluation,
)
from app.schemas.policy import (
    CriterionCategory,
    CriterionNode,
    CriterionNodeType,
    PolicyCriteriaExtraction,
    PolicyExclusion,
)
from app.services.evidence_service import EvidenceService
from app.services.retrieval_service import RetrievalService
from tests.helpers.evidence_sol import as_sol_response


def _section(section_id: str, text: str, *, page: int) -> Section:
    return Section(
        id=section_id,
        document_id="synthetic-patient",
        heading="Clinical notes",
        parent_heading=None,
        text=text,
        start_page=page,
        end_page=page,
        token_count=max(1, len(text.split())),
    )


def _passthrough_chunking() -> MagicMock:
    service = MagicMock()

    def chunk_sections(sections: list[Section]) -> list[Chunk]:
        return [
            Chunk(
                id=section.id,
                document_id=section.document_id,
                text=section.text,
                page_number=section.start_page,
                end_page_number=section.end_page,
                heading=section.heading,
                parent_heading=section.parent_heading,
                token_count=section.token_count,
                section_id=section.id,
                chunk_index=0,
                document_date=section.document_date,
            )
            for section in sections
        ]

    service.chunk_sections.side_effect = chunk_sections
    return service


def _embedding_service(dim: int = 4) -> MagicMock:
    service = MagicMock()

    def embed(texts: list[str], **_kwargs) -> np.ndarray:
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for index, text in enumerate(texts):
            lower = text.lower()
            matrix[index, 0] = (len(text) % 11) / 11.0
            matrix[index, 1] = 1.0 if "reflux" in lower else 0.0
            matrix[index, 2] = 1.0 if "compression" in lower else 0.0
            matrix[index, 3] = 1.0 if "ceap" in lower or "cosmetic" in lower else 0.0
            norm = np.linalg.norm(matrix[index])
            if norm > 0:
                matrix[index] /= norm
        return matrix

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


def _criteria_tree() -> CriterionNode:
    return CriterionNode(
        id="root",
        node_type=CriterionNodeType.AND,
        children=[
            CriterionNode(
                id="c_reflux",
                node_type=CriterionNodeType.CONDITION,
                description="Demonstrated saphenous reflux on duplex ultrasound.",
                category=CriterionCategory.DIAGNOSTIC,
                source_chunk_id="policy-1",
                source_text="Demonstrated saphenous reflux on duplex ultrasound.",
                page_number=1,
            ),
            CriterionNode(
                id="c_ceap",
                node_type=CriterionNodeType.CONDITION,
                description="CEAP clinical class is C2 or greater.",
                category=CriterionCategory.MEASUREMENT,
                source_chunk_id="policy-1",
                source_text="CEAP clinical class is C2 or greater.",
                page_number=1,
            ),
            CriterionNode(
                id="or_branch",
                node_type=CriterionNodeType.OR,
                children=[
                    CriterionNode(
                        id="c_compression",
                        node_type=CriterionNodeType.CONDITION,
                        description=(
                            "Compression therapy failed for at least three months."
                        ),
                        category=CriterionCategory.DURATION,
                        source_chunk_id="policy-1",
                        source_text=(
                            "Compression therapy failed for at least three months."
                        ),
                        page_number=1,
                    ),
                    CriterionNode(
                        id="c_conservative",
                        node_type=CriterionNodeType.CONDITION,
                        description=(
                            "Prior conservative treatment failed and symptoms "
                            "recurred."
                        ),
                        category=CriterionCategory.TREATMENT_HISTORY,
                        source_chunk_id="policy-1",
                        source_text=(
                            "Prior conservative treatment failed and symptoms "
                            "recurred."
                        ),
                        page_number=1,
                    ),
                ],
            ),
            CriterionNode(
                id="c_unclear",
                node_type=CriterionNodeType.CONDITION,
                description="Documented venous ulcer healing timeline is complete.",
                category=CriterionCategory.DOCUMENTATION,
                source_chunk_id="policy-1",
                source_text="Documented venous ulcer healing timeline is complete.",
                page_number=1,
            ),
        ],
    )


def _patient_sections() -> list[Section]:
    return [
        _section(
            "pat-1",
            "Duplex ultrasound demonstrated saphenous reflux in the GSV.",
            page=3,
        ),
        _section(
            "pat-2",
            "CEAP clinical class documented as C3 at the preoperative visit.",
            page=4,
        ),
        _section(
            "pat-3",
            "Patient notes mention possible ulcer changes without dates.",
            page=5,
        ),
        _section(
            "pat-4",
            "Allergy list and medication reconciliation only.",
            page=6,
        ),
    ]


@pytest.mark.asyncio
async def test_synthetic_patient_met_missing_unclear() -> None:
    embedding = _embedding_service()
    retrieval = RetrievalService(
        embedding,
        Settings(
            BM25_TOP_K=10,
            DENSE_TOP_K=10,
            RRF_TOP_K=10,
            RERANK_TOP_K=5,
        ),
    )

    policy_analysis = PolicyAnalysisResult(
        applicable=True,
        applicability_reason="Synthetic policy matches the procedure.",
        applicability_confidence=0.99,
        criteria=PolicyCriteriaExtraction(
            procedure_name="Example Vein Ablation",
            policy_summary="Synthetic ablation coverage criteria.",
            root=_criteria_tree(),
            exclusions=[
                PolicyExclusion(
                    id="ex_cosmetic",
                    description="Cosmetic treatment without medical necessity.",
                    source_chunk_id="policy-1",
                    source_text="Cosmetic treatment without medical necessity.",
                    page_number=1,
                )
            ],
            source_chunk_ids=["policy-1"],
        ),
        supporting_chunk_ids=["policy-1"],
        llm_call_count=1,
    )

    llm_batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c_reflux",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["pat-1"],
                evidence_text=["demonstrated saphenous reflux in the GSV"],
                page_numbers=[3],
                explanation="Duplex finding explicitly documents reflux.",
            ),
            CriterionEvaluation(
                criterion_id="c_ceap",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["pat-2"],
                evidence_text=["CEAP clinical class documented as C3"],
                page_numbers=[4],
                explanation="C3 satisfies C2 or greater.",
            ),
            CriterionEvaluation(
                criterion_id="c_compression",
                status=CriterionStatus.MISSING,
                explanation="No qualifying documentation found.",
            ),
            CriterionEvaluation(
                criterion_id="c_conservative",
                status=CriterionStatus.MISSING,
                explanation="No prior conservative failure documented.",
            ),
            CriterionEvaluation(
                criterion_id="c_unclear",
                status=CriterionStatus.UNCLEAR,
                evidence_chunk_ids=["pat-3"],
                evidence_text=["possible ulcer changes without dates"],
                page_numbers=[5],
                explanation="Ulcer mention lacks a complete healing timeline.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex_cosmetic",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="No cosmetic-only indication found.",
            )
        ],
    )

    reranker = MagicMock()

    def rerank_grouped(groups, top_k=None, **_kwargs):
        keep = top_k or 5
        return {
            group_id: candidates[:keep]
            for group_id, _query, candidates in groups
        }

    reranker.rerank_grouped.side_effect = rerank_grouped
    reranker.rerank.side_effect = (
        lambda query, candidates, top_k=None: candidates[: (top_k or 5)]
    )
    llm = MagicMock()
    llm.extract_structured = as_sol_response(llm_batch)

    service = EvidenceService(
        retrieval_service=retrieval,
        chunking_service=_passthrough_chunking(),
        reranking_service=reranker,
        llm_service=llm,
        settings=Settings(POLICY_REASONING_MODEL="gpt-6-sol"),
    )
    result = await service.evaluate_evidence(policy_analysis, _patient_sections())

    assert result.skipped is False
    assert result.llm_call_count == 1
    assert len(result.criteria) == 5
    statuses = {item.criterion_id: item.status for item in result.criteria}
    assert statuses["c_reflux"] == CriterionStatus.MET
    assert statuses["c_ceap"] == CriterionStatus.MET
    assert statuses["c_compression"] == CriterionStatus.MISSING
    assert statuses["c_conservative"] == CriterionStatus.MISSING
    assert statuses["c_unclear"] == CriterionStatus.UNCLEAR
    assert result.exclusions[0].triggered is False
    assert llm.extract_structured.await_count == 1

    met = [item for item in result.criteria if item.status == CriterionStatus.MET]
    assert all(item.evidence_chunk_ids and item.page_numbers for item in met)
    missing = [
        item for item in result.criteria if item.status == CriterionStatus.MISSING
    ]
    assert all(not item.evidence_chunk_ids for item in missing)

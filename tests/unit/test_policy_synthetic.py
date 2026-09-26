"""Synthetic applicable-policy fixture and criteria extraction coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.models.chunk import Chunk
from app.models.retrieval import RetrievalResult
from app.schemas.policy import (
    CriterionCategory,
    CriterionNodeType,
    SolCriterionNode,
    SolPolicyCriteriaExtraction,
    SolPolicyCriteriaResult,
    SolPolicyExclusion,
)
from app.schemas.procedure import ProcedureExtraction
from app.services.policy_service import PolicyService

SYNTHETIC_POLICY_TEXT = """
POLICY FOR EXAMPLE VEIN ABLATION PROCEDURE

Coverage is available for Example Vein Ablation when all of the following
are met:

1. Demonstrated saphenous reflux on duplex ultrasound.
2. CEAP clinical class is C2 or greater.
3. Either:
   a. Compression therapy failed for at least three months, or
   b. Prior conservative treatment failed and symptoms recurred.

Exclusions:
- Cosmetic treatment without medical necessity is not covered.
""".strip()


def _synthetic_candidates() -> list[RetrievalResult]:
    chunk = Chunk(
        id="synthetic-1",
        document_id="synthetic-policy",
        text=SYNTHETIC_POLICY_TEXT,
        page_number=1,
        end_page_number=1,
        heading="Example Vein Ablation",
        parent_heading=None,
        token_count=120,
        section_id="synthetic-sec",
        chunk_index=0,
    )
    return [
        RetrievalResult(
            chunk=chunk,
            rank=1,
            score=0.99,
            retrieval_method="rerank",
            reranker_score=0.99,
        )
    ]


def _synthetic_sol_tree() -> SolCriterionNode:
    return SolCriterionNode(
        id="root",
        node_type="AND",
        children=[
            SolCriterionNode(
                id="c1",
                node_type="CONDITION",
                description="Demonstrated saphenous reflux on duplex ultrasound.",
                category=CriterionCategory.DIAGNOSTIC,
                source_context_id="P1",
                source_quote=(
                    "Demonstrated saphenous reflux on duplex ultrasound."
                ),
            ),
            SolCriterionNode(
                id="c2",
                node_type="CONDITION",
                description="CEAP clinical class is C2 or greater.",
                category=CriterionCategory.MEASUREMENT,
                source_context_id="P1",
                source_quote="CEAP clinical class is C2 or greater.",
            ),
            SolCriterionNode(
                id="or-1",
                node_type="OR",
                children=[
                    SolCriterionNode(
                        id="c3",
                        node_type="CONDITION",
                        description=(
                            "Compression therapy failed for at least "
                            "three months."
                        ),
                        category=CriterionCategory.DURATION,
                        source_context_id="P1",
                        source_quote=(
                            "Compression therapy failed for at least "
                            "three months"
                        ),
                    ),
                    SolCriterionNode(
                        id="c4",
                        node_type="CONDITION",
                        description=(
                            "Prior conservative treatment failed and "
                            "symptoms recurred."
                        ),
                        category=CriterionCategory.TREATMENT_HISTORY,
                        source_context_id="P1",
                        source_quote=(
                            "Prior conservative treatment failed and "
                            "symptoms recurred"
                        ),
                    ),
                ],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_synthetic_applicable_policy_criteria_tree() -> None:
    llm = MagicMock()
    llm.extract_structured = AsyncMock(
        return_value=SolPolicyCriteriaResult(
            matching_policy_criteria_found=True,
            explanation="Policy explicitly covers Example Vein Ablation.",
            criteria_tree=SolPolicyCriteriaExtraction(
                procedure_name="Example Vein Ablation",
                policy_summary=(
                    "Ablation covered with diagnostic, CEAP, and "
                    "treatment-history criteria."
                ),
                root=_synthetic_sol_tree(),
                exclusions=[
                    SolPolicyExclusion(
                        id="ex1",
                        description=(
                            "Cosmetic treatment without medical necessity "
                            "is not covered."
                        ),
                        source_context_id="P1",
                        source_quote=(
                            "Cosmetic treatment without medical necessity "
                            "is not covered."
                        ),
                    )
                ],
                source_context_ids=["P1"],
            ),
        )
    )
    service = PolicyService(
        llm_service=llm,
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        settings=Settings(POLICY_REASONING_MODEL="gpt-6-sol"),
    )
    procedure = ProcedureExtraction(
        procedure_name="Example Vein Ablation",
        cpt_code=None,
        is_future_procedure=True,
        confidence=0.9,
    )

    result = await service._analyze_candidates(procedure, _synthetic_candidates())

    assert result.applicable is True
    assert result.llm_call_count == 1
    assert result.criteria is not None
    assert result.criteria.root.node_type == CriterionNodeType.AND
    assert len(result.criteria.root.children) == 3
    assert result.criteria.root.children[2].node_type == CriterionNodeType.OR
    assert len(result.criteria.root.children[2].children) == 2
    assert len(result.criteria.exclusions) == 1
    assert result.criteria.exclusions[0].id == "ex1"
    assert result.criteria.exclusions[0].source_chunk_id == "synthetic-1"

"""Regression tests for shared-corpus evidence candidate IDs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import EvidenceGroundingError
from app.models.chunk import Chunk
from app.schemas.evaluation import (
    CriterionStatus,
    EvidenceCitation,
    SolCriterionEvaluation,
    SolEvidenceEvaluationBatch,
    SolExclusionEvaluation,
    citation_from_chunk,
    materialize_sol_batch,
)
from app.services.evidence_service import EvidenceService


def _chunk(chunk_id: str, text: str, *, page: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="patient",
        text=text,
        page_number=page,
        end_page_number=page,
        heading="Notes",
        parent_heading=None,
        token_count=5,
        section_id=chunk_id,
        chunk_index=0,
        document_date="2025-01-01",
    )


def _service() -> EvidenceService:
    return EvidenceService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=MagicMock(),
    )


def test_candidate_ids_are_shared_across_multi_criterion_batch() -> None:
    service = _service()
    shared = _chunk("shared", "shared finding")
    mapping = {
        "av1a": ["shared", "a-only"],
        "snd1a": ["shared", "s-only"],
    }
    candidate_by_id = {
        "shared": shared,
        "a-only": _chunk("a-only", "av finding"),
        "s-only": _chunk("s-only", "snd finding"),
    }
    registry, allowed = service._build_candidate_registry(
        item_ids=["av1a", "snd1a"],
        mapping=mapping,
        candidate_by_id=candidate_by_id,
    )
    assert allowed["av1a"] == {"E1", "E2"}
    assert allowed["snd1a"] == {"E1", "E3"}
    assert set(registry) == {"E1", "E2", "E3"}
    # Same underlying chunk appears once with a shared E ID.
    assert registry["E1"].chunk.id == "shared"
    assert "E1" in allowed["av1a"] and "E1" in allowed["snd1a"]


def test_prompt_lists_shared_corpus_and_allowed_ids() -> None:
    service = _service()
    from app.schemas.policy import CriterionCategory
    from app.services.evidence_service import AtomicCriterion

    conditions = [
        AtomicCriterion(
            id="av1a",
            description="AV block criterion",
            category=CriterionCategory.CLINICAL,
            kind="condition",
        ),
        AtomicCriterion(
            id="snd1a",
            description="SND criterion",
            category=CriterionCategory.CLINICAL,
            kind="condition",
        ),
    ]
    mapping = {
        "av1a": ["av-chunk"],
        "snd1a": ["snd-chunk"],
    }
    candidate_by_id = {
        "av-chunk": _chunk("av-chunk", "AV evidence text"),
        "snd-chunk": _chunk("snd-chunk", "SND evidence text"),
    }
    registry, allowed = service._build_candidate_registry(
        item_ids=["av1a", "snd1a"],
        mapping=mapping,
        candidate_by_id=candidate_by_id,
    )
    prompt = service._build_evaluation_prompt(
        conditions=conditions,
        exclusions=[],
        candidate_map=registry,
        allowed_by_item=allowed,
        reference_date=None,
        local_context_by_id={},
    )
    assert "EVIDENCE CHUNKS:" not in prompt
    assert "evidence_candidates:" not in prompt
    assert "evidence_corpus:" in prompt
    assert "items:" in prompt
    assert "AV evidence text" in prompt
    assert "SND evidence text" in prompt
    assert "candidate_id: E1" in prompt
    assert "candidate_id: E2" in prompt
    after_av = prompt.split("criterion_id: av1a", 1)[1]
    av_block = after_av.split("criterion_id: snd1a", 1)[0]
    snd_block = prompt.split("criterion_id: snd1a", 1)[1]
    assert "allowed_candidate_ids: ['E1']" in av_block
    assert "allowed_candidate_ids: ['E2']" in snd_block
    assert "AV evidence text" not in av_block
    assert "SND evidence text" not in snd_block


def test_cross_criterion_candidate_raises_grounding_error() -> None:
    service = _service()
    batch = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="av1a",
                status=CriterionStatus.MET,
                evidence_candidate_ids=["E2"],
                explanation="Leak.",
            )
        ],
        exclusions=[],
    )
    with pytest.raises(EvidenceGroundingError):
        service._assert_candidate_ids_in_scope(
            batch,
            allowed_by_item={"av1a": {"E1"}, "snd1a": {"E2"}},
        )


def test_valid_candidate_resolves_to_deterministic_citation() -> None:
    chunk = _chunk("p1", "full chunk body", page=4)
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="av1a",
                status=CriterionStatus.MET,
                evidence_candidate_ids=["E1"],
                explanation="Supported.",
            )
        ],
        exclusions=[],
    )
    public = materialize_sol_batch(
        sol,
        candidate_map={"E1": citation_from_chunk(chunk)},
    )
    citation = public.criteria[0].evidence[0]
    assert citation == EvidenceCitation(
        chunk_id="p1",
        text="full chunk body",
        page_number=4,
        document_date="2025-01-01",
    )


def test_empty_evidence_candidate_ids_allowed_for_missing() -> None:
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="av1a",
                status=CriterionStatus.MISSING,
                evidence_candidate_ids=[],
                explanation="No support.",
            )
        ],
        exclusions=[
            SolExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                evidence_candidate_ids=[],
                explanation="Not triggered.",
            )
        ],
    )
    public = materialize_sol_batch(sol, candidate_map={})
    assert public.criteria[0].evidence == []
    assert public.exclusions[0].evidence == []


def test_multiple_citations_preserve_order_and_metadata() -> None:
    c1 = _chunk("p1", "first", page=1)
    c2 = _chunk("p2", "second", page=2)
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="av1a",
                status=CriterionStatus.MET,
                evidence_candidate_ids=["E1", "E2"],
                explanation="Two cites.",
            )
        ],
        exclusions=[],
    )
    public = materialize_sol_batch(
        sol,
        candidate_map={
            "E1": citation_from_chunk(c1),
            "E2": citation_from_chunk(c2),
        },
    )
    assert [item.chunk_id for item in public.criteria[0].evidence] == ["p1", "p2"]
    assert [item.page_number for item in public.criteria[0].evidence] == [1, 2]
    assert [item.text for item in public.criteria[0].evidence] == ["first", "second"]

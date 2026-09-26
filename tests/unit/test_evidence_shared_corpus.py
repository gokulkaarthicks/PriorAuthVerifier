"""Unit tests for shared E1..En evidence corpus packing and grounding."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
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
from app.schemas.policy import CriterionCategory
from app.services.evidence_service import AtomicCriterion, EvidenceService


def _chunk(
    chunk_id: str,
    text: str,
    *,
    page: int = 1,
    document_date: str | None = "2025-01-01",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="patient",
        text=text,
        page_number=page,
        end_page_number=page,
        heading="Notes",
        parent_heading=None,
        token_count=max(1, len(text.split())),
        section_id=chunk_id,
        chunk_index=0,
        document_date=document_date,
    )


def _atom(item_id: str, description: str = "Criterion text.") -> AtomicCriterion:
    return AtomicCriterion(
        id=item_id,
        description=description,
        category=CriterionCategory.CLINICAL,
        kind="condition",
    )


def _service(*, settings: Settings | None = None) -> EvidenceService:
    return EvidenceService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=MagicMock(),
        settings=settings
        or Settings(
            EVIDENCE_LLM_MAX_CONTEXT_TOKENS=20000,
            EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=20,
            POLICY_REASONING_MODEL="gpt-6-sol",
        ),
    )


def test_shared_chunk_appears_once_in_corpus_for_ten_criteria() -> None:
    service = _service()
    shared = _chunk("shared", "shared finding across criteria")
    item_ids = [f"c{i}" for i in range(10)]
    mapping = {item_id: ["shared"] for item_id in item_ids}
    registry, allowed = service._build_candidate_registry(
        item_ids=item_ids,
        mapping=mapping,
        candidate_by_id={"shared": shared},
    )
    assert list(registry) == ["E1"]
    assert registry["E1"].chunk.id == "shared"
    assert all(allowed[item_id] == {"E1"} for item_id in item_ids)


def test_each_criterion_has_own_allowed_set() -> None:
    service = _service()
    mapping = {
        "av1a": ["shared", "a-only"],
        "snd1a": ["shared", "s-only"],
    }
    candidate_by_id = {
        "shared": _chunk("shared", "shared finding"),
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
    assert allowed["av1a"] != allowed["snd1a"]
    assert set(registry) == {"E1", "E2", "E3"}


def test_criterion_cannot_cite_e_id_outside_allowed_set() -> None:
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


def test_one_e_id_allowed_for_multiple_criteria() -> None:
    service = _service()
    shared = _chunk("shared", "shared finding")
    registry, allowed = service._build_candidate_registry(
        item_ids=["c1", "c2", "c3"],
        mapping={"c1": ["shared"], "c2": ["shared"], "c3": ["shared"]},
        candidate_by_id={"shared": shared},
    )
    assert registry["E1"].chunk.id == "shared"
    assert allowed["c1"] == allowed["c2"] == allowed["c3"] == {"E1"}


def test_candidate_id_hydrates_exact_evidence_citation() -> None:
    chunk = _chunk("p1", "full chunk body", page=4, document_date="2024-11-02")
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="c1",
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
    assert public.criteria[0].evidence[0] == EvidenceCitation(
        chunk_id="p1",
        text="full chunk body",
        page_number=4,
        document_date="2024-11-02",
    )


def test_citation_metadata_cannot_be_modified_by_sol() -> None:
    service = _service()
    chunk = _chunk("p1", "authoritative chunk text", page=7, document_date="2023-05-01")
    # Sol invents different page/date/text/chunk_id — validation overwrites.
    forged = EvidenceCitation(
        chunk_id="p1",
        text="forged text from model",
        page_number=99,
        document_date="2099-01-01",
    )
    validated, _warnings = service._validate_evidence_grounding(
        item_id="c1",
        evidence=[forged],
        candidate_by_id={"p1": chunk},
        allowed_chunk_ids={"p1"},
        reference_date=date(2025, 1, 1),
    )
    assert validated == [citation_from_chunk(chunk)]
    assert validated[0].text == "authoritative chunk text"
    assert validated[0].page_number == 7
    assert validated[0].document_date == "2023-05-01"
    assert validated[0].chunk_id == "p1"


def test_token_aware_packing_respects_tiny_budget() -> None:
    service = _service(
        settings=Settings(
            EVIDENCE_LLM_MAX_CONTEXT_TOKENS=1024,
            EVIDENCE_LLM_MAX_OUTPUT_TOKENS=256,
            EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=20,
        )
    )
    conditions = [
        _atom(f"c{i}", description=f"Demonstrated clinical requirement {i}. " * 10)
        for i in range(4)
    ]
    candidate_by_id = {
        f"pat-{i}": _chunk(
            f"pat-{i}",
            ("Patient chart evidence note with supporting detail. " * 40) + str(i),
        )
        for i in range(4)
    }
    mapping = {f"c{i}": [f"pat-{i}"] for i in range(4)}
    work_items = service._build_sol_work_items(
        conditions,
        [],
        mapping=mapping,
        candidate_by_id=candidate_by_id,
        reference_date=date(2024, 6, 1),
        local_context_by_id={f"c{i}": "" for i in range(4)},
    )
    assert sum(len(batch[0]) for batch in work_items) == 4
    assert len(work_items) >= 2
    assert all(
        service._prompt_fits(
            conditions=batch_conditions,
            exclusions=batch_exclusions,
            mapping=batch_mapping,
            candidate_by_id=candidate_by_id,
            reference_date=date(2024, 6, 1),
            local_context_by_id={f"c{i}": "" for i in range(4)},
        )
        for batch_conditions, batch_exclusions, batch_mapping in work_items
    )


def test_batch_grows_beyond_five_when_it_fits() -> None:
    service = _service(
        settings=Settings(
            EVIDENCE_LLM_MAX_CONTEXT_TOKENS=20000,
            EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=20,
        )
    )
    conditions = [_atom(f"c{i}", description=f"Short criterion {i}.") for i in range(8)]
    candidate_by_id = {
        f"pat-{i}": _chunk(f"pat-{i}", f"brief evidence {i}") for i in range(8)
    }
    mapping = {f"c{i}": [f"pat-{i}"] for i in range(8)}
    work_items = service._build_sol_work_items(
        conditions,
        [],
        mapping=mapping,
        candidate_by_id=candidate_by_id,
        reference_date=None,
        local_context_by_id={},
    )
    assert len(work_items) == 1
    assert len(work_items[0][0]) == 8
    assert len(work_items[0][0]) > 5


def test_batch_splits_when_adding_exceeds_budget() -> None:
    service = _service(
        settings=Settings(
            EVIDENCE_LLM_MAX_CONTEXT_TOKENS=1100,
            EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=20,
        )
    )
    conditions = [
        _atom(
            f"c{i}",
            description=(
                f"Demonstrated clinical requirement number {i}. " * 8
            ),
        )
        for i in range(4)
    ]
    candidate_by_id = {
        f"pat-{i}": _chunk(
            f"pat-{i}",
            ("Patient chart evidence note with supporting detail. " * 30) + str(i),
        )
        for i in range(4)
    }
    mapping = {f"c{i}": [f"pat-{i}"] for i in range(4)}
    work_items = service._build_sol_work_items(
        conditions,
        [],
        mapping=mapping,
        candidate_by_id=candidate_by_id,
        reference_date=date(2024, 6, 1),
        local_context_by_id={f"c{i}": "" for i in range(4)},
    )
    assert len(work_items) >= 2
    assert all(len(batch[0]) < 4 for batch in work_items)


def test_max_items_safety_cap_respected() -> None:
    service = _service(
        settings=Settings(
            EVIDENCE_LLM_MAX_CONTEXT_TOKENS=20000,
            EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=3,
        )
    )
    conditions = [_atom(f"c{i}", description=f"Short criterion {i}.") for i in range(7)]
    candidate_by_id = {
        f"pat-{i}": _chunk(f"pat-{i}", f"brief evidence {i}") for i in range(7)
    }
    mapping = {f"c{i}": [f"pat-{i}"] for i in range(7)}
    work_items = service._build_sol_work_items(
        conditions,
        [],
        mapping=mapping,
        candidate_by_id=candidate_by_id,
        reference_date=None,
        local_context_by_id={},
    )
    assert all(len(batch[0]) <= 3 for batch in work_items)
    assert sum(len(batch[0]) for batch in work_items) == 7
    assert len(work_items) >= 3


def test_sol_schema_has_no_description_and_prompt_avoids_duplicate_texts() -> None:
    service = _service()
    assert "description" not in SolCriterionEvaluation.model_fields
    assert "description" not in SolExclusionEvaluation.model_fields
    assert "description" not in SolEvidenceEvaluationBatch.model_fields

    shared = _chunk("shared", "UNIQUE_SHARED_EVIDENCE_TEXT")
    only_a = _chunk("a-only", "UNIQUE_AV_EVIDENCE_TEXT")
    conditions = [_atom("av1a", "AV block"), _atom("snd1a", "SND criterion")]
    mapping = {"av1a": ["shared", "a-only"], "snd1a": ["shared"]}
    candidate_by_id = {"shared": shared, "a-only": only_a}
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
    assert "evidence_corpus:" in prompt
    assert prompt.count("UNIQUE_SHARED_EVIDENCE_TEXT") == 1
    assert prompt.count("UNIQUE_AV_EVIDENCE_TEXT") == 1
    # Items nest allowed IDs, not a second full copy of corpus texts.
    items_section = prompt.split("items:", 1)[1]
    assert "UNIQUE_SHARED_EVIDENCE_TEXT" not in items_section
    assert "UNIQUE_AV_EVIDENCE_TEXT" not in items_section
    assert "allowed_candidate_ids:" in items_section


@pytest.mark.asyncio
async def test_evidence_requests_low_reasoning_effort() -> None:
    llm = MagicMock()
    llm.extract_structured = AsyncMock(
        return_value=SolEvidenceEvaluationBatch(
            criteria=[
                SolCriterionEvaluation(
                    criterion_id="c1",
                    status=CriterionStatus.MISSING,
                    evidence_candidate_ids=[],
                    explanation="Missing.",
                )
            ],
            exclusions=[],
        )
    )
    service = EvidenceService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=llm,
        settings=Settings(
            EVIDENCE_LLM_MAX_CONTEXT_TOKENS=20000,
            EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=20,
            POLICY_REASONING_MODEL="gpt-6-sol",
        ),
    )
    await service._evaluate_with_llm(
        conditions=[_atom("c1")],
        exclusions=[],
        mapping={"c1": ["p1"]},
        candidate_by_id={"p1": _chunk("p1", "patient note")},
        reference_date=None,
        local_context_by_id={},
    )

    llm.extract_structured.assert_awaited()
    kwargs = llm.extract_structured.await_args.kwargs
    assert kwargs.get("reasoning_effort") == "low"

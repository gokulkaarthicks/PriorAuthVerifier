"""Regression tests for correctness fixes: context, citations, dates."""

from __future__ import annotations

import pytest

from app.core.exceptions import EvidenceValidationError
from app.models.chunk import Chunk
from app.schemas.evaluation import (
    CriterionEvaluation,
    CriterionStatus,
    EvidenceCitation,
    SolCriterionEvaluation,
    SolEvidenceEvaluationBatch,
    citation_from_chunk,
    enrich_citation,
    materialize_sol_batch,
)
from app.schemas.policy import (
    CriterionCategory,
    CriterionNode,
    CriterionNodeType,
    PolicyExclusion,
)
from app.services.evidence_service import EvidenceService
from app.utils.dates import (
    extract_analysis_reference_date,
    extract_document_date_with_warnings,
    extract_explicit_date,
    is_within_months,
)


def _condition(node_id: str, description: str, **kwargs: object) -> CriterionNode:
    return CriterionNode(
        id=node_id,
        node_type=CriterionNodeType.CONDITION,
        description=description,
        category=CriterionCategory.CLINICAL,
        source_chunk_id="policy-1",
        source_text=str(kwargs.get("source_text") or description),
        page_number=1,
    )


def test_local_policy_definitions_supplied_during_evidence_extraction() -> None:
    from unittest.mock import MagicMock

    root = CriterionNode(
        id="snd",
        node_type=CriterionNodeType.AND,
        description="Sinus node dysfunction",
        children=[
            CriterionNode(
                id="documented",
                node_type=CriterionNodeType.OR,
                description="Documented sinus node dysfunction including one of",
                children=[
                    _condition("hr", "Sinus bradycardia at rate <50 beats per minute"),
                    _condition("pause", "Sinus pauses >3 seconds"),
                ],
            ),
            _condition("symptoms", "Symptoms attributable to sinus node dysfunction"),
        ],
    )
    service = EvidenceService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=MagicMock(),
    )
    atoms = service.extract_atomic_criteria(root)
    by_id = {atom.id: atom for atom in atoms}
    assert "Sinus node dysfunction" in (by_id["hr"].local_policy_context or "")
    assert "parent:" in (by_id["hr"].local_policy_context or "")
    assert "Sinus pauses >3 seconds" in (by_id["hr"].local_policy_context or "")


def test_exclusion_local_context_includes_related_parent_conditions() -> None:
    from unittest.mock import MagicMock

    root = CriterionNode(
        id="root",
        node_type=CriterionNodeType.OR,
        children=[
            CriterionNode(
                id="snd",
                node_type=CriterionNodeType.AND,
                description="Sinus node dysfunction",
                children=[
                    _condition("hr", "Sinus bradycardia at rate <50"),
                ],
            )
        ],
    )
    exclusion = PolicyExclusion(
        id="ex1",
        description="Sinus node dysfunction when individual is asymptomatic",
        source_chunk_id="p1",
        source_text="Sinus node dysfunction when individual is asymptomatic",
        page_number=1,
    )
    service = EvidenceService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=MagicMock(),
    )
    context = service._exclusion_local_context(exclusion, root=root)
    assert "source_policy_text:" in context
    assert "Sinus node dysfunction" in context


def test_evidence_citation_keeps_chunk_text_page_date_together() -> None:
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_candidate_ids=["c1:e1"],
                explanation="Met under SND definition.",
            )
        ],
        exclusions=[],
    )
    chunk = Chunk(
        id="p1",
        document_id="doc",
        text="Holter showed HR 44 bpm",
        page_number=3,
        end_page_number=3,
        heading="Holter Nov 1-29, 2025",
        parent_heading=None,
        token_count=10,
        section_id="s1",
        chunk_index=0,
        document_date="2025-11-01",
    )
    public = materialize_sol_batch(
        sol,
        candidate_map={"c1:e1": citation_from_chunk(chunk)},
    )
    citation = public.criteria[0].evidence[0]
    assert citation.chunk_id == "p1"
    assert citation.text == "Holter showed HR 44 bpm"
    assert citation.page_number == 3
    assert citation.document_date == "2025-11-01"
    assert public.criteria[0].evidence_chunk_ids == ["p1"]
    assert public.criteria[0].evidence_text == ["Holter showed HR 44 bpm"]
    assert public.criteria[0].page_numbers == [3]


def test_evidence_with_text_cannot_lose_page_if_chunk_has_it() -> None:
    citation = EvidenceCitation(chunk_id="p1", text="pause 3.1 sec", page_number=None)
    chunk = Chunk(
        id="p1",
        document_id="doc",
        text="pause 3.1 sec documented",
        page_number=7,
        end_page_number=7,
        heading=None,
        parent_heading=None,
        token_count=5,
        section_id="s1",
        chunk_index=0,
        document_date=None,
    )
    enriched = enrich_citation(citation, chunk=chunk)
    assert enriched.page_number == 7
    assert enriched.text == "pause 3.1 sec documented"


def test_runtime_date_never_substituted_for_clinical_evidence_date() -> None:
    citation = EvidenceCitation(chunk_id="p1", text="finding", page_number=1)
    chunk = Chunk(
        id="p1",
        document_id="doc",
        text="finding",
        page_number=1,
        end_page_number=1,
        heading=None,
        parent_heading=None,
        token_count=1,
        section_id="s1",
        chunk_index=0,
        document_date=None,
    )
    enriched = enrich_citation(citation, chunk=chunk)
    assert enriched.document_date is None
    # Temporal window without reference date does not invent today().
    assert (
        is_within_months("2025-11-01", months=6, reference=None) is None
    )


def test_conflicting_source_dates_are_surfaced_not_rewritten() -> None:
    heading = "Holter Monitor November 1–29, 2025"
    body = (
        "Recording from November 1, 2025 through November 29, 2025. "
        "Electronically signed November 11, 2025."
    )
    primary, warnings = extract_document_date_with_warnings(heading, body)
    assert primary == "2025-11-01"
    assert warnings
    assert "Conflicting source dates" in warnings[0]
    assert "2025-11-11" in warnings[0]
    # Preferred date is preserved, not silently replaced by signature.
    assert extract_explicit_date(heading, body) == "2025-11-01"


def test_pa_submission_date_used_as_analysis_reference_when_present() -> None:
    text = (
        "Prior auth submission date: November 14, 2025. "
        "Planned procedure permanent pacemaker implantation."
    )
    assert extract_analysis_reference_date(text) == "2025-11-14"


def test_pa_status_submitted_with_line_broken_year() -> None:
    text = (
        "PA Status\n"
        "Submitted November 14,\n"
        "2025\n"
        "Auth # 12345\n"
        "Encounter date October 1, 2025."
    )
    assert extract_analysis_reference_date(text) == "2025-11-14"


def test_planned_procedure_date_fallback_when_no_pa_submission() -> None:
    text = (
        "Planned procedure date: December 2, 2025. "
        "Clinic visit October 1, 2025 documented sinus bradycardia."
    )
    assert extract_analysis_reference_date(text) == "2025-12-02"


def test_latest_clinical_date_fallback_when_no_pa_or_planned() -> None:
    # Keep clinical dates far from signature wording so the ±64 char window
    # does not incorrectly classify encounter dates as signature metadata.
    text = (
        "Clinic visit October 1, 2025 documented sinus bradycardia with "
        "ongoing symptoms across outpatient follow-up notes. "
        "Holter monitoring performed November 5, 2025 with significant pauses. "
        "Additional narrative padding keeps signature context distant from "
        "the encounter dates above. Electronically signed September 1, 2025."
    )
    assert extract_analysis_reference_date(text) == "2025-11-05"


def test_analysis_reference_date_null_when_no_defensible_date() -> None:
    text = "Patient reports fatigue. No calendar date recorded."
    assert extract_analysis_reference_date(text) is None
    assert extract_analysis_reference_date(None, "") is None


def test_legacy_parallel_arrays_fold_into_citations_or_fail_together() -> None:
    item = CriterionEvaluation(
        criterion_id="c1",
        status=CriterionStatus.MET,
        evidence_chunk_ids=["p1"],
        evidence_text=["quote"],
        page_numbers=[2],
        explanation="ok",
    )
    assert len(item.evidence) == 1
    assert item.evidence[0].page_number == 2
    with pytest.raises(EvidenceValidationError):
        CriterionEvaluation(
            criterion_id="c1",
            status=CriterionStatus.MET,
            evidence_chunk_ids=["p1", "p2"],
            evidence_text=["only one"],
            page_numbers=[1, 2],
            explanation="broken",
        )

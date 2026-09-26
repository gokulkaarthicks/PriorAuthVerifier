"""Regressions for conditional prerequisites, governing-seed filtering, packing."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import EvidenceGroundingError
from app.models.chunk import Chunk
from app.models.policy_analysis import PolicyAnalysisResult
from app.models.section import Section
from app.schemas.evaluation import (
    CriterionStatus,
    EvidenceCitation,
    ExclusionEvaluation,
    SolCriterionEvaluation,
    SolEvidenceEvaluationBatch,
    SolExclusionEvaluation,
)
from app.schemas.policy import (
    CriterionCategory,
    CriterionNode,
    CriterionNodeType,
    PolicyCriteriaExtraction,
    PolicyExclusion,
)
from app.schemas.procedure import ProcedureExtraction
from app.services.evidence_service import AtomicCriterion, EvidenceService
from app.services.policy_service import PolicyService
from app.services.readiness_service import ReadinessService
from app.services.retrieval_service import RetrievalService
from tests.helpers.evidence_sol import as_sol_response


def _chunk(chunk_id: str, text: str, *, page: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc",
        text=text,
        page_number=page,
        end_page_number=page,
        heading="Policy",
        parent_heading=None,
        token_count=10,
        section_id=f"sec-{chunk_id}",
        chunk_index=0,
    )


def _section(
    section_id: str,
    text: str,
    *,
    heading: str | None = None,
    page: int = 1,
) -> Section:
    return Section(
        id=section_id,
        document_id="policy",
        heading=heading,
        parent_heading=None,
        text=text,
        start_page=page,
        end_page=page,
        token_count=max(1, len(text.split())),
    )


def _condition(
    node_id: str,
    description: str,
    *,
    chunk_id: str = "p1",
    source_text: str | None = None,
) -> CriterionNode:
    return CriterionNode(
        id=node_id,
        node_type=CriterionNodeType.CONDITION,
        description=description,
        category=CriterionCategory.CLINICAL,
        source_chunk_id=chunk_id,
        source_text=source_text or description,
        page_number=1,
    )


def _policy_service() -> PolicyService:
    return PolicyService(
        llm_service=MagicMock(),
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
    )


def test_conditional_prerequisite_relocated_to_blocking_exclusion() -> None:
    service = _policy_service()
    tree = PolicyCriteriaExtraction(
        procedure_name="Example Device",
        policy_summary="Documentation and indication required.",
        root=CriterionNode(
            id="root",
            node_type=CriterionNodeType.AND,
            children=[
                CriterionNode(
                    id="g",
                    node_type=CriterionNodeType.AND,
                    description="General requirements",
                    children=[
                        _condition(
                            "g1",
                            "Current clinically relevant information is provided.",
                        ),
                        _condition(
                            "g2",
                            (
                                "If a same or similar device has already been placed, "
                                "clear documentation fulfills guideline criteria."
                            ),
                            source_text=(
                                "If a same or similar device has already been placed, "
                                "clear documentation fulfills guideline criteria."
                            ),
                        ),
                    ],
                ),
                _condition("ind1", "Qualifying implantation indication is met."),
            ],
        ),
        exclusions=[],
        source_chunk_ids=["p1"],
    )
    rewritten = service._relocate_conditional_prerequisites(tree)
    condition_ids = {
        node.id
        for node in _walk_conditions(rewritten.root)
    }
    assert "g2" not in condition_ids
    assert "g1" in condition_ids
    assert "ind1" in condition_ids
    assert any(
        "already been placed" in item.description.lower()
        and "without establishing" in item.description.lower()
        for item in rewritten.exclusions
    )


def test_absent_antecedent_does_not_fail_otherwise_ready_authorization() -> None:
    """No prior-device evidence → conditional block not triggered; READY if met."""
    policy = PolicyAnalysisResult(
        applicable=True,
        applicability_reason="ok",
        applicability_confidence=1.0,
        criteria=PolicyCriteriaExtraction(
            procedure_name="Example Device",
            policy_summary="Ready when indication documented.",
            root=CriterionNode(
                id="root",
                node_type=CriterionNodeType.AND,
                children=[
                    _condition("g1", "Current clinically relevant information."),
                    _condition("ind1", "Qualifying indication."),
                ],
            ),
            exclusions=[
                PolicyExclusion(
                    id="g2",
                    description=(
                        "a same or similar device has already been placed "
                        "without establishing that clear documentation fulfills "
                        "guideline criteria"
                    ),
                    source_chunk_id="p1",
                    source_text=(
                        "If a same or similar device has already been placed, "
                        "clear documentation fulfills guideline criteria."
                    ),
                    page_number=1,
                )
            ],
            source_chunk_ids=["p1"],
        ),
        supporting_chunk_ids=["p1"],
        llm_call_count=1,
    )
    evidence = EvidenceAnalysisResult_stub(
        criteria=[
            _crit("g1", CriterionStatus.MET),
            _crit("ind1", CriterionStatus.MET),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="g2",
                triggered=False,
                status=CriterionStatus.MISSING,
                evidence=[],
                explanation="No prior/similar device is documented.",
            )
        ],
    )
    result = ReadinessService().evaluate(
        ProcedureExtraction(
            procedure_name="Example Device",
            cpt_code="99999",
            is_future_procedure=True,
            confidence=0.9,
        ),
        policy,
        evidence,
    )
    assert result.overall_status.value == "READY"
    assert all(not item.triggered for item in result.exclusions)


def test_antecedent_present_requirement_absent_triggers_blocking_rule() -> None:
    policy = PolicyAnalysisResult(
        applicable=True,
        applicability_reason="ok",
        applicability_confidence=1.0,
        criteria=PolicyCriteriaExtraction(
            procedure_name="Example Device",
            policy_summary="Indication documented.",
            root=_condition("ind1", "Qualifying indication."),
            exclusions=[
                PolicyExclusion(
                    id="g2",
                    description=(
                        "a same or similar device has already been placed "
                        "without establishing that clear documentation fulfills "
                        "guideline criteria"
                    ),
                    source_chunk_id="p1",
                    source_text=(
                        "If a same or similar device has already been placed, "
                        "clear documentation fulfills guideline criteria."
                    ),
                    page_number=1,
                )
            ],
            source_chunk_ids=["p1"],
        ),
        supporting_chunk_ids=["p1"],
        llm_call_count=1,
    )
    evidence = EvidenceAnalysisResult_stub(
        criteria=[_crit("ind1", CriterionStatus.MET)],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="g2",
                triggered=True,
                status=CriterionStatus.MET,
                evidence=[
                    EvidenceCitation(
                        chunk_id="pat-1",
                        text="Prior pacemaker already implanted.",
                        page_number=2,
                    )
                ],
                explanation="Prior device present without guideline documentation.",
            )
        ],
    )
    result = ReadinessService().evaluate(
        ProcedureExtraction(
            procedure_name="Example Device",
            cpt_code="99999",
            is_future_procedure=True,
            confidence=0.9,
        ),
        policy,
        evidence,
    )
    assert result.overall_status.value == "NOT_READY"
    assert result.exclusions[0].triggered is True


def test_filter_keeps_primary_crid_block_and_drops_unrelated_procedure() -> None:
    sections = [
        _section(
            "topic",
            "Permanent Pacemaker Implantation",
            heading="Permanent Pacemaker Implantation",
            page=697,
        ),
        _section("crid7id", "(CRID-7)", heading="(CRID-7)", page=697),
        _section(
            "codes",
            "Codes included CPT 11111 22222",
            heading="Codes included",
            page=697,
        ),
        _section(
            "indications",
            (
                "Indications for Permanent Pacemaker (CRID-7)\n"
                "Sinus bradycardia at rate <50 beats per minute\n"
                "Sinus pauses >3 seconds"
            ),
            heading="Indications for Permanent Pacemaker (CRID-7)",
            page=697,
        ),
        _section(
            "continuation",
            "AVB continuation criteria",
            heading="Cardiac Implantable Devices (CID)",
            page=698,
        ),
        _section(
            "nonind",
            "Permanent pacemaker implantation is not indicated...",
            heading="Permanent Pacemaker Implantation - Non-indications (CRID-9)",
            page=700,
        ),
        _section(
            "leadless",
            (
                "Leadless right ventricular pacemaker (CRID-11.1.1)\n"
                "Indications for permanent right ventricular leadless pacemaker "
                "(CPT® 33274) implant - all of the following must be met"
            ),
            heading="Indications",
            page=703,
        ),
        _section(
            "leadless_named",
            (
                "Leadless right atrial pacemaker (CRID-11.1.3) Cardiac Implantable "
                "Devices (CID)\nIndications for permanent leadless right atrial "
                "pacemaker implant (CPT® 0823T)"
            ),
            heading=(
                "Leadless right atrial pacemaker (CRID-11.1.3) "
                "Cardiac Implantable Devices (CID)"
            ),
            page=704,
        ),
        _section(
            "proc_codes",
            "Procedure codes CPT 11111 Insertion ...",
            heading="Procedure codes (CRID-1.1)",
            page=690,
        ),
    ]
    service = RetrievalService(embedding_service=MagicMock())
    # Leadless ranks first (as in live UHC rerank); stronger procedure-name
    # overlap on the topic seed must still win the primary CRID.
    governing = [
        sections[6],  # leadless Indications + CPT 33274
        sections[7],  # leadless_named
        sections[8],  # proc_codes
        sections[0],  # topic
        sections[5],  # nonind
    ]
    filtered = service.filter_governing_seeds_for_procedure(
        governing,
        sections,
        procedure=ProcedureExtraction(
            procedure_name="Permanent Cardiac Pacemaker Implantation",
            cpt_code="11111",
            is_future_procedure=True,
            confidence=0.9,
        ),
        cpt_section_ids={"proc_codes"},
    )
    filtered_ids = {section.id for section in filtered}
    assert "topic" in filtered_ids
    assert "nonind" in filtered_ids
    assert "proc_codes" in filtered_ids
    assert "leadless" not in filtered_ids
    assert "leadless_named" not in filtered_ids

    # Include the CRID-7 indications seed so expansion can complete the block.
    filtered_with_indications = service.filter_governing_seeds_for_procedure(
        [*governing, sections[3]],
        sections,
        procedure=ProcedureExtraction(
            procedure_name="Permanent Cardiac Pacemaker Implantation",
            cpt_code="11111",
            is_future_procedure=True,
            confidence=0.9,
        ),
        cpt_section_ids={"proc_codes"},
    )
    expanded = service.expand_section_context(filtered_with_indications, sections)
    joined = "\n".join(section.text for section in expanded)
    assert "Sinus bradycardia at rate <50 beats per minute" in joined
    assert "Sinus pauses >3 seconds" in joined
    assert "Leadless right ventricular pacemaker (CRID-11.1.1)" not in joined
    assert "Leadless right atrial pacemaker (CRID-11.1.3)" not in joined

def test_evidence_packer_can_exceed_twenty_items_when_budget_allows() -> None:
    service = EvidenceService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=MagicMock(),
        settings=Settings(
            EVIDENCE_LLM_MAX_CONTEXT_TOKENS=20000,
            EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=40,
        ),
    )
    conditions = [
        AtomicCriterion(
            id=f"c{i}",
            description=f"Short criterion {i}.",
            category=CriterionCategory.CLINICAL,
            kind="condition",
        )
        for i in range(28)
    ]
    candidate_by_id = {
        f"pat-{i}": _chunk(f"pat-{i}", f"brief evidence {i}") for i in range(28)
    }
    mapping = {f"c{i}": [f"pat-{i}"] for i in range(28)}
    work_items = service._build_sol_work_items(
        conditions,
        [],
        mapping=mapping,
        candidate_by_id=candidate_by_id,
        reference_date=None,
        local_context_by_id={},
    )
    assert len(work_items) == 1
    assert len(work_items[0][0]) == 28
    assert len(work_items[0][0]) > 20


def test_token_budget_still_forces_split_with_higher_item_cap() -> None:
    service = EvidenceService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=MagicMock(),
        settings=Settings(
            EVIDENCE_LLM_MAX_CONTEXT_TOKENS=1024,
            EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=40,
        ),
    )
    conditions = [
        AtomicCriterion(
            id=f"c{i}",
            description=f"Demonstrated clinical requirement {i}. " * 20,
            category=CriterionCategory.CLINICAL,
            kind="condition",
        )
        for i in range(6)
    ]
    candidate_by_id = {
        f"pat-{i}": _chunk(
            f"pat-{i}",
            ("Patient chart evidence note with supporting detail. " * 80) + str(i),
        )
        for i in range(6)
    }
    mapping = {f"c{i}": [f"pat-{i}"] for i in range(6)}
    work_items = service._build_sol_work_items(
        conditions,
        [],
        mapping=mapping,
        candidate_by_id=candidate_by_id,
        reference_date=date(2024, 6, 1),
        local_context_by_id={f"c{i}": "" for i in range(6)},
    )
    assert sum(len(batch[0]) for batch in work_items) == 6
    assert len(work_items) >= 2


@pytest.mark.asyncio
async def test_conditional_exclusion_absent_antecedent_not_triggered() -> None:
    shared = _chunk("shared", "No prior implant. Current clinically relevant notes.")
    llm = MagicMock()
    llm.extract_structured = as_sol_response(
        SolEvidenceEvaluationBatch(
            criteria=[
                SolCriterionEvaluation(
                    criterion_id="g1",
                    status=CriterionStatus.MET,
                    evidence_candidate_ids=["E1"],
                    explanation="Current clinical info present.",
                ),
                SolCriterionEvaluation(
                    criterion_id="ind1",
                    status=CriterionStatus.MET,
                    evidence_candidate_ids=["E1"],
                    explanation="Indication supported.",
                ),
            ],
            exclusions=[
                SolExclusionEvaluation(
                    exclusion_id="g2",
                    triggered=False,
                    status=CriterionStatus.MISSING,
                    evidence_candidate_ids=[],
                    explanation="Antecedent prior device is not established.",
                )
            ],
        )
    )
    service = EvidenceService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=llm,
        settings=Settings(EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=40),
    )

    def fake_retrieve(atoms, sections, reference_date=None):  # type: ignore[no-untyped-def]
        mapping = {atom.id: ["shared"] for atom in atoms}
        return mapping, {"shared": shared}, 1, 1

    service._retrieve_evidence_for_atoms = fake_retrieve  # type: ignore[method-assign]
    policy = PolicyAnalysisResult(
        applicable=True,
        applicability_reason="ok",
        applicability_confidence=1.0,
        criteria=PolicyCriteriaExtraction(
            procedure_name="Example Device",
            policy_summary="ok",
            root=CriterionNode(
                id="root",
                node_type=CriterionNodeType.AND,
                children=[
                    _condition("g1", "Current clinically relevant information."),
                    _condition("ind1", "Qualifying indication."),
                ],
            ),
            exclusions=[
                PolicyExclusion(
                    id="g2",
                    description=(
                        "a same or similar device has already been placed without "
                        "establishing that clear documentation fulfills guideline "
                        "criteria"
                    ),
                    source_chunk_id="p1",
                    source_text=(
                        "If a same or similar device has already been placed, "
                        "clear documentation fulfills guideline criteria."
                    ),
                    page_number=1,
                )
            ],
            source_chunk_ids=["p1"],
        ),
        supporting_chunk_ids=["p1"],
        llm_call_count=1,
    )
    result = await service.evaluate_evidence(
        policy,
        [_section("s1", "No prior implant. Current clinically relevant notes.")],
    )
    assert result.exclusions[0].triggered is False
    assert "g2" not in {item.criterion_id for item in result.criteria}


def test_strict_allowed_candidate_grounding_unchanged() -> None:
    service = EvidenceService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=MagicMock(),
    )
    batch = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_candidate_ids=["E2"],
                explanation="Out of scope.",
            )
        ],
        exclusions=[],
    )
    with pytest.raises(EvidenceGroundingError):
        service._assert_candidate_ids_in_scope(
            batch,
            allowed_by_item={"c1": {"E1"}},
        )


def _walk_conditions(node: CriterionNode) -> list[CriterionNode]:
    if node.node_type == CriterionNodeType.CONDITION:
        return [node]
    found: list[CriterionNode] = []
    for child in node.children:
        found.extend(_walk_conditions(child))
    return found


def _crit(criterion_id: str, status: CriterionStatus) -> object:
    from app.schemas.evaluation import CriterionEvaluation

    return CriterionEvaluation(
        criterion_id=criterion_id,
        status=status,
        description=criterion_id,
        evidence=(
            [EvidenceCitation(chunk_id="x", text="ok", page_number=1)]
            if status == CriterionStatus.MET
            else []
        ),
        explanation="ok",
    )


def EvidenceAnalysisResult_stub(*, criteria, exclusions):  # type: ignore[no-untyped-def]
    from app.models.evidence_analysis import EvidenceAnalysisResult

    return EvidenceAnalysisResult(
        criteria=criteria,
        exclusions=exclusions,
        skipped=False,
        llm_call_count=1,
    )

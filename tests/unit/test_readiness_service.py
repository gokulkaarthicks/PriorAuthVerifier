"""Unit tests for deterministic ReadinessService aggregation."""

from __future__ import annotations

import pytest

from app.core.exceptions import ReadinessEvaluationError
from app.models.evidence_analysis import EvidenceAnalysisResult
from app.models.policy_analysis import PolicyAnalysisResult
from app.schemas.analysis import ReadinessStatus
from app.schemas.evaluation import (
    CriterionEvaluation,
    CriterionStatus,
    ExclusionEvaluation,
)
from app.schemas.policy import (
    CriterionCategory,
    CriterionNode,
    CriterionNodeType,
    PolicyCriteriaExtraction,
    PolicyExclusion,
)
from app.schemas.procedure import ProcedureExtraction
from app.services.readiness_service import ReadinessService


def _procedure() -> ProcedureExtraction:
    return ProcedureExtraction(
        procedure_name="Example Vein Ablation",
        cpt_code="99999",
        is_future_procedure=True,
        confidence=0.9,
    )


def _condition(
    node_id: str,
    description: str,
) -> CriterionNode:
    return CriterionNode(
        id=node_id,
        node_type=CriterionNodeType.CONDITION,
        description=description,
        category=CriterionCategory.CLINICAL,
        source_chunk_id="policy-1",
        source_text=description,
        page_number=1,
    )


def _eval(
    criterion_id: str,
    status: CriterionStatus,
    *,
    explanation: str | None = None,
) -> CriterionEvaluation:
    return CriterionEvaluation(
        criterion_id=criterion_id,
        status=status,
        evidence_chunk_ids=["p1"] if status != CriterionStatus.MISSING else [],
        evidence_text=["supported"] if status == CriterionStatus.MET else (
            ["ambiguous"] if status == CriterionStatus.UNCLEAR else []
        ),
        page_numbers=[1] if status != CriterionStatus.MISSING else [],
        explanation=explanation
        or (
            "Supported."
            if status == CriterionStatus.MET
            else (
                "Ambiguous evidence."
                if status == CriterionStatus.UNCLEAR
                else "No qualifying documentation found."
            )
        ),
    )


def _exclusion_eval(
    exclusion_id: str,
    *,
    triggered: bool = False,
    status: CriterionStatus = CriterionStatus.MISSING,
) -> ExclusionEvaluation:
    return ExclusionEvaluation(
        exclusion_id=exclusion_id,
        triggered=triggered,
        status=status,
        evidence_chunk_ids=["p1"] if status == CriterionStatus.MET else [],
        evidence_text=["exclusion evidence"] if status == CriterionStatus.MET else [],
        page_numbers=[1] if status == CriterionStatus.MET else [],
        explanation=(
            "Exclusion triggered."
            if triggered
            else (
                "Exclusion unclear."
                if status == CriterionStatus.UNCLEAR
                else "Exclusion not triggered."
            )
        ),
    )


def _policy(
    root: CriterionNode,
    *,
    exclusions: list[PolicyExclusion] | None = None,
) -> PolicyAnalysisResult:
    return PolicyAnalysisResult(
        applicable=True,
        applicability_reason="Policy covers the procedure.",
        applicability_confidence=0.95,
        criteria=PolicyCriteriaExtraction(
            procedure_name="Example Vein Ablation",
            policy_summary="Synthetic coverage criteria.",
            root=root,
            exclusions=exclusions or [],
            source_chunk_ids=["policy-1"],
        ),
        supporting_chunk_ids=["policy-1"],
        llm_call_count=1,
    )


def _evidence(
    criteria: list[CriterionEvaluation],
    exclusions: list[ExclusionEvaluation] | None = None,
) -> EvidenceAnalysisResult:
    return EvidenceAnalysisResult(
        criteria=criteria,
        exclusions=exclusions or [],
        skipped=False,
        llm_call_count=1,
    )


def _and_tree() -> CriterionNode:
    return CriterionNode(
        id="root",
        node_type=CriterionNodeType.AND,
        children=[
            _condition("c1", "Demonstrated saphenous reflux."),
            _condition("c2", "CEAP class is C2 or greater."),
        ],
    )


def _or_tree() -> CriterionNode:
    return CriterionNode(
        id="root",
        node_type=CriterionNodeType.OR,
        children=[
            _condition("c1", "Compression therapy failed for three months."),
            _condition("c2", "Conservative treatment failed."),
        ],
    )


def _nested_tree() -> CriterionNode:
    return CriterionNode(
        id="root",
        node_type=CriterionNodeType.AND,
        children=[
            _condition("c1", "Demonstrated saphenous reflux."),
            CriterionNode(
                id="or-1",
                node_type=CriterionNodeType.OR,
                children=[
                    _condition("c2", "Compression therapy failed for three months."),
                    _condition("c3", "Conservative treatment failed."),
                ],
            ),
        ],
    )


def test_all_met_ready() -> None:
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_and_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.MET),
            ]
        ),
    )
    assert result.overall_status == ReadinessStatus.READY
    assert result.matching_policy_criteria_found is True
    assert result.missing_items == []
    assert result.review_items == []
    assert "supported by the available chart evidence" in result.next_step


def test_and_with_one_missing_not_ready() -> None:
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_and_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.MISSING),
            ]
        ),
    )
    assert result.overall_status == ReadinessStatus.NOT_READY
    assert [item.criterion_id for item in result.missing_items] == ["c2"]
    assert "not documented" in result.missing_items[0].description.lower()
    assert "missing requirements" in result.next_step.lower()


def test_and_with_unclear_needs_review() -> None:
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_and_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.UNCLEAR),
            ]
        ),
    )
    assert result.overall_status == ReadinessStatus.NEEDS_REVIEW
    assert [item.criterion_id for item in result.review_items] == ["c2"]
    assert result.missing_items == []
    assert "manual review" in result.next_step.lower()


def test_or_with_one_met_ready() -> None:
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_or_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.MISSING),
            ]
        ),
    )
    assert result.overall_status == ReadinessStatus.READY
    assert result.missing_items == []


def test_or_with_all_missing_not_ready() -> None:
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_or_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MISSING),
                _eval("c2", CriterionStatus.MISSING),
            ]
        ),
    )
    assert result.overall_status == ReadinessStatus.NOT_READY
    # OR alternatives are not cumulative; surface one closest branch only.
    assert [item.criterion_id for item in result.missing_items] == ["c1"]


def test_or_with_missing_and_unclear_needs_review() -> None:
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_or_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MISSING),
                _eval("c2", CriterionStatus.UNCLEAR),
            ]
        ),
    )
    assert result.overall_status == ReadinessStatus.NEEDS_REVIEW
    # Closest branch prefers the UNCLEAR alternative (partial progress).
    assert [item.criterion_id for item in result.missing_items] == []
    assert [item.criterion_id for item in result.review_items] == ["c2"]


def test_nested_and_or_works() -> None:
    # AND(c1 MET, OR(c2 MISSING, c3 MET)) → READY
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_nested_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.MISSING),
                _eval("c3", CriterionStatus.MET),
            ]
        ),
    )
    assert result.overall_status == ReadinessStatus.READY
    assert result.missing_items == []

    # AND(c1 MET, OR(c2 MISSING, c3 MISSING)) → NOT_READY; one OR branch only
    failed = ReadinessService().evaluate(
        _procedure(),
        _policy(_nested_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.MISSING),
                _eval("c3", CriterionStatus.MISSING),
            ]
        ),
    )
    assert failed.overall_status == ReadinessStatus.NOT_READY
    assert [item.criterion_id for item in failed.missing_items] == ["c2"]


def test_or_prefers_closest_partially_met_branch() -> None:
    """Pacemaker-style OR: prefer SND branch with partial METs over empty peers."""
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
                    _condition("pause", "Sinus pauses >3 seconds"),
                    _condition("symptoms", "Symptoms attributable to SND"),
                ],
            ),
            CriterionNode(
                id="avb",
                node_type=CriterionNodeType.AND,
                description="Atrioventricular block",
                children=[
                    _condition("mobitz", "Second-degree Mobitz type II"),
                    _condition("chb", "Third-degree complete heart block"),
                ],
            ),
        ],
    )
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(root),
        _evidence(
            [
                _eval("hr", CriterionStatus.MET),
                _eval("pause", CriterionStatus.MET),
                _eval("symptoms", CriterionStatus.MISSING),
                _eval("mobitz", CriterionStatus.MISSING),
                _eval("chb", CriterionStatus.MISSING),
            ]
        ),
    )
    assert result.overall_status == ReadinessStatus.NOT_READY
    assert [item.criterion_id for item in result.missing_items] == ["symptoms"]
    assert "mobitz" not in {item.criterion_id for item in result.missing_items}
    assert "chb" not in {item.criterion_id for item in result.missing_items}
    assert "missing requirements" in result.next_step.lower()


def test_triggered_exclusion_not_ready() -> None:
    exclusion = PolicyExclusion(
        id="ex1",
        description="Cosmetic treatment is not covered.",
        source_chunk_id="policy-1",
        source_text="Cosmetic treatment is not covered.",
        page_number=1,
    )
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_and_tree(), exclusions=[exclusion]),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.MET),
            ],
            exclusions=[
                _exclusion_eval(
                    "ex1",
                    triggered=True,
                    status=CriterionStatus.MET,
                )
            ],
        ),
    )
    assert result.overall_status == ReadinessStatus.NOT_READY
    assert result.exclusions[0].triggered is True
    assert "non-indication" in result.next_step.lower()
    assert any(item.criterion_id == "ex1" for item in result.review_items)
    assert "missing requirements" not in result.next_step.lower()


def test_unclear_exclusion_needs_review() -> None:
    exclusion = PolicyExclusion(
        id="ex1",
        description="Cosmetic treatment is not covered.",
        source_chunk_id="policy-1",
        source_text="Cosmetic treatment is not covered.",
        page_number=1,
    )
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_and_tree(), exclusions=[exclusion]),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.MET),
            ],
            exclusions=[
                _exclusion_eval(
                    "ex1",
                    triggered=False,
                    status=CriterionStatus.UNCLEAR,
                )
            ],
        ),
    )
    assert result.overall_status == ReadinessStatus.NEEDS_REVIEW


def test_non_applicable_policy_not_applicable() -> None:
    policy = PolicyAnalysisResult(
        applicable=False,
        applicability_reason="Unrelated policy domain.",
        applicability_confidence=0.97,
        criteria=None,
        supporting_chunk_ids=["p1"],
        llm_call_count=1,
    )
    result = ReadinessService().evaluate(_procedure(), policy, None)
    assert result.overall_status == ReadinessStatus.NOT_APPLICABLE
    assert result.matching_policy_criteria_found is False
    assert result.criteria == []
    assert result.missing_items == []
    assert "no criteria for the detected planned procedure" in result.next_step.lower()


def test_missing_items_contain_only_failing_leaves() -> None:
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_nested_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MISSING),
                _eval("c2", CriterionStatus.MISSING),
                _eval("c3", CriterionStatus.UNCLEAR),
            ]
        ),
    )
    missing_ids = {item.criterion_id for item in result.missing_items}
    assert "root" not in missing_ids
    assert "or-1" not in missing_ids
    # AND missing leaf + closest OR branch (UNCLEAR preferred over empty MISSING).
    assert missing_ids == {"c1"}
    assert {item.criterion_id for item in result.review_items} == {"c3"}


def test_review_items_contain_only_unclear_leaves() -> None:
    result = ReadinessService().evaluate(
        _procedure(),
        _policy(_nested_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.MISSING),
                _eval("c3", CriterionStatus.UNCLEAR),
            ]
        ),
    )
    review_ids = {item.criterion_id for item in result.review_items}
    assert review_ids == {"c3"}
    assert "or-1" not in review_ids
    assert result.missing_items == []


def test_invalid_criterion_ids_fail() -> None:
    with pytest.raises(ReadinessEvaluationError, match="Unknown criterion"):
        ReadinessService().evaluate(
            _procedure(),
            _policy(_and_tree()),
            _evidence(
                [
                    _eval("c1", CriterionStatus.MET),
                    _eval("unknown", CriterionStatus.MET),
                ]
            ),
        )


def test_duplicate_evaluations_fail() -> None:
    with pytest.raises(ReadinessEvaluationError, match="Duplicate criterion"):
        ReadinessService().evaluate(
            _procedure(),
            _policy(_and_tree()),
            _evidence(
                [
                    _eval("c1", CriterionStatus.MET),
                    _eval("c1", CriterionStatus.MISSING),
                    _eval("c2", CriterionStatus.MET),
                ]
            ),
        )


def test_ready_not_ready_needs_review_synthetic_counts() -> None:
    service = ReadinessService()
    ready = service.evaluate(
        _procedure(),
        _policy(_and_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MET),
                _eval("c2", CriterionStatus.MET),
            ]
        ),
    )
    not_ready = service.evaluate(
        _procedure(),
        _policy(_and_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.MISSING),
                _eval("c2", CriterionStatus.MET),
            ]
        ),
    )
    needs_review = service.evaluate(
        _procedure(),
        _policy(_and_tree()),
        _evidence(
            [
                _eval("c1", CriterionStatus.UNCLEAR),
                _eval("c2", CriterionStatus.MET),
            ]
        ),
    )
    assert ready.overall_status == ReadinessStatus.READY
    assert not_ready.overall_status == ReadinessStatus.NOT_READY
    assert needs_review.overall_status == ReadinessStatus.NEEDS_REVIEW

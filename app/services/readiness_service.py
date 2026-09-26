"""Deterministic readiness aggregation over policy trees and evidence."""

from __future__ import annotations

import time
from enum import Enum

from app.core.exceptions import ReadinessEvaluationError
from app.core.logging import get_logger
from app.core.metrics import get_request_metrics, timed_substage
from app.models.evidence_analysis import EvidenceAnalysisResult
from app.models.policy_analysis import PolicyAnalysisResult
from app.schemas.analysis import (
    AnalysisResponse,
    MissingItem,
    ReadinessStatus,
    ReviewItem,
)
from app.schemas.evaluation import (
    CriterionEvaluation,
    CriterionStatus,
    ExclusionEvaluation,
)
from app.schemas.policy import CriterionNode, CriterionNodeType, PolicyExclusion
from app.schemas.procedure import ProcedureExtraction

logger = get_logger(__name__)

_NEXT_STEP_READY = (
    "Required policy criteria are supported by the available chart evidence."
)
_NEXT_STEP_NOT_APPLICABLE = (
    "No criteria for the detected planned procedure were found in the uploaded policy."
)
_NEXT_STEP_MISSING_DOCS = (
    "Obtain documentation for the missing requirements before submission."
)
_NEXT_STEP_TRIGGERED_EXCLUSION = (
    "Manual review is required because a policy non-indication is documented."
)
_NEXT_STEP_NEEDS_REVIEW = (
    "Manual review is required for ambiguous or incomplete evidence."
)


class _TriState(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class ReadinessService:
    """Aggregate criterion and exclusion evaluations into overall readiness."""

    def evaluate(
        self,
        procedure: ProcedureExtraction,
        policy_analysis: PolicyAnalysisResult,
        evidence_analysis: EvidenceAnalysisResult | None,
    ) -> AnalysisResponse:
        """Deterministically compute readiness without LLM involvement."""
        if not policy_analysis.applicable or policy_analysis.criteria is None:
            logger.info(
                "readiness_not_applicable",
                reason=policy_analysis.applicability_reason,
            )
            return AnalysisResponse(
                procedure=procedure,
                matching_policy_criteria_found=False,
                overall_status=ReadinessStatus.NOT_APPLICABLE,
                criteria=[],
                exclusions=[],
                missing_items=[],
                review_items=[],
                next_step=_NEXT_STEP_NOT_APPLICABLE,
            )

        if evidence_analysis is None or evidence_analysis.skipped:
            raise ReadinessEvaluationError(
                "Evidence analysis is required when the policy is applicable"
            )

        criteria_tree = policy_analysis.criteria
        evaluations = self._validate_criterion_evaluations(
            root=criteria_tree.root,
            evaluations=evidence_analysis.criteria,
        )
        exclusions = self._validate_exclusion_evaluations(
            policy_exclusions=criteria_tree.exclusions,
            evaluations=evidence_analysis.exclusions,
        )

        descriptions = self._condition_descriptions(criteria_tree.root)
        with timed_substage("readiness_tree_eval_ms"):
            tree_value = self._evaluate_node(criteria_tree.root, evaluations)

        with timed_substage("readiness_exclusions_ms"):
            triggered_exclusions = [
                item
                for item in exclusions
                if item.triggered or item.status == CriterionStatus.MET
            ]
            exclusion_triggered = bool(triggered_exclusions)
            exclusion_unclear = any(
                item.status == CriterionStatus.UNCLEAR for item in exclusions
            )

        with timed_substage("readiness_missing_items_ms"):
            missing_items = self._collect_missing_items(
                criteria_tree.root,
                evaluations=evaluations,
                descriptions=descriptions,
            )
            review_items = self._collect_review_items(
                criteria_tree.root,
                evaluations=evaluations,
                descriptions=descriptions,
            )

        with timed_substage("readiness_exclusions_ms"):
            # Surface triggered non-indications as review items, not "missing docs".
            for exclusion in triggered_exclusions:
                review_items.append(
                    ReviewItem(
                        criterion_id=exclusion.exclusion_id,
                        description=(
                            "Policy non-indication triggered: "
                            f"{exclusion.explanation.strip() or exclusion.exclusion_id}"
                        ),
                        explanation=exclusion.explanation,
                    )
                )

        response_started = time.perf_counter()
        overall = self._resolve_status(
            tree_value=tree_value,
            exclusion_triggered=exclusion_triggered,
            exclusion_unclear=exclusion_unclear,
        )
        next_step = self._next_step(
            overall,
            exclusion_triggered=exclusion_triggered,
            has_missing=bool(missing_items),
        )

        logger.info(
            "readiness_evaluated",
            overall_status=overall.value,
            tree_value=tree_value.value,
            exclusion_triggered=exclusion_triggered,
            exclusion_unclear=exclusion_unclear,
            missing_count=len(missing_items),
            review_count=len(review_items),
            criterion_count=len(evaluations),
            exclusion_count=len(exclusions),
        )

        response = AnalysisResponse(
            procedure=procedure,
            matching_policy_criteria_found=True,
            overall_status=overall,
            criteria=self._enrich_criteria(
                evidence_analysis.criteria,
                descriptions=descriptions,
            ),
            exclusions=list(exclusions),
            missing_items=missing_items,
            review_items=review_items,
            next_step=next_step,
        )
        metrics = get_request_metrics()
        if metrics is not None:
            metrics.add_substage_ms(
                "readiness_response_build_ms",
                int((time.perf_counter() - response_started) * 1000),
            )
        return response

    def _validate_criterion_evaluations(
        self,
        *,
        root: CriterionNode,
        evaluations: list[CriterionEvaluation],
    ) -> dict[str, CriterionEvaluation]:
        expected_ids = self._collect_condition_ids(root)
        seen: set[str] = set()
        by_id: dict[str, CriterionEvaluation] = {}

        for item in evaluations:
            if item.criterion_id in seen:
                raise ReadinessEvaluationError(
                    f"Duplicate criterion evaluation: {item.criterion_id}"
                )
            seen.add(item.criterion_id)
            if item.criterion_id not in expected_ids:
                raise ReadinessEvaluationError(
                    f"Unknown criterion evaluation id: {item.criterion_id}"
                )
            by_id[item.criterion_id] = item

        missing = expected_ids - set(by_id)
        if missing:
            raise ReadinessEvaluationError(
                "Missing evaluations for criterion ids: "
                f"{sorted(missing)}"
            )
        return by_id

    def _validate_exclusion_evaluations(
        self,
        *,
        policy_exclusions: list[PolicyExclusion],
        evaluations: list[ExclusionEvaluation],
    ) -> list[ExclusionEvaluation]:
        expected_ids = {item.id for item in policy_exclusions}
        seen: set[str] = set()
        by_id: dict[str, ExclusionEvaluation] = {}

        for item in evaluations:
            if item.exclusion_id in seen:
                raise ReadinessEvaluationError(
                    f"Duplicate exclusion evaluation: {item.exclusion_id}"
                )
            seen.add(item.exclusion_id)
            if item.exclusion_id not in expected_ids:
                raise ReadinessEvaluationError(
                    f"Unknown exclusion evaluation id: {item.exclusion_id}"
                )
            by_id[item.exclusion_id] = item

        missing = expected_ids - set(by_id)
        if missing:
            raise ReadinessEvaluationError(
                "Missing evaluations for exclusion ids: "
                f"{sorted(missing)}"
            )
        return [by_id[item.id] for item in policy_exclusions]

    def _collect_condition_ids(self, root: CriterionNode) -> set[str]:
        ids: set[str] = set()

        def walk(node: CriterionNode) -> None:
            if node.node_type == CriterionNodeType.CONDITION:
                if node.id in ids:
                    raise ReadinessEvaluationError(
                        f"Duplicate condition node id in tree: {node.id}"
                    )
                ids.add(node.id)
                return
            for child in node.children:
                walk(child)

        walk(root)
        return ids

    def _condition_descriptions(self, root: CriterionNode) -> dict[str, str]:
        descriptions: dict[str, str] = {}

        def walk(node: CriterionNode) -> None:
            if node.node_type == CriterionNodeType.CONDITION:
                descriptions[node.id] = (node.description or "").strip()
                return
            for child in node.children:
                walk(child)

        walk(root)
        return descriptions

    def _evaluate_node(
        self,
        node: CriterionNode,
        evaluations: dict[str, CriterionEvaluation],
    ) -> _TriState:
        if node.node_type == CriterionNodeType.CONDITION:
            status = evaluations[node.id].status
            if status == CriterionStatus.MET:
                return _TriState.TRUE
            if status == CriterionStatus.MISSING:
                return _TriState.FALSE
            if status == CriterionStatus.UNCLEAR:
                return _TriState.UNKNOWN
            raise ReadinessEvaluationError(
                f"Unsupported criterion status for {node.id}: {status}"
            )

        if not node.children:
            raise ReadinessEvaluationError(
                f"{node.node_type.value} node {node.id} has no children"
            )

        child_values = [
            self._evaluate_node(child, evaluations) for child in node.children
        ]

        if node.node_type == CriterionNodeType.AND:
            if any(value is _TriState.FALSE for value in child_values):
                return _TriState.FALSE
            if any(value is _TriState.UNKNOWN for value in child_values):
                return _TriState.UNKNOWN
            return _TriState.TRUE

        if node.node_type == CriterionNodeType.OR:
            if any(value is _TriState.TRUE for value in child_values):
                return _TriState.TRUE
            if any(value is _TriState.UNKNOWN for value in child_values):
                return _TriState.UNKNOWN
            return _TriState.FALSE

        raise ReadinessEvaluationError(f"Unknown node type: {node.node_type}")

    def _resolve_status(
        self,
        *,
        tree_value: _TriState,
        exclusion_triggered: bool,
        exclusion_unclear: bool,
    ) -> ReadinessStatus:
        if exclusion_triggered or tree_value is _TriState.FALSE:
            return ReadinessStatus.NOT_READY
        if exclusion_unclear or tree_value is _TriState.UNKNOWN:
            return ReadinessStatus.NEEDS_REVIEW
        return ReadinessStatus.READY

    def _collect_missing_items(
        self,
        node: CriterionNode,
        *,
        evaluations: dict[str, CriterionEvaluation],
        descriptions: dict[str, str],
    ) -> list[MissingItem]:
        if node.node_type == CriterionNodeType.CONDITION:
            evaluation = evaluations[node.id]
            if evaluation.status != CriterionStatus.MISSING:
                return []
            description = descriptions.get(node.id) or evaluation.explanation
            return [
                MissingItem(
                    criterion_id=node.id,
                    description=self._missing_description(
                        description,
                        explanation=evaluation.explanation,
                    ),
                )
            ]

        if node.node_type == CriterionNodeType.OR:
            tree_value = self._evaluate_node(node, evaluations)
            if tree_value is _TriState.TRUE:
                # Satisfied OR branch does not surface alternative missing leaves.
                return []
            closest = self._closest_or_children(node, evaluations=evaluations)
            or_missing: list[MissingItem] = []
            for child in closest:
                or_missing.extend(
                    self._collect_missing_items(
                        child,
                        evaluations=evaluations,
                        descriptions=descriptions,
                    )
                )
            return or_missing

        and_missing: list[MissingItem] = []
        for child in node.children:
            and_missing.extend(
                self._collect_missing_items(
                    child,
                    evaluations=evaluations,
                    descriptions=descriptions,
                )
            )
        return and_missing

    def _collect_review_items(
        self,
        node: CriterionNode,
        *,
        evaluations: dict[str, CriterionEvaluation],
        descriptions: dict[str, str],
    ) -> list[ReviewItem]:
        if node.node_type == CriterionNodeType.CONDITION:
            evaluation = evaluations[node.id]
            if evaluation.status != CriterionStatus.UNCLEAR:
                return []
            description = descriptions.get(node.id) or evaluation.explanation
            return [
                ReviewItem(
                    criterion_id=node.id,
                    description=description,
                    explanation=evaluation.explanation,
                )
            ]

        if node.node_type == CriterionNodeType.OR:
            tree_value = self._evaluate_node(node, evaluations)
            if tree_value is _TriState.TRUE:
                return []
            closest = self._closest_or_children(node, evaluations=evaluations)
            or_review: list[ReviewItem] = []
            for child in closest:
                or_review.extend(
                    self._collect_review_items(
                        child,
                        evaluations=evaluations,
                        descriptions=descriptions,
                    )
                )
            return or_review

        and_review: list[ReviewItem] = []
        for child in node.children:
            and_review.extend(
                self._collect_review_items(
                    child,
                    evaluations=evaluations,
                    descriptions=descriptions,
                )
            )
        return and_review
    def _closest_or_children(
        self,
        node: CriterionNode,
        *,
        evaluations: dict[str, CriterionEvaluation],
    ) -> list[CriterionNode]:
        """Pick the OR alternative(s) closest to satisfaction.

        Alternative OR pathways are alternatives, not cumulative requirements.
        Prefer the branch with the most MET leaves, then fewest remaining gaps.
        Ties keep only the earliest branch so users are not told to satisfy
        every alternative pathway at once.
        """
        if not node.children:
            return []

        scored: list[tuple[tuple[int, int, int, int], int, CriterionNode]] = []
        for index, child in enumerate(node.children):
            met, unclear, missing = self._leaf_status_counts(child, evaluations)
            # Prefer more MET, then UNCLEAR (partial evidence) over empty MISSING.
            score = (-met, -unclear, missing, index)
            scored.append((score, index, child))

        scored.sort(key=lambda item: item[0])
        best_score = scored[0][0]
        # When every alternative is equally empty, keep only the first pathway.
        if best_score[0] == 0 and best_score[1] == 0:
            return [scored[0][2]]
        return [scored[0][2]]

    def _leaf_status_counts(
        self,
        node: CriterionNode,
        evaluations: dict[str, CriterionEvaluation],
    ) -> tuple[int, int, int]:
        met = 0
        unclear = 0
        missing = 0

        def walk(current: CriterionNode) -> None:
            nonlocal met, unclear, missing
            if current.node_type == CriterionNodeType.CONDITION:
                status = evaluations[current.id].status
                if status == CriterionStatus.MET:
                    met += 1
                elif status == CriterionStatus.UNCLEAR:
                    unclear += 1
                else:
                    missing += 1
                return
            for child in current.children:
                walk(child)

        walk(node)
        return met, unclear, missing

    def _missing_description(
        self,
        description: str,
        *,
        explanation: str = "",
    ) -> str:
        text = description.strip()
        explanation_lower = explanation.lower()
        # Affirmative contrary evidence is not "documentation missing".
        if any(
            token in explanation_lower
            for token in (
                "affirmatively",
                "contradict",
                "disprov",
                "documented absence",
                "documented as absent",
                "not present",
                "rules out",
            )
        ):
            if not text:
                return "Chart evidence affirmatively conflicts with this requirement."
            lowered = text.rstrip(".")
            return (
                f"{lowered} is affirmatively not met by the available chart evidence."
            )
        if not text:
            return "Required documentation is missing."
        lowered = text.rstrip(".")
        if lowered.lower().endswith("is not documented"):
            return text if text.endswith(".") else f"{text}."
        return f"{lowered} is not documented."

    def _next_step(
        self,
        status: ReadinessStatus,
        *,
        exclusion_triggered: bool = False,
        has_missing: bool = False,
    ) -> str:
        if status == ReadinessStatus.READY:
            return _NEXT_STEP_READY
        if status == ReadinessStatus.NOT_APPLICABLE:
            return _NEXT_STEP_NOT_APPLICABLE
        if exclusion_triggered:
            return _NEXT_STEP_TRIGGERED_EXCLUSION
        if status == ReadinessStatus.NOT_READY and has_missing:
            return _NEXT_STEP_MISSING_DOCS
        if status == ReadinessStatus.NOT_READY:
            return _NEXT_STEP_MISSING_DOCS
        return _NEXT_STEP_NEEDS_REVIEW

    def _enrich_criteria(
        self,
        criteria: list[CriterionEvaluation],
        *,
        descriptions: dict[str, str],
    ) -> list[CriterionEvaluation]:
        enriched: list[CriterionEvaluation] = []
        for item in criteria:
            enriched.append(
                item.model_copy(
                    update={
                        "description": descriptions.get(
                            item.criterion_id,
                            item.description,
                        )
                    }
                )
            )
        return enriched

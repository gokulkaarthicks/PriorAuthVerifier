"""Analysis response API schemas for final readiness aggregation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.evaluation import (
    CriterionEvaluation,
    ExclusionEvaluation,
)
from app.schemas.policy import CriterionNode
from app.schemas.procedure import ProcedureExtraction


class ReadinessStatus(StrEnum):
    """Overall prior-authorization readiness outcome."""

    READY = "READY"
    NOT_READY = "NOT_READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class MissingItem(BaseModel):
    """An actionable leaf criterion that lacks supporting evidence."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    description: str


class ReviewItem(BaseModel):
    """A leaf criterion requiring manual review due to ambiguity."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    description: str
    explanation: str


class AnalysisResponse(BaseModel):
    """Top-level prior-authorization analysis response."""

    model_config = ConfigDict(extra="forbid")

    procedure: ProcedureExtraction
    matching_policy_criteria_found: bool
    overall_status: ReadinessStatus
    criteria: list[CriterionEvaluation] = Field(default_factory=list)
    exclusions: list[ExclusionEvaluation] = Field(default_factory=list)
    missing_items: list[MissingItem] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)
    next_step: str = ""
    # Existing evaluated tree for presentation/audit only; not used in readiness.
    criteria_tree: CriterionNode | None = None

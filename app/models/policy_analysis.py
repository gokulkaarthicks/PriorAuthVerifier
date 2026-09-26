"""Internal policy analysis result model."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.policy import PolicyCriteriaExtraction


@dataclass(frozen=True, slots=True)
class PolicyAnalysisResult:
    """Outcome of the single policy-criteria Sol call."""

    applicable: bool
    applicability_reason: str
    applicability_confidence: float
    criteria: PolicyCriteriaExtraction | None
    supporting_chunk_ids: list[str] = field(default_factory=list)
    llm_call_count: int = 0

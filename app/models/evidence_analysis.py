"""Internal evidence analysis result model."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.evaluation import CriterionEvaluation, ExclusionEvaluation


@dataclass(frozen=True, slots=True)
class EvidenceAnalysisResult:
    """Outcome of evaluating policy criteria against patient evidence."""

    criteria: list[CriterionEvaluation] = field(default_factory=list)
    exclusions: list[ExclusionEvaluation] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
    llm_call_count: int = 0
    retrieval_duration_ms: int = 0
    reranking_duration_ms: int = 0
    llm_duration_ms: int = 0

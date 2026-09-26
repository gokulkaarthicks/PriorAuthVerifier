"""Evaluation-related API schemas for criterion evidence judgments."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from app.core.exceptions import EvidenceValidationError
from app.models.chunk import Chunk


class CriterionStatus(StrEnum):
    """Outcome of evaluating a single policy criterion."""

    MET = "MET"
    MISSING = "MISSING"
    UNCLEAR = "UNCLEAR"


class EvidenceCitation(BaseModel):
    """One grounded quote with inseparable source metadata."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    page_number: int | None = None
    document_date: str | None = None


class CriterionEvaluation(BaseModel):
    """Evaluation result for one atomic policy criterion."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    status: CriterionStatus
    description: str | None = None
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    explanation: str
    temporal_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _fold_legacy_parallel_arrays(cls, data: object) -> object:
        return _fold_parallel_arrays_into_evidence(data, item_id_key="criterion_id")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_chunk_ids(self) -> list[str]:
        """Derived from ``evidence`` — never maintained independently."""
        return [item.chunk_id for item in self.evidence]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_text(self) -> list[str]:
        """Derived from ``evidence`` — never maintained independently."""
        return [item.text for item in self.evidence]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def page_numbers(self) -> list[int]:
        """Derived from ``evidence`` — never maintained independently."""
        return [
            int(item.page_number)
            for item in self.evidence
            if item.page_number is not None
        ]


class ExclusionEvaluation(BaseModel):
    """Evaluation result for one policy exclusion."""

    model_config = ConfigDict(extra="forbid")

    exclusion_id: str
    triggered: bool
    status: CriterionStatus
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    explanation: str
    temporal_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _fold_legacy_parallel_arrays(cls, data: object) -> object:
        return _fold_parallel_arrays_into_evidence(data, item_id_key="exclusion_id")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_chunk_ids(self) -> list[str]:
        return [item.chunk_id for item in self.evidence]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_text(self) -> list[str]:
        return [item.text for item in self.evidence]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def page_numbers(self) -> list[int]:
        return [
            int(item.page_number)
            for item in self.evidence
            if item.page_number is not None
        ]


class EvidenceEvaluationBatch(BaseModel):
    """Public batch of criterion/exclusion evaluations."""

    model_config = ConfigDict(extra="forbid")

    criteria: list[CriterionEvaluation] = Field(default_factory=list)
    exclusions: list[ExclusionEvaluation] = Field(default_factory=list)


class SolCriterionEvaluation(BaseModel):
    """Sol structured output: cites shared corpus candidate IDs only."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    status: CriterionStatus
    evidence_candidate_ids: list[str] = Field(default_factory=list)
    explanation: str


class SolExclusionEvaluation(BaseModel):
    """Sol structured output: cites shared corpus candidate IDs only."""

    model_config = ConfigDict(extra="forbid")

    exclusion_id: str
    triggered: bool
    status: CriterionStatus
    evidence_candidate_ids: list[str] = Field(default_factory=list)
    explanation: str


class SolEvidenceEvaluationBatch(BaseModel):
    """Sol structured response using criterion-scoped candidate IDs."""

    model_config = ConfigDict(extra="forbid")

    criteria: list[SolCriterionEvaluation] = Field(default_factory=list)
    exclusions: list[SolExclusionEvaluation] = Field(default_factory=list)


def _fold_parallel_arrays_into_evidence(
    data: object,
    *,
    item_id_key: str,
) -> object:
    """Single conversion path from legacy parallel arrays → EvidenceCitation."""
    if not isinstance(data, dict):
        return data
    if data.get("evidence") is not None:
        # Drop stale parallel keys so they cannot drift from ``evidence``.
        data = {
            key: value
            for key, value in data.items()
            if key
            not in {"evidence_chunk_ids", "evidence_text", "page_numbers"}
        }
        return data

    chunk_ids = list(data.get("evidence_chunk_ids") or [])
    texts = list(data.get("evidence_text") or [])
    pages = list(data.get("page_numbers") or [])
    item_id = str(data.get(item_id_key) or "item")
    if (texts or chunk_ids) and len(texts) != len(chunk_ids):
        raise EvidenceValidationError(
            f"{item_id} evidence_text length must match evidence_chunk_ids"
        )
    if pages and len(pages) != len(chunk_ids):
        raise EvidenceValidationError(
            f"{item_id} page_numbers length must match evidence_chunk_ids"
        )
    evidence = [
        {
            "chunk_id": chunk_id,
            "text": texts[index] if index < len(texts) else "",
            "page_number": pages[index] if index < len(pages) else None,
        }
        for index, chunk_id in enumerate(chunk_ids)
    ]
    folded = {
        key: value
        for key, value in data.items()
        if key not in {"evidence_chunk_ids", "evidence_text", "page_numbers"}
    }
    folded["evidence"] = evidence
    return folded


def citation_from_chunk(chunk: Chunk) -> EvidenceCitation:
    """Build a citation whose text/page/date always come from the chunk."""
    return EvidenceCitation(
        chunk_id=chunk.id,
        text=chunk.text,
        page_number=chunk.page_number,
        document_date=chunk.document_date,
    )


def enrich_citation(
    citation: EvidenceCitation,
    *,
    chunk: Chunk | None,
) -> EvidenceCitation:
    """Fill page/date from the source chunk so metadata cannot drift apart.

    Prefer chunk metadata for page/date/text alignment. ``document_date`` comes
    only from the source chunk — never from the runtime clock.
    """
    if chunk is None:
        return citation
    return EvidenceCitation(
        chunk_id=chunk.id,
        text=chunk.text,
        page_number=chunk.page_number,
        document_date=chunk.document_date,
    )


def materialize_sol_criterion(
    item: SolCriterionEvaluation,
    *,
    candidate_map: Mapping[str, EvidenceCitation],
    description: str | None = None,
) -> CriterionEvaluation:
    """Resolve Sol candidate IDs into the public EvidenceCitation schema."""
    evidence = [
        candidate_map[candidate_id] for candidate_id in item.evidence_candidate_ids
    ]
    return CriterionEvaluation(
        criterion_id=item.criterion_id,
        status=item.status,
        description=description,
        evidence=evidence,
        explanation=item.explanation,
    )


def materialize_sol_exclusion(
    item: SolExclusionEvaluation,
    *,
    candidate_map: Mapping[str, EvidenceCitation],
) -> ExclusionEvaluation:
    """Resolve Sol candidate IDs into the public EvidenceCitation schema."""
    evidence = [
        candidate_map[candidate_id] for candidate_id in item.evidence_candidate_ids
    ]
    return ExclusionEvaluation(
        exclusion_id=item.exclusion_id,
        triggered=item.triggered,
        status=item.status,
        evidence=evidence,
        explanation=item.explanation,
    )


def materialize_sol_batch(
    batch: SolEvidenceEvaluationBatch,
    *,
    candidate_map: Mapping[str, EvidenceCitation],
    descriptions_by_id: Mapping[str, str] | None = None,
) -> EvidenceEvaluationBatch:
    """Convert Sol candidate-ID batch into the public citation schema."""
    descriptions = descriptions_by_id or {}
    return EvidenceEvaluationBatch(
        criteria=[
            materialize_sol_criterion(
                item,
                candidate_map=candidate_map,
                description=descriptions.get(item.criterion_id),
            )
            for item in batch.criteria
        ],
        exclusions=[
            materialize_sol_exclusion(item, candidate_map=candidate_map)
            for item in batch.exclusions
        ],
    )


def parallel_to_sol_batch(batch: EvidenceEvaluationBatch) -> SolEvidenceEvaluationBatch:
    """Test helper: map public citations to sequential shared corpus IDs.

    IDs are ``E{n}`` in first-seen citation order across the batch. Prefer the
    prompt-aware adapter in evidence tests when retrieval order may differ.
    """
    chunk_to_candidate: dict[str, str] = {}

    def resolve(evidence: list[EvidenceCitation]) -> list[str]:
        resolved: list[str] = []
        for citation in evidence:
            candidate_id = chunk_to_candidate.get(citation.chunk_id)
            if candidate_id is None:
                candidate_id = f"E{len(chunk_to_candidate) + 1}"
                chunk_to_candidate[citation.chunk_id] = candidate_id
            resolved.append(candidate_id)
        return resolved

    return SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id=item.criterion_id,
                status=item.status,
                evidence_candidate_ids=resolve(item.evidence),
                explanation=item.explanation,
            )
            for item in batch.criteria
        ],
        exclusions=[
            SolExclusionEvaluation(
                exclusion_id=item.exclusion_id,
                triggered=item.triggered,
                status=item.status,
                evidence_candidate_ids=resolve(item.evidence),
                explanation=item.explanation,
            )
            for item in batch.exclusions
        ],
    )

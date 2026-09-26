"""Policy-related schemas for criteria matching and extraction."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CriterionNodeType(StrEnum):
    """Logical node kinds in a policy criteria tree."""

    CONDITION = "CONDITION"
    AND = "AND"
    OR = "OR"


class CriterionCategory(StrEnum):
    """Generic categories for atomic policy conditions."""

    CLINICAL = "CLINICAL"
    DIAGNOSTIC = "DIAGNOSTIC"
    TREATMENT_HISTORY = "TREATMENT_HISTORY"
    DURATION = "DURATION"
    MEASUREMENT = "MEASUREMENT"
    DOCUMENTATION = "DOCUMENTATION"
    EXCLUSION = "EXCLUSION"
    OTHER = "OTHER"


_SOURCE_QUOTE_MAX_CHARS = 200


class CriterionNode(BaseModel):
    """Public/normalized node in a nested policy criteria tree."""

    model_config = ConfigDict(extra="forbid")

    id: str
    node_type: CriterionNodeType
    description: str | None = Field(
        default=None,
        description="Short requirement text; keep AND/OR descriptions brief.",
    )
    category: CriterionCategory | None = None
    temporal_requirement: str | None = Field(
        default=None,
        description="Explicit policy timing only when stated; otherwise null.",
    )
    children: list[CriterionNode] = Field(default_factory=list)
    source_chunk_id: str | None = None
    source_text: str | None = None
    page_number: int | None = Field(default=None, ge=1)


class PolicyExclusion(BaseModel):
    """Public/normalized policy exclusion kept separate from positive criteria."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    source_chunk_id: str
    source_text: str
    page_number: int = Field(ge=1)


class PolicyCriteriaExtraction(BaseModel):
    """Structured criteria tree extracted from governing policy context."""

    model_config = ConfigDict(extra="forbid")

    procedure_name: str
    policy_summary: str = Field(
        description="One concise sentence summarizing the governing criteria."
    )
    root: CriterionNode
    exclusions: list[PolicyExclusion] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class PolicyCriteriaResult(BaseModel):
    """Public matching decision + optional criteria tree (post-hydration)."""

    model_config = ConfigDict(extra="forbid")

    matching_policy_criteria_found: bool
    explanation: str | None = Field(
        default=None,
        description="Optional one-sentence explanation; omit verbose analysis.",
    )
    criteria_tree: PolicyCriteriaExtraction | None = None

    @model_validator(mode="after")
    def enforce_tree_consistency(self) -> PolicyCriteriaResult:
        if self.matching_policy_criteria_found:
            if self.criteria_tree is None:
                raise ValueError(
                    "criteria_tree is required when matching_policy_criteria_found "
                    "is true"
                )
        elif self.criteria_tree is not None:
            raise ValueError(
                "criteria_tree must be null when matching_policy_criteria_found "
                "is false"
            )
        return self


class SolCriterionNode(BaseModel):
    """Compact Sol tree node using P-aliases instead of chunk UUIDs/pages.

    AND/OR nodes omit grounding. CONDITION leaves require source_context_id
    and a short source_quote. Kept as one recursive model so OpenAI structured
    outputs accept the schema (discriminated unions + recursion are rejected).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    node_type: CriterionNodeType
    description: str | None = Field(
        default=None,
        description="Short requirement text; keep AND/OR descriptions brief.",
    )
    category: CriterionCategory | None = None
    temporal_requirement: str | None = Field(
        default=None,
        description="Explicit policy timing only when stated; otherwise null.",
    )
    children: list[SolCriterionNode] = Field(default_factory=list)
    source_context_id: str | None = Field(
        default=None,
        description="Policy context alias such as P1 for CONDITION leaves only.",
    )
    source_quote: str | None = Field(
        default=None,
        description=(
            "Short exact supporting policy clause from the cited context "
            f"(<= {_SOURCE_QUOTE_MAX_CHARS} characters). CONDITION only."
        ),
    )

    @field_validator("source_quote")
    @classmethod
    def _bound_source_quote(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        if len(text) > _SOURCE_QUOTE_MAX_CHARS:
            return text[:_SOURCE_QUOTE_MAX_CHARS]
        return text

    @model_validator(mode="after")
    def _enforce_condition_grounding(self) -> SolCriterionNode:
        if self.node_type == CriterionNodeType.CONDITION:
            if self.children:
                raise ValueError("CONDITION nodes must not have children")
            if not (self.description or "").strip():
                raise ValueError("CONDITION nodes require a description")
            if not (self.source_context_id or "").strip():
                raise ValueError("CONDITION nodes require source_context_id")
            if not (self.source_quote or "").strip():
                raise ValueError("CONDITION nodes require source_quote")
        return self


class SolPolicyExclusion(BaseModel):
    """Compact Sol exclusion grounded by context alias + short quote."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    source_context_id: str
    source_quote: str

    @field_validator("source_quote")
    @classmethod
    def _bound_source_quote(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("source_quote must be non-empty")
        if len(text) > _SOURCE_QUOTE_MAX_CHARS:
            return text[:_SOURCE_QUOTE_MAX_CHARS]
        return text


class SolPolicyCriteriaExtraction(BaseModel):
    """Compact Sol criteria tree using P-aliases instead of chunk UUIDs."""

    model_config = ConfigDict(extra="forbid")

    procedure_name: str
    policy_summary: str = Field(
        description="One concise sentence summarizing the governing criteria."
    )
    root: SolCriterionNode
    exclusions: list[SolPolicyExclusion] = Field(default_factory=list)
    source_context_ids: list[str] = Field(default_factory=list)


class SolPolicyCriteriaResult(BaseModel):
    """Sol response model: matching decision + compact criteria tree."""

    model_config = ConfigDict(extra="forbid")

    matching_policy_criteria_found: bool
    explanation: str | None = Field(
        default=None,
        description="Optional one-sentence explanation; omit verbose analysis.",
    )
    criteria_tree: SolPolicyCriteriaExtraction | None = None

    @model_validator(mode="after")
    def enforce_tree_consistency(self) -> SolPolicyCriteriaResult:
        if self.matching_policy_criteria_found:
            if self.criteria_tree is None:
                raise ValueError(
                    "criteria_tree is required when matching_policy_criteria_found "
                    "is true"
                )
        elif self.criteria_tree is not None:
            raise ValueError(
                "criteria_tree must be null when matching_policy_criteria_found "
                "is false"
            )
        return self

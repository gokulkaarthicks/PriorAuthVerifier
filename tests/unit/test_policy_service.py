"""Unit tests for PolicyService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import PolicyValidationError
from app.models.chunk import Chunk
from app.models.retrieval import RetrievalResult
from app.schemas.policy import (
    CriterionCategory,
    CriterionNode,
    CriterionNodeType,
    PolicyCriteriaExtraction,
    SolCriterionNode,
    SolPolicyCriteriaExtraction,
    SolPolicyCriteriaResult,
    SolPolicyExclusion,
)
from app.schemas.procedure import ProcedureExtraction
from app.services.policy_service import PolicyService


def _chunk(
    chunk_id: str,
    text: str,
    *,
    heading: str = "Policy",
    page: int = 1,
    end_page: int | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="policy",
        text=text,
        page_number=page,
        end_page_number=end_page or page,
        heading=heading,
        parent_heading=None,
        token_count=20,
        section_id=f"sec-{chunk_id}",
        chunk_index=0,
    )


def _candidate(chunk: Chunk, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk=chunk,
        rank=rank,
        score=0.9,
        retrieval_method="rerank",
        reranker_score=0.9,
    )


def _procedure() -> ProcedureExtraction:
    return ProcedureExtraction(
        procedure_name="Example Procedure",
        cpt_code="12345",
        is_future_procedure=True,
        confidence=0.9,
    )


def _sol_condition(
    node_id: str,
    description: str,
    *,
    source_context_id: str = "P1",
    source_quote: str,
    category: CriterionCategory | None = CriterionCategory.CLINICAL,
) -> SolCriterionNode:
    return SolCriterionNode(
        id=node_id,
        node_type="CONDITION",
        description=description,
        category=category,
        source_context_id=source_context_id,
        source_quote=source_quote,
    )


def _sol_extraction(
    root: SolCriterionNode,
    *,
    procedure_name: str = "Example Procedure",
    policy_summary: str = "Summary",
    exclusions: list[SolPolicyExclusion] | None = None,
    source_context_ids: list[str] | None = None,
) -> SolPolicyCriteriaExtraction:
    return SolPolicyCriteriaExtraction(
        procedure_name=procedure_name,
        policy_summary=policy_summary,
        root=root,
        exclusions=exclusions or [],
        source_context_ids=source_context_ids or ["P1"],
    )


def _match_result(
    root: SolCriterionNode,
    *,
    explanation: str = "Policy governs the detected procedure.",
    **extraction_kwargs: object,
) -> SolPolicyCriteriaResult:
    return SolPolicyCriteriaResult(
        matching_policy_criteria_found=True,
        explanation=explanation,
        criteria_tree=_sol_extraction(root, **extraction_kwargs),  # type: ignore[arg-type]
    )


def _no_match_result(explanation: str) -> SolPolicyCriteriaResult:
    return SolPolicyCriteriaResult(
        matching_policy_criteria_found=False,
        explanation=explanation,
        criteria_tree=None,
    )


def _public_condition(
    node_id: str,
    description: str,
    *,
    chunk_id: str,
    source_text: str,
    page: int = 1,
) -> CriterionNode:
    return CriterionNode(
        id=node_id,
        node_type=CriterionNodeType.CONDITION,
        description=description,
        category=CriterionCategory.CLINICAL,
        source_chunk_id=chunk_id,
        source_text=source_text,
        page_number=page,
    )


@pytest.fixture
def llm() -> MagicMock:
    mock = MagicMock()
    mock.extract_structured = AsyncMock()
    return mock


@pytest.fixture
def service(llm: MagicMock) -> PolicyService:
    return PolicyService(
        llm_service=llm,
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        settings=Settings(POLICY_REASONING_MODEL="gpt-6-sol"),
    )


@pytest.mark.asyncio
async def test_policy_requests_low_reasoning_effort(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Example Procedure coverage criteria.")
    llm.extract_structured.return_value = _no_match_result("Unrelated.")

    await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    kwargs = llm.extract_structured.await_args.kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_output_tokens"] == 16000
    assert llm.extract_structured.await_count == 1


@pytest.mark.asyncio
async def test_policy_prompt_is_deterministic_and_non_duplicative(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Example Procedure coverage criteria.")
    llm.extract_structured.return_value = _no_match_result("Unrelated.")

    await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    system_prompt = llm.extract_structured.await_args.kwargs["system_prompt"]
    user_prompt = llm.extract_structured.await_args.kwargs["user_prompt"]
    assert "structured policy extraction" in system_prompt.lower()
    assert "do not explain internal reasoning" in system_prompt.lower()
    assert system_prompt.lower().count("use only the supplied") == 1
    assert "bm25" not in system_prompt.lower()
    assert "rerank" not in system_prompt.lower()
    assert "think step by step" not in system_prompt.lower()
    assert "procedure_name: Example Procedure" in user_prompt
    assert "cpt_code: 12345" in user_prompt
    assert "Extract the governing medical-necessity criteria" in user_prompt
    assert user_prompt.count("Extract the governing medical-necessity criteria") == 1


@pytest.mark.asyncio
async def test_policy_input_and_output_budgets_are_independent(
    llm: MagicMock,
) -> None:
    settings = Settings(
        POLICY_REASONING_MODEL="gpt-6-sol",
        POLICY_LLM_MAX_CONTEXT_TOKENS=28000,
        POLICY_LLM_MAX_OUTPUT_TOKENS=16000,
    )
    service = PolicyService(
        llm_service=llm,
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        settings=settings,
    )
    assert settings.POLICY_LLM_MAX_CONTEXT_TOKENS == 28000
    assert settings.POLICY_LLM_MAX_OUTPUT_TOKENS == 16000
    # Packing uses the full context budget; output is not subtracted.
    prepared = service._prepare_policy_context(
        _procedure(),
        [_candidate(_chunk("c1", "Example Procedure criteria."))],
    )
    assert len(prepared.candidates) == 1
    assert "P1" in prepared.aliases
    llm.extract_structured.return_value = _no_match_result("No match.")
    await service._analyze_candidates(_procedure(), prepared.candidates)
    assert llm.extract_structured.await_args.kwargs["max_output_tokens"] == 16000


@pytest.mark.asyncio
async def test_unified_policy_prompt_targets_procedure_not_payer(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk(
        "c1",
        "Aetna Medical Policy. Coverage for Example Procedure CPT 12345.",
    )
    llm.extract_structured.return_value = _match_result(
        _sol_condition(
            "c-1",
            "Documented clinical indication.",
            source_quote="Coverage for Example Procedure CPT 12345",
        ),
        policy_summary="Coverage for Example Procedure.",
    )

    await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    call = llm.extract_structured.await_args
    system_prompt = call.kwargs["system_prompt"]
    user_prompt = call.kwargs["user_prompt"]
    assert "matching_policy_criteria_found" in system_prompt
    assert "payer" in system_prompt.lower()
    assert "publisher" in system_prompt.lower()
    assert call.kwargs["response_model"] is SolPolicyCriteriaResult
    assert "procedure_name: Example Procedure" in user_prompt
    assert "cpt_code: 12345" in user_prompt
    assert "POLICY CONTEXT P1" in user_prompt
    assert "chunk_id:" not in user_prompt
    assert "page_number:" not in user_prompt
    assert llm.extract_structured.await_count == 1


@pytest.mark.asyncio
async def test_single_sol_call_passes_output_token_limit(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Example Procedure is covered when indication is documented.")
    llm.extract_structured.return_value = _match_result(
        _sol_condition(
            "c-1",
            "Indication is documented.",
            source_quote="indication is documented",
        ),
        policy_summary="Covered with documented indication.",
    )

    await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    kwargs = llm.extract_structured.await_args.kwargs
    assert kwargs["response_model"] is SolPolicyCriteriaResult
    assert kwargs["model"] == "gpt-6-sol"
    assert kwargs["operation"] == "policy_criteria"
    assert kwargs["max_output_tokens"] == service._settings.POLICY_LLM_MAX_OUTPUT_TOKENS
    assert llm.extract_structured.await_count == 1


@pytest.mark.asyncio
async def test_applicable_policy_returns_true(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk(
        "c1",
        "Coverage applies to Example Procedure CPT 12345 when criteria are met.",
    )
    llm.extract_structured.return_value = _match_result(
        _sol_condition(
            "c-1",
            "Documented clinical indication.",
            source_quote="when criteria are met",
        ),
        explanation="CPT 12345 and procedure name appear in policy.",
        policy_summary="Coverage for Example Procedure.",
    )

    result = await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    assert result.applicable is True
    assert result.criteria is not None
    assert result.criteria.root.source_chunk_id == "c1"
    assert result.criteria.root.page_number == 1
    assert result.llm_call_count == 1


@pytest.mark.asyncio
async def test_empty_criteria_tree_coerced_to_non_matching(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Example Procedure is mentioned without criteria.")
    llm.extract_structured.return_value = SolPolicyCriteriaResult(
        matching_policy_criteria_found=True,
        explanation="Named but no extractable criteria.",
        criteria_tree=SolPolicyCriteriaExtraction(
            procedure_name="Example Procedure",
            policy_summary="No affirmative criteria present.",
            root=SolCriterionNode(
                id="root",
                node_type="AND",
                description="Empty",
                children=[],
            ),
            exclusions=[],
            source_context_ids=["P1"],
        ),
    )

    result = await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    assert result.applicable is False
    assert result.criteria is None
    assert result.llm_call_count == 1


@pytest.mark.asyncio
async def test_unrelated_policy_returns_false(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "This policy covers emergency ambulance transport only.")
    llm.extract_structured.return_value = _no_match_result(
        "Policy covers ambulance transport, not the detected procedure."
    )

    result = await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    assert result.applicable is False
    assert result.criteria is None
    assert result.llm_call_count == 1
    assert llm.extract_structured.await_count == 1


@pytest.mark.asyncio
async def test_duplicate_chunk_id_and_text_removed_from_prompt(
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Example Procedure coverage criteria.")
    duplicate = _candidate(chunk, rank=1)
    duplicate_copy = _candidate(
        _chunk("c1", "Example Procedure coverage criteria."),
        rank=2,
    )
    service = PolicyService(
        llm_service=llm,
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        settings=Settings(POLICY_REASONING_MODEL="gpt-6-sol"),
    )
    llm.extract_structured.return_value = _no_match_result("Unrelated domain.")

    await service._analyze_candidates(
        _procedure(),
        [duplicate, duplicate_copy],
    )

    user_prompt = llm.extract_structured.await_args.kwargs["user_prompt"]
    assert user_prompt.count("POLICY CONTEXT P1") == 1
    assert "POLICY CONTEXT P2" not in user_prompt


@pytest.mark.asyncio
async def test_token_budget_preserves_highest_ranked_chunk(
    llm: MagicMock,
) -> None:
    settings = Settings(
        POLICY_REASONING_MODEL="gpt-6-sol",
        POLICY_LLM_MAX_CONTEXT_TOKENS=1024,
        POLICY_LLM_MAX_OUTPUT_TOKENS=16000,
    )
    service = PolicyService(
        llm_service=llm,
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        settings=settings,
    )
    long_text = "Example Procedure coverage. " + ("detail " * 800)
    candidates = [
        _candidate(_chunk("rank-1", long_text), rank=1),
        _candidate(_chunk("rank-2", long_text), rank=2),
        _candidate(_chunk("rank-3", long_text), rank=3),
    ]
    llm.extract_structured.return_value = _no_match_result("No match.")

    await service._analyze_candidates(_procedure(), candidates)

    user_prompt = llm.extract_structured.await_args.kwargs["user_prompt"]
    assert "POLICY CONTEXT P1" in user_prompt
    assert "POLICY CONTEXT P2" not in user_prompt
    assert "POLICY CONTEXT P3" not in user_prompt
    assert "chunk_id:" not in user_prompt


@pytest.mark.asyncio
async def test_and_or_and_nested_trees_validate(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    text = (
        "Require documented reflux. Require CEAP class C2 or greater. "
        "Either failed compression for three months or prior treatment failure "
        "with documented recurrent symptoms."
    )
    chunk = _chunk("c1", text)
    root = SolCriterionNode(
        id="root",
        node_type="AND",
        children=[
            _sol_condition(
                "c1",
                "Documented reflux.",
                source_quote="documented reflux",
            ),
            _sol_condition(
                "c2",
                "CEAP class C2 or greater.",
                source_quote="CEAP class C2 or greater",
            ),
            SolCriterionNode(
                id="or-1",
                node_type="OR",
                children=[
                    _sol_condition(
                        "c3",
                        "Failed compression for three months.",
                        source_quote="failed compression for three months",
                    ),
                    SolCriterionNode(
                        id="and-1",
                        node_type="AND",
                        children=[
                            _sol_condition(
                                "c4",
                                "Prior treatment failure.",
                                source_quote="prior treatment failure",
                            ),
                            _sol_condition(
                                "c5",
                                "Documented recurrent symptoms.",
                                source_quote="documented recurrent symptoms",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    llm.extract_structured.return_value = _match_result(
        root,
        explanation="Matching procedure domain.",
        policy_summary="Nested criteria example.",
    )

    result = await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    assert result.criteria is not None
    assert result.criteria.root.node_type == CriterionNodeType.AND
    assert result.criteria.root.children[2].node_type == CriterionNodeType.OR


@pytest.mark.asyncio
async def test_unsupported_source_text_is_rejected(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Actual policy text about covered services.")
    llm.extract_structured.return_value = _match_result(
        _sol_condition(
            "c1",
            "Invented requirement.",
            source_quote="this source text does not exist",
        ),
    )

    with pytest.raises(PolicyValidationError, match="source_text"):
        await service._analyze_candidates(_procedure(), [_candidate(chunk)])


@pytest.mark.asyncio
async def test_invalid_context_aliases_are_rejected(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Example Procedure coverage criteria.")
    llm.extract_structured.return_value = _match_result(
        _sol_condition(
            "c1",
            "Coverage criteria.",
            source_context_id="P99",
            source_quote="coverage criteria",
        ),
        source_context_ids=["P99"],
    )

    with pytest.raises(PolicyValidationError, match="unknown source_context_id"):
        await service._analyze_candidates(_procedure(), [_candidate(chunk)])


def test_invalid_page_numbers_are_rejected_on_public_tree(
    service: PolicyService,
) -> None:
    chunk = _chunk("c1", "Require documented indication for coverage.", page=2)
    criteria = PolicyCriteriaExtraction(
        procedure_name="Example Procedure",
        policy_summary="Summary",
        root=_public_condition(
            "c1",
            "Documented indication.",
            chunk_id="c1",
            source_text="documented indication",
            page=9,
        ),
        source_chunk_ids=["c1"],
    )

    with pytest.raises(PolicyValidationError, match="page_number"):
        service._validate_criteria(criteria, {"c1": chunk})


@pytest.mark.asyncio
async def test_duplicate_node_ids_rejected(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Require A. Require B.")
    llm.extract_structured.return_value = _match_result(
        SolCriterionNode(
            id="root",
            node_type="AND",
            children=[
                _sol_condition(
                    "dup",
                    "Require A.",
                    source_quote="Require A",
                ),
                _sol_condition(
                    "dup",
                    "Require B.",
                    source_quote="Require B",
                ),
            ],
        ),
    )

    with pytest.raises(PolicyValidationError, match="Duplicate"):
        await service._analyze_candidates(_procedure(), [_candidate(chunk)])


def test_condition_with_children_rejected_on_public_tree(
    service: PolicyService,
) -> None:
    chunk = _chunk("c1", "Require documented indication.")
    bad = CriterionNode(
        id="bad",
        node_type=CriterionNodeType.CONDITION,
        description="Require documented indication.",
        category=CriterionCategory.CLINICAL,
        children=[
            _public_condition(
                "child",
                "Require documented indication.",
                chunk_id="c1",
                source_text="documented indication",
            )
        ],
        source_chunk_id="c1",
        source_text="documented indication",
        page_number=1,
    )
    criteria = PolicyCriteriaExtraction(
        procedure_name="Example Procedure",
        policy_summary="Summary",
        root=bad,
        source_chunk_ids=["c1"],
    )

    with pytest.raises(PolicyValidationError, match="must not have children"):
        service._validate_criteria(criteria, {"c1": chunk})


@pytest.mark.asyncio
async def test_and_without_children_coerced_to_non_matching(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Policy text.")
    llm.extract_structured.return_value = _match_result(
        SolCriterionNode(
            id="root",
            node_type="AND",
            children=[],
        ),
    )

    result = await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    assert result.applicable is False
    assert result.criteria is None
    assert result.llm_call_count == 1


def test_missing_source_evidence_rejected_on_public_tree(
    service: PolicyService,
) -> None:
    chunk = _chunk("c1", "Require documented indication.")
    criteria = PolicyCriteriaExtraction(
        procedure_name="Example Procedure",
        policy_summary="Summary",
        root=CriterionNode(
            id="c1",
            node_type=CriterionNodeType.CONDITION,
            description="Require documented indication.",
            category=CriterionCategory.CLINICAL,
            source_chunk_id=None,
            source_text=None,
            page_number=None,
        ),
        source_chunk_ids=["c1"],
    )

    with pytest.raises(PolicyValidationError, match="missing source grounding"):
        service._validate_criteria(criteria, {"c1": chunk})


@pytest.mark.asyncio
async def test_exclusions_are_preserved_separately(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk(
        "c1",
        "Example Procedure is covered with documented indication. "
        "Exclusions: cosmetic treatment is not covered.",
    )
    llm.extract_structured.return_value = SolPolicyCriteriaResult(
        matching_policy_criteria_found=True,
        explanation="Match",
        criteria_tree=SolPolicyCriteriaExtraction(
            procedure_name="Example Procedure",
            policy_summary="Covered with exclusions.",
            root=_sol_condition(
                "c1",
                "Documented indication.",
                source_quote="documented indication",
            ),
            exclusions=[
                SolPolicyExclusion(
                    id="ex1",
                    description="Cosmetic treatment is not covered.",
                    source_context_id="P1",
                    source_quote="cosmetic treatment is not covered",
                )
            ],
            source_context_ids=["P1"],
        ),
    )

    result = await service._analyze_candidates(_procedure(), [_candidate(chunk)])

    assert result.criteria is not None
    assert len(result.criteria.exclusions) == 1
    assert result.criteria.exclusions[0].id == "ex1"
    assert result.criteria.exclusions[0].source_chunk_id == "c1"
    assert result.criteria.root.node_type == CriterionNodeType.CONDITION


@pytest.mark.asyncio
async def test_matching_result_requires_explanation_or_policy_summary(
    service: PolicyService,
    llm: MagicMock,
) -> None:
    chunk = _chunk("c1", "Example Procedure coverage.")
    llm.extract_structured.return_value = SolPolicyCriteriaResult(
        matching_policy_criteria_found=True,
        explanation="",
        criteria_tree=SolPolicyCriteriaExtraction(
            procedure_name="Example Procedure",
            policy_summary="",
            root=_sol_condition(
                "c1",
                "Coverage criteria.",
                source_quote="Example Procedure coverage",
            ),
            source_context_ids=["P1"],
        ),
    )

    with pytest.raises(PolicyValidationError, match="explanation"):
        await service._analyze_candidates(_procedure(), [_candidate(chunk)])

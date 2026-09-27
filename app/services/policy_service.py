"""Policy governing-criteria discovery and structured criteria extraction."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.core.debug_trace import is_enabled as debug_trace_enabled
from app.core.debug_trace import write_json as write_debug_trace
from app.core.exceptions import PolicyAnalysisError, PolicyValidationError
from app.core.logging import get_logger
from app.core.metrics import get_request_metrics, timed_substage
from app.models.chunk import Chunk
from app.models.policy_analysis import PolicyAnalysisResult
from app.models.retrieval import RetrievalResult
from app.models.section import Section
from app.schemas.policy import (
    CriterionNode,
    CriterionNodeType,
    PolicyCriteriaExtraction,
    PolicyCriteriaResult,
    PolicyExclusion,
    SolCriterionNode,
    SolPolicyCriteriaResult,
    SolPolicyExclusion,
)
from app.schemas.procedure import ProcedureExtraction
from app.services.chunking_service import ChunkingService
from app.services.llm_service import LLMService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService
from app.utils.llm_context import dedupe_by_id_and_text, select_items_within_budget
from app.utils.text import normalize_text, text_supported
from app.utils.tokens import count_tokens

logger = get_logger(__name__)

_NO_CANDIDATES_REASON = (
    "No governing policy criteria candidates were found for the detected procedure."
)

# Conditional prerequisite leaves that must not become unconditional AND criteria.
_CONDITIONAL_PREREQ_START_RE = re.compile(r"^\s*(?:if|when)\b", re.IGNORECASE)
_CONDITIONAL_PREREQ_MARKERS_RE = re.compile(
    r"\b("
    r"already|previously|prior|same\s+or\s+similar|"
    r"documentation|documented|document\b|"
    r"not\s+supported\s+without|without\s+(?:clear\s+)?documentation"
    r")\b",
    re.IGNORECASE,
)
_CONDITIONAL_REQUESTS_RE = re.compile(
    r"\brequests?\b.+\bwhen\b.+\b(?:not\s+supported|without)\b",
    re.IGNORECASE | re.DOTALL,
)
_CONDITIONAL_IF_WHEN_SPLIT_RE = re.compile(
    r"^\s*(?:If|When)\s+(.+?)[,;:]\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

_POLICY_SYSTEM_PROMPT = """\
You are performing structured policy extraction, not open-ended analysis.

Use only the supplied policy excerpts labeled P1, P2, P3, …

1. Identify whether the supplied excerpts contain affirmative \
medical-necessity criteria governing the detected procedure and/or CPT.
2. If no governing affirmative criteria are present: \
matching_policy_criteria_found=false, criteria_tree=null.
3. If governing criteria are present: matching_policy_criteria_found=true; \
extract the complete applicable criteria tree.
4. Preserve explicit AND/OR relationships.
5. Make CONDITION leaves atomic.
6. Keep non-indications/exclusions separate.
7. Conditional policy rules of the form "If/When X, Y is required" or \
"requests when X are not supported without Y" are NOT unconditional AND \
prerequisites. Represent them as exclusions/blocking conditions that trigger \
only when antecedent X applies AND required supporting condition Y is absent. \
If X is not established, the blocking condition is not triggered.
8. Ground CONDITION leaves and exclusions with source_context_id (P-alias) \
and a short exact source_quote (<=200 characters) from that excerpt.
9. Do NOT output chunk UUIDs, page numbers, or full excerpt reproductions.
10. AND/OR group nodes need only id, node_type, optional short description, \
and children - never source grounding fields.
11. Extract explicit temporal requirements only when stated.
12. Do not invent requirements.
13. Do not compare payer identity or policy publisher.
14. Do not broadly analyze unrelated procedures.
15. Do not explain internal reasoning.
16. Return only the structured schema.
"""


@dataclass(frozen=True, slots=True)
class _PolicyTraceMeta:
    """Request-local metadata for DEBUG_TRACE dumps only."""

    procedure_name: str | None
    cpt_code: str
    cpt_section_ids: frozenset[str]
    bm25_score_by_section: dict[str, float]
    governing_section_ids: frozenset[str]
    reranker_rank_by_section: dict[str, int]
    reranker_score_by_section: dict[str, float]


@dataclass(frozen=True, slots=True)
class _PolicyContextAlias:
    """Deterministic Sol alias → source chunk metadata."""

    alias: str
    chunk: Chunk


@dataclass(frozen=True, slots=True)
class _PreparedPolicyContext:
    """Budgeted policy candidates with compact Sol aliases."""

    candidates: list[RetrievalResult]
    aliases: dict[str, _PolicyContextAlias]


class PolicyService:
    """CPT+BM25 candidate recall, dense+rerank governing selection, then Sol."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
        chunking_service: ChunkingService,
        reranking_service: RerankingService,
        settings: Settings | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._retrieval_service = retrieval_service
        self._chunking_service = chunking_service
        self._reranking_service = reranking_service
        self._settings = settings or get_settings()

    async def analyze_policy(
        self,
        procedure: ProcedureExtraction,
        policy_sections: list[Section],
    ) -> PolicyAnalysisResult:
        """Retrieve governing policy context, then run one Sol criteria call."""
        self._validate_procedure(procedure)
        logger.info(
            "policy_sections_built",
            sections=len(policy_sections),
        )

        candidates, trace_meta = self._retrieve_policy_candidates_impl(
            procedure, policy_sections
        )
        if not candidates:
            return PolicyAnalysisResult(
                applicable=False,
                applicability_reason=_NO_CANDIDATES_REASON,
                applicability_confidence=0.0,
                criteria=None,
                supporting_chunk_ids=[],
                llm_call_count=0,
            )

        return await self._analyze_candidates(
            procedure, candidates, trace_meta=trace_meta
        )

    def retrieve_policy_candidates(
        self,
        procedure: ProcedureExtraction,
        policy_sections: list[Section],
    ) -> list[RetrievalResult]:
        """CPT+BM25 → union → dense → rerank → expand → chunks."""
        candidates, _trace_meta = self._retrieve_policy_candidates_impl(
            procedure, policy_sections
        )
        return candidates

    def _retrieve_policy_candidates_impl(
        self,
        procedure: ProcedureExtraction,
        policy_sections: list[Section],
    ) -> tuple[list[RetrievalResult], _PolicyTraceMeta | None]:
        """Same retrieval as production; optional debug meta when tracing."""
        cpt = (procedure.cpt_code or "").strip()
        with timed_substage("policy_exact_cpt_search_ms"):
            cpt_hits = (
                self._retrieval_service.find_sections_with_cpt(policy_sections, cpt)
                if cpt
                else []
            )
        cpt_section_ids = frozenset(section.id for section in cpt_hits)

        bm25_query = self._retrieval_service.policy_bm25_query(procedure)
        bm25_candidates = self._retrieval_service.bm25_select_sections(
            policy_sections,
            query=bm25_query,
            top_k=self._settings.POLICY_BM25_TOP_K,
            profile_prefix="policy",
        )
        logger.info(
            "policy_bm25_retrieval_completed",
            corpus_sections=len(policy_sections),
            candidates=len(bm25_candidates),
        )

        # Debug-only BM25 score map; does not alter BM25 selection.
        bm25_score_by_section: dict[str, float] = {}
        if debug_trace_enabled():
            bm25_score_by_section = self._debug_bm25_scores(
                policy_sections, query=bm25_query
            )
            self._write_policy_candidates_trace(
                procedure=procedure,
                cpt_code=cpt,
                cpt_hits=cpt_hits,
                bm25_candidates=bm25_candidates,
                bm25_score_by_section=bm25_score_by_section,
                cpt_section_ids=cpt_section_ids,
            )

        with timed_substage("policy_candidate_union_ms"):
            union = self._retrieval_service.union_sections(cpt_hits, bm25_candidates)
        logger.info(
            "policy_candidate_union_completed",
            cpt_candidates=len(cpt_hits),
            bm25_candidates=len(bm25_candidates),
            unique_candidates=len(union),
        )
        if not union:
            return [], None

        dense_query = self._retrieval_service.policy_dense_query(procedure)
        dense_selected = self._retrieval_service.dense_rank_sections(
            union,
            query=dense_query,
            top_k=self._settings.POLICY_DENSE_TOP_K,
            profile_prefix="policy",
        )
        logger.info(
            "policy_dense_ranking_completed",
            input_candidates=len(union),
            output_candidates=len(dense_selected),
        )
        if not dense_selected:
            return [], None

        section_candidates = self._retrieval_service.sections_to_retrieval_results(
            dense_selected
        )
        rerank_query = self._retrieval_service.policy_rerank_query(procedure)
        governing = self._reranking_service.rerank(
            query=rerank_query,
            candidates=section_candidates,
            top_k=self._settings.POLICY_RERANK_TOP_K,
            stage="policy",
        )
        logger.info(
            "policy_rerank_completed",
            input_candidates=len(section_candidates),
            governing_candidates=len(governing),
        )
        if not governing:
            return [], None

        dense_by_id = {section.id: section for section in dense_selected}
        if debug_trace_enabled():
            self._write_policy_reranked_trace(
                governing=governing,
                dense_by_id=dense_by_id,
                cpt_section_ids=cpt_section_ids,
            )

        governing_ids = {item.chunk.section_id for item in governing}
        governing_sections = [
            section for section in dense_selected if section.id in governing_ids
        ]
        # Preserve cross-encoder order.
        order = {
            item.chunk.section_id: item.rank
            for item in governing
            if item.chunk.section_id
        }
        governing_sections.sort(key=lambda section: order.get(section.id, 10**9))

        # Drop unrelated procedure seeds (e.g. leadless) once the governing
        # procedure block for the detected CPT/procedure is established.
        with timed_substage("policy_seed_filter_ms"):
            governing_sections = (
                self._retrieval_service.filter_governing_seeds_for_procedure(
                    governing_sections,
                    policy_sections,
                    procedure=procedure,
                    cpt_section_ids=cpt_section_ids,
                )
            )
        governing_ids = {section.id for section in governing_sections}

        rejected_ids = {section.id for section in union} - governing_ids
        with timed_substage("policy_structural_expansion_ms"):
            expanded = self._retrieval_service.expand_section_context(
                governing_sections,
                policy_sections,
                exclude_ids=rejected_ids,
            )
        logger.info(
            "policy_context_expansion_completed",
            governing_sections=len(governing_sections),
            governing_headings=[
                (section.heading or "")[:80] for section in governing_sections
            ],
            expanded_sections=len(expanded),
        )

        with timed_substage("policy_chunk_build_ms"):
            chunks = self._chunking_service.chunk_sections(expanded)
            if not chunks:
                return [], None

            section_rank = {
                section.id: index for index, section in enumerate(governing_sections)
            }
            chunks.sort(
                key=lambda chunk: (
                    section_rank.get(chunk.section_id, 10**9),
                    chunk.chunk_index,
                )
            )
            results = [
                RetrievalResult(
                    chunk=chunk,
                    rank=index,
                    score=1.0,
                    retrieval_method="policy_governing",
                )
                for index, chunk in enumerate(chunks, start=1)
            ]

        trace_meta: _PolicyTraceMeta | None = None
        if debug_trace_enabled():
            reranker_rank_by_section = {
                item.chunk.section_id: item.rank
                for item in governing
                if item.chunk.section_id
            }
            reranker_score_by_section = {
                item.chunk.section_id: float(item.reranker_score or 0.0)
                for item in governing
                if item.chunk.section_id
            }
            trace_meta = _PolicyTraceMeta(
                procedure_name=procedure.procedure_name,
                cpt_code=cpt,
                cpt_section_ids=cpt_section_ids,
                bm25_score_by_section=bm25_score_by_section,
                governing_section_ids=frozenset(
                    sid for sid in governing_ids if sid is not None
                ),
                reranker_rank_by_section=reranker_rank_by_section,
                reranker_score_by_section=reranker_score_by_section,
            )
        return results, trace_meta

    async def _analyze_candidates(
        self,
        procedure: ProcedureExtraction,
        candidates: list[RetrievalResult],
        *,
        trace_meta: _PolicyTraceMeta | None = None,
    ) -> PolicyAnalysisResult:
        prepared = self._prepare_policy_context(
            procedure, candidates, trace_meta=trace_meta
        )
        chunk_by_id = {item.chunk.id: item.chunk for item in prepared.candidates}

        started = time.perf_counter()
        sol_result = await self._extract_policy_criteria(procedure, prepared)
        duration_ms = int((time.perf_counter() - started) * 1000)

        with timed_substage("policy_grounding_hydration_ms"):
            result = self._hydrate_policy_result(sol_result, prepared.aliases)
        norm_started = time.perf_counter()
        self._validate_policy_result(result, chunk_by_id)
        result = self._normalize_empty_match(result)

        if not result.matching_policy_criteria_found:
            metrics = get_request_metrics()
            if metrics is not None:
                metrics.add_substage_ms(
                    "policy_sol_normalization_ms",
                    int((time.perf_counter() - norm_started) * 1000),
                )
            logger.info(
                "policy_criteria_completed",
                matching_policy_criteria_found=False,
                criteria_duration_ms=duration_ms,
                context_items=len(prepared.candidates),
            )
            return PolicyAnalysisResult(
                applicable=False,
                applicability_reason=(result.explanation or "").strip()
                or "Policy context does not govern the detected procedure.",
                applicability_confidence=0.0,
                criteria=None,
                supporting_chunk_ids=[],
                llm_call_count=1,
            )

        assert result.criteria_tree is not None
        validated = self._validate_criteria(result.criteria_tree, chunk_by_id)
        validated = self._relocate_conditional_prerequisites(validated)
        counts = self._count_nodes(validated.root)
        metrics = get_request_metrics()
        if metrics is not None:
            metrics.add_substage_ms(
                "policy_sol_normalization_ms",
                int((time.perf_counter() - norm_started) * 1000),
            )
            metrics.record_policy_tree_counts(
                group_nodes=counts["AND"] + counts["OR"],
                condition_nodes=counts["CONDITION"],
                exclusions=len(validated.exclusions),
            )
        supporting = list(dict.fromkeys(validated.source_chunk_ids))
        write_debug_trace(
            "criteria_normalized.json",
            validated.model_dump(),
        )
        logger.info(
            "policy_criteria_completed",
            matching_policy_criteria_found=True,
            criteria_duration_ms=duration_ms,
            condition_count=counts["CONDITION"],
            and_count=counts["AND"],
            or_count=counts["OR"],
            exclusion_count=len(validated.exclusions),
            context_items=len(prepared.candidates),
        )

        return PolicyAnalysisResult(
            applicable=True,
            applicability_reason=(result.explanation or "").strip()
            or validated.policy_summary,
            applicability_confidence=1.0,
            criteria=validated,
            supporting_chunk_ids=supporting,
            llm_call_count=1,
        )

    def _validate_procedure(self, procedure: ProcedureExtraction) -> None:
        if not (procedure.procedure_name or "").strip():
            raise PolicyAnalysisError(
                "Procedure name is required for policy analysis"
            )

    def _prepare_policy_context(
        self,
        procedure: ProcedureExtraction,
        candidates: Sequence[RetrievalResult],
        *,
        trace_meta: _PolicyTraceMeta | None = None,
    ) -> _PreparedPolicyContext:
        with timed_substage("policy_context_dedupe_ms"):
            deduped = dedupe_by_id_and_text(
                list(candidates),
                get_id=lambda item: item.chunk.id,
                get_text=lambda item: item.chunk.text,
            )
        # Input packing budget is independent of the completion ceiling.
        max_input = self._settings.POLICY_LLM_MAX_CONTEXT_TOKENS
        with timed_substage("policy_context_render_ms"):
            prefix = self._build_prompt_prefix(procedure)
            fixed_prefix = f"{_POLICY_SYSTEM_PROMPT}\n{prefix}"
            pre_rendered = "\n\n".join(
                self._render_candidate_block(item, alias="P0") for item in deduped
            )
        with timed_substage("policy_context_token_count_ms"):
            pre_budget_tokens = count_tokens(f"{fixed_prefix}\n{pre_rendered}")
        with timed_substage("policy_context_render_ms"):
            budgeted = select_items_within_budget(
                deduped,
                render_item=lambda item: self._render_candidate_block(
                    item, alias="P0"
                ),
                fixed_prefix=fixed_prefix,
                max_input_tokens=max_input,
            )
            if not budgeted.items and deduped:
                # Always keep the highest-ranked governing chunk.
                budgeted_items = [deduped[0]]
                truncated_item = budgeted.truncated_item
            else:
                budgeted_items = budgeted.items
                truncated_item = budgeted.truncated_item

            aliases = {
                f"P{index}": _PolicyContextAlias(
                    alias=f"P{index}",
                    chunk=item.chunk,
                )
                for index, item in enumerate(budgeted_items, start=1)
            }
            prepared = _PreparedPolicyContext(
                candidates=budgeted_items,
                aliases=aliases,
            )
            user_prompt = self._build_user_prompt(procedure, prepared)
        with timed_substage("policy_context_token_count_ms"):
            post_budget_tokens = count_tokens(
                f"{_POLICY_SYSTEM_PROMPT}\n{user_prompt}"
            )
        dropped_count = max(0, len(deduped) - len(budgeted_items))
        logger.info(
            "policy_context_budgeted",
            pre_budget_tokens=pre_budget_tokens,
            post_budget_tokens=post_budget_tokens,
            context_item_count=len(budgeted_items),
            dropped_context_item_count=dropped_count,
            truncated_item=truncated_item,
            input_budget_tokens=max_input,
            output_budget_tokens=self._settings.POLICY_LLM_MAX_OUTPUT_TOKENS,
        )

        if debug_trace_enabled():
            system_prompt = _POLICY_SYSTEM_PROMPT
            governing_ids = (
                trace_meta.governing_section_ids
                if trace_meta is not None
                else frozenset()
            )
            cpt_ids = (
                trace_meta.cpt_section_ids if trace_meta is not None else frozenset()
            )
            rerank_ranks = (
                trace_meta.reranker_rank_by_section if trace_meta is not None else {}
            )
            items: list[dict[str, object]] = []
            for order, candidate in enumerate(budgeted_items, start=1):
                chunk = candidate.chunk
                section_id = chunk.section_id
                alias = f"P{order}"
                items.append(
                    {
                        "order": order,
                        "context_id": alias,
                        "chunk_id": chunk.id,
                        "section_id": section_id,
                        "heading": chunk.heading,
                        "parent_heading": chunk.parent_heading,
                        "page_number": chunk.page_number,
                        "origin": (
                            "governing" if section_id in governing_ids else "expanded"
                        ),
                        "reranker_rank": rerank_ranks.get(section_id),
                        "exact_cpt_match": section_id in cpt_ids,
                        # Exact chunk body embedded in the Sol user prompt.
                        "text": chunk.text,
                    }
                )
            write_debug_trace(
                "policy_context_sent_to_sol.json",
                {
                    "procedure_name": procedure.procedure_name,
                    "cpt_code": procedure.cpt_code,
                    "context_item_count": len(budgeted_items),
                    "dropped_context_item_count": dropped_count,
                    "pre_budget_tokens": pre_budget_tokens,
                    "post_budget_tokens": post_budget_tokens,
                    "rendered_input_token_count": post_budget_tokens,
                    "configured_context_budget": (
                        self._settings.POLICY_LLM_MAX_CONTEXT_TOKENS
                    ),
                    "output_token_limit": self._settings.POLICY_LLM_MAX_OUTPUT_TOKENS,
                    "available_input_tokens": budgeted.available_tokens,
                    "truncated_item": truncated_item,
                    "items": items,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
            )
        return prepared

    async def _extract_policy_criteria(
        self,
        procedure: ProcedureExtraction,
        prepared: _PreparedPolicyContext,
    ) -> SolPolicyCriteriaResult:
        user_prompt = self._build_user_prompt(procedure, prepared)
        try:
            result = await self._llm_service.extract_structured(
                system_prompt=_POLICY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=SolPolicyCriteriaResult,
                model=self._settings.POLICY_REASONING_MODEL,
                operation="policy_criteria",
                max_output_tokens=self._settings.POLICY_LLM_MAX_OUTPUT_TOKENS,
                reasoning_effort="low",
                context_items=len(prepared.candidates),
            )
        except Exception as exc:
            raise PolicyAnalysisError(
                "Policy criteria extraction request failed"
            ) from exc

        write_debug_trace(
            "policy_sol_raw_response.json",
            result.model_dump(),
        )
        return result

    def _debug_bm25_scores(
        self,
        sections: Sequence[Section],
        *,
        query: str,
    ) -> dict[str, float]:
        """Score all sections for debug dumps; unused by production ranking."""
        index = self._retrieval_service.build_section_bm25_index(sections)
        if not index.sections or index.bm25 is None:
            return {}
        query_tokens = self._retrieval_service.tokenize(query)
        if not query_tokens:
            return {}
        scores = index.bm25.get_scores(query_tokens)
        return {
            section.id: float(scores[index_pos])
            for index_pos, section in enumerate(index.sections)
        }

    def _section_trace_entry(
        self,
        section: Section,
        *,
        cpt_section_ids: frozenset[str],
        bm25_score_by_section: dict[str, float],
    ) -> dict[str, object]:
        return {
            "section_id": section.id,
            "heading": section.heading,
            "parent_heading": section.parent_heading,
            "page_number": section.start_page,
            "exact_cpt_match": section.id in cpt_section_ids,
            "bm25_score": bm25_score_by_section.get(section.id),
            "text": section.text,
        }

    def _write_policy_candidates_trace(
        self,
        *,
        procedure: ProcedureExtraction,
        cpt_code: str,
        cpt_hits: Sequence[Section],
        bm25_candidates: Sequence[Section],
        bm25_score_by_section: dict[str, float],
        cpt_section_ids: frozenset[str],
    ) -> None:
        write_debug_trace(
            "policy_candidates.json",
            {
                "procedure_name": procedure.procedure_name,
                "cpt_code": cpt_code,
                "cpt_hit_count": len(cpt_hits),
                "bm25_candidate_count": len(bm25_candidates),
                "cpt_hits": [
                    self._section_trace_entry(
                        section,
                        cpt_section_ids=cpt_section_ids,
                        bm25_score_by_section=bm25_score_by_section,
                    )
                    for section in cpt_hits
                ],
                "bm25_candidates": [
                    self._section_trace_entry(
                        section,
                        cpt_section_ids=cpt_section_ids,
                        bm25_score_by_section=bm25_score_by_section,
                    )
                    for section in bm25_candidates
                ],
            },
        )

    def _write_policy_reranked_trace(
        self,
        *,
        governing: Sequence[RetrievalResult],
        dense_by_id: dict[str, Section],
        cpt_section_ids: frozenset[str],
    ) -> None:
        entries: list[dict[str, object]] = []
        for item in governing:
            section_id = item.chunk.section_id
            section = dense_by_id.get(section_id or "")
            text = section.text if section is not None else item.chunk.text
            heading = (
                section.heading if section is not None else item.chunk.heading
            )
            parent_heading = (
                section.parent_heading
                if section is not None
                else item.chunk.parent_heading
            )
            page_number = (
                section.start_page if section is not None else item.chunk.page_number
            )
            entries.append(
                {
                    "rank": item.rank,
                    "section_id": section_id,
                    "heading": heading,
                    "parent_heading": parent_heading,
                    "page_number": page_number,
                    "exact_cpt_match": (section_id or "") in cpt_section_ids,
                    "reranker_score": item.reranker_score,
                    "text": text,
                }
            )
        write_debug_trace("policy_reranked.json", entries)

    def _build_prompt_prefix(self, procedure: ProcedureExtraction) -> str:
        return "\n".join(
            [
                f"procedure_name: {procedure.procedure_name}",
                f"cpt_code: {procedure.cpt_code}",
                "",
                "Extract the governing medical-necessity criteria from the "
                "supplied policy context. Cite excerpts only via P-aliases "
                "(P1, P2, …) and short source_quote text.",
                "",
            ]
        )

    def _build_user_prompt(
        self,
        procedure: ProcedureExtraction,
        prepared: _PreparedPolicyContext,
    ) -> str:
        blocks: list[str] = [self._build_prompt_prefix(procedure)]
        for index, candidate in enumerate(prepared.candidates, start=1):
            alias = f"P{index}"
            blocks.append(self._render_candidate_block(candidate, alias=alias))
            blocks.append("")
        return "\n".join(blocks).rstrip() + "\n"

    def _render_candidate_block(
        self,
        candidate: RetrievalResult,
        *,
        alias: str,
        index: int | None = None,
    ) -> str:
        del index  # Alias is the public Sol label; ordinal is unused.
        chunk = candidate.chunk
        return "\n".join(
            [
                f"POLICY CONTEXT {alias}",
                f"heading: {chunk.heading}",
                f"parent_heading: {chunk.parent_heading}",
                "text:",
                chunk.text,
            ]
        )

    def _hydrate_policy_result(
        self,
        sol_result: SolPolicyCriteriaResult,
        aliases: dict[str, _PolicyContextAlias],
    ) -> PolicyCriteriaResult:
        if not sol_result.matching_policy_criteria_found:
            return PolicyCriteriaResult(
                matching_policy_criteria_found=False,
                explanation=sol_result.explanation,
                criteria_tree=None,
            )
        assert sol_result.criteria_tree is not None
        tree = sol_result.criteria_tree
        hydrated_root = self._hydrate_sol_node(tree.root, aliases)
        hydrated_exclusions = [
            self._hydrate_sol_exclusion(item, aliases) for item in tree.exclusions
        ]
        source_chunk_ids: list[str] = []
        for alias in tree.source_context_ids:
            mapped = aliases.get(alias)
            if mapped is not None:
                source_chunk_ids.append(mapped.chunk.id)
        # Also collect from leaves when the envelope list is incomplete.
        self._collect_chunk_ids(hydrated_root, source_chunk_ids)
        for exclusion in hydrated_exclusions:
            source_chunk_ids.append(exclusion.source_chunk_id)
        return PolicyCriteriaResult(
            matching_policy_criteria_found=True,
            explanation=sol_result.explanation,
            criteria_tree=PolicyCriteriaExtraction(
                procedure_name=tree.procedure_name,
                policy_summary=tree.policy_summary,
                root=hydrated_root,
                exclusions=hydrated_exclusions,
                source_chunk_ids=list(dict.fromkeys(source_chunk_ids)),
            ),
        )

    def _hydrate_sol_node(
        self,
        node: SolCriterionNode,
        aliases: dict[str, _PolicyContextAlias],
    ) -> CriterionNode:
        if node.node_type == CriterionNodeType.CONDITION:
            mapped = aliases.get(node.source_context_id or "")
            if mapped is None:
                raise PolicyValidationError(
                    f"CONDITION {node.id} cites unknown source_context_id: "
                    f"{node.source_context_id}"
                )
            return CriterionNode(
                id=node.id,
                node_type=CriterionNodeType.CONDITION,
                description=node.description,
                category=node.category,
                temporal_requirement=node.temporal_requirement,
                children=[],
                source_chunk_id=mapped.chunk.id,
                source_text=node.source_quote,
                page_number=mapped.chunk.page_number,
            )

        return CriterionNode(
            id=node.id,
            node_type=node.node_type,
            description=node.description,
            children=[
                self._hydrate_sol_node(child, aliases) for child in node.children
            ],
        )

    def _hydrate_sol_exclusion(
        self,
        exclusion: SolPolicyExclusion,
        aliases: dict[str, _PolicyContextAlias],
    ) -> PolicyExclusion:
        mapped = aliases.get(exclusion.source_context_id)
        if mapped is None:
            raise PolicyValidationError(
                f"Exclusion {exclusion.id} cites unknown source_context_id: "
                f"{exclusion.source_context_id}"
            )
        return PolicyExclusion(
            id=exclusion.id,
            description=exclusion.description,
            source_chunk_id=mapped.chunk.id,
            source_text=exclusion.source_quote,
            page_number=mapped.chunk.page_number,
        )

    def _collect_chunk_ids(
        self,
        node: CriterionNode,
        sink: list[str],
    ) -> None:
        if node.node_type == CriterionNodeType.CONDITION and node.source_chunk_id:
            sink.append(node.source_chunk_id)
        for child in node.children:
            self._collect_chunk_ids(child, sink)

    def _normalize_empty_match(
        self,
        result: PolicyCriteriaResult,
    ) -> PolicyCriteriaResult:
        """Treat matching=true with zero CONDITION leaves as non-matching."""
        if not result.matching_policy_criteria_found or result.criteria_tree is None:
            return result
        counts = self._count_nodes(result.criteria_tree.root)
        if counts["CONDITION"] > 0:
            return result
        explanation = (result.explanation or "").strip() or (
            result.criteria_tree.policy_summary or ""
        ).strip() or (
            "Policy excerpts did not yield extractable governing criteria."
        )
        return PolicyCriteriaResult(
            matching_policy_criteria_found=False,
            explanation=explanation,
            criteria_tree=None,
        )

    def _relocate_conditional_prerequisites(
        self,
        criteria: PolicyCriteriaExtraction,
    ) -> PolicyCriteriaExtraction:
        """Move If/When-X-then-Y leaves out of AND prerequisites into exclusions.

        Absence of antecedent X must not make authorization NOT_READY; the
        blocking exclusion triggers only when X applies and Y is unsatisfied.
        """
        relocated: list[PolicyExclusion] = []
        seen_exclusion_ids = {item.id for item in criteria.exclusions}

        def walk(node: CriterionNode) -> CriterionNode | None:
            if node.node_type == CriterionNodeType.CONDITION:
                if self._is_conditional_prerequisite(node.description or ""):
                    exclusion = self._conditional_leaf_to_exclusion(
                        node,
                        seen_ids=seen_exclusion_ids,
                    )
                    relocated.append(exclusion)
                    seen_exclusion_ids.add(exclusion.id)
                    return None
                return node

            kept_children: list[CriterionNode] = []
            for child in node.children:
                rewritten = walk(child)
                if rewritten is not None:
                    kept_children.append(rewritten)
            if not kept_children:
                return None
            if len(kept_children) == 1 and node.node_type in {
                CriterionNodeType.AND,
                CriterionNodeType.OR,
            }:
                # Collapse unary groups created by relocating siblings.
                return kept_children[0]
            return node.model_copy(update={"children": kept_children})

        new_root = walk(criteria.root)
        if new_root is None:
            raise PolicyValidationError(
                "Conditional-prerequisite relocation removed all criteria"
            )
        return PolicyCriteriaExtraction(
            procedure_name=criteria.procedure_name,
            policy_summary=criteria.policy_summary,
            root=new_root,
            exclusions=[*criteria.exclusions, *relocated],
            source_chunk_ids=list(criteria.source_chunk_ids),
        )

    @staticmethod
    def _is_conditional_prerequisite(description: str) -> bool:
        text = (description or "").strip()
        if not text:
            return False
        if _CONDITIONAL_REQUESTS_RE.search(text):
            return True
        if not _CONDITIONAL_PREREQ_START_RE.match(text):
            return False
        return bool(_CONDITIONAL_PREREQ_MARKERS_RE.search(text))

    @staticmethod
    def _conditional_leaf_to_exclusion(
        node: CriterionNode,
        *,
        seen_ids: set[str],
    ) -> PolicyExclusion:
        description = (node.description or "").strip()
        match = _CONDITIONAL_IF_WHEN_SPLIT_RE.match(description)
        if match is not None:
            antecedent = match.group(1).strip().rstrip(".")
            requirement = match.group(2).strip().rstrip(".")
            blocking = (
                f"{antecedent} without establishing that {requirement}"
            )
        else:
            blocking = description

        exclusion_id = node.id
        if exclusion_id in seen_ids:
            exclusion_id = f"{node.id}-conditional"
        suffix = 2
        while exclusion_id in seen_ids:
            exclusion_id = f"{node.id}-conditional-{suffix}"
            suffix += 1

        if not node.source_chunk_id or not node.source_text or node.page_number is None:
            raise PolicyValidationError(
                f"Conditional leaf {node.id} missing source grounding for relocation"
            )
        return PolicyExclusion(
            id=exclusion_id,
            description=blocking,
            source_chunk_id=node.source_chunk_id,
            source_text=node.source_text,
            page_number=node.page_number,
        )

    def _validate_policy_result(
        self,
        result: PolicyCriteriaResult,
        chunk_by_id: dict[str, Chunk],
    ) -> None:
        if not result.matching_policy_criteria_found:
            if result.criteria_tree is not None:
                raise PolicyValidationError(
                    "Non-matching policy result must not include criteria_tree"
                )
            return

        if result.criteria_tree is None:
            raise PolicyValidationError(
                "Matching policy result requires criteria_tree"
            )
        if not (result.explanation or "").strip() and not (
            result.criteria_tree.policy_summary or ""
        ).strip():
            raise PolicyValidationError(
                "Matching policies must include a non-empty explanation "
                "or policy_summary"
            )
        for chunk_id in result.criteria_tree.source_chunk_ids:
            if chunk_id not in chunk_by_id:
                raise PolicyValidationError(
                    f"Criteria cited unknown source_chunk_id: {chunk_id}"
                )

    def _validate_criteria(
        self,
        criteria: PolicyCriteriaExtraction,
        chunk_by_id: dict[str, Chunk],
    ) -> PolicyCriteriaExtraction:
        if not criteria.procedure_name.strip():
            raise PolicyValidationError("Criteria extraction missing procedure_name")
        if not criteria.policy_summary.strip():
            raise PolicyValidationError("Criteria extraction missing policy_summary")

        seen_ids: set[str] = set()
        self._validate_node(
            criteria.root,
            chunk_by_id=chunk_by_id,
            seen_ids=seen_ids,
            ancestors=set(),
        )

        for chunk_id in criteria.source_chunk_ids:
            if chunk_id not in chunk_by_id:
                raise PolicyValidationError(
                    f"Criteria cited unknown source_chunk_id: {chunk_id}"
                )

        deduped_exclusions = self._dedupe_exclusions(
            criteria.exclusions,
            chunk_by_id=chunk_by_id,
            seen_ids=seen_ids,
        )

        return PolicyCriteriaExtraction(
            procedure_name=criteria.procedure_name,
            policy_summary=criteria.policy_summary,
            root=criteria.root,
            exclusions=deduped_exclusions,
            source_chunk_ids=list(dict.fromkeys(criteria.source_chunk_ids)),
        )

    def _validate_node(
        self,
        node: CriterionNode,
        *,
        chunk_by_id: dict[str, Chunk],
        seen_ids: set[str],
        ancestors: set[str],
    ) -> None:
        if node.id in ancestors:
            raise PolicyValidationError(f"Cycle detected at node {node.id}")
        if node.id in seen_ids:
            raise PolicyValidationError(f"Duplicate criterion node id: {node.id}")
        seen_ids.add(node.id)

        if node.node_type == CriterionNodeType.CONDITION:
            if node.children:
                raise PolicyValidationError(
                    f"CONDITION node {node.id} must not have children"
                )
            if not (node.description or "").strip():
                raise PolicyValidationError(
                    f"CONDITION node {node.id} requires a description"
                )
            self._validate_grounding(
                node_id=node.id,
                source_chunk_id=node.source_chunk_id,
                source_text=node.source_text,
                page_number=node.page_number,
                chunk_by_id=chunk_by_id,
            )
            return

        if not node.children:
            raise PolicyValidationError(
                f"{node.node_type.value} node {node.id} requires children"
            )

        next_ancestors = set(ancestors)
        next_ancestors.add(node.id)
        for child in node.children:
            self._validate_node(
                child,
                chunk_by_id=chunk_by_id,
                seen_ids=seen_ids,
                ancestors=next_ancestors,
            )

    def _dedupe_exclusions(
        self,
        exclusions: Sequence[PolicyExclusion],
        *,
        chunk_by_id: dict[str, Chunk],
        seen_ids: set[str],
    ) -> list[PolicyExclusion]:
        unique: list[PolicyExclusion] = []
        seen_descriptions: set[str] = set()
        for exclusion in exclusions:
            if exclusion.id in seen_ids:
                raise PolicyValidationError(
                    f"Duplicate exclusion/node id: {exclusion.id}"
                )
            seen_ids.add(exclusion.id)
            if not exclusion.description.strip():
                raise PolicyValidationError(
                    f"Exclusion {exclusion.id} requires a description"
                )
            self._validate_grounding(
                node_id=exclusion.id,
                source_chunk_id=exclusion.source_chunk_id,
                source_text=exclusion.source_text,
                page_number=exclusion.page_number,
                chunk_by_id=chunk_by_id,
            )
            key = normalize_text(exclusion.description)
            if key in seen_descriptions:
                continue
            seen_descriptions.add(key)
            unique.append(exclusion)
        return unique

    def _validate_grounding(
        self,
        *,
        node_id: str,
        source_chunk_id: str | None,
        source_text: str | None,
        page_number: int | None,
        chunk_by_id: dict[str, Chunk],
    ) -> None:
        if not source_chunk_id or not source_text or page_number is None:
            raise PolicyValidationError(
                f"Node {node_id} is missing source grounding fields"
            )
        chunk = chunk_by_id.get(source_chunk_id)
        if chunk is None:
            raise PolicyValidationError(
                f"Node {node_id} cites unknown chunk_id: {source_chunk_id}"
            )
        if not text_supported(source_text, chunk.text):
            raise PolicyValidationError(
                f"Node {node_id} source_text was not found in cited chunk"
            )
        if not (chunk.page_number <= page_number <= chunk.end_page_number):
            raise PolicyValidationError(
                f"Node {node_id} page_number is outside cited chunk range"
            )

    def _count_nodes(self, root: CriterionNode) -> dict[str, int]:
        counts = {"CONDITION": 0, "AND": 0, "OR": 0}

        def walk(node: CriterionNode) -> None:
            counts[node.node_type.value] += 1
            for child in node.children:
                walk(child)

        walk(root)
        return counts

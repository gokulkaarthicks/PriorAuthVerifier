"""Patient-chart evidence retrieval and criterion evaluation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.core.config import Settings, get_settings
from app.core.debug_trace import is_enabled as debug_trace_enabled
from app.core.debug_trace import write_json as write_debug_trace
from app.core.exceptions import (
    EvidenceAnalysisError,
    EvidenceGroundingError,
    EvidenceValidationError,
)
from app.core.logging import get_logger
from app.core.metrics import get_request_metrics, timed_substage
from app.models.chunk import Chunk
from app.models.evidence_analysis import EvidenceAnalysisResult
from app.models.policy_analysis import PolicyAnalysisResult
from app.models.retrieval import RetrievalResult
from app.models.section import Section
from app.schemas.evaluation import (
    CriterionEvaluation,
    CriterionStatus,
    EvidenceCitation,
    EvidenceEvaluationBatch,
    ExclusionEvaluation,
    SolEvidenceEvaluationBatch,
    citation_from_chunk,
    materialize_sol_batch,
)
from app.schemas.policy import (
    CriterionCategory,
    CriterionNode,
    CriterionNodeType,
    PolicyExclusion,
)
from app.services.chunking_service import ChunkingService
from app.services.llm_service import LLMService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService
from app.utils.dates import is_within_months, parse_within_months_window
from app.utils.text import normalize_text
from app.utils.tokens import count_tokens

logger = get_logger(__name__)

_EVIDENCE_SYSTEM_PROMPT = """\
You evaluate whether patient-chart evidence satisfies policy criteria.

Rules:
1. A shared evidence_corpus lists unique patient excerpts as E1, E2, … \
Each criterion/exclusion lists only its allowed_candidate_ids from that \
corpus. Cite only IDs from that item's allowed set. Never use another \
item's candidates. If none support the judgment, return an empty \
evidence_candidate_ids list.
2. Never use outside medical knowledge or invent undocumented facts.
3. Use LOCAL POLICY CONTEXT to interpret each atom. Parent definitions and \
source policy wording define the condition - do not require the chart to \
repeat exact diagnosis labels when the policy definition plus patient facts \
establish it.
4. MET requires at least one supporting evidence_candidate_id.
5. MISSING means supporting evidence is absent from that criterion's \
allowed candidates.
6. UNCLEAR means evidence is incomplete, conflicting, or ambiguous.
7. Evaluate each criterion independently.
8. Evaluate exclusions separately. For ordinary non-indications, \
triggered=true only when evidence shows the exclusion applies. For \
conditional blocking exclusions of the form "if/when X, Y is required" \
(including "X without establishing Y"), triggered=true only when evidence \
establishes antecedent X AND does not establish required Y. If X is not \
established, triggered=false.
9. Return only criterion_id/exclusion_id, status, evidence_candidate_ids, \
and one concise explanation sentence. Do NOT repeat criterion descriptions, \
evidence text, chunk UUIDs, page numbers, or document dates.
10. When a criterion includes a temporal_requirement, prefer candidates whose \
encounter/document date satisfies that timing. Use analysis_reference_date \
for phrases such as "current" or "within the previous N months". Unknown-date \
evidence is not assumed current.
"""


@dataclass(frozen=True, slots=True)
class AtomicCriterion:
    """A leaf CONDITION or exclusion description used for retrieval."""

    id: str
    description: str
    category: CriterionCategory | None
    kind: Literal["condition", "exclusion"]
    temporal_requirement: str | None = None
    local_policy_context: str | None = None


@dataclass(frozen=True, slots=True)
class ScopedEvidenceCandidate:
    """Criterion-scoped evidence entry exposed to Sol."""

    candidate_id: str
    item_id: str
    chunk: Chunk

    def to_citation(self) -> EvidenceCitation:
        return citation_from_chunk(self.chunk)


class EvidenceService:
    """Shared BM25 + deduped chunking + batched dense/rerank/Sol evidence path."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        chunking_service: ChunkingService,
        reranking_service: RerankingService,
        llm_service: LLMService,
        settings: Settings | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._chunking_service = chunking_service
        self._reranking_service = reranking_service
        self._llm_service = llm_service
        self._settings = settings or get_settings()

    async def evaluate_evidence(
        self,
        policy_analysis: PolicyAnalysisResult,
        patient_sections: list[Section],
        *,
        analysis_reference_date: date | None = None,
    ) -> EvidenceAnalysisResult:
        """Evaluate atomic criteria against narrowed patient sections."""
        if not policy_analysis.applicable or policy_analysis.criteria is None:
            reason = (
                policy_analysis.applicability_reason
                or "Policy is not applicable; evidence evaluation skipped."
            )
            logger.info(
                "evidence_evaluation_skipped",
                reason="policy_not_applicable",
            )
            return EvidenceAnalysisResult(
                criteria=[],
                exclusions=[],
                skipped=True,
                skip_reason=reason,
                llm_call_count=0,
            )

        if not patient_sections:
            raise EvidenceAnalysisError(
                "Patient sections must contain at least one section"
            )

        criteria_tree = policy_analysis.criteria
        conditions = self.extract_atomic_criteria(criteria_tree.root)
        metrics = get_request_metrics()
        if metrics is not None:
            metrics.criteria_count = len(conditions)
        exclusions = list(criteria_tree.exclusions)
        exclusion_atoms = [
            AtomicCriterion(
                id=item.id,
                description=item.description,
                category=CriterionCategory.EXCLUSION,
                kind="exclusion",
                local_policy_context=self._exclusion_local_context(
                    item,
                    root=criteria_tree.root,
                ),
            )
            for item in exclusions
        ]
        atoms = [*conditions, *exclusion_atoms]
        # Analysis reference date is the PA/planned-procedure date when known.
        # Never invent a runtime/current date as a clinical evidence date.
        reference_date = analysis_reference_date

        logger.info(
            "evidence_evaluation_started",
            criteria_count=len(conditions),
            exclusion_count=len(exclusions),
            patient_section_count=len(patient_sections),
            analysis_reference_date=(
                reference_date.isoformat() if reference_date else None
            ),
        )

        if not atoms:
            return EvidenceAnalysisResult(
                criteria=[],
                exclusions=[],
                skipped=False,
                llm_call_count=0,
            )

        mapping, candidate_by_id, retrieval_duration_ms, reranking_duration_ms = (
            self._retrieve_evidence_for_atoms(
                atoms,
                patient_sections,
                reference_date=reference_date,
            )
        )

        llm_started = time.perf_counter()
        local_context_by_id = {
            atom.id: atom.local_policy_context or ""
            for atom in atoms
        }
        usage_before = self._llm_usage_snapshot()
        batch, llm_call_count, batch_traces = await self._evaluate_with_llm(
            conditions=conditions,
            exclusions=exclusions,
            mapping=mapping,
            candidate_by_id=candidate_by_id,
            reference_date=reference_date,
            local_context_by_id=local_context_by_id,
        )
        llm_ms = int((time.perf_counter() - llm_started) * 1000)
        evidence_usage = self._llm_usage_delta(usage_before)

        metrics = get_request_metrics()
        temporal_before = (
            int(metrics.substage_ms.get("evidence_temporal_validation_ms", 0))
            if metrics is not None
            else 0
        )
        norm_started = time.perf_counter()
        validated_criteria, validated_exclusions = self._validate_batch(
            batch=batch,
            expected_condition_ids={item.id for item in conditions},
            expected_exclusion_ids={item.id for item in exclusions},
            candidate_by_id=candidate_by_id,
            mapping=mapping,
            reference_date=reference_date,
        )
        if metrics is not None:
            norm_wall = int((time.perf_counter() - norm_started) * 1000)
            temporal_delta = (
                int(metrics.substage_ms.get("evidence_temporal_validation_ms", 0))
                - temporal_before
            )
            metrics.add_substage_ms(
                "evidence_normalization_ms",
                max(0, norm_wall - temporal_delta),
            )

        if debug_trace_enabled():
            self._write_evidence_debug_trace(
                conditions=conditions,
                exclusions=exclusions,
                mapping=mapping,
                candidate_by_id=candidate_by_id,
                criteria=validated_criteria,
                exclusion_evals=validated_exclusions,
                reference_date=reference_date,
                batch_traces=batch_traces,
            )

        met = sum(
            1 for item in validated_criteria if item.status == CriterionStatus.MET
        )
        missing = sum(
            1
            for item in validated_criteria
            if item.status == CriterionStatus.MISSING
        )
        unclear = sum(
            1
            for item in validated_criteria
            if item.status == CriterionStatus.UNCLEAR
        )
        logger.info(
            "evidence_evaluation_completed",
            atom_count=len(atoms),
            criteria_count=len(validated_criteria),
            exclusion_count=len(validated_exclusions),
            met_count=met,
            missing_count=missing,
            unclear_count=unclear,
            batch_count=llm_call_count,
            candidate_reference_count=sum(
                len(chunk_ids) for chunk_ids in mapping.values()
            ),
            unique_evidence_chunk_count=len(candidate_by_id),
            input_tokens=evidence_usage.get("input_tokens", 0),
            output_tokens=evidence_usage.get("output_tokens", 0),
            reasoning_tokens=evidence_usage.get("reasoning_tokens", 0),
            retrieval_duration_ms=retrieval_duration_ms,
            reranking_duration_ms=reranking_duration_ms,
            llm_duration_ms=llm_ms,
            llm_call_count=llm_call_count,
        )

        return EvidenceAnalysisResult(
            criteria=validated_criteria,
            exclusions=validated_exclusions,
            skipped=False,
            llm_call_count=llm_call_count,
            retrieval_duration_ms=retrieval_duration_ms,
            reranking_duration_ms=reranking_duration_ms,
            llm_duration_ms=llm_ms,
        )

    def extract_atomic_criteria(self, root: CriterionNode) -> list[AtomicCriterion]:
        """Collect CONDITION leaves with minimal local policy context."""
        atoms: list[AtomicCriterion] = []

        def walk(
            node: CriterionNode,
            *,
            ancestors: list[CriterionNode],
            siblings: list[CriterionNode],
        ) -> None:
            if node.node_type == CriterionNodeType.CONDITION:
                description = (node.description or "").strip()
                if not description:
                    raise EvidenceAnalysisError(
                        f"CONDITION node {node.id} has an empty description"
                    )
                temporal = (node.temporal_requirement or "").strip() or None
                atoms.append(
                    AtomicCriterion(
                        id=node.id,
                        description=description,
                        category=node.category,
                        kind="condition",
                        temporal_requirement=temporal,
                        local_policy_context=self._build_local_policy_context(
                            node,
                            ancestors=ancestors,
                            siblings=siblings,
                        ),
                    )
                )
                return
            for child in node.children:
                walk(
                    child,
                    ancestors=[*ancestors, node],
                    siblings=list(node.children),
                )

        walk(root, ancestors=[], siblings=[root])
        return atoms

    def _build_local_policy_context(
        self,
        node: CriterionNode,
        *,
        ancestors: list[CriterionNode],
        siblings: list[CriterionNode],
    ) -> str:
        """Minimal parent/sibling/source context, not the whole criteria tree."""
        lines: list[str] = []
        parent = next(
            (
                item
                for item in reversed(ancestors)
                if (item.description or "").strip()
            ),
            None,
        )
        if parent is not None:
            lines.append(
                f"parent: {(parent.description or '').strip()} "
                f"[{parent.node_type.value}]"
            )
        # Immediate ancestor chain (descriptions only) for branch identity.
        branch = [
            (item.description or "").strip()
            for item in ancestors
            if (item.description or "").strip()
        ]
        if branch:
            lines.append("branch: " + " > ".join(branch))

        sibling_defs = [
            (item.description or "").strip()
            for item in siblings
            if item.id != node.id
            and item.node_type == CriterionNodeType.CONDITION
            and (item.description or "").strip()
        ]
        if sibling_defs:
            # Keep sibling context small; definitions needed to interpret OR/AND.
            preview = sibling_defs[:6]
            lines.append("sibling_conditions: " + "; ".join(preview))

        source = (node.source_text or "").strip()
        if source:
            lines.append(f"source_policy_text: {source}")
        return "\n".join(lines) if lines else ""

    def _exclusion_local_context(
        self,
        exclusion: PolicyExclusion,
        *,
        root: CriterionNode,
    ) -> str:
        lines = [f"source_policy_text: {exclusion.source_text.strip()}"]
        # Attach parent-condition definitions that the exclusion wording references.
        related: list[str] = []
        exclusion_lower = exclusion.description.lower()

        def walk(node: CriterionNode, ancestors: list[str]) -> None:
            description = (node.description or "").strip()
            if node.node_type == CriterionNodeType.CONDITION and description:
                tokens = [
                    token
                    for token in description.lower().split()
                    if len(token) > 4
                ]
                if any(token in exclusion_lower for token in tokens[:4]):
                    branch = " > ".join([*ancestors, description])
                    related.append(branch)
                return
            next_ancestors = (
                [*ancestors, description] if description else list(ancestors)
            )
            for child in node.children:
                walk(child, next_ancestors)

        walk(root, [])
        if related:
            lines.append("related_policy_conditions: " + "; ".join(related[:4]))
        return "\n".join(lines)

    def build_query(self, atom: AtomicCriterion) -> str:
        """Build the retrieval query, including explicit temporal wording."""
        description = atom.description.strip()
        temporal = (atom.temporal_requirement or "").strip()
        if temporal and temporal.lower() not in description.lower():
            return f"{description} {temporal}".strip()
        return description

    def _retrieve_evidence_for_atoms(
        self,
        atoms: list[AtomicCriterion],
        patient_sections: list[Section],
        *,
        reference_date: date | None,
    ) -> tuple[
        dict[str, list[str]],
        dict[str, Chunk],
        int,
        int,
    ]:
        """Shared BM25, deduped chunking, batched dense and rerank."""
        started = time.perf_counter()
        with timed_substage("evidence_bm25_index_build_ms"):
            bm25_index = self._retrieval_service.build_section_bm25_index(
                patient_sections
            )
        logger.info(
            "evidence_bm25_index_built",
            corpus_sections=len(bm25_index.sections),
        )

        section_by_id = {section.id: section for section in patient_sections}
        selected_section_ids: dict[str, list[str]] = {}
        for atom in atoms:
            with timed_substage("evidence_query_build_ms"):
                query = self.build_query(atom)
            with timed_substage("evidence_bm25_selection_ms"):
                selected = self._retrieval_service.query_section_bm25(
                    bm25_index,
                    query=query,
                    top_k=self._settings.EVIDENCE_SECTION_BM25_TOP_K,
                )
            selected_section_ids[atom.id] = [section.id for section in selected]
            logger.info(
                "criterion_section_bm25_completed",
                criterion_id=atom.id,
                corpus_sections=len(bm25_index.sections),
                selected_sections=len(selected),
            )

        union_ids: list[str] = []
        seen_sections: set[str] = set()
        for section_ids in selected_section_ids.values():
            for section_id in section_ids:
                if section_id in seen_sections:
                    continue
                seen_sections.add(section_id)
                union_ids.append(section_id)

        union_sections = [
            section_by_id[section_id]
            for section_id in union_ids
            if section_id in section_by_id
        ]
        chunks_by_section: dict[str, list[Chunk]] = {
            section.id: [] for section in union_sections
        }
        with timed_substage("evidence_section_chunk_build_ms"):
            if union_sections:
                created = self._chunking_service.chunk_sections(union_sections)
                for chunk in created:
                    chunks_by_section.setdefault(chunk.section_id, []).append(chunk)

        logger.info(
            "evidence_unique_sections_chunked",
            unique_sections=len(union_sections),
            unique_chunks=sum(len(items) for items in chunks_by_section.values()),
        )

        allowed_chunks: dict[str, list[Chunk]] = {}
        dense_groups: dict[str, tuple[str, list[Chunk]]] = {}
        for atom in atoms:
            section_ids = selected_section_ids.get(atom.id, [])
            chunks: list[Chunk] = []
            for section_id in section_ids:
                chunks.extend(chunks_by_section.get(section_id, []))
            with timed_substage("evidence_temporal_validation_ms"):
                chunks = self._apply_temporal_window(
                    chunks,
                    atom=atom,
                    reference_date=reference_date,
                )
            allowed_chunks[atom.id] = chunks
            with timed_substage("evidence_query_build_ms"):
                dense_query = self.build_query(atom)
            dense_groups[atom.id] = (dense_query, chunks)
            logger.info(
                "criterion_sections_chunked",
                criterion_id=atom.id,
                selected_sections=len(section_ids),
                chunks=len(chunks),
            )

        dense_started = time.perf_counter()
        dense_by_atom = self._retrieval_service.dense_rank_chunks_batched(
            dense_groups,
            top_k=self._settings.EVIDENCE_DENSE_TOP_K,
        )
        dense_ms = int((time.perf_counter() - dense_started) * 1000)
        for atom_id, hits in dense_by_atom.items():
            logger.info(
                "criterion_dense_retrieval_completed",
                criterion_id=atom_id,
                input_chunks=len(allowed_chunks.get(atom_id, [])),
                output_chunks=len(hits),
            )

        rerank_groups: list[tuple[str, str, list[RetrievalResult]]] = []
        for atom in atoms:
            hits = dense_by_atom.get(atom.id, [])
            if hits:
                with timed_substage("evidence_query_build_ms"):
                    rerank_query = self.build_query(atom)
                rerank_groups.append((atom.id, rerank_query, hits))

        rerank_started = time.perf_counter()
        if rerank_groups:
            reranked_by_atom = self._reranking_service.rerank_grouped(
                rerank_groups,
                top_k=self._settings.EVIDENCE_RERANK_TOP_K,
                stage="evidence",
            )
        else:
            reranked_by_atom = {}
        rerank_ms = int((time.perf_counter() - rerank_started) * 1000)

        mapping: dict[str, list[str]] = {}
        candidate_by_id: dict[str, Chunk] = {}
        for atom in atoms:
            hits = reranked_by_atom.get(atom.id, [])
            chunk_ids: list[str] = []
            for hit in hits:
                chunk_ids.append(hit.chunk.id)
                candidate_by_id[hit.chunk.id] = hit.chunk
            mapping[atom.id] = chunk_ids
            logger.info(
                "criterion_rerank_completed",
                criterion_id=atom.id,
                input_chunks=len(dense_by_atom.get(atom.id, [])),
                output_chunks=len(hits),
            )

        retrieval_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "evidence_retrieval_completed",
            atom_count=len(atoms),
            unique_evidence_chunks=len(candidate_by_id),
            corpus_sections=len(bm25_index.sections),
            unique_selected_sections=len(union_sections),
            dense_duration_ms=dense_ms,
            reranking_duration_ms=rerank_ms,
            bm25_top_k=self._settings.EVIDENCE_SECTION_BM25_TOP_K,
            dense_top_k=self._settings.EVIDENCE_DENSE_TOP_K,
            rerank_top_k=self._settings.EVIDENCE_RERANK_TOP_K,
            total_retrieval_duration_ms=retrieval_ms,
        )
        return mapping, candidate_by_id, retrieval_ms, rerank_ms

    def _apply_temporal_window(
        self,
        chunks: list[Chunk],
        *,
        atom: AtomicCriterion,
        reference_date: date | None,
    ) -> list[Chunk]:
        """Optionally drop dated evidence clearly outside an explicit window.

        Unknown-date evidence is kept. Without an analysis reference date the
        window cannot be applied; never invent a runtime clock date.
        """
        months = parse_within_months_window(atom.temporal_requirement)
        if months is None or reference_date is None:
            return chunks
        kept: list[Chunk] = []
        for chunk in chunks:
            within = is_within_months(
                chunk.document_date,
                months=months,
                reference=reference_date,
            )
            if within is False:
                continue
            kept.append(chunk)
        return kept

    async def _evaluate_with_llm(
        self,
        *,
        conditions: list[AtomicCriterion],
        exclusions: list[PolicyExclusion],
        mapping: dict[str, list[str]],
        candidate_by_id: dict[str, Chunk],
        reference_date: date | None,
        local_context_by_id: dict[str, str],
    ) -> tuple[EvidenceEvaluationBatch, int, list[dict[str, object]]]:
        metrics = get_request_metrics()
        corpus_before = (
            int(metrics.substage_ms.get("evidence_shared_corpus_build_ms", 0))
            if metrics is not None
            else 0
        )
        token_before = (
            int(metrics.substage_ms.get("evidence_token_count_ms", 0))
            if metrics is not None
            else 0
        )
        dedupe_before = (
            int(metrics.substage_ms.get("evidence_candidate_dedupe_ms", 0))
            if metrics is not None
            else 0
        )
        pack_started = time.perf_counter()
        work_items = self._build_sol_work_items(
            conditions,
            exclusions,
            mapping=mapping,
            candidate_by_id=candidate_by_id,
            reference_date=reference_date,
            local_context_by_id=local_context_by_id,
        )
        pack_wall = int((time.perf_counter() - pack_started) * 1000)
        if metrics is not None:
            corpus_delta = (
                int(metrics.substage_ms.get("evidence_shared_corpus_build_ms", 0))
                - corpus_before
            )
            token_delta = (
                int(metrics.substage_ms.get("evidence_token_count_ms", 0))
                - token_before
            )
            dedupe_delta = (
                int(metrics.substage_ms.get("evidence_candidate_dedupe_ms", 0))
                - dedupe_before
            )
            metrics.add_substage_ms(
                "evidence_batch_pack_ms",
                max(0, pack_wall - corpus_delta - token_delta - dedupe_delta),
            )
        if not work_items:
            return EvidenceEvaluationBatch(), 0, []

        semaphore = asyncio.Semaphore(self._settings.EVIDENCE_SOL_MAX_CONCURRENCY)
        batch_durations_ms: list[int] = [0] * len(work_items)

        async def run_one(
            batch_index: int,
            batch_conditions: list[AtomicCriterion],
            batch_exclusions: list[PolicyExclusion],
            batch_mapping: dict[str, list[str]],
        ) -> tuple[EvidenceEvaluationBatch, dict[str, object], int]:
            async with semaphore:
                run_started = time.perf_counter()
                with timed_substage("evidence_shared_corpus_build_ms"):
                    candidate_map, allowed_by_item = self._build_candidate_registry(
                        item_ids=[
                            *[atom.id for atom in batch_conditions],
                            *[item.id for item in batch_exclusions],
                        ],
                        mapping=batch_mapping,
                        candidate_by_id=candidate_by_id,
                    )
                citation_map = {
                    candidate_id: scoped.to_citation()
                    for candidate_id, scoped in candidate_map.items()
                }
                user_prompt = self._build_evaluation_prompt(
                    conditions=batch_conditions,
                    exclusions=batch_exclusions,
                    candidate_map=candidate_map,
                    allowed_by_item=allowed_by_item,
                    reference_date=reference_date,
                    local_context_by_id=local_context_by_id,
                )
                item_count = len(batch_conditions) + len(batch_exclusions)
                unique_chunk_ids = {
                    scoped.chunk.id for scoped in candidate_map.values()
                }
                try:
                    raw = await self._llm_service.extract_structured(
                        system_prompt=_EVIDENCE_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        response_model=SolEvidenceEvaluationBatch,
                        model=self._settings.POLICY_REASONING_MODEL,
                        operation="evidence_evaluation",
                        max_output_tokens=self._settings.EVIDENCE_LLM_MAX_OUTPUT_TOKENS,
                        reasoning_effort="low",
                        context_items=item_count,
                        batch_index=batch_index,
                        item_count=item_count,
                        unique_evidence_chunks=len(unique_chunk_ids),
                    )
                except EvidenceAnalysisError:
                    raise
                except Exception as exc:
                    raise EvidenceAnalysisError(
                        "Evidence evaluation request failed"
                    ) from exc

                wanted_criteria = {item.id for item in batch_conditions}
                wanted_exclusions = {item.id for item in batch_exclusions}
                filtered = SolEvidenceEvaluationBatch(
                    criteria=[
                        item
                        for item in raw.criteria
                        if item.criterion_id in wanted_criteria
                    ],
                    exclusions=[
                        item
                        for item in raw.exclusions
                        if item.exclusion_id in wanted_exclusions
                    ],
                )
                self._assert_candidate_ids_in_scope(
                    filtered,
                    allowed_by_item=allowed_by_item,
                )
                descriptions_by_id = {
                    atom.id: atom.description for atom in batch_conditions
                }
                with timed_substage("evidence_citation_hydration_ms"):
                    resolved = materialize_sol_batch(
                        filtered,
                        candidate_map=citation_map,
                        descriptions_by_id=descriptions_by_id,
                    )
                corpus_by_id = {
                    candidate_id: {
                        "candidate_id": candidate_id,
                        "chunk_id": scoped.chunk.id,
                        "page_number": scoped.chunk.page_number,
                        "document_date": scoped.chunk.document_date,
                        "text": scoped.chunk.text,
                    }
                    for candidate_id, scoped in candidate_map.items()
                }
                # Stable E1..En order by numeric suffix.
                corpus_entries = [
                    corpus_by_id[candidate_id]
                    for candidate_id in sorted(
                        corpus_by_id,
                        key=lambda value: int(value[1:]),
                    )
                ]
                with timed_substage("evidence_token_count_ms"):
                    rendered_input_tokens = count_tokens(
                        f"{_EVIDENCE_SYSTEM_PROMPT}\n{user_prompt}"
                    )
                run_ms = int((time.perf_counter() - run_started) * 1000)
                trace: dict[str, object] = {
                    "rendered_input_tokens": rendered_input_tokens,
                    "item_count": item_count,
                    "unique_evidence_chunk_count": len(unique_chunk_ids),
                    "evidence_corpus": corpus_entries,
                    "items": [
                        {
                            "item_id": item_id,
                            "allowed_candidate_ids": sorted(
                                candidate_ids,
                                key=lambda value: int(value[1:]),
                            ),
                        }
                        for item_id, candidate_ids in allowed_by_item.items()
                    ],
                    "sol_raw": filtered.model_dump(),
                    "normalized": resolved.model_dump(),
                }
                return resolved, trace, run_ms

        wall_started = time.perf_counter()
        gathered = await asyncio.gather(
            *[
                run_one(
                    batch_index,
                    batch_conditions,
                    batch_exclusions,
                    batch_mapping,
                )
                for batch_index, (
                    batch_conditions,
                    batch_exclusions,
                    batch_mapping,
                ) in enumerate(work_items, start=1)
            ]
        )
        wall_ms = int((time.perf_counter() - wall_started) * 1000)

        merged_criteria: list[CriterionEvaluation] = []
        merged_exclusions: list[ExclusionEvaluation] = []
        batch_traces: list[dict[str, object]] = []
        for index, (result, trace, run_ms) in enumerate(gathered):
            merged_criteria.extend(result.criteria)
            merged_exclusions.extend(result.exclusions)
            batch_traces.append(trace)
            batch_durations_ms[index] = run_ms
        metrics = get_request_metrics()
        if metrics is not None:
            # Prefer pure Sol call durations for concurrency analysis.
            llm_batch_ms = [
                int(metrics.substage_ms.get(f"evidence_sol_batch_{i}_ms", 0))
                for i in range(1, len(work_items) + 1)
            ]
            if any(llm_batch_ms):
                batch_durations_ms = llm_batch_ms
            metrics.record_evidence_sol_wall(
                wall_ms=wall_ms,
                batch_durations_ms=batch_durations_ms,
            )
        return (
            EvidenceEvaluationBatch(
                criteria=merged_criteria,
                exclusions=merged_exclusions,
            ),
            len(gathered),
            batch_traces,
        )

    def _build_candidate_registry(
        self,
        *,
        item_ids: Sequence[str],
        mapping: dict[str, list[str]],
        candidate_by_id: dict[str, Chunk],
    ) -> tuple[dict[str, ScopedEvidenceCandidate], dict[str, set[str]]]:
        """Build a shared E1..En evidence corpus for one Sol batch.

        Each physical patient chunk appears once. Multiple criteria may share
        the same E ID while retaining per-item ``allowed_candidate_ids``.
        """
        candidate_map: dict[str, ScopedEvidenceCandidate] = {}
        allowed_by_item: dict[str, set[str]] = {}
        chunk_to_candidate: dict[str, str] = {}
        for item_id in item_ids:
            allowed: set[str] = set()
            for chunk_id in mapping.get(item_id, []):
                chunk = candidate_by_id[chunk_id]
                candidate_id = chunk_to_candidate.get(chunk_id)
                if candidate_id is None:
                    candidate_id = f"E{len(chunk_to_candidate) + 1}"
                    chunk_to_candidate[chunk_id] = candidate_id
                    candidate_map[candidate_id] = ScopedEvidenceCandidate(
                        candidate_id=candidate_id,
                        item_id=item_id,
                        chunk=chunk,
                    )
                allowed.add(candidate_id)
            allowed_by_item[item_id] = allowed
        return candidate_map, allowed_by_item

    def _assert_candidate_ids_in_scope(
        self,
        batch: SolEvidenceEvaluationBatch,
        *,
        allowed_by_item: dict[str, set[str]],
    ) -> None:
        for criterion_item in batch.criteria:
            allowed = allowed_by_item.get(criterion_item.criterion_id, set())
            for candidate_id in criterion_item.evidence_candidate_ids:
                if candidate_id not in allowed:
                    logger.info(
                        "evidence_grounding_failed",
                        item_id=criterion_item.criterion_id,
                        reason="candidate_outside_criterion",
                        candidate_id=candidate_id,
                    )
                    raise EvidenceGroundingError(
                        "Evidence grounding failed for criterion "
                        f"{criterion_item.criterion_id}"
                    )
        for exclusion_item in batch.exclusions:
            allowed = allowed_by_item.get(exclusion_item.exclusion_id, set())
            for candidate_id in exclusion_item.evidence_candidate_ids:
                if candidate_id not in allowed:
                    logger.info(
                        "evidence_grounding_failed",
                        item_id=exclusion_item.exclusion_id,
                        reason="candidate_outside_criterion",
                        candidate_id=candidate_id,
                    )
                    raise EvidenceGroundingError(
                        "Evidence grounding failed for exclusion "
                        f"{exclusion_item.exclusion_id}"
                    )

    def _build_sol_work_items(
        self,
        conditions: Sequence[AtomicCriterion],
        exclusions: Sequence[PolicyExclusion],
        *,
        mapping: dict[str, list[str]],
        candidate_by_id: dict[str, Chunk],
        reference_date: date | None,
        local_context_by_id: dict[str, str],
    ) -> list[
        tuple[list[AtomicCriterion], list[PolicyExclusion], dict[str, list[str]]]
    ]:
        """Pack criteria into Sol batches by rendered-token budget."""
        max_batch = max(1, self._settings.EVIDENCE_SOL_MAX_ITEMS_PER_BATCH)
        with timed_substage("evidence_candidate_dedupe_ms"):
            deduped_mapping = {
                key: self._dedupe_chunk_ids(chunk_ids, candidate_by_id)
                for key, chunk_ids in mapping.items()
            }

        if conditions:
            batches = self._pack_conditions_within_budget(
                list(conditions),
                exclusions=list(exclusions),
                mapping=deduped_mapping,
                candidate_by_id=candidate_by_id,
                reference_date=reference_date,
                local_context_by_id=local_context_by_id,
                max_batch=max_batch,
            )
            return batches

        if not exclusions:
            return []

        packed: list[
            tuple[list[AtomicCriterion], list[PolicyExclusion], dict[str, list[str]]]
        ] = []
        remaining = list(exclusions)
        while remaining:
            batch: list[PolicyExclusion] = []
            batch_mapping: dict[str, list[str]] = {}
            while remaining and len(batch) < max_batch:
                candidate = remaining[0]
                trial = batch + [candidate]
                trial_mapping = {
                    **batch_mapping,
                    candidate.id: list(deduped_mapping.get(candidate.id, [])),
                }
                if batch and not self._prompt_fits(
                    conditions=[],
                    exclusions=trial,
                    mapping=trial_mapping,
                    candidate_by_id=candidate_by_id,
                    reference_date=reference_date,
                    local_context_by_id=local_context_by_id,
                ):
                    break
                remaining.pop(0)
                batch = trial
                batch_mapping = trial_mapping
                if len(batch) == 1 and not self._prompt_fits(
                    conditions=[],
                    exclusions=batch,
                    mapping=batch_mapping,
                    candidate_by_id=candidate_by_id,
                    reference_date=reference_date,
                    local_context_by_id=local_context_by_id,
                ):
                    batch_mapping[candidate.id] = self._trim_chunk_ids_to_fit(
                        conditions=[],
                        exclusions=batch,
                        item_id=candidate.id,
                        chunk_ids=list(deduped_mapping.get(candidate.id, [])),
                        mapping={candidate.id: []},
                        candidate_by_id=candidate_by_id,
                        reference_date=reference_date,
                        local_context_by_id=local_context_by_id,
                    )
                    break
            packed.append(([], batch, batch_mapping))
        return packed

    def _pack_conditions_within_budget(
        self,
        conditions: list[AtomicCriterion],
        *,
        exclusions: list[PolicyExclusion],
        mapping: dict[str, list[str]],
        candidate_by_id: dict[str, Chunk],
        reference_date: date | None,
        local_context_by_id: dict[str, str],
        max_batch: int,
    ) -> list[
        tuple[list[AtomicCriterion], list[PolicyExclusion], dict[str, list[str]]]
    ]:
        packed: list[
            tuple[list[AtomicCriterion], list[PolicyExclusion], dict[str, list[str]]]
        ] = []
        remaining = list(conditions)
        while remaining:
            batch: list[AtomicCriterion] = []
            batch_mapping: dict[str, list[str]] = {}
            is_last = False
            while remaining and len(batch) < max_batch:
                candidate = remaining[0]
                trial = batch + [candidate]
                trial_mapping = {
                    **batch_mapping,
                    candidate.id: list(mapping.get(candidate.id, [])),
                }
                # Attach exclusions only when this would consume the rest.
                attach = exclusions if len(remaining) == 1 else []
                if batch and not self._prompt_fits(
                    conditions=trial,
                    exclusions=attach,
                    mapping=trial_mapping,
                    candidate_by_id=candidate_by_id,
                    reference_date=reference_date,
                    local_context_by_id=local_context_by_id,
                ):
                    break
                remaining.pop(0)
                batch = trial
                batch_mapping = trial_mapping
                is_last = len(remaining) == 0
                if len(batch) == 1 and not self._prompt_fits(
                    conditions=batch,
                    exclusions=attach if is_last else [],
                    mapping=batch_mapping,
                    candidate_by_id=candidate_by_id,
                    reference_date=reference_date,
                    local_context_by_id=local_context_by_id,
                ):
                    batch_mapping[candidate.id] = self._trim_chunk_ids_to_fit(
                        conditions=batch,
                        exclusions=attach if is_last else [],
                        item_id=candidate.id,
                        chunk_ids=list(mapping.get(candidate.id, [])),
                        mapping={candidate.id: []},
                        candidate_by_id=candidate_by_id,
                        reference_date=reference_date,
                        local_context_by_id=local_context_by_id,
                    )
                    break
            attached = list(exclusions) if is_last else []
            if attached:
                for exclusion in attached:
                    batch_mapping[exclusion.id] = list(
                        mapping.get(exclusion.id, [])
                    )
                if not self._prompt_fits(
                    conditions=batch,
                    exclusions=attached,
                    mapping=batch_mapping,
                    candidate_by_id=candidate_by_id,
                    reference_date=reference_date,
                    local_context_by_id=local_context_by_id,
                ):
                    # Prefer keeping criteria; spill exclusions to follow-up.
                    packed.append((batch, [], {
                        key: value
                        for key, value in batch_mapping.items()
                        if key in {item.id for item in batch}
                    }))
                    # Re-queue exclusions via a synthetic empty-condition path.
                    for spill in self._build_sol_work_items(
                        [],
                        attached,
                        mapping=mapping,
                        candidate_by_id=candidate_by_id,
                        reference_date=reference_date,
                        local_context_by_id=local_context_by_id,
                    ):
                        packed.append(spill)
                    exclusions = []
                    continue
            packed.append((batch, attached, batch_mapping))
            if is_last:
                exclusions = []
        return packed

    def _prompt_fits(
        self,
        *,
        conditions: list[AtomicCriterion],
        exclusions: list[PolicyExclusion],
        mapping: dict[str, list[str]],
        candidate_by_id: dict[str, Chunk],
        reference_date: date | None,
        local_context_by_id: dict[str, str],
    ) -> bool:
        # Input packing budget is independent of the completion ceiling.
        max_input = self._settings.EVIDENCE_LLM_MAX_CONTEXT_TOKENS
        with timed_substage("evidence_shared_corpus_build_ms"):
            candidate_map, allowed_by_item = self._build_candidate_registry(
                item_ids=[
                    *[atom.id for atom in conditions],
                    *[item.id for item in exclusions],
                ],
                mapping=mapping,
                candidate_by_id=candidate_by_id,
            )
        prompt = self._build_evaluation_prompt(
            conditions=conditions,
            exclusions=exclusions,
            candidate_map=candidate_map,
            allowed_by_item=allowed_by_item,
            reference_date=reference_date,
            local_context_by_id=local_context_by_id,
        )
        with timed_substage("evidence_token_count_ms"):
            token_count = count_tokens(f"{_EVIDENCE_SYSTEM_PROMPT}\n{prompt}")
        return token_count <= max_input

    def _trim_chunk_ids_to_fit(
        self,
        *,
        conditions: list[AtomicCriterion],
        exclusions: list[PolicyExclusion],
        item_id: str,
        chunk_ids: list[str],
        mapping: dict[str, list[str]],
        candidate_by_id: dict[str, Chunk],
        reference_date: date | None,
        local_context_by_id: dict[str, str],
    ) -> list[str]:
        kept = list(chunk_ids)
        while kept:
            trial_mapping = {**mapping, item_id: kept}
            if self._prompt_fits(
                conditions=conditions,
                exclusions=exclusions,
                mapping=trial_mapping,
                candidate_by_id=candidate_by_id,
                reference_date=reference_date,
                local_context_by_id=local_context_by_id,
            ):
                return kept
            if len(kept) <= 1:
                return kept
            kept = kept[:-1]
        return kept

    @staticmethod
    def _dedupe_chunk_ids(
        chunk_ids: Sequence[str],
        candidate_by_id: dict[str, Chunk],
    ) -> list[str]:
        unique: list[str] = []
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        for chunk_id in chunk_ids:
            if chunk_id in seen_ids:
                continue
            chunk = candidate_by_id.get(chunk_id)
            text_key = normalize_text(chunk.text) if chunk is not None else ""
            if text_key and text_key in seen_text:
                continue
            seen_ids.add(chunk_id)
            if text_key:
                seen_text.add(text_key)
            unique.append(chunk_id)
        return unique

    def _build_evaluation_prompt(
        self,
        *,
        conditions: list[AtomicCriterion],
        exclusions: list[PolicyExclusion],
        candidate_map: dict[str, ScopedEvidenceCandidate],
        allowed_by_item: dict[str, set[str]],
        reference_date: date | None,
        local_context_by_id: dict[str, str],
    ) -> str:
        reference_label = (
            reference_date.isoformat() if reference_date is not None else "unknown"
        )
        corpus_ids = sorted(
            candidate_map,
            key=lambda value: int(value[1:]),
        )
        blocks: list[str] = [
            "Evaluate each criterion and exclusion using only that item's "
            "allowed_candidate_ids from the shared evidence_corpus. Interpret "
            "atoms using LOCAL POLICY CONTEXT.",
            f"analysis_reference_date: {reference_label}",
            "",
            "evidence_corpus:",
        ]
        if not corpus_ids:
            blocks.append("- none")
        else:
            for candidate_id in corpus_ids:
                scoped = candidate_map[candidate_id]
                chunk = scoped.chunk
                blocks.extend(
                    [
                        f"- candidate_id: {candidate_id}",
                        f"  page_number: {chunk.page_number}",
                        f"  document_date: {chunk.document_date}",
                        "  text:",
                        chunk.text,
                    ]
                )

        blocks.append("")
        blocks.append("items:")
        if not conditions and not exclusions:
            blocks.append("- none")
        for atom in conditions:
            context = local_context_by_id.get(atom.id) or atom.local_policy_context
            allowed = sorted(
                allowed_by_item.get(atom.id, set()),
                key=lambda value: int(value[1:]),
            )
            blocks.extend(
                [
                    f"- criterion_id: {atom.id}",
                    f"  description: {atom.description}",
                    f"  category: {atom.category}",
                    f"  temporal_requirement: {atom.temporal_requirement}",
                    "  local_policy_context:",
                    context or "    none",
                    f"  allowed_candidate_ids: {allowed}",
                ]
            )
        for exclusion in exclusions:
            context = local_context_by_id.get(exclusion.id) or exclusion.source_text
            allowed = sorted(
                allowed_by_item.get(exclusion.id, set()),
                key=lambda value: int(value[1:]),
            )
            blocks.extend(
                [
                    f"- exclusion_id: {exclusion.id}",
                    f"  description: {exclusion.description}",
                    "  local_policy_context:",
                    context or "    none",
                    f"  allowed_candidate_ids: {allowed}",
                ]
            )
        return "\n".join(blocks)

    @staticmethod
    def _llm_usage_snapshot() -> dict[str, int]:
        metrics = get_request_metrics()
        if metrics is None:
            return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        for usage in metrics.model_usage.values():
            totals["input_tokens"] += usage.input_tokens
            totals["output_tokens"] += usage.output_tokens
            totals["reasoning_tokens"] += usage.reasoning_tokens
        return totals

    def _llm_usage_delta(self, before: dict[str, int]) -> dict[str, int]:
        after = self._llm_usage_snapshot()
        return {
            key: max(0, after.get(key, 0) - before.get(key, 0))
            for key in ("input_tokens", "output_tokens", "reasoning_tokens")
        }

    def _validate_batch(
        self,
        *,
        batch: EvidenceEvaluationBatch,
        expected_condition_ids: set[str],
        expected_exclusion_ids: set[str],
        candidate_by_id: dict[str, Chunk],
        mapping: dict[str, list[str]],
        reference_date: date | None,
    ) -> tuple[list[CriterionEvaluation], list[ExclusionEvaluation]]:
        criteria_by_id = {item.criterion_id: item for item in batch.criteria}
        if set(criteria_by_id) != expected_condition_ids:
            missing = expected_condition_ids - set(criteria_by_id)
            extra = set(criteria_by_id) - expected_condition_ids
            raise EvidenceValidationError(
                "Criterion evaluation IDs mismatch "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )

        exclusions_by_id = {item.exclusion_id: item for item in batch.exclusions}
        if set(exclusions_by_id) != expected_exclusion_ids:
            missing = expected_exclusion_ids - set(exclusions_by_id)
            extra = set(exclusions_by_id) - expected_exclusion_ids
            raise EvidenceValidationError(
                "Exclusion evaluation IDs mismatch "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )

        validated_criteria: list[CriterionEvaluation] = []
        for criterion_id in sorted(expected_condition_ids):
            criterion = criteria_by_id[criterion_id]
            allowed = set(mapping.get(criterion_id, []))
            validated_criteria.append(
                self._validate_criterion_evaluation(
                    criterion,
                    candidate_by_id=candidate_by_id,
                    allowed_chunk_ids=allowed,
                    reference_date=reference_date,
                )
            )

        validated_exclusions: list[ExclusionEvaluation] = []
        for exclusion_id in sorted(expected_exclusion_ids):
            exclusion = exclusions_by_id[exclusion_id]
            allowed = set(mapping.get(exclusion_id, []))
            validated_exclusions.append(
                self._validate_exclusion_evaluation(
                    exclusion,
                    candidate_by_id=candidate_by_id,
                    allowed_chunk_ids=allowed,
                    reference_date=reference_date,
                )
            )
        return validated_criteria, validated_exclusions

    def _validate_criterion_evaluation(
        self,
        item: CriterionEvaluation,
        *,
        candidate_by_id: dict[str, Chunk],
        allowed_chunk_ids: set[str],
        reference_date: date | None,
    ) -> CriterionEvaluation:
        if not item.criterion_id.strip():
            raise EvidenceValidationError("Criterion evaluation missing criterion_id")
        if not item.explanation.strip():
            raise EvidenceValidationError(
                f"Criterion {item.criterion_id} missing explanation"
            )

        if item.status == CriterionStatus.MET:
            if not item.evidence:
                logger.info(
                    "evidence_validation_failed",
                    criterion_id=item.criterion_id,
                    reason="met_without_evidence",
                )
                raise EvidenceValidationError(
                    f"MET criterion {item.criterion_id} requires evidence"
                )
            evidence, warnings = self._validate_evidence_grounding(
                item_id=item.criterion_id,
                evidence=item.evidence,
                candidate_by_id=candidate_by_id,
                allowed_chunk_ids=allowed_chunk_ids,
                reference_date=reference_date,
            )
            return item.model_copy(
                update={"evidence": evidence, "temporal_warnings": warnings}
            )
        if item.status in {CriterionStatus.UNCLEAR, CriterionStatus.MISSING}:
            if item.evidence:
                evidence, warnings = self._validate_evidence_grounding(
                    item_id=item.criterion_id,
                    evidence=item.evidence,
                    candidate_by_id=candidate_by_id,
                    allowed_chunk_ids=allowed_chunk_ids,
                    reference_date=reference_date,
                )
                return item.model_copy(
                    update={"evidence": evidence, "temporal_warnings": warnings}
                )
            return item
        raise EvidenceValidationError(
            f"Unsupported criterion status for {item.criterion_id}"
        )

    def _validate_exclusion_evaluation(
        self,
        item: ExclusionEvaluation,
        *,
        candidate_by_id: dict[str, Chunk],
        allowed_chunk_ids: set[str],
        reference_date: date | None,
    ) -> ExclusionEvaluation:
        if not item.exclusion_id.strip():
            raise EvidenceValidationError("Exclusion evaluation missing exclusion_id")
        if not item.explanation.strip():
            raise EvidenceValidationError(
                f"Exclusion {item.exclusion_id} missing explanation"
            )

        if item.triggered and item.status != CriterionStatus.MET:
            raise EvidenceValidationError(
                f"Triggered exclusion {item.exclusion_id} must have status MET"
            )
        if item.status == CriterionStatus.MET:
            if not item.evidence:
                raise EvidenceValidationError(
                    f"MET exclusion {item.exclusion_id} requires evidence"
                )
            evidence, warnings = self._validate_evidence_grounding(
                item_id=item.exclusion_id,
                evidence=item.evidence,
                candidate_by_id=candidate_by_id,
                allowed_chunk_ids=allowed_chunk_ids,
                reference_date=reference_date,
            )
            return item.model_copy(
                update={"evidence": evidence, "temporal_warnings": warnings}
            )
        if item.evidence:
            evidence, warnings = self._validate_evidence_grounding(
                item_id=item.exclusion_id,
                evidence=item.evidence,
                candidate_by_id=candidate_by_id,
                allowed_chunk_ids=allowed_chunk_ids,
                reference_date=reference_date,
            )
            return item.model_copy(
                update={"evidence": evidence, "temporal_warnings": warnings}
            )
        return item

    def _validate_evidence_grounding(
        self,
        *,
        item_id: str,
        evidence: list[EvidenceCitation],
        candidate_by_id: dict[str, Chunk],
        allowed_chunk_ids: set[str],
        reference_date: date | None,
    ) -> tuple[list[EvidenceCitation], list[str]]:
        validated: list[EvidenceCitation] = []
        warnings: list[str] = []
        for citation in evidence:
            chunk_id = citation.chunk_id
            if chunk_id not in allowed_chunk_ids:
                logger.info(
                    "evidence_grounding_failed",
                    item_id=item_id,
                    reason="chunk_outside_candidates",
                    chunk_id=chunk_id,
                )
                raise EvidenceGroundingError(
                    f"Evidence grounding failed for {item_id}"
                )
            chunk = candidate_by_id.get(chunk_id)
            if chunk is None:
                logger.info(
                    "evidence_grounding_failed",
                    item_id=item_id,
                    reason="unknown_chunk_id",
                    chunk_id=chunk_id,
                )
                raise EvidenceGroundingError(
                    f"Evidence grounding failed for {item_id}"
                )

            # Metadata always comes from the retrieved chunk, never Sol.
            resolved = citation_from_chunk(chunk)
            with timed_substage("evidence_temporal_validation_ms"):
                for warning in chunk.temporal_warnings:
                    if warning not in warnings:
                        warnings.append(warning)
            validated.append(resolved)
        _ = reference_date  # retained for debug traces / future window checks
        return validated, warnings

    def _write_evidence_debug_trace(
        self,
        *,
        conditions: list[AtomicCriterion],
        exclusions: list[PolicyExclusion],
        mapping: dict[str, list[str]],
        candidate_by_id: dict[str, Chunk],
        criteria: list[CriterionEvaluation],
        exclusion_evals: list[ExclusionEvaluation],
        reference_date: date | None,
        batch_traces: list[dict[str, object]] | None = None,
    ) -> None:
        reference_label = (
            reference_date.isoformat() if reference_date is not None else None
        )
        criteria_by_id = {item.criterion_id: item for item in criteria}
        exclusions_by_id = {item.exclusion_id: item for item in exclusion_evals}
        # Aggregate item-level allowed sets for overview (E IDs are batch-local;
        # here we expose underlying chunk IDs for debugging).
        per_item: list[dict[str, object]] = []
        for atom in conditions:
            criterion_result = criteria_by_id.get(atom.id)
            per_item.append(
                {
                    "criterion_id": atom.id,
                    "allowed_chunk_ids": list(mapping.get(atom.id, [])),
                    "normalized": (
                        None
                        if criterion_result is None
                        else {
                            "criterion_id": criterion_result.criterion_id,
                            "status": criterion_result.status.value,
                            "evidence": [
                                citation.model_dump()
                                for citation in criterion_result.evidence
                            ],
                            "explanation": criterion_result.explanation,
                        }
                    ),
                }
            )
        for exclusion in exclusions:
            exclusion_result = exclusions_by_id.get(exclusion.id)
            per_item.append(
                {
                    "exclusion_id": exclusion.id,
                    "allowed_chunk_ids": list(mapping.get(exclusion.id, [])),
                    "normalized": (
                        None
                        if exclusion_result is None
                        else {
                            "exclusion_id": exclusion_result.exclusion_id,
                            "status": exclusion_result.status.value,
                            "triggered": exclusion_result.triggered,
                            "evidence": [
                                citation.model_dump()
                                for citation in exclusion_result.evidence
                            ],
                            "explanation": exclusion_result.explanation,
                        }
                    ),
                }
            )

        indexed_batches: list[dict[str, object]] = []
        for index, batch in enumerate(batch_traces or [], start=1):
            indexed_batches.append(
                {
                    "batch_index": index,
                    "rendered_input_tokens": batch.get("rendered_input_tokens"),
                    "item_count": batch.get("item_count"),
                    "unique_evidence_chunk_count": batch.get(
                        "unique_evidence_chunk_count"
                    ),
                    "evidence_corpus": batch.get("evidence_corpus"),
                    "items": batch.get("items"),
                    "sol_raw": batch.get("sol_raw"),
                    "normalized": batch.get("normalized"),
                }
            )

        write_debug_trace(
            "evidence_context_sent_to_sol.json",
            {
                "analysis_reference_date": reference_label,
                "items": per_item,
                "batches": indexed_batches,
            },
        )
        write_debug_trace(
            "evidence_sol_raw_batches.json",
            {
                "analysis_reference_date": reference_label,
                "batches": [
                    {
                        "batch_index": batch.get("batch_index"),
                        "sol_raw": batch.get("sol_raw"),
                        "normalized": batch.get("normalized"),
                    }
                    for batch in indexed_batches
                ],
            },
        )
        write_debug_trace(
            "evidence_citations.json",
            {
                "analysis_reference_date": reference_label,
                "citations": [
                    {
                        "item_id": item.criterion_id,
                        "chunk_id": citation.chunk_id,
                        "text": citation.text,
                        "page_number": citation.page_number,
                        "document_date": citation.document_date,
                    }
                    for item in criteria
                    for citation in item.evidence
                ]
                + [
                    {
                        "item_id": item.exclusion_id,
                        "chunk_id": citation.chunk_id,
                        "text": citation.text,
                        "page_number": citation.page_number,
                        "document_date": citation.document_date,
                    }
                    for item in exclusion_evals
                    for citation in item.evidence
                ],
            },
        )

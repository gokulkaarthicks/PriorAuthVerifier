"""Request-scoped analysis metrics and usage accounting."""

from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def is_performance_profile_enabled() -> bool:
    """Return True when detailed performance profiling should be emitted."""
    return bool(get_settings().PERFORMANCE_PROFILE_ENABLED)


@dataclass
class LlmModelUsage:
    """Aggregated OpenAI usage for one model name."""

    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LocalModelUsage:
    """Aggregated local-model workload (not API billing tokens)."""

    batches: int = 0
    texts: int = 0
    pairs: int = 0
    tokens: int | None = None


@dataclass
class LlmCallDetail:
    """Per-call LLM wall-clock and token accounting."""

    operation: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    duration_ms: int
    context_items: int | None = None
    item_count: int | None = None
    unique_evidence_chunks: int | None = None
    batch_index: int | None = None


@dataclass
class EmbeddingCallDetail:
    """Per-encode embedding call accounting."""

    operation: str
    text_count: int
    unique_text_count: int
    duplicate_text_count: int
    duration_ms: int


@dataclass
class RerankerProfile:
    """Cross-encoder pair and timing profile for one request."""

    calls: int = 0
    batches: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    attempted_pairs: int = 0
    unique_pairs: int = 0
    duplicate_pairs: int = 0
    inference_ms: int = 0
    preprocessing_ms: int = 0
    procedure_pairs: int = 0
    policy_pairs: int = 0
    evidence_pairs: int = 0
    _seen_pair_hashes: set[str] = field(default_factory=set, repr=False)

    def record_pairs(
        self,
        *,
        stage: str,
        pairs: list[tuple[str, str]],
        batch_sizes: list[int],
        preprocessing_ms: int,
        inference_ms: int,
    ) -> None:
        self.calls += 1
        self.batches += len(batch_sizes)
        self.batch_sizes.extend(batch_sizes)
        self.preprocessing_ms += max(0, int(preprocessing_ms))
        self.inference_ms += max(0, int(inference_ms))
        self.attempted_pairs += len(pairs)

        unique_in_call = 0
        duplicate_in_call = 0
        for query_text, document_text in pairs:
            pair_hash = _pair_identity(query_text, document_text)
            if pair_hash in self._seen_pair_hashes:
                duplicate_in_call += 1
            else:
                self._seen_pair_hashes.add(pair_hash)
                unique_in_call += 1
        self.unique_pairs += unique_in_call
        self.duplicate_pairs += duplicate_in_call

        if stage == "procedure":
            self.procedure_pairs += len(pairs)
        elif stage == "policy":
            self.policy_pairs += len(pairs)
        elif stage == "evidence":
            self.evidence_pairs += len(pairs)


# Leaf substages shown in the sorted wall-clock table (no parent rollups).
_LEAF_SUBSTAGES: tuple[str, ...] = (
    "patient_pdf_parse_ms",
    "patient_page_text_extract_ms",
    "patient_ocr_ms",
    "patient_section_build_ms",
    "patient_date_metadata_ms",
    "procedure_heading_embedding_ms",
    "procedure_heading_dense_ms",
    "procedure_selected_chunk_build_ms",
    "procedure_bm25_ms",
    "procedure_chunk_embedding_ms",
    "procedure_dense_ms",
    "procedure_rerank_ms",
    "procedure_context_render_ms",
    "procedure_luna_ms",
    "procedure_normalization_ms",
    "policy_pdf_parse_ms",
    "policy_page_text_extract_ms",
    "policy_ocr_ms",
    "policy_section_build_ms",
    "policy_bm25_index_build_ms",
    "policy_exact_cpt_search_ms",
    "policy_bm25_search_ms",
    "policy_candidate_union_ms",
    "policy_candidate_embedding_ms",
    "policy_dense_ms",
    "policy_rerank_ms",
    "policy_seed_filter_ms",
    "policy_structural_expansion_ms",
    "policy_chunk_build_ms",
    "policy_context_dedupe_ms",
    "policy_context_render_ms",
    "policy_context_token_count_ms",
    "policy_sol_ms",
    "policy_sol_normalization_ms",
    "policy_grounding_hydration_ms",
    "evidence_bm25_index_build_ms",
    "evidence_query_build_ms",
    "evidence_bm25_selection_ms",
    "evidence_section_chunk_build_ms",
    "evidence_patient_embedding_ms",
    "evidence_criterion_embedding_ms",
    "evidence_dense_scoring_ms",
    "evidence_reranker_prepare_ms",
    "evidence_reranker_inference_ms",
    "evidence_candidate_dedupe_ms",
    "evidence_shared_corpus_build_ms",
    "evidence_batch_pack_ms",
    "evidence_token_count_ms",
    "evidence_sol_wall_ms",
    "evidence_normalization_ms",
    "evidence_citation_hydration_ms",
    "evidence_temporal_validation_ms",
    "readiness_tree_eval_ms",
    "readiness_missing_items_ms",
    "readiness_exclusions_ms",
    "readiness_response_build_ms",
)


@dataclass
class RequestMetrics:
    """In-process accounting for a single /api/analyze request."""

    request_id: str | None = None
    status: str | None = None
    error_type: str | None = None
    patient_pages: int = 0
    policy_pages: int = 0
    criteria_count: int = 0
    stage_ms: dict[str, int] = field(default_factory=dict)
    substage_ms: dict[str, int] = field(default_factory=dict)
    model_usage: dict[str, LlmModelUsage] = field(default_factory=dict)
    local_model_usage: dict[str, LocalModelUsage] = field(default_factory=dict)
    llm_calls: list[LlmCallDetail] = field(default_factory=list)
    embedding_calls: list[EmbeddingCallDetail] = field(default_factory=list)
    reranker: RerankerProfile = field(default_factory=RerankerProfile)
    patient_pdf_parse_count: int = 0
    policy_pdf_parse_count: int = 0
    patient_ocr_page_count: int = 0
    policy_ocr_page_count: int = 0
    embedding_text_hashes: set[str] = field(default_factory=set, repr=False)
    embedding_total_text_requests: int = 0
    embedding_unique_texts: int = 0
    embedding_duplicate_requests: int = 0
    embedding_total_ms: int = 0
    evidence_sol_batch_durations_ms: list[int] = field(default_factory=list)
    evidence_sol_wall_ms: int = 0
    evidence_sol_concurrent: bool | None = None
    policy_sol_group_nodes: int = 0
    policy_sol_condition_nodes: int = 0
    policy_sol_exclusions: int = 0
    policy_sol_total_nodes: int = 0
    _started: float = field(default_factory=time.perf_counter, repr=False)

    def record_stage_time(self, stage: str, duration_ms: int) -> None:
        self.stage_ms[stage] = int(duration_ms)

    def add_substage_ms(self, name: str, duration_ms: int) -> None:
        key = name if name.endswith("_ms") else f"{name}_ms"
        self.substage_ms[key] = self.substage_ms.get(key, 0) + max(0, int(duration_ms))

    def set_substage_ms(self, name: str, duration_ms: int) -> None:
        key = name if name.endswith("_ms") else f"{name}_ms"
        self.substage_ms[key] = max(0, int(duration_ms))

    def record_llm_usage(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        total_tokens: int = 0,
        failed: bool = False,
    ) -> None:
        entry = self.model_usage.setdefault(model, LlmModelUsage())
        entry.calls += 1
        if failed:
            entry.failed_calls += 1
        entry.input_tokens += max(0, int(input_tokens))
        entry.output_tokens += max(0, int(output_tokens))
        entry.reasoning_tokens += max(0, int(reasoning_tokens))
        entry.total_tokens += max(0, int(total_tokens))

    def record_llm_call(
        self,
        *,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        total_tokens: int,
        duration_ms: int,
        context_items: int | None = None,
        item_count: int | None = None,
        unique_evidence_chunks: int | None = None,
        batch_index: int | None = None,
    ) -> None:
        self.llm_calls.append(
            LlmCallDetail(
                operation=operation,
                model=model,
                input_tokens=max(0, int(input_tokens)),
                output_tokens=max(0, int(output_tokens)),
                reasoning_tokens=max(0, int(reasoning_tokens)),
                total_tokens=max(0, int(total_tokens)),
                duration_ms=max(0, int(duration_ms)),
                context_items=context_items,
                item_count=item_count,
                unique_evidence_chunks=unique_evidence_chunks,
                batch_index=batch_index,
            )
        )
        if operation == "policy_criteria":
            self.set_substage_ms("policy_sol_ms", duration_ms)
        elif operation == "procedure_extraction":
            self.set_substage_ms("procedure_luna_ms", duration_ms)
        elif operation == "evidence_evaluation" and batch_index is not None:
            self.set_substage_ms(f"evidence_sol_batch_{batch_index}_ms", duration_ms)
            while len(self.evidence_sol_batch_durations_ms) < batch_index:
                self.evidence_sol_batch_durations_ms.append(0)
            self.evidence_sol_batch_durations_ms[batch_index - 1] = int(duration_ms)

    def record_local_embedding_batch(
        self,
        model: str,
        *,
        batches: int,
        texts: int,
        tokens: int | None = None,
    ) -> None:
        entry = self.local_model_usage.setdefault(model, LocalModelUsage())
        entry.batches += max(0, int(batches))
        entry.texts += max(0, int(texts))
        if tokens is not None:
            entry.tokens = (entry.tokens or 0) + max(0, int(tokens))

    def record_embedding_call(
        self,
        *,
        operation: str,
        texts: list[str],
        duration_ms: int,
    ) -> None:
        text_count = len(texts)
        unique_in_call = 0
        duplicate_in_call = 0
        for text in texts:
            digest = _text_hash(text)
            if digest in self.embedding_text_hashes:
                duplicate_in_call += 1
                self.embedding_duplicate_requests += 1
            else:
                self.embedding_text_hashes.add(digest)
                unique_in_call += 1
                self.embedding_unique_texts += 1
        self.embedding_total_text_requests += text_count
        self.embedding_total_ms += max(0, int(duration_ms))
        self.embedding_calls.append(
            EmbeddingCallDetail(
                operation=operation or "embed",
                text_count=text_count,
                unique_text_count=unique_in_call,
                duplicate_text_count=duplicate_in_call,
                duration_ms=max(0, int(duration_ms)),
            )
        )

    def record_reranker_batch(
        self,
        model: str,
        *,
        batches: int,
        pairs: int,
        tokens: int | None = None,
    ) -> None:
        entry = self.local_model_usage.setdefault(model, LocalModelUsage())
        entry.batches += max(0, int(batches))
        entry.pairs += max(0, int(pairs))
        if tokens is not None:
            entry.tokens = (entry.tokens or 0) + max(0, int(tokens))

    def record_parser_invocation(self, document_type: str) -> None:
        normalized = (document_type or "").strip().upper()
        if normalized in {"PATIENT_CHART", "PATIENT"}:
            self.patient_pdf_parse_count += 1
        elif normalized == "POLICY":
            self.policy_pdf_parse_count += 1

    def record_ocr_pages(self, document_type: str, page_count: int) -> None:
        normalized = (document_type or "").strip().upper()
        if normalized in {"PATIENT_CHART", "PATIENT"}:
            self.patient_ocr_page_count += max(0, int(page_count))
        elif normalized == "POLICY":
            self.policy_ocr_page_count += max(0, int(page_count))

    def record_policy_tree_counts(
        self,
        *,
        group_nodes: int,
        condition_nodes: int,
        exclusions: int,
    ) -> None:
        self.policy_sol_group_nodes = int(group_nodes)
        self.policy_sol_condition_nodes = int(condition_nodes)
        self.policy_sol_exclusions = int(exclusions)
        self.policy_sol_total_nodes = int(group_nodes + condition_nodes + exclusions)

    def record_evidence_sol_wall(
        self,
        *,
        wall_ms: int,
        batch_durations_ms: list[int],
    ) -> None:
        self.evidence_sol_wall_ms = max(0, int(wall_ms))
        self.set_substage_ms("evidence_sol_wall_ms", wall_ms)
        if batch_durations_ms:
            self.evidence_sol_batch_durations_ms = [int(v) for v in batch_durations_ms]
        summed = sum(self.evidence_sol_batch_durations_ms)
        if len(self.evidence_sol_batch_durations_ms) >= 2 and summed > 0:
            # Concurrent when wall is closer to max than to sum.
            max_batch = max(self.evidence_sol_batch_durations_ms)
            self.evidence_sol_concurrent = (
                abs(self.evidence_sol_wall_ms - max_batch)
                <= abs(self.evidence_sol_wall_ms - summed)
            )
        elif len(self.evidence_sol_batch_durations_ms) <= 1:
            self.evidence_sol_concurrent = None

    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)

    @property
    def llm_tokens_total(self) -> int:
        return sum(item.total_tokens for item in self.model_usage.values())

    @property
    def policy_llm_ms(self) -> int:
        return int(self.substage_ms.get("policy_sol_ms", 0))

    @property
    def policy_local_ms(self) -> int:
        policy_total = int(self.stage_ms.get("policy", 0))
        return max(0, policy_total - self.policy_llm_ms)

    @property
    def evidence_llm_wall_ms(self) -> int:
        if self.evidence_sol_wall_ms:
            return int(self.evidence_sol_wall_ms)
        return int(self.substage_ms.get("evidence_sol_wall_ms", 0))

    @property
    def evidence_local_ms(self) -> int:
        evidence_total = int(self.stage_ms.get("evidence", 0))
        return max(0, evidence_total - self.evidence_llm_wall_ms)

    def leaf_substage_rows(self) -> list[tuple[str, int]]:
        """Non-overlapping leaf substages sorted by duration descending."""
        # Concurrent Sol batch walls are reported in section H; only the gather
        # wall belongs in the leaf % table to avoid double-counting.
        excluded = {
            name
            for name in self.substage_ms
            if name.startswith("evidence_sol_batch_") and name.endswith("_ms")
        }
        rows = [
            (name, int(self.substage_ms.get(name, 0)))
            for name in _LEAF_SUBSTAGES
            if int(self.substage_ms.get(name, 0)) > 0 and name not in excluded
        ]
        known = set(_LEAF_SUBSTAGES) | excluded
        for name, value in self.substage_ms.items():
            if name not in known and int(value) > 0:
                rows.append((name, int(value)))
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows

    def format_performance_report(self) -> str:
        total = max(1, self.total_ms)
        lines = [
            "SUBSTAGE                      MS       % TOTAL",
            "------------------------------------------------",
        ]
        for name, ms in self.leaf_substage_rows():
            pct = (100.0 * ms) / total
            lines.append(f"{name:<30} {ms:>8}  {pct:6.2f}%")

        lines.append("")
        lines.append(
            f"policy_local_ms={self.policy_local_ms}  "
            f"policy_llm_ms={self.policy_llm_ms}"
        )
        lines.append(
            f"evidence_local_ms={self.evidence_local_ms}  "
            f"evidence_llm_wall_ms={self.evidence_llm_wall_ms}"
        )

        remote_llm_ms = (
            int(self.substage_ms.get("procedure_luna_ms", 0))
            + self.policy_llm_ms
            + self.evidence_llm_wall_ms
        )
        local_model_ms = (
            self.embedding_total_ms
            + self.reranker.inference_ms
            + self.reranker.preprocessing_ms
        )
        pdf_parse_ms = (
            int(self.substage_ms.get("patient_pdf_parse_ms", 0))
            + int(self.substage_ms.get("patient_page_text_extract_ms", 0))
            + int(self.substage_ms.get("policy_pdf_parse_ms", 0))
            + int(self.substage_ms.get("policy_page_text_extract_ms", 0))
        )

        top5 = self.leaf_substage_rows()[:5]
        lines.append("")
        lines.append("A. top 5 wall-clock bottlenecks:")
        for index, (name, ms) in enumerate(top5, start=1):
            lines.append(f"   {index}. {name}={ms}ms ({100.0 * ms / total:.2f}%)")

        lines.append(f"B. remote LLM total wall time: {remote_llm_ms}ms")
        lines.append(f"C. local model total wall time: {local_model_ms}ms")
        lines.append(f"D. PDF parsing total wall time: {pdf_parse_ms}ms")
        lines.append(
            "E. reranker: "
            f"attempted={self.reranker.attempted_pairs} "
            f"unique={self.reranker.unique_pairs} "
            f"duplicate={self.reranker.duplicate_pairs} "
            f"inference_ms={self.reranker.inference_ms} "
            f"preprocessing_ms={self.reranker.preprocessing_ms} "
            f"procedure_pairs={self.reranker.procedure_pairs} "
            f"policy_pairs={self.reranker.policy_pairs} "
            f"evidence_pairs={self.reranker.evidence_pairs} "
            f"batches={self.reranker.batches} "
            f"batch_sizes={self.reranker.batch_sizes}"
        )
        lines.append(
            "F. embeddings: "
            f"requested={self.embedding_total_text_requests} "
            f"unique={self.embedding_unique_texts} "
            f"duplicate_encodes={self.embedding_duplicate_requests} "
            f"total_encode_ms={self.embedding_total_ms}"
        )
        lines.append(
            "G. parser counts: "
            f"patient={self.patient_pdf_parse_count} "
            f"policy={self.policy_pdf_parse_count}; "
            f"OCR pages patient={self.patient_ocr_page_count} "
            f"policy={self.policy_ocr_page_count}"
        )
        summed_batches = sum(self.evidence_sol_batch_durations_ms)
        lines.append(
            "H. evidence concurrency: "
            f"batch_durations_ms={self.evidence_sol_batch_durations_ms} "
            f"summed={summed_batches} "
            f"wall={self.evidence_sol_wall_ms} "
            f"concurrent={self.evidence_sol_concurrent}"
        )
        policy_call = next(
            (call for call in self.llm_calls if call.operation == "policy_criteria"),
            None,
        )
        if policy_call is not None:
            lines.append(
                "I. policy Sol: "
                f"input={policy_call.input_tokens} "
                f"output={policy_call.output_tokens} "
                f"reasoning={policy_call.reasoning_tokens} "
                f"duration_ms={policy_call.duration_ms} "
                f"group_nodes={self.policy_sol_group_nodes} "
                f"condition_nodes={self.policy_sol_condition_nodes} "
                f"exclusions={self.policy_sol_exclusions} "
                f"total_nodes={self.policy_sol_total_nodes}"
            )
        else:
            lines.append(
                "I. policy Sol: "
                f"group_nodes={self.policy_sol_group_nodes} "
                f"condition_nodes={self.policy_sol_condition_nodes} "
                f"exclusions={self.policy_sol_exclusions} "
                f"total_nodes={self.policy_sol_total_nodes}"
            )
        return "\n".join(lines)

    def summary_fields(self, *, include_profile: bool | None = None) -> dict[str, Any]:
        """Build PHI-free structured fields for analysis_request_summary.

        Lightweight request metrics are always included. Detailed substage,
        embedding/reranker pair stats, and per-call dumps are included only when
        PERFORMANCE_PROFILE_ENABLED is true (or include_profile=True).
        """
        profile = (
            is_performance_profile_enabled()
            if include_profile is None
            else include_profile
        )
        model_usage = {
            name: {
                "calls": usage.calls,
                "failed_calls": usage.failed_calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "total_tokens": usage.total_tokens,
            }
            for name, usage in sorted(self.model_usage.items())
        }
        local_usage: dict[str, dict[str, int]] = {}
        for name, usage in sorted(self.local_model_usage.items()):
            payload: dict[str, int] = {"batches": usage.batches}
            if usage.texts:
                payload["texts"] = usage.texts
            if usage.pairs:
                payload["pairs"] = usage.pairs
            if usage.tokens is not None:
                payload["tokens"] = usage.tokens
            local_usage[name] = payload

        fields: dict[str, Any] = {
            "request_id": self.request_id,
            "status": self.status or "UNKNOWN",
            "total_ms": self.total_ms,
            "stage_ms": {
                "procedure": self.stage_ms.get("procedure", 0),
                "policy": self.stage_ms.get("policy", 0),
                "evidence": self.stage_ms.get("evidence", 0),
                "readiness": self.stage_ms.get("readiness", 0),
            },
            "patient_pages": self.patient_pages,
            "policy_pages": self.policy_pages,
            "criteria_count": self.criteria_count,
            "llm_tokens_total": self.llm_tokens_total,
            "model_usage": model_usage,
            "local_model_usage": local_usage,
        }
        if self.error_type:
            fields["error_type"] = self.error_type

        if not profile:
            return fields

        fields.update(
            {
                "substage_ms": dict(sorted(self.substage_ms.items())),
                "policy_local_ms": self.policy_local_ms,
                "policy_llm_ms": self.policy_llm_ms,
                "evidence_local_ms": self.evidence_local_ms,
                "evidence_llm_wall_ms": self.evidence_llm_wall_ms,
                "patient_pdf_parse_count": self.patient_pdf_parse_count,
                "policy_pdf_parse_count": self.policy_pdf_parse_count,
                "patient_ocr_page_count": self.patient_ocr_page_count,
                "policy_ocr_page_count": self.policy_ocr_page_count,
                "embedding_total_text_requests": self.embedding_total_text_requests,
                "embedding_unique_texts": self.embedding_unique_texts,
                "embedding_duplicate_requests": self.embedding_duplicate_requests,
                "embedding_total_ms": self.embedding_total_ms,
                "reranker": {
                    "calls": self.reranker.calls,
                    "batches": self.reranker.batches,
                    "batch_sizes": list(self.reranker.batch_sizes),
                    "attempted_pairs": self.reranker.attempted_pairs,
                    "unique_pairs": self.reranker.unique_pairs,
                    "duplicate_pairs": self.reranker.duplicate_pairs,
                    "inference_ms": self.reranker.inference_ms,
                    "preprocessing_ms": self.reranker.preprocessing_ms,
                    "procedure_pairs": self.reranker.procedure_pairs,
                    "policy_pairs": self.reranker.policy_pairs,
                    "evidence_pairs": self.reranker.evidence_pairs,
                },
                "evidence_sol_batch_durations_ms": list(
                    self.evidence_sol_batch_durations_ms
                ),
                "evidence_sol_wall_ms": self.evidence_sol_wall_ms,
                "evidence_sol_concurrent": self.evidence_sol_concurrent,
                "policy_sol_group_nodes": self.policy_sol_group_nodes,
                "policy_sol_condition_nodes": self.policy_sol_condition_nodes,
                "policy_sol_exclusions": self.policy_sol_exclusions,
                "policy_sol_total_nodes": self.policy_sol_total_nodes,
                "llm_calls": [
                    {
                        "operation": call.operation,
                        "model": call.model,
                        "input_tokens": call.input_tokens,
                        "output_tokens": call.output_tokens,
                        "reasoning_tokens": call.reasoning_tokens,
                        "total_tokens": call.total_tokens,
                        "duration_ms": call.duration_ms,
                        "context_items": call.context_items,
                        "item_count": call.item_count,
                        "unique_evidence_chunks": call.unique_evidence_chunks,
                        "batch_index": call.batch_index,
                    }
                    for call in self.llm_calls
                ],
            }
        )
        return fields

    def emit_summary(self) -> None:
        """Emit exactly one structured analysis_request_summary event.

        Detailed substage / pair-duplication reports print only when
        PERFORMANCE_PROFILE_ENABLED=true.
        """
        profile = is_performance_profile_enabled()
        logger.info(
            "analysis_request_summary",
            **self.summary_fields(include_profile=profile),
        )
        if not profile:
            return
        report = self.format_performance_report()
        logger.info("analysis_performance_report", report=report)
        with contextlib.suppress(ValueError, OSError):
            # Tests may close stdout; never fail analysis for report printing.
            print("\n" + report + "\n", flush=True)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pair_identity(query_text: str, document_text: str) -> str:
    query_hash = _text_hash(query_text)
    document_hash = _text_hash(document_text)
    return f"{query_hash}:{document_hash}"


_request_metrics: ContextVar[RequestMetrics | None] = ContextVar(
    "request_metrics",
    default=None,
)
_preprocess_doc_kind: ContextVar[str | None] = ContextVar(
    "preprocess_doc_kind",
    default=None,
)
_embedding_operation: ContextVar[str | None] = ContextVar(
    "embedding_operation",
    default=None,
)
_rerank_stage: ContextVar[str | None] = ContextVar(
    "rerank_stage",
    default=None,
)


def get_request_metrics() -> RequestMetrics | None:
    return _request_metrics.get()


def bind_request_metrics(metrics: RequestMetrics) -> Token[RequestMetrics | None]:
    return _request_metrics.set(metrics)


def reset_request_metrics(token: Token[RequestMetrics | None]) -> None:
    _request_metrics.reset(token)


def set_preprocess_doc_kind(kind: str | None) -> Token[str | None]:
    return _preprocess_doc_kind.set(kind)


def get_preprocess_doc_kind() -> str | None:
    return _preprocess_doc_kind.get()


def reset_preprocess_doc_kind(token: Token[str | None]) -> None:
    _preprocess_doc_kind.reset(token)


@contextmanager
def embedding_operation(operation: str) -> Iterator[None]:
    token = _embedding_operation.set(operation)
    try:
        yield
    finally:
        _embedding_operation.reset(token)


def get_embedding_operation() -> str | None:
    return _embedding_operation.get()


@contextmanager
def rerank_stage(stage: str) -> Iterator[None]:
    token = _rerank_stage.set(stage)
    try:
        yield
    finally:
        _rerank_stage.reset(token)


def get_rerank_stage() -> str | None:
    return _rerank_stage.get()


@contextmanager
def timed_substage(name: str) -> Iterator[None]:
    """Accumulate wall-clock ms for a named leaf substage."""
    metrics = get_request_metrics()
    started = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        if metrics is not None:
            metrics.add_substage_ms(name, duration_ms)

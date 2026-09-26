"""Cross-encoder reranking with a configured BGE reranker model."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import RerankingError
from app.core.logging import get_logger
from app.core.metrics import get_request_metrics, get_rerank_stage
from app.models.retrieval import RetrievalResult

logger = get_logger(__name__)


class RerankingService:
    """Rank query-document pairs with a lazy-loaded cross-encoder."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any | None = None
        self.init_count: int = 0

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        *,
        stage: str | None = None,
    ) -> list[RetrievalResult]:
        grouped = self.rerank_grouped(
            [("default", query, candidates)],
            top_k=top_k,
            stage=stage,
        )
        return grouped.get("default", [])

    def rerank_grouped(
        self,
        groups: Sequence[tuple[str, str, list[RetrievalResult]]],
        top_k: int | None = None,
        *,
        stage: str | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Score criterion/chunk pairs in model batches, then regroup by id.

        ``groups`` items are ``(group_id, query, candidates)``. Isolation is
        preserved: each group only receives scores for its own pairs.
        """
        selected_top_k = top_k if top_k is not None else self._settings.RERANK_TOP_K
        if selected_top_k <= 0:
            return {group_id: [] for group_id, _, _ in groups}

        started = time.perf_counter()
        try:
            prep_started = time.perf_counter()
            pair_meta: list[tuple[str, int]] = []
            pairs: list[list[str]] = []
            for group_id, query, candidates in groups:
                for index, candidate in enumerate(candidates):
                    pairs.append([query, self._candidate_text(candidate)])
                    pair_meta.append((group_id, index))
            preprocessing_ms = int((time.perf_counter() - prep_started) * 1000)

            if not pairs:
                return {group_id: [] for group_id, _, _ in groups}

            inference_started = time.perf_counter()
            scores, batch_sizes = self._score_pairs(pairs)
            inference_ms = int((time.perf_counter() - inference_started) * 1000)
            if len(scores) != len(pairs):
                raise RerankingError("Reranker returned an unexpected score count")

            by_group: dict[str, list[tuple[int, float]]] = {
                group_id: [] for group_id, _, _ in groups
            }
            for meta, score in zip(pair_meta, scores, strict=True):
                group_id, original_index = meta
                by_group.setdefault(group_id, []).append(
                    (original_index, float(score))
                )

            candidates_by_id = {
                group_id: candidates for group_id, _, candidates in groups
            }
            result: dict[str, list[RetrievalResult]] = {}
            for group_id, scored in by_group.items():
                scored.sort(key=lambda item: (-item[1], item[0]))
                candidates = candidates_by_id.get(group_id, [])
                reranked: list[RetrievalResult] = []
                for new_rank, (original_index, score) in enumerate(
                    scored[:selected_top_k],
                    start=1,
                ):
                    if original_index >= len(candidates):
                        continue
                    original = candidates[original_index]
                    reranked.append(
                        RetrievalResult(
                            chunk=original.chunk,
                            rank=new_rank,
                            score=float(score),
                            retrieval_method="rerank",
                            dense_rank=original.dense_rank,
                            bm25_rank=original.bm25_rank,
                            rrf_score=original.rrf_score,
                            reranker_score=float(score),
                        )
                    )
                result[group_id] = reranked

            resolved_stage = stage or get_rerank_stage() or "unknown"
            metrics = get_request_metrics()
            if metrics is not None:
                pair_tuples = [(pair[0], pair[1]) for pair in pairs]
                metrics.reranker.record_pairs(
                    stage=resolved_stage,
                    pairs=pair_tuples,
                    batch_sizes=batch_sizes,
                    preprocessing_ms=preprocessing_ms,
                    inference_ms=inference_ms,
                )
                if resolved_stage == "evidence":
                    metrics.add_substage_ms(
                        "evidence_reranker_prepare_ms", preprocessing_ms
                    )
                    metrics.add_substage_ms(
                        "evidence_reranker_inference_ms", inference_ms
                    )
                elif resolved_stage == "procedure":
                    metrics.add_substage_ms(
                        "procedure_rerank_ms", preprocessing_ms + inference_ms
                    )
                elif resolved_stage == "policy":
                    metrics.add_substage_ms(
                        "policy_rerank_ms", preprocessing_ms + inference_ms
                    )

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "reranking_completed",
                group_count=len(groups),
                pair_count=len(pairs),
                unique_pair_count=(
                    metrics.reranker.unique_pairs if metrics is not None else None
                ),
                duplicate_pair_count=(
                    metrics.reranker.duplicate_pairs if metrics is not None else None
                ),
                preprocessing_ms=preprocessing_ms,
                inference_ms=inference_ms,
                reranking_duration_ms=elapsed_ms,
                stage=resolved_stage,
            )
            return result
        except RerankingError:
            raise
        except Exception as exc:
            logger.error("reranking_failed", error=str(exc))
            raise RerankingError("Failed to rerank retrieval candidates") from exc

    def _candidate_text(self, result: RetrievalResult) -> str:
        chunk = result.chunk
        parts: list[str] = []
        if chunk.document_date:
            parts.append(f"Encounter date: {chunk.document_date}")
        if chunk.heading:
            parts.append(f"Section: {chunk.heading}")
        elif chunk.parent_heading:
            parts.append(f"Section: {chunk.parent_heading}")
        if chunk.heading and chunk.parent_heading:
            parts.append(f"Parent: {chunk.parent_heading}")
        parts.append("Evidence:")
        parts.append(chunk.text)
        return "\n".join(parts)

    def _score_pairs(self, pairs: list[list[str]]) -> tuple[list[float], list[int]]:
        model = self._get_model()
        batch_size = max(1, self._settings.EMBEDDING_BATCH_SIZE)
        scores: list[float] = []
        batch_sizes: list[int] = []
        batch_count = 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            raw_scores = model.compute_score(batch)
            batch_count += 1
            batch_sizes.append(len(batch))
            if isinstance(raw_scores, (float, int)):
                scores.append(float(raw_scores))
            else:
                scores.extend(float(score) for score in raw_scores)

        metrics = get_request_metrics()
        if metrics is not None and pairs:
            metrics.record_reranker_batch(
                self._settings.RERANKER_MODEL,
                batches=batch_count,
                pairs=len(pairs),
            )
        return scores, batch_sizes

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        model_name = self._settings.RERANKER_MODEL
        try:
            from FlagEmbedding import FlagReranker

            logger.info("reranker_model_loading", model=model_name)
            # FlagReranker fp16 is for CUDA; keep False on CPU.
            self._model = FlagReranker(model_name, use_fp16=False)
            self.init_count += 1
            logger.info("reranker_model_loaded", model=model_name)
            return self._model
        except Exception as exc:
            logger.error(
                "reranker_model_load_failed",
                model=model_name,
                error=str(exc),
            )
            raise RerankingError(
                f"Failed to load reranker model: {model_name}"
            ) from exc

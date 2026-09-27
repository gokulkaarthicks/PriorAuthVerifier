"""Retrieval helpers for procedure, policy, and evidence pipelines.

Procedure: heading-only BGE, then BM25+dense+RRF over selected-section chunks.
Policy: exact CPT + BM25 over all sections, then dense over the candidate union.
Evidence: shared request-scoped section BM25, deduped chunking, batched dense.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

import numpy as np

from app.core.config import Settings, get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.core.metrics import get_request_metrics
from app.models.chunk import Chunk
from app.models.retrieval import HybridIndex, RetrievalResult, SectionBm25Index
from app.models.section import Section
from app.schemas.procedure import ProcedureExtraction
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*", re.IGNORECASE)

_PROCEDURE_HEADING_QUERY = (
    "planned procedure, recommended intervention, scheduled procedure, "
    "treatment plan, next procedural step"
)

_PROCEDURE_CHUNK_QUERY = (
    "planned future procedure, recommended intervention, scheduled procedure, "
    "treatment/procedural plan, procedure to be performed"
)

_PROCEDURE_RERANK_QUERY = (
    "What procedure is planned, recommended, or scheduled for this patient?"
)

_POLICY_DENSE_SUFFIX = (
    "Medical necessity indications, coverage criteria, requirements, "
    "and eligibility for this procedure."
)

_POLICY_RERANK_TEMPLATE = (
    "Which policy section directly contains the medical-necessity criteria, "
    "indications, requirements, or coverage rules governing this exact "
    "planned procedure: {procedure}?"
)

# Structural markers for document-local policy continuation expansion.
_CRID_RE = re.compile(r"\bCRID[-\s]?(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"^references?\b", re.IGNORECASE)
_MAX_CONTINUATION_PAGE_GAP = 3
_MAX_CONTINUATION_SECTIONS = 12


class RetrievalService:
    """Section narrowing, dense ranking, and procedure chunk hybrid retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._settings = settings or get_settings()

    def build_hybrid_index(
        self,
        chunks: list[Chunk],
        *,
        index_kind: str = "selected",
        profile_prefix: str | None = None,
    ) -> HybridIndex:
        """Build BM25 + dense embeddings for an already-narrowed chunk set."""
        if not chunks:
            raise RetrievalError(
                f"Cannot build a {index_kind} index from an empty chunk list"
            )

        started = time.perf_counter()
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RetrievalError("Failed to import rank_bm25") from exc

        try:
            bm25_started = time.perf_counter()
            retrieval_texts = [self._build_retrieval_text(chunk) for chunk in chunks]
            tokenized_corpus = [self.tokenize(text) for text in retrieval_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_ms = int((time.perf_counter() - bm25_started) * 1000)

            embed_started = time.perf_counter()
            embeddings = self._embedding_service.embed(
                retrieval_texts,
                operation=f"{profile_prefix or index_kind}_chunk_embedding",
            )
            embed_ms = int((time.perf_counter() - embed_started) * 1000)
            if embeddings.shape[0] != len(chunks):
                raise RetrievalError(
                    f"Embedding count does not match {index_kind} chunk count"
                )

            metrics = get_request_metrics()
            if metrics is not None and profile_prefix == "procedure":
                metrics.add_substage_ms("procedure_bm25_ms", bm25_ms)
                metrics.add_substage_ms("procedure_chunk_embedding_ms", embed_ms)

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "procedure_chunk_index_built",
                index_kind=index_kind,
                candidate_chunks=len(chunks),
                indexing_duration_ms=elapsed_ms,
                bm25_duration_ms=bm25_ms,
                embedding_duration_ms=embed_ms,
                embedding_dim=int(embeddings.shape[1]) if embeddings.size else 0,
            )
            return HybridIndex(
                chunks=chunks,
                embeddings=embeddings,
                bm25=bm25,
                retrieval_texts=retrieval_texts,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            logger.error(
                "procedure_chunk_index_failed",
                index_kind=index_kind,
                error=str(exc),
            )
            raise RetrievalError(f"Failed to build {index_kind} index") from exc

    def retrieve_for_query(
        self,
        query: str,
        index: HybridIndex,
        *,
        log_prefix: str = "procedure_chunk",
        profile_prefix: str | None = None,
    ) -> list[RetrievalResult]:
        """BM25 + dense + RRF over a narrowed hybrid index (procedure path only)."""
        if not query.strip() or not index.chunks:
            return []

        bm25_started = time.perf_counter()
        bm25_results = self._bm25_search(query, index)
        bm25_ms = int((time.perf_counter() - bm25_started) * 1000)

        dense_started = time.perf_counter()
        dense_results = self._dense_search(
            query,
            index,
            operation=f"{profile_prefix or 'procedure'}_dense_query",
        )
        dense_ms = int((time.perf_counter() - dense_started) * 1000)

        fusion_started = time.perf_counter()
        fused = self._rrf_fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
            chunks=index.chunks,
        )
        fusion_ms = int((time.perf_counter() - fusion_started) * 1000)

        metrics = get_request_metrics()
        if metrics is not None and profile_prefix == "procedure":
            # Index build already recorded corpus BM25; this is query BM25.
            metrics.add_substage_ms("procedure_bm25_ms", bm25_ms)
            metrics.add_substage_ms("procedure_dense_ms", dense_ms)

        logger.info(
            f"{log_prefix}_retrieval_completed",
            query_present=True,
            bm25_candidate_count=len(bm25_results),
            dense_candidate_count=len(dense_results),
            rrf_candidate_count=len(fused),
            bm25_duration_ms=bm25_ms,
            dense_duration_ms=dense_ms,
            fusion_duration_ms=fusion_ms,
        )
        return fused

    def bm25_select_sections(
        self,
        sections: Sequence[Section],
        *,
        query: str,
        top_k: int | None = None,
        profile_prefix: str | None = None,
    ) -> list[Section]:
        """BM25 top-K over compact representations of the full section corpus."""
        index_started = time.perf_counter()
        index = self.build_section_bm25_index(sections)
        index_ms = int((time.perf_counter() - index_started) * 1000)
        search_started = time.perf_counter()
        selected = self.query_section_bm25(
            index,
            query=query,
            top_k=top_k if top_k is not None else self._settings.POLICY_BM25_TOP_K,
        )
        search_ms = int((time.perf_counter() - search_started) * 1000)
        metrics = get_request_metrics()
        if metrics is not None and profile_prefix == "policy":
            metrics.add_substage_ms("policy_bm25_index_build_ms", index_ms)
            metrics.add_substage_ms("policy_bm25_search_ms", search_ms)
        return selected

    def build_section_bm25_index(
        self,
        sections: Sequence[Section],
    ) -> SectionBm25Index:
        """Build a request-scoped BM25 index over compact section texts."""
        usable = [section for section in sections if section.text.strip()]
        if not usable:
            return SectionBm25Index(sections=[], bm25=None)

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RetrievalError("Failed to import rank_bm25") from exc

        compact_texts = [self.compact_section_text(section) for section in usable]
        tokenized = [self.tokenize(text) for text in compact_texts]
        if not any(tokenized):
            return SectionBm25Index(sections=usable, bm25=None)
        return SectionBm25Index(sections=usable, bm25=BM25Okapi(tokenized))

    def query_section_bm25(
        self,
        index: SectionBm25Index,
        *,
        query: str,
        top_k: int | None = None,
    ) -> list[Section]:
        """Query a request-scoped section BM25 index."""
        if not index.sections:
            return []

        selected_top_k = top_k if top_k is not None else len(index.sections)
        selected_top_k = min(selected_top_k, len(index.sections))
        if selected_top_k <= 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens or index.bm25 is None:
            return index.sections[:selected_top_k]

        scores = np.asarray(index.bm25.get_scores(query_tokens), dtype=np.float64)
        ranked = np.argsort(-scores, kind="stable")
        return [index.sections[int(index_pos)] for index_pos in ranked[:selected_top_k]]

    def dense_rank_sections(
        self,
        sections: Sequence[Section],
        *,
        query: str,
        top_k: int | None = None,
        profile_prefix: str | None = None,
    ) -> list[Section]:
        """BGE-M3 dense ranking over a pre-narrowed section candidate set only."""
        usable = [section for section in sections if section.text.strip()]
        if not usable:
            return []

        selected_top_k = (
            top_k if top_k is not None else self._settings.POLICY_DENSE_TOP_K
        )
        selected_top_k = min(selected_top_k, len(usable))
        if selected_top_k <= 0:
            return []

        retrieval_texts = [
            self.policy_section_retrieval_text(section) for section in usable
        ]
        dense_ranked = self._rank_texts_dense(
            query,
            retrieval_texts,
            profile_prefix=profile_prefix,
        )
        if not dense_ranked:
            return usable[:selected_top_k]
        return [usable[index] for index in dense_ranked[:selected_top_k]]

    def dense_rank_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """BGE-M3 dense ranking over an already-narrowed chunk set (no BM25/RRF)."""
        grouped = self.dense_rank_chunks_batched(
            {"default": (query, list(chunks))},
            top_k=top_k,
        )
        return grouped.get("default", [])

    def dense_rank_chunks_batched(
        self,
        groups: dict[str, tuple[str, list[Chunk]]],
        *,
        top_k: int | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Encode unique queries/chunks once, then score only allowed pairs.

        ``groups`` maps criterion_id → (query, allowed_chunks). Criterion
        isolation is preserved: a chunk is scored for a criterion only when it
        appears in that criterion's allowed set.
        """
        selected_top_k = (
            top_k if top_k is not None else self._settings.EVIDENCE_DENSE_TOP_K
        )
        if selected_top_k <= 0 or not groups:
            return {group_id: [] for group_id in groups}

        unique_chunks: dict[str, Chunk] = {}
        for _, chunks in groups.values():
            for chunk in chunks:
                if chunk.text.strip() and chunk.id not in unique_chunks:
                    unique_chunks[chunk.id] = chunk

        if not unique_chunks:
            return {group_id: [] for group_id in groups}

        chunk_ids = list(unique_chunks.keys())
        chunk_texts = [
            self._build_retrieval_text(unique_chunks[chunk_id])
            for chunk_id in chunk_ids
        ]
        patient_embed_started = time.perf_counter()
        chunk_embeddings = self._embedding_service.embed(
            chunk_texts,
            operation="evidence_patient_embedding",
        )
        patient_embed_ms = int((time.perf_counter() - patient_embed_started) * 1000)
        chunk_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}

        query_texts: list[str] = []
        query_keys: list[str] = []
        for group_id, (query, _) in groups.items():
            cleaned = query.strip()
            if cleaned:
                query_keys.append(group_id)
                query_texts.append(cleaned)

        query_embeddings_by_id: dict[str, np.ndarray] = {}
        criterion_embed_ms = 0
        if query_texts:
            criterion_embed_started = time.perf_counter()
            query_matrix = self._embedding_service.embed(
                query_texts,
                operation="evidence_criterion_embedding",
            )
            criterion_embed_ms = int(
                (time.perf_counter() - criterion_embed_started) * 1000
            )
            for group_id, embedding in zip(query_keys, query_matrix, strict=True):
                query_embeddings_by_id[group_id] = embedding

        score_started = time.perf_counter()
        results: dict[str, list[RetrievalResult]] = {}
        for group_id, (query, chunks) in groups.items():
            usable = [chunk for chunk in chunks if chunk.text.strip()]
            query_embedding = query_embeddings_by_id.get(group_id)
            if not usable or query_embedding is None or not query.strip():
                results[group_id] = []
                continue

            doc_rows: list[np.ndarray] = []
            ordered_chunks: list[Chunk] = []
            for chunk in usable:
                index = chunk_index.get(chunk.id)
                if index is None:
                    continue
                doc_rows.append(chunk_embeddings[index])
                ordered_chunks.append(chunk)
            if not ordered_chunks:
                results[group_id] = []
                continue

            docs = np.vstack(doc_rows)
            scores = self._embedding_service.similarity(query_embedding, docs)
            ranked = np.argsort(-scores, kind="stable")
            keep = min(selected_top_k, len(ordered_chunks))
            group_results: list[RetrievalResult] = []
            for rank, index in enumerate(ranked[:keep], start=1):
                group_results.append(
                    RetrievalResult(
                        chunk=ordered_chunks[int(index)],
                        rank=rank,
                        score=float(scores[int(index)]),
                        retrieval_method="dense",
                        dense_rank=rank,
                    )
                )
            results[group_id] = group_results
        score_ms = int((time.perf_counter() - score_started) * 1000)

        metrics = get_request_metrics()
        if metrics is not None:
            metrics.add_substage_ms("evidence_patient_embedding_ms", patient_embed_ms)
            metrics.add_substage_ms(
                "evidence_criterion_embedding_ms", criterion_embed_ms
            )
            metrics.add_substage_ms("evidence_dense_scoring_ms", score_ms)
        return results

    def union_sections(
        self,
        *groups: Sequence[Section],
    ) -> list[Section]:
        """Deterministic dedupe by section id, preserving first-seen order."""
        merged: list[Section] = []
        seen: set[str] = set()
        for group in groups:
            for section in group:
                if section.id in seen:
                    continue
                seen.add(section.id)
                merged.append(section)
        return merged

    def select_sections_by_heading_similarity(
        self,
        sections: Sequence[Section],
        *,
        query: str,
        top_k: int | None = None,
        profile_prefix: str | None = None,
    ) -> list[Section]:
        """Select sections by BGE-M3 similarity over heading strings only.

        Dense/BGE-M3 never sees parent headings, body previews, or section
        bodies at this stage; only the heading text.
        """
        headed = [
            section
            for section in sections
            if (section.heading or "").strip() and section.text.strip()
        ]
        if not headed:
            logger.info(
                "patient_heading_retrieval_completed",
                headings=0,
                selected_sections=0,
            )
            return []

        selected_top_k = (
            top_k if top_k is not None else self._settings.PATIENT_HEADING_TOP_K
        )
        selected_top_k = min(selected_top_k, len(headed))
        if selected_top_k <= 0:
            return []

        headings = [(section.heading or "").strip() for section in headed]
        dense_ranked = self._rank_texts_dense(
            query,
            headings,
            profile_prefix=profile_prefix,
        )
        if not dense_ranked:
            selected = headed[:selected_top_k]
        else:
            selected = [headed[index] for index in dense_ranked[:selected_top_k]]

        logger.info(
            "patient_heading_retrieval_completed",
            headings=len(headed),
            selected_sections=len(selected),
        )
        return selected

    def find_sections_with_cpt(
        self,
        sections: Sequence[Section],
        cpt_code: str,
    ) -> list[Section]:
        """Deterministic exact CPT match over section text (cheap lexical signal)."""
        cpt = (cpt_code or "").strip()
        if not cpt:
            return []
        pattern = re.compile(rf"(?<!\d){re.escape(cpt)}(?!\d)")
        hits = [
            section
            for section in sections
            if section.text and pattern.search(section.text)
        ]
        logger.info(
            "policy_cpt_search_completed",
            cpt=cpt,
            hits=len(hits),
        )
        return hits

    def expand_section_context(
        self,
        selected: Sequence[Section],
        all_sections: Sequence[Section],
        *,
        neighbor_window: int | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[Section]:
        """Expand document-local continuation around governing sections.

        Expansion is structural and ordered by section position, never by
        global heading / ``parent_heading`` string equality. Contiguous
        continuation sections (including repeated running-header wrappers)
        stay attached until a peer policy topic, References, or different
        CRID boundary. ``exclude_ids`` still blocks non-continuation neighbor
        padding, but does not sever local structural continuation.
        """
        if not selected:
            return []

        window = (
            neighbor_window
            if neighbor_window is not None
            else self._settings.SECTION_NEIGHBOR_WINDOW
        )
        blocked = exclude_ids or set()
        by_id = {section.id: index for index, section in enumerate(all_sections)}
        heading_counts = self._heading_frequency(all_sections)
        keep: set[int] = set()

        for section in selected:
            index = by_id.get(section.id)
            if index is None:
                continue
            keep.add(index)
            # Prefer an explicit CRID on the selected section; otherwise discover
            # the governing CRID of the contiguous procedure topic block ahead
            # (e.g. topic title → "(CRID-7)" → Codes included → Indications).
            anchor_crid = self._section_crid(section) or self._discover_forward_crid(
                section,
                all_sections,
                start_index=index,
                heading_counts=heading_counts,
            )

            # Immediate neighbor padding (respect exclude_ids and policy
            # boundaries so adjacent unrelated CRID blocks are not pulled in).
            for offset in range(1, window + 1):
                if index - offset >= 0:
                    neighbor = all_sections[index - offset]
                    if neighbor.id not in blocked and not self._is_policy_boundary(
                        neighbor,
                        selected=section,
                        anchor_crid=anchor_crid,
                        heading_counts=heading_counts,
                    ):
                        keep.add(index - offset)
                if index + offset < len(all_sections):
                    neighbor = all_sections[index + offset]
                    if neighbor.id not in blocked and not self._is_policy_boundary(
                        neighbor,
                        selected=section,
                        anchor_crid=anchor_crid,
                        heading_counts=heading_counts,
                    ):
                        keep.add(index + offset)

            # Forward structural continuation (overrides exclude_ids).
            for offset, forward_index in enumerate(
                range(index + 1, len(all_sections)),
                start=1,
            ):
                if offset > _MAX_CONTINUATION_SECTIONS:
                    break
                candidate = all_sections[forward_index]
                if not self._within_page_proximity(section, candidate):
                    break
                if self._is_policy_boundary(
                    candidate,
                    selected=section,
                    anchor_crid=anchor_crid,
                    heading_counts=heading_counts,
                ):
                    break
                keep.add(forward_index)

            # Backward: local parent + running-header wrappers only.
            for backward_index in range(index - 1, -1, -1):
                candidate = all_sections[backward_index]
                if not self._within_page_proximity(section, candidate):
                    break
                if self._is_running_header(candidate, heading_counts):
                    keep.add(backward_index)
                    continue
                parent = (section.parent_heading or "").strip()
                if parent and (candidate.heading or "").strip() == parent:
                    keep.add(backward_index)
                break

        return [all_sections[index] for index in sorted(keep)]

    def filter_governing_seeds_for_procedure(
        self,
        governing_sections: Sequence[Section],
        all_sections: Sequence[Section],
        *,
        procedure: ProcedureExtraction,
        cpt_section_ids: set[str] | frozenset[str],
    ) -> list[Section]:
        """Keep governing seeds for the detected procedure block only.

        Once a primary procedure CRID/topic is established (from procedure-name
        matches and CPT identity sections), drop unrelated top-K seeds that
        open a different major CRID procedure block. Preserves CPT mapping,
        general/untitled requirements, the complete primary CRID block, and
        associated non-indications that share the procedure topic stem.
        """
        if not governing_sections:
            return []

        by_id = {section.id: index for index, section in enumerate(all_sections)}
        heading_counts = self._heading_frequency(all_sections)
        procedure_tokens = self._procedure_topic_tokens(procedure.procedure_name)

        primary_majors: set[str] = set()
        primary_topic_stems: set[str] = set()

        scored: list[tuple[tuple[int, float], Section]] = []
        for section in governing_sections:
            if section.id in cpt_section_ids:
                continue
            overlap = self._section_procedure_overlap_score(section, procedure_tokens)
            if overlap <= 0:
                continue
            stem = self._procedure_topic_stem(section.heading or section.text[:120])
            stem_tokens = set(stem.split()) if stem else set()
            if procedure_tokens and stem_tokens:
                jaccard = len(stem_tokens & procedure_tokens) / len(
                    stem_tokens | procedure_tokens
                )
            else:
                jaccard = 0.0
            scored.append(((overlap, jaccard), section))
        best_key = max((key for key, _ in scored), default=(0, 0.0))
        # Prefer the strongest procedure-name matches as the governing block;
        # weaker semantic neighbors (e.g. leadless vs transvenous pacemaker)
        # must not define the primary CRID set; break ties by stem Jaccard so
        # shared document-family tokens (e.g. "cardiac") do not elevate them.
        primary_seeds = (
            [section for key, section in scored if key == best_key]
            if best_key[0] >= 2
            else []
        )

        for section in primary_seeds:
            stem = self._procedure_topic_stem(section.heading or section.text[:120])
            if stem:
                primary_topic_stems.add(stem)
            index = by_id.get(section.id)
            if index is None:
                continue
            crid = self._section_block_crid(
                section,
                all_sections,
                start_index=index,
                heading_counts=heading_counts,
            )
            if crid:
                primary_majors.add(crid.split(".", 1)[0])

        # Fallback: first governing seed that exposes a CRID.
        if not primary_majors:
            for section in governing_sections:
                index = by_id.get(section.id)
                if index is None:
                    continue
                crid = self._section_block_crid(
                    section,
                    all_sections,
                    start_index=index,
                    heading_counts=heading_counts,
                )
                if not crid:
                    continue
                primary_majors.add(crid.split(".", 1)[0])
                stem = self._procedure_topic_stem(section.heading or section.text[:120])
                if stem:
                    primary_topic_stems.add(stem)
                break

        kept: list[Section] = []
        for section in governing_sections:
            if section.id in cpt_section_ids:
                kept.append(section)
                continue
            if self._is_codes_or_global_identity_section(section):
                kept.append(section)
                continue

            index = by_id.get(section.id, -1)
            crid = (
                self._section_block_crid(
                    section,
                    all_sections,
                    start_index=index if index >= 0 else 0,
                    heading_counts=heading_counts,
                )
                if index >= 0
                else self._section_text_crid(section)
            )
            # Prefer an explicit CRID on the section itself over forward discovery
            # when deciding whether this seed opens a foreign procedure block.
            own_crid = self._section_crid(section) or self._section_text_crid(section)
            effective_crid = own_crid or crid

            if effective_crid is None:
                # Untitled / general requirements: keep.
                kept.append(section)
                continue

            major = effective_crid.split(".", 1)[0]
            if major in primary_majors:
                kept.append(section)
                continue

            # Associated non-indications / exclusions for the same procedure
            # family (e.g. CRID-9 non-indications next to CRID-7). Never keep a
            # different procedure block merely via weak stem overlap.
            stem = self._procedure_topic_stem(section.heading or section.text[:120])
            if (
                primary_majors
                and self._looks_like_nonindication_or_exclusion(section)
                and stem
                and any(
                    self._topic_stems_compatible(stem, primary)
                    for primary in primary_topic_stems
                )
            ):
                kept.append(section)
                continue

            logger.info(
                "policy_governing_seed_dropped",
                section_id=section.id,
                heading=section.heading,
                section_crid=effective_crid,
                primary_majors=sorted(primary_majors),
            )

        return kept if kept else list(governing_sections)

    def _section_block_crid(
        self,
        section: Section,
        all_sections: Sequence[Section],
        *,
        start_index: int,
        heading_counts: dict[str, int],
    ) -> str | None:
        return (
            self._section_crid(section)
            or self._section_text_crid(section)
            or self._discover_forward_crid(
                section,
                all_sections,
                start_index=start_index,
                heading_counts=heading_counts,
            )
        )

    @staticmethod
    def _section_text_crid(section: Section) -> str | None:
        match = _CRID_RE.search((section.text or "")[:400])
        return match.group(1) if match else None

    @staticmethod
    def _procedure_topic_tokens(procedure_name: str | None) -> set[str]:
        stop = {
            "the",
            "and",
            "for",
            "with",
            "of",
            "a",
            "an",
            "to",
            "in",
            "on",
            "or",
        }
        tokens = {
            token
            for token in _TOKEN_RE.findall((procedure_name or "").lower())
            if len(token) > 2 and token not in stop
        }
        return tokens

    @classmethod
    def _section_procedure_overlap_score(
        cls,
        section: Section,
        procedure_tokens: set[str],
    ) -> int:
        if not procedure_tokens:
            return 0
        sample = f"{section.heading or ''}\n{section.text[:240]}".lower()
        sample_tokens = set(_TOKEN_RE.findall(sample))
        return len(procedure_tokens & sample_tokens)

    @classmethod
    def _section_matches_procedure_topic(
        cls,
        section: Section,
        procedure_tokens: set[str],
    ) -> bool:
        score = cls._section_procedure_overlap_score(section, procedure_tokens)
        if score >= 2:
            return True
        return bool(len(procedure_tokens) <= 2 and score > 0)

    @staticmethod
    def _procedure_topic_stem(text: str | None) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        # First line / heading only.
        raw = raw.split("\n", 1)[0]
        raw = _CRID_RE.sub(" ", raw)
        raw = re.sub(
            r"[-–—]\s*non-?indications?\b.*$",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        raw = re.sub(r"^indications?\s*(for\s+)?", "", raw, flags=re.IGNORECASE)
        tokens = _TOKEN_RE.findall(raw.lower())
        return " ".join(tokens[:8])

    @staticmethod
    def _topic_stems_compatible(left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left == right:
            return True
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return False
        overlap = left_tokens & right_tokens
        # Same procedure family (e.g. permanent pacemaker implantation vs
        # permanent pacemaker implantation non-indications).
        return len(overlap) >= min(3, min(len(left_tokens), len(right_tokens)))

    @staticmethod
    def _looks_like_nonindication_or_exclusion(section: Section) -> bool:
        sample = f"{section.heading or ''}\n{(section.text or '')[:160]}".lower()
        return bool(
            re.search(r"\bnon-?indications?\b", sample)
            or re.search(r"\bnot indicated\b", sample)
            or re.search(r"\bexclusions?\b", sample)
            or re.search(r"\bcontraindications?\b", sample)
        )

    @staticmethod
    def _is_codes_or_global_identity_section(section: Section) -> bool:
        """True only for CPT/code-table identity sections, not indication bodies.

        Indication blocks often mention a CPT parenthetically; those must not
        be treated as global identity and thereby bypass CRID filtering.
        """
        heading = (section.heading or "").strip().lower()
        if not heading:
            return False
        if "code" in heading and (
            "procedure" in heading or "included" in heading or "cpt" in heading
        ):
            return True
        # Untitled / body code tables: require a codes-table cue in the heading
        # or opening line, not merely a CPT mention inside indication text.
        opening = (section.text or "").strip().split("\n", 1)[0].lower()
        heading_codes_cue = re.search(
            r"\b(codes?\s+included|procedure\s+codes?|cpt\s*®?\s*codes?)\b",
            heading,
        )
        opening_codes_cue = re.search(
            r"\b(codes?\s+included|procedure\s+codes?|procedure\s+description)\b",
            opening,
        )
        codes_cue = bool(heading_codes_cue or opening_codes_cue)
        if not codes_cue:
            return False
        sample = f"{heading}\n{(section.text or '')[:200]}".lower()
        return bool(
            re.search(r"\bcpt\b", sample)
            and re.search(r"\b\d{5}\b", sample)
        )

    @staticmethod
    def _heading_frequency(sections: Sequence[Section]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for section in sections:
            heading = (section.heading or "").strip()
            if heading:
                counts[heading] = counts.get(heading, 0) + 1
        return counts

    @staticmethod
    def _section_crid(section: Section) -> str | None:
        for text in (section.heading, section.parent_heading):
            if not text:
                continue
            match = _CRID_RE.search(text)
            if match:
                return match.group(1)
        return None

    def _discover_forward_crid(
        self,
        selected: Section,
        all_sections: Sequence[Section],
        *,
        start_index: int,
        heading_counts: dict[str, int],
    ) -> str | None:
        """Find the governing CRID for a selected topic title without a CRID.

        Scans contiguous forward sections (metadata, code tables, identity
        lines) until the first CRID appears or a hard boundary is reached.
        """
        for offset, forward_index in enumerate(
            range(start_index + 1, len(all_sections)),
            start=1,
        ):
            if offset > _MAX_CONTINUATION_SECTIONS:
                break
            candidate = all_sections[forward_index]
            if not self._within_page_proximity(selected, candidate):
                break
            heading = (candidate.heading or "").strip()
            if heading and _REFERENCES_RE.match(heading):
                break
            heading_crid = None
            if heading:
                match = _CRID_RE.search(heading)
                if match:
                    heading_crid = match.group(1)
            if heading_crid is not None:
                return heading_crid
            # Stop discovery at a peer titled topic that is not a wrapper /
            # internal metadata heading for the selected procedure block.
            if (
                heading
                and not self._is_running_header(candidate, heading_counts)
                and not self._is_same_topic_child(candidate, selected)
                and not self._is_internal_topic_heading(heading)
            ):
                break
        return None

    @staticmethod
    def _is_internal_topic_heading(heading: str) -> bool:
        """Headings that belong inside a procedure block, not peer topics."""
        normalized = heading.strip().lower()
        if not normalized:
            return True
        if normalized in {"codes included", "code included", "codes"}:
            return True
        # Bare policy/version identifiers (e.g. CID.PM.111.A) or CRID-only lines.
        stripped = normalized.strip("()[]")
        if _CRID_RE.fullmatch(stripped):
            return True
        # Compact version/code tokens must include a digit to avoid peer titles
        # like "Exclusions" or "Appendix".
        return bool(
            re.fullmatch(r"[a-z0-9._-]{3,40}", normalized)
            and re.search(r"\d", normalized)
        )

    @staticmethod
    def _within_page_proximity(anchor: Section, candidate: Section) -> bool:
        # Compare page spans; continue only while still near the anchor.
        anchor_lo = min(anchor.start_page, anchor.end_page)
        anchor_hi = max(anchor.start_page, anchor.end_page)
        cand_lo = min(candidate.start_page, candidate.end_page)
        cand_hi = max(candidate.start_page, candidate.end_page)
        if cand_lo > anchor_hi:
            gap = cand_lo - anchor_hi
        elif cand_hi < anchor_lo:
            gap = anchor_lo - cand_hi
        else:
            gap = 0
        return gap <= _MAX_CONTINUATION_PAGE_GAP

    def _is_running_header(
        self,
        section: Section,
        heading_counts: dict[str, int],
    ) -> bool:
        heading = (section.heading or "").strip()
        if not heading:
            return True
        if _REFERENCES_RE.match(heading):
            return False
        if _CRID_RE.search(heading):
            return False
        # Repeated short headings (e.g. chapter running titles) are wrappers.
        return len(heading) <= 80 and heading_counts.get(heading, 0) >= 2

    def _is_same_topic_child(
        self,
        candidate: Section,
        selected: Section,
    ) -> bool:
        parent = (candidate.parent_heading or "").strip()
        if not parent:
            return False
        selected_heading = (selected.heading or "").strip()
        selected_parent = (selected.parent_heading or "").strip()
        return parent == selected_heading or (
            bool(selected_parent) and parent == selected_parent
        )

    def _is_policy_boundary(
        self,
        section: Section,
        *,
        selected: Section,
        anchor_crid: str | None,
        heading_counts: dict[str, int],
    ) -> bool:
        heading = (section.heading or "").strip()
        if heading and _REFERENCES_RE.match(heading):
            return True

        heading_crid = None
        if heading:
            match = _CRID_RE.search(heading)
            if match:
                heading_crid = match.group(1)
        # Body-leading CRID identity (e.g. "Leadless ... (CRID-11.1.1)") also
        # starts a peer procedure block even when the heading is generic.
        section_crid = heading_crid or self._section_text_crid(section)
        if section_crid is not None:
            if anchor_crid is None:
                # Unknown governing CRID: any CRID starts a new topic block.
                return True
            anchor_major = anchor_crid.split(".", 1)[0]
            section_major = section_crid.split(".", 1)[0]
            # Same major CRID (identity line, Indications heading, etc.) stays
            # inside the contiguous governing block.
            return section_major != anchor_major

        if self._is_running_header(section, heading_counts):
            return False
        if self._is_same_topic_child(section, selected):
            return False
        if self._is_internal_topic_heading(heading):
            return False
        # Inside a known CRID topic block, non-CRID titled sections (code
        # tables, metadata) continue until a different major CRID/References.
        if anchor_crid is not None:
            return False
        # Without a governing CRID, a distinct titled section is a peer topic.
        return bool(heading)

    def sections_to_retrieval_results(
        self,
        sections: Sequence[Section],
    ) -> list[RetrievalResult]:
        """Adapt sections into RetrievalResult wrappers for cross-encoder rerank."""
        results: list[RetrievalResult] = []
        for rank, section in enumerate(sections, start=1):
            chunk = Chunk(
                id=section.id,
                document_id=section.document_id,
                text=self.policy_section_retrieval_text(section),
                page_number=section.start_page,
                end_page_number=section.end_page,
                heading=section.heading,
                parent_heading=section.parent_heading,
                token_count=section.token_count,
                section_id=section.id,
                chunk_index=0,
                document_date=section.document_date,
                temporal_warnings=section.temporal_warnings,
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    rank=rank,
                    score=0.0,
                    retrieval_method="section",
                )
            )
        return results

    def procedure_heading_query(self) -> str:
        """Query for BGE-M3 semantic ranking over section headings only."""
        return _PROCEDURE_HEADING_QUERY

    def procedure_chunk_query(self) -> str:
        """Query for hybrid BM25 + dense retrieval over selected chunks."""
        return _PROCEDURE_CHUNK_QUERY

    def procedure_rerank_query(self) -> str:
        """Query for cross-encoder reranking of procedure chunk candidates."""
        return _PROCEDURE_RERANK_QUERY

    def policy_bm25_query(self, procedure: ProcedureExtraction) -> str:
        """Lexical BM25 query: procedure name plus CPT when present."""
        name = (procedure.procedure_name or "").strip()
        cpt = (procedure.cpt_code or "").strip()
        if name and cpt:
            return f"{name} {cpt}"
        return name or cpt

    def policy_dense_query(self, procedure: ProcedureExtraction) -> str:
        """Dense query for BGE-M3 ranking over policy candidate sections."""
        name = (procedure.procedure_name or "").strip()
        return f"{name}. {_POLICY_DENSE_SUFFIX}"

    def policy_rerank_query(self, procedure: ProcedureExtraction) -> str:
        """Cross-encoder query for selecting governing policy sections."""
        name = (procedure.procedure_name or "").strip()
        cpt = (procedure.cpt_code or "").strip()
        procedure_label = f"{name} CPT {cpt}" if cpt else name
        return _POLICY_RERANK_TEMPLATE.format(procedure=procedure_label)

    def policy_section_retrieval_text(self, section: Section) -> str:
        """Heading + parent + full section body for policy dense/rerank stages."""
        parts: list[str] = []
        if section.heading:
            parts.append(f"Heading: {section.heading}")
        if section.parent_heading:
            parts.append(f"Parent: {section.parent_heading}")
        body = section.text.strip()
        if body:
            parts.append(f"Content: {body}")
        return "\n".join(parts) if parts else ""

    def tokenize(self, text: str) -> list[str]:
        """Lowercase tokenize while preserving numbers and CPT-like tokens."""
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def compact_section_text(self, section: Section) -> str:
        """Compact heading + preview text for section-level BM25."""
        parts: list[str] = []
        if section.heading:
            parts.append(f"Heading: {section.heading}")
        if section.parent_heading:
            parts.append(f"Parent: {section.parent_heading}")
        preview_limit = self._settings.COMPACT_SECTION_CHARS
        preview = " ".join(section.text.split())
        if len(preview) > preview_limit:
            preview = preview[:preview_limit].rstrip()
        if preview:
            parts.append(f"Preview: {preview}")
        return "\n".join(parts) if parts else ""

    def _build_retrieval_text(self, chunk: Chunk) -> str:
        parts: list[str] = []
        if chunk.document_date:
            parts.append(f"Encounter date: {chunk.document_date}")
        if chunk.heading:
            parts.append(f"Heading: {chunk.heading}")
        if chunk.parent_heading:
            parts.append(f"Parent: {chunk.parent_heading}")
        parts.append(f"Content: {chunk.text}")
        return "\n".join(parts)

    def _rank_texts_dense(
        self,
        query: str,
        texts: Sequence[str],
        *,
        profile_prefix: str | None = None,
    ) -> list[int]:
        if not query.strip() or not texts:
            return []
        embed_started = time.perf_counter()
        embeddings = self._embedding_service.embed(
            list(texts),
            operation=f"{profile_prefix or 'dense'}_corpus",
        )
        query_embedding = self._embedding_service.embed(
            [query],
            operation=f"{profile_prefix or 'dense'}_query",
        )[0]
        embed_ms = int((time.perf_counter() - embed_started) * 1000)
        score_started = time.perf_counter()
        scores = self._embedding_service.similarity(query_embedding, embeddings)
        ranked = [int(index) for index in np.argsort(-scores, kind="stable")]
        score_ms = int((time.perf_counter() - score_started) * 1000)

        metrics = get_request_metrics()
        if metrics is not None:
            if profile_prefix == "procedure_heading":
                metrics.add_substage_ms("procedure_heading_embedding_ms", embed_ms)
                metrics.add_substage_ms("procedure_heading_dense_ms", score_ms)
            elif profile_prefix == "policy":
                metrics.add_substage_ms("policy_candidate_embedding_ms", embed_ms)
                metrics.add_substage_ms("policy_dense_ms", score_ms)
        return ranked

    def _bm25_search(
        self,
        query: str,
        index: HybridIndex,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        try:
            scores = np.asarray(
                index.bm25.get_scores(query_tokens),
                dtype=np.float64,
            )
        except Exception as exc:
            raise RetrievalError("BM25 retrieval failed") from exc

        selected_top_k = top_k if top_k is not None else self._settings.BM25_TOP_K
        selected_top_k = min(selected_top_k, len(index.chunks))
        if selected_top_k <= 0:
            return []

        ranked = np.argsort(-scores, kind="stable")
        results: list[RetrievalResult] = []
        for rank_offset, index_pos in enumerate(ranked[:selected_top_k], start=1):
            idx = int(index_pos)
            results.append(
                RetrievalResult(
                    chunk=index.chunks[idx],
                    rank=rank_offset,
                    score=float(scores[idx]),
                    retrieval_method="bm25",
                    bm25_rank=rank_offset,
                )
            )
        return results

    def _dense_search(
        self,
        query: str,
        index: HybridIndex,
        top_k: int | None = None,
        *,
        operation: str | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self._embedding_service.embed(
            [query],
            operation=operation or "dense_query",
        )[0]
        scores = self._embedding_service.similarity(
            query_embedding,
            index.embeddings,
        )
        selected_top_k = top_k if top_k is not None else self._settings.DENSE_TOP_K
        selected_top_k = min(selected_top_k, len(index.chunks))
        if selected_top_k <= 0:
            return []

        ranked = np.argsort(-scores, kind="stable")
        results: list[RetrievalResult] = []
        for rank_offset, index_pos in enumerate(ranked[:selected_top_k], start=1):
            idx = int(index_pos)
            results.append(
                RetrievalResult(
                    chunk=index.chunks[idx],
                    rank=rank_offset,
                    score=float(scores[idx]),
                    retrieval_method="dense",
                    dense_rank=rank_offset,
                )
            )
        return results

    def _rrf_fuse(
        self,
        *,
        dense_results: Sequence[RetrievalResult],
        bm25_results: Sequence[RetrievalResult],
        chunks: Sequence[Chunk],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        rrf_k = self._settings.RRF_K
        chunk_order = {chunk.id: index for index, chunk in enumerate(chunks)}
        chunk_by_id = {chunk.id: chunk for chunk in chunks}

        rrf_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}

        for result in dense_results:
            chunk_id = result.chunk.id
            dense_ranks[chunk_id] = result.rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + result.rank)
            )

        for result in bm25_results:
            chunk_id = result.chunk.id
            bm25_ranks[chunk_id] = result.rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + result.rank)
            )

        def sort_key(chunk_id: str) -> tuple[float, int, int]:
            best_rank = min(
                rank
                for rank in (
                    dense_ranks.get(chunk_id),
                    bm25_ranks.get(chunk_id),
                )
                if rank is not None
            )
            return (
                -rrf_scores[chunk_id],
                best_rank,
                chunk_order.get(chunk_id, 10**9),
            )

        ordered_ids = sorted(rrf_scores.keys(), key=sort_key)
        selected_top_k = top_k if top_k is not None else self._settings.RRF_TOP_K
        selected_top_k = min(selected_top_k, len(ordered_ids))
        fused: list[RetrievalResult] = []
        for rank, chunk_id in enumerate(ordered_ids[:selected_top_k], start=1):
            fused.append(
                RetrievalResult(
                    chunk=chunk_by_id[chunk_id],
                    rank=rank,
                    score=rrf_scores[chunk_id],
                    retrieval_method="rrf",
                    dense_rank=dense_ranks.get(chunk_id),
                    bm25_rank=bm25_ranks.get(chunk_id),
                    rrf_score=rrf_scores[chunk_id],
                )
            )
        return fused

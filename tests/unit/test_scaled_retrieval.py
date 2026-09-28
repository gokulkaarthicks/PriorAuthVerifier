"""Focused unit tests for section-first retrieval invariants."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from app.core.config import Settings
from app.models.chunk import Chunk
from app.models.retrieval import RetrievalResult
from app.models.section import Section
from app.schemas.procedure import ProcedureExtraction
from app.services.policy_service import PolicyService
from app.services.procedure_service import ProcedureService
from app.services.retrieval_service import RetrievalService


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
                "SECTION_NEIGHBOR_WINDOW": 1,
        "RERANK_TOP_K": 5,
        "BM25_TOP_K": 10,
        "DENSE_TOP_K": 10,
        "RRF_TOP_K": 10,
        "PROCEDURE_RERANK_TOP_K": 5,
        "PATIENT_HEADING_TOP_K": 3,
        "POLICY_BM25_TOP_K": 8,
        "POLICY_DENSE_TOP_K": 4,
        "POLICY_RERANK_TOP_K": 2,
        "COMPACT_SECTION_CHARS": 400,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _section(
    section_id: str,
    text: str,
    *,
    heading: str | None = None,
    parent_heading: str | None = None,
    page: int = 1,
    token_count: int | None = None,
) -> Section:
    tokens = token_count if token_count is not None else max(1, len(text.split()))
    return Section(
        id=section_id,
        document_id="doc",
        heading=heading,
        parent_heading=parent_heading,
        text=text,
        start_page=page,
        end_page=page,
        token_count=tokens,
    )


def _chunk_from_section(section: Section, *, chunk_index: int = 0) -> Chunk:
    return Chunk(
        id=f"{section.id}:chunk:{chunk_index:04d}",
        document_id=section.document_id,
        text=section.text,
        page_number=section.start_page,
        end_page_number=section.end_page,
        heading=section.heading,
        parent_heading=section.parent_heading,
        token_count=section.token_count,
        section_id=section.id,
        chunk_index=chunk_index,
    )


def _embedding_service(dim: int = 4) -> MagicMock:
    service = MagicMock()

    def embed(texts: list[str], **_kwargs) -> np.ndarray:
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for index, text in enumerate(texts):
            lower = text.lower()
            matrix[index, 0] = (len(text) % 11) / 11.0
            matrix[index, 1] = 1.0 if "33207" in text or "cpt" in lower else 0.0
            matrix[index, 2] = (
                1.0
                if any(
                    token in lower
                    for token in (
                        "pacemaker",
                        "planned",
                        "procedure",
                        "intervention",
                        "upcoming",
                        "surgery",
                    )
                )
                else 0.0
            )
            matrix[index, 3] = (
                1.0 if "indication" in lower or "coverage" in lower else 0.0
            )
            norm = np.linalg.norm(matrix[index])
            if norm > 0:
                matrix[index] /= norm
        return matrix

    def similarity(
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        docs = np.asarray(document_embeddings, dtype=np.float32)
        return docs @ query

    service.embed.side_effect = embed
    service.similarity.side_effect = similarity
    return service


def _passthrough_chunking() -> MagicMock:
    service = MagicMock()

    def chunk_sections(sections: list[Section]) -> list[Chunk]:
        return [_chunk_from_section(section) for section in sections]

    service.chunk_sections.side_effect = chunk_sections
    return service


def _identity_reranker() -> MagicMock:
    reranker = MagicMock()

    def rerank(
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        **_kwargs: object,
    ):
        limit = top_k if top_k is not None else len(candidates)
        return candidates[:limit]

    def rerank_grouped(groups, top_k=None, **_kwargs):
        keep = top_k if top_k is not None else None
        result = {}
        for group_id, _query, candidates in groups:
            limit = keep if keep is not None else len(candidates)
            result[group_id] = candidates[:limit]
        return result

    reranker.rerank.side_effect = rerank
    reranker.rerank_grouped.side_effect = rerank_grouped
    return reranker


def _procedure() -> ProcedureExtraction:
    return ProcedureExtraction(
        procedure_name="Permanent Cardiac Pacemaker Implantation",
        cpt_code="33207",
        is_future_procedure=True,
        confidence=0.95,
    )


def test_procedure_discovery_from_semantically_similar_heading() -> None:
    sections = [
        _section("s1", "Allergy list only.", heading="Allergies", page=1),
        _section(
            "s2",
            "Schedule permanent cardiac pacemaker implantation CPT 33207.",
            heading="Upcoming Intervention Plan",
            page=2,
            token_count=40,
        ),
        _section("s3", "Labs within normal limits.", heading="Labs", page=3),
        _section("s4", "Social history noncontributory.", heading="Social", page=4),
        _section("s5", "Review of systems negative.", heading="ROS", page=5),
    ]
    embedding = _embedding_service()
    retrieval = RetrievalService(embedding, _settings())
    selected = retrieval.select_sections_by_heading_similarity(
        sections,
        query=retrieval.procedure_heading_query(),
        top_k=3,
    )
    assert any(section.heading == "Upcoming Intervention Plan" for section in selected)
    assert not any(section.heading == "Planned Procedure" for section in selected)


def test_policy_canonical_path_cpt_bm25_union_dense_rerank_then_expand() -> None:
    sections = [
        _section(
            f"s{i}",
            (
                "Permanent Cardiac Pacemaker Implantation CPT 33207 medical necessity."
                if i == 4
                else f"Unrelated policy paragraph {i} about transport billing."
            ),
            heading=(
                "Permanent Pacemaker Implantation" if i == 4 else f"Noise {i}"
            ),
            page=i + 1,
        )
        for i in range(20)
    ]
    embedding = _embedding_service()
    settings = _settings(
        POLICY_BM25_TOP_K=6,
        POLICY_DENSE_TOP_K=3,
        POLICY_RERANK_TOP_K=1,
    )
    retrieval = RetrievalService(embedding, settings)
    chunking = _passthrough_chunking()

    bm25_corpus_sizes: list[int] = []
    dense_input_sizes: list[int] = []
    expansion_calls: list[tuple[list[str], set[str] | None]] = []
    original_bm25_select = retrieval.bm25_select_sections
    original_dense = retrieval._rank_texts_dense
    original_expand = retrieval.expand_section_context

    def spy_bm25_select(secs, *, query, top_k=None, **kwargs):
        bm25_corpus_sizes.append(len(secs))
        return original_bm25_select(secs, query=query, top_k=top_k, **kwargs)

    def spy_dense(query: str, texts: list[str], **kwargs) -> list[int]:
        dense_input_sizes.append(len(texts))
        return original_dense(query, texts, **kwargs)

    def spy_expand(selected, all_sections, *, neighbor_window=None, exclude_ids=None):
        expansion_calls.append(
            ([section.id for section in selected], exclude_ids)
        )
        return original_expand(
            selected,
            all_sections,
            neighbor_window=neighbor_window,
            exclude_ids=exclude_ids,
        )

    retrieval.bm25_select_sections = spy_bm25_select  # type: ignore[method-assign]
    retrieval._rank_texts_dense = spy_dense  # type: ignore[method-assign]
    retrieval.expand_section_context = spy_expand  # type: ignore[method-assign]
    retrieval.build_hybrid_index = MagicMock(  # type: ignore[method-assign]
        wraps=retrieval.build_hybrid_index
    )
    reranker = _identity_reranker()

    service = PolicyService(
        llm_service=MagicMock(),
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=reranker,
        settings=settings,
    )
    candidates = service.retrieve_policy_candidates(_procedure(), sections)

    assert bm25_corpus_sizes == [20]
    assert dense_input_sizes
    assert dense_input_sizes[0] <= settings.POLICY_BM25_TOP_K + 1
    assert dense_input_sizes[0] < len(sections)
    assert expansion_calls
    assert len(expansion_calls[0][0]) <= settings.POLICY_RERANK_TOP_K
    retrieval.build_hybrid_index.assert_not_called()
    assert not hasattr(RetrievalService, "rank_sections")
    rerank_call = reranker.rerank.call_args
    assert rerank_call is not None
    assert len(rerank_call.kwargs["candidates"]) <= settings.POLICY_DENSE_TOP_K
    assert rerank_call.kwargs["top_k"] == settings.POLICY_RERANK_TOP_K
    assert "governing" in rerank_call.kwargs["query"].lower()
    assert candidates
    assert all(item.retrieval_method == "policy_governing" for item in candidates)
    chunked = chunking.chunk_sections.call_args.args[0]
    assert len(chunked) < len(sections)
    assert "s4" in {section.id for section in chunked}


def test_duplicate_cpt_prefers_direct_governing_section() -> None:
    """Same CPT in code table, governing section, and related CRT-P section."""
    code_table = _section(
        "code-table",
        "General code table listing CPT 33207 among many codes.",
        heading="Covered Codes",
        page=1,
    )
    governing = _section(
        "pacemaker",
        (
            "Permanent Pacemaker Implantation. CPT 33207. "
            "Medical necessity indications and coverage requirements for "
            "Permanent Cardiac Pacemaker Implantation single chamber ventricular."
        ),
        heading="Permanent Pacemaker Implantation",
        page=2,
    )
    governing_child = _section(
        "pacemaker-child",
        "Document symptomatic bradycardia before implantation.",
        heading="Bradycardia Criteria",
        parent_heading="Permanent Pacemaker Implantation",
        page=3,
    )
    related = _section(
        "crt-p",
        (
            "CRT-P / conduction system pacing. CPT 33207. "
            "Different QRS and heart-failure criteria for biventricular pacing."
        ),
        heading="CRT-P / Conduction System Pacing",
        page=4,
    )
    filler = [
        _section(f"n{i}", f"Unrelated policy filler {i}.", heading=f"F{i}", page=10 + i)
        for i in range(10)
    ]
    sections = [code_table, governing, governing_child, related, *filler]

    embedding = _embedding_service()
    settings = _settings(
        POLICY_BM25_TOP_K=8,
        POLICY_DENSE_TOP_K=4,
        POLICY_RERANK_TOP_K=1,
        SECTION_NEIGHBOR_WINDOW=1,
    )
    retrieval = RetrievalService(embedding, settings)

    # Prefer the direct governing section at rerank time without pacemaker hardcoding
    # in production code: score candidates by procedure-name token overlap.
    def governing_rerank(*, query: str, candidates, top_k=None, **_kwargs):
        procedure_tokens = {
            "permanent",
            "cardiac",
            "pacemaker",
            "implantation",
        }

        def score(result: RetrievalResult) -> float:
            text = f"{result.chunk.heading or ''} {result.chunk.text}".lower()
            return float(sum(1 for token in procedure_tokens if token in text))

        ordered = sorted(candidates, key=score, reverse=True)
        limit = top_k if top_k is not None else len(ordered)
        return ordered[:limit]

    reranker = MagicMock()
    reranker.rerank.side_effect = governing_rerank
    chunking = _passthrough_chunking()
    service = PolicyService(
        llm_service=MagicMock(),
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=reranker,
        settings=settings,
    )

    cpt_hits = retrieval.find_sections_with_cpt(sections, "33207")
    assert {section.id for section in cpt_hits} >= {
        "code-table",
        "pacemaker",
        "crt-p",
    }

    candidates = service.retrieve_policy_candidates(_procedure(), sections)
    assert candidates
    sol_section_ids = {item.chunk.section_id for item in candidates}
    assert "pacemaker" in sol_section_ids
    assert "crt-p" not in sol_section_ids
    assert "code-table" not in sol_section_ids
    assert sol_section_ids <= {"pacemaker", "pacemaker-child"}
    assert "crt-p" not in {
        section.id for section in chunking.chunk_sections.call_args.args[0]
    }
    sol_text = " ".join(item.chunk.text for item in candidates).lower()
    assert "biventricular" not in sol_text
    assert "permanent pacemaker implantation" in sol_text


def test_policy_path_without_cpt_still_uses_same_pipeline() -> None:
    sections = [
        _section(
            "relevant",
            "Coverage criteria for Permanent Cardiac Pacemaker Implantation.",
            heading="Pacemaker Coverage",
            page=1,
        ),
        _section("noise1", "Ambulance transport billing.", heading="EMS", page=2),
        *[
            _section(f"n{i}", f"filler {i}", heading=f"F{i}", page=10 + i)
            for i in range(8)
        ],
    ]
    embedding = _embedding_service()
    retrieval = RetrievalService(embedding, _settings())
    retrieval.build_hybrid_index = MagicMock(wraps=retrieval.build_hybrid_index)  # type: ignore[method-assign]
    reranker = _identity_reranker()
    procedure = ProcedureExtraction(
        procedure_name="Permanent Cardiac Pacemaker Implantation",
        cpt_code=None,
        is_future_procedure=True,
        confidence=0.9,
    )
    service = PolicyService(
        llm_service=MagicMock(),
        retrieval_service=retrieval,
        chunking_service=_passthrough_chunking(),
        reranking_service=reranker,
        settings=_settings(),
    )
    candidates = service.retrieve_policy_candidates(procedure, sections)

    retrieval.build_hybrid_index.assert_not_called()
    reranker.rerank.assert_called_once()
    assert candidates
    assert all(item.retrieval_method == "policy_governing" for item in candidates)


def test_empty_policy_candidates_return_matching_false() -> None:
    import asyncio

    service = PolicyService(
        llm_service=MagicMock(),
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        settings=_settings(),
    )
    service.retrieve_policy_candidates = MagicMock(return_value=[])  # type: ignore[method-assign]

    async def _run() -> None:
        outcome = await service.analyze_policy(_procedure(), [])
        assert outcome.applicable is False
        assert outcome.criteria is None
        assert outcome.llm_call_count == 0
        assert "No governing policy criteria" in outcome.applicability_reason

    asyncio.run(_run())


def test_policy_no_full_corpus_dense_or_hybrid_index() -> None:
    sections = [
        _section(
            f"sec-{i}",
            (
                "Permanent Cardiac Pacemaker Implantation CPT 33207 medical necessity."
                if i == 7
                else f"Policy paragraph {i} transport details."
            ),
            heading=f"Heading {i}",
            page=i + 1,
        )
        for i in range(40)
    ]
    embedding = _embedding_service()
    retrieval = RetrievalService(
        embedding,
        _settings(POLICY_BM25_TOP_K=5, POLICY_DENSE_TOP_K=3),
    )
    dense_sizes: list[int] = []
    original_dense = retrieval._rank_texts_dense

    def spy_dense(query: str, texts: list[str], **kwargs) -> list[int]:
        dense_sizes.append(len(texts))
        return original_dense(query, texts, **kwargs)

    retrieval._rank_texts_dense = spy_dense  # type: ignore[method-assign]
    retrieval.build_hybrid_index = MagicMock(wraps=retrieval.build_hybrid_index)  # type: ignore[method-assign]
    service = PolicyService(
        llm_service=MagicMock(),
        retrieval_service=retrieval,
        chunking_service=_passthrough_chunking(),
        reranking_service=_identity_reranker(),
        settings=_settings(POLICY_BM25_TOP_K=5, POLICY_DENSE_TOP_K=3),
    )
    candidates = service.retrieve_policy_candidates(_procedure(), sections)

    assert candidates
    assert dense_sizes
    assert all(size < len(sections) for size in dense_sizes)
    retrieval.build_hybrid_index.assert_not_called()
    chunked = service._chunking_service.chunk_sections.call_args.args[0]
    assert len(chunked) < len(sections)


def test_neighboring_context_retained_via_expand_section_context() -> None:
    sections = [
        _section("s0", "Parent coverage intro.", heading="Coverage", page=1),
        _section(
            "s1",
            "CPT 33207 permanent pacemaker indications.",
            heading="Indications",
            parent_heading="Coverage",
            page=2,
        ),
        _section(
            "s2",
            "Child detail under indications.",
            heading="Detail",
            parent_heading="Indications",
            page=3,
        ),
        _section("s3", "Far appendix.", heading="Appendix", page=9),
    ]
    embedding = _embedding_service()
    retrieval = RetrievalService(embedding, _settings(SECTION_NEIGHBOR_WINDOW=1))
    expanded = retrieval.expand_section_context([sections[1]], sections)
    assert [section.id for section in expanded] == ["s0", "s1", "s2"]
    assert "s3" not in {section.id for section in expanded}


def test_patient_evidence_retrieval_uses_bm25_then_dense_chunks() -> None:
    from app.schemas.policy import CriterionCategory
    from app.services.evidence_service import AtomicCriterion, EvidenceService

    sections = [
        _section(
            "relevant",
            "Duplex ultrasound demonstrated saphenous reflux.",
            heading="Imaging",
            page=1,
            token_count=30,
        ),
        *[
            _section(
                f"noise-{i}",
                f"Unrelated outpatient note number {i} about diet and exercise.",
                heading=f"Noise {i}",
                page=i + 2,
                token_count=20,
            )
            for i in range(40)
        ],
    ]
    embedding = _embedding_service()
    settings = _settings(
        EVIDENCE_SECTION_BM25_TOP_K=3,
        EVIDENCE_DENSE_TOP_K=2,
        EVIDENCE_RERANK_TOP_K=1,
    )
    retrieval = RetrievalService(embedding, settings)
    chunking = _passthrough_chunking()
    dense_sizes: list[int] = []
    chunked_sizes: list[int] = []
    original_dense = retrieval.dense_rank_chunks_batched
    original_chunk_fn = chunking.chunk_sections.side_effect

    def spy_dense(groups, *, top_k=None):
        for _group_id, (_query, chunks) in groups.items():
            dense_sizes.append(len(chunks))
        return original_dense(groups, top_k=top_k)

    def spy_chunk(secs: list[Section]) -> list[Chunk]:
        chunked_sizes.append(len(secs))
        return original_chunk_fn(secs)

    retrieval.dense_rank_chunks_batched = spy_dense  # type: ignore[method-assign]
    chunking.chunk_sections.side_effect = spy_chunk
    retrieval._rrf_fuse = MagicMock(wraps=retrieval._rrf_fuse)  # type: ignore[method-assign]
    retrieval.build_hybrid_index = MagicMock(  # type: ignore[method-assign]
        wraps=retrieval.build_hybrid_index
    )

    service = EvidenceService(
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=_identity_reranker(),
        llm_service=MagicMock(),
        settings=settings,
    )
    atoms = [
        AtomicCriterion(
            id="c1",
            description="Demonstrated saphenous reflux.",
            category=CriterionCategory.DIAGNOSTIC,
            kind="condition",
        )
    ]
    from datetime import date

    service._retrieve_evidence_for_atoms(
        atoms,
        sections,
        reference_date=date(2026, 9, 26),
    )

    assert chunked_sizes
    assert all(size <= settings.EVIDENCE_SECTION_BM25_TOP_K for size in chunked_sizes)
    assert dense_sizes
    assert all(size <= settings.EVIDENCE_SECTION_BM25_TOP_K for size in dense_sizes)
    retrieval._rrf_fuse.assert_not_called()
    retrieval.build_hybrid_index.assert_not_called()


def test_procedure_path_heading_only_then_chunk_hybrid_rerank() -> None:
    sections = [
        _section(
            f"s{i}",
            (
                "UNIQUE_BODY_NEVER_IN_HEADING_STAGE schedule pacemaker planned "
                "procedure pending authorization"
                if i == 2
                else f"noise note {i} UNIQUE_BODY_NEVER_IN_HEADING_STAGE"
            ),
            heading=(
                "Upcoming Intervention Plan" if i == 2 else f"H{i}"
            ),
            page=i + 1,
        )
        for i in range(20)
    ]
    embedding = _embedding_service()
    settings = _settings(
        PATIENT_HEADING_TOP_K=3,
        PROCEDURE_RERANK_TOP_K=2,
    )
    retrieval = RetrievalService(embedding, settings)
    chunking = _passthrough_chunking()

    heading_dense_inputs: list[list[str]] = []
    hybrid_sizes: list[int] = []
    chunked_section_ids: list[list[str]] = []
    original_dense = retrieval._rank_texts_dense
    original_build = retrieval.build_hybrid_index
    original_rrf = retrieval._rrf_fuse
    chunk_rrf_calls = 0

    def spy_dense(query: str, texts: list[str], **kwargs) -> list[int]:
        # First-stage heading dense: texts are bare headings (no body markers).
        if texts and all(
            "UNIQUE_BODY_NEVER_IN_HEADING_STAGE" not in text for text in texts
        ):
            heading_dense_inputs.append(list(texts))
        return original_dense(query, texts, **kwargs)

    def spy_rrf(**kwargs):
        nonlocal chunk_rrf_calls
        chunk_rrf_calls += 1
        return original_rrf(**kwargs)

    def spy_build(chunks: list[Chunk], *, index_kind: str = "selected", **kwargs):
        hybrid_sizes.append(len(chunks))
        return original_build(chunks, index_kind=index_kind, **kwargs)

    def spy_chunk(secs: list[Section]) -> list[Chunk]:
        chunked_section_ids.append([section.id for section in secs])
        return [_chunk_from_section(section) for section in secs]

    retrieval._rank_texts_dense = spy_dense  # type: ignore[method-assign]
    retrieval._rrf_fuse = spy_rrf  # type: ignore[method-assign]
    retrieval.build_hybrid_index = spy_build  # type: ignore[method-assign]
    chunking.chunk_sections.side_effect = spy_chunk
    retrieval.build_section_bm25_index = MagicMock(  # type: ignore[method-assign]
        wraps=retrieval.build_section_bm25_index
    )
    retrieval.compact_section_text = MagicMock(  # type: ignore[method-assign]
        wraps=retrieval.compact_section_text
    )
    reranker = _identity_reranker()

    service = ProcedureService(
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=reranker,
        llm_service=MagicMock(),
        settings=settings,
    )
    selected = service._retrieve_candidates(sections)

    # First-stage BGE sees ONLY headings.
    assert heading_dense_inputs
    assert len(heading_dense_inputs[0]) == len(sections)
    assert all(
        text in {section.heading for section in sections}
        for text in heading_dense_inputs[0]
    )
    assert all(
        "UNIQUE_BODY_NEVER_IN_HEADING_STAGE" not in text
        for text in heading_dense_inputs[0]
    )
    # Heading stage does not build a section BM25 index.
    retrieval.build_section_bm25_index.assert_not_called()
    retrieval.compact_section_text.assert_not_called()

    # Only heading-selected sections are chunked; never full document.
    assert chunked_section_ids
    assert len(chunked_section_ids[0]) == settings.PATIENT_HEADING_TOP_K
    assert len(chunked_section_ids[0]) < len(sections)
    assert "s2" in chunked_section_ids[0]
    chunked_sections = chunking.chunk_sections.call_args.args[0]
    for section in chunked_sections:
        assert section.id
        assert section.heading
        assert section.start_page >= 1
        assert section.text.strip()

    # Hybrid BM25+dense+RRF over selected chunks only.
    assert hybrid_sizes == [settings.PATIENT_HEADING_TOP_K]
    assert chunk_rrf_calls == 1
    rerank_call = reranker.rerank.call_args
    assert rerank_call is not None
    assert len(rerank_call.kwargs["candidates"]) <= settings.PATIENT_HEADING_TOP_K
    assert rerank_call.kwargs["top_k"] == settings.PROCEDURE_RERANK_TOP_K
    assert "planned" in rerank_call.kwargs["query"].lower()
    assert len(selected) <= settings.PROCEDURE_RERANK_TOP_K
    for chunk in selected:
        assert chunk.section_id
        assert chunk.heading
        assert chunk.page_number >= 1


def test_no_usable_headings_returns_empty_candidates() -> None:
    sections = [
        _section(f"page-{i}", f"planned procedure text {i}", heading=None, page=i + 1)
        for i in range(5)
    ]
    embedding = _embedding_service()
    retrieval = RetrievalService(embedding, _settings(PATIENT_HEADING_TOP_K=3))
    retrieval.build_hybrid_index = MagicMock(  # type: ignore[method-assign]
        wraps=retrieval.build_hybrid_index
    )
    chunking = _passthrough_chunking()
    service = ProcedureService(
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=_identity_reranker(),
        llm_service=MagicMock(),
        settings=_settings(PATIENT_HEADING_TOP_K=3),
    )
    selected = service._retrieve_candidates(sections)

    assert selected == []
    chunking.chunk_sections.assert_not_called()
    retrieval.build_hybrid_index.assert_not_called()


def test_crid7_topic_anchor_recovers_full_block_in_policy_sol_context() -> None:
    """CRID-7 indications lost at rerank still reach Sol via topic-anchor expand.

    Mimics the production failure mode: cross-encoder keeps the parent topic
    title (and optionally CRID-9 non-indications) but drops the structurally
    downstream Indications (CRID-7) section that holds SND criteria.
    """
    topic = _section(
        "s1739",
        "Permanent Pacemaker Implantation",
        heading="Permanent Pacemaker Implantation",
        page=697,
    )
    crid_identity = _section(
        "s1740",
        "(CRID-7)",
        heading="(CRID-7)",
        page=697,
    )
    version = _section(
        "s1741",
        "CID.PM.111.A v1.0.2025",
        heading="CID.PM.111.A",
        page=697,
    )
    codes = _section(
        "s1742",
        "Codes included Description CPT 33206 33207 Permanent Cardiac Pacemaker",
        heading="Codes included",
        page=697,
    )
    cid_wrapper = _section(
        "s1743",
        "Cardiac Implantable Devices (CID)",
        heading="Cardiac Implantable Devices (CID)",
        page=697,
    )
    indications = _section(
        "s1744",
        (
            "Indications for Permanent Pacemaker (CRID-7)\n"
            "Sinus bradycardia at rate <50 beats per minute\n"
            "Sinus pauses >3 seconds"
        ),
        heading="Indications for Permanent Pacemaker (CRID-7)",
        page=697,
    )
    avb_continuation = _section(
        "s1745",
        "Symptomatic AVB continuation criteria",
        heading="Cardiac Implantable Devices (CID)",
        page=698,
    )
    non_indications = _section(
        "s1747",
        "Permanent pacemaker implantation is not indicated for asymptomatic SND.",
        heading="Permanent Pacemaker Implantation - Non-indications (CRID-9)",
        page=700,
    )
    filler = [
        _section(
            f"noise-{i}",
            f"Unrelated transport billing policy paragraph {i}.",
            heading=f"Noise {i}",
            page=10 + i,
        )
        for i in range(8)
    ]
    sections = [
        topic,
        crid_identity,
        version,
        codes,
        cid_wrapper,
        indications,
        avb_continuation,
        non_indications,
        *filler,
    ]
    # Indications sit structurally downstream of the topic + metadata/code table.
    topic_index = sections.index(topic)
    codes_index = sections.index(codes)
    indications_index = sections.index(indications)
    assert topic_index < codes_index < indications_index

    settings = _settings(
        POLICY_BM25_TOP_K=12,
        POLICY_DENSE_TOP_K=10,
        POLICY_RERANK_TOP_K=2,
        SECTION_NEIGHBOR_WINDOW=1,
    )
    embedding = _embedding_service()
    retrieval = RetrievalService(embedding, settings)
    chunking = _passthrough_chunking()

    block_ids = {
        "s1739",
        "s1740",
        "s1741",
        "s1742",
        "s1743",
        "s1744",
        "s1745",
        "s1747",
    }
    original_dense = retrieval.dense_rank_sections

    def dense_prefer_block(secs, *, query, top_k=None, **_kwargs):
        preferred = [section for section in secs if section.id in block_ids]
        remainder = [section for section in secs if section.id not in block_ids]
        ordered = preferred + remainder
        if not preferred:
            return original_dense(secs, query=query, top_k=top_k)
        limit = top_k if top_k is not None else len(ordered)
        return ordered[:limit]

    retrieval.dense_rank_sections = dense_prefer_block  # type: ignore[method-assign]

    governing_from_rerank: list[str] = []

    def topic_anchor_rerank(*, query: str, candidates, top_k=None, **_kwargs):
        del query
        # Parent/topic survives; direct Indications (CRID-7) loses top-K.
        selected = [
            item
            for item in candidates
            if (item.chunk.heading or "") == "Permanent Pacemaker Implantation"
        ]
        assert selected, "parent/topic anchor must survive into rerank candidates"
        limit = top_k if top_k is not None else 1
        chosen = selected[:limit]
        governing_from_rerank[:] = [
            item.chunk.section_id or "" for item in chosen if item.chunk.section_id
        ]
        return chosen

    reranker = MagicMock()
    reranker.rerank.side_effect = topic_anchor_rerank

    service = PolicyService(
        llm_service=MagicMock(),
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=reranker,
        settings=settings,
    )
    candidates = service.retrieve_policy_candidates(_procedure(), sections)
    assert candidates
    assert governing_from_rerank == ["s1739"]
    assert "s1744" not in governing_from_rerank

    prepared = service._prepare_policy_context(_procedure(), candidates)
    joined = "\n".join(item.chunk.text for item in prepared.candidates)
    section_ids = {
        item.chunk.section_id for item in prepared.candidates if item.chunk.section_id
    }

    # Parent topic + intermediate metadata/code-table + governing indications
    # + associated non-indications (must not require independent rerank survival).
    assert "s1739" in section_ids
    assert "s1740" in section_ids
    assert "s1742" in section_ids
    assert "s1744" in section_ids
    assert "s1747" in section_ids
    assert "Indications for Permanent Pacemaker (CRID-7)" in joined
    assert "Sinus bradycardia at rate <50 beats per minute" in joined
    assert "Sinus pauses >3 seconds" in joined
    assert "asymptomatic SND" in joined
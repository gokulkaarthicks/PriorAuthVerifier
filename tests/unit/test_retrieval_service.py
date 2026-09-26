"""Unit tests for RetrievalService.

Hybrid-index tests below operate on small pre-selected fixture chunks. They
validate BM25 / dense / RRF primitives used after section narrowing — they do
not recreate full-document chunk→embed orchestration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import RetrievalError
from app.models.chunk import Chunk
from app.models.retrieval import RetrievalResult
from app.models.section import Section
from app.schemas.procedure import ProcedureExtraction
from app.services.retrieval_service import RetrievalService


def _chunk(
    chunk_id: str,
    text: str,
    *,
    heading: str | None = None,
    parent_heading: str | None = None,
    page: int = 1,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="policy-doc",
        text=text,
        page_number=page,
        end_page_number=page,
        heading=heading,
        parent_heading=parent_heading,
        token_count=10,
        section_id=f"section-{chunk_id}",
        chunk_index=0,
    )


def _section(
    section_id: str,
    text: str,
    *,
    heading: str | None = None,
    parent_heading: str | None = None,
    page: int = 1,
    token_count: int | None = None,
) -> Section:
    return Section(
        id=section_id,
        document_id="doc",
        heading=heading,
        parent_heading=parent_heading,
        text=text,
        start_page=page,
        end_page=page,
        token_count=(
            token_count
            if token_count is not None
            else max(1, len(text.split()))
        ),
    )


def _embedding_service(dim: int = 4) -> MagicMock:
    service = MagicMock()

    def embed(texts: list[str], **_kwargs) -> np.ndarray:
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for index, text in enumerate(texts):
            # Deterministic pseudo-embedding from text length and digits.
            matrix[index, 0] = (len(text) % 7) / 7.0
            matrix[index, 1] = 1.0 if "33207" in text else 0.0
            matrix[index, 2] = 1.0 if "pacemaker" in text.lower() else 0.0
            matrix[index, 3] = 1.0 if "ambulance" in text.lower() else 0.0
            if "intervention" in text.lower() or "planned" in text.lower():
                matrix[index, 2] = 1.0
            if "upcoming" in text.lower():
                matrix[index, 2] = 1.0
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


def _settings(**overrides: object) -> Settings:
    defaults = {
        "BM25_TOP_K": 2,
        "DENSE_TOP_K": 2,
        "RRF_TOP_K": 2,
                "PATIENT_HEADING_TOP_K": 3,
        "POLICY_BM25_TOP_K": 8,
        "POLICY_DENSE_TOP_K": 4,
        "POLICY_RERANK_TOP_K": 2,
        "SECTION_NEIGHBOR_WINDOW": 1,
        "RERANK_TOP_K": 5,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_bm25_exact_terminology_ranks_correctly() -> None:
    chunks = [
        _chunk("c1", "Coverage for ambulance transport services.", heading="Transport"),
        _chunk(
            "c2",
            "Indications for Permanent Cardiac Pacemaker Implantation.",
            heading="Cardiac",
        ),
        _chunk("c3", "General exclusions and limitations.", heading="Exclusions"),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(BM25_TOP_K=2, DENSE_TOP_K=2, RRF_TOP_K=2),
    )
    index = service.build_hybrid_index(chunks, index_kind="policy")
    bm25 = service._bm25_search(
        "Permanent Cardiac Pacemaker Implantation",
        index,
    )

    assert bm25[0].chunk.id == "c2"
    assert len(bm25) == 2


def test_bm25_preserves_numeric_cpt_tokens() -> None:
    chunks = [
        _chunk("c1", "Procedure code 99999 is unrelated."),
        _chunk("c2", "Authorized CPT 33207 for device implant."),
        _chunk("c3", "No numeric identifiers here."),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(BM25_TOP_K=3),
    )
    index = service.build_hybrid_index(chunks, index_kind="policy")
    bm25 = service._bm25_search("CPT 33207", index)

    assert bm25[0].chunk.id == "c2"
    assert "33207" in service.tokenize(index.retrieval_texts[1])


def test_bm25_top_k_respected() -> None:
    chunks = [_chunk(f"c{i}", f"policy text {i} coverage") for i in range(5)]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(BM25_TOP_K=3, DENSE_TOP_K=3, RRF_TOP_K=3),
    )
    index = service.build_hybrid_index(chunks, index_kind="policy")
    results = service._bm25_search("coverage", index)
    assert len(results) == 3


def test_empty_query_handled_safely() -> None:
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(),
    )
    index = service.build_hybrid_index(
        [_chunk("c1", "Some policy text")],
        index_kind="policy",
    )
    assert service._bm25_search("   ", index) == []
    assert service.retrieve_for_query("", index) == []
    assert service.retrieve_for_query("   ", index) == []
    assert service.policy_bm25_query(ProcedureExtraction()) == ""
    assert service.policy_dense_query(ProcedureExtraction()).startswith(". ")


def test_empty_chunk_list_handled_safely() -> None:
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(),
    )
    with pytest.raises(RetrievalError):
        service.build_hybrid_index([], index_kind="policy")


def test_dense_top_k_and_ordering() -> None:
    chunks = [
        _chunk("c1", "ambulance coverage rules"),
        _chunk("c2", "permanent cardiac pacemaker implantation CPT 33207"),
        _chunk("c3", "billing schedule"),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(DENSE_TOP_K=2),
    )
    index = service.build_hybrid_index(chunks, index_kind="policy")
    dense = service._dense_search(
        "Permanent Cardiac Pacemaker Implantation CPT 33207",
        index,
    )

    assert len(dense) == 2
    assert dense[0].chunk.id == "c2"
    assert dense[0].score >= dense[1].score


def test_embeddings_not_regenerated_on_retrieve() -> None:
    chunks = [_chunk("c1", "policy text one"), _chunk("c2", "policy text two")]
    embedding = _embedding_service()
    service = RetrievalService(
        embedding_service=embedding,
        settings=_settings(BM25_TOP_K=2, DENSE_TOP_K=2, RRF_TOP_K=2),
    )
    index = service.build_hybrid_index(chunks, index_kind="policy")
    embed_calls_after_index = embedding.embed.call_count

    procedure = ProcedureExtraction(
        procedure_name="Example Procedure",
        cpt_code="12345",
        is_future_procedure=True,
        confidence=0.9,
    )
    bm25_query = service.policy_bm25_query(procedure)
    service.retrieve_for_query(bm25_query, index)

    # One query embedding call only; corpus embeddings reused from index.
    assert embedding.embed.call_count == embed_calls_after_index + 1


def test_rrf_combines_both_rankings() -> None:
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(RRF_K=60, RRF_TOP_K=5),
    )
    chunks = [
        _chunk("a", "alpha"),
        _chunk("b", "beta"),
        _chunk("c", "gamma"),
    ]
    dense = [
        RetrievalResult(chunk=chunks[0], rank=1, score=0.9, retrieval_method="dense"),
        RetrievalResult(chunk=chunks[1], rank=2, score=0.8, retrieval_method="dense"),
    ]
    bm25 = [
        RetrievalResult(chunk=chunks[1], rank=1, score=5.0, retrieval_method="bm25"),
        RetrievalResult(chunk=chunks[2], rank=2, score=4.0, retrieval_method="bm25"),
    ]

    fused = service._rrf_fuse(
        dense_results=dense,
        bm25_results=bm25,
        chunks=chunks,
    )

    by_id = {item.chunk.id: item for item in fused}
    assert by_id["b"].rrf_score == pytest.approx(
        (1 / 62) + (1 / 61),
        rel=1e-6,
    )
    assert by_id["a"].bm25_rank is None
    assert by_id["a"].dense_rank == 1
    assert by_id["c"].dense_rank is None
    assert by_id["c"].bm25_rank == 2


def test_rrf_formula_and_top_k() -> None:
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(RRF_K=60, RRF_TOP_K=2),
    )
    chunks = [_chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")]
    dense = [
        RetrievalResult(chunk=chunks[0], rank=1, score=1.0, retrieval_method="dense"),
        RetrievalResult(chunk=chunks[1], rank=2, score=0.5, retrieval_method="dense"),
        RetrievalResult(chunk=chunks[2], rank=3, score=0.1, retrieval_method="dense"),
    ]
    fused = service._rrf_fuse(dense_results=dense, bm25_results=[], chunks=chunks)

    assert len(fused) == 2
    assert fused[0].chunk.id == "a"
    assert fused[0].rrf_score == pytest.approx(1 / 61)


def test_rrf_results_are_deterministic() -> None:
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(RRF_K=60, RRF_TOP_K=5, BM25_TOP_K=5, DENSE_TOP_K=5),
    )
    chunks = [
        _chunk("c1", "pacemaker implant policy text"),
        _chunk("c2", "ambulance transport policy text"),
        _chunk("c3", "billing and reimbursement rules"),
    ]
    index = service.build_hybrid_index(chunks, index_kind="policy")
    procedure = ProcedureExtraction(
        procedure_name="Pacemaker Implant",
        cpt_code="33207",
        is_future_procedure=True,
        confidence=0.8,
    )
    bm25_query = service.policy_bm25_query(procedure)
    first = service.retrieve_for_query(bm25_query, index)
    second = service.retrieve_for_query(bm25_query, index)
    assert [item.chunk.id for item in first] == [item.chunk.id for item in second]


def test_policy_queries_include_cpt_in_bm25_and_governing_rerank() -> None:
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(),
    )
    procedure = ProcedureExtraction(
        procedure_name="Example Procedure",
        cpt_code="12345",
    )
    bm25_query = service.policy_bm25_query(procedure)
    dense_query = service.policy_dense_query(procedure)
    rerank_query = service.policy_rerank_query(procedure)
    assert bm25_query == "Example Procedure 12345"
    assert "12345" not in dense_query
    assert "Example Procedure" in dense_query
    assert "Medical necessity" in dense_query
    assert "governing this exact planned procedure" in rerank_query
    assert "Example Procedure CPT 12345" in rerank_query


def test_policy_bm25_and_dense_helpers_respect_corpus_scope() -> None:
    sections = [
        _section(
            f"s{i}",
            (
                "Permanent Cardiac Pacemaker Implantation CPT 33207 indications."
                if i == 3
                else f"unrelated outpatient policy paragraph {i}"
            ),
            heading=(
                "Permanent Pacemaker Implantation" if i == 3 else f"Noise {i}"
            ),
        )
        for i in range(12)
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(POLICY_BM25_TOP_K=4, POLICY_DENSE_TOP_K=2),
    )
    procedure = ProcedureExtraction(
        procedure_name="Permanent Cardiac Pacemaker Implantation",
        cpt_code="33207",
    )
    bm25_sizes: list[int] = []
    dense_sizes: list[int] = []
    original_build = service.build_section_bm25_index
    original_dense = service._rank_texts_dense

    def spy_build(sections_in):
        bm25_sizes.append(len(sections_in))
        return original_build(sections_in)

    def spy_dense(query: str, texts: list[str], **kwargs) -> list[int]:
        dense_sizes.append(len(texts))
        return original_dense(query, texts, **kwargs)

    service.build_section_bm25_index = spy_build  # type: ignore[method-assign]
    service._rank_texts_dense = spy_dense  # type: ignore[method-assign]

    bm25_selected = service.bm25_select_sections(
        sections,
        query=service.policy_bm25_query(procedure),
        top_k=4,
    )
    assert bm25_sizes == [12]
    assert len(bm25_selected) == 4

    dense_selected = service.dense_rank_sections(
        bm25_selected,
        query=service.policy_dense_query(procedure),
        top_k=2,
    )
    assert dense_sizes == [4]
    assert len(dense_selected) == 2
    assert "s3" in {section.id for section in dense_selected}


def test_union_sections_keeps_cpt_hits_absent_from_bm25() -> None:
    cpt_only = _section("cpt", "CPT 33207 table row only.", heading="Codes")
    bm25_hit = _section(
        "bm25",
        "Permanent Cardiac Pacemaker Implantation criteria.",
        heading="Pacemaker",
    )
    duplicate = _section(
        "cpt",
        "CPT 33207 table row only.",
        heading="Codes",
    )
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(),
    )
    union = service.union_sections([cpt_only], [bm25_hit, duplicate])
    assert [section.id for section in union] == ["cpt", "bm25"]


def test_find_sections_with_cpt_exact_match() -> None:
    sections = [
        _section("s1", "Unrelated billing notes."),
        _section("s2", "Coverage for CPT 33207 permanent pacemaker."),
        _section("s3", "Near miss 332070 or 133207 should not match."),
        _section("s4", "Another mention of 33207 in criteria."),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(),
    )
    hits = service.find_sections_with_cpt(sections, "33207")
    assert [section.id for section in hits] == ["s2", "s4"]
    assert service.find_sections_with_cpt(sections, "") == []
    assert service.find_sections_with_cpt(sections, "99999") == []


def test_bm25_select_sections_ranks_full_corpus() -> None:
    """BM25 section selection over the full corpus (policy/evidence lexical recall)."""
    sections = [
        _section(
            "s1",
            "Allergy list and medication reconciliation.",
            heading="Allergies",
        ),
        _section(
            "s2",
            "Upcoming Intervention Plan: schedule permanent pacemaker.",
            heading="Upcoming Intervention Plan",
        ),
        _section("s3", "Vital signs within normal limits.", heading="Vitals"),
        _section("s4", "Past surgical history only.", heading="History"),
        _section("s5", "Labs reviewed today.", heading="Labs"),
        _section("s6", "Social history noncontributory.", heading="Social"),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(POLICY_BM25_TOP_K=2),
    )
    selected = service.bm25_select_sections(
        sections,
        query="planned procedure intervention pacemaker",
        top_k=2,
    )
    assert len(selected) == 2
    assert "s2" in {section.id for section in selected}


def test_heading_dense_receives_only_heading_strings() -> None:
    body_marker = "UNIQUE_BODY_TOKEN_SHOULD_NEVER_REACH_DENSE"
    parent = "Parent Heading Never Embedded"
    sections = [
        _section(
            "s1",
            f"{body_marker} allergies only.",
            heading="Allergies",
            parent_heading=parent,
        ),
        _section(
            "s2",
            f"{body_marker} schedule permanent pacemaker surgery.",
            heading="Upcoming Intervention Plan",
            parent_heading=parent,
        ),
        _section(
            "s3",
            f"{body_marker} vitals normal.",
            heading="Vitals",
            parent_heading=parent,
        ),
        _section(
            "s4",
            f"{body_marker} history only.",
            heading="History",
            parent_heading=parent,
        ),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(PATIENT_HEADING_TOP_K=2),
    )
    dense_inputs: list[list[str]] = []
    original_dense = service._rank_texts_dense

    def spy_dense(query: str, texts: list[str], **kwargs) -> list[int]:
        dense_inputs.append(list(texts))
        return original_dense(query, texts, **kwargs)

    service._rank_texts_dense = spy_dense  # type: ignore[method-assign]
    service.build_section_bm25_index = MagicMock(  # type: ignore[method-assign]
        wraps=service.build_section_bm25_index
    )

    selected = service.select_sections_by_heading_similarity(
        sections,
        query=service.procedure_heading_query(),
        top_k=2,
    )

    service.build_section_bm25_index.assert_not_called()
    assert len(dense_inputs) == 1
    assert dense_inputs[0] == [
        "Allergies",
        "Upcoming Intervention Plan",
        "Vitals",
        "History",
    ]
    assert all(body_marker not in text for text in dense_inputs[0])
    assert all(parent not in text for text in dense_inputs[0])
    assert all("Preview:" not in text for text in dense_inputs[0])
    assert all("Heading:" not in text for text in dense_inputs[0])
    assert len(selected) == 2
    assert "s2" in {section.id for section in selected}
    for section in selected:
        assert section.id
        assert section.heading
        assert section.start_page >= 1
        assert section.text.strip()


def test_heading_top_k_and_small_corpus() -> None:
    sections = [
        _section(f"s{i}", f"body {i}", heading=f"Plan {i}" if i < 3 else f"Noise {i}")
        for i in range(8)
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(PATIENT_HEADING_TOP_K=3),
    )
    selected = service.select_sections_by_heading_similarity(
        sections,
        query=service.procedure_heading_query(),
        top_k=3,
    )
    assert len(selected) == 3

    small = [
        _section("a", "planned procedure surgery", heading="Plan", page=1),
        _section("b", "vitals only", heading="Vitals", page=2),
    ]
    small_selected = service.select_sections_by_heading_similarity(
        small,
        query=service.procedure_heading_query(),
        top_k=10,
    )
    assert len(small_selected) == 2
    assert {section.id for section in small_selected} == {"a", "b"}


def test_heading_selection_empty_and_no_headings() -> None:
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(PATIENT_HEADING_TOP_K=5),
    )
    query = service.procedure_heading_query()
    assert service.select_sections_by_heading_similarity([], query=query) == []
    assert (
        service.select_sections_by_heading_similarity(
            [_section("blank", "   ", heading="Blank")],
            query=query,
        )
        == []
    )
    # No usable headings → clean empty result (no alternate path).
    no_headings = [
        _section("p1", "planned procedure scheduled surgery", heading=None),
        _section("p2", "vitals normal", heading="   "),
    ]
    assert (
        service.select_sections_by_heading_similarity(no_headings, query=query) == []
    )


def test_heading_selection_is_deterministic() -> None:
    sections = [
        _section("s1", "body one", heading="Planned Procedure"),
        _section("s2", "body two", heading="Planned Procedure"),
        _section("s3", "body three", heading="Planned Procedure"),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(PATIENT_HEADING_TOP_K=3),
    )
    query = service.procedure_heading_query()
    first = service.select_sections_by_heading_similarity(sections, query=query)
    second = service.select_sections_by_heading_similarity(sections, query=query)
    assert [section.id for section in first] == [section.id for section in second]


def test_expand_section_context_retains_neighbors_and_parent_child() -> None:
    sections = [
        _section("s0", "Intro", heading="Coverage", page=1),
        _section(
            "s1",
            "Indications body",
            heading="Indications",
            parent_heading="Coverage",
            page=2,
        ),
        _section(
            "s2",
            "Child criterion",
            heading="Criterion A",
            parent_heading="Indications",
            page=3,
        ),
        _section("s3", "Neighbor after", heading="Exclusions", page=4),
        _section("s4", "Far away", heading="Appendix", page=5),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(SECTION_NEIGHBOR_WINDOW=1),
    )
    expanded = service.expand_section_context([sections[1]], sections)
    expanded_ids = [section.id for section in expanded]
    # Selected s1 + neighbor window + local same-topic child; peer Exclusions stops.
    assert expanded_ids == ["s0", "s1", "s2"]
    assert "s4" not in expanded_ids


def test_expand_section_context_excludes_rejected_candidates() -> None:
    sections = [
        _section("s0", "Intro", heading="Coverage", page=1),
        _section(
            "s1",
            "Indications body",
            heading="Indications",
            parent_heading="Coverage",
            page=2,
        ),
        _section(
            "s2",
            "Child criterion",
            heading="Criterion A",
            parent_heading="Indications",
            page=3,
        ),
        _section("s3", "Rejected CPT neighbor", heading="CRT-P", page=4),
        _section("s4", "Far away", heading="Appendix", page=5),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(SECTION_NEIGHBOR_WINDOW=1),
    )
    expanded = service.expand_section_context(
        [sections[1]],
        sections,
        exclude_ids={"s3"},
    )
    assert [section.id for section in expanded] == ["s0", "s1", "s2"]
    assert "s3" not in {section.id for section in expanded}


def test_expand_keeps_continuation_across_running_headers() -> None:
    """Governing CRID section keeps contiguous continuation past page headers."""
    sections = [
        _section(
            "s1743",
            "CID wrapper",
            heading="Cardiac Implantable Devices (CID)",
            page=697,
        ),
        _section(
            "s1744",
            "Indications for Permanent Pacemaker (CRID-7)\nSinus node dysfunction...",
            heading="Indications for Permanent Pacemaker (CRID-7)",
            page=697,
        ),
        _section(
            "s1745",
            "• Symptomatic AVB continuation criteria across page break",
            heading="Cardiac Implantable Devices (CID)",
            page=698,
        ),
        _section(
            "s1746",
            "Neuromuscular diseases known to involve the heart",
            heading="Cardiac Implantable Devices (CID)",
            page=700,
        ),
        _section(
            "s1747",
            "Permanent pacemaker implantation is not indicated...",
            heading=(
                "Permanent Pacemaker Implantation - Non-indications (CRID-9)"
            ),
            page=700,
        ),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(SECTION_NEIGHBOR_WINDOW=1),
    )
    # 1745/1746 were rejected by rerank but remain structural continuation.
    expanded = service.expand_section_context(
        [sections[1]],
        sections,
        exclude_ids={"s1745", "s1746", "s1747"},
    )
    expanded_ids = [section.id for section in expanded]
    assert "s1744" in expanded_ids
    assert "s1745" in expanded_ids
    assert "s1746" in expanded_ids
    assert "s1747" not in expanded_ids


def test_expand_from_topic_anchor_recovers_full_crid_block() -> None:
    """Parent topic without CRID still recovers governing indications block."""
    sections = [
        _section(
            "s1739",
            "Permanent Pacemaker Implantation",
            heading="Permanent Pacemaker Implantation",
            page=697,
        ),
        _section(
            "s1740",
            "(CRID-7)",
            heading="(CRID-7)",
            page=697,
        ),
        _section(
            "s1741",
            "CID.PM.111.A v1.0.2025",
            heading="CID.PM.111.A",
            page=697,
        ),
        _section(
            "s1742",
            "Codes included Description CPT 33206 33207",
            heading="Codes included",
            page=697,
        ),
        _section(
            "s1743",
            "Cardiac Implantable Devices (CID)",
            heading="Cardiac Implantable Devices (CID)",
            page=697,
        ),
        _section(
            "s1744",
            (
                "Indications for Permanent Pacemaker (CRID-7)\n"
                "Sinus bradycardia at rate <50 beats per minute\n"
                "Sinus pauses >3 seconds"
            ),
            heading="Indications for Permanent Pacemaker (CRID-7)",
            page=697,
        ),
        _section(
            "s1745",
            "Symptomatic AVB continuation criteria",
            heading="Cardiac Implantable Devices (CID)",
            page=698,
        ),
        _section(
            "s1747",
            "Permanent pacemaker implantation is not indicated...",
            heading=(
                "Permanent Pacemaker Implantation - Non-indications (CRID-9)"
            ),
            page=700,
        ),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(SECTION_NEIGHBOR_WINDOW=1),
    )
    # Only the topic title survives rerank; indications/code-table sections lost.
    expanded = service.expand_section_context(
        [sections[0]],
        sections,
        exclude_ids={
            "s1740",
            "s1741",
            "s1742",
            "s1743",
            "s1744",
            "s1745",
            "s1747",
        },
    )
    expanded_ids = {section.id for section in expanded}
    assert "s1739" in expanded_ids
    assert "s1740" in expanded_ids
    assert "s1742" in expanded_ids
    assert "s1744" in expanded_ids
    assert "s1745" in expanded_ids
    assert "s1747" not in expanded_ids
    joined = "\n".join(section.text for section in expanded)
    assert "Sinus bradycardia at rate <50 beats per minute" in joined
    assert "Sinus pauses >3 seconds" in joined


def test_expand_never_pulls_distant_same_named_headings() -> None:
    """Same-named distant Indications children must not enter context."""
    sections = [
        _section(
            "amyloid",
            "known systemic amyloidosis Cardiac MRI",
            heading="transplant when both",
            parent_heading="Indications",
            page=644,
        ),
        _section("gap1", "unrelated mid doc", heading="Other Topic", page=650),
        _section(
            "crid7",
            "Indications for Permanent Pacemaker (CRID-7) sinus node",
            heading="Indications for Permanent Pacemaker (CRID-7)",
            page=697,
        ),
        _section(
            "leadless",
            "Leadless right ventricular pacemaker (CRID-11.1.1)",
            heading="Indications",
            page=703,
        ),
    ]
    service = RetrievalService(
        embedding_service=_embedding_service(),
        settings=_settings(SECTION_NEIGHBOR_WINDOW=0),
    )
    expanded = service.expand_section_context(
        [sections[2], sections[3]],
        sections,
    )
    expanded_ids = {section.id for section in expanded}
    assert "crid7" in expanded_ids
    assert "amyloid" not in expanded_ids

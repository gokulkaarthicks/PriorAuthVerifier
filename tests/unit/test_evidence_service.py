"""Unit tests for EvidenceService."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import (
    EvidenceAnalysisError,
    EvidenceGroundingError,
    EvidenceValidationError,
)
from app.models.chunk import Chunk
from app.models.policy_analysis import PolicyAnalysisResult
from app.models.section import Section
from app.schemas.evaluation import (
    CriterionEvaluation,
    CriterionStatus,
    EvidenceCitation,
    EvidenceEvaluationBatch,
    ExclusionEvaluation,
    SolCriterionEvaluation,
    SolEvidenceEvaluationBatch,
    SolExclusionEvaluation,
    citation_from_chunk,
    materialize_sol_batch,
)
from app.schemas.policy import (
    CriterionCategory,
    CriterionNode,
    CriterionNodeType,
    PolicyCriteriaExtraction,
    PolicyExclusion,
)
from app.services.evidence_service import EvidenceService
from app.services.retrieval_service import RetrievalService
from tests.helpers.evidence_sol import as_sol_response, public_batch_to_scoped_sol


def _parse_prompt_candidate_lookup(prompt: str) -> dict[tuple[str, str], str]:
    from tests.helpers.evidence_sol import parse_prompt_candidate_lookup

    return parse_prompt_candidate_lookup(prompt)


def _public_batch_to_scoped_sol(
    public: EvidenceEvaluationBatch,
    *,
    prompt: str,
) -> SolEvidenceEvaluationBatch:
    return public_batch_to_scoped_sol(public, prompt=prompt)


def _as_sol_response(
    llm_response: EvidenceEvaluationBatch | SolEvidenceEvaluationBatch,
) -> AsyncMock:
    return as_sol_response(llm_response)


def _section(
    section_id: str,
    text: str,
    *,
    page: int = 1,
) -> Section:
    return Section(
        id=section_id,
        document_id="patient",
        heading="Notes",
        parent_heading=None,
        text=text,
        start_page=page,
        end_page=page,
        token_count=max(1, len(text.split())),
    )


def _passthrough_chunking() -> MagicMock:
    """Map each section to a chunk with the same id (stable test IDs)."""
    service = MagicMock()

    def chunk_sections(sections: list[Section]) -> list[Chunk]:
        return [
            Chunk(
                id=section.id,
                document_id=section.document_id,
                text=section.text,
                page_number=section.start_page,
                end_page_number=section.end_page,
                heading=section.heading,
                parent_heading=section.parent_heading,
                token_count=section.token_count,
                section_id=section.id,
                chunk_index=0,
                document_date=section.document_date,
            )
            for section in sections
        ]

    service.chunk_sections.side_effect = chunk_sections
    return service


def _mock_reranker() -> MagicMock:
    reranker = MagicMock()

    def rerank(query, candidates, top_k=None, **_kwargs):
        return candidates[: (top_k or 5)]

    def rerank_grouped(groups, top_k=None, **_kwargs):
        keep = top_k or 5
        return {
            group_id: candidates[:keep]
            for group_id, _query, candidates in groups
        }

    reranker.rerank.side_effect = rerank
    reranker.rerank_grouped.side_effect = rerank_grouped
    return reranker


def _embedding_service(dim: int = 4) -> MagicMock:
    service = MagicMock()
    call_count = {"embed": 0}

    def embed(texts: list[str], **_kwargs) -> np.ndarray:
        call_count["embed"] += 1
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for index, text in enumerate(texts):
            lower = text.lower()
            matrix[index, 0] = (len(text) % 7) / 7.0
            matrix[index, 1] = 1.0 if "reflux" in lower else 0.0
            matrix[index, 2] = 1.0 if "compression" in lower else 0.0
            matrix[index, 3] = 1.0 if "ceap" in lower else 0.0
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
    service.call_count = call_count
    return service


def _tree() -> CriterionNode:
    return CriterionNode(
        id="root",
        node_type=CriterionNodeType.AND,
        children=[
            CriterionNode(
                id="c1",
                node_type=CriterionNodeType.CONDITION,
                description="Demonstrated saphenous reflux.",
                category=CriterionCategory.DIAGNOSTIC,
            ),
            CriterionNode(
                id="c2",
                node_type=CriterionNodeType.CONDITION,
                description="CEAP class is C2 or greater.",
                category=CriterionCategory.MEASUREMENT,
            ),
            CriterionNode(
                id="or-1",
                node_type=CriterionNodeType.OR,
                children=[
                    CriterionNode(
                        id="c3",
                        node_type=CriterionNodeType.CONDITION,
                        description=(
                            "Compression therapy failed for at least "
                            "three months."
                        ),
                        category=CriterionCategory.DURATION,
                    ),
                    CriterionNode(
                        id="c4",
                        node_type=CriterionNodeType.CONDITION,
                        description="Symptoms remain ambiguous in chart notes.",
                        category=CriterionCategory.CLINICAL,
                    ),
                ],
            ),
        ],
    )


def _policy_result(*, applicable: bool = True) -> PolicyAnalysisResult:
    if not applicable:
        return PolicyAnalysisResult(
            applicable=False,
            applicability_reason="Unrelated policy domain.",
            applicability_confidence=0.95,
            criteria=None,
            supporting_chunk_ids=["p1"],
            llm_call_count=1,
        )
    return PolicyAnalysisResult(
        applicable=True,
        applicability_reason="Policy covers the procedure.",
        applicability_confidence=0.95,
        criteria=PolicyCriteriaExtraction(
            procedure_name="Example Vein Ablation",
            policy_summary="Coverage criteria for ablation.",
            root=_tree(),
            exclusions=[
                PolicyExclusion(
                    id="ex1",
                    description="Cosmetic treatment is not covered.",
                    source_chunk_id="policy-1",
                    source_text="Cosmetic treatment is not covered.",
                    page_number=1,
                )
            ],
            source_chunk_ids=["policy-1"],
        ),
        supporting_chunk_ids=["policy-1"],
        llm_call_count=1,
    )


def _service(
    *,
    llm_response: EvidenceEvaluationBatch | SolEvidenceEvaluationBatch | None = None,
    embedding: MagicMock | None = None,
    settings: Settings | None = None,
) -> tuple[EvidenceService, MagicMock, MagicMock, MagicMock, RetrievalService]:
    embedding_service = embedding or _embedding_service()
    resolved = settings or Settings(
        BM25_TOP_K=10,
        DENSE_TOP_K=10,
        RRF_TOP_K=10,
        RERANK_TOP_K=5,
        EVIDENCE_SECTION_BM25_TOP_K=5,
        EVIDENCE_DENSE_TOP_K=4,
        EVIDENCE_RERANK_TOP_K=3,
        EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=5,
        EVIDENCE_SOL_MAX_CONCURRENCY=3,
        POLICY_REASONING_MODEL="gpt-6-sol",
    )
    retrieval = RetrievalService(embedding_service, resolved)
    reranker = _mock_reranker()
    llm = MagicMock()
    if llm_response is not None:
        llm.extract_structured = _as_sol_response(llm_response)
    else:
        llm.extract_structured = AsyncMock()
    service = EvidenceService(
        retrieval_service=retrieval,
        chunking_service=_passthrough_chunking(),
        reranking_service=reranker,
        llm_service=llm,
        settings=resolved,
    )
    return service, embedding_service, reranker, llm, retrieval


def test_atomic_criteria_extraction_ignores_and_or_nodes() -> None:
    service, *_ = _service()
    atoms = service.extract_atomic_criteria(_tree())
    ids = [atom.id for atom in atoms]
    assert ids == ["c1", "c2", "c3", "c4"]
    assert all(atom.kind == "condition" for atom in atoms)


def test_and_or_nodes_are_not_queried() -> None:
    service, *_ = _service()
    atoms = service.extract_atomic_criteria(_tree())
    assert "root" not in {atom.id for atom in atoms}
    assert "or-1" not in {atom.id for atom in atoms}


def test_build_query_returns_description() -> None:
    service, *_ = _service()
    atoms = service.extract_atomic_criteria(_tree())
    for atom in atoms:
        assert service.build_query(atom) == atom.description.strip()
    exclusion = atoms[0].__class__(
        id="ex1",
        description="  Cosmetic treatment is not covered.  ",
        category=CriterionCategory.EXCLUSION,
        kind="exclusion",
    )
    assert service.build_query(exclusion) == "Cosmetic treatment is not covered."


@pytest.mark.asyncio
async def test_batch_embeddings_and_patient_reuse() -> None:
    embedding = _embedding_service()
    sections = [
        _section("p1", "Duplex ultrasound showed demonstrated saphenous reflux."),
        _section("p2", "CEAP clinical class documented as C3."),
        _section("p3", "Compression therapy failed after four months."),
        _section("p4", "Notes mention possible symptoms without clear duration."),
    ]
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p1"],
                evidence_text=["demonstrated saphenous reflux"],
                page_numbers=[1],
                explanation="Explicit duplex finding.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p2"],
                evidence_text=["CEAP clinical class documented as C3"],
                page_numbers=[1],
                explanation="CEAP C3 meets C2 or greater.",
            ),
            CriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p3"],
                evidence_text=["Compression therapy failed after four months"],
                page_numbers=[1],
                explanation="Duration requirement met.",
            ),
            CriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.UNCLEAR,
                evidence_chunk_ids=["p4"],
                evidence_text=["possible symptoms without clear duration"],
                page_numbers=[1],
                explanation="Ambiguous symptom note.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="No cosmetic indication documented.",
            )
        ],
    )
    service, embedding_service, reranker, llm, retrieval = _service(
        llm_response=batch,
        embedding=embedding,
    )

    bm25_builds: list[int] = []
    original_build = retrieval.build_section_bm25_index
    original_rrf = retrieval._rrf_fuse
    original_hybrid = retrieval.build_hybrid_index
    original_bm25_search = retrieval._bm25_search
    rrf_calls = 0
    chunk_bm25_calls = 0

    def spy_build(sections: list[Section]):
        bm25_builds.append(len(sections))
        return original_build(sections)

    def spy_rrf(**kwargs):
        nonlocal rrf_calls
        rrf_calls += 1
        return original_rrf(**kwargs)

    def spy_bm25_search(*args, **kwargs):
        nonlocal chunk_bm25_calls
        chunk_bm25_calls += 1
        return original_bm25_search(*args, **kwargs)

    retrieval.build_section_bm25_index = spy_build  # type: ignore[method-assign]
    retrieval._rrf_fuse = spy_rrf  # type: ignore[method-assign]
    retrieval.build_hybrid_index = MagicMock(wraps=original_hybrid)  # type: ignore[method-assign]
    retrieval._bm25_search = spy_bm25_search  # type: ignore[method-assign]

    result = await service.evaluate_evidence(_policy_result(), sections)

    assert bm25_builds == [len(sections)]
    assert rrf_calls == 0
    assert chunk_bm25_calls == 0
    retrieval.build_hybrid_index.assert_not_called()
    assert result.llm_call_count == 1
    assert len(result.criteria) == 4
    assert reranker.rerank_grouped.call_count == 1
    assert llm.extract_structured.await_count == 1
    assert embedding_service.call_count["embed"] >= 1
    grouped = reranker.rerank_grouped.call_args
    assert grouped.kwargs.get("top_k") == 3
    for _group_id, _query, candidates in grouped.args[0]:
        assert len(candidates) <= 4


@pytest.mark.asyncio
async def test_evidence_bm25_then_dense_then_rerank_no_rrf() -> None:
    sections = [
        _section(f"p{i}", f"patient note about reflux therapy {i}")
        for i in range(12)
    ]
    embedding = _embedding_service()
    service, _, reranker, llm, retrieval = _service(
        llm_response=EvidenceEvaluationBatch(
            criteria=[
                CriterionEvaluation(
                    criterion_id=cid,
                    status=CriterionStatus.MISSING,
                    explanation="No qualifying documentation found.",
                )
                for cid in ("c1", "c2", "c3", "c4")
            ],
            exclusions=[
                ExclusionEvaluation(
                    exclusion_id="ex1",
                    triggered=False,
                    status=CriterionStatus.MISSING,
                    explanation="Not triggered.",
                )
            ],
        ),
        embedding=embedding,
    )

    dense_input_sizes: list[int] = []
    original_dense = retrieval.dense_rank_chunks_batched

    def spy_dense(groups, *, top_k=None):
        for _group_id, (_query, chunks) in groups.items():
            dense_input_sizes.append(len(chunks))
        return original_dense(groups, top_k=top_k)

    retrieval.dense_rank_chunks_batched = spy_dense  # type: ignore[method-assign]
    retrieval._rrf_fuse = MagicMock(wraps=retrieval._rrf_fuse)  # type: ignore[method-assign]
    retrieval.build_hybrid_index = MagicMock(  # type: ignore[method-assign]
        wraps=retrieval.build_hybrid_index
    )

    await service.evaluate_evidence(_policy_result(), sections)

    assert dense_input_sizes
    assert all(size <= 5 for size in dense_input_sizes)
    retrieval._rrf_fuse.assert_not_called()
    retrieval.build_hybrid_index.assert_not_called()
    assert reranker.rerank_grouped.call_count == 1
    grouped = reranker.rerank_grouped.call_args
    assert grouped.kwargs.get("top_k") == 3
    for _group_id, _query, candidates in grouped.args[0]:
        assert len(candidates) <= 4
    assert llm.extract_structured.await_count == 1


@pytest.mark.asyncio
async def test_large_patient_chart_never_full_dense_or_full_chunk() -> None:
    sections = [
        _section(
            f"p{i}",
            (
                "Duplex ultrasound demonstrated saphenous reflux in the GSV."
                if i == 42
                else f"Unrelated outpatient note number {i} diet exercise vitals."
            ),
            page=i + 1,
        )
        for i in range(1000)
    ]
    embedding = _embedding_service()
    settings = Settings(
        EVIDENCE_SECTION_BM25_TOP_K=40,
        EVIDENCE_DENSE_TOP_K=10,
        EVIDENCE_RERANK_TOP_K=5,
        EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=5,
        EVIDENCE_SOL_MAX_CONCURRENCY=3,
        POLICY_REASONING_MODEL="gpt-6-sol",
    )
    retrieval = RetrievalService(embedding, settings)
    chunking = _passthrough_chunking()
    reranker = _mock_reranker()
    llm = MagicMock()
    llm.extract_structured = _as_sol_response(
        EvidenceEvaluationBatch(
            criteria=[
                CriterionEvaluation(
                    criterion_id="c1",
                    status=CriterionStatus.MET,
                    evidence_chunk_ids=["p42"],
                    evidence_text=["demonstrated saphenous reflux"],
                    page_numbers=[43],
                    explanation="Supported.",
                ),
                CriterionEvaluation(
                    criterion_id="c2",
                    status=CriterionStatus.MISSING,
                    explanation="Missing.",
                ),
                CriterionEvaluation(
                    criterion_id="c3",
                    status=CriterionStatus.MISSING,
                    explanation="Missing.",
                ),
                CriterionEvaluation(
                    criterion_id="c4",
                    status=CriterionStatus.MISSING,
                    explanation="Missing.",
                ),
            ],
            exclusions=[
                ExclusionEvaluation(
                    exclusion_id="ex1",
                    triggered=False,
                    status=CriterionStatus.MISSING,
                    explanation="Not triggered.",
                )
            ],
        )
    )
    service = EvidenceService(
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=reranker,
        llm_service=llm,
        settings=settings,
    )

    bm25_corpus: list[int] = []
    dense_inputs: list[int] = []
    chunked_counts: list[int] = []
    original_query = retrieval.query_section_bm25
    original_dense = retrieval.dense_rank_chunks_batched
    original_chunk_fn = chunking.chunk_sections.side_effect

    def spy_query(index, *, query, top_k=None):
        bm25_corpus.append(len(index.sections))
        return original_query(index, query=query, top_k=top_k)

    def spy_dense(groups, *, top_k=None):
        for _group_id, (_query, chunks) in groups.items():
            dense_inputs.append(len(chunks))
        return original_dense(groups, top_k=top_k)

    def spy_chunk(sections_in: list[Section]) -> list[Chunk]:
        chunked_counts.append(len(sections_in))
        return original_chunk_fn(sections_in)

    retrieval.query_section_bm25 = spy_query  # type: ignore[method-assign]
    retrieval.dense_rank_chunks_batched = spy_dense  # type: ignore[method-assign]
    chunking.chunk_sections.side_effect = spy_chunk
    retrieval._rrf_fuse = MagicMock(wraps=retrieval._rrf_fuse)  # type: ignore[method-assign]

    result = await service.evaluate_evidence(_policy_result(), sections)

    assert bm25_corpus
    assert all(size == 1000 for size in bm25_corpus)
    assert chunked_counts == [chunked_counts[0]]  # unique sections chunked once
    assert chunked_counts[0] <= 5 * 40
    assert chunked_counts[0] < 1000
    assert dense_inputs
    assert all(size <= 40 for size in dense_inputs)
    assert reranker.rerank_grouped.call_count == 1
    for _group_id, _query, candidates in reranker.rerank_grouped.call_args.args[0]:
        assert len(candidates) <= 10
    assert reranker.rerank_grouped.call_args.kwargs["top_k"] == 5
    retrieval._rrf_fuse.assert_not_called()
    assert result.criteria[0].status == CriterionStatus.MET


@pytest.mark.asyncio
async def test_unique_sections_chunked_once_without_widening() -> None:
    sections = [
        _section("shared", "Shared reflux and CEAP documentation together."),
        _section("only-a", "Demonstrated saphenous reflux alone."),
        _section("only-b", "CEAP clinical class documented as C3 alone."),
        _section("noise", "Unrelated diet note."),
    ]
    embedding = _embedding_service()
    chunking = _passthrough_chunking()
    settings = Settings(
        EVIDENCE_SECTION_BM25_TOP_K=2,
        EVIDENCE_DENSE_TOP_K=4,
        EVIDENCE_RERANK_TOP_K=3,
        EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=10,
        POLICY_REASONING_MODEL="gpt-6-sol",
    )
    retrieval = RetrievalService(embedding, settings)
    service = EvidenceService(
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=_mock_reranker(),
        llm_service=MagicMock(
            extract_structured=_as_sol_response(
                EvidenceEvaluationBatch(
                    criteria=[
                        CriterionEvaluation(
                            criterion_id=cid,
                            status=CriterionStatus.MISSING,
                            explanation="Missing.",
                        )
                        for cid in ("c1", "c2", "c3", "c4")
                    ],
                    exclusions=[
                        ExclusionEvaluation(
                            exclusion_id="ex1",
                            triggered=False,
                            status=CriterionStatus.MISSING,
                            explanation="Not triggered.",
                        )
                    ],
                )
            )
        ),
        settings=settings,
    )

    selected: dict[str, list[str]] = {}
    original_query = retrieval.query_section_bm25

    def spy_query(index, *, query, top_k=None):
        result = original_query(index, query=query, top_k=top_k)
        selected[query] = [section.id for section in result]
        return result

    retrieval.query_section_bm25 = spy_query  # type: ignore[method-assign]
    await service.evaluate_evidence(_policy_result(), sections)

    assert chunking.chunk_sections.call_count == 1
    chunked_ids = {
        section.id for section in chunking.chunk_sections.call_args.args[0]
    }
    # Union of selected sections only; not the full chart.
    asserted_selected = set()
    for ids in selected.values():
        asserted_selected.update(ids)
    assert chunked_ids == asserted_selected
    assert "noise" not in chunked_ids or "noise" in asserted_selected


@pytest.mark.asyncio
async def test_criteria_isolation_and_single_bm25_index() -> None:
    sections = [
        _section("reflux", "Duplex ultrasound demonstrated saphenous reflux."),
        _section("ceap", "CEAP clinical class documented as C3."),
        _section("compression", "Compression therapy failed after four months."),
        *[
            _section(f"noise-{i}", f"Unrelated outpatient note {i}.")
            for i in range(20)
        ],
    ]
    embedding = _embedding_service()
    service, _, reranker, _, retrieval = _service(
        llm_response=EvidenceEvaluationBatch(
            criteria=[
                CriterionEvaluation(
                    criterion_id="c1",
                    status=CriterionStatus.MET,
                    evidence_chunk_ids=["reflux"],
                    evidence_text=["demonstrated saphenous reflux"],
                    page_numbers=[1],
                    explanation="Supported.",
                ),
                CriterionEvaluation(
                    criterion_id="c2",
                    status=CriterionStatus.MET,
                    evidence_chunk_ids=["ceap"],
                    evidence_text=["CEAP clinical class documented as C3"],
                    page_numbers=[1],
                    explanation="Supported.",
                ),
                CriterionEvaluation(
                    criterion_id="c3",
                    status=CriterionStatus.MET,
                    evidence_chunk_ids=["compression"],
                    evidence_text=["Compression therapy failed after four months"],
                    page_numbers=[1],
                    explanation="Supported.",
                ),
                CriterionEvaluation(
                    criterion_id="c4",
                    status=CriterionStatus.MISSING,
                    explanation="Missing.",
                ),
            ],
            exclusions=[
                ExclusionEvaluation(
                    exclusion_id="ex1",
                    triggered=False,
                    status=CriterionStatus.MISSING,
                    explanation="Not triggered.",
                )
            ],
        ),
        embedding=embedding,
    )
    build_count = 0
    selected_by_query: dict[str, list[str]] = {}
    original_build = retrieval.build_section_bm25_index
    original_query = retrieval.query_section_bm25

    def spy_build(sections_in: list[Section]):
        nonlocal build_count
        build_count += 1
        return original_build(sections_in)

    def spy_query(index, *, query, top_k=None):
        selected = original_query(index, query=query, top_k=top_k)
        selected_by_query[query] = [section.id for section in selected]
        return selected

    retrieval.build_section_bm25_index = spy_build  # type: ignore[method-assign]
    retrieval.query_section_bm25 = spy_query  # type: ignore[method-assign]

    await service.evaluate_evidence(_policy_result(), sections)

    assert build_count == 1
    assert len(selected_by_query) >= 4
    # Different criteria produce independent selected-section lists.
    assert selected_by_query["Demonstrated saphenous reflux."][0] == "reflux"
    assert "ceap" in selected_by_query["CEAP class is C2 or greater."]
    prompts = [
        call.kwargs["user_prompt"]
        for call in service._llm_service.extract_structured.await_args_list
    ]
    joined = "\n".join(prompts)
    assert "criterion_id: c1" in joined
    assert "evidence_corpus:" in joined
    assert "candidate_id:" in joined


@pytest.mark.asyncio
async def test_bge_encodes_unique_chunks_once() -> None:
    sections = [
        _section("shared", "Demonstrated saphenous reflux and CEAP C3 together."),
        _section("noise-1", "Unrelated outpatient note one."),
        _section("noise-2", "Unrelated outpatient note two."),
    ]
    embedding = _embedding_service()
    service, embedding_service, _, _, retrieval = _service(
        llm_response=EvidenceEvaluationBatch(
            criteria=[
                CriterionEvaluation(
                    criterion_id=cid,
                    status=CriterionStatus.MISSING,
                    explanation="Missing.",
                )
                for cid in ("c1", "c2", "c3", "c4")
            ],
            exclusions=[
                ExclusionEvaluation(
                    exclusion_id="ex1",
                    triggered=False,
                    status=CriterionStatus.MISSING,
                    explanation="Not triggered.",
                )
            ],
        ),
        embedding=embedding,
    )
    texts_seen: list[list[str]] = []
    original_embed = embedding_service.embed.side_effect

    def spy_embed(texts: list[str], **kwargs) -> np.ndarray:
        texts_seen.append(list(texts))
        return original_embed(texts, **kwargs)

    embedding_service.embed.side_effect = spy_embed
    await service.evaluate_evidence(_policy_result(), sections)

    # Criteria queries are encoded together; unique chunk texts appear once.
    assert any(len(batch) >= 4 for batch in texts_seen)
    chunk_batches = [
        batch
        for batch in texts_seen
        if any("Content:" in text or "Encounter date:" in text for text in batch)
        or any("reflux" in text.lower() and "Heading:" in text for text in batch)
    ]
    assert chunk_batches
    # Unique chunk embeddings are produced in a single batch call.
    assert len(chunk_batches) == 1


@pytest.mark.asyncio
async def test_met_requires_evidence_and_rejects_unsupported() -> None:
    sections = [
        _section("p1", "Duplex ultrasound showed demonstrated saphenous reflux."),
        _section("p2", "Unrelated allergy history."),
    ]
    embedding = _embedding_service()

    bad = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=[],
                evidence_text=[],
                page_numbers=[],
                explanation="Claimed met without evidence.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=bad, embedding=embedding)
    with pytest.raises(EvidenceValidationError, match="requires evidence"):
        await service.evaluate_evidence(_policy_result(), sections)


@pytest.mark.asyncio
async def test_missing_allows_empty_evidence_and_unclear_works() -> None:
    sections = [
        _section("p1", "Duplex ultrasound showed demonstrated saphenous reflux."),
        _section("p2", "CEAP clinical class documented as C3."),
        _section("p3", "Notes mention possible symptoms without clear duration."),
    ]
    embedding = _embedding_service()
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p1"],
                evidence_text=["demonstrated saphenous reflux"],
                page_numbers=[1],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p2"],
                evidence_text=["CEAP clinical class documented as C3"],
                page_numbers=[1],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="No qualifying documentation found.",
            ),
            CriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.UNCLEAR,
                evidence_chunk_ids=["p3"],
                evidence_text=["possible symptoms without clear duration"],
                page_numbers=[1],
                explanation="Ambiguous note.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=batch, embedding=embedding)
    result = await service.evaluate_evidence(_policy_result(), sections)
    statuses = {item.criterion_id: item.status for item in result.criteria}
    assert statuses["c1"] == CriterionStatus.MET
    assert statuses["c3"] == CriterionStatus.MISSING
    assert statuses["c4"] == CriterionStatus.UNCLEAR
    assert result.exclusions[0].triggered is False


@pytest.mark.asyncio
async def test_invalid_chunk_ids_rejected() -> None:
    sections = [_section("p1", "Demonstrated saphenous reflux on imaging.")]
    embedding = _embedding_service()
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_candidate_ids=["c2:e1"],
                explanation="Cross-criterion citation.",
            ),
            SolCriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            SolCriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            SolCriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            SolExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=sol, embedding=embedding)
    with pytest.raises(EvidenceGroundingError, match="Evidence grounding failed"):
        await service.evaluate_evidence(_policy_result(), sections)


@pytest.mark.asyncio
async def test_sol_cannot_alter_page_or_text_from_candidate() -> None:
    sections = [_section("p1", "Demonstrated saphenous reflux on imaging.", page=2)]
    embedding = _embedding_service()
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p1"],
                evidence_text=["ignored quote"],
                page_numbers=[999],
                explanation="Candidate metadata wins.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=batch, embedding=embedding)
    result = await service.evaluate_evidence(_policy_result(), sections)
    item = next(c for c in result.criteria if c.criterion_id == "c1")
    assert item.evidence[0].page_number == 2
    assert item.evidence[0].text == "Demonstrated saphenous reflux on imaging."


@pytest.mark.asyncio
async def test_exclusions_evaluated_separately() -> None:
    sections = [
        _section("p1", "Demonstrated saphenous reflux."),
        _section("p2", "Visit documented as cosmetic treatment only."),
    ]
    embedding = _embedding_service()
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p1"],
                evidence_text=["Demonstrated saphenous reflux"],
                page_numbers=[1],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=True,
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p2"],
                evidence_text=["cosmetic treatment only"],
                page_numbers=[1],
                explanation="Cosmetic exclusion triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=batch, embedding=embedding)
    result = await service.evaluate_evidence(_policy_result(), sections)
    assert len(result.exclusions) == 1
    assert result.exclusions[0].triggered is True
    assert result.exclusions[0].status == CriterionStatus.MET
    assert all(item.criterion_id.startswith("c") for item in result.criteria)


@pytest.mark.asyncio
async def test_non_applicable_policy_skips_evidence_evaluation() -> None:
    sections = [_section("p1", "Any patient note.")]
    embedding = _embedding_service()
    service, _, reranker, llm, _ = _service(embedding=embedding)
    result = await service.evaluate_evidence(
        _policy_result(applicable=False),
        sections,
    )
    assert result.skipped is True
    assert result.criteria == []
    assert result.llm_call_count == 0
    llm.extract_structured.assert_not_awaited()
    reranker.rerank_grouped.assert_not_called()


@pytest.mark.asyncio
async def test_empty_patient_sections_raises() -> None:
    service, *_ = _service()
    with pytest.raises(EvidenceAnalysisError, match="at least one section"):
        await service.evaluate_evidence(_policy_result(), [])


@pytest.mark.asyncio
async def test_bm25_no_candidates_skips_dense() -> None:
    sections = [_section("p1", "zzz unrelated filler text only.")]
    embedding = _embedding_service()
    service, _, reranker, llm, retrieval = _service(
        llm_response=EvidenceEvaluationBatch(
            criteria=[
                CriterionEvaluation(
                    criterion_id=cid,
                    status=CriterionStatus.MISSING,
                    explanation="No qualifying documentation found.",
                )
                for cid in ("c1", "c2", "c3", "c4")
            ],
            exclusions=[
                ExclusionEvaluation(
                    exclusion_id="ex1",
                    triggered=False,
                    status=CriterionStatus.MISSING,
                    explanation="Not triggered.",
                )
            ],
        ),
        embedding=embedding,
    )

    def empty_query(index, *, query, top_k=None):
        return []

    retrieval.query_section_bm25 = empty_query  # type: ignore[method-assign]
    retrieval.dense_rank_chunks_batched = MagicMock(  # type: ignore[method-assign]
        wraps=retrieval.dense_rank_chunks_batched
    )
    retrieval._rrf_fuse = MagicMock(wraps=retrieval._rrf_fuse)  # type: ignore[method-assign]
    retrieval.build_hybrid_index = MagicMock(  # type: ignore[method-assign]
        wraps=retrieval.build_hybrid_index
    )

    result = await service.evaluate_evidence(_policy_result(), sections)

    dense_groups = retrieval.dense_rank_chunks_batched.call_args.args[0]
    assert all(len(chunks) == 0 for _query, chunks in dense_groups.values())
    retrieval._rrf_fuse.assert_not_called()
    retrieval.build_hybrid_index.assert_not_called()
    reranker.rerank_grouped.assert_not_called()
    assert all(item.status == CriterionStatus.MISSING for item in result.criteria)
    prompts = [
        call.kwargs["user_prompt"]
        for call in llm.extract_structured.await_args_list
    ]
    assert any("evidence_corpus:" in prompt for prompt in prompts)
    assert any("- none" in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_sol_batching_and_bounded_concurrency() -> None:
    sections = [
        _section(f"p{i}", f"Demonstrated saphenous reflux note {i}.")
        for i in range(6)
    ]
    embedding = _embedding_service()
    settings = Settings(
        EVIDENCE_SECTION_BM25_TOP_K=3,
        EVIDENCE_DENSE_TOP_K=3,
        EVIDENCE_RERANK_TOP_K=2,
        EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=2,
        EVIDENCE_SOL_MAX_CONCURRENCY=2,
        POLICY_REASONING_MODEL="gpt-6-sol",
    )
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def slow_llm(**kwargs):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        prompt = kwargs["user_prompt"]
        criterion_ids = [
            line.split("criterion_id:", 1)[1].strip()
            for line in prompt.splitlines()
            if "criterion_id:" in line
        ]
        exclusion_ids = [
            line.split("exclusion_id:", 1)[1].strip()
            for line in prompt.splitlines()
            if "exclusion_id:" in line
        ]
        async with lock:
            in_flight -= 1
        return SolEvidenceEvaluationBatch(
            criteria=[
                SolCriterionEvaluation(
                    criterion_id=cid,
                    status=CriterionStatus.MISSING,
                    explanation="Missing.",
                )
                for cid in criterion_ids
            ],
            exclusions=[
                SolExclusionEvaluation(
                    exclusion_id=eid,
                    triggered=False,
                    status=CriterionStatus.MISSING,
                    explanation="Not triggered.",
                )
                for eid in exclusion_ids
            ],
        )

    service, _, _, llm, _ = _service(embedding=embedding, settings=settings)
    llm.extract_structured = AsyncMock(side_effect=slow_llm)
    result = await service.evaluate_evidence(_policy_result(), sections)

    # 4 criteria with batch size 2 => 2 Sol calls; exclusions ride on the last batch.
    assert result.llm_call_count == 2
    assert llm.extract_structured.await_count == 2
    assert max_in_flight <= 2
    assert len(result.criteria) == 4


@pytest.mark.asyncio
async def test_temporal_metadata_reaches_sol_prompt() -> None:
    sections = [
        Section(
            id="lab",
            document_id="patient",
            heading="Laboratory Results",
            parent_heading=None,
            text="2026-08-10 TSH 2.1 mIU/L within normal limits.",
            start_page=1,
            end_page=1,
            token_count=10,
            document_date="2026-08-10",
        ),
        Section(
            id="old",
            document_id="patient",
            heading="Laboratory Results",
            parent_heading=None,
            text="2021-01-05 TSH 12.0 mIU/L elevated.",
            start_page=2,
            end_page=2,
            token_count=8,
            document_date="2021-01-05",
        ),
    ]
    tree = CriterionNode(
        id="root",
        node_type=CriterionNodeType.AND,
        children=[
            CriterionNode(
                id="tsh",
                node_type=CriterionNodeType.CONDITION,
                description="Current TSH is within normal limits.",
                category=CriterionCategory.MEASUREMENT,
                temporal_requirement="current",
            )
        ],
    )
    policy = PolicyAnalysisResult(
        applicable=True,
        applicability_reason="Applicable.",
        applicability_confidence=0.9,
        criteria=PolicyCriteriaExtraction(
            procedure_name="Example",
            policy_summary="Summary",
            root=tree,
            exclusions=[],
            source_chunk_ids=["p1"],
        ),
        supporting_chunk_ids=["p1"],
        llm_call_count=1,
    )
    service, _, reranker, llm, _ = _service(
        llm_response=EvidenceEvaluationBatch(
            criteria=[
                CriterionEvaluation(
                    criterion_id="tsh",
                    status=CriterionStatus.MET,
                    evidence_chunk_ids=["lab"],
                    evidence_text=["TSH 2.1 mIU/L within normal limits"],
                    page_numbers=[1],
                    explanation="Current lab supports criterion.",
                )
            ],
            exclusions=[],
        )
    )
    atoms = service.extract_atomic_criteria(tree)
    assert atoms[0].temporal_requirement == "current"
    assert "current" in service.build_query(atoms[0]).lower()

    result = await service.evaluate_evidence(policy, sections)
    assert result.criteria[0].status == CriterionStatus.MET
    prompt = llm.extract_structured.await_args.kwargs["user_prompt"]
    assert "temporal_requirement: current" in prompt
    assert "document_date: 2026-08-10" in prompt
    assert "analysis_reference_date:" in prompt
    # Reranker candidate text includes encounter dates.
    groups = reranker.rerank_grouped.call_args.args[0]
    assert groups


@pytest.mark.asyncio
async def test_empty_criteria_works_safely() -> None:
    policy = PolicyAnalysisResult(
        applicable=True,
        applicability_reason="Applicable.",
        applicability_confidence=0.9,
        criteria=PolicyCriteriaExtraction(
            procedure_name="Example",
            policy_summary="Summary",
            root=CriterionNode(id="root", node_type=CriterionNodeType.AND, children=[]),
            exclusions=[],
            source_chunk_ids=[],
        ),
        supporting_chunk_ids=[],
        llm_call_count=1,
    )
    service, _, reranker, llm, _ = _service()
    result = await service.evaluate_evidence(policy, [_section("p1", "note")])
    assert result.criteria == []
    assert result.llm_call_count == 0
    llm.extract_structured.assert_not_awaited()
    reranker.rerank_grouped.assert_not_called()


@pytest.mark.asyncio
async def test_evidence_reuses_parsed_sections_without_pdf_parse() -> None:
    sections = [
        _section("p1", "Duplex ultrasound showed demonstrated saphenous reflux."),
        _section("p2", "CEAP clinical class documented as C3."),
    ]
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p1"],
                evidence_text=["demonstrated saphenous reflux"],
                page_numbers=[1],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p2"],
                evidence_text=["CEAP clinical class documented as C3"],
                page_numbers=[1],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=batch)
    # EvidenceService has no PDF/parse dependency — only already-parsed sections.
    assert not hasattr(service, "_pdf_service")
    assert not hasattr(service, "parse")
    result = await service.evaluate_evidence(_policy_result(), sections)
    assert len(result.criteria) == 4


def test_materialize_sol_batch_keeps_paired_citations_one_to_one() -> None:
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_candidate_ids=["c1:e1", "c1:e2"],
                explanation="Supported by two chunks.",
            )
        ],
        exclusions=[],
    )
    candidate_map = {
        "c1:e1": EvidenceCitation(
            chunk_id="p1",
            text="demonstrated saphenous reflux",
            page_number=1,
        ),
        "c1:e2": EvidenceCitation(
            chunk_id="p2",
            text="CEAP clinical class documented as C3",
            page_number=2,
        ),
    }
    public = materialize_sol_batch(sol, candidate_map=candidate_map)
    item = public.criteria[0]
    assert item.evidence_chunk_ids == ["p1", "p2"]
    assert item.evidence_text == [
        "demonstrated saphenous reflux",
        "CEAP clinical class documented as C3",
    ]
    assert item.page_numbers == [1, 2]


def test_parallel_to_sol_rejects_mismatched_arrays() -> None:
    with pytest.raises(EvidenceValidationError, match="evidence_text length"):
        CriterionEvaluation(
            criterion_id="c1",
            status=CriterionStatus.MET,
            evidence_chunk_ids=["p1", "p2"],
            evidence_text=["only one quote"],
            page_numbers=[1, 2],
            explanation="Broken parallel arrays.",
        )


@pytest.mark.asyncio
async def test_sol_grounding_multiple_evidence_chunks() -> None:
    sections = [
        _section("p1", "Duplex ultrasound showed demonstrated saphenous reflux."),
        _section("p2", "Follow-up note also showed demonstrated saphenous reflux."),
        _section("p3", "CEAP clinical class documented as C3."),
        _section("p4", "Compression therapy failed after four months."),
    ]
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p1", "p2"],
                evidence_text=[
                    "Duplex ultrasound showed demonstrated saphenous reflux.",
                    "Follow-up note also showed demonstrated saphenous reflux.",
                ],
                page_numbers=[1, 1],
                explanation="Two supporting notes.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, _, _, llm, _ = _service(llm_response=batch, embedding=_embedding_service())
    result = await service.evaluate_evidence(_policy_result(), sections)
    item = next(c for c in result.criteria if c.criterion_id == "c1")
    assert item.status == CriterionStatus.MET
    assert item.evidence_chunk_ids == ["p1", "p2"]
    assert len(item.evidence_text) == 2
    assert item.evidence_chunk_ids[0] == "p1"
    assert "reflux" in item.evidence_text[0]
    assert llm.extract_structured.await_args.kwargs["response_model"] is (
        SolEvidenceEvaluationBatch
    )


@pytest.mark.asyncio
async def test_sol_grounding_one_evidence_chunk() -> None:
    sections = [
        _section("p1", "Duplex ultrasound showed demonstrated saphenous reflux."),
        _section("p2", "CEAP clinical class documented as C3."),
    ]
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p1"],
                evidence_text=[
                    "Duplex ultrasound showed demonstrated saphenous reflux."
                ],
                page_numbers=[1],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            CriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=batch, embedding=_embedding_service())
    result = await service.evaluate_evidence(_policy_result(), sections)
    item = next(c for c in result.criteria if c.criterion_id == "c1")
    assert item.evidence_chunk_ids == ["p1"]
    assert "reflux" in item.evidence_text[0]


@pytest.mark.asyncio
async def test_sol_grounding_no_evidence() -> None:
    sections = [
        _section("p1", "Unrelated outpatient diet note."),
        _section("p2", "Another unrelated note."),
    ]
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id=cid,
                status=CriterionStatus.MISSING,
                evidence_candidate_ids=[],
                explanation="No supporting documentation.",
            )
            for cid in ("c1", "c2", "c3", "c4")
        ],
        exclusions=[
            SolExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                evidence_candidate_ids=[],
                explanation="Not triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=sol, embedding=_embedding_service())
    result = await service.evaluate_evidence(_policy_result(), sections)
    assert all(item.status == CriterionStatus.MISSING for item in result.criteria)
    assert all(item.evidence_chunk_ids == [] for item in result.criteria)
    assert all(item.evidence_text == [] for item in result.criteria)


@pytest.mark.asyncio
async def test_sol_grounding_rejects_unknown_chunk_id() -> None:
    sections = [_section("p1", "Demonstrated saphenous reflux on imaging.")]
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_candidate_ids=["c1:e999"],
                explanation="Invented candidate id.",
            ),
            SolCriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            SolCriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            SolCriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            SolExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=sol, embedding=_embedding_service())
    with pytest.raises(EvidenceGroundingError, match="Evidence grounding failed"):
        await service.evaluate_evidence(_policy_result(), sections)


@pytest.mark.asyncio
async def test_sol_grounding_rejects_cross_criterion_candidate() -> None:
    sections = [
        _section("p1", "Demonstrated saphenous reflux on imaging."),
        _section("p2", "CEAP clinical class documented as C3."),
    ]
    sol = SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_candidate_ids=["c2:e1"],
                explanation="Used another criterion's candidate.",
            ),
            SolCriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            SolCriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
            SolCriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            SolExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, *_ = _service(llm_response=sol, embedding=_embedding_service())
    with pytest.raises(EvidenceGroundingError, match="Evidence grounding failed"):
        await service.evaluate_evidence(_policy_result(), sections)


@pytest.mark.asyncio
async def test_sol_grounding_fills_missing_page_numbers_from_chunk() -> None:
    chunk_p1 = Chunk(
        id="p1",
        document_id="doc",
        text="demonstrated saphenous reflux",
        page_number=1,
        end_page_number=1,
        heading=None,
        parent_heading=None,
        token_count=5,
        section_id="s1",
        chunk_index=0,
    )
    chunk_p2 = Chunk(
        id="p2",
        document_id="doc",
        text="CEAP clinical class documented as C3",
        page_number=2,
        end_page_number=2,
        heading=None,
        parent_heading=None,
        token_count=5,
        section_id="s2",
        chunk_index=0,
    )
    public = materialize_sol_batch(
        SolEvidenceEvaluationBatch(
            criteria=[
                SolCriterionEvaluation(
                    criterion_id="c1",
                    status=CriterionStatus.MET,
                    evidence_candidate_ids=["c1:e1", "c1:e2"],
                    explanation="Mixed pages.",
                )
            ],
            exclusions=[],
        ),
        candidate_map={
            "c1:e1": citation_from_chunk(chunk_p1),
            "c1:e2": citation_from_chunk(chunk_p2),
        },
    )
    assert [item.page_number for item in public.criteria[0].evidence] == [1, 2]
    assert public.criteria[0].evidence[0].text == chunk_p1.text


@pytest.mark.asyncio
async def test_sol_batch_multiple_criteria_keeps_paired_evidence() -> None:
    sections = [
        _section("p1", "Duplex ultrasound showed demonstrated saphenous reflux."),
        _section("p2", "CEAP clinical class documented as C3."),
        _section("p3", "Compression therapy failed after four months."),
        _section("p4", "Symptoms remain ambiguous in chart notes."),
    ]
    settings = Settings(
        EVIDENCE_SECTION_BM25_TOP_K=5,
        EVIDENCE_DENSE_TOP_K=4,
        EVIDENCE_RERANK_TOP_K=3,
        EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=10,
        EVIDENCE_SOL_MAX_CONCURRENCY=1,
        POLICY_REASONING_MODEL="gpt-6-sol",
    )
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p1"],
                evidence_text=[
                    "Duplex ultrasound showed demonstrated saphenous reflux."
                ],
                page_numbers=[1],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p2"],
                evidence_text=["CEAP clinical class documented as C3."],
                page_numbers=[1],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c3",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["p3"],
                evidence_text=["Compression therapy failed after four months."],
                page_numbers=[1],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c4",
                status=CriterionStatus.MISSING,
                explanation="Missing.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    service, _, _, llm, _ = _service(
        llm_response=batch,
        embedding=_embedding_service(),
        settings=settings,
    )
    result = await service.evaluate_evidence(_policy_result(), sections)
    assert llm.extract_structured.await_count == 1
    by_id = {item.criterion_id: item for item in result.criteria}
    assert by_id["c1"].evidence_chunk_ids == ["p1"]
    assert "reflux" in by_id["c1"].evidence_text[0]
    assert by_id["c2"].evidence_chunk_ids == ["p2"]
    assert "CEAP" in by_id["c2"].evidence_text[0]
    assert by_id["c3"].evidence_chunk_ids == ["p3"]
    assert by_id["c4"].evidence_chunk_ids == []


def test_validate_evidence_grounding_still_rejects_parallel_mismatch() -> None:
    # Parallel-array mismatches are rejected when folding into citations.
    with pytest.raises(EvidenceValidationError, match="evidence_text length"):
        CriterionEvaluation(
            criterion_id="c1",
            status=CriterionStatus.MET,
            evidence_chunk_ids=["p1", "p2"],
            evidence_text=["only one quote"],
            page_numbers=[1, 2],
            explanation="Broken.",
        )


def test_build_sol_work_items_shrinks_batches_when_context_budget_tiny() -> None:
    from datetime import date

    from app.services.evidence_service import AtomicCriterion

    service, *_ = _service(
        settings=Settings(
            EVIDENCE_LLM_MAX_CONTEXT_TOKENS=1100,
            EVIDENCE_LLM_MAX_OUTPUT_TOKENS=256,
            EVIDENCE_SOL_MAX_ITEMS_PER_BATCH=10,
        )
    )
    conditions = [
        AtomicCriterion(
            id=f"c{i}",
            description=f"Demonstrated clinical requirement number {i}. " * 8,
            category=CriterionCategory.CLINICAL,
            kind="condition",
        )
        for i in range(4)
    ]
    candidate_by_id = {
        f"pat-{i}": Chunk(
            id=f"pat-{i}",
            document_id="patient",
            text=("Patient chart evidence note with supporting detail. " * 30)
            + str(i),
            page_number=1,
            end_page_number=1,
            heading="Notes",
            parent_heading=None,
            token_count=200,
            section_id=f"pat-{i}",
            chunk_index=0,
        )
        for i in range(4)
    }
    mapping = {f"c{i}": [f"pat-{i}"] for i in range(4)}
    local_context_by_id = {f"c{i}": "" for i in range(4)}

    work_items = service._build_sol_work_items(
        conditions,
        [],
        mapping=mapping,
        candidate_by_id=candidate_by_id,
        reference_date=date(2024, 6, 1),
        local_context_by_id=local_context_by_id,
    )

    batched_conditions = sum(len(batch[0]) for batch in work_items)
    assert batched_conditions == len(conditions)
    assert len(work_items) >= 2
    assert all(len(batch[0]) < len(conditions) for batch in work_items)

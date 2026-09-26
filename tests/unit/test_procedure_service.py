"""Unit tests for ProcedureService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import ProcedureExtractionError
from app.models.chunk import Chunk
from app.models.section import Section
from app.schemas.procedure import ProcedureExtraction
from app.services.procedure_service import ProcedureService


def _chunk(
    *,
    chunk_id: str,
    text: str,
    heading: str | None = None,
    parent_heading: str | None = None,
    page_number: int = 1,
    end_page_number: int | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc",
        text=text,
        page_number=page_number,
        end_page_number=end_page_number or page_number,
        heading=heading,
        parent_heading=parent_heading,
        token_count=10,
        section_id="sec",
        chunk_index=0,
    )


def _section(
    *,
    section_id: str,
    text: str,
    heading: str | None = None,
    parent_heading: str | None = None,
    page: int = 1,
) -> Section:
    return Section(
        id=section_id,
        document_id="doc",
        heading=heading,
        parent_heading=parent_heading,
        text=text,
        start_page=page,
        end_page=page,
        token_count=max(1, len(text.split())),
    )


def _service_with_mocks(
    *,
    candidates: list[Chunk],
    extraction: ProcedureExtraction,
    top_k: int = 5,
) -> tuple[ProcedureService, AsyncMock]:
    retrieval = MagicMock()
    chunking = MagicMock()
    reranking = MagicMock()
    llm = MagicMock()
    llm.extract_structured = AsyncMock(return_value=extraction)

    service = ProcedureService(
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=reranking,
        llm_service=llm,
        settings=Settings(PROCEDURE_RERANK_TOP_K=top_k),
    )
    service._retrieve_candidates = MagicMock(return_value=candidates)  # type: ignore[method-assign]
    return service, llm.extract_structured


@pytest.mark.asyncio
async def test_llm_prompt_includes_only_supplied_candidates() -> None:
    sections = [
        _section(section_id=f"s{i}", text=f"text {i}", heading=f"H{i}")
        for i in range(6)
    ]
    candidates = [
        _chunk(chunk_id="c1", text="text 1", heading="H1"),
        _chunk(chunk_id="c3", text="text 3", heading="H3"),
        _chunk(chunk_id="c5", text="text 5", heading="H5"),
    ]
    extraction = ProcedureExtraction(
        procedure_name=None,
        is_future_procedure=False,
        confidence=0.0,
    )
    service, llm_mock = _service_with_mocks(
        candidates=candidates,
        extraction=extraction,
        top_k=3,
    )

    await service.detect_planned_procedure(sections)

    prompt = llm_mock.await_args.kwargs["user_prompt"]
    assert prompt.count("CANDIDATE") == 3
    assert "chunk_id: c1" in prompt
    assert "chunk_id: c3" in prompt
    assert "chunk_id: c5" in prompt
    assert "chunk_id: c0" not in prompt
    service._retrieve_candidates.assert_called_once_with(sections)


@pytest.mark.asyncio
async def test_llm_prompt_candidate_order_is_deterministic() -> None:
    sections = [
        _section(section_id="a", text="alpha", heading="A"),
        _section(section_id="b", text="beta", heading="B"),
        _section(section_id="c", text="gamma", heading="C"),
    ]
    candidates = [
        _chunk(chunk_id="b", text="beta", heading="B"),
        _chunk(chunk_id="c", text="gamma", heading="C"),
        _chunk(chunk_id="a", text="alpha", heading="A"),
    ]
    extraction = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    service, llm_mock = _service_with_mocks(
        candidates=candidates,
        extraction=extraction,
    )

    await service.detect_planned_procedure(sections)
    first_prompt = llm_mock.await_args.kwargs["user_prompt"]

    await service.detect_planned_procedure(sections)
    second_prompt = llm_mock.await_args.kwargs["user_prompt"]

    assert first_prompt == second_prompt
    assert first_prompt.index("chunk_id: b") < first_prompt.index("chunk_id: c")


@pytest.mark.asyncio
async def test_valid_extraction_passes() -> None:
    text = "Next step is Example Procedure. CPT 12345. Status: pending auth."
    section = _section(section_id="sec-1", text=text, heading="Plan", page=3)
    chunk = _chunk(
        chunk_id="chunk-1",
        text=text,
        heading="Plan",
        page_number=3,
        end_page_number=3,
    )
    extraction = ProcedureExtraction(
        procedure_name="Example Procedure",
        cpt_code="12345",
        status="pending auth",
        source_chunk_id="chunk-1",
        source_text="Example Procedure. CPT 12345",
        page_number=3,
        is_future_procedure=True,
        confidence=0.91,
    )
    service, _ = _service_with_mocks(candidates=[chunk], extraction=extraction)

    result = await service.detect_planned_procedure([section])

    assert result.procedure_name == "Example Procedure"
    assert result.cpt_code == "12345"
    assert result.is_future_procedure is True


@pytest.mark.asyncio
async def test_unsupported_source_text_is_rejected() -> None:
    section = _section(section_id="sec-1", text="Actual chart text about monitoring.")
    chunk = _chunk(chunk_id="chunk-1", text="Actual chart text about monitoring.")
    extraction = ProcedureExtraction(
        procedure_name="Invented Surgery",
        source_chunk_id="chunk-1",
        source_text="This source text does not exist in the chunk",
        page_number=1,
        is_future_procedure=True,
        confidence=0.5,
    )
    service, _ = _service_with_mocks(candidates=[chunk], extraction=extraction)

    with pytest.raises(ProcedureExtractionError, match="source_text"):
        await service.detect_planned_procedure([section])


@pytest.mark.asyncio
async def test_invalid_cpt_is_rejected() -> None:
    text = "Planned Example Procedure with code ABCD1 pending."
    section = _section(section_id="sec-1", text=text)
    chunk = _chunk(chunk_id="chunk-1", text=text)
    extraction = ProcedureExtraction(
        procedure_name="Example Procedure",
        cpt_code="ABCD1",
        source_chunk_id="chunk-1",
        source_text="Example Procedure with code ABCD1",
        page_number=1,
        is_future_procedure=True,
        confidence=0.7,
    )
    service, _ = _service_with_mocks(candidates=[chunk], extraction=extraction)

    with pytest.raises(ProcedureExtractionError, match="five digits"):
        await service.detect_planned_procedure([section])


@pytest.mark.asyncio
async def test_missing_cpt_is_allowed() -> None:
    text = "Scheduled Example Procedure is the next step."
    section = _section(section_id="sec-1", text=text)
    chunk = _chunk(chunk_id="chunk-1", text=text)
    extraction = ProcedureExtraction(
        procedure_name="Example Procedure",
        cpt_code=None,
        source_chunk_id="chunk-1",
        source_text="Scheduled Example Procedure is the next step.",
        page_number=1,
        is_future_procedure=True,
        confidence=0.8,
    )
    service, _ = _service_with_mocks(candidates=[chunk], extraction=extraction)

    result = await service.detect_planned_procedure([section])

    assert result.cpt_code is None
    assert result.procedure_name == "Example Procedure"


@pytest.mark.asyncio
async def test_no_planned_procedure_is_handled() -> None:
    section = _section(
        section_id="sec-1",
        text="No upcoming interventions documented.",
    )
    chunk = _chunk(chunk_id="chunk-1", text="No upcoming interventions documented.")
    extraction = ProcedureExtraction(
        procedure_name=None,
        is_future_procedure=False,
        confidence=0.2,
    )
    service, _ = _service_with_mocks(candidates=[chunk], extraction=extraction)

    result = await service.detect_planned_procedure([section])

    assert result.procedure_name is None
    assert result.is_future_procedure is False


@pytest.mark.asyncio
async def test_historical_procedure_not_forced() -> None:
    section = _section(
        section_id="sec-1",
        text="Patient previously underwent appendectomy in 2019.",
    )
    chunk = _chunk(
        chunk_id="chunk-1",
        text="Patient previously underwent appendectomy in 2019.",
    )
    extraction = ProcedureExtraction(
        procedure_name=None,
        is_future_procedure=False,
        confidence=0.1,
    )
    service, _ = _service_with_mocks(candidates=[chunk], extraction=extraction)

    result = await service.detect_planned_procedure([section])

    assert result.procedure_name is None
    assert result.is_future_procedure is False


@pytest.mark.asyncio
async def test_multiple_candidate_chunks_work() -> None:
    sections = [
        _section(section_id="s1", text="History only."),
        _section(
            section_id="s2",
            text="Planned Example Procedure CPT 22222 pending authorization.",
            heading="Plan",
            page=4,
        ),
        _section(section_id="s3", text="Labs reviewed."),
    ]
    candidates = [
        _chunk(
            chunk_id="c2",
            text="Planned Example Procedure CPT 22222 pending authorization.",
            heading="Plan",
            page_number=4,
        ),
        _chunk(chunk_id="c1", text="History only."),
    ]
    extraction = ProcedureExtraction(
        procedure_name="Example Procedure",
        cpt_code="22222",
        status="pending authorization",
        source_chunk_id="c2",
        source_text="Planned Example Procedure CPT 22222 pending authorization.",
        page_number=4,
        is_future_procedure=True,
        confidence=0.95,
    )
    service, llm_mock = _service_with_mocks(
        candidates=candidates,
        extraction=extraction,
        top_k=2,
    )

    result = await service.detect_planned_procedure(sections)

    assert result.source_chunk_id == "c2"
    prompt = llm_mock.await_args.kwargs["user_prompt"]
    assert "chunk_id: c2" in prompt
    assert "chunk_id: c1" in prompt


@pytest.mark.asyncio
async def test_llm_response_conforms_to_schema() -> None:
    from pydantic import ValidationError

    section = _section(section_id="sec-1", text="Monitoring only.")
    chunk = _chunk(chunk_id="chunk-1", text="Monitoring only.")
    with pytest.raises(ValidationError):
        ProcedureExtraction(confidence=1.5)

    extraction = ProcedureExtraction(
        procedure_name=None,
        is_future_procedure=False,
        confidence=0.0,
    )
    service, _ = _service_with_mocks(candidates=[chunk], extraction=extraction)
    result = await service.detect_planned_procedure([section])
    assert isinstance(result, ProcedureExtraction)


@pytest.mark.asyncio
async def test_empty_sections_return_empty_result() -> None:
    service = ProcedureService(
        retrieval_service=MagicMock(),
        chunking_service=MagicMock(),
        reranking_service=MagicMock(),
        llm_service=MagicMock(),
        settings=Settings(),
    )
    result = await service.detect_planned_procedure([])
    assert result.procedure_name is None
    assert result.is_future_procedure is False


@pytest.mark.asyncio
async def test_retrieve_candidates_heading_then_hybrid_rerank() -> None:
    """Heading select → chunk selected → BM25+dense+RRF → rerank."""
    sections = [
        _section(
            section_id=f"s{i}",
            text=f"note {i} planned procedure",
            heading=f"H{i}",
        )
        for i in range(4)
    ]
    chunks = [
        _chunk(chunk_id=f"c{i}", text=f"note {i} planned procedure", heading=f"H{i}")
        for i in range(4)
    ]
    retrieval = MagicMock()
    retrieval.procedure_heading_query.return_value = "planned procedure"
    retrieval.procedure_chunk_query.return_value = "planned future procedure"
    retrieval.procedure_rerank_query.return_value = (
        "What procedure is planned, recommended, or scheduled for this patient?"
    )
    retrieval.select_sections_by_heading_similarity.return_value = sections[:3]
    retrieval.build_hybrid_index.return_value = MagicMock()
    fused = [
        MagicMock(chunk=chunks[i], rank=i + 1, score=1.0 - (i * 0.1))
        for i in range(3)
    ]
    retrieval.retrieve_for_query.return_value = fused

    chunking = MagicMock()
    chunking.chunk_sections.return_value = chunks[:3]

    reranking = MagicMock()
    reranking.rerank.return_value = fused[:2]

    service = ProcedureService(
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=reranking,
        llm_service=MagicMock(),
        settings=Settings(
            PROCEDURE_RERANK_TOP_K=2,
            PATIENT_HEADING_TOP_K=3,
        ),
    )

    selected = service._retrieve_candidates(sections)

    assert [chunk.id for chunk in selected] == ["c0", "c1"]
    retrieval.select_sections_by_heading_similarity.assert_called_once()
    call_kwargs = retrieval.select_sections_by_heading_similarity.call_args.kwargs
    assert call_kwargs["top_k"] == 3
    chunking.chunk_sections.assert_called_once_with(sections[:3])
    retrieval.build_hybrid_index.assert_called_once()
    assert retrieval.build_hybrid_index.call_args.kwargs["index_kind"] == "procedure"
    retrieval.retrieve_for_query.assert_called_once()
    assert (
        retrieval.retrieve_for_query.call_args.kwargs["log_prefix"]
        == "procedure_chunk"
    )
    reranking.rerank.assert_called_once()
    assert reranking.rerank.call_args.kwargs["top_k"] == 2
    assert reranking.rerank.call_args.kwargs["candidates"] == fused
    assert (
        reranking.rerank.call_args.kwargs["query"]
        == retrieval.procedure_rerank_query.return_value
    )
    assert len(selected) == 2


@pytest.mark.asyncio
async def test_no_heading_candidates_returns_empty() -> None:
    sections = [
        _section(section_id="s1", text="planned surgery", heading=None),
        _section(section_id="s2", text="vitals", heading="   "),
    ]
    retrieval = MagicMock()
    retrieval.procedure_heading_query.return_value = "planned procedure"
    retrieval.select_sections_by_heading_similarity.return_value = []
    chunking = MagicMock()
    reranking = MagicMock()
    llm = MagicMock()
    llm.extract_structured = AsyncMock(
        return_value=ProcedureExtraction(
            procedure_name=None,
            is_future_procedure=False,
            confidence=0.0,
        )
    )
    service = ProcedureService(
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=reranking,
        llm_service=llm,
        settings=Settings(PATIENT_HEADING_TOP_K=5),
    )

    result = await service.detect_planned_procedure(sections)

    assert result.procedure_name is None
    assert result.is_future_procedure is False
    chunking.chunk_sections.assert_not_called()
    retrieval.build_hybrid_index.assert_not_called()
    retrieval.retrieve_for_query.assert_not_called()
    reranking.rerank.assert_not_called()
    llm.extract_structured.assert_not_awaited()

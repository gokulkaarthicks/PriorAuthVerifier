"""Optional live integration test for planned-procedure detection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.models.document import DocumentType
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.heading_service import HeadingService
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService
from app.services.procedure_service import ProcedureService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
PATIENT_PDF = SAMPLE_DIR / "patient.pdf"

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_INTEGRATION"),
    reason="Set RUN_LIVE_INTEGRATION=1 to run live OpenAI integration",
)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for live procedure integration test",
)
@pytest.mark.skipif(not PATIENT_PDF.exists(), reason="sample patient.pdf missing")
async def test_patient_chart_planned_procedure_live() -> None:
    settings = get_settings()
    document = PDFService(settings=settings).parse(
        PATIENT_PDF,
        DocumentType.PATIENT_CHART,
    )
    sections = HeadingService().build_sections(document)
    embedding = EmbeddingService(settings=settings)
    retrieval = RetrievalService(embedding_service=embedding, settings=settings)
    chunking = ChunkingService(settings=settings)
    reranking = RerankingService(settings=settings)

    service = ProcedureService(
        retrieval_service=retrieval,
        chunking_service=chunking,
        reranking_service=reranking,
        llm_service=LLMService(settings=settings),
        settings=settings,
    )
    result = await service.detect_planned_procedure(sections)

    assert result.procedure_name is not None
    assert "permanent cardiac pacemaker implantation" in (
        result.procedure_name.lower()
    )
    assert result.cpt_code == "33207"
    assert result.is_future_procedure is True
    assert result.source_chunk_id is not None
    assert result.page_number is not None
    assert result.source_text is not None

    # Source text must appear in some patient section body.
    assert any(
        result.source_text.lower() in section.text.lower()
        or result.procedure_name.lower() in section.text.lower()
        for section in sections
        if section.text
    )
    assert any(
        section.start_page <= result.page_number <= section.end_page
        for section in sections
    )

"""Integration tests for PDF → sections → chunks (parser/heading/chunking only).

These tests intentionally chunk entire sample PDFs to validate section-aware
chunking. They do not build retrieval indexes or dense-embed documents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.document import DocumentType
from app.services.chunking_service import ChunkingService
from app.services.heading_service import HeadingService
from app.services.pdf_service import PDFService

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
PATIENT_PDF = SAMPLE_DIR / "patient.pdf"
POLICY_PDF = SAMPLE_DIR / "bcbs_policy.pdf"


@pytest.fixture
def pdf_service() -> PDFService:
    return PDFService()


@pytest.fixture
def heading_service() -> HeadingService:
    return HeadingService()


@pytest.fixture
def chunking_service() -> ChunkingService:
    return ChunkingService()


@pytest.mark.skipif(not PATIENT_PDF.exists(), reason="sample patient.pdf missing")
def test_patient_chart_sections_and_chunks(
    pdf_service: PDFService,
    heading_service: HeadingService,
    chunking_service: ChunkingService,
) -> None:
    document = pdf_service.parse(PATIENT_PDF, DocumentType.PATIENT_CHART)
    sections = heading_service.build_sections(document)
    chunks = chunking_service.chunk_sections(sections)

    assert document.page_count == 7
    assert len(sections) >= 5
    assert len(chunks) >= 5

    headed = [section for section in sections if section.heading]
    assert len(headed) >= 4

    page_3_headings = [
        section.heading
        for section in sections
        if section.heading and section.start_page <= 3 <= section.end_page
    ]
    assert len(page_3_headings) >= 2

    combined = "\n".join(chunk.text for chunk in chunks).lower()
    assert "pacemaker" in combined
    assert "33207" in combined or "planned procedure" in combined

    assert all(chunk.page_number >= 1 for chunk in chunks)
    assert all(chunk.end_page_number >= chunk.page_number for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)
    assert all(chunk.section_id for chunk in chunks)


@pytest.mark.skipif(not POLICY_PDF.exists(), reason="sample bcbs_policy.pdf missing")
def test_policy_sections_and_chunks(
    pdf_service: PDFService,
    heading_service: HeadingService,
    chunking_service: ChunkingService,
) -> None:
    document = pdf_service.parse(POLICY_PDF, DocumentType.POLICY)
    sections = heading_service.build_sections(document)
    chunks = chunking_service.chunk_sections(sections)

    assert document.page_count >= 2
    assert len(sections) >= 3
    assert len(chunks) >= 3

    headings = " ".join(section.heading or "" for section in sections).upper()
    assert "POSITION STATEMENT" in headings or "BILLING" in headings

    combined = "\n".join(chunk.text for chunk in chunks).lower()
    assert "billing" in combined or "reimbursement" in combined
    # Sample policy uses sectioned coverage language rather than a titled
    # "Position Statement"; preserve substantive policy content instead.
    assert (
        "coverage" in combined
        or "authorization" in combined
        or "medical necessity" in combined
    )
    assert (
        "definitions" in combined
        or "exclusions" in combined
        or "coding" in combined
    )

    for chunk in chunks:
        assert 1 <= chunk.page_number <= document.page_count
        assert chunk.end_page_number <= document.page_count
        assert chunk.end_page_number >= chunk.page_number

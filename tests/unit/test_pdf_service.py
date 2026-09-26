"""Unit tests for PDFService."""

from pathlib import Path

import pymupdf
import pytest

from app.core.config import Settings
from app.core.exceptions import InvalidPDFError, PDFNotFoundError
from app.models.document import DocumentType
from app.services.pdf_service import PDFService


@pytest.fixture
def pdf_service() -> PDFService:
    return PDFService(settings=Settings(MIN_EXTRACTED_PAGE_CHARS=30))


def _write_text_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()


def _write_empty_page_pdf(path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    doc.close()


def test_pdf_parses_successfully(tmp_path: Path, pdf_service: PDFService) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_text_pdf(pdf_path, ["Hello PriorAuthVerifier"])

    parsed = pdf_service.parse(pdf_path, DocumentType.PATIENT_CHART)

    assert parsed.filename == "sample.pdf"
    assert parsed.document_type == DocumentType.PATIENT_CHART
    assert parsed.id
    assert parsed.page_count == 1


def test_correct_page_count(tmp_path: Path, pdf_service: PDFService) -> None:
    pdf_path = tmp_path / "multi.pdf"
    _write_text_pdf(
        pdf_path,
        [
            "Page one content for testing.",
            "Page two content for testing.",
            "Page three content for testing.",
        ],
    )

    parsed = pdf_service.parse(pdf_path, DocumentType.POLICY)

    assert parsed.page_count == 3
    assert len(parsed.pages) == 3


def test_page_numbers_start_at_one(tmp_path: Path, pdf_service: PDFService) -> None:
    pdf_path = tmp_path / "pages.pdf"
    _write_text_pdf(pdf_path, ["First page text here.", "Second page text here."])

    parsed = pdf_service.parse(pdf_path, DocumentType.PATIENT_CHART)

    assert [page.page_number for page in parsed.pages] == [1, 2]


def test_text_extraction_works(tmp_path: Path, pdf_service: PDFService) -> None:
    pdf_path = tmp_path / "text.pdf"
    _write_text_pdf(pdf_path, ["Planned procedure: lumbar fusion"])

    parsed = pdf_service.parse(pdf_path, DocumentType.PATIENT_CHART)

    assert "lumbar fusion" in parsed.pages[0].text


def test_span_metadata_exists(tmp_path: Path, pdf_service: PDFService) -> None:
    pdf_path = tmp_path / "spans.pdf"
    _write_text_pdf(pdf_path, ["Span metadata should be present"])

    parsed = pdf_service.parse(pdf_path, DocumentType.POLICY)
    spans = parsed.pages[0].spans

    assert spans
    span = spans[0]
    assert span.text.strip()
    assert span.font_size > 0
    assert span.font_name
    assert isinstance(span.is_bold, bool)
    assert len(span.bbox) == 4


def test_missing_file_raises_pdf_not_found(pdf_service: PDFService) -> None:
    with pytest.raises(PDFNotFoundError):
        pdf_service.parse(
            Path("/tmp/does-not-exist-priorauth.pdf"),
            DocumentType.POLICY,
        )


def test_non_pdf_raises_invalid_pdf(tmp_path: Path, pdf_service: PDFService) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(InvalidPDFError):
        pdf_service.parse(text_path, DocumentType.PATIENT_CHART)


def test_fake_pdf_extension_raises_invalid_pdf(
    tmp_path: Path, pdf_service: PDFService
) -> None:
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("this is not a pdf file", encoding="utf-8")

    with pytest.raises(InvalidPDFError):
        pdf_service.parse(fake_pdf, DocumentType.POLICY)


def test_empty_page_requires_ocr(tmp_path: Path, pdf_service: PDFService) -> None:
    pdf_path = tmp_path / "empty.pdf"
    _write_empty_page_pdf(pdf_path)

    parsed = pdf_service.parse(pdf_path, DocumentType.PATIENT_CHART)

    assert parsed.pages[0].requires_ocr is True
    assert parsed.pages[0].text.strip() == ""


def test_sparse_text_requires_ocr(tmp_path: Path, pdf_service: PDFService) -> None:
    pdf_path = tmp_path / "sparse.pdf"
    _write_text_pdf(pdf_path, ["abc"])

    parsed = pdf_service.parse(pdf_path, DocumentType.POLICY)

    assert parsed.pages[0].requires_ocr is True

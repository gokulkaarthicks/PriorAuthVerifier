"""PDF parsing domain models."""

from dataclasses import dataclass

from app.models.document import DocumentType


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A layout-aware text span extracted from a PDF page."""

    text: str
    font_size: float
    font_name: str
    is_bold: bool
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PDFPage:
    """Page-level extracted content from a PDF."""

    page_number: int
    text: str
    spans: list[TextSpan]
    requires_ocr: bool


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Fully parsed PDF with page-level content."""

    id: str
    filename: str
    document_type: DocumentType
    page_count: int
    pages: list[PDFPage]

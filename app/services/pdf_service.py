"""PDF ingestion service using PyMuPDF."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import pymupdf

from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidPDFError, PDFNotFoundError, PDFParsingError
from app.core.logging import get_logger
from app.core.metrics import get_request_metrics
from app.models.document import DocumentType
from app.models.pdf import ParsedDocument, PDFPage, TextSpan

logger = get_logger(__name__)

_PDF_MAGIC = b"%PDF"
_BOLD_MARKERS = ("bold", "semibold", "demi")


class PDFService:
    """Parse PDF files into page-level domain models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def parse(self, file_path: Path, document_type: DocumentType) -> ParsedDocument:
        """Validate and parse a PDF into a ParsedDocument."""
        metrics = get_request_metrics()
        if metrics is not None:
            metrics.record_parser_invocation(document_type.value)

        path = self._resolve_path(file_path)
        self._validate_pdf_file(path)

        document_id = str(uuid.uuid4())
        prefix = (
            "patient"
            if document_type == DocumentType.PATIENT_CHART
            else "policy"
        )
        parse_started = time.perf_counter()
        try:
            # PyMuPDF stubs are incomplete for Document construction.
            with pymupdf.open(path) as pdf_doc:  # type: ignore[no-untyped-call]
                open_ms = int((time.perf_counter() - parse_started) * 1000)
                if metrics is not None:
                    metrics.add_substage_ms(f"{prefix}_pdf_parse_ms", open_ms)
                if pdf_doc.page_count < 1:
                    raise PDFParsingError(f"PDF has no pages: {path.name}")

                extract_started = time.perf_counter()
                pages = [
                    self._extract_page(pdf_doc, page_index)
                    for page_index in range(pdf_doc.page_count)
                ]
                extract_ms = int((time.perf_counter() - extract_started) * 1000)
                if metrics is not None:
                    metrics.add_substage_ms(
                        f"{prefix}_page_text_extract_ms", extract_ms
                    )
        except PDFParsingError:
            raise
        except Exception as exc:
            logger.error(
                "pdf_parse_failed",
                filename=path.name,
                error=str(exc),
            )
            raise PDFParsingError(f"Failed to parse PDF: {path.name}") from exc

        ocr_pages = [page.page_number for page in pages if page.requires_ocr]
        logger.info(
            "pdf_parsed",
            filename=path.name,
            document_type=document_type.value,
            page_count=len(pages),
            ocr_candidate_pages=ocr_pages,
        )

        return ParsedDocument(
            id=document_id,
            filename=path.name,
            document_type=document_type,
            page_count=len(pages),
            pages=pages,
        )

    def _resolve_path(self, file_path: Path) -> Path:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise PDFNotFoundError(f"PDF file not found: {path.name}")
        return path

    def _validate_pdf_file(self, path: Path) -> None:
        if path.suffix.lower() != ".pdf":
            raise InvalidPDFError(f"File is not a PDF: {path.name}")

        try:
            header = path.read_bytes()[:1024]
        except OSError as exc:
            raise InvalidPDFError(f"Unable to read file: {path.name}") from exc

        if _PDF_MAGIC not in header:
            raise InvalidPDFError(f"File is not a valid PDF: {path.name}")

    def _extract_page(self, pdf_doc: pymupdf.Document, page_index: int) -> PDFPage:
        page_number = page_index + 1
        try:
            page = pdf_doc.load_page(page_index)  # type: ignore[no-untyped-call]
            text = str(page.get_text("text"))
            page_dict = page.get_text("dict")
            if not isinstance(page_dict, dict):
                raise PDFParsingError(
                    f"Unexpected page dict type on page {page_number}"
                )
            spans = self._extract_spans(page_dict)
        except PDFParsingError:
            raise
        except Exception as exc:
            logger.error(
                "pdf_page_parse_failed",
                page_number=page_number,
                error=str(exc),
            )
            raise PDFParsingError(
                f"Failed to parse PDF page {page_number}"
            ) from exc

        return PDFPage(
            page_number=page_number,
            text=text,
            spans=spans,
            requires_ocr=self._requires_ocr(text),
        )

    def _extract_spans(self, page_dict: dict[str, Any]) -> list[TextSpan]:
        spans: list[TextSpan] = []
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text or not text.strip():
                        continue
                    font_name = str(span.get("font", ""))
                    bbox_raw = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    bbox = (
                        float(bbox_raw[0]),
                        float(bbox_raw[1]),
                        float(bbox_raw[2]),
                        float(bbox_raw[3]),
                    )
                    spans.append(
                        TextSpan(
                            text=text,
                            font_size=float(span.get("size", 0.0)),
                            font_name=font_name,
                            is_bold=self._is_bold(font_name),
                            bbox=bbox,
                        )
                    )
        return spans

    def _is_bold(self, font_name: str) -> bool:
        lowered = font_name.lower()
        return any(marker in lowered for marker in _BOLD_MARKERS)

    def _requires_ocr(self, text: str) -> bool:
        meaningful = "".join(ch for ch in text if not ch.isspace())
        return len(meaningful) < self._settings.MIN_EXTRACTED_PAGE_CHARS

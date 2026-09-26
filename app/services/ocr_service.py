"""Optional Mistral OCR enrichment for unreadable PDF pages."""

from __future__ import annotations

import base64
import time
from dataclasses import replace

import pymupdf

from app.core.config import Settings, get_settings
from app.core.exceptions import OCRProcessingError
from app.core.logging import get_logger
from app.core.metrics import get_request_metrics
from app.models.pdf import ParsedDocument, PDFPage

logger = get_logger(__name__)


class OCRService:
    """Fill OCR-required pages using the configured Mistral OCR API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: object | None = None

    async def enrich_document(
        self,
        document: ParsedDocument,
        source_path: str | None = None,
    ) -> ParsedDocument:
        ocr_pages = [page for page in document.pages if page.requires_ocr]
        if not ocr_pages:
            return document

        if not (self._settings.MISTRAL_API_KEY or "").strip():
            page_numbers = [page.page_number for page in ocr_pages]
            raise OCRProcessingError(
                "OCR is required for pages "
                f"{page_numbers} but MISTRAL_API_KEY is not configured"
            )

        if source_path is None:
            raise OCRProcessingError(
                "OCR enrichment requires the original PDF path"
            )

        started = time.perf_counter()
        updated_pages: list[PDFPage] = []
        for page in document.pages:
            if not page.requires_ocr:
                updated_pages.append(page)
                continue
            text = await self._ocr_page(source_path, page.page_number)
            if not text.strip():
                raise OCRProcessingError(
                    f"OCR returned empty text for page {page.page_number}"
                )
            updated_pages.append(
                replace(
                    page,
                    text=text,
                    requires_ocr=False,
                    spans=[],
                )
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        metrics = get_request_metrics()
        if metrics is not None:
            metrics.record_ocr_pages(document.document_type.value, len(ocr_pages))
        logger.info(
            "ocr_enrichment_completed",
            filename=document.filename,
            ocr_page_count=len(ocr_pages),
            duration_ms=elapsed_ms,
        )
        return replace(document, pages=updated_pages)

    async def _ocr_page(self, source_path: str, page_number: int) -> str:
        image_b64 = self._render_page_png_base64(source_path, page_number)
        client = self._get_client()
        try:
            response = await client.ocr.process(  # type: ignore[attr-defined]
                model=self._settings.OCR_MODEL,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_b64}",
                },
                include_image_base64=False,
            )
            return self._extract_text(response)
        except OCRProcessingError:
            raise
        except Exception as exc:
            logger.error(
                "ocr_page_failed",
                page_number=page_number,
                error=str(exc),
            )
            raise OCRProcessingError(
                f"OCR failed for page {page_number}"
            ) from exc

    def _render_page_png_base64(self, source_path: str, page_number: int) -> str:
        try:
            with pymupdf.open(source_path) as pdf_doc:  # type: ignore[no-untyped-call]
                page = pdf_doc.load_page(page_number - 1)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # type: ignore[no-untyped-call]
                return base64.b64encode(pix.tobytes("png")).decode("ascii")
        except Exception as exc:
            raise OCRProcessingError(
                f"Failed to render page {page_number} for OCR"
            ) from exc

    def _extract_text(self, response: object) -> str:
        pages = getattr(response, "pages", None)
        if not pages:
            markdown = getattr(response, "markdown", None)
            if isinstance(markdown, str):
                return markdown
            raise OCRProcessingError("OCR response contained no pages")
        parts: list[str] = []
        for page in pages:
            markdown = getattr(page, "markdown", None)
            if isinstance(markdown, str) and markdown.strip():
                parts.append(markdown)
        return "\n\n".join(parts).strip()

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        try:
            from mistralai.client import Mistral

            self._client = Mistral(
                api_key=self._settings.MISTRAL_API_KEY,
                timeout_ms=int(self._settings.OCR_REQUEST_TIMEOUT_SECONDS * 1000),
            )
            return self._client
        except Exception as exc:
            raise OCRProcessingError(
                "Failed to initialize Mistral OCR client"
            ) from exc

"""Unit tests for OCRService configuration and failure behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import Settings, get_settings
from app.core.exceptions import OCRProcessingError
from app.models.document import DocumentType
from app.models.pdf import ParsedDocument, PDFPage
from app.services.ocr_service import OCRService


def _document(*, requires_ocr: bool) -> ParsedDocument:
    return ParsedDocument(
        id="doc-1",
        filename="scan.pdf",
        document_type=DocumentType.PATIENT_CHART,
        page_count=1,
        pages=[
            PDFPage(
                page_number=1,
                text="",
                spans=[],
                requires_ocr=requires_ocr,
            )
        ],
    )


def test_settings_allow_startup_without_mistral_key() -> None:
    get_settings.cache_clear()
    settings = Settings(OPENAI_API_KEY="test-key", MISTRAL_API_KEY=None)
    assert settings.MISTRAL_API_KEY is None
    assert settings.OCR_MODEL == "mistral-ocr-latest"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_enrich_skips_when_no_ocr_pages() -> None:
    settings = Settings(MISTRAL_API_KEY=None, OCR_MODEL="custom-ocr")
    service = OCRService(settings=settings)
    document = _document(requires_ocr=False)

    with patch("mistralai.client.Mistral") as mistral_ctor:
        result = await service.enrich_document(document, source_path="unused.pdf")

    assert result is document
    mistral_ctor.assert_not_called()


@pytest.mark.asyncio
async def test_enrich_without_key_fails_clearly() -> None:
    settings = Settings(MISTRAL_API_KEY=None)
    service = OCRService(settings=settings)

    with pytest.raises(OCRProcessingError, match="MISTRAL_API_KEY"):
        await service.enrich_document(
            _document(requires_ocr=True),
            source_path="scan.pdf",
        )


@pytest.mark.asyncio
async def test_ocr_uses_settings_model_and_api_key() -> None:
    settings = Settings(
        MISTRAL_API_KEY="mistral-secret",
        OCR_MODEL="configured-ocr-model",
        OCR_REQUEST_TIMEOUT_SECONDS=12.0,
    )
    service = OCRService(settings=settings)
    fake_response = SimpleNamespace(
        pages=[SimpleNamespace(markdown="OCR page text")]
    )
    fake_client = SimpleNamespace(
        ocr=SimpleNamespace(process=AsyncMock(return_value=fake_response))
    )

    with (
        patch.object(
            service,
            "_render_page_png_base64",
            return_value="abc123",
        ),
            patch("mistralai.client.Mistral", return_value=fake_client) as mistral_ctor,
    ):
        result = await service.enrich_document(
            _document(requires_ocr=True),
            source_path="scan.pdf",
        )

    mistral_ctor.assert_called_once_with(
        api_key="mistral-secret",
        timeout_ms=12000,
    )
    fake_client.ocr.process.assert_awaited_once()
    call_kwargs = fake_client.ocr.process.await_args.kwargs
    assert call_kwargs["model"] == "configured-ocr-model"
    assert result.pages[0].text == "OCR page text"
    assert result.pages[0].requires_ocr is False


@pytest.mark.asyncio
async def test_ocr_client_reused_across_pages() -> None:
    settings = Settings(MISTRAL_API_KEY="mistral-secret", OCR_MODEL="ocr-model")
    service = OCRService(settings=settings)
    document = ParsedDocument(
        id="doc-2",
        filename="scan.pdf",
        document_type=DocumentType.PATIENT_CHART,
        page_count=2,
        pages=[
            PDFPage(page_number=1, text="", spans=[], requires_ocr=True),
            PDFPage(page_number=2, text="", spans=[], requires_ocr=True),
        ],
    )
    fake_response = SimpleNamespace(
        pages=[SimpleNamespace(markdown="page text")]
    )
    fake_client = SimpleNamespace(
        ocr=SimpleNamespace(process=AsyncMock(return_value=fake_response))
    )

    with (
        patch.object(
            service,
            "_render_page_png_base64",
            return_value="img",
        ),
            patch("mistralai.client.Mistral", return_value=fake_client) as mistral_ctor,
    ):
        await service.enrich_document(document, source_path="scan.pdf")

    assert mistral_ctor.call_count == 1
    assert fake_client.ocr.process.await_count == 2

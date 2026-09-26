"""Central FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    EmbeddingError,
    EvidenceAnalysisError,
    EvidenceGroundingError,
    EvidenceValidationError,
    InvalidPDFError,
    LLMServiceError,
    OCRProcessingError,
    PDFNotFoundError,
    PDFParsingError,
    PolicyAnalysisError,
    PolicyValidationError,
    PriorAuthVerifierError,
    ProcedureExtractionError,
    ReadinessEvaluationError,
    RerankingError,
    RetrievalError,
    UploadValidationError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    headers: dict[str, str] = {}
    request_id = _request_id(request)
    if request_id:
        headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach domain exception handlers to the FastAPI application."""

    @app.exception_handler(UploadValidationError)
    async def upload_validation_handler(
        request: Request,
        exc: UploadValidationError,
    ) -> JSONResponse:
        logger.warning(
            "upload_validation_failed",
            error=str(exc),
            status_code=exc.status_code,
        )
        return _error_response(
            request=request,
            status_code=exc.status_code,
            code="upload_validation_error",
            message=str(exc),
        )

    @app.exception_handler(InvalidPDFError)
    async def invalid_pdf_handler(
        request: Request,
        exc: InvalidPDFError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=400,
            code="invalid_pdf",
            message=str(exc),
        )

    @app.exception_handler(PDFNotFoundError)
    async def pdf_not_found_handler(
        request: Request,
        exc: PDFNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=400,
            code="pdf_not_found",
            message=str(exc),
        )

    @app.exception_handler(DocumentProcessingError)
    async def document_processing_handler(
        request: Request,
        exc: DocumentProcessingError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=422,
            code="document_processing_error",
            message=str(exc),
        )

    @app.exception_handler(ProcedureExtractionError)
    async def procedure_handler(
        request: Request,
        exc: ProcedureExtractionError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=422,
            code="procedure_extraction_error",
            message=str(exc),
        )

    @app.exception_handler(PolicyAnalysisError)
    async def policy_handler(
        request: Request,
        exc: PolicyAnalysisError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=422,
            code="policy_analysis_error",
            message=str(exc),
        )

    @app.exception_handler(PolicyValidationError)
    async def policy_validation_handler(
        request: Request,
        exc: PolicyValidationError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=422,
            code="policy_validation_error",
            message=str(exc),
        )

    @app.exception_handler(EvidenceAnalysisError)
    async def evidence_handler(
        request: Request,
        exc: EvidenceAnalysisError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=422,
            code="evidence_analysis_error",
            message=str(exc),
        )

    @app.exception_handler(EvidenceValidationError)
    async def evidence_validation_handler(
        request: Request,
        exc: EvidenceValidationError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=422,
            code="evidence_validation_error",
            message=str(exc),
        )

    @app.exception_handler(EvidenceGroundingError)
    async def evidence_grounding_handler(
        request: Request,
        exc: EvidenceGroundingError,
    ) -> JSONResponse:
        logger.error("evidence_grounding_failed", error=str(exc))
        return _error_response(
            request=request,
            status_code=500,
            code="evidence_grounding_error",
            message="Evidence grounding failed",
        )

    @app.exception_handler(ReadinessEvaluationError)
    async def readiness_handler(
        request: Request,
        exc: ReadinessEvaluationError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=422,
            code="readiness_evaluation_error",
            message=str(exc),
        )

    @app.exception_handler(ConfigurationError)
    async def configuration_handler(
        request: Request,
        exc: ConfigurationError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=503,
            code="configuration_error",
            message=str(exc),
        )

    @app.exception_handler(LLMServiceError)
    async def llm_handler(
        request: Request,
        exc: LLMServiceError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=502,
            code="provider_error",
            message="Upstream language-model provider request failed",
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_handler(
        request: Request,
        exc: EmbeddingError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=502,
            code="provider_error",
            message="Embedding provider request failed",
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_handler(
        request: Request,
        exc: RetrievalError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=502,
            code="provider_error",
            message="Retrieval pipeline failed",
        )

    @app.exception_handler(RerankingError)
    async def reranking_handler(
        request: Request,
        exc: RerankingError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=502,
            code="provider_error",
            message="Reranking provider request failed",
        )

    @app.exception_handler(OCRProcessingError)
    async def ocr_handler(
        request: Request,
        exc: OCRProcessingError,
    ) -> JSONResponse:
        # Map OCR failures to 422 (document issue), not 502.
        return _error_response(
            request=request,
            status_code=422,
            code="ocr_processing_error",
            message=str(exc),
        )

    @app.exception_handler(PDFParsingError)
    async def pdf_parsing_handler(
        request: Request,
        exc: PDFParsingError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=422,
            code="pdf_parsing_error",
            message=str(exc),
        )

    @app.exception_handler(PriorAuthVerifierError)
    async def app_error_handler(
        request: Request,
        exc: PriorAuthVerifierError,
    ) -> JSONResponse:
        logger.error("application_error", error=str(exc))
        return _error_response(
            request=request,
            status_code=500,
            code="application_error",
            message="Analysis failed",
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.error("unexpected_error", error=str(exc))
        return _error_response(
            request=request,
            status_code=500,
            code="internal_error",
            message="Unexpected server error",
        )

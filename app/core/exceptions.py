"""Application and domain exceptions."""


class PriorAuthVerifierError(Exception):
    """Base exception for PriorAuthVerifier application errors."""


class ConfigurationError(PriorAuthVerifierError):
    """Raised when required runtime configuration is missing or invalid."""


class DocumentProcessingError(PriorAuthVerifierError):
    """Raised when document ingestion or preprocessing fails."""


class PDFNotFoundError(DocumentProcessingError):
    """Raised when a PDF file path does not exist."""


class InvalidPDFError(DocumentProcessingError):
    """Raised when a file is not a valid PDF."""


class PDFParsingError(DocumentProcessingError):
    """Raised when PDF parsing fails for a document or page."""


class OCRProcessingError(DocumentProcessingError):
    """Raised when OCR is required but unavailable or fails."""


class EmbeddingError(PriorAuthVerifierError):
    """Raised when embedding generation or similarity fails."""


class LLMServiceError(PriorAuthVerifierError):
    """Raised when an LLM request fails or is misconfigured."""


class LLMCompletionLimitError(LLMServiceError):
    """Raised when the model exhausts max_completion_tokens without parseable output.

    Non-retryable: resending the same prompt would waste the same budget.
    """


class ProcedureExtractionError(PriorAuthVerifierError):
    """Raised when procedure extraction output fails validation."""


class RetrievalError(PriorAuthVerifierError):
    """Raised when hybrid retrieval or index construction fails."""


class RerankingError(PriorAuthVerifierError):
    """Raised when cross-encoder reranking fails."""


class PolicyAnalysisError(PriorAuthVerifierError):
    """Raised when policy applicability or criteria analysis fails."""


class PolicyValidationError(PriorAuthVerifierError):
    """Raised when structured policy output fails deterministic validation."""


class EvidenceAnalysisError(PriorAuthVerifierError):
    """Raised when evidence retrieval or criterion evaluation fails."""


class EvidenceValidationError(PriorAuthVerifierError):
    """Raised when structured evidence judgments fail deterministic validation."""


class EvidenceGroundingError(PriorAuthVerifierError):
    """Raised when Sol cites evidence outside a criterion's allowed candidates.

    This is an internal model/output contract failure, not a user upload error.
    """


class ReadinessEvaluationError(PriorAuthVerifierError):
    """Raised when readiness aggregation inputs are inconsistent."""


class UploadValidationError(PriorAuthVerifierError):
    """Raised when an uploaded file fails pre-analysis validation."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code

"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Prevent Transformers from background-converting PyTorch .bin weights to
# safetensors (and downloading conversion-PR artifacts) at runtime.
# Must run before SentenceTransformer / Transformers model load.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


class Settings(BaseSettings):
    """Runtime settings for PriorAuthVerifier."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "PriorAuthVerifier"
    LOG_LEVEL: str = "INFO"
    DEBUG_TRACE_ENABLED: bool = False
    PERFORMANCE_PROFILE_ENABLED: bool = False

    STRUCTURED_EXTRACTION_MODEL: str = "gpt-6-luna"
    POLICY_REASONING_MODEL: str = "gpt-6-sol"

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    OCR_MODEL: str = "mistral-ocr-latest"

    MAX_SECTION_TOKENS: int = 900
    SECTION_OVERLAP_TOKENS: int = 100

    # Procedure chunk hybrid (BM25 + dense + RRF) after heading selection.
    BM25_TOP_K: int = 10
    DENSE_TOP_K: int = 10
    RRF_TOP_K: int = 10
    RRF_K: int = 60
    PATIENT_HEADING_TOP_K: int = 10
    PROCEDURE_RERANK_TOP_K: int = 5

    # Shared reranker default when a caller omits top_k.
    RERANK_TOP_K: int = 5

    SECTION_NEIGHBOR_WINDOW: int = 1
    COMPACT_SECTION_CHARS: int = 400

    # Policy: CPT + BM25 over all sections, then dense + rerank on the union.
    POLICY_BM25_TOP_K: int = 40
    POLICY_DENSE_TOP_K: int = 15
    POLICY_RERANK_TOP_K: int = 5

    # Evidence: section BM25, then dense + rerank on selected-section chunks.
    EVIDENCE_SECTION_BM25_TOP_K: int = 40
    EVIDENCE_DENSE_TOP_K: int = 10
    EVIDENCE_RERANK_TOP_K: int = 5
    # Safety cap only; token budget is the primary evidence packing control.
    EVIDENCE_SOL_MAX_ITEMS_PER_BATCH: int = Field(default=40, ge=1)
    EVIDENCE_SOL_MAX_CONCURRENCY: int = Field(default=3, ge=1)

    # LLM budgets (tiktoken cl100k_base). CONTEXT is the input packing ceiling;
    # OUTPUT is a separate max_completion_tokens ceiling (reasoning + visible).
    # They are independent: raising OUTPUT must not shrink the input budget.
    POLICY_LLM_MAX_CONTEXT_TOKENS: int = Field(default=28000, ge=1024)
    POLICY_LLM_MAX_OUTPUT_TOKENS: int = Field(default=16000, ge=256)
    EVIDENCE_LLM_MAX_CONTEXT_TOKENS: int = Field(default=20000, ge=1024)
    EVIDENCE_LLM_MAX_OUTPUT_TOKENS: int = Field(default=8000, ge=256)

    MIN_EXTRACTED_PAGE_CHARS: int = 30
    EMBEDDING_BATCH_SIZE: int = 16

    OPENAI_API_KEY: str | None = None
    MISTRAL_API_KEY: str | None = None
    HF_TOKEN: str | None = None
    HF_DOWNLOAD_MAX_WORKERS: int = Field(default=8, ge=1)

    LLM_REQUEST_TIMEOUT_SECONDS: float = 120.0
    OCR_REQUEST_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 2

    MAX_ACTIVE_ANALYSIS_JOBS: int = Field(default=20, ge=1, le=20)
    ANALYSIS_JOB_CONCURRENCY: int = Field(default=2, ge=1, le=20)

    MAX_UPLOAD_SIZE_MB: int = 50
    CORS_ORIGINS: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

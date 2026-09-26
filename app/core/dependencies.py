"""Shared service dependency graph for CLI and API wiring."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import Request

from app.core.config import Settings, get_settings
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.evidence_service import EvidenceService
from app.services.heading_service import HeadingService
from app.services.llm_service import LLMService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.policy_service import PolicyService
from app.services.procedure_service import ProcedureService
from app.services.readiness_service import ReadinessService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService

if TYPE_CHECKING:
    from app.services.analysis_service import AnalysisService


@dataclass(slots=True)
class ServiceContainer:
    """Reusable service instances for one process lifetime."""

    settings: Settings
    pdf_service: PDFService
    ocr_service: OCRService
    heading_service: HeadingService
    chunking_service: ChunkingService
    embedding_service: EmbeddingService
    llm_service: LLMService
    procedure_service: ProcedureService
    retrieval_service: RetrievalService
    reranking_service: RerankingService
    policy_service: PolicyService
    evidence_service: EvidenceService
    readiness_service: ReadinessService


def build_services(settings: Settings | None = None) -> ServiceContainer:
    """Construct shared services with single embedding/reranker/LLM instances."""
    resolved = settings or get_settings()
    embedding_service = EmbeddingService(settings=resolved)
    llm_service = LLMService(settings=resolved)
    chunking_service = ChunkingService(settings=resolved)
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        settings=resolved,
    )
    reranking_service = RerankingService(settings=resolved)
    return ServiceContainer(
        settings=resolved,
        pdf_service=PDFService(settings=resolved),
        ocr_service=OCRService(settings=resolved),
        heading_service=HeadingService(),
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        llm_service=llm_service,
        procedure_service=ProcedureService(
            retrieval_service=retrieval_service,
            chunking_service=chunking_service,
            reranking_service=reranking_service,
            llm_service=llm_service,
            settings=resolved,
        ),
        retrieval_service=retrieval_service,
        reranking_service=reranking_service,
        policy_service=PolicyService(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            chunking_service=chunking_service,
            reranking_service=reranking_service,
            settings=resolved,
        ),
        evidence_service=EvidenceService(
            retrieval_service=retrieval_service,
            chunking_service=chunking_service,
            reranking_service=reranking_service,
            llm_service=llm_service,
            settings=resolved,
        ),
        readiness_service=ReadinessService(),
    )


@lru_cache
def get_service_container() -> ServiceContainer:
    """Return a process-scoped shared service container."""
    return build_services(get_settings())


def get_analysis_service(request: Request) -> AnalysisService:
    """FastAPI dependency that returns a shared AnalysisService."""
    from app.services.analysis_service import AnalysisService

    services: ServiceContainer = request.app.state.services
    return AnalysisService(services)

"""End-to-end prior-authorization analysis orchestration."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import date, datetime
from pathlib import Path

import structlog

from app.core.config import Settings
from app.core.dependencies import ServiceContainer
from app.core.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    OCRProcessingError,
    PriorAuthVerifierError,
)
from app.core.logging import get_logger
from app.core.metrics import (
    RequestMetrics,
    bind_request_metrics,
    get_request_metrics,
    reset_preprocess_doc_kind,
    reset_request_metrics,
    set_preprocess_doc_kind,
    timed_substage,
)
from app.models.document import DocumentType
from app.models.pdf import ParsedDocument
from app.models.section import Section
from app.schemas.analysis import AnalysisResponse
from app.schemas.procedure import ProcedureExtraction
from app.utils.dates import extract_analysis_reference_date

logger = get_logger(__name__)


class AnalysisService:
    """Coordinate the full patient+policy analysis pipeline."""

    def __init__(self, services: ServiceContainer) -> None:
        self._services = services
        self._settings: Settings = services.settings

    async def analyze(
        self,
        patient_path: Path,
        policy_path: Path,
    ) -> AnalysisResponse:
        """Run the full analysis pipeline and return a readiness response."""
        existing_id = structlog.contextvars.get_contextvars().get("analysis_id")
        bound_externally = existing_id is not None
        request_id = str(existing_id) if existing_id is not None else str(uuid.uuid4())
        if not bound_externally:
            structlog.contextvars.bind_contextvars(analysis_id=request_id)

        metrics = RequestMetrics(request_id=request_id)
        metrics_token = bind_request_metrics(metrics)
        self._validate_runtime_config()

        logger.info(
            "analysis_started",
            patient=patient_path.name,
            policy=policy_path.name,
        )

        try:
            (patient_doc, patient_sections), (policy_doc, policy_sections) = (
                await asyncio.gather(
                    self._preprocess_document(
                        patient_path,
                        DocumentType.PATIENT_CHART,
                    ),
                    self._preprocess_document(
                        policy_path,
                        DocumentType.POLICY,
                    ),
                )
            )
            metrics.patient_pages = patient_doc.page_count
            metrics.policy_pages = policy_doc.page_count

            procedure_started = time.perf_counter()
            try:
                procedure = (
                    await self._services.procedure_service.detect_planned_procedure(
                        patient_sections
                    )
                )
            finally:
                procedure_ms = int((time.perf_counter() - procedure_started) * 1000)
                metrics.record_stage_time("procedure", procedure_ms)
            logger.info(
                "procedure_extraction_completed",
                duration_ms=procedure_ms,
                procedure_name=procedure.procedure_name,
                has_cpt=bool(procedure.cpt_code),
            )

            policy_started = time.perf_counter()
            try:
                policy_analysis = await self._services.policy_service.analyze_policy(
                    procedure,
                    policy_sections,
                )
            finally:
                policy_ms = int((time.perf_counter() - policy_started) * 1000)
                metrics.record_stage_time("policy", policy_ms)
            logger.info(
                "policy_analysis_completed",
                duration_ms=policy_ms,
                applicable=policy_analysis.applicable,
                llm_call_count=policy_analysis.llm_call_count,
            )

            evidence_analysis = None
            evidence_ms = 0
            if policy_analysis.applicable:
                evidence_started = time.perf_counter()
                reference_date = self._resolve_analysis_reference_date(
                    procedure=procedure,
                    patient_sections=patient_sections,
                )
                try:
                    evidence_analysis = (
                        await self._services.evidence_service.evaluate_evidence(
                            policy_analysis,
                            patient_sections,
                            analysis_reference_date=reference_date,
                        )
                    )
                finally:
                    evidence_ms = int((time.perf_counter() - evidence_started) * 1000)
                    metrics.record_stage_time("evidence", evidence_ms)
                assert evidence_analysis is not None
                logger.info(
                    "evidence_analysis_completed",
                    duration_ms=evidence_ms,
                    llm_call_count=evidence_analysis.llm_call_count,
                    skipped=evidence_analysis.skipped,
                )
            else:
                metrics.record_stage_time("evidence", 0)
                logger.info(
                    "evidence_analysis_skipped",
                    reason="policy_not_applicable",
                )

            readiness_started = time.perf_counter()
            result = self._services.readiness_service.evaluate(
                procedure=procedure,
                policy_analysis=policy_analysis,
                evidence_analysis=evidence_analysis,
            )
            readiness_ms = int((time.perf_counter() - readiness_started) * 1000)
            metrics.record_stage_time("readiness", readiness_ms)
            metrics.status = result.overall_status.value

            logger.info(
                "analysis_completed",
                overall_status=result.overall_status.value,
                matching_policy_criteria_found=result.matching_policy_criteria_found,
                readiness_duration_ms=readiness_ms,
                total_duration_ms=metrics.total_ms,
                patient_filename=patient_doc.filename,
                policy_filename=policy_doc.filename,
                patient_section_count=len(patient_sections),
                policy_section_count=len(policy_sections),
                embedding_init_count=self._services.embedding_service.init_count,
                reranker_init_count=self._services.reranking_service.init_count,
                llm_client_init_count=self._services.llm_service.init_count,
            )
            return result
        except PriorAuthVerifierError as exc:
            metrics.status = "ERROR"
            metrics.error_type = type(exc).__name__
            raise
        except Exception as exc:
            metrics.status = "ERROR"
            metrics.error_type = type(exc).__name__
            logger.error("analysis_unexpected_failure", error=str(exc))
            raise PriorAuthVerifierError("Analysis failed unexpectedly") from exc
        finally:
            metrics.emit_summary()
            reset_request_metrics(metrics_token)
            if not bound_externally:
                structlog.contextvars.clear_contextvars()

    def _validate_runtime_config(self) -> None:
        if not (self._settings.OPENAI_API_KEY or "").strip():
            raise ConfigurationError(
                "OPENAI_API_KEY is required to run prior-authorization analysis"
            )

    async def _preprocess_document(
        self,
        path: Path,
        document_type: DocumentType,
    ) -> tuple[ParsedDocument, list[Section]]:
        """Parse once into logical sections; defer chunking until after narrowing."""
        prefix = (
            "patient"
            if document_type == DocumentType.PATIENT_CHART
            else "policy"
        )
        kind_token = set_preprocess_doc_kind(prefix)
        try:
            # Parse + page extract timings are recorded inside PDFService.parse.
            document = await asyncio.to_thread(
                self._services.pdf_service.parse,
                path,
                document_type,
            )

            with timed_substage(f"{prefix}_ocr_ms"):
                # OCR service also records when pages are actually OCR'd; this
                # catches the no-op path so the substage is always present.
                ocr_started = time.perf_counter()
                document = await self._services.ocr_service.enrich_document(
                    document,
                    source_path=str(path),
                )
                ocr_ms = int((time.perf_counter() - ocr_started) * 1000)

            remaining_ocr = [
                page.page_number for page in document.pages if page.requires_ocr
            ]
            if remaining_ocr:
                raise OCRProcessingError(
                    "Document still has unreadable pages after OCR handling: "
                    f"{remaining_ocr}"
                )

            with timed_substage(f"{prefix}_section_build_ms"):
                heading_started = time.perf_counter()
                sections = await asyncio.to_thread(
                    self._services.heading_service.build_sections,
                    document,
                )
                heading_ms = int((time.perf_counter() - heading_started) * 1000)

            # Date extraction is timed inside heading build; exclude it from the
            # section-build leaf so the report does not double-count.
            metrics = get_request_metrics()
            if metrics is not None and prefix == "patient":
                date_ms = int(metrics.substage_ms.get("patient_date_metadata_ms", 0))
                section_key = "patient_section_build_ms"
                if date_ms and section_key in metrics.substage_ms:
                    metrics.substage_ms[section_key] = max(
                        0, metrics.substage_ms[section_key] - date_ms
                    )

            readable = [section for section in sections if section.text.strip()]
            if not readable:
                raise DocumentProcessingError(
                    f"No readable text sections were produced from {path.name}"
                )

            logger.info(
                "document_preprocess_completed",
                document_type=document_type.value,
                filename=document.filename,
                ocr_duration_ms=ocr_ms,
                heading_duration_ms=heading_ms,
                page_count=document.page_count,
                section_count=len(sections),
            )
            return document, sections
        finally:
            reset_preprocess_doc_kind(kind_token)

    def _resolve_analysis_reference_date(
        self,
        *,
        procedure: ProcedureExtraction,
        patient_sections: list[Section],
    ) -> date | None:
        """Prefer PA submission, then planned DOS, then latest chart date."""
        candidates = [
            procedure.source_text,
            procedure.status,
            *(
                f"{section.heading or ''}\n{section.text}"
                for section in patient_sections[:40]
            ),
        ]
        iso = extract_analysis_reference_date(*candidates)
        if not iso:
            return None
        try:
            return datetime.strptime(iso, "%Y-%m-%d").date()
        except ValueError:
            return None

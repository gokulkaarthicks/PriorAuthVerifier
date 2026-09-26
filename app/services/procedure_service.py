"""Planned procedure detection from patient-chart sections."""

from __future__ import annotations

import re
import time

from app.core.config import Settings, get_settings
from app.core.exceptions import ProcedureExtractionError
from app.core.logging import get_logger
from app.core.metrics import embedding_operation, timed_substage
from app.models.chunk import Chunk
from app.models.section import Section
from app.schemas.procedure import ProcedureExtraction
from app.services.chunking_service import ChunkingService
from app.services.llm_service import LLMService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService
from app.utils.text import text_supported

logger = get_logger(__name__)

_CPT_RE = re.compile(r"^\d{5}$")

_SYSTEM_PROMPT = """\
You extract the patient's next intended clinical procedure from candidate \
chart excerpts.

Rules:
1. Use only the supplied patient-chart text.
2. Do not infer undocumented procedures.
3. Prefer explicitly planned, scheduled, ordered, recommended, or \
pending-authorization procedures.
4. Do not treat completed or historical procedures as upcoming.
5. Do not treat diagnostic tests as procedures unless the chart explicitly \
intends them as the next procedure.
6. Preserve the procedure name as written where possible.
7. Extract CPT only if explicitly present in the supplied text.
8. Extract status only if explicitly present.
9. Return null for absent fields.
10. Include exact supporting source text copied from one candidate chunk.
11. Include the source chunk ID of that evidence.
12. Do not invent page numbers; use the candidate page range.
13. If multiple future procedures appear, choose the one with strongest \
immediate intent.
14. If no clear future procedure exists, or the case is genuinely ambiguous, \
set procedure_name to null and is_future_procedure to false.
15. Confidence must be between 0 and 1.
"""


class ProcedureService:
    """Heading BGE → selected chunks → BM25+BGE+RRF → rerank → Luna."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        chunking_service: ChunkingService,
        reranking_service: RerankingService,
        llm_service: LLMService,
        settings: Settings | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._chunking_service = chunking_service
        self._reranking_service = reranking_service
        self._llm_service = llm_service
        self._settings = settings or get_settings()

    async def detect_planned_procedure(
        self,
        sections: list[Section],
    ) -> ProcedureExtraction:
        """Detect the next planned procedure from patient sections."""
        if not sections:
            return ProcedureExtraction(
                procedure_name=None,
                is_future_procedure=False,
                confidence=0.0,
            )

        logger.info(
            "patient_sections_built",
            sections=len(sections),
        )

        started = time.perf_counter()
        candidates = self._retrieve_candidates(sections)
        retrieve_ms = int((time.perf_counter() - started) * 1000)

        logger.info(
            "procedure_candidates_selected",
            candidate_count=len(candidates),
            retrieval_duration_ms=retrieve_ms,
            candidate_headings=[chunk.heading for chunk in candidates],
        )

        llm_started = time.perf_counter()
        extraction = await self._extract_with_llm(candidates)
        llm_ms = int((time.perf_counter() - llm_started) * 1000)

        with timed_substage("procedure_normalization_ms"):
            validated = self._validate_extraction(extraction, candidates)
        logger.info(
            "procedure_extraction_completed",
            llm_duration_ms=llm_ms,
            has_procedure=validated.procedure_name is not None,
            is_future_procedure=validated.is_future_procedure,
            confidence=validated.confidence,
        )
        return validated

    def _retrieve_candidates(self, sections: list[Section]) -> list[Chunk]:
        heading_query = self._retrieval_service.procedure_heading_query()
        with embedding_operation("procedure_heading"):
            selected_sections = (
                self._retrieval_service.select_sections_by_heading_similarity(
                    sections,
                    query=heading_query,
                    top_k=self._settings.PATIENT_HEADING_TOP_K,
                    profile_prefix="procedure_heading",
                )
            )
        if not selected_sections:
            return []

        with timed_substage("procedure_selected_chunk_build_ms"):
            chunks = self._chunking_service.chunk_sections(selected_sections)
        logger.info(
            "patient_sections_chunked",
            sections=len(selected_sections),
            chunks=len(chunks),
        )
        if not chunks:
            return []

        chunk_query = self._retrieval_service.procedure_chunk_query()
        with embedding_operation("procedure_chunk"):
            index = self._retrieval_service.build_hybrid_index(
                chunks,
                index_kind="procedure",
                profile_prefix="procedure",
            )
        fused = self._retrieval_service.retrieve_for_query(
            chunk_query,
            index,
            log_prefix="procedure_chunk",
            profile_prefix="procedure",
        )
        rerank_query = self._retrieval_service.procedure_rerank_query()
        reranked = self._reranking_service.rerank(
            query=rerank_query,
            candidates=fused,
            top_k=self._settings.PROCEDURE_RERANK_TOP_K,
            stage="procedure",
        )
        logger.info(
            "procedure_rerank_completed",
            input_chunks=len(fused),
            output_chunks=len(reranked),
        )
        return [item.chunk for item in reranked]

    async def _extract_with_llm(
        self,
        candidates: list[Chunk],
    ) -> ProcedureExtraction:
        if not candidates:
            return ProcedureExtraction(
                procedure_name=None,
                is_future_procedure=False,
                confidence=0.0,
            )
        with timed_substage("procedure_context_render_ms"):
            user_prompt = self._build_user_prompt(candidates)
        return await self._llm_service.extract_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=ProcedureExtraction,
            model=self._settings.STRUCTURED_EXTRACTION_MODEL,
            operation="procedure_extraction",
            context_items=len(candidates),
        )

    def _build_user_prompt(self, candidates: list[Chunk]) -> str:
        blocks: list[str] = [
            "Identify the procedure that is currently intended next "
            "from these candidate excerpts.",
            "",
        ]
        for index, chunk in enumerate(candidates, start=1):
            blocks.extend(
                [
                    f"CANDIDATE {index}",
                    f"chunk_id: {chunk.id}",
                    f"heading: {chunk.heading}",
                    f"parent_heading: {chunk.parent_heading}",
                    f"start_page: {chunk.page_number}",
                    f"end_page: {chunk.end_page_number}",
                    "exact_chunk_text:",
                    chunk.text,
                    "",
                ]
            )
        return "\n".join(blocks)

    def _validate_extraction(
        self,
        extraction: ProcedureExtraction,
        candidates: list[Chunk],
    ) -> ProcedureExtraction:
        if extraction.procedure_name is None:
            return ProcedureExtraction(
                procedure_name=None,
                cpt_code=None,
                status=None,
                source_chunk_id=None,
                source_text=None,
                page_number=None,
                is_future_procedure=False,
                confidence=extraction.confidence,
            )

        if not extraction.source_chunk_id or not extraction.source_text:
            raise ProcedureExtractionError(
                "Procedure extraction missing source chunk or source text"
            )

        candidate_by_id = {chunk.id: chunk for chunk in candidates}
        chunk = candidate_by_id.get(extraction.source_chunk_id)
        if chunk is None:
            raise ProcedureExtractionError(
                "source_chunk_id is not among supplied candidates"
            )

        if not text_supported(extraction.source_text, chunk.text):
            raise ProcedureExtractionError(
                "source_text was not found in the cited chunk"
            )

        if extraction.page_number is None:
            raise ProcedureExtractionError(
                "page_number is required when a procedure is returned"
            )
        if not (
            chunk.page_number <= extraction.page_number <= chunk.end_page_number
        ):
            raise ProcedureExtractionError(
                "page_number is outside the cited chunk page range"
            )

        if not text_supported(extraction.procedure_name, chunk.text):
            raise ProcedureExtractionError(
                "procedure_name is not supported by the cited chunk text"
            )

        if extraction.cpt_code is not None:
            if not _CPT_RE.match(extraction.cpt_code):
                raise ProcedureExtractionError("cpt_code must be exactly five digits")
            if extraction.cpt_code not in extraction.source_text:
                raise ProcedureExtractionError(
                    "cpt_code does not appear in source_text"
                )
            if extraction.cpt_code not in chunk.text:
                raise ProcedureExtractionError(
                    "cpt_code does not appear in the cited chunk"
                )

        return extraction

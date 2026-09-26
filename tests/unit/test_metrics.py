"""Unit tests for request-scoped analysis metrics."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import EvidenceValidationError, PriorAuthVerifierError
from app.core.metrics import (
    RequestMetrics,
    bind_request_metrics,
    get_request_metrics,
    reset_request_metrics,
)
from app.models.document import DocumentType
from app.models.pdf import ParsedDocument, PDFPage
from app.models.policy_analysis import PolicyAnalysisResult
from app.models.section import Section
from app.schemas.analysis import AnalysisResponse, ReadinessStatus
from app.schemas.policy import (
    CriterionCategory,
    CriterionNode,
    CriterionNodeType,
    PolicyCriteriaExtraction,
)
from app.schemas.procedure import ProcedureExtraction
from app.services.analysis_service import AnalysisService
from app.services.evidence_service import EvidenceService
from app.services.llm_service import LLMService


def test_request_metrics_aggregates_llm_and_local_usage() -> None:
    metrics = RequestMetrics(request_id="req-1")
    metrics.record_stage_time("procedure", 10)
    metrics.record_stage_time("policy", 20)
    metrics.record_stage_time("evidence", 30)
    metrics.record_stage_time("readiness", 5)
    metrics.record_llm_usage(
        "gpt-6-luna",
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
    )
    metrics.record_llm_usage(
        "gpt-6-sol",
        input_tokens=200,
        output_tokens=40,
        total_tokens=240,
    )
    metrics.record_llm_usage(
        "gpt-6-sol",
        input_tokens=50,
        output_tokens=10,
        total_tokens=60,
    )
    metrics.record_local_embedding_batch("BAAI/bge-m3", batches=2, texts=16)
    metrics.record_local_embedding_batch("BAAI/bge-m3", batches=1, texts=4)
    metrics.record_reranker_batch("BAAI/bge-reranker-v2-m3", batches=1, pairs=8)
    metrics.record_parser_invocation("PATIENT_CHART")
    metrics.record_parser_invocation("POLICY")
    metrics.patient_pages = 12
    metrics.policy_pages = 40
    metrics.criteria_count = 7
    metrics.status = "READY"

    assert metrics.llm_tokens_total == 420
    assert metrics.model_usage["gpt-6-sol"].calls == 2
    assert metrics.model_usage["gpt-6-sol"].total_tokens == 300
    assert metrics.local_model_usage["BAAI/bge-m3"].batches == 3
    assert metrics.local_model_usage["BAAI/bge-m3"].texts == 20
    assert metrics.local_model_usage["BAAI/bge-reranker-v2-m3"].pairs == 8
    assert metrics.patient_pdf_parse_count == 1
    assert metrics.policy_pdf_parse_count == 1

    payload = metrics.summary_fields(include_profile=False)
    assert payload["total_ms"] >= 0
    assert payload["stage_ms"]["procedure"] == 10
    assert payload["llm_tokens_total"] == 420
    assert payload["patient_pages"] == 12
    assert payload["policy_pages"] == 40
    assert payload["criteria_count"] == 7
    assert payload["model_usage"]["gpt-6-sol"]["failed_calls"] == 0
    assert payload["model_usage"]["gpt-6-sol"]["reasoning_tokens"] == 0
    assert "substage_ms" not in payload
    assert "patient text" not in str(payload).lower()
    assert "policy text" not in str(payload).lower()

    profiled = metrics.summary_fields(include_profile=True)
    assert profiled["patient_pdf_parse_count"] == 1
    assert profiled["policy_pdf_parse_count"] == 1
    assert "substage_ms" in profiled


@pytest.mark.asyncio
async def test_llm_service_records_usage_from_api_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = RequestMetrics(request_id="req-llm")
    token = bind_request_metrics(metrics)
    try:
        usage = MagicMock()
        usage.prompt_tokens = 11
        usage.completion_tokens = 7
        usage.total_tokens = 18
        message = MagicMock()
        message.refusal = None
        message.parsed = MagicMock()
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage

        client = MagicMock()
        client.beta.chat.completions.parse = AsyncMock(return_value=response)

        service = LLMService(settings=MagicMock(
            STRUCTURED_EXTRACTION_MODEL="gpt-6-luna",
            LLM_MAX_RETRIES=0,
            OPENAI_API_KEY="test-key",
            LLM_REQUEST_TIMEOUT_SECONDS=1.0,
        ))
        service._client = client
        await service.extract_structured(
            system_prompt="sys",
            user_prompt="user",
            response_model=MagicMock,
            model="gpt-6-luna",
        )
        assert metrics.model_usage["gpt-6-luna"].calls == 1
        assert metrics.model_usage["gpt-6-luna"].input_tokens == 11
        assert metrics.model_usage["gpt-6-luna"].output_tokens == 7
        assert metrics.model_usage["gpt-6-luna"].total_tokens == 18
    finally:
        reset_request_metrics(token)


@pytest.mark.asyncio
async def test_analysis_emits_summary_once_on_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.core import metrics as metrics_mod

    emitted: list[dict[str, object]] = []

    def fake_emit(self: RequestMetrics) -> None:
        emitted.append(self.summary_fields())

    monkeypatch.setattr(RequestMetrics, "emit_summary", fake_emit)

    patient_doc = ParsedDocument(
        id="patient",
        filename="patient.pdf",
        document_type=DocumentType.PATIENT_CHART,
        page_count=3,
        pages=[
            PDFPage(page_number=1, text="a", spans=[], requires_ocr=False),
            PDFPage(page_number=2, text="b", spans=[], requires_ocr=False),
            PDFPage(page_number=3, text="c", spans=[], requires_ocr=False),
        ],
    )
    policy_doc = ParsedDocument(
        id="policy",
        filename="policy.pdf",
        document_type=DocumentType.POLICY,
        page_count=5,
        pages=[
            PDFPage(page_number=i, text="p", spans=[], requires_ocr=False)
            for i in range(1, 6)
        ],
    )
    section = Section(
        id="s1",
        document_id="patient",
        heading="Plan",
        parent_heading=None,
        text="Planned procedure note",
        start_page=1,
        end_page=1,
        token_count=3,
    )

    services = MagicMock()
    services.settings = MagicMock(OPENAI_API_KEY="test")
    services.pdf_service.parse.side_effect = [patient_doc, policy_doc]
    services.ocr_service.enrich_document = AsyncMock(
        side_effect=lambda document, source_path=None: document
    )
    services.heading_service.build_sections.side_effect = [
        [section],
        [section],
    ]
    services.procedure_service.detect_planned_procedure = AsyncMock(
        return_value=ProcedureExtraction(
            procedure_name="Ablation",
            cpt_code="36475",
            is_future_procedure=True,
            confidence=0.9,
            source_chunk_id="s1",
            source_text="Planned procedure note",
            page_number=1,
        )
    )
    services.policy_service.analyze_policy = AsyncMock(
        return_value=PolicyAnalysisResult(
            applicable=False,
            applicability_reason="Unrelated.",
            applicability_confidence=0.9,
            criteria=None,
            supporting_chunk_ids=[],
            llm_call_count=1,
        )
    )
    services.readiness_service.evaluate.return_value = AnalysisResponse(
        procedure=ProcedureExtraction(
            procedure_name="Ablation",
            cpt_code="36475",
            is_future_procedure=True,
            confidence=0.9,
            source_chunk_id="s1",
            source_text="Planned procedure note",
            page_number=1,
        ),
        overall_status=ReadinessStatus.NOT_APPLICABLE,
        matching_policy_criteria_found=False,
        criteria=[],
        exclusions=[],
    )
    services.embedding_service.init_count = 0
    services.reranking_service.init_count = 0
    services.llm_service.init_count = 0

    analysis = AnalysisService(services)
    result = await analysis.analyze(Path("patient.pdf"), Path("policy.pdf"))

    assert result.overall_status == ReadinessStatus.NOT_APPLICABLE
    assert len(emitted) == 1
    summary = emitted[0]
    assert summary["status"] == ReadinessStatus.NOT_APPLICABLE.value
    assert "total_ms" in summary
    assert "stage_ms" in summary
    assert "model_usage" in summary
    assert "llm_tokens_total" in summary
    assert summary["patient_pages"] == 3
    assert summary["policy_pages"] == 5
    assert get_request_metrics() is None
    _ = metrics_mod


@pytest.mark.asyncio
async def test_analysis_emits_summary_once_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, object]] = []

    def fake_emit(self: RequestMetrics) -> None:
        emitted.append(self.summary_fields())

    monkeypatch.setattr(RequestMetrics, "emit_summary", fake_emit)

    services = MagicMock()
    services.settings = MagicMock(OPENAI_API_KEY="test")
    services.pdf_service.parse.side_effect = RuntimeError("boom")

    analysis = AnalysisService(services)
    with pytest.raises(PriorAuthVerifierError):
        await analysis.analyze(Path("patient.pdf"), Path("policy.pdf"))

    assert len(emitted) == 1
    assert emitted[0]["status"] == "ERROR"
    assert emitted[0]["error_type"] == "RuntimeError"
    assert "total_ms" in emitted[0]


@pytest.mark.asyncio
async def test_analysis_preserves_criteria_and_evidence_ms_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, object]] = []

    def fake_emit(self: RequestMetrics) -> None:
        emitted.append(self.summary_fields())

    monkeypatch.setattr(RequestMetrics, "emit_summary", fake_emit)

    patient_doc = ParsedDocument(
        id="patient",
        filename="patient.pdf",
        document_type=DocumentType.PATIENT_CHART,
        page_count=2,
        pages=[
            PDFPage(page_number=1, text="a", spans=[], requires_ocr=False),
            PDFPage(page_number=2, text="b", spans=[], requires_ocr=False),
        ],
    )
    policy_doc = ParsedDocument(
        id="policy",
        filename="policy.pdf",
        document_type=DocumentType.POLICY,
        page_count=4,
        pages=[
            PDFPage(page_number=i, text="p", spans=[], requires_ocr=False)
            for i in range(1, 5)
        ],
    )
    section = Section(
        id="s1",
        document_id="patient",
        heading="Plan",
        parent_heading=None,
        text="Planned procedure note",
        start_page=1,
        end_page=1,
        token_count=3,
    )
    root = CriterionNode(
        id="root",
        node_type=CriterionNodeType.AND,
        children=[
            CriterionNode(
                id=f"c{i}",
                node_type=CriterionNodeType.CONDITION,
                description=f"Criterion {i}",
                category=CriterionCategory.CLINICAL,
            )
            for i in range(1, 16)
        ],
    )
    criteria = PolicyCriteriaExtraction(
        procedure_name="Ablation",
        policy_summary="Summary",
        root=root,
        exclusions=[],
        source_chunk_ids=["pol-1"],
    )

    async def failing_evaluate_evidence(
        policy_analysis: PolicyAnalysisResult,
        _patient_sections: list[Section],
        **_kwargs: object,
    ) -> object:
        assert policy_analysis.criteria is not None
        conditions = EvidenceService.extract_atomic_criteria(
            MagicMock(),
            policy_analysis.criteria.root,
        )
        metrics = get_request_metrics()
        assert metrics is not None
        metrics.criteria_count = len(conditions)
        metrics.record_llm_usage(
            "gpt-6-sol",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )
        await asyncio.sleep(0.05)
        raise EvidenceValidationError(
            "c1 evidence_text length must match evidence_chunk_ids"
        )

    services = MagicMock()
    services.settings = MagicMock(OPENAI_API_KEY="test")
    services.pdf_service.parse.side_effect = [patient_doc, policy_doc]
    services.ocr_service.enrich_document = AsyncMock(
        side_effect=lambda document, source_path=None: document
    )
    services.heading_service.build_sections.side_effect = [
        [section],
        [section],
    ]
    services.procedure_service.detect_planned_procedure = AsyncMock(
        return_value=ProcedureExtraction(
            procedure_name="Ablation",
            cpt_code="36475",
            is_future_procedure=True,
            confidence=0.9,
            source_chunk_id="s1",
            source_text="Planned procedure note",
            page_number=1,
        )
    )
    services.policy_service.analyze_policy = AsyncMock(
        return_value=PolicyAnalysisResult(
            applicable=True,
            applicability_reason="Applicable.",
            applicability_confidence=0.95,
            criteria=criteria,
            supporting_chunk_ids=["pol-1"],
            llm_call_count=1,
        )
    )
    services.evidence_service.evaluate_evidence = AsyncMock(
        side_effect=failing_evaluate_evidence
    )
    services.embedding_service.init_count = 0
    services.reranking_service.init_count = 0
    services.llm_service.init_count = 0

    analysis = AnalysisService(services)
    with pytest.raises(EvidenceValidationError, match="evidence_text length"):
        await analysis.analyze(Path("patient.pdf"), Path("policy.pdf"))

    assert len(emitted) == 1
    summary = emitted[0]
    assert summary["status"] == "ERROR"
    assert summary["error_type"] == "EvidenceValidationError"
    assert summary["criteria_count"] == 15
    stage_ms = summary["stage_ms"]
    assert isinstance(stage_ms, dict)
    assert int(stage_ms["evidence"]) >= 40
    assert summary["llm_tokens_total"] == 120
    model_usage = summary["model_usage"]
    assert isinstance(model_usage, dict)
    assert model_usage["gpt-6-sol"]["calls"] == 1
    assert get_request_metrics() is None


@pytest.mark.asyncio
async def test_failed_policy_stage_records_nonzero_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.exceptions import PolicyAnalysisError

    emitted: list[dict[str, object]] = []

    def fake_emit(self: RequestMetrics) -> None:
        emitted.append(self.summary_fields())

    monkeypatch.setattr(RequestMetrics, "emit_summary", fake_emit)

    patient_doc = ParsedDocument(
        id="patient",
        filename="patient.pdf",
        document_type=DocumentType.PATIENT_CHART,
        page_count=1,
        pages=[PDFPage(page_number=1, text="a", spans=[], requires_ocr=False)],
    )
    policy_doc = ParsedDocument(
        id="policy",
        filename="policy.pdf",
        document_type=DocumentType.POLICY,
        page_count=1,
        pages=[PDFPage(page_number=1, text="p", spans=[], requires_ocr=False)],
    )
    section = Section(
        id="s1",
        document_id="patient",
        heading="Plan",
        parent_heading=None,
        text="Planned procedure note",
        start_page=1,
        end_page=1,
        token_count=3,
    )

    async def slow_failing_policy(
        _procedure: ProcedureExtraction,
        _sections: list[Section],
    ) -> PolicyAnalysisResult:
        await asyncio.sleep(0.05)
        raise PolicyAnalysisError("Policy criteria extraction request failed")

    services = MagicMock()
    services.settings = MagicMock(OPENAI_API_KEY="test")
    services.pdf_service.parse.side_effect = [patient_doc, policy_doc]
    services.ocr_service.enrich_document = AsyncMock(
        side_effect=lambda document, source_path=None: document
    )
    services.heading_service.build_sections.side_effect = [[section], [section]]
    services.procedure_service.detect_planned_procedure = AsyncMock(
        return_value=ProcedureExtraction(
            procedure_name="Ablation",
            cpt_code="36475",
            is_future_procedure=True,
            confidence=0.9,
            source_chunk_id="s1",
            source_text="Planned procedure note",
            page_number=1,
        )
    )
    services.policy_service.analyze_policy = AsyncMock(side_effect=slow_failing_policy)
    services.embedding_service.init_count = 0
    services.reranking_service.init_count = 0
    services.llm_service.init_count = 0

    analysis = AnalysisService(services)
    with pytest.raises(PolicyAnalysisError):
        await analysis.analyze(Path("patient.pdf"), Path("policy.pdf"))

    assert len(emitted) == 1
    stage_ms = emitted[0]["stage_ms"]
    assert isinstance(stage_ms, dict)
    assert int(stage_ms["policy"]) >= 40
    assert int(stage_ms["procedure"]) >= 0


def test_record_llm_usage_tracks_failed_calls_and_reasoning() -> None:
    metrics = RequestMetrics(request_id="failed-usage")
    metrics.record_llm_usage(
        "gpt-6-sol",
        input_tokens=7706,
        output_tokens=12000,
        reasoning_tokens=12000,
        total_tokens=19706,
        failed=True,
    )
    payload = metrics.summary_fields()["model_usage"]["gpt-6-sol"]
    assert payload["calls"] == 1
    assert payload["failed_calls"] == 1
    assert payload["reasoning_tokens"] == 12000
    assert payload["total_tokens"] == 19706

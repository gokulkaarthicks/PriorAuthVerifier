"""Integration-style tests for AnalysisService orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.core.config import Settings
from app.core.dependencies import ServiceContainer, build_services
from app.core.exceptions import ConfigurationError, OCRProcessingError
from app.models.chunk import Chunk
from app.models.document import DocumentType
from app.models.pdf import ParsedDocument, PDFPage
from app.models.section import Section
from app.schemas.analysis import ReadinessStatus
from app.schemas.evaluation import (
    CriterionEvaluation,
    CriterionStatus,
    EvidenceEvaluationBatch,
    ExclusionEvaluation,
    SolEvidenceEvaluationBatch,
)
from app.schemas.policy import (
    CriterionCategory,
    SolCriterionNode,
    SolPolicyCriteriaExtraction,
    SolPolicyCriteriaResult,
    SolPolicyExclusion,
)
from app.schemas.procedure import ProcedureExtraction
from app.services.analysis_service import AnalysisService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.evidence_service import EvidenceService
from app.services.llm_service import LLMService
from app.services.ocr_service import OCRService
from app.services.policy_service import PolicyService
from app.services.procedure_service import ProcedureService
from app.services.readiness_service import ReadinessService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService
from tests.helpers.evidence_sol import public_batch_to_scoped_sol

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
PATIENT_PDF = SAMPLE_DIR / "patient.pdf"
POLICY_PDF = SAMPLE_DIR / "bcbs_policy.pdf"


def _section(section_id: str, text: str, *, page: int = 1) -> Section:
    return Section(
        id=section_id,
        document_id="doc",
        heading="Section",
        parent_heading=None,
        text=text,
        start_page=page,
        end_page=page,
        token_count=max(1, len(text.split())),
    )


def _passthrough_chunking() -> MagicMock:
    service = MagicMock()

    def chunk_sections(sections: list[Section]) -> list[Chunk]:
        return [
            Chunk(
                id=section.id,
                document_id=section.document_id,
                text=section.text,
                page_number=section.start_page,
                end_page_number=section.end_page,
                heading=section.heading,
                parent_heading=section.parent_heading,
                token_count=section.token_count,
                section_id=section.id,
                chunk_index=0,
            )
            for section in sections
        ]

    service.chunk_sections.side_effect = chunk_sections
    return service


def _fake_embedding(dim: int = 4) -> EmbeddingService:
    service = EmbeddingService(settings=Settings())
    service._model = object()

    def embed(texts: list[str], **_kwargs) -> np.ndarray:
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for index, _text in enumerate(texts):
            matrix[index, index % dim] = 1.0
        return matrix

    def similarity(
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        docs = np.asarray(document_embeddings, dtype=np.float32)
        return docs @ query

    service.embed = embed  # type: ignore[method-assign]
    service.similarity = similarity  # type: ignore[method-assign]
    service.init_count = 1
    return service


def _sol_tree() -> SolCriterionNode:
    return SolCriterionNode(
        id="root",
        node_type="AND",
        children=[
            SolCriterionNode(
                id="c1",
                node_type="CONDITION",
                description="Demonstrated saphenous reflux.",
                category=CriterionCategory.DIAGNOSTIC,
                source_context_id="P1",
                source_quote="Demonstrated saphenous reflux.",
            ),
            SolCriterionNode(
                id="c2",
                node_type="CONDITION",
                description="CEAP class is C2 or greater.",
                category=CriterionCategory.MEASUREMENT,
                source_context_id="P1",
                source_quote="CEAP class is C2 or greater.",
            ),
        ],
    )


def _patient_sections() -> list[Section]:
    return [
        _section(
            "pat-0",
            "Planned procedure: Example Vein Ablation CPT 99999.",
            page=1,
        ),
        _section("pat-1", "Duplex ultrasound demonstrated saphenous reflux.", page=2),
        _section("pat-2", "CEAP clinical class documented as C3.", page=3),
    ]


def _policy_sections() -> list[Section]:
    return [
        _section(
            "pol-1",
            (
                "Coverage for Example Vein Ablation. "
                "Demonstrated saphenous reflux. "
                "CEAP class is C2 or greater. "
                "Cosmetic treatment is not covered."
            ),
            page=1,
        )
    ]


def _procedure_extraction() -> ProcedureExtraction:
    return ProcedureExtraction(
        procedure_name="Example Vein Ablation",
        cpt_code="99999",
        status="planned",
        source_chunk_id="pat-0",
        source_text="Planned procedure: Example Vein Ablation CPT 99999.",
        page_number=1,
        is_future_procedure=True,
        confidence=0.95,
    )


def _build_container(
    *,
    applicability: bool,
    evidence_batch: EvidenceEvaluationBatch | None = None,
    mock_procedure: bool = True,
) -> ServiceContainer:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        BM25_TOP_K=5,
        DENSE_TOP_K=5,
        RRF_TOP_K=5,
        RERANK_TOP_K=3,
        POLICY_BM25_TOP_K=10,
        POLICY_DENSE_TOP_K=5,
        POLICY_RERANK_TOP_K=3,
        SECTION_NEIGHBOR_WINDOW=1,
    )
    embedding = _fake_embedding()
    llm = LLMService(settings=settings)
    llm._client = object()
    llm.init_count = 1

    policy_responses: list[object] = []
    if applicability:
        policy_responses.append(
            SolPolicyCriteriaResult(
                matching_policy_criteria_found=True,
                explanation="Policy covers Example Vein Ablation.",
                criteria_tree=SolPolicyCriteriaExtraction(
                    procedure_name="Example Vein Ablation",
                    policy_summary="Synthetic ablation criteria.",
                    root=_sol_tree(),
                    exclusions=[
                        SolPolicyExclusion(
                            id="ex1",
                            description="Cosmetic treatment is not covered.",
                            source_context_id="P1",
                            source_quote="Cosmetic treatment is not covered.",
                        )
                    ],
                    source_context_ids=["P1"],
                ),
            )
        )
    else:
        policy_responses.append(
            SolPolicyCriteriaResult(
                matching_policy_criteria_found=False,
                explanation="Unrelated policy domain.",
                criteria_tree=None,
            )
        )

    queue = list(policy_responses)

    async def extract_structured(**kwargs: object) -> object:
        response_model = kwargs.get("response_model")
        if response_model is SolEvidenceEvaluationBatch:
            assert evidence_batch is not None
            prompt = str(kwargs.get("user_prompt") or "")
            return public_batch_to_scoped_sol(evidence_batch, prompt=prompt)
        if response_model is ProcedureExtraction:
            return _procedure_extraction()
        if not queue:
            raise AssertionError("Unexpected extra LLM call")
        return queue.pop(0)

    llm.extract_structured = AsyncMock(side_effect=extract_structured)

    retrieval = RetrievalService(embedding, settings)
    reranker = RerankingService(settings=settings)
    reranker._model = MagicMock(
        compute_score=MagicMock(side_effect=lambda pairs: [1.0] * len(pairs))
    )
    reranker.init_count = 1
    chunking = _passthrough_chunking()

    if mock_procedure:
        procedure_service = MagicMock()
        procedure_service.detect_planned_procedure = AsyncMock(
            return_value=_procedure_extraction()
        )
    else:
        procedure_service = ProcedureService(
            retrieval_service=retrieval,
            chunking_service=chunking,
            reranking_service=reranker,
            llm_service=llm,
            settings=settings,
        )

    return ServiceContainer(
        settings=settings,
        pdf_service=MagicMock(),
        ocr_service=OCRService(settings=settings),
        heading_service=MagicMock(),
        chunking_service=chunking,
        embedding_service=embedding,
        llm_service=llm,
        procedure_service=procedure_service,
        retrieval_service=retrieval,
        reranking_service=reranker,
        policy_service=PolicyService(
            llm_service=llm,
            retrieval_service=retrieval,
            chunking_service=chunking,
            reranking_service=reranker,
            settings=settings,
        ),
        evidence_service=EvidenceService(
            retrieval_service=retrieval,
            chunking_service=chunking,
            reranking_service=reranker,
            llm_service=llm,
            settings=settings,
        ),
        readiness_service=ReadinessService(),
    )


async def _stub_preprocess(
    service: AnalysisService,
) -> None:
    patient_doc = ParsedDocument(
        id="patient",
        filename="patient.pdf",
        document_type=DocumentType.PATIENT_CHART,
        page_count=1,
        pages=[
            PDFPage(
                page_number=1,
                text="patient text",
                spans=[],
                requires_ocr=False,
            )
        ],
    )
    policy_doc = ParsedDocument(
        id="policy",
        filename="policy.pdf",
        document_type=DocumentType.POLICY,
        page_count=1,
        pages=[
            PDFPage(
                page_number=1,
                text="policy text",
                spans=[],
                requires_ocr=False,
            )
        ],
    )

    async def preprocess(
        path: Path,
        document_type: DocumentType,
    ) -> tuple[ParsedDocument, list[Section]]:
        if document_type == DocumentType.PATIENT_CHART:
            return patient_doc, _patient_sections()
        return policy_doc, _policy_sections()

    service._preprocess_document = preprocess  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_synthetic_ready_flow(tmp_path: Path) -> None:
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["pat-1"],
                evidence_text=["demonstrated saphenous reflux"],
                page_numbers=[2],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["pat-2"],
                evidence_text=["CEAP clinical class documented as C3"],
                page_numbers=[3],
                explanation="Supported.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    services = _build_container(applicability=True, evidence_batch=batch)
    analysis = AnalysisService(services)
    await _stub_preprocess(analysis)

    result = await analysis.analyze(tmp_path / "p.pdf", tmp_path / "pol.pdf")
    assert result.overall_status == ReadinessStatus.READY
    assert result.matching_policy_criteria_found is True
    assert services.embedding_service.init_count == 1
    assert services.reranking_service.init_count == 1
    assert services.llm_service.init_count == 1
    # Mocked procedure → policy criteria + evidence Sol only.
    assert services.llm_service.extract_structured.await_count == 2


@pytest.mark.asyncio
async def test_synthetic_not_ready_flow(tmp_path: Path) -> None:
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["pat-1"],
                evidence_text=["demonstrated saphenous reflux"],
                page_numbers=[2],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MISSING,
                explanation="No CEAP documentation found.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    services = _build_container(applicability=True, evidence_batch=batch)
    analysis = AnalysisService(services)
    await _stub_preprocess(analysis)

    result = await analysis.analyze(tmp_path / "p.pdf", tmp_path / "pol.pdf")
    assert result.overall_status == ReadinessStatus.NOT_READY
    assert [item.criterion_id for item in result.missing_items] == ["c2"]


@pytest.mark.asyncio
async def test_synthetic_needs_review_flow(tmp_path: Path) -> None:
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["pat-1"],
                evidence_text=["demonstrated saphenous reflux"],
                page_numbers=[2],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.UNCLEAR,
                evidence_chunk_ids=["pat-2"],
                evidence_text=["CEAP clinical class documented as C3"],
                page_numbers=[3],
                explanation="Ambiguous CEAP notation.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    services = _build_container(applicability=True, evidence_batch=batch)
    analysis = AnalysisService(services)
    await _stub_preprocess(analysis)

    result = await analysis.analyze(tmp_path / "p.pdf", tmp_path / "pol.pdf")
    assert result.overall_status == ReadinessStatus.NEEDS_REVIEW
    assert [item.criterion_id for item in result.review_items] == ["c2"]


@pytest.mark.asyncio
async def test_non_applicable_uses_exactly_two_openai_calls(tmp_path: Path) -> None:
    services = _build_container(applicability=False, mock_procedure=False)
    analysis = AnalysisService(services)
    await _stub_preprocess(analysis)

    result = await analysis.analyze(tmp_path / "p.pdf", tmp_path / "pol.pdf")

    assert result.overall_status == ReadinessStatus.NOT_APPLICABLE
    assert result.matching_policy_criteria_found is False
    assert result.criteria == []
    assert result.exclusions == []
    assert result.missing_items == []
    assert result.review_items == []
    assert (
        result.next_step
        == (
            "No criteria for the detected planned procedure were found "
            "in the uploaded policy."
        )
    )

    models = [
        call.kwargs["response_model"]
        for call in services.llm_service.extract_structured.await_args_list
    ]
    assert models == [ProcedureExtraction, SolPolicyCriteriaResult]
    assert services.llm_service.extract_structured.await_count == 2


@pytest.mark.asyncio
async def test_applicable_uses_exactly_three_openai_calls(tmp_path: Path) -> None:
    batch = EvidenceEvaluationBatch(
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["pat-1"],
                evidence_text=["demonstrated saphenous reflux"],
                page_numbers=[2],
                explanation="Supported.",
            ),
            CriterionEvaluation(
                criterion_id="c2",
                status=CriterionStatus.MET,
                evidence_chunk_ids=["pat-2"],
                evidence_text=["CEAP clinical class documented as C3"],
                page_numbers=[3],
                explanation="Supported.",
            ),
        ],
        exclusions=[
            ExclusionEvaluation(
                exclusion_id="ex1",
                triggered=False,
                status=CriterionStatus.MISSING,
                explanation="Not triggered.",
            )
        ],
    )
    services = _build_container(
        applicability=True,
        evidence_batch=batch,
        mock_procedure=False,
    )
    analysis = AnalysisService(services)
    await _stub_preprocess(analysis)

    result = await analysis.analyze(tmp_path / "p.pdf", tmp_path / "pol.pdf")

    assert result.overall_status == ReadinessStatus.READY
    models = [
        call.kwargs["response_model"]
        for call in services.llm_service.extract_structured.await_args_list
    ]
    assert models == [
        ProcedureExtraction,
        SolPolicyCriteriaResult,
        SolEvidenceEvaluationBatch,
    ]
    assert services.llm_service.extract_structured.await_count == 3


@pytest.mark.asyncio
async def test_non_applicable_skips_evidence_llm(tmp_path: Path) -> None:
    services = _build_container(applicability=False)
    analysis = AnalysisService(services)
    await _stub_preprocess(analysis)

    result = await analysis.analyze(tmp_path / "p.pdf", tmp_path / "pol.pdf")
    assert result.overall_status == ReadinessStatus.NOT_APPLICABLE
    # Procedure service is mocked; only policy criteria uses the LLM.
    assert services.llm_service.extract_structured.await_count == 1
    assert (
        services.llm_service.extract_structured.await_args.kwargs["response_model"]
        is SolPolicyCriteriaResult
    )


@pytest.mark.asyncio
async def test_missing_openai_key_fails_before_analysis(tmp_path: Path) -> None:
    services = build_services(Settings(OPENAI_API_KEY=None))
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        await AnalysisService(services).analyze(
            tmp_path / "p.pdf",
            tmp_path / "pol.pdf",
        )


@pytest.mark.asyncio
async def test_unreadable_pdf_without_mistral_fails(tmp_path: Path) -> None:
    settings = Settings(OPENAI_API_KEY="test-key", MISTRAL_API_KEY=None)
    services = build_services(settings)
    services.pdf_service = MagicMock()
    services.pdf_service.parse.return_value = ParsedDocument(
        id="ocr-doc",
        filename="scan.pdf",
        document_type=DocumentType.PATIENT_CHART,
        page_count=1,
        pages=[
            PDFPage(
                page_number=1,
                text="",
                spans=[],
                requires_ocr=True,
            )
        ],
    )
    patient = tmp_path / "patient.pdf"
    policy = tmp_path / "policy.pdf"
    patient.write_bytes(b"%PDF-1.4")
    policy.write_bytes(b"%PDF-1.4")

    with pytest.raises(OCRProcessingError, match="MISTRAL_API_KEY"):
        await AnalysisService(services).analyze(patient, policy)


@pytest.mark.skipif(
    not PATIENT_PDF.exists() or not POLICY_PDF.exists(),
    reason="sample PDFs missing",
)
@pytest.mark.asyncio
async def test_sample_documents_not_applicable_with_mocks() -> None:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        BM25_TOP_K=10,
        DENSE_TOP_K=10,
        RRF_TOP_K=10,
        RERANK_TOP_K=5,
        POLICY_BM25_TOP_K=40,
        POLICY_DENSE_TOP_K=15,
        POLICY_RERANK_TOP_K=5,
        SECTION_NEIGHBOR_WINDOW=1,
    )
    services = build_services(settings)
    embedding = _fake_embedding()
    services.embedding_service = embedding
    services.retrieval_service = RetrievalService(embedding, settings)
    services.chunking_service = ChunkingService(settings=settings)
    services.reranking_service._model = MagicMock(
        compute_score=MagicMock(
            side_effect=lambda pairs: list(range(len(pairs), 0, -1))
        )
    )
    services.reranking_service.init_count = 1
    services.llm_service._client = object()
    services.llm_service.init_count = 1

    services.procedure_service.detect_planned_procedure = AsyncMock(
        return_value=ProcedureExtraction(
            procedure_name="Permanent Cardiac Pacemaker Implantation",
            cpt_code="33207",
            status="planned",
            is_future_procedure=True,
            confidence=0.95,
            source_chunk_id="x",
            source_text="Permanent Cardiac Pacemaker Implantation",
            page_number=1,
        )
    )

    async def analyze_policy(procedure, policy_sections):  # type: ignore[no-untyped-def]
        candidates = services.policy_service.retrieve_policy_candidates(
            procedure,
            policy_sections,
        )
        services.llm_service.extract_structured = AsyncMock(
            return_value=SolPolicyCriteriaResult(
                matching_policy_criteria_found=False,
                explanation=(
                    "Policy addresses EMS transport, not cardiac implants."
                ),
                criteria_tree=None,
            )
        )
        return await PolicyService(
            llm_service=services.llm_service,
            retrieval_service=services.retrieval_service,
            chunking_service=services.chunking_service,
            reranking_service=services.reranking_service,
            settings=settings,
        )._analyze_candidates(procedure, candidates)

    services.policy_service.analyze_policy = analyze_policy  # type: ignore[method-assign]
    services.evidence_service = EvidenceService(
        retrieval_service=services.retrieval_service,
        chunking_service=services.chunking_service,
        reranking_service=services.reranking_service,
        llm_service=services.llm_service,
        settings=settings,
    )

    result = await AnalysisService(services).analyze(PATIENT_PDF, POLICY_PDF)
    assert result.overall_status == ReadinessStatus.NOT_APPLICABLE
    assert result.matching_policy_criteria_found is False
    assert services.embedding_service.init_count == 1
    assert services.reranking_service.init_count == 1

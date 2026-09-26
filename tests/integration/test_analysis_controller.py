"""Integration tests for the analysis API controller."""

from __future__ import annotations

import io
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.dependencies import get_analysis_service
from app.core.exceptions import (
    ConfigurationError,
    InvalidPDFError,
    LLMServiceError,
    PolicyAnalysisError,
)
from app.main import create_app
from app.schemas.analysis import AnalysisResponse, ReadinessStatus
from app.schemas.procedure import ProcedureExtraction

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
PATIENT_PDF = SAMPLE_DIR / "patient.pdf"
POLICY_PDF = SAMPLE_DIR / "bcbs_policy.pdf"


def _pdf_bytes(content: bytes = b"%PDF-1.4 sample") -> bytes:
    return content


def _pdf_upload(
    name: str = "patient.pdf",
    content: bytes | None = None,
    content_type: str = "application/pdf",
) -> tuple[str, io.BytesIO, str]:
    payload = content if content is not None else _pdf_bytes()
    return (name, io.BytesIO(payload), content_type)


def _sample_response() -> AnalysisResponse:
    return AnalysisResponse(
        procedure=ProcedureExtraction(
            procedure_name="Permanent Cardiac Pacemaker Implantation",
            cpt_code="33207",
            status="planned",
            is_future_procedure=True,
            confidence=0.95,
        ),
        matching_policy_criteria_found=False,
        overall_status=ReadinessStatus.NOT_APPLICABLE,
        criteria=[],
        exclusions=[],
        missing_items=[],
        review_items=[],
        next_step=(
            "No criteria for the detected planned procedure were found "
            "in the uploaded policy."
        ),
    )


@pytest.fixture
def analysis_mock() -> AsyncMock:
    mock_service = AsyncMock()
    mock_service.analyze = AsyncMock(return_value=_sample_response())
    return mock_service


@pytest.fixture
def client(analysis_mock: AsyncMock) -> TestClient:
    application = create_app(
        Settings(
            OPENAI_API_KEY="test-key",
            MAX_UPLOAD_SIZE_MB=1,
            CORS_ORIGINS="http://localhost:5173",
        )
    )
    application.dependency_overrides[get_analysis_service] = lambda: analysis_mock
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


def test_valid_upload_returns_200(
    client: TestClient,
    analysis_mock: AsyncMock,
) -> None:
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf"),
            "policy_file": _pdf_upload("policy.pdf"),
        },
        headers={"X-Request-ID": "req-123"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_status"] == "NOT_APPLICABLE"
    assert payload["matching_policy_criteria_found"] is False
    assert payload["procedure"]["procedure_name"]
    assert response.headers["X-Request-ID"] == "req-123"
    analysis_mock.analyze.assert_awaited_once()


def test_response_matches_schema(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf"),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    parsed = AnalysisResponse.model_validate(response.json())
    assert parsed.overall_status == ReadinessStatus.NOT_APPLICABLE


def test_missing_patient_file_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        files={"policy_file": _pdf_upload("policy.pdf")},
    )
    assert response.status_code == 422


def test_missing_policy_file_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        files={"patient_file": _pdf_upload("patient.pdf")},
    )
    assert response.status_code == 422


def test_invalid_pdf_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf", content=b"not-a-pdf"),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upload_validation_error"


def test_invalid_extension_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload(
                "patient.txt",
                content_type="text/plain",
            ),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    assert response.status_code == 400


def test_oversized_file_returns_413(client: TestClient) -> None:
    oversized = b"%PDF-1.4\n" + (b"x" * (1024 * 1024 + 10))
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf", content=oversized),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    assert response.status_code == 413
    assert "upload size limit" in response.json()["error"]["message"]


def test_application_error_mapping(
    client: TestClient,
    analysis_mock: AsyncMock,
) -> None:
    analysis_mock.analyze = AsyncMock(
        side_effect=PolicyAnalysisError("criteria failed")
    )
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf"),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "policy_analysis_error"


def test_invalid_pdf_error_maps_to_400(
    client: TestClient,
    analysis_mock: AsyncMock,
) -> None:
    analysis_mock.analyze = AsyncMock(side_effect=InvalidPDFError("bad pdf"))
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf"),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_pdf"


def test_provider_error_returns_502(
    client: TestClient,
    analysis_mock: AsyncMock,
) -> None:
    analysis_mock.analyze = AsyncMock(side_effect=LLMServiceError("provider down"))
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf"),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_error"


def test_configuration_error_returns_503(
    client: TestClient,
    analysis_mock: AsyncMock,
) -> None:
    analysis_mock.analyze = AsyncMock(
        side_effect=ConfigurationError("OPENAI_API_KEY missing")
    )
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf"),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    assert response.status_code == 503


def test_request_id_generated_when_missing(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf"),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")


def test_temporary_files_cleaned_up(
    client: TestClient,
    analysis_mock: AsyncMock,
) -> None:
    captured: dict[str, Path] = {}

    async def fake_analyze(patient_path: Path, policy_path: Path) -> AnalysisResponse:
        captured["patient"] = patient_path
        captured["policy"] = policy_path
        assert patient_path.exists()
        assert policy_path.exists()
        return _sample_response()

    analysis_mock.analyze = AsyncMock(side_effect=fake_analyze)
    response = client.post(
        "/api/analyze",
        files={
            "patient_file": _pdf_upload("patient.pdf"),
            "policy_file": _pdf_upload("policy.pdf"),
        },
    )
    assert response.status_code == 200
    assert not captured["patient"].exists()
    assert not captured["policy"].exists()


def test_health_and_ready_endpoints() -> None:
    app = create_app(Settings(OPENAI_API_KEY="test-key"))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"


def test_ready_returns_503_without_openai_key() -> None:
    app = create_app(Settings(OPENAI_API_KEY=None))
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert "OPENAI_API_KEY" in ready.json()["missing"]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_INTEGRATION"),
    reason="Set RUN_LIVE_INTEGRATION=1 to run live OpenAI integration",
)
@pytest.mark.skipif(
    not PATIENT_PDF.exists() or not POLICY_PDF.exists(),
    reason="sample PDFs missing",
)
def test_optional_full_api_sample_documents_not_applicable() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for live API integration")
    app = create_app()
    with (
        TestClient(app) as client,
        PATIENT_PDF.open("rb") as patient_handle,
        POLICY_PDF.open("rb") as policy_handle,
    ):
        response = client.post(
            "/api/analyze",
            files={
                "patient_file": (
                    "patient.pdf",
                    patient_handle,
                    "application/pdf",
                ),
                "policy_file": (
                    "policy.pdf",
                    policy_handle,
                    "application/pdf",
                ),
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["matching_policy_criteria_found"] is False
    assert payload["overall_status"] == "NOT_APPLICABLE"
    assert payload["procedure"]["procedure_name"]

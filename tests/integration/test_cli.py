"""CLI integration tests for exit codes and output formats."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.cli import (
    EXIT_CONFIGURATION,
    EXIT_DOCUMENT,
    EXIT_SUCCESS,
    EXIT_USAGE,
    main,
    print_human_readable,
    print_json,
)
from app.core.exceptions import ConfigurationError, OCRProcessingError
from app.schemas.analysis import (
    AnalysisResponse,
    MissingItem,
    ReadinessStatus,
    ReviewItem,
)
from app.schemas.evaluation import CriterionEvaluation, CriterionStatus
from app.schemas.procedure import ProcedureExtraction

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
PATIENT_PDF = SAMPLE_DIR / "patient.pdf"
POLICY_PDF = SAMPLE_DIR / "bcbs_policy.pdf"


def _sample_result(
    status: ReadinessStatus = ReadinessStatus.NOT_APPLICABLE,
) -> AnalysisResponse:
    return AnalysisResponse(
        procedure=ProcedureExtraction(
            procedure_name="Permanent Cardiac Pacemaker Implantation",
            cpt_code="33207",
            status="planned",
            is_future_procedure=True,
            confidence=0.95,
        ),
        matching_policy_criteria_found=status != ReadinessStatus.NOT_APPLICABLE,
        overall_status=status,
        criteria=[
            CriterionEvaluation(
                criterion_id="c1",
                status=CriterionStatus.MISSING,
                description="Compression therapy failed for three months.",
                explanation="No qualifying documentation found.",
            )
        ]
        if status != ReadinessStatus.NOT_APPLICABLE
        else [],
        exclusions=[],
        missing_items=[
            MissingItem(
                criterion_id="c1",
                description=(
                    "Compression therapy failed for three months "
                    "is not documented."
                ),
            )
        ]
        if status == ReadinessStatus.NOT_READY
        else [],
        review_items=[
            ReviewItem(
                criterion_id="c2",
                description="Ambiguous note",
                explanation="Unclear duration.",
            )
        ]
        if status == ReadinessStatus.NEEDS_REVIEW
        else [],
        next_step=(
            "No criteria for the detected planned procedure were found "
            "in the uploaded policy."
            if status == ReadinessStatus.NOT_APPLICABLE
            else "Obtain documentation for the missing requirements before submission."
        ),
    )


def test_missing_args_exit_usage() -> None:
    assert main([]) == EXIT_USAGE


def test_json_output_is_valid(capsys: pytest.CaptureFixture[str]) -> None:
    result = _sample_result(ReadinessStatus.NOT_READY)
    print_json(result)
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "NOT_READY"
    assert payload["procedure"]["procedure_name"]
    assert "missing_items" in payload


def test_human_cli_output_renders(capsys: pytest.CaptureFixture[str]) -> None:
    print_human_readable(_sample_result(ReadinessStatus.NOT_READY))
    out = capsys.readouterr().out
    assert "Procedure:" in out
    assert "Overall readiness: NOT_READY" in out
    assert "Missing items:" in out
    assert "Next step:" in out


def test_cli_success_exit_code(tmp_path: Path) -> None:
    patient = tmp_path / "patient.pdf"
    policy = tmp_path / "policy.pdf"
    patient.write_bytes(b"%PDF")
    policy.write_bytes(b"%PDF")
    with patch(
        "app.cli.run_analysis",
        new=AsyncMock(return_value=_sample_result()),
    ):
        code = main(["--patient", str(patient), "--policy", str(policy)])
    assert code == EXIT_SUCCESS


def test_cli_configuration_exit_code(tmp_path: Path) -> None:
    patient = tmp_path / "patient.pdf"
    policy = tmp_path / "policy.pdf"
    patient.write_bytes(b"%PDF")
    policy.write_bytes(b"%PDF")
    with patch(
        "app.cli.run_analysis",
        new=AsyncMock(side_effect=ConfigurationError("OPENAI_API_KEY missing")),
    ):
        code = main(["--patient", str(patient), "--policy", str(policy)])
    assert code == EXIT_CONFIGURATION


def test_cli_document_exit_code(tmp_path: Path) -> None:
    patient = tmp_path / "patient.pdf"
    policy = tmp_path / "policy.pdf"
    patient.write_bytes(b"%PDF")
    policy.write_bytes(b"%PDF")
    with patch(
        "app.cli.run_analysis",
        new=AsyncMock(side_effect=OCRProcessingError("OCR required")),
    ):
        code = main(["--patient", str(patient), "--policy", str(policy)])
    assert code == EXIT_DOCUMENT


def test_cli_json_flag_prints_only_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patient = tmp_path / "patient.pdf"
    policy = tmp_path / "policy.pdf"
    patient.write_bytes(b"%PDF")
    policy.write_bytes(b"%PDF")
    with patch(
        "app.cli.run_analysis",
        new=AsyncMock(return_value=_sample_result()),
    ):
        code = main(
            ["--patient", str(patient), "--policy", str(policy), "--json"]
        )
    assert code == EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "NOT_APPLICABLE"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_INTEGRATION"),
    reason="Set RUN_LIVE_INTEGRATION=1 to run live OpenAI integration",
)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY")
    or not PATIENT_PDF.exists()
    or not POLICY_PDF.exists(),
    reason="OPENAI_API_KEY and sample PDFs required for live E2E",
)
def test_live_sample_documents_not_applicable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "--patient",
            str(PATIENT_PDF),
            "--policy",
            str(POLICY_PDF),
            "--json",
        ]
    )
    assert code == EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "NOT_APPLICABLE"
    assert payload["matching_policy_criteria_found"] is False

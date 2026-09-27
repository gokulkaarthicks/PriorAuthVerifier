"""Prior Authorization analysis HTTP endpoints."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.core.config import Settings, get_settings
from app.core.dependencies import get_analysis_service
from app.core.exceptions import UploadValidationError
from app.core.logging import get_logger
from app.schemas.analysis import AnalysisResponse
from app.services.analysis_service import AnalysisService

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "application/vnd.pdf",
    "text/pdf",
    "application/octet-stream",
}

PatientUpload = Annotated[
    UploadFile,
    File(description="Patient chart PDF"),
]
PolicyUpload = Annotated[
    UploadFile,
    File(description="Policy PDF"),
]
AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]


def _request_settings(request: Request) -> Settings:
    stored = getattr(request.app.state, "settings", None)
    if isinstance(stored, Settings):
        return stored
    return get_settings()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze Prior Authorization readiness",
    description=(
        "Analyzes a patient chart against an uploaded medical-necessity policy."
    ),
)
async def analyze(
    request: Request,
    patient_file: PatientUpload,
    policy_file: PolicyUpload,
    analysis_service: AnalysisServiceDep,
) -> AnalysisResponse:
    """Accept patient and policy PDFs and return readiness analysis."""
    settings = _request_settings(request)
    with tempfile.TemporaryDirectory(prefix="priorauth_") as temp_dir:
        temp_root = Path(temp_dir)
        patient_path = await _persist_upload(
            upload=patient_file,
            destination=temp_root / "patient.pdf",
            field_name="patient_file",
            settings=settings,
        )
        policy_path = await _persist_upload(
            upload=policy_file,
            destination=temp_root / "policy.pdf",
            field_name="policy_file",
            settings=settings,
        )
        logger.info(
            "analyze_request_accepted",
            patient_filename=patient_file.filename,
            policy_filename=policy_file.filename,
        )
        # Share the execution loop/clients with jobs while preserving this response API.
        from app.services.analysis_job_manager import AnalysisJobManager

        manager: AnalysisJobManager = request.app.state.job_manager
        return await manager.execute_legacy(analysis_service, patient_path, policy_path)


async def _persist_upload(
    *,
    upload: UploadFile,
    destination: Path,
    field_name: str,
    settings: Settings,
) -> Path:
    _validate_upload_metadata(upload, field_name=field_name)
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    size = await _write_upload_streaming(
        upload=upload,
        destination=destination,
        max_bytes=max_bytes,
        field_name=field_name,
    )
    if size == 0:
        raise UploadValidationError(
            f"{field_name} is empty",
            status_code=400,
        )
    # Early %PDF check; PDFService re-validates during parse.
    header = await asyncio.to_thread(_read_prefix, destination, 1024)
    if b"%PDF" not in header:
        raise UploadValidationError(
            f"{field_name} is not a valid PDF",
            status_code=400,
        )
    return destination


def _validate_upload_metadata(upload: UploadFile, *, field_name: str) -> None:
    filename = (upload.filename or "").strip()
    if not filename:
        raise UploadValidationError(
            f"{field_name} filename is required",
            status_code=400,
        )
    if not filename.lower().endswith(".pdf"):
        raise UploadValidationError(
            f"{field_name} must be a PDF file",
            status_code=400,
        )
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        raise UploadValidationError(
            f"{field_name} has unsupported content type: {content_type}",
            status_code=400,
        )


async def _write_upload_streaming(
    *,
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
    field_name: str,
) -> int:
    total = 0
    chunk_size = 1024 * 1024
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise UploadValidationError(
                        f"{field_name} exceeds upload size limit of "
                        f"{max_bytes // (1024 * 1024)} MB",
                        status_code=413,
                    )
                handle.write(chunk)
    finally:
        await upload.close()
    return total


def _read_prefix(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)

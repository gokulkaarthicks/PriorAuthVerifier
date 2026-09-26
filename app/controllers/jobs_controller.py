"""Create and inspect in-memory prior-authorization jobs."""

from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.controllers.analysis_controller import (
    AnalysisServiceDep,
    PatientUpload,
    PolicyUpload,
    _persist_upload,
    _request_settings,
)
from app.schemas.jobs import JobDetail, JobList, JobSummary
from app.services.analysis_job_manager import AnalysisJobManager, JobCapacityError

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def get_job_manager(request: Request) -> AnalysisJobManager:
    manager: AnalysisJobManager = request.app.state.job_manager
    return manager


ManagerDep = Annotated[AnalysisJobManager, Depends(get_job_manager)]


def _filename(name: str | None) -> str:
    # Only retain a display basename; client-supplied paths never become storage paths.
    return PurePosixPath((name or "document.pdf").replace("\\", "/")).name


@router.post("", response_model=JobSummary, status_code=202)
async def create_job(
    request: Request,
    patient_file: PatientUpload,
    policy_file: PolicyUpload,
    analysis_service: AnalysisServiceDep,
    manager: ManagerDep,
) -> JobSummary:
    try:
        job_id = manager.reserve(
            _filename(patient_file.filename), _filename(policy_file.filename)
        )
    except JobCapacityError as exc:
        raise HTTPException(429, "Wait for an active case to finish.") from exc
    try:
        patient, policy = manager.upload_paths(job_id)
        settings = _request_settings(request)
        await _persist_upload(
            upload=patient_file, destination=patient,
            field_name="patient_file", settings=settings,
        )
        await _persist_upload(
            upload=policy_file, destination=policy,
            field_name="policy_file", settings=settings,
        )
        return manager.enqueue(job_id, analysis_service)
    except BaseException:
        manager.discard_upload(job_id)
        raise
    finally:
        await patient_file.close()
        await policy_file.close()


@router.get("", response_model=JobList)
async def list_jobs(manager: ManagerDep) -> JobList:
    return manager.list_jobs()


@router.get("/{job_id}", response_model=JobDetail)
async def get_job(job_id: str, manager: ManagerDep) -> JobDetail:
    try:
        return manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Case not found in this server session.") from exc

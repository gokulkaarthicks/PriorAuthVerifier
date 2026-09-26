"""Session job metadata; completed details reuse the canonical analysis schema."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.schemas.analysis import AnalysisResponse, ReadinessStatus


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    patient_filename: str
    policy_filename: str
    procedure_name: str | None = None
    cpt_code: str | None = None
    overall_status: ReadinessStatus | None = None
    error_message: str | None = None


class JobDetail(JobSummary):
    result: AnalysisResponse | None = None


class JobList(BaseModel):
    jobs: list[JobSummary]
    active_count: int
    max_active_jobs: int
    execution_concurrency: int

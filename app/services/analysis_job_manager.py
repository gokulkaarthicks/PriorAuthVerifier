"""One in-memory job queue per FastAPI lifespan; no analysis decisions here."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.schemas.analysis import AnalysisResponse
from app.schemas.jobs import JobDetail, JobList, JobStatus, JobSummary
from app.services.analysis_runner import AnalysisRunner
from app.services.analysis_service import AnalysisService

SAFE_JOB_ERROR = "Analysis could not be completed."


class JobCapacityError(Exception):
    """No active slot remains for an incoming upload."""


@dataclass
class _Job:
    summary: JobSummary
    directory: Path | None
    result: AnalysisResponse | None = None
    service: AnalysisService | None = None


class AnalysisJobManager:
    """All metadata mutations occur on the HTTP loop, never in execution threads."""

    def __init__(self, settings: Settings) -> None:
        self.max_active = settings.MAX_ACTIVE_ANALYSIS_JOBS
        self.concurrency = settings.ANALYSIS_JOB_CONCURRENCY
        self._jobs: dict[str, _Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._root = tempfile.TemporaryDirectory(prefix="priorauth_jobs_")
        self._runner = AnalysisRunner(self.concurrency)
        self._closed = False
        self._workers = [
            asyncio.create_task(self._worker(), name=f"analysis-job-worker-{index}")
            for index in range(self.concurrency)
        ]

    @property
    def active_count(self) -> int:
        return sum(
            job.summary.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            for job in self._jobs.values()
        )

    def reserve(self, patient_filename: str, policy_filename: str) -> str:
        # No await between capacity check and reservation: atomic on the owner loop.
        # Reserving before streaming protects capacity from simultaneous uploads.
        if self._closed or self.active_count >= self.max_active:
            raise JobCapacityError
        job_id = str(uuid.uuid4())
        directory = Path(self._root.name) / job_id
        directory.mkdir(mode=0o700)
        self._jobs[job_id] = _Job(
            summary=JobSummary(
                job_id=job_id,
                created_at=datetime.now(UTC),
                patient_filename=patient_filename,
                policy_filename=policy_filename,
            ),
            directory=directory,
        )
        return job_id

    def upload_paths(self, job_id: str) -> tuple[Path, Path]:
        directory = self._jobs[job_id].directory
        assert directory is not None
        return directory / "patient.pdf", directory / "policy.pdf"

    def enqueue(self, job_id: str, service: AnalysisService) -> JobSummary:
        job = self._jobs[job_id]
        job.service = service
        self._queue.put_nowait(job_id)
        return job.summary.model_copy()

    def discard_upload(self, job_id: str) -> None:
        job = self._jobs.pop(job_id)
        self._clean_documents(job)

    def list_jobs(self) -> JobList:
        return JobList(
            jobs=[job.summary for job in reversed(self._jobs.values())],
            active_count=self.active_count,
            max_active_jobs=self.max_active,
            execution_concurrency=self.concurrency,
        )

    def get_job(self, job_id: str) -> JobDetail:
        job = self._jobs[job_id]
        return JobDetail(**job.summary.model_dump(), result=job.result)

    async def execute_legacy(
        self, service: AnalysisService, patient: Path, policy: Path
    ) -> AnalysisResponse:
        return await self._runner.execute(service, str(patient), str(policy))

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs[job_id]
            try:
                assert job.service is not None
                patient, policy = self.upload_paths(job_id)
                job.summary.status = JobStatus.RUNNING
                job.summary.started_at = datetime.now(UTC)
                result = await self._runner.execute(
                    job.service, str(patient), str(policy)
                )
                job.result = result  # Keep exactly one canonical final response.
                job.summary.procedure_name = result.procedure.procedure_name
                job.summary.cpt_code = result.procedure.cpt_code
                job.summary.overall_status = result.overall_status
                job.summary.status = JobStatus.COMPLETED
            except asyncio.CancelledError:
                job.summary.status = JobStatus.FAILED
                job.summary.error_message = SAFE_JOB_ERROR
                raise
            except Exception:
                # Deliberately do not retain exceptions, tracebacks, or provider text.
                job.summary.status = JobStatus.FAILED
                job.summary.error_message = SAFE_JOB_ERROR
            finally:
                job.summary.completed_at = datetime.now(UTC)
                self._clean_documents(job)
                job.service = None
                self._queue.task_done()

    @staticmethod
    def _clean_documents(job: _Job) -> None:
        if job.directory is not None:
            shutil.rmtree(job.directory, ignore_errors=True)
            job.directory = None

    async def close(self) -> None:
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        # Remove queued, running, and partially uploaded PDFs before stopping CPU work.
        for job in self._jobs.values():
            self._clean_documents(job)
        self._root.cleanup()
        await self._runner.close()

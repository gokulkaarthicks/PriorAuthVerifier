"""Orchestration tests with mocked analysis; no models or provider requests."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.schemas.analysis import AnalysisResponse, ReadinessStatus
from app.schemas.jobs import JobStatus
from app.schemas.procedure import ProcedureExtraction
from app.services.analysis_job_manager import (
    SAFE_JOB_ERROR,
    AnalysisJobManager,
    JobCapacityError,
)
from app.services.analysis_runner import AnalysisRunner
from app.services.analysis_service import AnalysisService


def response() -> AnalysisResponse:
    return AnalysisResponse(
        procedure=ProcedureExtraction(procedure_name="Pacemaker", cpt_code="33207"),
        matching_policy_criteria_found=False,
        overall_status=ReadinessStatus.NOT_APPLICABLE,
        next_step="No matching criteria.",
    )


async def until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(4):
        while not predicate():
            await asyncio.sleep(0.01)


def submit(manager: AnalysisJobManager, service: AnalysisService) -> str:
    job_id = manager.reserve("patient.pdf", "policy.pdf")
    for path in manager.upload_paths(job_id):
        path.write_bytes(b"%PDF-1.4 test")
    manager.enqueue(job_id, service)
    return job_id


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2, 3])
async def test_capacity_concurrency_lifecycle_and_history(concurrency: int) -> None:
    manager = AnalysisJobManager(Settings(ANALYSIS_JOB_CONCURRENCY=concurrency))
    release = threading.Event()
    service = MagicMock(spec=AnalysisService)
    running = 0
    peak = 0
    started: list[Path] = []
    canonical = response()

    async def analyze(patient: Path, policy: Path) -> AnalysisResponse:
        nonlocal running, peak
        assert patient.exists() and policy.exists()
        started.append(patient)
        running += 1
        peak = max(peak, running)
        while not release.is_set():
            await asyncio.sleep(0.01)
        running -= 1
        return canonical

    service.analyze = AsyncMock(side_effect=analyze)
    try:
        ids = [submit(manager, service) for _ in range(20)]
        paths = [manager.upload_paths(job_id) for job_id in ids]
        await until(lambda: len(started) == concurrency)
        assert manager.active_count == 20
        assert peak == concurrency
        assert sum(j.status == JobStatus.RUNNING for j in manager.list_jobs().jobs) == (
            concurrency
        )
        assert manager.get_job(ids[-1]).status == JobStatus.QUEUED
        with pytest.raises(JobCapacityError):
            submit(manager, service)
        release.set()
        await until(lambda: manager.active_count == 0)
        assert service.analyze.await_count == 20
        assert peak <= concurrency
        assert [j.job_id for j in manager.list_jobs().jobs] == list(reversed(ids))
        assert manager.get_job(ids[0]).result is canonical
        assert manager.get_job(ids[0]).overall_status == ReadinessStatus.NOT_APPLICABLE
        assert manager.get_job(ids[0]).started_at is not None
        assert manager.get_job(ids[0]).completed_at is not None
        assert not any(path.exists() for pair in paths for path in pair)
        # Metadata polling never includes the large stored evidence response.
        assert "result" not in manager.list_jobs().jobs[0].model_dump()
        submit(manager, service)
        assert len(manager.list_jobs().jobs) == 21
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
async def test_failure_is_safe_retained_and_frees_capacity() -> None:
    manager = AnalysisJobManager(Settings(MAX_ACTIVE_ANALYSIS_JOBS=1))
    service = MagicMock(spec=AnalysisService)
    service.analyze = AsyncMock(side_effect=RuntimeError("SECRET /private/provider.py"))
    try:
        job_id = submit(manager, service)
        paths = manager.upload_paths(job_id)
        await until(lambda: manager.active_count == 0)
        job = manager.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert job.error_message == SAFE_JOB_ERROR
        assert "SECRET" not in job.model_dump_json()
        assert job.result is None
        assert not any(path.exists() for path in paths)
        assert len(manager.list_jobs().jobs) == 1
        submit(manager, service)
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_shutdown_removes_running_queued_and_incomplete_uploads() -> None:
    manager = AnalysisJobManager(Settings(ANALYSIS_JOB_CONCURRENCY=1))
    service = MagicMock(spec=AnalysisService)
    service.analyze = AsyncMock(side_effect=lambda *_: None)

    async def pending(*_: Path) -> AnalysisResponse:
        await asyncio.sleep(60)
        return response()

    service.analyze = AsyncMock(side_effect=pending)
    ids = [submit(manager, service) for _ in range(3)]
    ids.append(manager.reserve("partial.pdf", "policy.pdf"))
    paths = [manager.upload_paths(job_id)[0].parent for job_id in ids]
    await until(lambda: manager.get_job(ids[0]).status == JobStatus.RUNNING)
    await manager.close()
    assert all(not path.exists() for path in paths)
    with pytest.raises(JobCapacityError):
        manager.reserve("new.pdf", "policy.pdf")


@pytest.mark.asyncio
async def test_shared_execution_loop_keeps_http_responsive() -> None:
    runner = AnalysisRunner(1)
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []
    service = MagicMock(spec=AnalysisService)

    async def cpu_work(*_: Path) -> AnalysisResponse:
        calls.append(threading.get_ident())
        entered.set()
        release.wait(timeout=3)  # Simulate a synchronous CPU stage.
        return response()

    service.analyze = AsyncMock(side_effect=cpu_work)
    try:
        task = asyncio.create_task(runner.execute(service, "patient", "policy"))
        await until(entered.is_set)
        # This HTTP-loop coroutine remains schedulable during the blocking stage.
        assert not task.done()
        assert calls[0] != threading.get_ident()
        release.set()
        await task
        await runner.execute(service, "patient", "policy")
        assert calls[0] == calls[1]
    finally:
        release.set()
        await runner.close()


def test_shutdown_does_not_leave_non_daemon_executor_work(tmp_path: Path) -> None:
    marker = tmp_path / "directory"
    script = '''
import asyncio, threading
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from app.core.config import Settings
from app.services.analysis_job_manager import AnalysisJobManager
from app.services.analysis_service import AnalysisService
async def main():
    manager = AnalysisJobManager(Settings())
    service = MagicMock(spec=AnalysisService)
    entered = threading.Event()
    def blocked():
        entered.set()
        threading.Event().wait(60)
    async def analyze(*args):
        await asyncio.to_thread(blocked)
    service.analyze = AsyncMock(side_effect=analyze)
    job = manager.reserve("p.pdf", "policy.pdf")
    directory = manager.upload_paths(job)[0].parent
    Path(MARKER).write_text(str(directory))
    manager.enqueue(job, service)
    while not entered.is_set():
        await asyncio.sleep(.01)
    await manager.close()
asyncio.run(main())
'''
    completed = subprocess.run(
        [sys.executable, "-c", f"MARKER = {str(marker)!r}\n" + script],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not Path(marker.read_text()).exists()

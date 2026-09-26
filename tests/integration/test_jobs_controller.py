"""Job API upload validation, immediate admission and server-session history."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.dependencies import get_analysis_service
from app.main import create_app
from app.services.analysis_service import AnalysisService
from tests.unit.test_analysis_jobs import response


def files() -> dict[str, tuple[str, bytes, str]]:
    return {
        "patient_file": ("patient.pdf", b"%PDF-1.4 patient", "application/pdf"),
        "policy_file": ("policy.pdf", b"%PDF-1.4 policy", "application/pdf"),
    }


@pytest.fixture
def harness() -> Iterator[tuple[TestClient, threading.Event, list[Path]]]:
    application = create_app(Settings(MAX_ACTIVE_ANALYSIS_JOBS=2,
                                      ANALYSIS_JOB_CONCURRENCY=1))
    release = threading.Event()
    paths: list[Path] = []
    service = MagicMock(spec=AnalysisService)

    async def analyze(patient: Path, policy: Path) -> object:
        paths.extend([patient, policy])
        while not release.is_set():
            await asyncio.sleep(0.01)
        return response()

    service.analyze = AsyncMock(side_effect=analyze)
    application.dependency_overrides[get_analysis_service] = lambda: service
    with TestClient(application) as client:
        try:
            yield client, release, paths
        finally:
            release.set()


def wait_completed(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        data: dict[str, object] = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] == "COMPLETED":
            return data
        time.sleep(0.02)
    pytest.fail("Job did not complete")


def test_immediate_admission_queue_history_detail_and_capacity(
    harness: tuple[TestClient, threading.Event, list[Path]],
) -> None:
    client, release, paths = harness
    # A hanging analysis must not delay the upload response.
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(client.post, "/api/jobs", files=files())
        try:
            first = future.result(timeout=1)
        except BaseException:
            release.set()
            raise
    assert first.status_code == 202
    first_id = first.json()["job_id"]
    second = client.post("/api/jobs", files=files())
    assert second.status_code == 202
    second_id = second.json()["job_id"]
    listing = client.get("/api/jobs").json()
    assert listing["active_count"] == listing["max_active_jobs"] == 2
    assert listing["execution_concurrency"] == 1
    assert [job["job_id"] for job in listing["jobs"]] == [second_id, first_id]
    assert listing["jobs"][0]["status"] == "QUEUED"
    assert "result" not in listing["jobs"][0]
    assert "priorauth_jobs_" not in str(listing)
    assert client.post("/api/jobs", files=files()).status_code == 429
    release.set()
    detail = wait_completed(client, first_id)
    wait_completed(client, second_id)
    assert detail["result"] == response().model_dump(mode="json")
    assert not any(path.exists() for path in paths)
    assert client.get("/api/jobs").json()["active_count"] == 0
    assert client.get(f"/api/jobs/{first_id}").json() == detail
    assert client.post("/api/jobs", files=files()).status_code == 202
    assert client.get("/api/jobs/unknown").status_code == 404


@pytest.mark.parametrize("kind", ["invalid", "empty", "missing", "oversized"])
def test_invalid_upload_frees_reservation_and_removes_pdfs(
    harness: tuple[TestClient, threading.Event, list[Path]], kind: str,
) -> None:
    client, _, _ = harness
    payload = files()
    if kind == "missing":
        del payload["policy_file"]
    elif kind == "oversized":
        client.app.state.settings.MAX_UPLOAD_SIZE_MB = 1  # type: ignore[union-attr]
        payload["policy_file"] = ("policy.pdf", b"%PDF" + b"x" * 1024**2,
                                  "application/pdf")
    else:
        payload["policy_file"] = ("policy.pdf", b"" if kind == "empty" else b"bad",
                                  "application/pdf")
    assert client.post("/api/jobs", files=payload).status_code in (400, 413, 422)
    assert client.get("/api/jobs").json()["jobs"] == []
    manager = client.app.state.job_manager  # type: ignore[union-attr]
    assert list(Path(manager._root.name).iterdir()) == []


def test_restart_clears_session_history() -> None:
    application = create_app(Settings())
    service = MagicMock(spec=AnalysisService)
    service.analyze = AsyncMock(return_value=response())
    application.dependency_overrides[get_analysis_service] = lambda: service
    with TestClient(application) as client:
        created = client.post("/api/jobs", files=files())
        wait_completed(client, created.json()["job_id"])
        assert len(client.get("/api/jobs").json()["jobs"]) == 1
    with TestClient(application) as client:
        assert client.get("/api/jobs").json()["jobs"] == []

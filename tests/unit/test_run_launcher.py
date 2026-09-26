"""Exercise launcher shutdown with stand-in services and real process groups."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("shutdown", ["interrupt", "service_failure"])
def test_launcher_cleans_service_descendants(tmp_path: Path, shutdown: str) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    service = fake_bin / "service"
    service.write_text(
        "#!/bin/bash\n"
        'sleep 120 &\nchild=$!\n'
        'echo "$child" >> "$LAUNCHER_TEST_PIDS"\n'
        'trap \'kill "$child" 2>/dev/null; wait "$child"; exit 0\' TERM\n'
        'if [ "$1" = backend ] && [ "$LAUNCHER_TEST_MODE" = service_failure ]; '
        'then sleep 1; kill "$child"; wait "$child"; exit 7; fi\n'
        'wait "$child"\n'
    )
    uv = fake_bin / "uv"
    uv.write_text(
        '#!/bin/bash\nif [ "${2:-}" = uvicorn ]; then '
        'exec service backend; fi\nexit 0\n'
    )
    npm = fake_bin / "npm"
    npm.write_text(
        '#!/bin/bash\nif [ "${3:-}" = run ]; then '
        'exec service frontend; fi\nexit 0\n'
    )
    for executable in (service, uv, npm):
        executable.chmod(0o755)
    pid_file = tmp_path / "pids"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LAUNCHER_TEST_PIDS": str(pid_file),
        "LAUNCHER_TEST_MODE": shutdown,
    }
    process = subprocess.Popen(
        ["bash", str(ROOT / "scripts/run.sh")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    children: list[int] = []
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if pid_file.exists():
                children = [int(pid) for pid in pid_file.read_text().splitlines()]
            if len(children) == 2:
                break
            time.sleep(0.05)
        assert len(children) == 2, "Both services must start"
        if shutdown == "interrupt":
            process.send_signal(signal.SIGINT)
        assert process.wait(timeout=12) == (130 if shutdown == "interrupt" else 7)
        for pid in children:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for pid in children:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)

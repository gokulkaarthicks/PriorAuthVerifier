"""Run the unchanged async pipeline away from the HTTP event loop.

One execution loop owns the shared async clients and model instances. Its
synchronous CPU stages stay serialized on that loop; LLM waits can overlap.
Daemon workers also handle the pipeline's existing asyncio.to_thread calls,
so an uninterruptible native operation cannot keep the server alive at exit.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Queue
from typing import ParamSpec, TypeVar

from app.schemas.analysis import AnalysisResponse
from app.services.analysis_service import AnalysisService

P = ParamSpec("P")
T = TypeVar("T")


class _DaemonExecutor(ThreadPoolExecutor):
    """Bounded daemon pool for to_thread, with no interpreter-exit join hook.

    asyncio requires a ThreadPoolExecutor for set_default_executor. Override
    its public scheduling/shutdown API rather than modifying private internals.
    No work is submitted to the superclass's non-daemon pool.
    """

    def __init__(self, workers: int) -> None:
        super().__init__(max_workers=workers)
        self._pending: Queue[tuple[Callable[[], None], Callable[[], bool]] | None] = (
            Queue()
        )
        self._guard = threading.Lock()
        self._stopping = False
        self._workers = [
            threading.Thread(target=self._work, name=f"analysis-io-{i}", daemon=True)
            for i in range(workers)
        ]
        for worker in self._workers:
            worker.start()

    def submit(
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future: Future[T] = Future()

        def execute() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)

        with self._guard:
            if self._stopping:
                raise RuntimeError("Analysis executor is shutting down")
            self._pending.put((execute, future.cancel))
        return future

    def _work(self) -> None:
        while (item := self._pending.get()) is not None:
            item[0]()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._guard:
            if not self._stopping:
                self._stopping = True
                if cancel_futures:
                    while True:
                        try:
                            item = self._pending.get_nowait()
                        except Empty:
                            break
                        if item is not None:
                            item[1]()
                for _ in self._workers:
                    self._pending.put(None)
        if wait:
            for worker in self._workers:
                worker.join()


class AnalysisRunner:
    """Single process-local execution loop, shared by jobs and legacy requests."""

    def __init__(self, concurrency: int) -> None:
        self._loop = asyncio.new_event_loop()
        self._executor = _DaemonExecutor(max(2, concurrency * 2))
        self._limit = asyncio.Semaphore(concurrency)
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_loop, name="analysis-loop", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.set_default_executor(self._executor)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def execute(
        self, service: AnalysisService, patient: str, policy: str
    ) -> AnalysisResponse:
        from pathlib import Path

        if self._closed:
            raise RuntimeError("Analysis runner is closed")

        async def invoke() -> AnalysisResponse:
            async with self._limit:
                return await service.analyze(Path(patient), Path(policy))

        # Do not inherit another request's logging/metrics context into a job.
        future = contextvars.Context().run(
            asyncio.run_coroutine_threadsafe, invoke(), self._loop
        )
        return await asyncio.wrap_future(future)

    async def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

        async def stop() -> None:
            current = asyncio.current_task()
            tasks = [task for task in asyncio.all_tasks() if task is not current]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._loop.stop()

        # CPU-native work cannot be forcibly interrupted safely in Python. Cleanup
        # of files happens in the manager first; remaining execution is daemonized.
        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(stop()))
        deadline = asyncio.get_running_loop().time() + 0.5
        while self._thread.is_alive() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)

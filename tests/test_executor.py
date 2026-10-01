"""进程内后台执行器测试（SPEC 3.1 / 15 并发限制）。"""

from __future__ import annotations

import threading
import time

import pytest

from app.errors import ResourceLimitError
from app.services.executor import LocalTaskExecutor


def test_executor_runs_submitted_task() -> None:
    executor = LocalTaskExecutor(max_concurrent=1)
    finished = threading.Event()
    observed: list[str] = []

    def work(value: str) -> None:
        observed.append(value)
        finished.set()

    executor.submit("task-1", work, "hello")
    assert finished.wait(5.0)
    assert observed == ["hello"]

    # 结束后并发位应被释放
    deadline = time.time() + 5.0
    while executor.running_count != 0 and time.time() < deadline:
        time.sleep(0.02)
    assert executor.has_capacity()
    executor.shutdown()


def test_executor_refuses_when_concurrency_limit_reached() -> None:
    executor = LocalTaskExecutor(max_concurrent=1)
    release = threading.Event()

    executor.submit("task-1", release.wait, 10)
    deadline = time.time() + 5.0
    while not executor.running_ids() and time.time() < deadline:
        time.sleep(0.02)

    assert not executor.has_capacity()
    with pytest.raises(ResourceLimitError) as excinfo:
        executor.submit("task-2", lambda: None)
    assert excinfo.value.http_status == 429
    assert excinfo.value.detail["max_concurrent_tasks"] == 1

    release.set()
    executor.shutdown(wait=True)


def test_executor_counts_slots_after_submit() -> None:
    executor = LocalTaskExecutor(max_concurrent=2)
    assert executor.max_concurrent == 2
    assert executor.has_capacity()
    executor.shutdown()

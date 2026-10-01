"""进程内后台任务执行器（阶段 1，替代 Redis 之前的过渡路径）。

为什么需要它：SPEC 3.1 规定「Web API 不直接执行长时间量化、编译和 Docker 构建任务」。
阶段 1 尚未接入 Redis，而 Celery 在 eager 模式下会**同步**执行任务，
入队这个 HTTP 请求就要一直等到量化和编译结束 —— 既违反 SPEC，前端也会超时。

本模块提供与 Celery 语义一致的本地替代：
- `submit()` 立即返回，任务在后台线程中执行；
- 并发数由 `max_concurrent_tasks` 限制，超限直接以 RESOURCE_LIMIT_EXCEEDED 拒绝（不排队堆积）；
- 业务代码只调用 `run_pipeline`（见 app/pipeline/runner.py），
  接入 Redis 后把 `executor_mode` 改成 `celery` 即可，业务层不变。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.config.settings import get_settings
from app.errors import ResourceLimitError

logger = logging.getLogger(__name__)


class LocalTaskExecutor:
    """带并发上限的进程内执行器。"""

    def __init__(self, max_concurrent: int, *, thread_name_prefix: str = "qforge-task") -> None:
        self._max_concurrent = max(1, max_concurrent)
        self._pool = ThreadPoolExecutor(
            max_workers=self._max_concurrent, thread_name_prefix=thread_name_prefix
        )
        self._slots = threading.BoundedSemaphore(self._max_concurrent)
        self._lock = threading.Lock()
        self._running: set[str] = set()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def running_count(self) -> int:
        with self._lock:
            return len(self._running)

    def running_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._running)

    def has_capacity(self) -> bool:
        """是否还有并发空位（入队前先问，避免任务卡在 QUEUED）。"""
        with self._lock:
            return len(self._running) < self._max_concurrent

    def submit(self, key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """提交任务；并发已满时抛 ResourceLimitError（429），而不是无限排队。

        注意：必须**先登记**再提交给线程池。否则任务可能在登记之前就跑完，
        导致 `_running` 残留一个再也不会被清理的键（并发位被永久占用）。
        """
        if not self._slots.acquire(blocking=False):
            raise ResourceLimitError(
                f"并发任务数已达上限 {self._max_concurrent}，请等待当前任务结束后重试",
                detail={"max_concurrent_tasks": self._max_concurrent, "running": self.running_ids()},
            )

        with self._lock:
            self._running.add(key)

        try:
            self._pool.submit(self._run, key, fn, args, kwargs)
        except Exception:
            with self._lock:
                self._running.discard(key)
            self._slots.release()
            raise

    def _run(
        self,
        key: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        try:
            fn(*args, **kwargs)
        except Exception:  # pragma: no cover - 由 pipeline 内部记录为任务失败
            logger.exception("后台任务执行失败", extra={"task_id": key})
        finally:
            with self._lock:
                self._running.discard(key)
            self._slots.release()
            logger.info("后台任务结束", extra={"task_id": key})

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)


_executor: LocalTaskExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> LocalTaskExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                settings = get_settings()
                _executor = LocalTaskExecutor(settings.max_concurrent_tasks)
    return _executor


def reset_executor() -> None:
    """释放全局执行器（测试隔离用）。"""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None

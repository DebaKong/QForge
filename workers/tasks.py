"""Celery 任务定义。

阶段 1 起，Worker 执行的是**真实流水线**，与进程内执行器调用同一个 `run_pipeline`：

- `probe`：接线自检；
- `run_task_pipeline`：执行完整流水线（校验 → 预处理 → 量化 → Engine → 代码生成 →
  编译 → 运行验证 → 打包），进度、精度与失败原因全部落库并写日志。

无 Redis 时由 `app.services.executor.LocalTaskExecutor` 在进程内后台线程调用同一函数，
因此两种执行方式行为一致（SPEC 3.1：Web API 不直接执行长任务）。
"""

from __future__ import annotations

import logging
from typing import Any

from workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="qforge.probe", bind=True)
def probe(self: Any) -> dict[str, Any]:
    """接线自检：验证 Worker 能被调度并返回结果。"""
    return {
        "task": "qforge.probe",
        "worker": self.request.hostname,
        "status": "ok",
    }


@celery_app.task(name="qforge.run_task_pipeline", bind=True)
def run_task_pipeline(self: Any, task_id: str) -> dict[str, Any]:
    """执行完整部署流水线（真实实现见 app/pipeline/runner.py）。"""
    from app.pipeline import run_pipeline

    worker_id = self.request.hostname or "celery-worker"
    logger.info("Celery 领取任务", extra={"task_id": task_id, "worker_id": worker_id})
    return run_pipeline(task_id, worker_id=worker_id)

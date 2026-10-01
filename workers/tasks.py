"""Celery 任务定义。

阶段 0 只有两个任务，用途是把「API -> 队列 -> Worker -> 任务状态/日志」的接线打通：
- `probe`：不触碰数据库的接线自检；
- `run_task_pipeline`：占位流水线。

明确不做的事（AGENTS.md：非本 Phase 不实现、禁止伪造结果）：
- 不执行真实校验、量化、Engine 构建、代码生成、编译或推理；
- 不把任务标记为 SUCCESS；
- 因此被领取的任务会停留在 QUEUED，并在 job_logs 中记录原因，等待阶段 1 接续实现。
"""

from __future__ import annotations

import logging
from typing import Any

from workers.celery_app import celery_app

logger = logging.getLogger(__name__)

PHASE0_PLACEHOLDER_NOTE = (
    "阶段 0 占位实现：真实流水线（VALIDATING 起）在阶段 1 落地，任务保持 QUEUED"
)


@celery_app.task(name="qforge.probe", bind=True)
def probe(self: Any) -> dict[str, Any]:
    """接线自检：验证 Worker 能被调度并返回结果。"""
    return {
        "task": "qforge.probe",
        "worker": self.request.hostname,
        "status": "ok",
        "phase": "phase-0",
    }


@celery_app.task(name="qforge.run_task_pipeline", bind=True)
def run_task_pipeline(self: Any, task_id: str) -> dict[str, Any]:
    """阶段 0 占位流水线：只登记领取记录，不推进阶段状态。"""
    from app.db.base import session_scope
    from app.services import task_service
    from app.services.task_state import TaskStatus

    worker_id = self.request.hostname or "inline-eager"

    with session_scope() as session:
        task = task_service.get_task(session, task_id)

        if task.status is not TaskStatus.QUEUED:
            task_service.append_log(
                session,
                task,
                f"占位流水线跳过：任务状态为 {task.status.value}，仅 QUEUED 任务会被领取",
                stage=task.status.value,
                level="WARNING",
            )
            return {
                "task_id": task_id,
                "status": task.status.value,
                "executed": False,
                "phase": "phase-0",
            }

        task.worker_id = worker_id
        task_service.append_log(
            session,
            task,
            f"Worker({worker_id}) 已领取任务。{PHASE0_PLACEHOLDER_NOTE}",
            stage=TaskStatus.QUEUED.value,
        )
        logger.info(
            "占位流水线已领取任务",
            extra={"task_id": task_id, "stage": TaskStatus.QUEUED.value, "worker_id": worker_id},
        )
        return {
            "task_id": task_id,
            "status": task.status.value,
            "executed": True,
            "worker_id": worker_id,
            "next_stage": task_service.next_status(task.status).value if task_service.next_status(task.status) else None,
            "phase": "phase-0",
        }

"""Task API（SPEC 16.1 + 状态机查询）。

阶段 0 说明：
- `POST /tasks` 只创建任务与配置快照，不启动真实流水线；
- `POST /tasks/{id}/enqueue` 校验 CREATED -> QUEUED 并投递 Celery；
  阶段 0 的 Worker 任务为占位实现，任务会停留在 QUEUED（见 workers/tasks.py），
  阶段 1 接入真实阶段实现后才会继续推进；
- `GET /tasks/{id}/report` 属阶段 2（精度报告），阶段 0 明确返回 501。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Query, status

from app.api.deps import SessionDep
from app.config.settings import get_settings
from app.errors import NotImplementedInPhaseError
from app.schemas.task import (
    ArtifactRead,
    JobLogRead,
    TaskConfigRead,
    TaskCreate,
    TaskRead,
    TaskTransitionInfo,
)
from app.services import task_service
from app.services.storage import safe_join
from app.services.task_state import TaskStatus

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskRead, status_code=status.HTTP_201_CREATED, summary="创建任务")
def create_task(payload: TaskCreate, session: SessionDep) -> TaskRead:
    task = task_service.create_task(session, payload)
    return TaskRead.model_validate(task)


@router.get("", response_model=list[TaskRead], summary="任务列表")
def list_tasks(
    session: SessionDep,
    project_id: str | None = Query(default=None),
    task_status: TaskStatus | None = Query(default=None, alias="status"),
) -> list[TaskRead]:
    tasks = task_service.list_tasks(session, project_id=project_id, status=task_status)
    return [TaskRead.model_validate(item) for item in tasks]


@router.get("/{task_id}", response_model=TaskRead, summary="任务详情")
def get_task(task_id: str, session: SessionDep) -> TaskRead:
    return TaskRead.model_validate(task_service.get_task(session, task_id))


@router.get("/{task_id}/config", response_model=TaskConfigRead, summary="任务配置快照")
def get_task_config(task_id: str, session: SessionDep) -> TaskConfigRead:
    return TaskConfigRead.model_validate(task_service.get_task_config(session, task_id))


@router.get(
    "/{task_id}/transitions",
    response_model=TaskTransitionInfo,
    summary="任务可迁移状态（前端据此渲染操作，避免前端复制状态机）",
)
def get_task_transitions(task_id: str, session: SessionDep) -> TaskTransitionInfo:
    return TaskTransitionInfo.model_validate(
        task_service.describe_transitions(session, task_id)
    )


@router.post("/{task_id}/enqueue", response_model=TaskRead, summary="入队（CREATED -> QUEUED）")
def enqueue_task(task_id: str, session: SessionDep) -> TaskRead:
    return TaskRead.model_validate(task_service.enqueue_task(session, task_id))


@router.post("/{task_id}/cancel", response_model=TaskRead, summary="取消任务")
def cancel_task(task_id: str, session: SessionDep) -> TaskRead:
    return TaskRead.model_validate(task_service.cancel_task(session, task_id))


@router.get("/{task_id}/logs", response_model=list[JobLogRead], summary="任务日志")
def get_task_logs(task_id: str, session: SessionDep) -> list[JobLogRead]:
    return [JobLogRead.model_validate(item) for item in task_service.list_logs(session, task_id)]


@router.get(
    "/{task_id}/artifacts", response_model=list[ArtifactRead], summary="任务产物列表"
)
def get_task_artifacts(task_id: str, session: SessionDep) -> list[ArtifactRead]:
    return [
        ArtifactRead.model_validate(item)
        for item in task_service.list_artifacts(session, task_id)
    ]


@router.get(
    "/{task_id}/report",
    summary="精度验证报告（阶段 1 提供 Engine 与 FP32 基准的误差指标；完整对比报告属阶段 2）",
)
def get_task_report(task_id: str, session: SessionDep) -> dict:
    """返回任务已产出的精度验证结果。

    阶段 1 已按 SPEC 9.2 计算 MAE / MSE / RMSE / 最大绝对误差 / 余弦相似度
    （FP32 基准 = onnxruntime CPU 推理）。任务尚未跑到 TESTING 阶段时，
    明确返回 501，而不是返回空报告。
    """
    task_service.get_task(session, task_id)  # 先确认任务存在，避免 404 被 501 掩盖

    settings = get_settings()
    storage_root = settings.resolved_storage_root
    for artifact in task_service.list_artifacts(session, task_id):
        if not artifact.relative_path.endswith("report/accuracy.json"):
            continue
        segments = [segment for segment in artifact.relative_path.split("/") if segment]
        path = safe_join(storage_root, *segments)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    raise NotImplementedInPhaseError(
        "该任务尚无精度验证结果（需先完成到 TESTING 阶段）；FP32/FP16/INT8 三方对比报告属阶段 2",
        detail={"task_id": task_id, "planned_phase": "phase-2"},
    )


@router.get(
    "/{task_id}/compatibility",
    summary="算子兼容性报告（SPEC 7.2）",
    description=(
        "返回该任务的算子兼容性报告：每个算子的 operator / domain / opset / supported / "
        "condition / severity / suggestion，以及汇总结论（OK / WARNINGS / BLOCKED）。"
    ),
)
def get_task_compatibility(task_id: str, session: SessionDep) -> dict:
    task_service.get_task(session, task_id)  # 先确认任务存在

    storage_root = get_settings().resolved_storage_root
    for artifact in task_service.list_artifacts(session, task_id):
        if not artifact.relative_path.endswith("report/compatibility.json"):
            continue
        segments = [segment for segment in artifact.relative_path.split("/") if segment]
        path = safe_join(storage_root, *segments)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    raise NotImplementedInPhaseError(
        "该任务尚无算子兼容性报告（需先完成模型校验阶段）",
        detail={"task_id": task_id},
    )

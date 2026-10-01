"""任务编排（SPEC 13）。

职责边界：
- `transition()` 是 Task.status / progress / current_stage 的唯一写入口，
  任何状态变化都必须经过状态机校验，禁止在别处直接赋值；
- 任务创建时落一份不可变 TaskConfig 快照，并建立 storage/<task_id> 目录布局；
- 入队前必须先提交事务，确保 Worker 能看到任务行（eager 模式下尤其关键）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.errors import (
    ConflictError,
    ErrorCode,
    NotFoundError,
    ResourceLimitError,
    ValidationFailedError,
)
from app.models.artifact import Artifact, JobLog
from app.models.dataset import Dataset
from app.models.onnx_model import OnnxModel
from app.models.task import Task, TaskConfig
from app.models.mixins import utcnow
from app.schemas.task import TaskCreate
from app.services import catalog
from app.services.storage import ensure_task_layout
from app.services.task_state import (
    TaskStatus,
    allowed_transitions,
    ensure_transition,
    is_terminal,
    next_status,
    progress_for,
)


# ---------------- 日志 ----------------


def append_log(
    session: Session,
    task: Task,
    message: str,
    *,
    stage: str | None = None,
    level: str = "INFO",
) -> JobLog:
    """写入阶段日志（SPEC 13.2）。调用方负责提交事务。"""
    log = JobLog(
        task_id=task.id,
        stage=stage if stage is not None else task.current_stage,
        level=level,
        message=message,
    )
    session.add(log)
    session.flush()
    return log


# ---------------- 配置快照 ----------------


def build_task_config_snapshot(
    *,
    task_id: str,
    data: TaskCreate,
    onnx_model: OnnxModel | None,
    dataset: Dataset | None,
) -> dict[str, Any]:
    """构造 SPEC 5.1 的 TaskConfig 快照，供前端、API、Worker、代码生成共用。"""
    definition: dict[str, Any] = dict(onnx_model.model_definition or {}) if onnx_model else {}

    preprocess = data.preprocess
    if preprocess is None:
        preprocess = definition.get("preprocessing", {})

    postprocess = data.postprocess
    if postprocess is None:
        postprocess = definition.get("postprocessing", {})

    calibration = data.calibration
    if calibration is None:
        calibration = {
            "dataset_id": dataset.id if dataset else None,
            "kind": dataset.kind if dataset else None,
            "entry": "images/" if dataset else None,
        }

    return {
        "task_id": task_id,
        "task_type": data.task_type,
        "model": {
            "model_id": onnx_model.id if onnx_model else None,
            "path": onnx_model.file_path if onnx_model else None,
            "architecture": onnx_model.architecture if onnx_model else None,
            "definition": definition or None,
        },
        "backend": {
            "name": data.backend_name,
            # 具体版本由阶段 1 的 Backend Adapter 按实际环境写入（SPEC 4.1 / 10.1）
            "version": None,
        },
        "precision": data.precision,
        "preprocess": preprocess or {},
        "postprocess": postprocess or {},
        "calibration": calibration or {},
        "build": data.build or {},
        "validation": data.validation or {},
    }


# ---------------- 状态迁移 ----------------


def transition(
    session: Session,
    task: Task,
    target: TaskStatus,
    *,
    message: str | None = None,
    worker_id: str | None = None,
    error_code: ErrorCode | str | None = None,
) -> Task:
    """唯一的状态写入口：校验迁移合法性并同步进度、阶段、时间戳与日志。"""
    previous = task.status
    ensure_transition(previous, target)

    task.status = target
    task.progress = progress_for(target)
    task.current_stage = target.value
    task.message = message or f"{previous.value} -> {target.value}"

    if target is TaskStatus.FAILED:
        code = error_code or ErrorCode.INTERNAL_ERROR
        task.error_code = code.value if isinstance(code, ErrorCode) else str(code)
    elif target is not TaskStatus.CANCELLED:
        # 取消是主动终止，不代表失败；其余前进路径清空历史错误码
        task.error_code = None

    if target is not TaskStatus.CREATED and task.started_at is None:
        task.started_at = utcnow()
    if is_terminal(target):
        task.finished_at = utcnow()

    if worker_id:
        task.worker_id = worker_id

    append_log(
        session,
        task,
        task.message,
        stage=target.value,
        level="ERROR" if target is TaskStatus.FAILED else "INFO",
    )
    session.flush()
    return task


def fail_task(
    session: Session, task: Task, *, message: str, error_code: ErrorCode | str
) -> Task:
    """失败路径统一入口：任务必须处于非终态。"""
    return transition(session, task, TaskStatus.FAILED, message=message, error_code=error_code)


# ---------------- 任务读写 ----------------


def _validate_links(
    session: Session,
    data: TaskCreate,
    onnx_model: OnnxModel | None,
    dataset: Dataset | None,
) -> None:
    if data.project_id:
        catalog.get_project(session, data.project_id)
    if onnx_model and data.project_id and onnx_model.project_id != data.project_id:
        raise ConflictError(
            "模型不属于指定项目",
            detail={"model_id": onnx_model.id, "project_id": data.project_id},
        )
    if dataset and data.project_id and dataset.project_id != data.project_id:
        raise ConflictError(
            "数据集不属于指定项目",
            detail={"dataset_id": dataset.id, "project_id": data.project_id},
        )
    if data.precision == "int8" and dataset is None:
        # SPEC 9：INT8 为静态校准量化，必须提供校准集
        raise ValidationFailedError(
            "INT8 静态量化必须提供校准数据集（dataset_id）",
            detail={"precision": data.precision},
        )


def create_task(session: Session, data: TaskCreate) -> Task:
    onnx_model = catalog.get_model(session, data.model_id) if data.model_id else None
    dataset = catalog.get_dataset(session, data.dataset_id) if data.dataset_id else None
    _validate_links(session, data, onnx_model, dataset)

    task = Task(
        project_id=data.project_id,
        model_id=onnx_model.id if onnx_model else None,
        dataset_id=dataset.id if dataset else None,
        task_type=data.task_type,
        precision=data.precision,
        backend_name=data.backend_name,
        status=TaskStatus.CREATED,
        progress=progress_for(TaskStatus.CREATED),
        current_stage=TaskStatus.CREATED.value,
        message="任务已创建，等待入队",
    )
    session.add(task)
    session.flush()

    snapshot = build_task_config_snapshot(
        task_id=task.id, data=data, onnx_model=onnx_model, dataset=dataset
    )
    session.add(TaskConfig(task_id=task.id, config=snapshot))

    # 存储布局（SPEC 14.2）：目录名由 task_id 生成，不使用任何用户提供的名称
    ensure_task_layout(get_settings().resolved_storage_root, task.id)

    append_log(
        session,
        task,
        "任务已创建，TaskConfig 快照与存储目录已建立",
        stage=TaskStatus.CREATED.value,
    )
    session.flush()
    return task


def get_task(session: Session, task_id: str) -> Task:
    task = session.get(Task, task_id)
    if task is None:
        raise NotFoundError("任务不存在", detail={"task_id": task_id})
    return task


def get_task_config(session: Session, task_id: str) -> TaskConfig:
    config = session.scalar(select(TaskConfig).where(TaskConfig.task_id == task_id))
    if config is None:
        raise NotFoundError("任务配置快照不存在", detail={"task_id": task_id})
    return config


def list_tasks(
    session: Session, *, project_id: str | None = None, status: TaskStatus | None = None
) -> Sequence[Task]:
    stmt = select(Task).order_by(Task.created_at.desc())
    if project_id:
        stmt = stmt.where(Task.project_id == project_id)
    if status is not None:
        stmt = stmt.where(Task.status == status)
    return session.scalars(stmt).all()


def list_logs(session: Session, task_id: str) -> Sequence[JobLog]:
    get_task(session, task_id)
    return session.scalars(
        select(JobLog).where(JobLog.task_id == task_id).order_by(JobLog.id.asc())
    ).all()


def list_artifacts(session: Session, task_id: str) -> Sequence[Artifact]:
    get_task(session, task_id)
    return session.scalars(
        select(Artifact).where(Artifact.task_id == task_id).order_by(Artifact.created_at.asc())
    ).all()


def describe_transitions(session: Session, task_id: str) -> dict[str, Any]:
    task = get_task(session, task_id)
    return {
        "status": task.status,
        "allowed_transitions": sorted(allowed_transitions(task.status), key=lambda s: s.value),
        "next_status": next_status(task.status),
        "is_terminal": is_terminal(task.status),
        "progress": task.progress,
    }


# ---------------- 入队与取消 ----------------


def enqueue_task(session: Session, task_id: str) -> Task:
    """CREATED -> QUEUED 并投递执行。

    执行方式由 `executor_mode` 决定（SPEC 3.1：Web API 不直接执行长任务）：
    - local：进程内后台线程执行器，本请求立即返回；
    - celery：投递到真实 broker（接入 Redis 后使用）。

    顺序要求：先提交事务再投递——Worker（含本地执行器线程）必须能读到已提交的任务行；
    并发已满时在状态迁移**之前**就拒绝，避免任务卡在 QUEUED 无法回退。
    """
    task = get_task(session, task_id)
    if task.status is not TaskStatus.CREATED:
        raise ConflictError(
            f"只有 CREATED 状态的任务可以入队，当前状态 {task.status.value}",
            detail={"task_id": task.id, "status": task.status.value},
        )

    settings = get_settings()
    executor = None
    if settings.executor_mode == "local":
        from app.services.executor import get_executor

        executor = get_executor()
        if not executor.has_capacity():
            raise ResourceLimitError(
                f"并发任务数已达上限 {executor.max_concurrent}，请等待当前任务结束后重试",
                detail={
                    "max_concurrent_tasks": executor.max_concurrent,
                    "running_tasks": executor.running_ids(),
                },
            )

    transition(session, task, TaskStatus.QUEUED, message="任务已入队，等待 Worker 领取")
    session.commit()

    if executor is not None:
        from app.pipeline import run_pipeline

        executor.submit(task.id, run_pipeline, task.id)
        session.refresh(task)
        return task

    # 延迟导入：避免 API 进程启动即依赖 Worker 模块
    from workers.tasks import run_task_pipeline

    run_task_pipeline.delay(task.id)

    # 刷新：eager 模式下 Worker 已在另一个会话中更新该行（worker_id / 日志），
    # 而本会话的实例因 expire_on_commit=False 仍是旧值；非 eager 时该刷新读到当前值。
    session.refresh(task)
    return task


def cancel_task(session: Session, task_id: str) -> Task:
    task = get_task(session, task_id)
    if is_terminal(task.status):
        raise ConflictError(
            f"任务已处于终态 {task.status.value}，无法取消",
            detail={"task_id": task.id, "status": task.status.value},
        )
    return transition(session, task, TaskStatus.CANCELLED, message="任务已被用户取消")

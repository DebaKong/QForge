"""项目删除（SPEC 5.2 项目组织单元）：级联清理元数据与磁盘文件。

与「模型/数据集单独删除」的区别：删项目是**一次清掉该项目下的全部痕迹**，
因此这里坚持两条原则：

1. **先算账再动手**：`preview()` 先算出会连带删除多少模型/数据集/任务/产物、
   占用多少磁盘，界面拿它做确认框，避免用户误删。
2. **该拒绝就拒绝**：只要项目下还有排队或运行中的任务，就拒绝删除并说清楚原因
   （Worker 可能正在写这些目录；先取消任务再删）。CREATED（建了没入队）不算运行中。

磁盘删除全部走 `storage.safe_join` / `layout.*_root`，路径解析后必须仍在
storage_root 之内（防路径穿越），且目录不存在时按幂等处理（不报错）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.errors import ConflictError, NotFoundError
from app.models.artifact import Artifact, JobLog
from app.models.dataset import Dataset
from app.models.onnx_model import OnnxModel
from app.models.project import Project
from app.models.task import Task, TaskConfig
from app.services import catalog, layout, storage, uploads
from app.services.task_state import TERMINAL_STATUSES, TaskStatus

logger = logging.getLogger(__name__)

# 这些状态说明任务正在（或马上）被 Worker 处理：删项目会删到正在写入的目录
BLOCKING_STATUSES: tuple[TaskStatus, ...] = tuple(
    status
    for status in TaskStatus
    if status not in TERMINAL_STATUSES and status is not TaskStatus.CREATED
)
_BLOCKING_VALUES = {status.value for status in BLOCKING_STATUSES}

# 批量删除时每次 IN 查询的任务数量（避免 SQLite 变量上限）
_TASK_ID_CHUNK = 400


@dataclass
class ProjectDeletionPreview:
    """删除前的影响面：给界面确认框用。"""

    project_id: str
    project_name: str
    models: int = 0
    datasets: int = 0
    tasks: int = 0
    artifacts: int = 0
    logs: int = 0
    disk_bytes: int = 0
    blocking_tasks: list[dict[str, str]] = field(default_factory=list)

    @property
    def can_delete(self) -> bool:
        return not self.blocking_tasks

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "models": self.models,
            "datasets": self.datasets,
            "tasks": self.tasks,
            "artifacts": self.artifacts,
            "logs": self.logs,
            "disk_bytes": self.disk_bytes,
            "blocking_tasks": self.blocking_tasks,
            "can_delete": self.can_delete,
        }


def _directory_size(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:  # pragma: no cover - 文件刚被清理
                continue
    return total


def _project_dirs(storage_root: Path, task_ids: list[str], model_ids: list[str],
                  dataset_ids: list[str]) -> list[Path]:
    """该项目在磁盘上占用的目录（全部经安全拼接，越界直接抛异常）。"""
    directories = [storage.task_dir(storage_root, task_id) for task_id in task_ids]
    directories += [layout.model_root(storage_root, model_id) for model_id in model_ids]
    directories += [layout.dataset_root(storage_root, dataset_id) for dataset_id in dataset_ids]
    return directories


def _collect(session: Session, project_id: str) -> tuple[list[str], list[str], list[str], list[Task]]:
    task_ids = list(session.scalars(select(Task.id).where(Task.project_id == project_id)))
    model_ids = list(
        session.scalars(select(OnnxModel.id).where(OnnxModel.project_id == project_id))
    )
    dataset_ids = list(session.scalars(select(Dataset.id).where(Dataset.project_id == project_id)))
    tasks = list(session.scalars(select(Task).where(Task.project_id == project_id)))
    return task_ids, model_ids, dataset_ids, tasks


def preview(session: Session, storage_root: Path, project_id: str) -> ProjectDeletionPreview:
    """算清删这个项目会连带删掉什么（不修改任何数据）。"""
    project = catalog.get_project(session, project_id)
    task_ids, model_ids, dataset_ids, tasks = _collect(session, project_id)

    artifacts = 0
    logs = 0
    if task_ids:
        artifacts = sum(
            1
            for chunk in _chunks(task_ids)
            for _ in session.scalars(select(Artifact.id).where(Artifact.task_id.in_(chunk)))
        )
        logs = sum(
            1
            for chunk in _chunks(task_ids)
            for _ in session.scalars(select(JobLog.id).where(JobLog.task_id.in_(chunk)))
        )

    disk_bytes = sum(
        _directory_size(path)
        for path in _project_dirs(storage_root, task_ids, model_ids, dataset_ids)
    )

    blocking = [
        {"id": task.id, "status": task.status.value, "stage": task.current_stage or ""}
        for task in tasks
        if task.status.value in _BLOCKING_VALUES
    ]

    return ProjectDeletionPreview(
        project_id=project.id,
        project_name=project.name,
        models=len(model_ids),
        datasets=len(dataset_ids),
        tasks=len(task_ids),
        artifacts=artifacts,
        logs=logs,
        disk_bytes=disk_bytes,
        blocking_tasks=blocking,
    )


def _chunks(values: list[str]) -> list[list[str]]:
    return [values[index : index + _TASK_ID_CHUNK] for index in range(0, len(values), _TASK_ID_CHUNK)]


def delete_project(session: Session, storage_root: Path, project_id: str) -> dict[str, object]:
    """删除项目：先做安全检查，再删磁盘、最后删元数据。

    抛 `ConflictError`（409）表示项目下有排队/运行中的任务，需先取消。
    """
    project = catalog.get_project(session, project_id)
    preview_result = preview(session, storage_root, project_id)
    if preview_result.blocking_tasks:
        detail = {
            "project_id": project_id,
            "blocking_tasks": preview_result.blocking_tasks,
        }
        raise ConflictError(
            "项目下还有排队或运行中的任务，请先取消这些任务再删除项目",
            detail=detail,
        )

    task_ids, model_ids, dataset_ids, _ = _collect(session, project_id)

    # 1) 磁盘：只删该项目自己的目录，且必须仍在存储根之内
    removed_dirs = 0
    freed_bytes = 0
    for directory in _project_dirs(storage_root, task_ids, model_ids, dataset_ids):
        if not storage.is_within(storage_root, directory):
            # 理论上不可达（路径由 id 拼接），保留防护以免将来改动引入越界删除
            raise ConflictError(
                "拒绝删除存储根之外的目录", detail={"path": str(directory)}
            )
        if directory.exists():
            freed_bytes += _directory_size(directory)
            uploads.remove_tree(directory)
            removed_dirs += 1

    # 2) 元数据：先清任务的下挂对象，再清任务/模型/数据集，最后清项目
    #    任务里的 model_id / dataset_id 若指向本项目要删的模型或数据集，
    #    在其他项目的任务上显式置空，避免留下悬空引用（SQLite 默认不开外键级联）
    others = select(Task.id).where(Task.project_id != project_id)
    if model_ids:
        session.execute(
            Task.__table__.update()
            .where(Task.id.in_(others), Task.model_id.in_(model_ids))
            .values(model_id=None)
        )
    if dataset_ids:
        session.execute(
            Task.__table__.update()
            .where(Task.id.in_(others), Task.dataset_id.in_(dataset_ids))
            .values(dataset_id=None)
        )

    deleted_artifacts = 0
    deleted_logs = 0
    deleted_configs = 0
    for chunk in _chunks(task_ids):
        deleted_artifacts += session.execute(
            delete(Artifact).where(Artifact.task_id.in_(chunk))
        ).rowcount or 0
        deleted_logs += session.execute(
            delete(JobLog).where(JobLog.task_id.in_(chunk))
        ).rowcount or 0
        deleted_configs += session.execute(
            delete(TaskConfig).where(TaskConfig.task_id.in_(chunk))
        ).rowcount or 0
        session.execute(delete(Task).where(Task.id.in_(chunk)))

    deleted_models = (
        session.execute(delete(OnnxModel).where(OnnxModel.project_id == project_id)).rowcount or 0
    )
    deleted_datasets = (
        session.execute(delete(Dataset).where(Dataset.project_id == project_id)).rowcount or 0
    )
    session.execute(delete(Project).where(Project.id == project_id))
    session.flush()

    result = {
        "project_id": project_id,
        "project_name": project.name,
        "deleted": {
            "tasks": len(task_ids),
            "models": deleted_models,
            "datasets": deleted_datasets,
            "artifacts": deleted_artifacts,
            "logs": deleted_logs,
            "task_configs": deleted_configs,
        },
        "removed_directories": removed_dirs,
        "freed_bytes": freed_bytes,
    }
    logger.info(
        "项目已删除",
        extra={"project_id": project_id, "project_name": project.name, **result["deleted"]},
    )
    return result


def delete_projects(
    session: Session, storage_root: Path, project_ids: list[str]
) -> dict[str, object]:
    """批量删除：逐个执行，单个失败不影响其余（返回每项结果供界面展示）。"""
    deleted: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    seen: set[str] = set()

    for project_id in project_ids:
        if project_id in seen:
            continue
        seen.add(project_id)
        try:
            deleted.append(delete_project(session, storage_root, project_id))
            # 逐个提交：某个项目失败时不能把前面已删成功的记录一起回滚
            # （磁盘文件已经删掉了，回滚数据库只会造成两边不一致）
            session.commit()
        except (ConflictError, NotFoundError) as error:
            session.rollback()
            failed.append({"project_id": project_id, "reason": error.message})
            logger.warning("删除项目失败：%s（%s）", project_id, error.message)

    return {
        "deleted": deleted,
        "failed": failed,
        "deleted_count": len(deleted),
        "failed_count": len(failed),
        "freed_bytes": sum(int(item["freed_bytes"]) for item in deleted),
    }

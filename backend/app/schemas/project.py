"""Project 请求/响应模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import ORMModel


class ProjectCreate(ORMModel):
    name: str = Field(min_length=1, max_length=128, description="项目名，全局唯一")
    description: str | None = Field(default=None, max_length=2000)


class ProjectRead(ORMModel):
    id: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class BlockingTaskRead(ORMModel):
    """阻止删除项目的原因：该项目下正在排队/运行的任务。"""

    id: str
    status: str
    stage: str = ""


class ProjectDeletionPreviewRead(ORMModel):
    """删除确认框需要的信息：删什么、占多少磁盘、能不能删。"""

    project_id: str
    project_name: str
    models: int = 0
    datasets: int = 0
    tasks: int = 0
    artifacts: int = 0
    logs: int = 0
    disk_bytes: int = 0
    blocking_tasks: list[BlockingTaskRead] = Field(default_factory=list)
    can_delete: bool = True


class ProjectDeletionResult(ORMModel):
    project_id: str
    project_name: str
    deleted: dict[str, int] = Field(default_factory=dict)
    removed_directories: int = 0
    freed_bytes: int = 0


class ProjectBatchDeleteRequest(ORMModel):
    project_ids: list[str] = Field(min_length=1, max_length=200, description="要删除的项目 ID")


class ProjectBatchDeleteResult(ORMModel):
    deleted: list[ProjectDeletionResult] = Field(default_factory=list)
    failed: list[dict[str, str]] = Field(default_factory=list)
    deleted_count: int = 0
    failed_count: int = 0
    freed_bytes: int = 0

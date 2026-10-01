"""Task 相关请求/响应模型（SPEC 5.1 / 13.1）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.models.task import PRECISION_MODES, TASK_TYPES
from app.schemas.common import ORMModel
from app.services.task_state import TaskStatus


class TaskCreate(ORMModel):
    """创建任务。阶段 0 只登记配置快照，不触发真实流水线。"""

    task_type: str = Field(default="detection")
    precision: str = Field(default="fp16", description="fp32 | fp16 | int8（不得混称）")
    backend_name: str = Field(default="tensorrt", max_length=32)

    project_id: str | None = Field(default=None, max_length=32)
    model_id: str | None = Field(default=None, max_length=32)
    dataset_id: str | None = Field(default=None, max_length=32)

    # SPEC 5.1 的配置分节；未提供的分节从模型定义的对应分节继承
    preprocess: dict[str, Any] | None = None
    postprocess: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    build: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None

    @field_validator("task_type")
    @classmethod
    def _check_task_type(cls, value: str) -> str:
        if value not in TASK_TYPES:
            raise ValueError(f"task_type 必须是 {TASK_TYPES} 之一，阶段 0 只支持 2D 检测")
        return value

    @field_validator("precision")
    @classmethod
    def _check_precision(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in PRECISION_MODES:
            raise ValueError(
                f"precision 必须是 {PRECISION_MODES} 之一；"
                "FP32 基准 / FP16 半精度 / INT8 静态量化不得混称"
            )
        return normalized

    @field_validator("backend_name")
    @classmethod
    def _check_backend(cls, value: str) -> str:
        normalized = value.lower()
        if normalized != "tensorrt":
            raise ValueError("阶段 0 / MVP 只登记 tensorrt 后端（SPEC 2.1）")
        return normalized


class TaskRead(ORMModel):
    id: str
    project_id: str | None
    model_id: str | None
    dataset_id: str | None
    task_type: str
    precision: str
    backend_name: str
    status: TaskStatus
    progress: int
    current_stage: str | None
    message: str | None
    error_code: str | None
    started_at: datetime | None
    finished_at: datetime | None
    worker_id: str | None
    artifact_id: str | None
    created_at: datetime
    updated_at: datetime


class TaskTransitionInfo(ORMModel):
    """任务可执行的下一步（前端据此显示按钮，避免前端猜状态机）。"""

    status: TaskStatus
    allowed_transitions: list[TaskStatus]
    next_status: TaskStatus | None
    is_terminal: bool
    progress: int


class TaskConfigRead(ORMModel):
    id: str
    task_id: str
    config: dict[str, Any]
    created_at: datetime


class JobLogRead(ORMModel):
    id: int
    task_id: str
    stage: str | None
    level: str
    message: str
    created_at: datetime


class ArtifactRead(ORMModel):
    id: str
    task_id: str
    kind: str
    relative_path: str
    size_bytes: int | None
    sha256: str | None
    created_at: datetime

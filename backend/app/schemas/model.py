"""OnnxModel 请求/响应模型。

阶段 0：只登记模型元数据与 SPEC 6.1 的 Model Definition，不接受文件上传。
上传、ONNX Checker、Runtime Load Test、图分析在阶段 1 实现（SPEC 7.1）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.models.task import TASK_TYPES
from app.schemas.common import ORMModel


class ModelCreate(ORMModel):
    project_id: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)
    architecture: str | None = Field(default=None, max_length=64, description="如 yolov8")
    task_type: str = Field(default="detection")
    # 存储根目录下的相对路径；阶段 1 由上传流程生成
    file_path: str | None = Field(default=None, max_length=512)
    opset: int | None = Field(default=None, ge=1, le=21)
    model_definition: dict[str, Any] | None = Field(
        default=None, description="SPEC 6.1 模型语义定义（预处理/输出/后处理）"
    )

    @field_validator("task_type")
    @classmethod
    def _check_task_type(cls, value: str) -> str:
        if value not in TASK_TYPES:
            raise ValueError(f"task_type 必须是 {TASK_TYPES} 之一，阶段 0 只支持 2D 检测")
        return value


class ModelRead(ORMModel):
    id: str
    project_id: str
    name: str
    task_type: str
    architecture: str | None
    file_path: str | None
    sha256: str | None
    size_bytes: int | None
    opset: int | None
    input_spec: dict[str, Any] | None
    output_spec: dict[str, Any] | None
    model_definition: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

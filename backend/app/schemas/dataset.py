"""Dataset 请求/响应模型。

阶段 0：只登记数据集元数据；calibration.zip 上传、ZIP 炸弹/路径穿越防护与
图像校验在阶段 1 按 SPEC 8.3 / 15 实现。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.models.dataset import DATASET_KINDS
from app.schemas.common import ORMModel


class DatasetCreate(ORMModel):
    project_id: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)
    kind: str = Field(default="calibration")
    root_path: str | None = Field(default=None, max_length=512)

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, value: str) -> str:
        if value not in DATASET_KINDS:
            raise ValueError(f"kind 必须是 {DATASET_KINDS} 之一")
        return value


class DatasetRead(ORMModel):
    id: str
    project_id: str
    name: str
    kind: str
    root_path: str | None
    image_count: int
    meta: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

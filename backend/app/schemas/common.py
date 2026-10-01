"""通用响应模型与错误响应结构。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    """可直接由 ORM 实体构造的响应基类。"""

    model_config = ConfigDict(from_attributes=True)


class PageMeta(BaseModel):
    total: int = Field(description="符合筛选条件的记录总数")


class ErrorResponse(BaseModel):
    """统一错误结构（SPEC 13.1 error_code + message）。"""

    error_code: str = Field(description="结构化错误码，见 SPEC 15.1")
    message: str = Field(description="用户可读错误信息")
    detail: dict[str, Any] = Field(default_factory=dict, description="结构化上下文")


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str
    python_version: str
    database: str
    storage_root: str
    celery_eager: bool = Field(description="阶段 0 为 true：无 Redis，任务同步执行")
    phase: str = Field(description="当前实现阶段，阶段 0 为脚手架")

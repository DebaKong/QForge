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

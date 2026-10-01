"""Backend：推理后端及版本/能力信息（SPEC 5.2 / 7.1 Capability Registry）。

阶段 0 只提供实体与注册入口；TensorRT 的真实能力表由阶段 1 的 Backend Adapter 填充。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import TimestampMixin, new_id


class Backend(Base, TimestampMixin):
    __tablename__ = "backends"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    # SPEC 4.1：必须记录具体版本，不允许只写 "TensorRT 8.x"
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    capabilities: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Backend name={self.name!r} version={self.version!r}>"

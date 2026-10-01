"""Artifact 与 JobLog：任务产物与结构化日志（SPEC 5.2 / 12.2 / 13.2）。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import new_id, utcnow

if TYPE_CHECKING:
    from app.models.task import Task

# 产物类型与 SPEC 12.2 的 artifact 目录一致
ARTIFACT_KINDS: tuple[str, ...] = (
    "engine",
    "source",
    "config",
    "docker",
    "test",
    "report",
    "archive",
)

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")


class Artifact(Base):
    """任务产物。`relative_path` 相对存储根目录，避免入库宿主机绝对路径。"""

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    task: Mapped["Task"] = relationship(back_populates="artifacts")

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Artifact id={self.id} kind={self.kind!r} path={self.relative_path!r}>"


class JobLog(Base):
    """阶段日志（SPEC 13.2）：必须能回答任务、阶段、级别、时间与失败原因。"""

    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="INFO")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    task: Mapped["Task"] = relationship(back_populates="logs")

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<JobLog task_id={self.task_id} level={self.level} stage={self.stage}>"

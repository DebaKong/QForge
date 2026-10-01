"""Task 与 TaskConfig：一次流水线执行记录与其配置快照（SPEC 5.1 / 13.1）。

TaskConfig 为不可变快照：前端、API、Worker、代码生成共用同一份配置（AGENTS.md 边界），
因此任务创建时写入后不再修改，避免执行过程中配置漂移。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin, new_id, utcnow
from app.services.task_state import TaskStatus

if TYPE_CHECKING:
    from app.models.artifact import Artifact, JobLog
    from app.models.dataset import Dataset
    from app.models.onnx_model import OnnxModel
    from app.models.project import Project

# 精度模式（SPEC 9）：FP32 基准 / FP16 半精度 / INT8 静态量化，不得混称
PRECISION_MODES: tuple[str, ...] = ("fp32", "fp16", "int8")
TASK_TYPES: tuple[str, ...] = ("detection",)

_status_column = Enum(
    TaskStatus,
    name="task_status",
    native_enum=False,
    length=32,
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
    validate_strings=True,
)


class Task(Base, TimestampMixin):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)

    # 阶段 0 允许不挂项目的临时任务；挂项目时随项目级联
    project_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    model_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("models.id", ondelete="SET NULL"), nullable=True, index=True
    )
    dataset_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )

    # SPEC 5.1 顶层字段
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="detection")
    precision: Mapped[str] = mapped_column(String(16), nullable=False, default="fp16")
    backend_name: Mapped[str] = mapped_column(String(32), nullable=False, default="tensorrt")

    # SPEC 13.1 状态字段
    status: Mapped[TaskStatus] = mapped_column(
        _status_column, nullable=False, default=TaskStatus.CREATED
    )
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # 最终产物 ID。此处不加外键约束：artifacts.task_id 已指向本表，
    # 双向外键会在 SQLite/PG 上引入建表顺序与删除顺序耦合，阶段 0 用应用层维护一致性。
    artifact_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    project: Mapped["Project | None"] = relationship(back_populates="tasks")
    onnx_model: Mapped["OnnxModel | None"] = relationship()
    dataset: Mapped["Dataset | None"] = relationship()
    task_config: Mapped["TaskConfig | None"] = relationship(
        back_populates="task", cascade="all, delete-orphan", uselist=False, passive_deletes=True
    )
    logs: Mapped[list["JobLog"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", passive_deletes=True
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Task id={self.id} status={self.status.value} precision={self.precision!r}>"


class TaskConfig(Base):
    """任务配置快照（SPEC 5.1）。整份 JSON 只写一次，不做原地修改。"""

    __tablename__ = "task_configs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    # 完整 TaskConfig 快照：task_type / model / backend / precision / preprocess /
    # postprocess / calibration / build / validation
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    task: Mapped["Task"] = relationship(back_populates="task_config")

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<TaskConfig task_id={self.task_id}>"

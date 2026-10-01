"""Project：项目级组织单元（SPEC 5.2）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin, new_id

if TYPE_CHECKING:
    from app.models.dataset import Dataset
    from app.models.onnx_model import OnnxModel
    from app.models.task import Task


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    models: Mapped[list["OnnxModel"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="project", passive_deletes=True
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Project id={self.id} name={self.name!r}>"

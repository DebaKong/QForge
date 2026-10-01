"""Dataset：校准集/测试集及其元数据（SPEC 5.2 / 8.2）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin, new_id

if TYPE_CHECKING:
    from app.models.project import Project

# SPEC 8.2：MVP 校准集为图像目录（calibration.zip -> images/）
DATASET_KINDS: tuple[str, ...] = ("calibration", "test")


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_datasets_project_name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="calibration")

    # 存储根目录下的相对路径（阶段 1 由上传与解压流程写入）
    root_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 校准集校验结果（SPEC 8.3：数量/分辨率/通道/异常图片清单等）
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="datasets")

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Dataset id={self.id} name={self.name!r} kind={self.kind!r}>"

"""OnnxModel：原始 ONNX 模型及其元数据（SPEC 5.2 / 6）。

阶段 0 只保存元数据与模型定义的注册信息，文件上传与 ONNX 解析在阶段 1 落地。
`model_definition` 保存 SPEC 6.1 的模型语义（预处理/输出/后处理），
不允许由 ONNX Shape 猜测后处理（AGENTS.md 边界）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin, new_id

if TYPE_CHECKING:
    from app.models.project import Project


class OnnxModel(Base, TimestampMixin):
    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_models_project_name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    # SPEC 5.1：model.architecture / task_type
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="detection")
    architecture: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 存储根目录下的相对路径（不保存宿主机绝对路径）
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    opset: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ONNX 解析结果与模型定义（SPEC 6.1）；阶段 1 由 Model Analyzer 填充
    input_spec: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    output_spec: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_definition: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="models")

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<OnnxModel id={self.id} name={self.name!r} arch={self.architecture!r}>"

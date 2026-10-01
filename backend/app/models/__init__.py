"""ORM 实体（SPEC 5.2 / 14.1）。

表清单与 SPEC 14.1 一致：projects / models / datasets / tasks / task_configs /
artifacts / job_logs / backends。

约定：
- 主键为 32 位十六进制 UUID 字符串（跨 SQLite/PostgreSQL 一致，便于产物与日志引用）；
- 时间戳统一为带时区 UTC；
- 结构化内容使用 JSON 列（SQLite 与 PostgreSQL 均支持），阶段 1 迁移到 PostgreSQL 无需改模型。
"""

from app.models.artifact import Artifact, JobLog
from app.models.backend import Backend
from app.models.dataset import Dataset
from app.models.onnx_model import OnnxModel
from app.models.project import Project
from app.models.task import Task, TaskConfig

__all__ = [
    "Artifact",
    "Backend",
    "Dataset",
    "JobLog",
    "OnnxModel",
    "Project",
    "Task",
    "TaskConfig",
]

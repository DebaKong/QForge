"""流水线上下文：一次任务执行所需的全部输入与产物位置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.adapters.base import BackendAdapter, ModelAdapter
from app.adapters.definition import ModelDefinition
from app.config.settings import Settings
from app.services.layout import task_subdir


@dataclass
class PipelineContext:
    """任务执行上下文（在 VALIDATING 阶段逐步补全）。"""

    task_id: str
    settings: Settings
    storage_root: Path
    task_config: dict[str, Any]
    precision: str
    backend_name: str
    architecture: str

    definition: ModelDefinition | None = None
    model_adapter: ModelAdapter | None = None
    backend_adapter: BackendAdapter | None = None

    model_id: str | None = None
    dataset_id: str | None = None
    dataset_kind: str | None = None
    calibration_dataset_root: Path | None = None
    inspection: dict[str, Any] | None = None

    onnx_path: Path | None = None
    calibration_images: list[Path] = field(default_factory=list)
    # INT8 显式量化（Q/DQ）产物：由 QUANTIZING 阶段生成，BUILDING_ENGINE 阶段优先解析它
    quantized_onnx_path: Path | None = None
    sample_image: Path | None = None

    engine_path: Path | None = None
    engine_metadata: dict[str, Any] = field(default_factory=dict)
    source_dir: Path | None = None
    build_dir: Path | None = None
    executable_path: Path | None = None
    verification: dict[str, Any] = field(default_factory=dict)
    cpp_build_status: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    started_monotonic: float = 0.0

    # ---- 目录 ----
    @property
    def task_dir(self) -> Path:
        return task_subdir(self.storage_root, self.task_id, "input").parent

    @property
    def build_options(self) -> dict[str, Any]:
        """TaskConfig.build 分节（SPEC 5.1）。"""
        section = self.task_config.get("build")
        return section if isinstance(section, dict) else {}

    def subdir(self, name: str) -> Path:
        return task_subdir(self.storage_root, self.task_id, name)

    @property
    def input_dir(self) -> Path:
        return self.subdir("input")

    @property
    def calibration_dir(self) -> Path:
        return self.subdir("calibration")

    @property
    def intermediate_dir(self) -> Path:
        return self.subdir("intermediate")

    @property
    def engine_dir(self) -> Path:
        return self.subdir("engine")

    @property
    def source_root(self) -> Path:
        return self.subdir("source")

    @property
    def docker_dir(self) -> Path:
        return self.subdir("docker")

    @property
    def logs_dir(self) -> Path:
        return self.subdir("logs")

    @property
    def report_dir(self) -> Path:
        return self.subdir("report")

    def log_file(self, name: str) -> Path:
        return self.logs_dir / name

    def report_file(self, name: str) -> Path:
        return self.report_dir / name

"""适配器接口（SPEC 6.2 / 19）。

`ModelAdapter`：模型语义与模板参数；
`BackendAdapter`：量化/Engine 构建等后端相关能力。

两者都通过注册表按名称获取，业务层不出现针对具体模型或后端的分支判断。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.adapters.definition import ModelDefinition


@dataclass
class Detection:
    """一条检测结果（后处理输出）。"""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "score": self.score,
            "class_id": self.class_id,
        }


@dataclass
class DecodeResult:
    """解码 + NMS 的结果，用于运行验证与精度报告。"""

    detections: list[Detection] = field(default_factory=list)
    raw_shape: tuple[int, ...] = ()
    candidates: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "raw_shape": list(self.raw_shape),
            "candidates": self.candidates,
            "detections": len(self.detections),
            "preview": [item.to_dict() for item in self.detections[:10]],
        }


class ModelAdapter(ABC):
    """模型适配器：把模型语义交给平台，而不是让平台猜。"""

    architecture: str = ""
    task: str = "detection"

    @abstractmethod
    def default_definition(self, inspection: dict[str, Any] | None = None) -> dict[str, Any]:
        """给出该架构的默认 Model Definition（可由用户在任务中覆盖）。"""

    @abstractmethod
    def validate(
        self, definition: ModelDefinition, inspection: dict[str, Any] | None = None
    ) -> list[str]:
        """模型特定合法性检查；返回警告列表，硬错误直接抛领域异常。"""

    @abstractmethod
    def template_context(self, definition: ModelDefinition) -> dict[str, Any]:
        """提供给代码生成模板的参数。"""

    @abstractmethod
    def decode(
        self, outputs: list[np.ndarray], definition: ModelDefinition
    ) -> DecodeResult:
        """按模型语义解码输出并执行 NMS（用于运行验证与精度报告）。"""


class BackendAdapter(ABC):
    """后端适配器：隔离 TensorRT / OpenVINO 等差异（SPEC 19）。"""

    name: str = ""

    @abstractmethod
    def availability(self) -> tuple[bool, dict[str, Any]]:
        """返回 (是否可用, 环境信息)。不可用时必须给出可读原因。"""

    def operator_capabilities(
        self, inspection: dict[str, Any], capability: dict[str, Any]
    ) -> dict[str, Any]:
        """算子兼容性报告（SPEC 7.1 步骤 6 / 7.2）。

        默认返回空报告（表示该后端未提供算子能力表）；有注册表的后端覆盖此方法。
        放在适配器里是为了让"哪个后端支持哪些算子"这种私有知识不出现在通用流水线中
        （SPEC 3.1 / AGENTS.md：避免 `if backend == ...`）。
        """
        return {
            "operators": [],
            "summary": {
                "backend": self.name,
                "verdict": "UNKNOWN",
                "reason": "该后端未提供算子能力注册表",
            },
        }

    @abstractmethod
    def build_engine(
        self,
        *,
        onnx_path: Path,
        engine_path: Path,
        definition: ModelDefinition,
        precision: str,
        calibration_images: list[Path] | None,
        workspace_bytes: int,
        calibration_cache: Path | None,
        log_path: Path,
    ) -> dict[str, Any]:
        """构建 Engine，返回构建元数据（engine_metadata，SPEC 4.1 / 10.1）。"""

    @abstractmethod
    def run_inference(
        self, *, engine_path: Path, inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        """用 Engine 做一次真实推理（SPEC 11.3 步骤 4~6）。"""

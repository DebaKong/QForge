"""INT8 显式量化（Q/DQ）：把 ONNX 转成带 QuantizeLinear / DequantizeLinear 的模型。

为什么要改用 Q/DQ（SPEC 4 / 9 的 INT8 路径升级）
------------------------------------------------
- TensorRT 10.1 起 `IInt8Calibrator` 被标记为 **deprecated**，官方推荐**显式量化（Q/DQ）**；
- Q/DQ 把量化参数**固化在图里**：Engine 构建不再依赖运行期校准回调，构建更可复现，
  量化后的模型也能交给别的工具链（或人工）检查；
- 本平台用 onnxruntime 的静态量化完成「校准集 → 激活尺度 → QDQ 图」，校准方法可配置
  （MinMax / Entropy / Percentile），并且与分层误差分析（SPEC 9.2）复用**同一套输入与实现**，
  因此两处的结论天然可比。

边界（如实记录，不含糊）
------------------------
- **当前状态（实测）**：Q/DQ 图能正常生成（含 QuantizeLinear / DequantizeLinear 节点），但
  TensorRT 10.16 的 ONNX Parser **仍拒绝** onnxruntime 生成的 Q/DQ 图：
  已修掉"激活非对称（zero point ≠ 0）"这一条（改为 `ActivationSymmetric=True`），
  随后卡在 Conv bias 的 `DequantizeLinear` 节点（`INVALID_NODE`）。
  因此**任务默认仍走 `quantization.mode=calibrator`**（真实 INT8 任务已验证可用），
  `mode=qdq` 作为显式可选项保留，失败时如实抛 `QuantizationFailedError`，不静默回退。
- QDQ 图由 onnxruntime 生成，其激活尺度算法与 TensorRT 熵校准**不保证完全一致**；
  平台保留两条路径，端到端精度都会写在报告里供对比。
- 量化失败必须显式失败（`QuantizationFailedError`），不静默退回旧路径——否则报告会失真。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("qforge.quantization")

# 可配置的校准方法（对应 onnxruntime 的 CalibrationMethod）
CALIBRATION_METHODS: tuple[str, ...] = ("minmax", "entropy", "percentile")
DEFAULT_METHOD = "minmax"
# 参与量化的输入样本上限（ORT 会逐样本统计激活范围）
MAX_CALIBRATION_SAMPLES = 64


@dataclass
class QdqResult:
    """Q/DQ 量化结果（直接落盘进 report/quantization.json）。"""

    status: str  # SUCCESS / BLOCKED
    reason: str = ""
    path: str | None = None
    method: str = DEFAULT_METHOD
    per_channel: bool = False
    weight_type: str = "QInt8"
    activation_type: str = "QInt8"
    samples: int = 0
    quantize_nodes: int = 0
    dequantize_nodes: int = 0
    total_nodes: int = 0
    size_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "path": self.path,
            "method": self.method,
            "per_channel": self.per_channel,
            "weight_type": self.weight_type,
            "activation_type": self.activation_type,
            "samples": self.samples,
            "quantize_nodes": self.quantize_nodes,
            "dequantize_nodes": self.dequantize_nodes,
            "total_nodes": self.total_nodes,
            "size_bytes": self.size_bytes,
            "notes": self.notes,
        }


class _ListCalibrationReader:
    """把预处理好的张量逐个喂给 onnxruntime 的静态量化器。"""

    def __init__(self, input_name: str, samples: list[np.ndarray]) -> None:
        self._input_name = input_name
        self._samples = samples
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._samples):
            return None
        sample = self._samples[self._index]
        self._index += 1
        return {self._input_name: sample}

    def rewind(self) -> None:
        self._index = 0


def _calibration_method(name: str) -> Any:
    from onnxruntime.quantization import CalibrationMethod

    mapping = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    return mapping.get(name.lower(), CalibrationMethod.MinMax)


def count_qdq_nodes(onnx_path: Path) -> tuple[int, int, int]:
    """统计 Q/DQ 节点与总节点数（用于证明「量化真的发生了」）。"""
    import onnx

    model = onnx.load(str(onnx_path))
    quantize = 0
    dequantize = 0
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            quantize += 1
        elif node.op_type == "DequantizeLinear":
            dequantize += 1
    return quantize, dequantize, len(model.graph.node)


def quantize(
    onnx_path: Path,
    output_path: Path,
    samples: list[np.ndarray],
    *,
    input_name: str,
    method: str = DEFAULT_METHOD,
    per_channel: bool = False,
) -> QdqResult:
    """把 ONNX 量化成 Q/DQ 模型；失败时返回 `BLOCKED` 结果（由调用方决定是否失败任务）。"""
    result = QdqResult(status="BLOCKED", method=method.lower(), per_channel=per_channel)

    if not samples:
        result.reason = "没有可用的校准输入样本"
        return result
    if method.lower() not in CALIBRATION_METHODS:
        result.reason = f"不支持的校准方法：{method}（可选 {list(CALIBRATION_METHODS)}）"
        return result

    used = [np.asarray(item, dtype=np.float32) for item in samples[:MAX_CALIBRATION_SAMPLES]]
    result.samples = len(used)

    try:
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except Exception as exc:  # noqa: BLE001 - 依赖缺失按 BLOCKED 上报
        result.reason = f"onnxruntime.quantization 不可用：{exc}"
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        quantize_static(
            str(onnx_path),
            str(output_path),
            _ListCalibrationReader(input_name, used),
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=_calibration_method(method),
            # 关键：激活必须是**对称**量化（zero point = 0）。
            # 实测教训：ORT 默认可能给出非零 zero point，TensorRT 解析时会直接报
            # "Non-zero zero point is not supported"（只在 DLA 上才支持非对称量化）。
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
        )
    except Exception as exc:  # noqa: BLE001 - 量化失败必须显式暴露
        result.reason = f"Q/DQ 量化失败（{type(exc).__name__}）：{exc}"
        return result

    try:
        quantize_nodes, dequantize_nodes, total_nodes = count_qdq_nodes(output_path)
    except Exception as exc:  # noqa: BLE001
        result.reason = f"量化产物无法读取：{exc}"
        return result

    result.status = "SUCCESS"
    result.path = output_path.name
    result.quantize_nodes = quantize_nodes
    result.dequantize_nodes = dequantize_nodes
    result.total_nodes = total_nodes
    result.size_bytes = output_path.stat().st_size
    result.notes = [
        "Q/DQ 图由 onnxruntime 静态量化生成，激活尺度固化在图中，Engine 构建不再依赖校准回调",
        "与 TensorRT 熵校准（旧路径）不保证数值完全一致；两条路径的端到端精度都会写入报告",
    ]
    logger.info(
        "INT8 显式量化完成：%s 个 QuantizeLinear / %s 个 DequantizeLinear（共 %s 节点，方法 %s）",
        quantize_nodes,
        dequantize_nodes,
        total_nodes,
        result.method,
    )
    return result


def select_model_source(
    precision: str, onnx_path: Path, quantized_onnx_path: Path | None
) -> Path:
    """引擎构建要解析哪个模型：INT8 + 有 Q/DQ 产物时用 Q/DQ 模型，其余用原始 ONNX。

    抽成纯函数是为了能脱离 TensorRT 单测这条"该用哪张图"的决策逻辑。
    """
    if precision == "int8" and quantized_onnx_path is not None and quantized_onnx_path.exists():
        return quantized_onnx_path
    return onnx_path

"""ONNX 模型校验与图分析（SPEC 7.1 校验层级 1~4）。

对应关系：
1. 文件格式校验 — 由 uploads.save_upload 完成（扩展名/大小/可读性）；
2. ONNX Checker   — `onnx.checker.check_model`；
3. Runtime Load   — `onnxruntime.InferenceSession` 实际加载并提取输入输出；
4. Graph Analysis — 解析节点、算子、opset、动态维度与数据类型。

输出为结构化字典，可直接落库到 `models.input_spec / output_spec`，
并写入产物报告 `report/model_info.json`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

from app.errors import ModelInvalidError, ModelLoadFailedError

logger = logging.getLogger(__name__)


def _dtype_name(dtype: int) -> str:
    try:
        return str(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(dtype)))
    except Exception:  # pragma: no cover - 极少数自定义 dtype
        try:
            return TensorProto.DataType.Name(dtype)
        except Exception:
            return f"unknown({dtype})"


def _shape_of(value_info: onnx.ValueInfoProto) -> list[Any]:
    tensor_type = value_info.type.tensor_type
    shape: list[Any] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


def _is_dynamic(shape: list[Any]) -> bool:
    return any((not isinstance(dim, int)) or dim <= 0 for dim in shape)


@dataclass
class OnnxInspection:
    """校验结果汇总。"""

    checker_passed: bool = False
    runtime_load_passed: bool = False
    opset: int | None = None
    ir_version: int | None = None
    producer: str | None = None
    graph_name: str | None = None
    node_count: int = 0
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    operators: list[dict[str, Any]] = field(default_factory=list)
    dynamic_inputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checker_passed": self.checker_passed,
            "runtime_load_passed": self.runtime_load_passed,
            "opset": self.opset,
            "ir_version": self.ir_version,
            "producer": self.producer,
            "graph_name": self.graph_name,
            "node_count": self.node_count,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "operators": self.operators,
            "dynamic_inputs": self.dynamic_inputs,
            "warnings": self.warnings,
        }


def inspect(
    onnx_path: Path,
    *,
    total_operator_limit: int = 200,
    require_static_batch: bool = True,
) -> OnnxInspection:
    """执行完整校验链；任一硬性检查失败即抛领域异常（结构化错误码）。"""
    if not onnx_path.exists():
        raise ModelInvalidError("ONNX 文件不存在", detail={"path": str(onnx_path)})

    try:
        model = onnx.load(str(onnx_path))
    except Exception as exc:
        raise ModelInvalidError(
            "ONNX 文件无法解析", detail={"error": f"{type(exc).__name__}: {exc}"}
        ) from exc

    result = OnnxInspection()

    # ---- 2. ONNX Checker ----
    try:
        onnx.checker.check_model(model)
        result.checker_passed = True
    except Exception as exc:
        raise ModelInvalidError(
            "ONNX Checker 未通过",
            detail={"error": f"{type(exc).__name__}: {exc}"},
        ) from exc

    graph = model.graph
    result.ir_version = int(model.ir_version)
    result.producer = f"{model.producer_name} {model.producer_version}".strip() or None
    result.graph_name = graph.name or None
    result.node_count = len(graph.node)

    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            result.opset = int(opset.version)

    # ---- 4. Graph Analysis ----
    result.inputs = [
        {
            "name": item.name,
            "shape": _shape_of(item),
            "dtype": _dtype_name(item.type.tensor_type.elem_type),
        }
        for item in graph.input
    ]
    result.outputs = [
        {
            "name": item.name,
            "shape": _shape_of(item),
            "dtype": _dtype_name(item.type.tensor_type.elem_type),
        }
        for item in graph.output
    ]

    counters: dict[tuple[str, str], int] = {}
    for node in graph.node:
        domain = node.domain or "ai.onnx"
        key = (node.op_type, domain)
        counters[key] = counters.get(key, 0) + 1

    result.operators = [
        {"operator": op_type, "domain": domain, "count": count}
        for (op_type, domain), count in sorted(
            counters.items(), key=lambda item: (-item[1], item[0][0])
        )
    ]
    if len(result.operators) > total_operator_limit:  # pragma: no cover - 极端模型
        result.warnings.append(
            f"算子种类数 {len(result.operators)} 超过报告上限 {total_operator_limit}，已截断"
        )
        result.operators = result.operators[:total_operator_limit]

    for item in result.inputs:
        if _is_dynamic(item["shape"]):
            result.dynamic_inputs.append(item["name"])
    if result.dynamic_inputs and require_static_batch:
        # SPEC 10.1：MVP 默认 batch=1 且固定 shape，动态维度需在阶段 2 处理
        result.warnings.append(
            "存在动态维度输入：" + ", ".join(result.dynamic_inputs) + "（MVP 按固定 shape 处理）"
        )

    # ---- 3. Runtime Load Test ----
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ModelLoadFailedError(
            "onnxruntime 无法加载该模型",
            detail={"error": f"{type(exc).__name__}: {exc}"},
        ) from exc

    runtime_inputs = [{"name": item.name, "shape": item.shape, "dtype": item.type} for item in session.get_inputs()]
    runtime_outputs = [
        {"name": item.name, "shape": item.shape, "dtype": item.type} for item in session.get_outputs()
    ]
    result.runtime_load_passed = True

    # 以 runtime 的结论为准补充/校正 shape（ORT 会把符号维度展开为字符串）
    for declared, runtime in zip(result.inputs, runtime_inputs, strict=False):
        if declared["name"] != runtime["name"]:  # pragma: no cover - 非典型模型
            result.warnings.append(
                f"图定义与 runtime 的输入名不一致：{declared['name']} vs {runtime['name']}"
            )
    for declared, runtime in zip(result.outputs, runtime_outputs, strict=False):
        if declared["name"] != runtime["name"]:  # pragma: no cover
            result.warnings.append(
                f"图定义与 runtime 的输出名不一致：{declared['name']} vs {runtime['name']}"
            )

    logger.info(
        "ONNX 校验通过",
        extra={
            "opset": result.opset,
            "nodes": result.node_count,
            "inputs": len(result.inputs),
            "outputs": len(result.outputs),
        },
    )
    return result


__all__ = [
    "OnnxInspection",
    "inspect",
]

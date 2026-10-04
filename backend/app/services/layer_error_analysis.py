"""分层/逐算子误差分析（SPEC 9.2 量化误差分析）。

目的与做法
----------
SPEC 9.2 要的是「定位**敏感层**」：哪些层在量化后误差最大，将来做混合精度（这些层保留 FP16）
就看这份排序。做法是在 **ONNX 图层面**做逐层对比：

1. 把选定节点的输出**临时追加为图输出**（instrument），其余结构与权重完全不变；
2. 用 onnxruntime 跑 **FP32 基线**；
3. 用 onnxruntime 对**同一张图**做 INT8 静态量化（校准集就是本任务实际用的校准数据），再跑一次；
4. 逐节点计算 MAE / RMSE / Cosine 相似度 / 最大绝对误差，并按**相对 RMSE** 排序。

边界（必须如实写进报告，不能含糊）
--------------------------------
- 分层分析在 **ONNX + onnxruntime** 层面进行，用于定位敏感层；它**不是** TensorRT Engine 的逐层数据
  （TRT 运行期不暴露中间张量，要拿就得在 build 时 markOutput 重建引擎，代价与风险都不合适）；
- **端到端精度**仍以真实 Engine 与 FP32 基准的对比为准（`report/accuracy.json`）；
- 两者结论不一致时以端到端为准，并在报告里保留双方数据供人判断。
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 收集的中间张量下限/上限：太小的张量（标量、索引）没有误差分析价值
MIN_ELEMENTS = 64
# 相对 RMSE 排名前 N 名进入"敏感层"清单
TOP_SENSITIVE = 20
# 默认最多分析多少个节点（大模型逐层跑会明显变慢）
DEFAULT_MAX_LAYERS = 120
# 单个张量元素上限（超过就不纳入，避免显存/内存爆炸）
MAX_ELEMENTS_PER_TENSOR = 4_000_000
# 参与统计的输入样本数上限（取前 N 个校准输入）
DEFAULT_MAX_INPUTS = 4


@dataclass
class LayerErrorAnalysis:
    """分层误差分析结果（可直接落盘成 JSON）。"""

    status: str  # SUCCESS / BLOCKED / SKIPPED
    method: str = "onnxruntime FP32 基线 vs onnxruntime INT8 静态量化（同一张 ONNX 图，逐节点输出对比）"
    reason: str = ""
    analyzed_layers: int = 0
    requested_layers: int = 0
    inputs_used: int = 0
    input_shape: list[int] = field(default_factory=list)
    metrics: list[str] = field(default_factory=lambda: ["mae", "rmse", "cosine_similarity", "max_abs_error"])
    ranking: list[dict[str, Any]] = field(default_factory=list)
    layers: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "reason": self.reason,
            "analyzed_layers": self.analyzed_layers,
            "requested_layers": self.requested_layers,
            "inputs_used": self.inputs_used,
            "input_shape": self.input_shape,
            "metrics": self.metrics,
            "ranking": self.ranking,
            "layers": self.layers,
            "notes": self.notes,
        }


def compare_arrays(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """逐张量误差指标（SPEC 9.2 要求的 MAE / MSE / RMSE / Cosine）。"""
    reference_flat = np.asarray(reference, dtype=np.float64).ravel()
    actual_flat = np.asarray(actual, dtype=np.float64).ravel()
    if reference_flat.size != actual_flat.size:
        raise ValueError(
            f"张量元素数量不一致：baseline={reference_flat.size} actual={actual_flat.size}"
        )

    difference = actual_flat - reference_flat
    baseline_abs_max = float(np.max(np.abs(reference_flat))) if reference_flat.size else 0.0
    rmse = float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(actual_flat))
    cosine = float(np.dot(reference_flat, actual_flat) / denominator) if denominator else 1.0
    return {
        "elements": int(reference_flat.size),
        "mae": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "mse": float(np.mean(difference**2)) if difference.size else 0.0,
        "rmse": rmse,
        "max_abs_error": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "cosine_similarity": cosine,
        "baseline_abs_max": baseline_abs_max,
        # 相对 RMSE：跨层比较用（不同层量纲差别很大，绝对误差排不出真正的敏感层）
        "relative_rmse": rmse / baseline_abs_max if baseline_abs_max > 0 else rmse,
    }


def _float_tensor_elements(value_info: Any) -> int | None:
    """返回静态 float 张量的元素数；不是静态 float 张量则返回 None。"""
    try:
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type != 1:  # TensorProto.FLOAT
            return None
        dims = tensor_type.shape.dim
        if not dims:
            return None
        elements = 1
        for dim in dims:
            if not dim.HasField("dim_value"):
                return None  # 动态维：无法确定大小
            elements *= int(dim.dim_value)
        return elements
    except Exception:  # noqa: BLE001 - 形状信息缺失按不可用处理
        return None


def select_candidate_layers(
    model: Any, *, max_layers: int = DEFAULT_MAX_LAYERS
) -> list[tuple[str, str, int]]:
    """挑选要观测的中间张量：`[(tensor_name, op_type, node_index), ...]`。

    规则：静态 float 张量、元素数在 [MIN_ELEMENTS, MAX_ELEMENTS_PER_TENSOR] 之间，
    超过上限时**按图顺序均匀抽样**（而不是只取前 N 个，避免只看前半张图）。
    """
    import onnx
    from onnx import shape_inference

    inferred = shape_inference.infer_shapes(model)
    value_info: dict[str, Any] = {}
    for collection in (inferred.graph.value_info, inferred.graph.output, inferred.graph.input):
        for item in collection:
            value_info[item.name] = item

    candidates: list[tuple[str, str, int]] = []
    for index, node in enumerate(model.graph.node):
        for output_name in node.output:
            if not output_name:
                continue
            info = value_info.get(output_name)
            if info is None:
                continue
            elements = _float_tensor_elements(info)
            if elements is None or elements < MIN_ELEMENTS or elements > MAX_ELEMENTS_PER_TENSOR:
                continue
            candidates.append((output_name, node.op_type, index))

    if len(candidates) <= max_layers:
        return candidates

    step = len(candidates) / float(max_layers)
    sampled = [candidates[int(index * step)] for index in range(max_layers)]
    logger.info("中间张量候选 %s 个，均匀抽样为 %s 个", len(candidates), len(sampled))
    return sampled


def _instrument_model(model: Any, tensors: list[str]) -> Any:
    """把待观测张量追加为图输出（复制 ValueInfo，不改变原有输出与算子）。"""
    import onnx
    from onnx import shape_inference

    inferred = shape_inference.infer_shapes(model)
    by_name = {item.name: item for item in inferred.graph.value_info}
    by_name.update({item.name: item for item in model.graph.output})

    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    existing = {item.name for item in clone.graph.output}
    for name in tensors:
        if name in existing:
            continue
        info = by_name.get(name)
        if info is None:
            continue
        clone.graph.output.append(info)
    return clone


class _ListCalibrationReader:
    """给 onnxruntime 静态量化用的校准数据读取器（顺序喂入已预处理好的张量）。"""

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


def _run_session(session: Any, input_name: str, samples: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
    """跑若干输入，返回 `{输出名: [每个输入一个数组]}`。"""
    collected: dict[str, list[np.ndarray]] = {}
    output_names = [item.name for item in session.get_outputs()]
    for sample in samples:
        outputs = session.run(output_names, {input_name: sample})
        for name, array in zip(output_names, outputs, strict=True):
            collected.setdefault(name, []).append(np.asarray(array))
    return collected


def analyze(
    onnx_path: Path,
    samples: list[np.ndarray],
    *,
    input_name: str,
    max_layers: int = DEFAULT_MAX_LAYERS,
    max_inputs: int = DEFAULT_MAX_INPUTS,
) -> LayerErrorAnalysis:
    """执行分层误差分析。

    任何一步不可用（onnxruntime 缺失、量化失败、模型结构异常）都返回 `BLOCKED` 并写清原因，
    **不抛异常**：它是"质量分析"，不该把已经跑通的任务判失败。
    """
    if not samples:
        return LayerErrorAnalysis(status="SKIPPED", reason="没有可用的输入样本（校准集或样例输入）")

    used_samples = [np.asarray(item, dtype=np.float32) for item in samples[:max_inputs]]
    result = LayerErrorAnalysis(
        status="SUCCESS",
        inputs_used=len(used_samples),
        input_shape=list(used_samples[0].shape),
    )

    try:
        import onnx
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001 - 依赖缺失按 BLOCKED 上报
        result.status = "BLOCKED"
        result.reason = f"onnxruntime/onnx 不可用：{exc}"
        return result

    try:
        model = onnx.load(str(onnx_path))
    except Exception as exc:  # noqa: BLE001
        result.status = "BLOCKED"
        result.reason = f"ONNX 加载失败：{exc}"
        return result

    candidates = select_candidate_layers(model, max_layers=max_layers)
    result.requested_layers = len(candidates)
    if not candidates:
        result.status = "SKIPPED"
        result.reason = "模型中没有可观测的静态 float 中间张量（例如全是动态形状或只有极小张量）"
        return result

    names = [item[0] for item in candidates]
    try:
        instrumented = _instrument_model(model, names)
    except Exception as exc:  # noqa: BLE001
        result.status = "BLOCKED"
        result.reason = f"为中间张量追加图输出失败：{exc}"
        return result

    with tempfile.TemporaryDirectory(prefix="qforge-layer-analysis-") as workdir:
        work = Path(workdir)
        fp32_model = work / "fp32_instrumented.onnx"
        int8_model = work / "int8_instrumented.onnx"
        onnx.save(instrumented, str(fp32_model))

        try:
            baseline_session = ort.InferenceSession(
                str(fp32_model), providers=["CPUExecutionProvider"]
            )
            baseline = _run_session(baseline_session, input_name, used_samples)
        except Exception as exc:  # noqa: BLE001
            result.status = "BLOCKED"
            result.reason = f"FP32 基线推理失败：{exc}"
            return result

        try:
            from onnxruntime.quantization import (
                CalibrationMethod,
                QuantFormat,
                QuantType,
                quantize_static,
            )

            reader = _ListCalibrationReader(input_name, used_samples)
            quantize_static(
                str(fp32_model),
                str(int8_model),
                reader,
                quant_format=QuantFormat.QDQ,
                per_channel=False,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
            )
            quantized_session = ort.InferenceSession(
                str(int8_model), providers=["CPUExecutionProvider"]
            )
            quantized = _run_session(quantized_session, input_name, used_samples)
        except Exception as exc:  # noqa: BLE001 - 量化失败不阻断任务
            result.status = "BLOCKED"
            result.reason = f"INT8 静态量化失败（{type(exc).__name__}）：{exc}"
            result.notes.append("分层分析失败不影响任务结论：端到端精度仍以真实 Engine 对比为准")
            return result

    layers: list[dict[str, Any]] = []
    for tensor_name, op_type, node_index in candidates:
        reference_list = baseline.get(tensor_name)
        actual_list = quantized.get(tensor_name)
        if not reference_list or not actual_list:
            continue
        # 多个输入样本：先各自算指标再平均，避免把不同样本拼成一个大张量（会掩盖单样本退化）
        per_sample = []
        for reference, actual in zip(reference_list, actual_list, strict=True):
            if reference.shape != actual.shape:
                per_sample = []
                break
            per_sample.append(compare_arrays(reference, actual))
        if not per_sample:
            continue
        averaged = {
            key: float(np.mean([item[key] for item in per_sample]))
            for key in per_sample[0]
        }
        layers.append(
            {
                "tensor": tensor_name,
                "operator": op_type,
                "node_index": node_index,
                "samples": len(per_sample),
                **{key: value for key, value in averaged.items() if key != "elements"},
                "elements": int(per_sample[0]["elements"]),
            }
        )

    if not layers:
        result.status = "BLOCKED"
        result.reason = "没有成功匹配到任何中间张量（FP32 与 INT8 图输出名不一致）"
        return result

    layers.sort(key=lambda item: float(item["relative_rmse"]), reverse=True)
    result.layers = layers
    result.analyzed_layers = len(layers)
    result.ranking = [
        {
            "rank": index + 1,
            "tensor": item["tensor"],
            "operator": item["operator"],
            "relative_rmse": item["relative_rmse"],
            "rmse": item["rmse"],
            "mae": item["mae"],
            "cosine_similarity": item["cosine_similarity"],
            "elements": item["elements"],
        }
        for index, item in enumerate(layers[:TOP_SENSITIVE])
    ]
    result.notes.append(
        "分层分析在 ONNX/onnxruntime 层面进行（用于定位敏感层）；"
        "端到端精度以真实 Engine 与 FP32 基准的对比为准"
    )
    result.notes.append(f"按相对 RMSE 降序取前 {len(result.ranking)} 层作为敏感层候选（SPEC 9.2）")
    return result

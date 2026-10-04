"""INT8 显式量化（Q/DQ）测试：选图逻辑、量化产物、失败路径（纯 CPU）。"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from app.adapters.base import BackendAdapter
from app.services import qdq_quantization
from tools.synth_model import build_yolov8_like_onnx


def _onnx(tmp_path: Path, size: int = 64) -> Path:
    path = tmp_path / "synthetic.onnx"
    build_yolov8_like_onnx(path, input_size=size)
    return path


def _samples(count: int = 3, size: int = 64) -> list[np.ndarray]:
    rng = np.random.default_rng(seed=5)
    return [rng.random((1, 3, size, size), dtype=np.float32) for _ in range(count)]


def test_select_model_source_prefers_qdq_for_int8(tmp_path: Path) -> None:
    onnx_path = _onnx(tmp_path)
    qdq_path = tmp_path / "model_qdq.onnx"
    qdq_path.write_bytes(b"placeholder")

    assert qdq_quantization.select_model_source("int8", onnx_path, qdq_path) == qdq_path
    # 其他精度不受影响：即使存在 Q/DQ 产物也用原始 ONNX
    assert qdq_quantization.select_model_source("fp16", onnx_path, qdq_path) == onnx_path
    assert qdq_quantization.select_model_source("fp32", onnx_path, qdq_path) == onnx_path
    # INT8 但没有 Q/DQ 产物 → 回退到原始模型（旧校准路径）
    assert qdq_quantization.select_model_source("int8", onnx_path, None) == onnx_path
    # 产物路径存在但文件不在 → 同样回退
    assert (
        qdq_quantization.select_model_source("int8", onnx_path, tmp_path / "missing.onnx")
        == onnx_path
    )


def test_quantize_produces_qdq_graph(tmp_path: Path) -> None:
    """真实量化一遍：产物必须真的含 QuantizeLinear / DequantizeLinear 节点。"""
    onnx_path = _onnx(tmp_path)
    output = tmp_path / "model_qdq.onnx"

    result = qdq_quantization.quantize(
        onnx_path, output, _samples(), input_name="images", method="minmax"
    )

    assert result.status == "SUCCESS", result.reason
    assert output.is_file() and result.size_bytes > 0
    assert result.quantize_nodes > 0, "没有 QuantizeLinear 说明量化根本没发生"
    assert result.dequantize_nodes > 0
    assert result.total_nodes > 0
    assert result.samples == 3
    assert result.weight_type == "QInt8" and result.activation_type == "QInt8"
    assert result.notes, "要把'与 TRT 熵校准不保证一致'的边界写进报告"

    quantize_nodes, dequantize_nodes, _ = qdq_quantization.count_qdq_nodes(output)
    assert (quantize_nodes, dequantize_nodes) == (result.quantize_nodes, result.dequantize_nodes)


def test_quantize_without_samples_is_blocked(tmp_path: Path) -> None:
    result = qdq_quantization.quantize(
        _onnx(tmp_path), tmp_path / "out.onnx", [], input_name="images"
    )
    assert result.status == "BLOCKED"
    assert "校准输入样本" in result.reason


def test_quantize_with_unknown_method_is_blocked(tmp_path: Path) -> None:
    result = qdq_quantization.quantize(
        _onnx(tmp_path),
        tmp_path / "out.onnx",
        _samples(1),
        input_name="images",
        method="magic",
    )
    assert result.status == "BLOCKED"
    assert "不支持的校准方法" in result.reason


def test_quantize_failure_is_reported_not_swallowed(tmp_path: Path, monkeypatch) -> None:
    """量化失败必须显式返回 BLOCKED（调用方据此失败任务），不能悄悄退回旧路径。"""
    import onnxruntime.quantization as ort_quant

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("模拟量化器崩溃")

    monkeypatch.setattr(ort_quant, "quantize_static", _boom)

    result = qdq_quantization.quantize(
        _onnx(tmp_path), tmp_path / "out.onnx", _samples(1), input_name="images"
    )

    assert result.status == "BLOCKED"
    assert "RuntimeError" in result.reason and "模拟量化器崩溃" in result.reason


def test_calibration_methods_are_whitelisted() -> None:
    assert set(qdq_quantization.CALIBRATION_METHODS) == {"minmax", "entropy", "percentile"}
    assert qdq_quantization.DEFAULT_METHOD in qdq_quantization.CALIBRATION_METHODS
    # 方法名到 ORT 枚举的映射必须每项都能取到（避免写错关键字只在运行期才炸）
    for method in qdq_quantization.CALIBRATION_METHODS:
        assert qdq_quantization._calibration_method(method) is not None  # noqa: SLF001


def test_backend_adapter_accepts_quantized_onnx_path() -> None:
    """接口层必须显式接受 Q/DQ 产物，且默认值保持向后兼容（None）。"""
    signature = inspect.signature(BackendAdapter.build_engine)
    parameter = signature.parameters["quantized_onnx_path"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_entropy_and_percentile_methods_run(tmp_path: Path) -> None:
    """可配置指标：换校准方法也要能跑通（SPEC 9.2 的“可配置”要求）。"""
    for method in ("entropy", "percentile"):
        result = qdq_quantization.quantize(
            _onnx(tmp_path),
            tmp_path / f"model_{method}.onnx",
            _samples(2),
            input_name="images",
            method=method,
        )
        assert result.status == "SUCCESS", f"{method}: {result.reason}"
        assert result.method == method
        assert result.quantize_nodes > 0


@pytest.mark.parametrize("per_channel", [False, True])
def test_per_channel_option_is_recorded(tmp_path: Path, per_channel: bool) -> None:
    result = qdq_quantization.quantize(
        _onnx(tmp_path),
        tmp_path / "model.onnx",
        _samples(1),
        input_name="images",
        per_channel=per_channel,
    )
    assert result.status == "SUCCESS", result.reason
    assert result.per_channel is per_channel

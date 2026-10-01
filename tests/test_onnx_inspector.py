"""ONNX 校验链测试（SPEC 7.1）。"""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from app.errors import ModelInvalidError, ModelLoadFailedError
from app.services import onnx_inspector
from tools.synth_model import build_yolov8_like_onnx


def test_inspect_synth_model_passes_full_chain(tmp_path: Path) -> None:
    path = build_yolov8_like_onnx(tmp_path / "synth.onnx", input_size=64)
    result = onnx_inspector.inspect(path)

    assert result.checker_passed is True
    assert result.runtime_load_passed is True
    assert result.opset == 17
    assert result.inputs[0]["name"] == "images"
    assert result.inputs[0]["shape"] == [1, 3, 64, 64]
    assert result.outputs[0]["name"] == "output0"
    assert result.outputs[0]["shape"] == [1, 84, 256]

    operators = {item["operator"] for item in result.operators}
    assert {"Conv", "Reshape"} <= operators
    assert result.node_count == 2
    assert result.dynamic_inputs == []


def test_inspect_reports_dynamic_input_as_warning(tmp_path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"], name="identity")],
        "dynamic",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, ["batch", 3, 64, 64]
            )
        ],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 3, 64, 64])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "dynamic.onnx"
    onnx.save(model, str(path))

    result = onnx_inspector.inspect(path)
    assert result.dynamic_inputs == ["images"]
    assert any("动态维度" in item for item in result.warnings)


def test_inspect_rejects_unparsable_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.onnx"
    path.write_bytes(b"definitely not protobuf")
    with pytest.raises(ModelInvalidError) as excinfo:
        onnx_inspector.inspect(path)
    assert excinfo.value.code.value == "MODEL_INVALID"


def test_inspect_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ModelInvalidError):
        onnx_inspector.inspect(tmp_path / "absent.onnx")


def test_inspect_rejects_broken_graph(tmp_path: Path) -> None:
    # 缺少 initializer 的 Conv：结构不完整，checker 应拒绝
    graph = helper.make_graph(
        [helper.make_node("Conv", ["images", "w", "b"], ["output0"], name="conv")],
        "broken",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 16, 16])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "broken.onnx"
    onnx.save(model, str(path))

    with pytest.raises((ModelInvalidError, ModelLoadFailedError)):
        onnx_inspector.inspect(path)

"""语义分割适配器（SPEC 2.2 / 6）：定义、校验、解码、任务类型放行（纯 CPU）。"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.adapters.base import DecodeResult
from app.adapters.definition import ModelDefinition
from app.adapters.models.unet import UNetAdapter
from app.adapters.registry import get_model_adapter, supported_architectures
from app.errors import PreprocessConfigError, ValidationFailedError
from app.schemas.project import ProjectCreate
from app.services import catalog, onnx_inspector
from app.services.task_state import TaskStatus
from tools.synth_segmentation import build_segmentation_onnx, reference_mask


def _inspection(tmp_path: Path, *, class_count: int = 3, size: int = 32) -> dict:
    onnx_path = build_segmentation_onnx(
        tmp_path / "seg.onnx", input_size=size, class_count=class_count
    )
    return onnx_inspector.inspect(onnx_path).to_dict()


def _definition(tmp_path: Path, **kwargs) -> ModelDefinition:
    inspection = _inspection(tmp_path, **kwargs)
    adapter = UNetAdapter()
    return ModelDefinition.parse(adapter.default_definition(inspection))


def test_adapter_is_registered() -> None:
    assert "unet" in supported_architectures()
    adapter = get_model_adapter("unet")
    assert adapter.task == "segmentation"


def test_default_definition_matches_model(tmp_path: Path) -> None:
    inspection = _inspection(tmp_path, class_count=4, size=32)
    payload = UNetAdapter().default_definition(inspection)
    definition = ModelDefinition.parse(payload)

    assert definition.task == "segmentation"
    assert definition.architecture == "unet"
    assert definition.postprocessing.decoder == "argmax"
    assert definition.postprocessing.ignore_index == 255
    # 分割输出的通道维就是类别数（由架构约定取值，不是从 shape 猜后处理语义）
    assert definition.output.class_count == 4
    assert definition.input.shape[0] == 1
    assert definition.preprocessing.resize == "stretch"


def test_validate_accepts_matching_definition(tmp_path: Path) -> None:
    inspection = _inspection(tmp_path, class_count=3)
    definition = _definition(tmp_path, class_count=3)

    warnings = UNetAdapter().validate(definition, inspection)

    assert warnings == []


def test_validate_rejects_wrong_task(tmp_path: Path) -> None:
    definition = _definition(tmp_path)
    broken = ModelDefinition.parse({**definition.to_dict(), "task": "detection"})

    with pytest.raises(ValidationFailedError) as error:
        UNetAdapter().validate(broken)
    assert "segmentation" in str(error.value)


def test_validate_rejects_detection_decoder(tmp_path: Path) -> None:
    """检测的 decoder=yolov8 用在分割任务上必须被拒绝（后处理语义完全不同）。"""
    definition = _definition(tmp_path)
    payload = definition.to_dict()
    payload["postprocessing"] = {"decoder": "yolov8"}

    with pytest.raises(PreprocessConfigError):
        UNetAdapter().validate(ModelDefinition.parse(payload))


def test_validate_warns_on_channel_mismatch(tmp_path: Path) -> None:
    inspection = _inspection(tmp_path, class_count=3)
    definition = _definition(tmp_path, class_count=3)
    payload = definition.to_dict()
    payload["output"]["class_count"] = 2  # 声明 2 类但模型输出 3 通道

    warnings = UNetAdapter().validate(ModelDefinition.parse(payload), inspection)

    assert any("通道数" in warning for warning in warnings)


def test_validate_warns_on_letterbox(tmp_path: Path) -> None:
    definition = _definition(tmp_path)
    payload = definition.to_dict()
    payload["preprocessing"]["resize"] = "letterbox"

    warnings = UNetAdapter().validate(ModelDefinition.parse(payload))

    assert any("letterbox" in warning for warning in warnings)


def test_non_4d_input_is_rejected_by_schema(tmp_path: Path) -> None:
    """非 4D 输入在**解析 Model Definition 时**就被拒绝（比适配器校验更早，更安全）。"""
    definition = _definition(tmp_path)
    payload = definition.to_dict()
    payload["input"]["shape"] = [1, 32, 32]

    with pytest.raises(PreprocessConfigError):
        ModelDefinition.parse(payload)


def test_decode_produces_mask_statistics(tmp_path: Path) -> None:
    """解码必须给出掩膜与类别直方图（供运行验证/精度报告使用）。"""
    import onnxruntime as ort

    onnx_path = build_segmentation_onnx(tmp_path / "seg.onnx", input_size=32, class_count=3)
    inspection = onnx_inspector.inspect(onnx_path).to_dict()
    adapter = UNetAdapter()
    definition = ModelDefinition.parse(adapter.default_definition(inspection))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: np.zeros((1, 3, 32, 32), dtype=np.float32)})[0]

    result = adapter.decode([logits], definition)

    assert isinstance(result, DecodeResult)
    assert result.detections == []  # 分割不出框
    assert tuple(result.raw_shape) == (1, 3, 32, 32)
    summary = result.summary()
    assert summary["mask_shape"] == [1, 32, 32]
    assert summary["class_count"] == 3
    assert sum(summary["class_histogram"].values()) == 32 * 32
    assert summary["decoder"] == "argmax"


def test_decode_rejects_empty_outputs(tmp_path: Path) -> None:
    definition = _definition(tmp_path)
    with pytest.raises(ValidationFailedError):
        UNetAdapter().decode([], definition)


def test_decode_result_summary_keeps_detection_fields() -> None:
    """接口扩展必须向后兼容：检测路径的 summary() 字段一个都不能少。"""
    summary = DecodeResult(raw_shape=(1, 84, 8400), candidates=7).summary()

    assert summary["raw_shape"] == [1, 84, 8400]
    assert summary["candidates"] == 7
    assert summary["detections"] == 0
    assert summary["preview"] == []


def test_segmentation_task_type_is_accepted(
    client: TestClient, db_session, storage_root: Path
) -> None:
    """任务类型白名单必须放行 segmentation（否则整个分割链路进不来）。"""
    project = catalog.create_project(
        session=db_session,
        data=ProjectCreate(name=f"seg-{uuid.uuid4().hex[:8]}", description="分割任务用例"),
    )
    db_session.commit()

    response = client.post(
        "/api/tasks",
        json={
            "project_id": project.id,
            "task_type": "segmentation",
            "precision": "fp32",
            "backend_name": "tensorrt",
        },
    )

    assert response.status_code == 201, response.text
    assert response.json()["task_type"] == "segmentation"
    assert response.json()["status"] == TaskStatus.CREATED.value


def test_reference_mask_matches_declared_class_count() -> None:
    mask = reference_mask(32, class_count=3, bands=3)
    assert mask.shape == (32, 32)
    assert set(np.unique(mask)).issubset({0, 1, 2})

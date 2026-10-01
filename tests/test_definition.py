"""Model Definition 解析与校验测试（SPEC 6.1）。"""

from __future__ import annotations

import pytest

from app.adapters.definition import ModelDefinition
from app.adapters.registry import get_model_adapter, supported_architectures
from app.errors import PreprocessConfigError, ValidationFailedError

VALID_PAYLOAD = {
    "task": "detection",
    "architecture": "yolov8",
    "input": {"name": "images", "shape": [1, 3, 640, 640], "layout": "NCHW", "dtype": "FP32"},
    "preprocessing": {
        "resize": "letterbox",
        "color": "RGB",
        "scale": 255.0,
        "mean": [0, 0, 0],
        "std": [1, 1, 1],
    },
    "output": {
        "name": "output0",
        "format": "[1,84,8400]",
        "box_format": "xywh",
        "has_objectness": False,
        "class_count": 80,
    },
    "postprocessing": {
        "decoder": "yolov8",
        "nms": {"type": "classwise", "confidence_threshold": 0.25, "iou_threshold": 0.45},
    },
}


def test_parse_valid_definition() -> None:
    definition = ModelDefinition.parse(VALID_PAYLOAD)
    assert definition.architecture == "yolov8"
    assert definition.input.shape == (1, 3, 640, 640)
    assert (definition.input.height, definition.input.width) == (640, 640)
    assert definition.preprocessing.resize == "letterbox"
    assert definition.output.class_count == 80
    assert definition.output.shape == (1, 84, 8400)
    assert definition.postprocessing.nms.type == "classwise"


def test_round_trip_dict() -> None:
    definition = ModelDefinition.parse(VALID_PAYLOAD)
    assert ModelDefinition.parse(definition.to_dict()).to_dict() == definition.to_dict()


@pytest.mark.parametrize(
    ("patch", "error"),
    [
        ({"architecture": ""}, PreprocessConfigError),
        ({"input": {"name": "images", "shape": [1, 3, 640]}}, PreprocessConfigError),
        ({"input": {"name": "images", "shape": [1, 3, 0, 640]}}, PreprocessConfigError),
        ({"input": {"name": "images", "shape": [1, 3, 640, 640], "layout": "NCHWX"}}, PreprocessConfigError),
        ({"preprocessing": {"resize": "magic"}}, PreprocessConfigError),
        ({"preprocessing": {"color": "CMYK"}}, PreprocessConfigError),
        ({"preprocessing": {"std": [0, 1, 1]}}, PreprocessConfigError),
        ({"output": {"name": "o", "box_format": "xyz"}}, PreprocessConfigError),
        ({"output": {"name": "o", "class_count": 0}}, PreprocessConfigError),
        ({"postprocessing": {"nms": {"confidence_threshold": 1.5}}}, PreprocessConfigError),
    ],
)
def test_invalid_definition_rejected(patch: dict, error: type[Exception]) -> None:
    payload = {**VALID_PAYLOAD, **patch}
    with pytest.raises(error):
        ModelDefinition.parse(payload)


def test_scalar_mean_std_broadcast_to_channels() -> None:
    payload = {
        **VALID_PAYLOAD,
        "preprocessing": {"mean": 0.5, "std": 2.0},
    }
    definition = ModelDefinition.parse(payload)
    assert definition.preprocessing.mean == [0.5, 0.5, 0.5]
    assert definition.preprocessing.std == [2.0, 2.0, 2.0]


def test_registry_exposes_yolov8() -> None:
    assert "yolov8" in supported_architectures()
    adapter = get_model_adapter("yolov8")
    assert adapter.architecture == "yolov8"


def test_unknown_architecture_rejected() -> None:
    with pytest.raises(ValidationFailedError) as excinfo:
        get_model_adapter("faster-rcnn")
    assert "faster-rcnn" in str(excinfo.value)


def test_adapter_validation_detects_shape_mismatch() -> None:
    adapter = get_model_adapter("yolov8")
    definition = ModelDefinition.parse(VALID_PAYLOAD)
    inspection = {
        "inputs": [{"name": "images", "shape": [1, 3, 320, 320], "dtype": "float32"}],
        "outputs": [{"name": "output0", "shape": [1, 84, 1600], "dtype": "float32"}],
    }
    with pytest.raises(ValidationFailedError) as excinfo:
        adapter.validate(definition, inspection)
    assert "shape" in str(excinfo.value)


def test_adapter_validation_rejects_wrong_channel_count() -> None:
    adapter = get_model_adapter("yolov8")
    definition = ModelDefinition.parse(VALID_PAYLOAD)
    inspection = {
        "inputs": [{"name": "images", "shape": [1, 3, 640, 640], "dtype": "float32"}],
        "outputs": [{"name": "output0", "shape": [1, 85, 8400], "dtype": "float32"}],
    }
    with pytest.raises(ValidationFailedError) as excinfo:
        adapter.validate(definition, inspection)
    assert excinfo.value.detail["expected_channels"] == 84


def test_default_definition_derives_from_inspection() -> None:
    adapter = get_model_adapter("yolov8")
    payload = adapter.default_definition(
        {
            "inputs": [{"name": "input", "shape": [2, 3, 320, 320], "dtype": "float32"}],
            "outputs": [{"name": "pred", "shape": [2, 84, 1600], "dtype": "float32"}],
        }
    )
    assert payload["input"]["name"] == "input"
    assert payload["input"]["shape"] == [1, 3, 320, 320]  # MVP 固定 batch=1
    assert payload["output"]["name"] == "pred"
    assert payload["output"]["class_count"] == 80

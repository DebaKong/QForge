"""预处理与解码测试（SPEC 8.1 / 6.2）。

关键约定：平台 Python 侧与生成的 C++ 工程必须使用同一套预处理语义，
因此这里对 letterbox 的缩放/填充与归一化结果做精确断言。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.adapters.definition import ModelDefinition
from app.adapters.models.yolov8 import nms
from app.adapters.registry import get_model_adapter
from app.services.preprocess import letterbox, preprocess, to_tensor

DEFINITION = ModelDefinition.parse(
    {
        "task": "detection",
        "architecture": "yolov8",
        "input": {"name": "images", "shape": [1, 3, 64, 64], "layout": "NCHW", "dtype": "FP32"},
        "preprocessing": {"resize": "letterbox", "color": "RGB", "scale": 255.0},
        "output": {
            "name": "output0",
            "format": "[1,84,256]",
            "box_format": "xywh",
            "has_objectness": False,
            "class_count": 80,
        },
        "postprocessing": {
            "decoder": "yolov8",
            "nms": {"type": "classwise", "confidence_threshold": 0.25, "iou_threshold": 0.45},
        },
    }
)


def test_letterbox_keeps_aspect_ratio_and_pads() -> None:
    image = np.full((32, 64, 3), 255, dtype=np.uint8)
    canvas, scale, pad = letterbox(image, (64, 64), pad_value=114.0)

    assert canvas.shape == (64, 64, 3)
    # 64x32 → 缩放比例 min(64/32, 64/64) = 1.0
    assert scale == 1.0
    assert pad == (0.0, 16.0)  # 高度方向居中填充 16 px
    # 填充区为 pad_value，图像区为 255
    assert canvas[0, 0, 0] == 114
    assert canvas[32, 32, 0] == 255


def test_letterbox_downscales_wide_image() -> None:
    image = np.zeros((128, 256, 3), dtype=np.uint8)
    canvas, scale, pad = letterbox(image, (64, 64))
    assert canvas.shape == (64, 64, 3)
    # 128x256 → 缩放比 min(64/128, 64/256) = 0.25 → 32x64，高度方向还需填充 32 px
    assert scale == pytest.approx(0.25)
    assert pad == (0.0, 16.0)


def test_to_tensor_normalization_and_layout() -> None:
    canvas = np.full((4, 4, 3), 128, dtype=np.uint8)
    tensor = to_tensor(canvas, DEFINITION)
    assert tensor.shape == (1, 3, 4, 4)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0, 0] == pytest.approx(128 / 255.0)


def test_to_tensor_applies_mean_std_and_bgr() -> None:
    definition = ModelDefinition.parse(
        {
            "task": "detection",
            "architecture": "yolov8",
            "input": {"name": "images", "shape": [1, 3, 2, 2]},
            "preprocessing": {
                "color": "BGR",
                "scale": 1.0,
                "mean": [1.0, 2.0, 3.0],
                "std": [2.0, 2.0, 2.0],
            },
            "output": {"name": "o", "format": "[1,84,1]", "class_count": 80},
            "postprocessing": {"decoder": "yolov8"},
        }
    )
    canvas = np.zeros((2, 2, 3), dtype=np.uint8)
    canvas[:, :, 0] = 10  # R
    canvas[:, :, 2] = 20  # B
    tensor = to_tensor(canvas, definition)
    # BGR 交换后：通道0 取原 B=20 → (20-1)/2 = 9.5；通道2 取原 R=10 → (10-3)/2 = 3.5
    assert tensor[0, 0, 0, 0] == pytest.approx(9.5)
    assert tensor[0, 2, 0, 0] == pytest.approx(3.5)


def test_preprocess_returns_coordinate_mapping() -> None:
    image = np.zeros((100, 50, 3), dtype=np.uint8)
    result = preprocess(image, DEFINITION)
    assert result.tensor.shape == (1, 3, 64, 64)
    assert result.scale == pytest.approx(0.64)
    assert result.original_hw == (100, 50)

    # 反变换：letterbox 空间的中心点应回到原图中心
    restored = result.to_original([32.0, 32.0, 32.0, 32.0])
    assert restored[0] == pytest.approx(25.0, abs=1.5)
    assert restored[1] == pytest.approx(50.0, abs=1.5)


def test_nms_suppresses_overlapping_boxes_of_same_class() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.array([0, 0, 0])
    keep = nms(boxes, scores, classes, iou_threshold=0.45, classwise=True)
    assert keep == [0, 2]


def test_nms_keeps_overlapping_boxes_of_different_classes_when_classwise() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1])
    assert len(nms(boxes, scores, classes, iou_threshold=0.45, classwise=True)) == 2
    assert len(nms(boxes, scores, classes, iou_threshold=0.45, classwise=False)) == 1


def test_adapter_decode_yolov8_output() -> None:
    adapter = get_model_adapter("yolov8")
    channels = 4 + DEFINITION.output.class_count
    anchors = 4
    raw = np.zeros((1, channels, anchors), dtype=np.float32)

    # 第 0 个 anchor：中心 (32,32)，宽高 16，类别 3 置信度 0.9
    raw[0, 0, 0] = 32.0
    raw[0, 1, 0] = 32.0
    raw[0, 2, 0] = 16.0
    raw[0, 3, 0] = 16.0
    raw[0, 4 + 3, 0] = 0.9
    # 第 1 个 anchor：几乎重合，置信度 0.8（应被 NMS 抑制）
    raw[0, 0, 1] = 33.0
    raw[0, 1, 1] = 33.0
    raw[0, 2, 1] = 16.0
    raw[0, 3, 1] = 16.0
    raw[0, 4 + 3, 1] = 0.8
    # 第 2 个 anchor：低置信度（应被阈值过滤）
    raw[0, 4 + 5, 2] = 0.1

    result = adapter.decode([raw], DEFINITION)
    assert result.raw_shape == (1, channels, anchors)
    assert result.candidates == 2
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.class_id == 3
    assert detection.score == pytest.approx(0.9)
    assert detection.x1 == pytest.approx(24.0)
    assert detection.y2 == pytest.approx(40.0)


def test_adapter_decode_rejects_wrong_channel_count() -> None:
    adapter = get_model_adapter("yolov8")
    raw = np.zeros((1, 20, 8), dtype=np.float32)
    with pytest.raises(Exception) as excinfo:
        adapter.decode([raw], DEFINITION)
    assert "class_count" in str(excinfo.value)

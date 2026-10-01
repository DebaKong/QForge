"""统一预处理（SPEC 8.1）。

平台的预处理必须**只有一份实现语义**：Python 侧用于 INT8 校准与精度验证，
生成的 C++ 工程按同一份参数实现（写入 config/model.yaml），二者不一致会直接导致精度下降。

支持：resize（letterbox / stretch / center_crop）、颜色顺序（RGB / BGR）、
像素缩放与 mean/std 归一化、NCHW / NHWC 布局。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.adapters.definition import ModelDefinition
from app.errors import PreprocessConfigError

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """预处理结果与坐标映射信息（用于把检测框还原到原图坐标）。"""

    tensor: np.ndarray  # (1, C, H, W) float32
    scale: float
    pad: tuple[int, int]  # (pad_x, pad_y)
    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]

    def to_original(self, box: Sequence[float]) -> list[float]:
        """把 letterbox 空间的 xyxy 还原到原图坐标。"""
        pad_x, pad_y = self.pad
        scale = self.scale or 1.0
        height, width = self.original_hw
        x1 = (box[0] - pad_x) / scale
        y1 = (box[1] - pad_y) / scale
        x2 = (box[2] - pad_x) / scale
        y2 = (box[3] - pad_y) / scale
        return [
            float(np.clip(x1, 0, width)),
            float(np.clip(y1, 0, height)),
            float(np.clip(x2, 0, width)),
            float(np.clip(y2, 0, height)),
        ]


def load_image_rgb(path: Path) -> np.ndarray:
    """读取图像为 RGB uint8 (H, W, 3)，并按 EXIF 方向自动摆正。"""
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
    except Exception as exc:
        raise PreprocessConfigError(
            "图像无法读取", detail={"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        ) from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise PreprocessConfigError(
            "图像通道数异常", detail={"path": str(path), "shape": list(array.shape)}
        )
    return array


def letterbox(
    image: np.ndarray, target_hw: tuple[int, int], *, pad_value: float = 114.0
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比缩放并填充到目标尺寸，返回 (画布, 缩放比例, (pad_x, pad_y))。"""
    target_h, target_w = target_hw
    height, width = image.shape[:2]
    scale = min(target_h / height, target_w / width)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top

    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(float(pad_value), float(pad_value), float(pad_value)),
    )
    return canvas, float(scale), (float(left), float(top))


def _resize_stretch(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _center_crop(image: np.ndarray, target_hw: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    target_h, target_w = target_hw
    height, width = image.shape[:2]
    scale = max(target_h / height, target_w / width)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = max((new_w - target_w) // 2, 0)
    top = max((new_h - target_h) // 2, 0)
    cropped = resized[top : top + target_h, left : left + target_w]
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:  # pragma: no cover
        raise PreprocessConfigError(
            "center_crop 结果尺寸异常", detail={"shape": list(cropped.shape)}
        )
    return cropped, float(scale), (-float(left), -float(top))


def to_tensor(canvas: np.ndarray, definition: ModelDefinition) -> np.ndarray:
    """按 Model Definition 归一化并转成 (1, C, H, W) float32。"""
    spec = definition.preprocessing
    array = canvas.astype(np.float32)

    if spec.color == "BGR":
        array = array[:, :, ::-1]

    if spec.scale:
        array = array / float(spec.scale)

    mean = np.asarray(spec.mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(spec.std, dtype=np.float32).reshape(1, 1, -1)
    array = (array - mean) / std

    if definition.input.layout == "NCHW":
        array = np.transpose(array, (2, 0, 1))
    else:
        array = np.transpose(array, (0, 1, 2))  # NHWC 保持原序

    return np.ascontiguousarray(array[None, ...], dtype=np.float32)


def preprocess(image: np.ndarray, definition: ModelDefinition) -> PreprocessResult:
    """对单张图像执行完整预处理。"""
    target_hw = (definition.input.height, definition.input.width)
    spec = definition.preprocessing

    if spec.resize == "letterbox":
        canvas, scale, pad = letterbox(image, target_hw, pad_value=spec.pad_value)
    elif spec.resize == "stretch":
        canvas = _resize_stretch(image, target_hw)
        # stretch 会改变长宽比，坐标还原按两轴各自的缩放比处理，这里记录等效比例供报告使用
        scale = float(min(target_hw[0] / image.shape[0], target_hw[1] / image.shape[1]))
        pad = (0.0, 0.0)
    elif spec.resize == "center_crop":
        canvas, scale, pad = _center_crop(image, target_hw)
    else:  # pragma: no cover - 已被 definition 校验拦截
        raise PreprocessConfigError(f"不支持的 resize 方式：{spec.resize}")

    tensor = to_tensor(canvas, definition)
    return PreprocessResult(
        tensor=tensor,
        scale=scale,
        pad=(float(pad[0]), float(pad[1])),
        original_hw=(int(image.shape[0]), int(image.shape[1])),
        resized_hw=(int(canvas.shape[0]), int(canvas.shape[1])),
    )


def preprocess_file(path: Path, definition: ModelDefinition) -> PreprocessResult:
    return preprocess(load_image_rgb(path), definition)


def preprocess_batches(
    paths: Sequence[Path],
    definition: ModelDefinition,
    *,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """按批产出 (N, C, H, W) 张量，供 INT8 校准使用。"""
    batch: list[np.ndarray] = []
    for path in paths:
        batch.append(preprocess_file(path, definition).tensor[0])
        if len(batch) == batch_size:
            yield np.stack(batch, axis=0)
            batch = []
    if batch:
        yield np.stack(batch, axis=0)

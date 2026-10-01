"""校准数据集校验与供给（SPEC 8.2 / 8.3 / 9.1 步骤 3~4）。

校验内容（SPEC 8.3）：
- 图像是否可读取；
- 统计数量、分辨率、通道数与文件类型；
- 随机抽样执行预处理，检查 tensor 的 shape / dtype / 数值范围；
- 报告异常图片清单。

INT8 校准所需的批次张量由 `preprocess_batches` 产出，保证与 C++ 工程使用同一套预处理参数。
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.adapters.definition import ModelDefinition
from app.errors import CalibrationDataError
from app.services.preprocess import preprocess_file

logger = logging.getLogger(__name__)

# 允许的坏图比例：超过该比例直接判定校准集不可用（避免"能跑但精度崩"）
MAX_BAD_IMAGE_RATIO = 0.5


@dataclass
class CalibrationReport:
    """校准集扫描与抽样结果。"""

    root: Path
    image_count: int = 0
    total_files: int = 0
    formats: dict[str, int] = field(default_factory=dict)
    resolutions: dict[str, int] = field(default_factory=dict)
    channels: dict[str, int] = field(default_factory=dict)
    bad_files: list[dict[str, str]] = field(default_factory=list)
    sample: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "image_count": self.image_count,
            "total_files": self.total_files,
            "formats": self.formats,
            "resolutions": self.resolutions,
            "channels": self.channels,
            "bad_files": self.bad_files[:50],
            "bad_file_count": len(self.bad_files),
            "sample": self.sample,
            "warnings": self.warnings,
        }


def scan_images(
    root: Path, allowed_extensions: list[str], *, max_images: int
) -> list[Path]:
    """递归扫描图像文件（扩展名白名单），结果稳定排序。"""
    if not root.exists() or not root.is_dir():
        raise CalibrationDataError(
            "校准集目录不存在（应先上传并解压 calibration.zip）", detail={"root": str(root)}
        )

    allowed = {item.lower() for item in allowed_extensions}
    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]
    found.sort()

    if not found:
        raise CalibrationDataError(
            "校准集目录内没有找到图像文件",
            detail={"root": str(root), "allowed_extensions": sorted(allowed)},
        )
    if len(found) > max_images:
        raise CalibrationDataError(
            f"校准集图像数 {len(found)} 超过上限 {max_images}",
            detail={"image_count": len(found), "max_images": max_images},
        )
    return found


def validate_calibration_set(
    root: Path,
    definition: ModelDefinition,
    *,
    allowed_extensions: list[str],
    max_images: int,
    sample_size: int = 8,
    seed: int = 0,
) -> CalibrationReport:
    """执行 SPEC 8.3 的全部校验项。"""
    images = scan_images(root, allowed_extensions, max_images=max_images)
    report = CalibrationReport(root=root, image_count=len(images), total_files=len(images))

    format_counter: Counter[str] = Counter()
    resolution_counter: Counter[str] = Counter()
    channel_counter: Counter[str] = Counter()

    for path in images:
        try:
            with Image.open(path) as image:
                format_counter[image.format or path.suffix.lstrip(".").upper()] += 1
                resolution_counter[f"{image.width}x{image.height}"] += 1
                channel_counter[str(len(image.getbands()))] += 1
        except Exception as exc:
            report.bad_files.append(
                {"path": path.name, "reason": f"{type(exc).__name__}: {exc}"}
            )

    readable = report.image_count - len(report.bad_files)
    if readable <= 0:
        raise CalibrationDataError(
            "校准集内没有任何可读取的图像",
            detail={"image_count": report.image_count, "bad_files": report.bad_files[:10]},
        )
    if len(report.bad_files) / max(report.image_count, 1) > MAX_BAD_IMAGE_RATIO:
        raise CalibrationDataError(
            "校准集坏图比例过高，拒绝用于量化",
            detail={
                "bad": len(report.bad_files),
                "total": report.image_count,
                "limit_ratio": MAX_BAD_IMAGE_RATIO,
            },
        )
    if report.bad_files:
        report.warnings.append(f"存在 {len(report.bad_files)} 张不可读取的图像，已排除")

    report.formats = dict(format_counter.most_common())
    report.resolutions = dict(resolution_counter.most_common(20))
    report.channels = dict(channel_counter.most_common())

    if len(resolution_counter) > 1:
        report.warnings.append(
            f"校准集包含 {len(resolution_counter)} 种分辨率，letterbox 会统一到模型输入尺寸"
        )

    # 抽样预处理检查（SPEC 8.3：检查 tensor shape/dtype/range）
    good_images = [
        path for path in images if path.name not in {item["path"] for item in report.bad_files}
    ]
    rng = random.Random(seed)
    sample_paths = rng.sample(good_images, min(sample_size, len(good_images)))

    expected_shape = tuple(definition.input.shape)
    tensors: list[np.ndarray] = []
    for path in sample_paths:
        result = preprocess_file(path, definition)
        tensor = result.tensor
        if tuple(tensor.shape) != expected_shape:
            raise CalibrationDataError(
                "预处理结果与模型输入 shape 不一致",
                detail={
                    "image": path.name,
                    "tensor_shape": list(tensor.shape),
                    "expected": list(expected_shape),
                },
            )
        if tensor.dtype != np.float32:
            raise CalibrationDataError(
                "预处理结果 dtype 不是 float32", detail={"dtype": str(tensor.dtype)}
            )
        if not np.isfinite(tensor).all():
            raise CalibrationDataError(
                "预处理结果包含 NaN/Inf", detail={"image": path.name}
            )
        tensors.append(tensor)

    stacked = np.stack([item[0] for item in tensors], axis=0)
    report.sample = {
        "count": len(sample_paths),
        "images": [path.name for path in sample_paths],
        "tensor_shape": list(stacked.shape[1:]),
        "dtype": str(stacked.dtype),
        "value_min": float(stacked.min()),
        "value_max": float(stacked.max()),
        "value_mean": float(stacked.mean()),
        "value_std": float(stacked.std()),
    }

    logger.info(
        "校准集校验通过",
        extra={
            "image_count": report.image_count,
            "bad": len(report.bad_files),
            "formats": report.formats,
        },
    )
    return report

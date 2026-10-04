"""语义分割后处理与精度指标（SPEC 2.2「语义分割：V1.0 扩展」/ SPEC 9 精度定义）。

为什么单独一个模块：
- 分割与检测的**后处理语义完全不同**（检测是 NMS 出框，分割是逐像素 argmax 出掩膜），
  所以这里给出统一的掩膜后处理与指标实现，Python 运行验证与精度报告共用一套；
- 指标必须与"人算得出来"的定义完全一致（IoU / Dice / 像素准确率），
  这样报告里的数字才能被复核，而不是靠信任。

关键约定（写清楚，避免平台里两层语义打架）：
- 模型输出是 **logits**（未过 softmax），shape `(N, C, H, W)`；后处理取 argmax 得到类别标签；
- `ignore_index`（常见为 255）表示**不参与评估**的像素（边界/未标注），
  在指标计算里被完全排除，并且**不计入任何类别的分母**；
- 掩膜缩放回原图尺寸使用**最近邻**（标签不能插值）；
- 指标按"整个数据集累积混淆矩阵"计算（不是每张图算完取平均）——后者会被小图/空类扭曲。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 常见的忽略标签（Pascal VOC / Cityscapes 用 255）
DEFAULT_IGNORE_INDEX = 255


@dataclass
class SegmentationResult:
    """一张（或一批）图像的掩膜后处理结果。"""

    labels: np.ndarray  # (N, H', W') uint8/int32，argmax 后的类别标签
    class_count: int
    ignored_mask: np.ndarray | None = None  # True 表示该像素不参与评估
    resized: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.labels.shape),
            "class_count": self.class_count,
            "resized": self.resized,
            "ignored_pixels": int(self.ignored_mask.sum()) if self.ignored_mask is not None else 0,
            "notes": self.notes,
        }


def logits_to_labels(output: np.ndarray) -> np.ndarray:
    """把 `(N, C, H, W)` logits 取 argmax 变成 `(N, H, W)` 标签。

    维度顺序不做猜测试探：不是 4D 就明确报错（宁可失败也不要静默算错）。
    """
    array = np.asarray(output)
    if array.ndim != 4:
        raise ValueError(f"分割输出必须是 4D (N, C, H, W)，收到 shape={array.shape}")
    return np.argmax(array, axis=1).astype(np.int32)


def resize_labels_nearest(labels: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """最近邻缩放标签图（`(N, H, W)` → `(N, target_h, target_w)`）；标签不能插值。"""
    array = np.asarray(labels)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"标签必须是 (N, H, W) 或 (H, W)，收到 shape={array.shape}")

    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"目标尺寸必须为正，收到 {target_hw}")

    source_h, source_w = array.shape[1], array.shape[2]
    if (source_h, source_w) == (target_h, target_w):
        return array.astype(np.int32, copy=False)

    # 用行/列索引做最近邻：每行取 floor(i * source / target)，与原图网格对齐
    rows = np.minimum((np.arange(target_h) * source_h // target_h), source_h - 1)
    cols = np.minimum((np.arange(target_w) * source_w // target_w), source_w - 1)
    return array[:, rows][:, :, cols].astype(np.int32, copy=False)


def postprocess_mask(
    output: np.ndarray,
    *,
    class_count: int,
    target_hw: tuple[int, int] | None = None,
    ignore_index: int | None = None,
) -> SegmentationResult:
    """分割后处理：argmax → （可选）缩回原图 → 生成忽略掩膜。"""
    if class_count <= 0:
        raise ValueError(f"class_count 必须为正，收到 {class_count}")

    labels = logits_to_labels(output)
    channels = np.asarray(output).shape[1]
    notes: list[str] = []
    if channels != class_count:
        notes.append(
            f"模型输出通道数 {channels} 与声明的类别数 {class_count} 不一致，"
            "已按输出通道数解释类别（请在 Model Definition 中修正）"
        )
        class_count = int(channels)

    resized = False
    if target_hw is not None and tuple(labels.shape[1:]) != (int(target_hw[0]), int(target_hw[1])):
        labels = resize_labels_nearest(labels, target_hw)
        resized = True

    ignored = None
    if ignore_index is not None:
        ignored = labels == int(ignore_index)

    return SegmentationResult(
        labels=labels,
        class_count=int(class_count),
        ignored_mask=ignored,
        resized=resized,
        notes=notes,
    )


def confusion_matrix(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    class_count: int,
    ignore_index: int | None = DEFAULT_IGNORE_INDEX,
) -> np.ndarray:
    """累积混淆矩阵 `[class_count, class_count]`（行=真值，列=预测）。

    忽略像素（真值或预测为 `ignore_index`）整列整行都不统计——这是"不参与评估"的准确含义。
    """
    reference_array = np.asarray(reference).astype(np.int64).ravel()
    actual_array = np.asarray(actual).astype(np.int64).ravel()
    if reference_array.size != actual_array.size:
        raise ValueError(
            f"真值与预测像素数不一致：{reference_array.size} vs {actual_array.size}"
        )

    valid = (reference_array >= 0) & (reference_array < class_count)
    valid &= (actual_array >= 0) & (actual_array < class_count)
    if ignore_index is not None:
        valid &= reference_array != int(ignore_index)
        valid &= actual_array != int(ignore_index)

    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (reference_array[valid], actual_array[valid]), 1)
    return matrix


def metrics_from_confusion(
    matrix: np.ndarray, *, class_names: list[str] | None = None
) -> dict[str, Any]:
    """由混淆矩阵导出 IoU / Dice / 像素准确率（SPEC 9 的分割精度指标）。"""
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"混淆矩阵必须是方阵，收到 shape={matrix.shape}")

    class_count = matrix.shape[0]
    true_positive = np.diag(matrix).astype(np.float64)
    # 行和=该类真值像素数；列和=该类预测像素数
    actual_total = matrix.sum(axis=1).astype(np.float64)
    predicted_total = matrix.sum(axis=0).astype(np.float64)
    union = actual_total + predicted_total - true_positive
    dice_denominator = actual_total + predicted_total

    per_class: list[dict[str, Any]] = []
    ious: list[float] = []
    dices: list[float] = []
    for index in range(class_count):
        # 该类在真值与预测里都没出现 → 不参与平均（否则空类会把 mIoU 拉低）
        if union[index] == 0:
            per_class.append(
                {
                    "class_id": index,
                    "name": (class_names or [])[index] if class_names and index < len(class_names) else f"class_{index}",
                    "iou": None,
                    "dice": None,
                    "pixels": 0,
                    "present": False,
                }
            )
            continue
        iou = float(true_positive[index] / union[index])
        dice = (
            float(2 * true_positive[index] / dice_denominator[index])
            if dice_denominator[index] > 0
            else 0.0
        )
        ious.append(iou)
        dices.append(dice)
        per_class.append(
            {
                "class_id": index,
                "name": (class_names or [])[index] if class_names and index < len(class_names) else f"class_{index}",
                "iou": round(iou, 6),
                "dice": round(dice, 6),
                "pixels": int(actual_total[index]),
                "present": True,
            }
        )

    total = int(matrix.sum())
    accuracy = float(true_positive.sum() / total) if total else 0.0
    return {
        "class_count": class_count,
        "evaluated_pixels": total,
        "mean_iou": round(float(np.mean(ious)), 6) if ious else 0.0,
        "mean_dice": round(float(np.mean(dices)), 6) if dices else 0.0,
        "pixel_accuracy": round(accuracy, 6),
        "classes_evaluated": len(ious),
        "per_class": per_class,
    }


def evaluate_masks(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    class_count: int,
    ignore_index: int | None = DEFAULT_IGNORE_INDEX,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """评估两张掩膜（或两批掩膜）：IoU / Dice / 像素准确率 + 逐类明细。"""
    matrix = confusion_matrix(
        reference, actual, class_count=class_count, ignore_index=ignore_index
    )
    metrics = metrics_from_confusion(matrix, class_names=class_names)
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def evaluate_outputs(
    output: np.ndarray,
    reference: np.ndarray,
    *,
    class_count: int,
    target_hw: tuple[int, int] | None = None,
    ignore_index: int | None = DEFAULT_IGNORE_INDEX,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """直接评估模型输出（logits）与真值掩膜：后处理 + 指标一步到位。"""
    result = postprocess_mask(
        output, class_count=class_count, target_hw=target_hw, ignore_index=ignore_index
    )
    metrics = evaluate_masks(
        reference,
        result.labels,
        class_count=class_count,
        ignore_index=ignore_index,
        class_names=class_names,
    )
    metrics["postprocess"] = result.to_dict()
    return metrics


def agreement_metrics(
    output: np.ndarray,
    baseline_output: np.ndarray,
    *,
    class_count: int,
    ignore_index: int | None = None,
) -> dict[str, Any]:
    """**无标注**场景下的分割定量检查：引擎掩膜与 FP32 基线掩膜的一致度。

    为什么这样做：平台的校准/样例数据是**未标注图像**（SPEC 8.2），拿不到真值掩膜，
    因此和检测路径用 MAE/RMSE 对比张量一样，这里把 FP32 基线的 argmax 结果当作参照，
    用 IoU / Dice / 像素一致率衡量"精度模式带来的掩膜变化"。
    有标注数据时请改用 `evaluate_masks`（真值对比），两者的指标定义完全一致。
    """
    engine = postprocess_mask(output, class_count=class_count, ignore_index=ignore_index)
    baseline = postprocess_mask(
        baseline_output, class_count=class_count, ignore_index=ignore_index
    )
    if engine.labels.shape != baseline.labels.shape:
        raise ValueError(
            "引擎与基线的掩膜形状不一致："
            f"{engine.labels.shape} vs {baseline.labels.shape}"
        )

    metrics = evaluate_masks(
        baseline.labels, engine.labels, class_count=class_count, ignore_index=ignore_index
    )
    metrics["baseline"] = "onnxruntime CPU FP32（argmax 掩膜作为参照，非人工标注真值）"
    metrics["note"] = (
        "无标注数据下的一致性检查：指标含义是「引擎掩膜与 FP32 基线掩膜的吻合度」，"
        "不是与人工标注的精度；有标注时请用 evaluate_masks 做真值对比"
    )
    return metrics

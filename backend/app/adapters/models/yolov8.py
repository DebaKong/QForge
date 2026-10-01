"""YOLOv8 目标检测适配器（SPEC 6.2）。

输出布局按 YOLOv8 官方导出约定：`[1, 4 + class_count, anchors]`，
box 为 xywh（中心点 + 宽高），无 objectness，类别分数已过 sigmoid。

本适配器负责：
- 给出该架构的默认 Model Definition（输入信息取自真实 ONNX 图，不凭空猜）；
- 校验用户声明的定义与 ONNX 图是否一致；
- 为代码生成提供模板参数；
- 在平台侧实现同一套解码 + NMS，用于运行验证与精度报告（与生成的 C++ 逻辑一致）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.adapters.base import DecodeResult, Detection, ModelAdapter
from app.adapters.definition import ModelDefinition
from app.adapters.registry import register_model_adapter
from app.errors import PreprocessConfigError, ValidationFailedError

logger = logging.getLogger(__name__)

DEFAULT_INPUT_SHAPE = [1, 3, 640, 640]
DEFAULT_CLASS_COUNT = 80


def _iou_matrix(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    """boxes: (N,4) xyxy；box: (4,) xyxy → (N,) IoU。"""
    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    area_b = np.clip(box[2] - box[0], 0, None) * np.clip(box[3] - box[1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    iou_threshold: float,
    classwise: bool = True,
) -> list[int]:
    """贪心 NMS，返回保留的下标（按分数降序）。"""
    if boxes.size == 0:
        return []

    order = scores.argsort()[::-1]
    keep: list[int] = []

    if classwise:
        for class_id in np.unique(class_ids):
            class_order = [index for index in order if class_ids[index] == class_id]
            while class_order:
                current = class_order.pop(0)
                keep.append(int(current))
                if not class_order:
                    break
                others = np.array(class_order)
                ious = _iou_matrix(boxes[others], boxes[current])
                class_order = [
                    int(index) for index, iou in zip(others, ious, strict=True) if iou <= iou_threshold
                ]
        return sorted(keep, key=lambda index: -scores[index])

    remaining = list(order)
    while remaining:
        current = remaining.pop(0)
        keep.append(int(current))
        if not remaining:
            break
        others = np.array(remaining)
        ious = _iou_matrix(boxes[others], boxes[current])
        remaining = [
            int(index) for index, iou in zip(others, ious, strict=True) if iou <= iou_threshold
        ]
    return keep


@register_model_adapter
class YOLOv8Adapter(ModelAdapter):
    architecture = "yolov8"
    task = "detection"

    # ---------------- Model Definition ----------------

    def default_definition(self, inspection: dict[str, Any] | None = None) -> dict[str, Any]:
        input_info = (inspection or {}).get("inputs") or []
        output_info = (inspection or {}).get("outputs") or []

        name = input_info[0]["name"] if input_info else "images"
        shape = list(input_info[0]["shape"]) if input_info else list(DEFAULT_INPUT_SHAPE)
        shape = [dim if isinstance(dim, int) and dim > 0 else 640 for dim in shape]
        if len(shape) == 4:
            shape[0] = 1  # MVP：固定 batch=1（SPEC 10.1）

        class_count = DEFAULT_CLASS_COUNT
        output_name = output_info[0]["name"] if output_info else "output0"
        output_shape = list(output_info[0]["shape"]) if output_info else [1, 4 + class_count, 8400]
        # 仅在形状明确符合 [1, 4+C, N] 时据此推断类别数；这属于「按已知架构约定取值」，
        # 不是用 ONNX Shape 猜后处理语义（后处理仍由本文件显式声明）。
        if len(output_shape) == 3:
            channels = output_shape[1]
            if isinstance(channels, int) and channels > 4:
                class_count = channels - 4

        return {
            "task": self.task,
            "architecture": self.architecture,
            "input": {"name": name, "shape": shape, "layout": "NCHW", "dtype": "FP32"},
            "preprocessing": {
                "resize": "letterbox",
                "color": "RGB",
                "scale": 255.0,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "pad_value": 114.0,
            },
            "output": {
                "name": output_name,
                "format": f"[{','.join(str(dim) for dim in output_shape)}]",
                "box_format": "xywh",
                "has_objectness": False,
                "class_count": class_count,
            },
            "postprocessing": {
                "decoder": "yolov8",
                "nms": {
                    "type": "classwise",
                    "confidence_threshold": 0.25,
                    "iou_threshold": 0.45,
                },
            },
        }

    def validate(
        self, definition: ModelDefinition, inspection: dict[str, Any] | None = None
    ) -> list[str]:
        warnings: list[str] = []
        if definition.task != self.task:
            raise ValidationFailedError(
                f"YOLOv8 适配器只支持 detection 任务，收到 {definition.task}",
                detail={"task": definition.task},
            )
        if definition.postprocessing.decoder != self.architecture:
            raise PreprocessConfigError(
                f"decoder 必须是 {self.architecture}，收到 {definition.postprocessing.decoder}",
                detail={"decoder": definition.postprocessing.decoder},
            )

        if inspection:
            graph_inputs = {item["name"]: item for item in inspection.get("inputs") or []}
            graph_outputs = {item["name"]: item for item in inspection.get("outputs") or []}

            declared = definition.input.name
            if declared not in graph_inputs:
                raise ValidationFailedError(
                    "Model Definition 的输入名与 ONNX 图不一致",
                    detail={"declared": declared, "graph_inputs": sorted(graph_inputs)},
                )

            graph_shape = graph_inputs[declared].get("shape") or []
            declared_shape = list(definition.input.shape)
            if len(graph_shape) == len(declared_shape):
                mismatched = [
                    {"axis": axis, "graph": graph_dim, "declared": declared_shape[axis]}
                    for axis, graph_dim in enumerate(graph_shape)
                    if isinstance(graph_dim, int) and graph_dim > 0 and graph_dim != declared_shape[axis]
                ]
                if mismatched:
                    raise ValidationFailedError(
                        "Model Definition 的输入 shape 与 ONNX 图不一致",
                        detail={"mismatched": mismatched, "declared": declared_shape},
                    )
            else:
                warnings.append(
                    f"输入 shape 维数不同：图为 {graph_shape}，定义为 {declared_shape}"
                )

            out_name = definition.output.name
            if out_name not in graph_outputs:
                raise ValidationFailedError(
                    "Model Definition 的输出名与 ONNX 图不一致",
                    detail={"declared": out_name, "graph_outputs": sorted(graph_outputs)},
                )

            expected_channels = 4 + definition.output.class_count + (
                1 if definition.output.has_objectness else 0
            )
            graph_out_shape = graph_outputs[out_name].get("shape") or []
            if len(graph_out_shape) == 3:
                numeric = [dim for dim in graph_out_shape if isinstance(dim, int) and dim > 0]
                if numeric and expected_channels not in numeric:
                    raise ValidationFailedError(
                        "输出通道数与 class_count / has_objectness 不一致",
                        detail={
                            "expected_channels": expected_channels,
                            "graph_output_shape": graph_out_shape,
                            "class_count": definition.output.class_count,
                            "has_objectness": definition.output.has_objectness,
                        },
                    )
            else:
                warnings.append(
                    f"输出 shape 不是三维：{graph_out_shape}（YOLOv8 期望 [1, 4+C, N]）"
                )

        if definition.input.channels != 3:
            warnings.append(f"输入通道数为 {definition.input.channels}，YOLOv8 一般使用 3 通道 RGB")

        return warnings

    # ---------------- 代码生成参数 ----------------

    def template_context(self, definition: ModelDefinition) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "task": self.task,
            "input_name": definition.input.name,
            "input_shape": list(definition.input.shape),
            "input_layout": definition.input.layout,
            "input_height": definition.input.height,
            "input_width": definition.input.width,
            "input_channels": definition.input.channels,
            "class_count": definition.output.class_count,
            "has_objectness": definition.output.has_objectness,
            "box_format": definition.output.box_format,
            "output_name": definition.output.name,
            "decoder": definition.postprocessing.decoder,
            "nms_type": definition.postprocessing.nms.type,
            "confidence_threshold": definition.postprocessing.nms.confidence_threshold,
            "iou_threshold": definition.postprocessing.nms.iou_threshold,
            "resize": definition.preprocessing.resize,
            "color": definition.preprocessing.color,
            "scale": definition.preprocessing.scale,
            "mean": definition.preprocessing.mean,
            "std": definition.preprocessing.std,
            "pad_value": definition.preprocessing.pad_value,
        }

    # ---------------- 平台侧解码 + NMS ----------------

    def decode(self, outputs: list[np.ndarray], definition: ModelDefinition) -> DecodeResult:
        if not outputs:
            raise ValidationFailedError("推理没有返回任何输出张量")

        raw = np.asarray(outputs[0])
        result = DecodeResult(raw_shape=tuple(int(dim) for dim in raw.shape))

        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim != 2:
            raise ValidationFailedError(
                "YOLOv8 输出应为二维或三维张量",
                detail={"shape": list(result.raw_shape)},
            )

        attributes = 4 + definition.output.class_count + (
            1 if definition.output.has_objectness else 0
        )
        if raw.shape[0] == attributes:
            data = raw.T  # (anchors, attributes)
        elif raw.shape[1] == attributes:
            data = raw
        else:
            raise ValidationFailedError(
                "输出张量的通道数与 class_count 不匹配",
                detail={
                    "shape": list(raw.shape),
                    "expected_channels": attributes,
                    "class_count": definition.output.class_count,
                },
            )

        data = data.astype(np.float32, copy=False)
        boxes_raw = data[:, :4]
        scores_raw = data[:, 4:]

        if definition.output.has_objectness:
            objectness = scores_raw[:, 0]
            class_scores = scores_raw[:, 1:]
            confidence = objectness[:, None] * class_scores
        else:
            class_scores = scores_raw
            confidence = class_scores

        class_ids = confidence.argmax(axis=1)
        scores = confidence[np.arange(confidence.shape[0]), class_ids]

        threshold = definition.postprocessing.nms.confidence_threshold
        mask = scores >= threshold
        result.candidates = int(mask.sum())
        if not mask.any():
            return result

        boxes = boxes_raw[mask]
        if definition.output.box_format == "xywh":
            cx, cy, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes = np.stack(
                [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], axis=1
            )
        elif definition.output.box_format == "cxcywh":  # pragma: no cover - 预留
            cx, cy, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes = np.stack(
                [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], axis=1
            )

        kept_scores = scores[mask]
        kept_classes = class_ids[mask]

        keep = nms(
            boxes,
            kept_scores,
            kept_classes,
            iou_threshold=definition.postprocessing.nms.iou_threshold,
            classwise=definition.postprocessing.nms.type != "class_agnostic",
        )
        if definition.postprocessing.nms.type == "none":
            keep = list(range(len(kept_scores)))

        result.detections = [
            Detection(
                x1=float(boxes[index][0]),
                y1=float(boxes[index][1]),
                x2=float(boxes[index][2]),
                y2=float(boxes[index][3]),
                score=float(kept_scores[index]),
                class_id=int(kept_classes[index]),
            )
            for index in keep
        ]
        return result

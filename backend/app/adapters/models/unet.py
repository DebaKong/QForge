"""U-Net 风格语义分割适配器（SPEC 2.2「语义分割：V1.0 扩展」/ SPEC 6）。

为什么这样设计：
- **不猜后处理语义**：分割的后处理是"逐像素 argmax + 可选缩回原图"，与检测的"NMS 出框"完全
  不同，所以由本适配器显式声明 `postprocessing.decoder = argmax` 与 `ignore_index`，
  而不是让平台从 ONNX shape 反推（AGENTS.md 明确禁止用 shape 猜后处理）；
- **复用同一套指标实现**：解码与评估都调用 `app/services/segmentation.py`
  （logits → 标签 → IoU/Dice），Python 运行验证、精度报告、后续 C++ 侧语义保持一处一致；
- **接口向后兼容**：检测路径靠 `DecodeResult.detections`，分割结果放在 `DecodeResult.extra`
  （掩膜形状、类别直方图、忽略像素数），两边互不影响。

支持的架构名：`unet`（本适配器的 architecture）。同一份实现也接受 `decoder=argmax` 的
任意分割模型（例如 DeepLab/U-Net 变体），只要 Model Definition 声明正确。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.adapters.base import DecodeResult, ModelAdapter
from app.adapters.definition import ModelDefinition
from app.adapters.registry import register_model_adapter
from app.errors import PreprocessConfigError, ValidationFailedError
from app.services import segmentation

DEFAULT_INPUT_SHAPE = (1, 3, 256, 256)
DEFAULT_CLASS_COUNT = 2  # 常见二分类分割（前景/背景）；多分类由 Model Definition 覆盖
# 本适配器认可的解码器名（argmax 是语义描述，unet 是架构名，两者都指向同一实现）
SUPPORTED_DECODERS = ("argmax", "unet")


@register_model_adapter
class UNetAdapter(ModelAdapter):
    architecture = "unet"
    task = "segmentation"

    # ---------------- Model Definition ----------------

    def default_definition(self, inspection: dict[str, Any] | None = None) -> dict[str, Any]:
        input_info = (inspection or {}).get("inputs") or []
        output_info = (inspection or {}).get("outputs") or []

        input_name = input_info[0]["name"] if input_info else "images"
        shape = list(input_info[0]["shape"]) if input_info else list(DEFAULT_INPUT_SHAPE)
        # 动态维（batch/尺寸）按 MVP 固定下来（SPEC 10.1：先保证固定 shape 主链路稳定）
        shape = [dim if isinstance(dim, int) and dim > 0 else 256 for dim in shape]
        if len(shape) == 4:
            shape[0] = 1

        output_name = output_info[0]["name"] if output_info else "logits"
        output_shape = list(output_info[0]["shape"]) if output_info else None
        class_count = DEFAULT_CLASS_COUNT
        if output_shape and len(output_shape) == 4:
            channels = output_shape[1]
            # 分割输出的通道维就是类别数（这是分割的**架构约定**，不是用 shape 猜后处理）
            if isinstance(channels, int) and channels > 0:
                class_count = channels

        return {
            "task": self.task,
            "architecture": self.architecture,
            "input": {"name": input_name, "shape": shape, "layout": "NCHW", "dtype": "FP32"},
            "preprocessing": {
                # 分割用 stretch（整图缩放到网络输入尺寸），与检测的 letterbox 不同：
                # 掩膜要逐像素对应原图，补边会引入无效区域
                "resize": "stretch",
                "color": "RGB",
                "scale": 255.0,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
            "output": {
                "name": output_name,
                "format": (
                    f"[{','.join(str(dim) for dim in output_shape)}]" if output_shape else "[1,C,H,W]"
                ),
                "class_count": class_count,
            },
            "postprocessing": {
                "decoder": "argmax",
                # 255 是 Pascal VOC / Cityscapes 的忽略标签；二分类任务可设为 null
                "ignore_index": 255,
            },
        }

    # ---------------- 校验 ----------------

    def validate(
        self, definition: ModelDefinition, inspection: dict[str, Any] | None = None
    ) -> list[str]:
        warnings: list[str] = []
        if definition.task != self.task:
            raise ValidationFailedError(
                f"分割适配器只支持 segmentation 任务，收到 {definition.task}",
                detail={"task": definition.task, "expected": self.task},
            )
        if definition.postprocessing.decoder not in SUPPORTED_DECODERS:
            raise PreprocessConfigError(
                f"分割任务要求 decoder 属于 {SUPPORTED_DECODERS}，收到 "
                f"{definition.postprocessing.decoder}",
                detail={"decoder": definition.postprocessing.decoder},
            )

        if len(definition.input.shape) != 4:
            raise ValidationFailedError(
                "分割模型输入必须是 4D (N, C, H, W)",
                detail={"input_shape": list(definition.input.shape)},
            )
        if definition.input.shape[1] not in (1, 3):
            warnings.append(
                f"输入通道数为 {definition.input.shape[1]}，既不是 1 也不是 3；"
                "请确认预处理中的颜色顺序配置"
            )
        if definition.preprocessing.resize == "letterbox":
            warnings.append(
                "分割任务通常使用 stretch 缩放（letterbox 补边会让掩膜出现无效区域），"
                "若确实需要 letterbox 请确认后处理会用同一套坐标映射还原"
            )

        if inspection:
            outputs = inspection.get("outputs") or []
            if outputs:
                shape = outputs[0].get("shape") or []
                if len(shape) != 4:
                    raise ValidationFailedError(
                        "分割模型输出必须是 4D (N, C, H, W) 的 logits",
                        detail={"output_shape": shape, "output_name": outputs[0].get("name")},
                    )
                channels = shape[1]
                if isinstance(channels, int) and channels > 0:
                    if channels != definition.output.class_count:
                        warnings.append(
                            f"模型输出通道数 {channels} 与 Model Definition 的 class_count "
                            f"{definition.output.class_count} 不一致，将按输出通道数解释类别"
                        )
            inputs = inspection.get("inputs") or []
            if inputs and len(inputs[0].get("shape") or []) != 4:
                warnings.append("模型输入不是 4D，分割任务期望 (N, C, H, W)")

        return warnings

    # ---------------- 代码生成参数 ----------------

    def template_context(self, definition: ModelDefinition) -> dict[str, Any]:
        """给代码生成模板的参数。

        键集与检测适配器**保持一致**：公共模板（generated_config.h / model.yaml / README /
        start 脚本）是共用的，缺键会直接渲染失败（实测踩过：model.input_name 未定义）。
        其中与检测相关的键（box_format / nms_* / has_objectness）在分割里给出"无意义但明确"的值，
        避免模板里出现 Undefined。
        """
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
            # 分割不出框：这些键仍然要给（公共模板会读），值明确表示"不适用"
            "has_objectness": False,
            "box_format": "none",
            "output_name": definition.output.name,
            "output_shape": list(definition.output.shape),
            "decoder": definition.postprocessing.decoder,
            "ignore_index": definition.postprocessing.ignore_index,
            "nms_type": "none",
            "confidence_threshold": 0.0,
            "iou_threshold": 0.0,
            "resize": definition.preprocessing.resize,
            "color": definition.preprocessing.color,
            "scale": definition.preprocessing.scale,
            "mean": definition.preprocessing.mean,
            "std": definition.preprocessing.std,
            "pad_value": definition.preprocessing.pad_value,
        }

    # ---------------- 解码（供运行验证与精度报告使用） ----------------

    def decode(self, outputs: list[np.ndarray], definition: ModelDefinition) -> DecodeResult:
        """把 logits 解码成掩膜，并把掩膜统计放进 `extra`（检测路径不受影响）。"""
        if not outputs:
            raise ValidationFailedError("分割解码收到空输出")

        logits = np.asarray(outputs[0])
        result = segmentation.postprocess_mask(
            logits,
            class_count=definition.output.class_count,
            ignore_index=definition.postprocessing.ignore_index,
        )

        labels = result.labels
        histogram = {
            str(class_id): int(count)
            for class_id, count in zip(*np.unique(labels, return_counts=True), strict=True)
        }
        extra = {
            "mask_shape": list(labels.shape),
            "class_count": result.class_count,
            "class_histogram": histogram,
            "ignored_pixels": (
                int(result.ignored_mask.sum()) if result.ignored_mask is not None else 0
            ),
            "decoder": definition.postprocessing.decoder,
            "notes": result.notes,
        }
        return DecodeResult(detections=[], raw_shape=tuple(logits.shape), candidates=0, extra=extra)

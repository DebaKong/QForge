"""合成语义分割 ONNX（回归用）。

为什么不下载真实模型：阶段 2 的验收要求"每阶段有固定回归模型"（SPEC 17.1），
合成模型体积小、可复现、无网络依赖，能稳定地跑通"生成 → 编译 → 推理 → mIoU 报告"整条链路；
真实模型（联网允许时）另行验证泛化性，两者互补。

结构（故意做成最小可解释的分割网络）：
    images (1,3,H,W) → Conv(3→hidden, 3x3, pad=1) → Relu → Conv(hidden→classes, 1x1) → logits (1,classes,H,W)
输出为 **logits**（未过 softmax），与 `app/services/segmentation.py` 的约定一致。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _conv(
    weights: np.ndarray, bias: np.ndarray, name: str
) -> list[onnx.TensorProto]:  # pragma: no cover - 仅用于生成
    return [
        numpy_helper.from_array(weights.astype(np.float32), f"{name}.weight"),
        numpy_helper.from_array(bias.astype(np.float32), f"{name}.bias"),
    ]


def build_segmentation_onnx(
    path: Path,
    *,
    input_size: int = 64,
    class_count: int = 3,
    hidden: int = 8,
    seed: int = 0,
    opset: int = 17,
) -> Path:
    """生成一个确定性的合成分割模型（同一 seed 结果完全一致）。"""
    rng = np.random.default_rng(seed)

    stem_weight = rng.normal(0, 0.1, (hidden, 3, 3, 3)).astype(np.float32)
    stem_bias = np.zeros(hidden, dtype=np.float32)
    head_weight = rng.normal(0, 0.1, (class_count, hidden, 1, 1)).astype(np.float32)
    head_bias = np.zeros(class_count, dtype=np.float32)

    initializers = [
        *_conv(stem_weight, stem_bias, "stem"),
        *_conv(head_weight, head_bias, "head"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["images", "stem.weight", "stem.bias"],
            ["stem_out"],
            name="stem_conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["stem_out"], ["stem_relu"], name="stem_relu"),
        helper.make_node(
            "Conv",
            ["stem_relu", "head.weight", "head.bias"],
            ["logits"],
            name="head_conv",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "qforge_segmentation_like",
        inputs=[
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, input_size, input_size]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, [1, class_count, input_size, input_size]
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="qforge-synth",
        producer_version="1.0",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)

    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def reference_mask(
    input_size: int = 64, *, class_count: int = 3, bands: int = 3, seed: int = 0
) -> np.ndarray:
    """给合成模型配套的确定性"真值掩膜"（按竖直条带切分），用于 mIoU 回归。

    不是模型的真实输出（那是无意义的随机权重），而是**可复现的评测基准**：
    同一张图 + 同一真值 = 同一指标，才能作为回归基线。
    """
    if bands < 1:
        raise ValueError("bands 必须 >= 1")
    columns = np.arange(input_size)
    labels = (columns * bands // input_size) % class_count
    return np.tile(labels.astype(np.int32), (input_size, 1))

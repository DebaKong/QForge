"""生成「YOLOv8 形状」的合成 ONNX 模型，用于测试与端到端验证。

为什么需要它：真实 YOLOv8 权重较大且不便入库；而验证平台链路（校验 → 量化 →
Engine → 代码生成 → 编译 → 运行）只需要一个**结构正确**的模型：
输入 `[1,3,H,W]`，输出 `[1, 4+class_count, (H/4)*(W/4)]`，即 YOLOv8 的
`[1,84,8400]` 布局的最小等价物。

图结构：Conv(k=1, stride=4) 把 H×W 降到 H/4×W/4，再 Reshape 成三维输出。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DEFAULT_CLASS_COUNT = 80
DEFAULT_INPUT_SIZE = 64
DEFAULT_STRIDE = 4


def build_yolov8_like_onnx(
    path: Path,
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
    class_count: int = DEFAULT_CLASS_COUNT,
    batch: int = 1,
    stride: int = DEFAULT_STRIDE,
    seed: int = 0,
    opset: int = 17,
) -> Path:
    """生成并保存合成模型，返回文件路径。"""
    if input_size % stride != 0:
        raise ValueError("input_size 必须能被 stride 整除")
    channels = 4 + class_count
    spatial = input_size // stride
    anchors = spatial * spatial

    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, 0.1, (channels, 3, 1, 1)).astype(np.float32)
    bias = np.zeros(channels, dtype=np.float32)

    nodes = [
        helper.make_node(
            "Conv",
            ["images", "conv_w", "conv_b"],
            ["features"],
            name="conv",
            kernel_shape=[1, 1],
            strides=[stride, stride],
            pads=[0, 0, 0, 0],
        ),
        helper.make_node(
            "Reshape",
            ["features", "out_shape"],
            ["output0"],
            name="flatten",
        ),
    ]

    initializers = [
        numpy_helper.from_array(weight, "conv_w"),
        numpy_helper.from_array(bias, "conv_b"),
        numpy_helper.from_array(
            np.array([batch, channels, anchors], dtype=np.int64), "out_shape"
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "qforge_yolov8_like",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [batch, 3, input_size, input_size]
            )
        ],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [batch, channels, anchors])],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="qforge-synth",
        producer_version="1.0",
    )
    model.ir_version = 9  # 与较老的 onnxruntime / TensorRT 兼容

    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 YOLOv8 形状的合成 ONNX 模型")
    parser.add_argument("output", type=Path, help="输出 .onnx 路径")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--class-count", type=int, default=DEFAULT_CLASS_COUNT)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()

    target = build_yolov8_like_onnx(
        args.output, input_size=args.input_size, class_count=args.class_count, stride=args.stride
    )
    size_kb = target.stat().st_size / 1024
    print(f"已生成：{target}（{size_kb:.1f} KB，输入 {args.input_size}x{args.input_size}）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

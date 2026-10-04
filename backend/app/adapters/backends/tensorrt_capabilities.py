"""TensorRT 算子能力注册表（SPEC 7.1 步骤 6 / 7.2 兼容性报告）。

为什么单独一个模块：
- TRT 的 ONNX Parser 只能回答"**能不能解析**"；用户真正需要的是"**哪个算子有问题、
  什么条件下有问题、多严重、怎么改**"——即 SPEC 7.2 的七个字段：
  operator / domain / opset / supported / condition / severity / suggestion。
- 这些是**后端私有知识**（换 OpenVINO 就完全不同），所以放在 backend adapter 下，
  由流水线调用，不让上层出现 `if backend == "tensorrt"`。

登记原则（重要）：
1. **只登记有实际依据的条目**：NVIDIA TensorRT 10.x 支持矩阵中的公开限制与已知不可用项；
   写不出依据的算子**不登记**，避免用臆测结论告诉用户"不支持"。
2. 未登记算子按"**未登记**"标注：Parser 解析通过 → 视为支持（info）；
   Parser 解析失败 → 标为 error 并提示查阅对应版本的算子支持矩阵。
3. 结论只做**建议**，最终判定以真实 Parser/Engine 构建结果为准（本平台本来就要求编译与运行验证）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 严重程度（SPEC 7.2 severity），顺序即优先级
SEVERITY_ORDER: tuple[str, ...] = ("error", "warning", "info")
SEVERITY_RANK = {severity: index for index, severity in enumerate(SEVERITY_ORDER)}

# 结论级别
VERDICT_BLOCKED = "BLOCKED"
VERDICT_WARNINGS = "WARNINGS"
VERDICT_OK = "OK"


@dataclass(frozen=True)
class OperatorCapability:
    """单个算子的能力条目。`condition` 写清什么情况下会出问题。"""

    operator: str
    supported: bool = True
    condition: str = ""
    severity: str = "info"
    suggestion: str = ""
    min_opset: int | None = None
    max_opset: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "supported": self.supported,
            "condition": self.condition,
            "severity": self.severity,
            "suggestion": self.suggestion,
            "min_opset": self.min_opset,
            "max_opset": self.max_opset,
        }


def _entry(
    operator: str,
    *,
    supported: bool = True,
    condition: str = "",
    severity: str = "info",
    suggestion: str = "",
    min_opset: int | None = None,
    max_opset: int | None = None,
) -> tuple[str, OperatorCapability]:
    return operator, OperatorCapability(
        operator=operator,
        supported=supported,
        condition=condition,
        severity=severity,
        suggestion=suggestion,
        min_opset=min_opset,
        max_opset=max_opset,
    )


# --------------------------------------------------------------------------- #
# 能力表
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, OperatorCapability] = dict(
    [
        # ---- TensorRT 明确不支持：非确定性 / 序列 / 字符串 / 传统 ML 域 ----
        _entry(
            "RandomNormal",
            supported=False,
            severity="error",
            suggestion="推理引擎不支持随机算子；把随机初始化移出推理图（改为常量输入）",
        ),
        _entry(
            "RandomNormalLike",
            supported=False,
            severity="error",
            suggestion="同上：随机数应在训练阶段完成，推理图里用常量替代",
        ),
        _entry(
            "RandomUniform",
            supported=False,
            severity="error",
            suggestion="同上：随机数应在训练阶段完成，推理图里用常量替代",
        ),
        _entry(
            "RandomUniformLike",
            supported=False,
            severity="error",
            suggestion="同上：随机数应在训练阶段完成，推理图里用常量替代",
        ),
        _entry(
            "Multinomial",
            supported=False,
            severity="error",
            suggestion="采样类算子无法编译进推理引擎；改为在宿主程序里做采样",
        ),
        _entry(
            "SequenceConstruct",
            supported=False,
            severity="error",
            suggestion="TensorRT 不接受 ONNX 序列类型；把序列展开为多个张量输出",
        ),
        _entry(
            "SequenceAt",
            supported=False,
            severity="error",
            suggestion="同上：展开序列类型",
        ),
        _entry(
            "SequenceEmpty",
            supported=False,
            severity="error",
            suggestion="同上：展开序列类型",
        ),
        _entry(
            "SequenceInsert",
            supported=False,
            severity="error",
            suggestion="同上：展开序列类型",
        ),
        _entry(
            "SequenceLength",
            supported=False,
            severity="error",
            suggestion="同上：序列长度在导出时固化为常量",
        ),
        _entry(
            "SplitToSequence",
            supported=False,
            severity="error",
            suggestion="同上：展开序列类型",
        ),
        _entry(
            "ConcatFromSequence",
            supported=False,
            severity="error",
            suggestion="同上：展开序列类型",
        ),
        _entry(
            "StringNormalizer",
            supported=False,
            severity="error",
            suggestion="字符串算子属于 ai.onnx（文本域）能力，推理引擎不提供；放到前后处理里做",
        ),
        _entry(
            "Tokenizer",
            supported=False,
            severity="error",
            suggestion="同上：文本分词请放在宿主程序（Python/C++）里做",
        ),
        _entry(
            "RegexFullMatch",
            supported=False,
            severity="error",
            suggestion="同上：正则匹配请放在宿主程序里做",
        ),
        _entry(
            "LabelEncoder",
            supported=False,
            severity="error",
            suggestion="ai.onnx.ml 域算子不支持；把编码表固化为查表张量或放到宿主程序",
        ),
        _entry(
            "TreeEnsembleClassifier",
            supported=False,
            severity="error",
            suggestion="ai.onnx.ml 域算子不支持；建议把树模型转成等价张量运算或用 TensorRT 自带插件",
        ),
        _entry(
            "TreeEnsembleRegressor",
            supported=False,
            severity="error",
            suggestion="同上：ai.onnx.ml 域算子不支持",
        ),
        # ---- 支持但有条件（容易在上板后才发现，提前提示）----
        _entry(
            "NonMaxSuppression",
            supported=True,
            condition=(
                "TRT 10 通过 EfficientNMS 插件路径支持；坐标格式、iou 阈值与 max_output_boxes "
                "需为常量，且 opset>=11 才常见可用"
            ),
            severity="warning",
            suggestion=(
                "更稳的做法是把 NMS 移到模型外：本平台生成的推理工程已在 C++ 后处理里做 NMS"
                "（见 source/src/yolov8_decoder.cpp），导出 ONNX 时可去掉该节点"
            ),
            min_opset=11,
        ),
        _entry(
            "TopK",
            supported=True,
            condition="K 必须是常量（TensorRT 不支持运行期动态 K）；且只能对最后一维取 TopK",
            severity="warning",
            suggestion="把 K 固化为常量（导出时写死），动态 K 改为取固定上限后再截断",
        ),
        _entry(
            "NonZero",
            supported=True,
            condition="输出 shape 依赖数据，会引入动态形状，需要 Dynamic Shape Profile 才能构建 Engine",
            severity="warning",
            suggestion="若可能，改用固定 shape 的等价写法；否则为目标机配置最小/最优/最大 shape profile",
        ),
        _entry(
            "Resize",
            supported=True,
            condition=(
                "仅支持部分 coordinate_transformation_mode（如 half_pixel / align_corners / "
                "asymmetric）与 nearest / linear 模式"
            ),
            severity="warning",
            suggestion="导出前确认 Resize 的 mode 与坐标变换模式在支持列表内，否则改用固定尺寸预处理",
        ),
        _entry(
            "GridSample",
            supported=True,
            condition="仅支持 2D、bilinear/nearest 采样；5D 输入与部分 padding 模式不支持",
            severity="warning",
            suggestion="确认输入为 4D（NCHW）且采样模式为 bilinear/nearest",
        ),
        _entry(
            "ScatterND",
            supported=True,
            condition="reduction 仅支持 none/add/mul 等子集，索引 dtype 需为 int32/int64",
            severity="warning",
            suggestion="把 reduction 改为 none 并在图中显式相加，或确认所用子集在该 TRT 版本可用",
        ),
        _entry(
            "ScatterElements",
            supported=True,
            condition="同 ScatterND：reduction 与索引 dtype 受限",
            severity="warning",
            suggestion="同上",
        ),
        _entry(
            "Einsum",
            supported=True,
            condition="TRT 10 支持常见等式，但部分广播组合与高阶张量不受支持",
            severity="warning",
            suggestion="若构建报错，把 Einsum 拆成 MatMul + Transpose 等基础算子",
        ),
        _entry(
            "ArgMax",
            supported=True,
            condition="只支持沿单一轴求极值，且输出为 int（后续若参与浮点运算需显式 Cast）",
            severity="warning",
            suggestion="确认 axis 为常量且后续算子能接受整型张量",
        ),
        _entry(
            "ArgMin",
            supported=True,
            condition="同 ArgMax",
            severity="warning",
            suggestion="同上",
        ),
        _entry(
            "LSTM",
            supported=True,
            condition="受 layout、双向与 activation 组合限制；权重需为常量",
            severity="warning",
            suggestion="优先导出为单向、layout=0 的 LSTM；必要时手工展开为矩阵运算",
        ),
        _entry(
            "GRU",
            supported=True,
            condition="同 LSTM",
            severity="warning",
            suggestion="同上",
        ),
        _entry(
            "RNN",
            supported=True,
            condition="同 LSTM",
            severity="warning",
            suggestion="同上",
        ),
        _entry(
            "Loop",
            supported=True,
            condition="循环次数需可静态推导（常量 trip count）；循环体内含数据依赖形状时无法构建",
            severity="warning",
            suggestion="尽量把循环展开（unroll）后再导出",
        ),
        _entry(
            "Scan",
            supported=True,
            condition="同 Loop：迭代次数与状态形状需静态可推导",
            severity="warning",
            suggestion="同上：优先展开循环",
        ),
        _entry(
            "If",
            supported=True,
            condition="条件必须是常量或可静态求值；两个分支都必须能被 Parser 解析",
            severity="warning",
            suggestion="导出时用常量折叠消掉分支，或确保两个分支都只含受支持算子",
        ),
        _entry(
            "RoiAlign",
            supported=True,
            condition="sampling_ratio、coordinate_transformation_mode 等属性受支持子集限制",
            severity="warning",
            suggestion="按 TensorRT 支持矩阵核对属性；或用等价的自定义裁剪+Resize 实现",
        ),
        _entry(
            "Pad",
            supported=True,
            condition="mode=constant/edge/reflect 常见可用；mode=wrap 在部分版本不可用",
            severity="warning",
            suggestion="避免 wrap 模式；padding 值需为常量",
        ),
        _entry(
            "Cast",
            supported=True,
            condition="INT64/DOUBLE 参与计算时会退化或需要额外 Cast；TRT 以 FP32/FP16/INT32 为主",
            severity="warning",
            suggestion="把索引/形状类张量保持 INT32，避免在主干里出现 INT64 张量运算",
        ),
        _entry(
            "Softmax",
            supported=True,
            condition="对超大维度做 Softmax 可能因 workspace 不足失败（可调大 memory pool）",
            severity="info",
            suggestion="构建失败时提高 workspace 上限（平台已按精度模式设置默认值）",
        ),
    ]
)

# 未登记算子的说明模板
UNREGISTERED_NOTE = "未在本平台能力表中登记"


@dataclass
class CapabilityReport:
    """兼容性报告（SPEC 7.2）。"""

    items: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"operators": self.items, "summary": self.summary}


def _severity_for(capability: OperatorCapability, opset: int | None) -> tuple[bool, str, str]:
    """结合 opset 判定最终 (supported, severity, condition)。"""
    supported = capability.supported
    severity = capability.severity
    condition = capability.condition

    if opset is not None and supported:
        too_low = capability.min_opset is not None and opset < capability.min_opset
        too_high = capability.max_opset is not None and opset > capability.max_opset
        if too_low or too_high:
            bound = (
                f"该算子需要 opset>={capability.min_opset}"
                if too_low
                else f"该算子仅支持 opset<={capability.max_opset}"
            )
            supported = False
            severity = "error"
            condition = f"{bound}（当前 opset={opset}）"
    return supported, severity, condition


def evaluate(
    operators: list[dict[str, Any]],
    *,
    opset: int | None = None,
    backend_name: str = "tensorrt",
    backend_version: str | None = None,
    parser_accepted: bool = True,
) -> CapabilityReport:
    """把 ONNX 图算子清单评估成 SPEC 7.2 兼容性报告。

    `parser_accepted=False` 表示目标后端 Parser 已经解析失败：此时未登记算子按 error 处理，
    避免把"解析失败"误报成"全部支持"。
    """
    items: list[dict[str, Any]] = []
    for entry in operators:
        operator = str(entry.get("operator") or entry.get("op_type") or entry.get("name") or "")
        if not operator:
            continue
        domain = str(entry.get("domain") or "ai.onnx")
        occurrences = int(entry.get("count") or 1)

        capability = _REGISTRY.get(operator)
        if capability is None:
            # 未登记：不臆断。Parser 通过则以实测为准（info），失败则提示查阅支持矩阵。
            supported = bool(parser_accepted)
            item = {
                "operator": operator,
                "domain": domain,
                "opset": opset,
                "supported": supported,
                "condition": "",
                "severity": "info" if parser_accepted else "error",
                "suggestion": (
                    ""
                    if parser_accepted
                    else f"{UNREGISTERED_NOTE}且目标后端解析失败：请查阅该后端版本的算子支持矩阵，或替换为等价算子"
                ),
                "registered": False,
                "occurrences": occurrences,
            }
        else:
            supported, severity, condition = _severity_for(capability, opset)
            item = {
                "operator": operator,
                "domain": domain,
                "opset": opset,
                "supported": supported,
                "condition": condition,
                "severity": severity,
                "suggestion": capability.suggestion,
                "registered": True,
                "occurrences": occurrences,
            }
        items.append(item)

    # 问题算子排在前面：error → warning → info，同级按出现次数降序
    items.sort(
        key=lambda item: (
            SEVERITY_RANK.get(str(item["severity"]), len(SEVERITY_ORDER)),
            -int(item["occurrences"]),
            str(item["operator"]),
        )
    )

    errors = [item for item in items if item["severity"] == "error"]
    warnings = [item for item in items if item["severity"] == "warning"]
    verdict = VERDICT_BLOCKED if errors else (VERDICT_WARNINGS if warnings else VERDICT_OK)

    summary = {
        "backend": backend_name,
        "backend_version": backend_version,
        "opset": opset,
        "parser_accepted": bool(parser_accepted),
        "verdict": verdict,
        "operator_kinds": len(items),
        "operator_instances": sum(int(item["occurrences"]) for item in items),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "unsupported": [item["operator"] for item in errors],
        "conditional": [item["operator"] for item in warnings],
        "unregistered": [item["operator"] for item in items if not item["registered"]],
    }
    return CapabilityReport(items=items, summary=summary)


def registered_operators() -> list[str]:
    """能力表里登记过的算子名（用于自检与文档）。"""
    return sorted(_REGISTRY)

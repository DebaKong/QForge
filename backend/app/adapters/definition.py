"""Model Definition 数据结构与校验（SPEC 6.1）。

平台**不通过 ONNX Shape 猜测后处理语义**（AGENTS.md 边界）：模型语义必须显式声明，
本模块负责把声明解析成强类型对象，并在与 ONNX 图不一致时给出明确错误。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.errors import PreprocessConfigError

SUPPORTED_RESIZE = ("letterbox", "stretch", "center_crop")
SUPPORTED_COLOR = ("RGB", "BGR")
SUPPORTED_LAYOUT = ("NCHW", "NHWC")
SUPPORTED_BOX_FORMAT = ("xywh", "xyxy", "cxcywh")
SUPPORTED_NMS = ("classwise", "class_agnostic", "none")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreprocessConfigError(
            f"Model Definition 的 {name} 必须是对象", detail={"field": name, "value": value}
        )
    return value


def _as_float_list(value: Any, *, name: str, expected_len: int | None = None) -> list[float]:
    if value is None:
        raise PreprocessConfigError(f"{name} 不能为空", detail={"field": name})
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # 允许标量：mean: 0.5 / std: 2.0，等价于在每个通道上使用同一个值
        return [float(value)]
    if not isinstance(value, (list, tuple)):
        raise PreprocessConfigError(f"{name} 必须是数组", detail={"field": name, "value": value})
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise PreprocessConfigError(
            f"{name} 必须是数值数组", detail={"field": name, "value": value}
        ) from exc
    if expected_len is not None and len(numbers) not in (1, expected_len):
        raise PreprocessConfigError(
            f"{name} 长度必须是 1 或 {expected_len}", detail={"field": name, "value": value}
        )
    return numbers


@dataclass
class InputSpec:
    """模型输入语义（SPEC 6.1 input）。"""

    name: str
    shape: tuple[int, ...]
    layout: str = "NCHW"
    dtype: str = "FP32"

    @property
    def height(self) -> int:
        return self.shape[2] if self.layout == "NCHW" else self.shape[1]

    @property
    def width(self) -> int:
        return self.shape[3] if self.layout == "NCHW" else self.shape[2]

    @property
    def channels(self) -> int:
        return self.shape[1] if self.layout == "NCHW" else self.shape[3]

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> "InputSpec":
        payload = _require_mapping(payload, "input")
        name = payload.get("name")
        if not name or not isinstance(name, str):
            raise PreprocessConfigError("input.name 必填", detail={"field": "input.name"})

        shape = payload.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 4:
            raise PreprocessConfigError(
                "阶段 1（MVP）只支持 4 维输入 shape，如 [1,3,640,640]",
                detail={"field": "input.shape", "value": shape},
            )
        try:
            dims = tuple(int(item) for item in shape)
        except (TypeError, ValueError) as exc:
            raise PreprocessConfigError(
                "input.shape 必须是整数数组", detail={"value": shape}
            ) from exc
        if any(dim <= 0 for dim in dims):
            raise PreprocessConfigError(
                "阶段 1 只支持固定 shape（SPEC 10.1：MVP 默认 batch=1 固定输入）",
                detail={"value": shape},
            )

        layout = str(payload.get("layout", "NCHW")).upper()
        if layout not in SUPPORTED_LAYOUT:
            raise PreprocessConfigError(
                f"input.layout 必须是 {SUPPORTED_LAYOUT} 之一", detail={"value": layout}
            )
        return cls(name=name, shape=dims, layout=layout, dtype=str(payload.get("dtype", "FP32")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "layout": self.layout,
            "dtype": self.dtype,
        }


@dataclass
class PreprocessSpec:
    """预处理规则（SPEC 6.1 preprocessing / 8.1）。"""

    resize: str = "letterbox"
    color: str = "RGB"
    scale: float = 255.0
    mean: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    pad_value: float = 114.0

    @classmethod
    def parse(cls, payload: dict[str, Any] | None, *, channels: int = 3) -> "PreprocessSpec":
        payload = _require_mapping(payload or {}, "preprocessing")
        resize = str(payload.get("resize", "letterbox")).lower()
        if resize not in SUPPORTED_RESIZE:
            raise PreprocessConfigError(
                f"preprocessing.resize 必须是 {SUPPORTED_RESIZE} 之一", detail={"value": resize}
            )
        color = str(payload.get("color", "RGB")).upper()
        if color not in SUPPORTED_COLOR:
            raise PreprocessConfigError(
                f"preprocessing.color 必须是 {SUPPORTED_COLOR} 之一", detail={"value": color}
            )

        try:
            scale = float(payload.get("scale", 255.0))
        except (TypeError, ValueError) as exc:
            raise PreprocessConfigError(
                "preprocessing.scale 必须是数值", detail={"value": payload.get("scale")}
            ) from exc
        if scale == 0:
            raise PreprocessConfigError("preprocessing.scale 不能为 0")

        mean = (
            _as_float_list(payload.get("mean"), name="preprocessing.mean", expected_len=channels)
            if payload.get("mean") is not None
            else [0.0] * channels
        )
        std = (
            _as_float_list(payload.get("std"), name="preprocessing.std", expected_len=channels)
            if payload.get("std") is not None
            else [1.0] * channels
        )
        if any(item == 0 for item in std):
            raise PreprocessConfigError("preprocessing.std 不能包含 0")
        if len(mean) == 1:
            mean = mean * channels
        if len(std) == 1:
            std = std * channels

        return cls(
            resize=resize,
            color=color,
            scale=scale,
            mean=mean,
            std=std,
            pad_value=float(payload.get("pad_value", 114.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resize": self.resize,
            "color": self.color,
            "scale": self.scale,
            "mean": self.mean,
            "std": self.std,
            "pad_value": self.pad_value,
        }


@dataclass
class NmsSpec:
    type: str = "classwise"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    @classmethod
    def parse(cls, payload: dict[str, Any] | None) -> "NmsSpec":
        payload = _require_mapping(payload or {}, "postprocessing.nms")
        nms_type = str(payload.get("type", "classwise")).lower()
        if nms_type not in SUPPORTED_NMS:
            raise PreprocessConfigError(
                f"postprocessing.nms.type 必须是 {SUPPORTED_NMS} 之一", detail={"value": nms_type}
            )
        confidence = float(payload.get("confidence_threshold", 0.25))
        iou = float(payload.get("iou_threshold", 0.45))
        if not 0.0 < confidence <= 1.0:
            raise PreprocessConfigError(
                "confidence_threshold 必须落在 (0, 1]", detail={"value": confidence}
            )
        if not 0.0 < iou <= 1.0:
            raise PreprocessConfigError(
                "iou_threshold 必须落在 (0, 1]", detail={"value": iou}
            )
        return cls(type=nms_type, confidence_threshold=confidence, iou_threshold=iou)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
        }


@dataclass
class OutputSpec:
    """模型输出语义（SPEC 6.1 output）。"""

    name: str
    shape: tuple[int, ...]
    format: str | None = None
    box_format: str = "xywh"
    has_objectness: bool = False
    class_count: int = 80

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> "OutputSpec":
        payload = _require_mapping(payload, "output")
        name = payload.get("name")
        if not name or not isinstance(name, str):
            raise PreprocessConfigError("output.name 必填", detail={"field": "output.name"})

        raw_format = payload.get("format")
        shape: tuple[int, ...] = ()
        if isinstance(raw_format, str) and raw_format.strip().startswith("["):
            try:
                shape = tuple(int(item) for item in raw_format.strip("[]").split(",") if item.strip())
            except ValueError as exc:
                raise PreprocessConfigError(
                    "output.format 形如 [1,84,8400]", detail={"value": raw_format}
                ) from exc
        elif isinstance(raw_format, (list, tuple)):
            shape = tuple(int(item) for item in raw_format)

        box_format = str(payload.get("box_format", "xywh")).lower()
        if box_format not in SUPPORTED_BOX_FORMAT:
            raise PreprocessConfigError(
                f"output.box_format 必须是 {SUPPORTED_BOX_FORMAT} 之一", detail={"value": box_format}
            )

        class_count = int(payload.get("class_count", 80))
        if class_count <= 0:
            raise PreprocessConfigError("output.class_count 必须为正整数")

        return cls(
            name=name,
            shape=shape,
            format=raw_format if isinstance(raw_format, str) else None,
            box_format=box_format,
            has_objectness=bool(payload.get("has_objectness", False)),
            class_count=class_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format,
            "box_format": self.box_format,
            "has_objectness": self.has_objectness,
            "class_count": self.class_count,
        }


@dataclass
class PostprocessSpec:
    decoder: str
    nms: NmsSpec = field(default_factory=NmsSpec)

    @classmethod
    def parse(cls, payload: dict[str, Any] | None) -> "PostprocessSpec":
        payload = _require_mapping(payload or {}, "postprocessing")
        decoder = str(payload.get("decoder", "yolov8")).lower()
        if not decoder:
            raise PreprocessConfigError("postprocessing.decoder 必填")
        return cls(decoder=decoder, nms=NmsSpec.parse(payload.get("nms")))

    def to_dict(self) -> dict[str, Any]:
        return {"decoder": self.decoder, "nms": self.nms.to_dict()}


@dataclass
class ModelDefinition:
    """完整的模型语义（SPEC 6.1）。"""

    task: str
    architecture: str
    input: InputSpec
    preprocessing: PreprocessSpec
    output: OutputSpec
    postprocessing: PostprocessSpec

    @classmethod
    def parse(cls, payload: dict[str, Any] | None) -> "ModelDefinition":
        payload = _require_mapping(payload, "model_definition")
        task = str(payload.get("task", "detection")).lower()
        architecture = str(payload.get("architecture", "")).lower()
        if not architecture:
            raise PreprocessConfigError(
                "model_definition.architecture 必填（如 yolov8）", detail={"field": "architecture"}
            )

        input_spec = InputSpec.parse(payload.get("input") or {})
        preprocessing = PreprocessSpec.parse(
            payload.get("preprocessing"), channels=input_spec.channels
        )
        output_spec = OutputSpec.parse(payload.get("output") or {})
        postprocessing = PostprocessSpec.parse(payload.get("postprocessing"))

        return cls(
            task=task,
            architecture=architecture,
            input=input_spec,
            preprocessing=preprocessing,
            output=output_spec,
            postprocessing=postprocessing,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "architecture": self.architecture,
            "input": self.input.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "output": self.output.to_dict(),
            "postprocessing": self.postprocessing.to_dict(),
        }

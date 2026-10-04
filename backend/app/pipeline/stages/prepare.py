"""准备阶段：模型校验 / 预处理一致性 / 量化校准准备（SPEC 7 / 8 / 9）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.adapters.definition import ModelDefinition
from app.adapters.registry import supported_architectures
from app.db.base import session_scope
from app.errors import (
    CalibrationDataError,
    ModelInvalidError,
    QuantizationFailedError,
    ValidationFailedError,
)
from app.models.onnx_model import OnnxModel
from app.pipeline.context import PipelineContext
from app.services import calibration, onnx_inspector, qdq_quantization
from app.services.layout import place_input_file
from app.services.preprocess import load_image_rgb, preprocess, preprocess_file

logger = logging.getLogger("qforge.pipeline")


def write_json(path: Path, payload: Any) -> None:
    """把结构化报告写入任务 report 目录（UTF-8、缩进、保留中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def record_artifact(
    context: PipelineContext,
    *,
    kind: str,
    path: Path,
    description: str,
) -> None:
    context.artifacts.append(
        {"kind": kind, "path": str(path), "description": description}
    )


# ---------------- VALIDATING ----------------


def run_validation(context: PipelineContext) -> None:
    """SPEC 7.1 校验层级 1~6。"""
    if context.onnx_path is None or not context.onnx_path.exists():
        raise ModelInvalidError(
            "任务未绑定可用的 ONNX 模型文件（请先生成模型并上传）",
            detail={"model_id": context.model_id, "expected_path": str(context.onnx_path)},
        )

    # 任务自包含：把模型安置到 storage/<task_id>/input/（优先硬链接）
    placed = place_input_file(context.onnx_path, context.input_dir / context.onnx_path.name)
    logger.info("模型已安置到任务目录（%s）：%s", placed, context.onnx_path.name)

    inspection = onnx_inspector.inspect(context.onnx_path)
    context.inspection = inspection.to_dict()

    with session_scope() as session:
        if context.model_id:
            row = session.get(OnnxModel, context.model_id)
            if row is not None:
                row.input_spec = inspection.inputs
                row.output_spec = inspection.outputs
                row.opset = inspection.opset
                row.size_bytes = context.onnx_path.stat().st_size

    if context.model_adapter is None:
        raise ValidationFailedError(
            "任务未绑定受支持的模型架构，无法确定模型语义",
            detail={
                "architecture": context.architecture,
                "supported": supported_architectures(),
                "task_id": context.task_id,
            },
        )

    if context.definition is None:
        payload = context.model_adapter.default_definition(context.inspection)
        context.definition = ModelDefinition.parse(payload)
        logger.warning(
            "任务配置缺少 Model Definition，已按架构默认定义执行（架构=%s）",
            context.architecture,
        )

    warnings = context.model_adapter.validate(context.definition, context.inspection)
    for warning in warnings:
        logger.warning("模型定义提示：%s", warning)

    # 后端能力检查（SPEC 7.1 步骤 5~6）：后端不可用不在此阶段失败，但必须如实记录
    capability: dict[str, Any]
    assert context.backend_adapter is not None
    available, environment = context.backend_adapter.availability()
    if available:
        capability = context.backend_adapter.capability_check(  # type: ignore[attr-defined]
            context.onnx_path, log_path=context.log_file("trt_parser.log")
        )
        logger.info(
            "TensorRT 解析通过：网络层数 %s，输入 %s，输出 %s",
            capability.get("network_layers"),
            [item["name"] for item in capability.get("network_inputs", [])],
            [item["name"] for item in capability.get("network_outputs", [])],
        )
    else:
        capability = {
            "parsed": False,
            "reason": environment.get("reason", "后端不可用"),
            "environment": environment,
        }
        logger.warning("后端不可用，跳过 TensorRT 解析检查：%s", capability["reason"])

    # 算子兼容性报告（SPEC 7.1 步骤 6 / 7.2）：把 ONNX 图算子清单与目标后端的能力表对起来，
    # 给出 supported / condition / severity / suggestion —— 而不是只报"解析成功/失败"。
    compatibility = context.backend_adapter.operator_capabilities(context.inspection, capability)
    compatibility_summary = compatibility.get("summary", {})
    logger.info(
        "算子兼容性结论：%s（错误 %s，警告 %s，算子种类 %s）",
        compatibility_summary.get("verdict"),
        compatibility_summary.get("error_count"),
        compatibility_summary.get("warning_count"),
        compatibility_summary.get("operator_kinds"),
    )

    write_json(
        context.report_file("model_info.json"),
        {
            "task_id": context.task_id,
            "model_file": context.onnx_path.name,
            "model_sha256": None,
            "inspection": context.inspection,
            "model_definition": context.definition.to_dict(),
            "warnings": warnings,
        },
    )
    write_json(
        context.report_file("compatibility.json"),
        {
            "task_id": context.task_id,
            "backend": context.backend_name,
            "operators": compatibility.get("operators", []),
            "summary": compatibility_summary,
            # 后端 Parser 的原始细节（网络层数、输入输出、parser 错误）保留，便于排障
            "capability": capability,
        },
    )
    context.reports.extend(["report/model_info.json", "report/compatibility.json"])


# ---------------- PREPROCESSING ----------------


def _prepare_sample_input(context: PipelineContext) -> dict[str, Any]:
    """准备运行验证用的样例输入（PPM 图像 + 平台预处理后的张量）。"""
    assert context.definition is not None

    if context.calibration_images:
        source_image = context.calibration_images[0]
        image = load_image_rgb(source_image)
        origin = f"校准集首张图像 {source_image.name}"
    else:
        height = context.definition.input.height
        width = context.definition.input.width
        # 无校准集时使用确定性合成图（全灰），仅用于验证链路可跑通，不作为精度依据
        image = np.full((height, width, 3), 114, dtype=np.uint8)
        origin = "合成灰底图像（无校准集，仅用于链路验证）"

    result = preprocess(image, context.definition)

    ppm_path = context.intermediate_dir / "test_input.ppm"
    ppm_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(ppm_path, format="PPM")

    tensor_path = context.intermediate_dir / "test_input.f32"
    np.ascontiguousarray(result.tensor, dtype=np.float32).tofile(tensor_path)

    reference_path = context.intermediate_dir / "reference_output.f32"
    return {
        "origin": origin,
        "ppm": ppm_path.name,
        "tensor": tensor_path.name,
        "reference_tensor": reference_path.name,
        "letterbox_scale": result.scale,
        "pad": list(result.pad),
        "original_hw": list(result.original_hw),
        "tensor_shape": list(result.tensor.shape),
    }


def run_preprocessing(context: PipelineContext) -> None:
    """校准集校验（SPEC 8.3）+ 统一预处理（SPEC 8.1）。"""
    assert context.definition is not None
    settings = context.settings

    report: dict[str, Any] = {"task_id": context.task_id, "precision": context.precision}

    if context.calibration_dataset_root and context.calibration_dataset_root.exists():
        calibration_report = calibration.validate_calibration_set(
            context.calibration_dataset_root,
            context.definition,
            allowed_extensions=settings.allowed_image_extensions,
            max_images=settings.max_calibration_images,
            sample_size=8,
        )
        context.calibration_images = calibration.scan_images(
            context.calibration_dataset_root,
            settings.allowed_image_extensions,
            max_images=settings.max_calibration_images,
        )
        report["calibration"] = calibration_report.to_dict()
        logger.info(
            "校准集校验通过：%s 张图像，坏图 %s 张",
            calibration_report.image_count,
            len(calibration_report.bad_files),
        )
    elif context.precision == "int8":
        raise CalibrationDataError(
            "INT8 为静态校准量化，必须提供校准数据集（SPEC 9）",
            detail={"task_id": context.task_id, "dataset_id": context.dataset_id},
        )
    else:
        report["calibration"] = None
        logger.info("未提供校准集（精度 %s 不要求），跳过校准集校验", context.precision)

    # 把本任务实际使用的校准集清单落盘，保证可追溯（SPEC 13.2：能回答用了什么数据）
    if context.calibration_images:
        listing = context.calibration_dir / "calibration_list.txt"
        listing.write_text(
            "\n".join(str(path) for path in context.calibration_images), encoding="utf-8"
        )
        report["calibration_list"] = {
            "file": "calibration/calibration_list.txt",
            "count": len(context.calibration_images),
        }

    report["sample_input"] = _prepare_sample_input(context)
    logger.info("样例输入已准备：%s", report["sample_input"]["origin"])

    write_json(context.report_file("preprocessing.json"), report)
    context.reports.append("report/preprocessing.json")


# ---------------- QUANTIZING ----------------


def run_quantization(context: PipelineContext) -> None:
    """量化阶段。

    INT8 默认走**显式量化（Q/DQ）**：本阶段用校准集生成带 QuantizeLinear / DequantizeLinear
    的 ONNX（量化参数固化在图里），BUILDING_ENGINE 阶段直接解析它，不再使用 TensorRT 10.1 起
    已弃用的 `IInt8Calibrator`。保留 `quantization.mode=calibrator` 旧路径用于对比与回退验证，
    两条路径的端到端精度都会写进报告。
    """
    assert context.definition is not None
    settings = context.settings

    if context.precision != "int8":
        payload = {
            "task_id": context.task_id,
            "applied": False,
            "precision": context.precision,
            "reason": f"{context.precision.upper()} 不需要量化校准，本阶段无操作",
        }
        logger.info("精度 %s 无需量化，跳过校准", context.precision.upper())
        write_json(context.report_file("quantization.json"), payload)
        context.reports.append("report/quantization.json")
        return

    if not context.calibration_images:
        raise CalibrationDataError("INT8 校准缺少可用图像", detail={"task_id": context.task_id})

    options = context.task_config.get("quantization") or {}
    # 默认 qdq（显式量化，SPEC 9.1 升级路径，TensorRT 官方推荐）：实测端到端精度明显优于旧熵校准。
    # 需要与旧路径对比时显式传 quantization.mode=calibrator。
    mode = str(options.get("mode") or "qdq").lower()
    if mode not in ("qdq", "calibrator"):
        logger.warning("未知的 quantization.mode=%s，按 qdq 处理", mode)
        mode = "qdq"

    if mode == "calibrator":
        # 旧路径（保留用于对比）：真正的校准发生在 Engine 构建时，由 TensorRT 回调读取校准数据
        cache_path = context.intermediate_dir / settings.calibration_cache_filename
        batch_size = min(settings.calibration_batch_size, max(1, len(context.calibration_images)))
        payload = {
            "task_id": context.task_id,
            "applied": True,
            "precision": "INT8",
            "mode": "calibrator",
            "method": "TensorRT IInt8EntropyCalibrator2（静态熵校准，已弃用路径）",
            "note": "设计变更：SPEC 原指定 PPQ，PPQ 0.6.6 与本机工具链无法共存，详见 docs/phase-1.md",
            "images": len(context.calibration_images),
            "batch_size": batch_size,
            "cache_file": f"intermediate/{cache_path.name}",
            "executed_during": "BUILDING_ENGINE（由 TensorRT 回调读取校准数据）",
        }
        logger.info(
            "INT8 校准计划（calibrator 路径）：%s 张图像，batch=%s，缓存 %s",
            len(context.calibration_images),
            batch_size,
            cache_path.name,
        )
        write_json(context.report_file("quantization.json"), payload)
        context.reports.append("report/quantization.json")
        return

    # ---- Q/DQ 显式量化（默认路径）----
    method = str(options.get("method") or qdq_quantization.DEFAULT_METHOD)
    per_channel = bool(options.get("per_channel", False))
    samples: list[np.ndarray] = []
    for image in list(context.calibration_images)[: qdq_quantization.MAX_CALIBRATION_SAMPLES]:
        try:
            samples.append(
                np.asarray(preprocess_file(Path(image), context.definition).tensor, dtype=np.float32)
            )
        except Exception:  # noqa: BLE001 - 单张坏图跳过（校验阶段已报告过坏图）
            logger.warning("量化：校准图像预处理失败，已跳过 %s", image)
    if not samples:
        raise CalibrationDataError(
            "INT8 显式量化没有任何可用的校准样本", detail={"task_id": context.task_id}
        )

    qdq_path = context.intermediate_dir / "model_qdq.onnx"
    result = qdq_quantization.quantize(
        context.onnx_path,
        qdq_path,
        samples,
        input_name=context.definition.input.name,
        method=method,
        per_channel=per_channel,
    )
    if result.status != "SUCCESS":
        # 不静默退回旧路径：报告会失真。要对比就显式设置 quantization.mode=calibrator。
        raise QuantizationFailedError(
            f"INT8 显式量化（Q/DQ）失败：{result.reason}",
            detail=result.to_dict(),
        )

    context.quantized_onnx_path = qdq_path
    payload = {
        "task_id": context.task_id,
        "applied": True,
        "precision": "INT8",
        "mode": "qdq",
        "method": f"Q/DQ 显式量化（onnxruntime 静态量化，校准方法 {result.method}）",
        "note": (
            "TensorRT 10.1 起 IInt8Calibrator 已弃用，Q/DQ 是官方推荐路径；"
            "需要与旧路径对比时设置 quantization.mode=calibrator"
        ),
        "images": len(context.calibration_images),
        "samples_used": result.samples,
        "per_channel": result.per_channel,
        "weight_type": result.weight_type,
        "activation_type": result.activation_type,
        "qdq_model": f"intermediate/{qdq_path.name}",
        "quantize_nodes": result.quantize_nodes,
        "dequantize_nodes": result.dequantize_nodes,
        "total_nodes": result.total_nodes,
        "model_size_bytes": result.size_bytes,
        "executed_during": "QUANTIZING（Engine 构建期直接解析 Q/DQ 图，无校准回调）",
        "notes": result.notes,
    }
    logger.info(
        "INT8 显式量化完成：%s（%s 个 QuantizeLinear 节点，样本 %s）",
        qdq_path.name,
        result.quantize_nodes,
        result.samples,
    )
    write_json(context.report_file("quantization.json"), payload)
    context.reports.append("report/quantization.json")

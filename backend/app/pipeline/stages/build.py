"""构建阶段：Engine 构建 / 代码生成 / 编译 / 运行验证（SPEC 10 / 11）。

关于「编译验证」的策略（设计决策，已记录在 docs/phase-1.md）：
SPEC 11.3 要求生成的工程必须编译、加载 Engine 并真实推理。编译需要
TensorRT 开发文件（头文件 + 导入库）与 CUDA 运行时开发文件；当本机缺失时，
任务配置 `build.cpp_build` 决定行为：

- `required`：缺文件即 FAILED（BACKEND_UNAVAILABLE）——最严格；
- `auto`（默认）：标记 `cpp_verification: BLOCKED` 并继续，报告中明确写出缺什么，
  绝不把「没编译」写成「编译通过」；
- `skip`：明确跳过（只做 Engine 与 Python 侧验证）。

无论哪种模式，**Engine 都会用 TensorRT Python API 真实加载并推理**，
并与 onnxruntime FP32 基准比较，产出 SPEC 9.2 要求的精度指标。
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.adapters.backends import tensorrt_dev
from app.codegen import generate_project
from app.codegen.docker import generate_dockerfile
from app.errors import (
    BackendUnavailableError,
    RuntimeTestFailedError,
)
from app.pipeline.context import PipelineContext
from app.pipeline.stages.prepare import record_artifact, write_json
from app.services import opencv_dev, toolchain
from app.services.layout import place_input_file

logger = logging.getLogger("qforge.pipeline")

EXECUTABLE_NAME = "qforge_detector"
CPP_BUILD_MODES = ("auto", "required", "skip")


# ---------------- BUILDING_ENGINE ----------------


def run_engine_build(context: PipelineContext) -> None:
    """构建 Engine（SPEC 10.1：记录 TensorRT / CUDA / GPU 架构 / 精度 / 输入 profile）。"""
    assert context.backend_adapter is not None
    assert context.definition is not None
    assert context.onnx_path is not None

    available, environment = context.backend_adapter.availability()
    logger.info("后端环境：%s", json.dumps(environment, ensure_ascii=False, default=str))
    if not available:
        raise BackendUnavailableError(
            "目标后端在当前环境不可用，无法构建 Engine",
            detail={"environment": environment, "backend": context.backend_name},
        )

    engine_path = context.engine_dir / "model.engine"
    build_log = context.log_file("engine_build.log")
    workspace_bytes = context.settings.trt_workspace_gb * 1024**3

    metadata = context.backend_adapter.build_engine(
        onnx_path=context.onnx_path,
        engine_path=engine_path,
        definition=context.definition,
        precision=context.precision,
        calibration_images=context.calibration_images or None,
        workspace_bytes=workspace_bytes,
        calibration_cache=context.intermediate_dir
        / context.settings.calibration_cache_filename,
        log_path=build_log,
    )

    context.engine_path = engine_path
    context.engine_metadata = metadata
    record_artifact(context, kind="engine", path=engine_path, description="TensorRT Engine")

    write_json(
        context.report_file("engine_build.json"),
        {
            "task_id": context.task_id,
            "engine": engine_path.name,
            "engine_metadata": metadata,
            "build_log": "logs/engine_build.log",
        },
    )
    context.reports.append("report/engine_build.json")
    logger.info(
        "Engine 构建完成：%s（%.2f MB）",
        engine_path.name,
        metadata.get("engine_size_bytes", 0) / 1024 / 1024,
    )


# ---------------- GENERATING_CODE ----------------


def run_codegen(context: PipelineContext) -> None:
    """生成 C++ 推理工程与 Dockerfile（SPEC 11.2 / 12.2）。"""
    assert context.definition is not None
    assert context.model_adapter is not None
    assert context.backend_adapter is not None
    assert context.engine_path is not None

    backend_context = context.backend_adapter.build_context()  # type: ignore[attr-defined]
    project = generate_project(
        dest=context.source_root,
        definition=context.definition,
        adapter=context.model_adapter,
        task_config=context.task_config,
        engine_filename=context.engine_path.name,
        backend_context=backend_context,
        task_id=context.task_id,
        precision=context.precision,
    )
    context.source_dir = context.source_root

    dockerfile = generate_dockerfile(
        dest=context.docker_dir,
        definition=context.definition,
        backend_context=backend_context,
        task_id=context.task_id,
        precision=context.precision,
        engine_filename=context.engine_path.name,
        base_image=context.settings.docker_base_image,
        onnx_filename=context.onnx_path.name if context.onnx_path else "model.onnx",
        has_calibration=context.precision == "int8",
        calibration_cache_filename=context.settings.calibration_cache_filename,
        engine_metadata=context.engine_metadata,
    )

    logger.info("代码生成完成：%s 个文件；Dockerfile：%s", len(project.files), dockerfile.name)
    write_json(
        context.report_file("codegen.json"),
        {
            "task_id": context.task_id,
            "files": sorted(project.files),
            "dockerfile": str(dockerfile.relative_to(context.task_dir)).replace("\\", "/"),
            "backend_context": backend_context,
        },
    )
    context.reports.append("report/codegen.json")


# ---------------- BUILDING ----------------


def _compile_blockers(context: PipelineContext) -> list[str]:
    """列出阻止编译的环境问题（工具链 + TensorRT/CUDA 开发文件）。"""
    problems: list[str] = []

    info = toolchain.detect_toolchain()
    problems.extend(info.problems)

    dev = tensorrt_dev.locate_dev_files(context.settings)
    problems.extend(dev.problems)
    return problems


def _camera_setup(context: PipelineContext) -> tuple[dict[str, str], dict[str, Any]]:
    """实时推理（摄像头/视频/预览）开关：返回 CMake 定义与报告片段。

    `build.with_camera=true` 时尝试启用 OpenCV；**找不到 OpenCV 不算失败**——
    仍然编译（只是产物没有摄像头能力），并把原因写进报告与任务日志，避免
    因为一个可选能力把整个任务判为 FAILED。
    """
    if not bool(context.build_options.get("with_camera")):
        return {}, {"status": "OFF", "reason": "任务未开启实时推理（build.with_camera）"}

    files = opencv_dev.locate(context.settings)
    if not files.complete:
        logger.warning("要求实时推理，但未找到 OpenCV 开发文件：%s", files.problems)
        return {}, {
            "status": "BLOCKED",
            "reason": "任务要求实时推理（摄像头/视频/预览），但本机缺少 OpenCV 开发文件",
            "problems": files.problems,
            "note": "补齐 OpenCV 后重跑本任务即可得到带摄像头能力的产物；本次仍照常编译（无摄像头能力）",
        }

    defines = {
        "QFORGE_WITH_OPENCV": "ON",
        "QFORGE_OPENCV_ROOT": files.root.as_posix() if files.root else "",
    }
    logger.info("实时推理已启用：OpenCV %s（%s）", files.version, files.root)
    return defines, {
        "status": "ENABLED",
        "opencv": files.to_dict(),
        "note": "产物内置摄像头/视频/预览能力；交付 zip 会一并带上 OpenCV 运行库",
    }


def run_build(context: PipelineContext) -> None:
    """CMake Configure + Build（SPEC 11.3 步骤 2~3）。"""
    assert context.source_dir is not None

    camera_defines, camera_status = _camera_setup(context)

    mode = str(context.build_options.get("cpp_build") or "auto").lower()
    if mode not in CPP_BUILD_MODES:
        logger.warning("未知的 build.cpp_build=%s，按 auto 处理", mode)
        mode = "auto"

    if mode == "skip":
        logger.warning("任务配置要求跳过 C++ 编译（build.cpp_build=skip）")
        context.cpp_build_status = {
            "status": "SKIPPED",
            "reason": "任务配置 build.cpp_build=skip，未执行编译与运行验证",
            "mode": mode,
            "camera": camera_status,
        }
        write_json(context.report_file("cpp_build.json"), context.cpp_build_status)
        context.reports.append("report/cpp_build.json")
        return

    blockers = _compile_blockers(context)
    if blockers:
        logger.warning("编译环境检查未通过：%s", blockers)
        if mode == "required":
            raise BackendUnavailableError(
                "C++ 编译所需环境不完整，且任务要求必须编译（build.cpp_build=required）",
                detail={"problems": blockers},
            )
        logger.warning("C++ 编译验证标记为 BLOCKED，原因：%s", blockers)
        context.cpp_build_status = {
            "status": "BLOCKED",
            "mode": mode,
            "reason": "本机缺少编译生成的 C++ 工程所需文件，未执行编译与运行验证",
            "problems": blockers,
            "camera": camera_status,
            "note": "补齐后重跑本任务即可完成 SPEC 11.3 的编译与运行验证",
        }
        write_json(context.report_file("cpp_build.json"), context.cpp_build_status)
        context.reports.append("report/cpp_build.json")
        return

    toolchain_info = toolchain.require_toolchain()
    logger.info("构建工具链：%s", json.dumps(toolchain_info.to_dict(), ensure_ascii=False))

    build_dir = context.source_dir / "build"
    result = toolchain.build_cpp_project(
        project_dir=context.source_dir,
        build_dir=build_dir,
        log_path=context.log_file("cpp_build.log"),
        timeout_seconds=context.settings.build_timeout_seconds,
        extra_defines=camera_defines,
    )
    context.build_dir = build_dir

    executable = _find_executable(build_dir)
    if executable is None:
        raise RuntimeTestFailedError(
            "编译成功但未找到可执行文件",
            detail={"build_dir": str(build_dir), "expected": EXECUTABLE_NAME},
        )
    context.executable_path = executable
    context.cpp_build_status = {
        "status": "SUCCESS",
        "mode": mode,
        "executable": str(executable),
        "duration_seconds": round(result.duration_seconds, 3),
        "toolchain": toolchain_info.to_dict(),
        "camera": camera_status,
        "log": "logs/cpp_build.log",
    }
    logger.info("编译完成，可执行文件：%s", executable.name)

    write_json(context.report_file("cpp_build.json"), context.cpp_build_status)
    context.reports.append("report/cpp_build.json")


def _find_executable(build_dir: Path) -> Path | None:
    candidates = [
        build_dir / f"{EXECUTABLE_NAME}.exe",
        build_dir / EXECUTABLE_NAME,
        build_dir / "Release" / f"{EXECUTABLE_NAME}.exe",
        build_dir / "Release" / EXECUTABLE_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in build_dir.rglob(f"{EXECUTABLE_NAME}*"):
        if candidate.is_file() and candidate.suffix.lower() in ("", ".exe"):
            return candidate
    return None


# ---------------- TESTING ----------------


def _run_executable(
    context: PipelineContext, executable: Path, arguments: list[str], *, log_name: str
) -> subprocess.CompletedProcess[str]:
    log_path = context.log_file(log_name)
    command = [str(executable), *arguments]
    logger.info("运行验证：%s", " ".join(command))

    # 默认不把 TensorRT/CUDA 的 DLL 复制进每个任务的 build 目录（其中
    # nvinfer_builder_resource_*.dll 约 1.9 GB 且运行期无用）；改为运行时把
    # 这些 bin 目录加入 PATH，程序同样能找到依赖。
    # 实时推理产物还依赖 OpenCV 运行库：编译时探测到的 bin 目录也要加进 PATH。
    extra_dll_dirs: list[str] = []
    opencv_bin = ((context.cpp_build_status or {}).get("camera") or {}).get("opencv") or {}
    if opencv_bin.get("bin_dir"):
        extra_dll_dirs.append(str(opencv_bin["bin_dir"]))
    run_env = toolchain.build_environment(extra_dll_dirs=extra_dll_dirs)

    try:
        completed = subprocess.run(
            command,
            cwd=str(executable.parent),
            env=run_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=context.settings.run_verify_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeTestFailedError(
            f"运行验证超时（{context.settings.run_verify_timeout_seconds}s）",
            detail={"command": " ".join(command), "error": str(exc)},
        ) from exc

    log_path.write_text(
        "$ " + " ".join(command) + "\n\n--- stdout ---\n" + completed.stdout
        + "\n--- stderr ---\n" + completed.stderr,
        encoding="utf-8",
    )
    return completed


def _onnxruntime_baseline(context: PipelineContext, tensor: np.ndarray) -> np.ndarray:
    """用 onnxruntime(CPU, FP32) 跑同一输入，作为 FP32 基准输出（SPEC 9 精度定义）。"""
    import onnxruntime as ort

    assert context.onnx_path is not None
    session = ort.InferenceSession(str(context.onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    return np.asarray(outputs[0], dtype=np.float32)


def _accuracy_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    reference_flat = reference.astype(np.float64).ravel()
    actual_flat = actual.astype(np.float64).ravel()
    if reference_flat.size != actual_flat.size:
        raise RuntimeTestFailedError(
            "Engine 输出与 FP32 基准输出元素数量不一致，无法比较精度",
            detail={"reference": int(reference_flat.size), "actual": int(actual_flat.size)},
        )
    difference = actual_flat - reference_flat
    denominator = np.linalg.norm(reference_flat) * np.linalg.norm(actual_flat)
    cosine = float(np.dot(reference_flat, actual_flat) / denominator) if denominator else 1.0
    return {
        "baseline": "onnxruntime CPU FP32",
        "elements": int(reference_flat.size),
        "mae": float(np.mean(np.abs(difference))),
        "mse": float(np.mean(difference**2)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "cosine_similarity": cosine,
        "reference_abs_max": float(np.max(np.abs(reference_flat))),
        "actual_abs_max": float(np.max(np.abs(actual_flat))),
    }


def _verify_with_python(context: PipelineContext) -> dict[str, Any]:
    """用 TensorRT Python API 真实加载 Engine 并推理（不依赖 C++ 工具链）。"""
    assert context.backend_adapter is not None
    assert context.engine_path is not None
    assert context.definition is not None
    assert context.model_adapter is not None

    tensor_path = context.intermediate_dir / "test_input.f32"
    tensor = np.fromfile(tensor_path, dtype=np.float32).reshape(context.definition.input.shape)
    inputs = {context.definition.input.name: tensor}

    started = time.perf_counter()
    outputs = context.backend_adapter.run_inference(  # type: ignore[attr-defined]
        engine_path=context.engine_path, inputs=inputs
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    decode = context.model_adapter.decode(outputs, context.definition)
    baseline = _onnxruntime_baseline(context, tensor)
    metrics = _accuracy_metrics(baseline, outputs[0])

    logger.info(
        "Python 侧 Engine 验证通过：输出 %s，检测 %s 条，MAE=%.6f cosine=%.6f（%.1f ms）",
        list(outputs[0].shape),
        len(decode.detections),
        metrics["mae"],
        metrics["cosine_similarity"],
        latency_ms,
    )
    return {
        "status": "SUCCESS",
        "method": "TensorRT Python API（加载 Engine + 真实推理）",
        "engine": context.engine_path.name,
        "latency_ms": round(latency_ms, 3),
        "output_shapes": [list(item.shape) for item in outputs],
        "decode": decode.summary(),
        "accuracy": metrics,
    }


def _verify_generated_program(context: PipelineContext) -> dict[str, Any]:
    """运行生成的 C++ 程序并校验输出结构（SPEC 11.3 步骤 4~7）。"""
    assert context.executable_path is not None
    assert context.engine_path is not None
    assert context.definition is not None

    test_dir = context.source_dir / "test" if context.source_dir else context.intermediate_dir
    test_dir.mkdir(parents=True, exist_ok=True)

    ppm_source = context.intermediate_dir / "test_input.ppm"
    tensor_source = context.intermediate_dir / "test_input.f32"
    ppm_target = test_dir / "sample.ppm"
    tensor_target = test_dir / "sample.f32"
    if ppm_source.exists():
        place_input_file(ppm_source, ppm_target)
    if tensor_source.exists():
        place_input_file(tensor_source, tensor_target)

    result_json = test_dir / "result.json"
    dumped_raw = test_dir / "engine_output.f32"

    completed = _run_executable(
        context,
        context.executable_path,
        [
            "--engine",
            str(context.engine_path),
            "--image",
            str(ppm_target),
            "--output",
            str(result_json),
            "--dump-raw",
            str(dumped_raw),
            "--restore",
            "--iterations",
            "3",
        ],
        log_name="runtime_test.log",
    )

    if completed.returncode != 0:
        raise RuntimeTestFailedError(
            f"生成的程序运行失败（退出码 {completed.returncode}）",
            detail={"stderr": completed.stderr[-2000:], "log": "logs/runtime_test.log"},
        )

    if not result_json.exists():
        raise RuntimeTestFailedError("程序未输出结果文件", detail={"expected": str(result_json)})

    try:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeTestFailedError("程序输出不是合法 JSON", detail={"error": str(exc)}) from exc

    # SPEC 11.3 步骤 6：输出张量存在且 shape 正确
    expected_elements = int(np.prod(context.definition.output.shape)) if context.definition.output.shape else 0
    reported = int(payload.get("output_elements") or 0)
    if reported <= 0:
        raise RuntimeTestFailedError("程序报告的输出张量为空", detail={"payload": payload})
    if expected_elements and reported != expected_elements:
        # YOLOv8 导出常见 [1,84,8400]；若不一致说明 Engine 输出与模型定义不符
        raise RuntimeTestFailedError(
            "程序报告的输出元素数与模型定义不一致",
            detail={"expected": expected_elements, "reported": reported},
        )

    # SPEC 11.3 步骤 7：目标检测后处理结构合法
    class_count = context.definition.output.class_count
    detections = payload.get("detections") or []
    for detection in detections:
        box = detection.get("box") or []
        score = detection.get("score")
        class_id = detection.get("class_id")
        if len(box) != 4 or not all(np.isfinite(box)):
            raise RuntimeTestFailedError("检测框结构非法", detail={"detection": detection})
        if score is None or not (0.0 <= float(score) <= 1.0):
            raise RuntimeTestFailedError("检测置信度越界", detail={"detection": detection})
        if class_id is None or not (0 <= int(class_id) < class_count):
            raise RuntimeTestFailedError(
                "检测类别号越界", detail={"detection": detection, "class_count": class_count}
            )

    cpp_accuracy: dict[str, Any] | None = None
    if dumped_raw.exists() and tensor_target.exists():
        engine_output = np.fromfile(dumped_raw, dtype=np.float32).reshape(
            context.definition.output.shape
        )
        sample = np.fromfile(tensor_target, dtype=np.float32).reshape(
            context.definition.input.shape
        )
        baseline = _onnxruntime_baseline(context, sample)
        cpp_accuracy = _accuracy_metrics(baseline, engine_output)
        cpp_accuracy["baseline"] = "onnxruntime CPU FP32（对照生成的 C++ 程序输出）"

    return {
        "status": "SUCCESS",
        "method": "运行生成的 C++ 程序（SPEC 11.3 步骤 4~7）",
        "executable": str(context.executable_path),
        "returncode": completed.returncode,
        "result": payload,
        "accuracy": cpp_accuracy,
        "log": "logs/runtime_test.log",
    }


def run_testing(context: PipelineContext) -> None:
    """运行验证：Python 侧真实推理 + 精度指标；C++ 侧程序结构与输出校验。"""
    assert context.engine_path is not None
    assert context.definition is not None

    python_verification = _verify_with_python(context)

    if context.executable_path and context.executable_path.exists():
        cpp_verification = _verify_generated_program(context)
    else:
        cpp_verification = {
            "status": "BLOCKED",
            "reason": "未编译出可执行文件（详见 report/cpp_build.json 的 status 与 problems）",
            "note": "SPEC 11.3 要求的「生成的 C++ 程序真实运行」在补齐工具链后重跑本任务即可验证",
        }

    context.verification = {
        "python_engine_verification": python_verification,
        "cpp_program_verification": cpp_verification,
        "engine_metadata": context.engine_metadata,
    }

    write_json(
        context.report_file("runtime_verification.json"),
        {"task_id": context.task_id, "precision": context.precision, **context.verification},
    )
    write_json(
        context.report_file("accuracy.json"),
        {
            "task_id": context.task_id,
            "precision": context.precision,
            "engine_vs_fp32_baseline": python_verification["accuracy"],
            "cpp_program_vs_fp32_baseline": (cpp_verification or {}).get("accuracy"),
            "note": "FP32 基准 = onnxruntime CPU 推理；FP16/INT8 均以 FP32 为基准，不混称",
        },
    )
    context.reports.extend(["report/runtime_verification.json", "report/accuracy.json"])
    logger.info(
        "运行验证完成：Python 侧=%s，C++ 侧=%s",
        python_verification["status"],
        cpp_verification["status"],
    )

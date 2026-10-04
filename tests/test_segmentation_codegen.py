"""分割代码生成（2B）：按任务类型产出互斥源码集，并验证真实编译（纯 CPU/本地工具链）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.definition import ModelDefinition
from app.adapters.registry import get_model_adapter
from app.codegen import generate_project
from app.codegen.generator import render_plan
from app.services import onnx_inspector
from tools.synth_segmentation import build_segmentation_onnx

BACKEND_CONTEXT = {
    "name": "tensorrt",
    "version": "10.16.1.11",
    "gpu_arch": "sm_120",
    "cuda_version": "13.3",
    "tensorrt_include_dirs": ["D:/qforge-toolchain/tensorrt/TensorRT-10.16.1.11/include"],
    "tensorrt_lib_dirs": ["D:/qforge-toolchain/tensorrt/TensorRT-10.16.1.11/lib"],
    "tensorrt_libs": ["nvinfer_10", "nvinfer_plugin_10", "nvonnxparser_10"],
    "cuda_include_dirs": ["D:/qforge-toolchain/cuda/include"],
    "cuda_lib_dirs": ["D:/qforge-toolchain/cuda/lib"],
    "cuda_libs": ["cudart"],
    "runtime_dll_dirs": [],
    "copy_runtime_dlls": "OFF",
}


def _segmentation_definition(tmp_path: Path) -> ModelDefinition:
    onnx_path = build_segmentation_onnx(tmp_path / "seg.onnx", input_size=32, class_count=3)
    inspection = onnx_inspector.inspect(onnx_path).to_dict()
    adapter = get_model_adapter("unet")
    return ModelDefinition.parse(adapter.default_definition(inspection))


def _render_segmentation(tmp_path: Path) -> Path:
    definition = _segmentation_definition(tmp_path)
    dest = tmp_path / "source"
    generate_project(
        dest=dest,
        definition=definition,
        adapter=get_model_adapter("unet"),
        task_config={"task_id": "seg-1", "task_type": "segmentation", "precision": "fp16"},
        engine_filename="model.engine",
        backend_context=BACKEND_CONTEXT,
        task_id="seg-1",
        precision="fp16",
    )
    return dest


def test_render_plan_is_task_specific() -> None:
    detection = {item[1] for item in render_plan("detection")}
    segmentation = {item[1] for item in render_plan("segmentation")}

    assert "src/detector.cpp" in detection and "src/yolov8_decoder.cpp" in detection
    assert "src/detector.cpp" not in segmentation, "分割工程不应包含检测解码器"
    assert "src/segmenter.cpp" in segmentation
    assert "src/main.cpp" in detection and "src/main.cpp" in segmentation
    # 公共部分必须两边都有（引擎/预处理/实时输入/推送）
    for shared in ("src/engine.cpp", "src/preprocess.cpp", "src/push_client.cpp"):
        assert shared in detection and shared in segmentation


def test_render_plan_rejects_unknown_task() -> None:
    from app.errors import CodegenFailedError

    with pytest.raises(CodegenFailedError) as error:
        render_plan("classification")
    assert "classification" in str(error.value)


def test_segmentation_project_has_expected_files(tmp_path: Path) -> None:
    source = _render_segmentation(tmp_path)

    for expected in (
        "src/main.cpp",
        "src/segmenter.cpp",
        "include/qforge/segmenter.h",
        "include/qforge/engine.h",
        "CMakeLists.txt",
        "config/model.yaml",
    ):
        assert (source / expected).is_file(), f"缺少 {expected}"
    # 不能出现检测专用文件
    assert not (source / "src" / "detector.cpp").exists()
    assert not (source / "src" / "yolov8_decoder.cpp").exists()


def test_cmake_selects_segmentation_sources(tmp_path: Path) -> None:
    source = _render_segmentation(tmp_path)
    cmake = (source / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "src/segmenter.cpp" in cmake
    assert "src/detector.cpp" not in cmake
    assert "src/yolov8_decoder.cpp" not in cmake


def test_segmentation_main_uses_argmax_and_mask_output(tmp_path: Path) -> None:
    source = _render_segmentation(tmp_path)
    main = (source / "src" / "main.cpp").read_text(encoding="utf-8")
    segmenter = (source / "src" / "segmenter.cpp").read_text(encoding="utf-8")

    assert "decode_segmentation" in main
    assert "--mask" in main, "分割程序必须能把掩膜写成图片"
    assert "segmentation_json" in main
    # C++ 侧必须与 Python 侧同名同义：argmax + 直方图 + 忽略像素
    assert "best_class" in segmenter and "histogram" in segmenter
    assert "ignored_pixels" in segmenter
    # 忽略标签必须按模型定义渲染进 C++（默认 255；值为 -1 表示不设置）
    assert "255" in main, "ignore_index 没有渲染进生成的 main.cpp"


def test_cmake_segmentation_project_configures(tmp_path: Path) -> None:
    """真实跑一次 cmake 配置（不编译）：确认任务感知的 CMakeLists 语法与源文件都存在。"""
    import shutil
    import subprocess

    from app.services import toolchain

    source = _render_segmentation(tmp_path)
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("本机没有 cmake")

    result = subprocess.run(
        [cmake, "-S", str(source), "-B", str(source / "build"), "-DCMAKE_BUILD_TYPE=Release"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=toolchain.build_environment() if _toolchain_ready() else None,
    )

    assert result.returncode == 0, (result.stdout or "")[-2000:] + (result.stderr or "")[-1000:]
    assert "Configuring done" in (result.stdout or "")


def _toolchain_ready() -> bool:
    from app.services import toolchain

    try:
        return toolchain.detect_toolchain().available
    except Exception:  # noqa: BLE001 - 探测失败就退化为默认环境
        return False

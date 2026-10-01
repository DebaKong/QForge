"""代码生成测试（SPEC 11.1 / 11.2 / 12.2）。

验证分层模板能渲染出完整工程，且关键配置被正确固化进生成代码与 config/model.yaml。
"""

from __future__ import annotations

from pathlib import Path

from app.adapters.definition import ModelDefinition
from app.adapters.registry import get_model_adapter
from app.codegen import generate_project
from app.codegen.docker import generate_dockerfile

DEFINITION_PAYLOAD = {
    "task": "detection",
    "architecture": "yolov8",
    "input": {"name": "images", "shape": [1, 3, 640, 640], "layout": "NCHW", "dtype": "FP32"},
    "preprocessing": {
        "resize": "letterbox",
        "color": "RGB",
        "scale": 255.0,
        "mean": [0, 0, 0],
        "std": [1, 1, 1],
    },
    "output": {
        "name": "output0",
        "format": "[1,84,8400]",
        "box_format": "xywh",
        "has_objectness": False,
        "class_count": 80,
    },
    "postprocessing": {
        "decoder": "yolov8",
        "nms": {"type": "classwise", "confidence_threshold": 0.25, "iou_threshold": 0.45},
    },
}

BACKEND_CONTEXT = {
    "name": "tensorrt",
    "version": "10.16.1.11",
    "gpu_arch": "sm_120",
    "cuda_version": "13.0",
    "tensorrt_include_dirs": ["C:/qforge-toolchain/tensorrt/include"],
    "tensorrt_lib_dirs": ["C:/qforge-toolchain/tensorrt/lib"],
    "tensorrt_libs": ["nvinfer", "nvinfer_plugin", "nvonnxparser"],
    "cuda_include_dirs": ["C:/qforge-toolchain/cuda/include"],
    "cuda_lib_dirs": ["C:/qforge-toolchain/cuda/lib/x64"],
    "cuda_libs": ["cudart"],
    "runtime_dll_dirs": ["C:/qforge-toolchain/tensorrt/bin"],
}


def _render(tmp_path: Path) -> Path:
    definition = ModelDefinition.parse(DEFINITION_PAYLOAD)
    project = generate_project(
        dest=tmp_path / "source",
        definition=definition,
        adapter=get_model_adapter("yolov8"),
        task_config={"task_id": "task-1", "precision": "fp16"},
        engine_filename="model.engine",
        backend_context=BACKEND_CONTEXT,
        task_id="task-1",
        precision="fp16",
    )
    assert project.files
    assert "CMakeLists.txt" in project.files
    return tmp_path / "source"


def test_generates_expected_file_layers(tmp_path: Path) -> None:
    source = _render(tmp_path)
    expected = [
        "CMakeLists.txt",
        "config/model.yaml",
        "README.md",
        "test/README.md",
        "include/qforge/generated_config.h",
        "include/qforge/logger.h",
        "include/qforge/cuda_utils.h",
        "include/qforge/engine.h",
        "include/qforge/image_io.h",
        "include/qforge/preprocess.h",
        "include/qforge/detector.h",
        "include/qforge/yolov8_decoder.h",
        "src/main.cpp",
        "src/engine.cpp",
        "src/image_io.cpp",
        "src/preprocess.cpp",
        "src/detector.cpp",
        "src/yolov8_decoder.cpp",
    ]
    for relative in expected:
        assert (source / relative).is_file(), relative


def test_generated_config_header_bakes_model_parameters(tmp_path: Path) -> None:
    source = _render(tmp_path)
    header = (source / "include/qforge/generated_config.h").read_text(encoding="utf-8")
    assert "kInputHeight = 640" in header
    assert "kInputWidth = 640" in header
    assert "kInputChannels = 3" in header
    assert "kClassCount = 80" in header
    assert "kHasObjectness = false" in header
    assert 'kInputName = "images"' in header
    assert 'kOutputName = "output0"' in header
    assert "kConfidenceThreshold = 0.25" in header
    assert "kIouThreshold = 0.45" in header
    assert "kScale = 255.0f" in header
    # 浮点字面量带 f 后缀，避免 constexpr float 由 double 初始化触发 C4305 截断告警
    assert "kMean = { 0.0f, 0.0f, 0.0f }" in header
    assert "kStd = { 1.0f, 1.0f, 1.0f }" in header
    assert 'kPrecision = "FP16"' in header


def test_cmake_lists_uses_injected_dependency_paths(tmp_path: Path) -> None:
    source = _render(tmp_path)
    cmake = (source / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "C:/qforge-toolchain/tensorrt/include" in cmake
    assert "C:/qforge-toolchain/cuda/lib/x64" in cmake
    assert "nvinfer" in cmake
    assert "cudart" in cmake
    assert "qforge_detector" in cmake


def test_config_yaml_matches_definition(tmp_path: Path) -> None:
    source = _render(tmp_path)
    config = (source / "config/model.yaml").read_text(encoding="utf-8")
    assert "task_id: task-1" in config
    assert "precision: fp16" in config
    assert "architecture: yolov8" in config
    assert "shape: [1, 3, 640, 640]" in config
    assert "confidence_threshold: 0.25" in config
    assert "backend_version: 10.16.1.11" in config


def test_generated_cpp_references_model_adapter(tmp_path: Path) -> None:
    source = _render(tmp_path)
    decoder = (source / "src/yolov8_decoder.cpp").read_text(encoding="utf-8")
    assert "decode_yolov8" in decoder
    detector = (source / "src/detector.cpp").read_text(encoding="utf-8")
    assert "yolov8_decoder.h" in detector
    main_cpp = (source / "src/main.cpp").read_text(encoding="utf-8")
    assert "--engine" in main_cpp
    assert "dump-raw" in main_cpp


def test_dockerfile_generation(tmp_path: Path) -> None:
    definition = ModelDefinition.parse(DEFINITION_PAYLOAD)
    path = generate_dockerfile(
        dest=tmp_path / "docker",
        definition=definition,
        backend_context=BACKEND_CONTEXT,
        task_id="task-1",
        precision="int8",
        engine_filename="model.engine",
    )
    content = path.read_text(encoding="utf-8")
    assert "FROM ${BASE_IMAGE}" in content
    assert "REPLACE_WITH_CONFIRMED_TAG" in content  # 基础镜像 tag 必须人工确认
    assert "QFORGE_PRECISION=INT8" in content
    assert "USER qforge" in content

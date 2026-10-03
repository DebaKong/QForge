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


def test_start_scripts_are_generated_with_correct_line_endings(tmp_path: Path) -> None:
    """交付物要「解压即用」：工程根目录必须有 start.bat / start.sh，且行尾各自正确。

    实测教训：`.bat` 用 LF 会被 cmd.exe 拆坏（'etlocal' 不是内部或外部命令）；
    `.sh` 用 CRLF 会 bad interpreter；`.bat` 含中文会乱码。
    """
    source = _render(tmp_path)

    bat = (source / "start.bat").read_bytes()
    assert bat.startswith(b"@echo off")
    assert b"\r\n" in bat
    assert bat.replace(b"\r\n", b"").count(b"\n") == 0, "start.bat 必须是纯 CRLF"
    assert all(byte < 128 for byte in bat), "start.bat 必须是 ASCII（cmd 对 UTF-8 支持差）"

    # 实时模式启动脚本同理（.bat 必须 CRLF+ASCII）
    camera_bat = (source / "start_camera.bat").read_bytes()
    assert camera_bat.startswith(b"@echo off")
    assert camera_bat.replace(b"\r\n", b"").count(b"\n") == 0, "start_camera.bat 必须是纯 CRLF"
    assert all(byte < 128 for byte in camera_bat), "start_camera.bat 必须是 ASCII"
    assert b"--push" in camera_bat and b"--camera" in camera_bat

    shell = (source / "start.sh").read_bytes()
    assert shell.startswith(b"#!")
    assert b"\r\n" not in shell, "start.sh 必须是 LF"
    camera_shell = (source / "start_camera.sh").read_bytes()
    assert camera_shell.startswith(b"#!")
    assert b"\r\n" not in camera_shell, "start_camera.sh 必须是 LF"

    readme = (source / "README.md").read_text(encoding="utf-8")
    for expected in (
        "start.bat",
        "start.sh",
        "一键启动",
        "bin/",
        "model/model.engine",
        "实时模式",  # 摄像头 + 推送章节必须存在
        "--push",
        "stdin-frames",
    ):
        assert expected in readme, f"交付说明里应提到 {expected}"


def test_dockerfile_generation(tmp_path: Path) -> None:
    definition = ModelDefinition.parse(DEFINITION_PAYLOAD)
    path = generate_dockerfile(
        dest=tmp_path / "docker",
        definition=definition,
        backend_context=BACKEND_CONTEXT,
        task_id="task-1",
        precision="int8",
        engine_filename="model.engine",
        base_image="nvcr.io/nvidia/tensorrt:26.03-py3",
        onnx_filename="synth.onnx",
        has_calibration=True,
    )
    content = path.read_text(encoding="utf-8")
    # 多阶段构建：Windows 侧编出的是 .exe，必须在容器内重新编译才可运行
    assert "FROM ${BASE_IMAGE} AS builder" in content
    assert "FROM ${BASE_IMAGE} AS runtime" in content
    assert "nvcr.io/nvidia/tensorrt:26.03-py3" in content
    assert "cmake --build /build/source/build" in content
    # CUDA 头文件随构建上下文提供（TRT 官方镜像不含 cuda_runtime_api.h）
    assert "COPY cuda-include/" in content
    assert "-DQFORGE_CUDA_INCLUDE_DIRS=/opt/qforge-cuda-include" in content
    assert "USER qforge" in content
    assert "/opt/qforge/test/sample.ppm" in content
    assert 'ENTRYPOINT ["/opt/qforge/entrypoint.sh"]' in content

    # Engine 重建放在容器启动时（TRT 计划文件平台相关，且 docker build 阶段无 GPU）
    entrypoint = (tmp_path / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "trtexec" in entrypoint
    assert "--int8" in entrypoint
    assert "CALIB=/opt/qforge/calibration/" in entrypoint
    assert "REBUILD_ENGINE" in entrypoint
    assert "exec /opt/qforge/bin/qforge_detector" in entrypoint

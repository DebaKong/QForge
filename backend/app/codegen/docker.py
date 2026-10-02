"""Dockerfile 生成（SPEC 12.2 容器产物）。

要点：
- SPEC 2.1 对容器产物的要求是「Dockerfile、依赖说明；**可选**生成镜像」；
  本模块负责生成可审查、可构建的 Dockerfile，镜像是否真的构建由任务配置
  `build.docker_build` 决定（默认关闭）。
- 生成的镜像是**多阶段构建**：容器内重新编译同一份 C++ 源码（Windows 侧编出的是 .exe，
  在 Linux 容器里无法运行），因此同一份生成代码在两种平台上都能构建运行。
- 基础镜像 tag 属版本敏感项（SPEC 4.1）：默认值来自 `QFORGE_DOCKER_BASE_IMAGE`，
  该 tag 的 TensorRT/CUDA 版本已在 docs/versions.md 中记录并核对。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from app.adapters.definition import ModelDefinition
from app.errors import CodegenFailedError

logger = logging.getLogger(__name__)

TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"


def generate_dockerfile(
    *,
    dest: Path,
    definition: ModelDefinition,
    backend_context: dict[str, Any],
    task_id: str,
    precision: str,
    engine_filename: str,
    base_image: str,
    onnx_filename: str,
    has_calibration: bool = False,
    calibration_cache_filename: str = "calibration.cache",
    engine_metadata: dict[str, Any] | None = None,
) -> Path:
    """渲染 Dockerfile 到 dest/Dockerfile。"""
    dest.mkdir(parents=True, exist_ok=True)
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATES_ROOT)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
    )
    context = {
        "task_id": task_id,
        "precision": precision.lower(),
        "precision_label": precision.upper(),
        "engine_filename": engine_filename,
        "onnx_filename": onnx_filename,
        "base_image": base_image,
        "has_calibration": has_calibration,
        "calibration_cache_filename": calibration_cache_filename,
        # 默认**在容器内重建 Engine**：TensorRT 的计划文件是「平台相关」的——
        # 平台在 Windows 上构建的 .engine 在 Linux 容器里加载会报
        # `Platform specific tag mismatch`（实测），因此容器内用 trtexec 重建。
        "rebuild_engine_default": 1,
        "engine_metadata": engine_metadata or {},
        "backend": backend_context,
        "architecture": definition.architecture,
        "input_shape": list(definition.input.shape),
    }
    try:
        rendered = environment.get_template("docker/Dockerfile.j2").render(**context)
    except TemplateError as exc:
        raise CodegenFailedError(
            "Dockerfile 生成失败", detail={"error": f"{type(exc).__name__}: {exc}"}
        ) from exc

    target = dest / "Dockerfile"
    target.write_text(rendered, encoding="utf-8", newline="\n")

    # 启动脚本一并生成：Engine 是平台相关的，容器启动时才在目标机 GPU 上重建
    entrypoint = environment.get_template("docker/entrypoint.sh.j2").render(**context)
    entrypoint_path = dest / "entrypoint.sh"
    entrypoint_path.write_text(entrypoint, encoding="utf-8", newline="\n")

    logger.info("Dockerfile 与 entrypoint.sh 已生成（基础镜像 %s）", base_image)
    return target

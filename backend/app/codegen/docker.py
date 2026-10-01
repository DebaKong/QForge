"""Dockerfile 生成（SPEC 12.2 容器产物）。

SPEC 2.1 对容器产物的要求是「Dockerfile、依赖说明；**可选**生成镜像」，
因此本模块只负责生成可审查的 Dockerfile；
真实镜像构建由任务配置 `build.docker_build` 控制，且要求 Docker daemon 可用。

基础镜像 tag 属版本敏感项（SPEC 4.1），模板中以 ARG 暴露并要求构建前确认，
不在代码里写死一个未经拉取验证的 tag。
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
        "precision": precision.upper(),
        "engine_filename": engine_filename,
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
    logger.info("Dockerfile 已生成：%s", target)
    return target

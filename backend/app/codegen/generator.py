"""代码生成器（SPEC 11.1 / 11.2）。

分层渲染顺序（对应 SPEC 11.1 的 codegen 目录分层）：

    common/      公共推理底座：日志、CUDA 检查、Engine 封装、图像读取、预处理
    task/<task>/ 任务模板：检测任务的解码 + NMS 与 main
    model/<arch>/模型适配器模板：YOLOv8 输出布局相关代码
    config/      配置与说明：model.yaml、README.md

配置**编译期固化**进生成代码（避免在 C++ 侧引入 YAML/JSON 解析依赖），
同时在 `config/model.yaml` 输出同一份可读配置作为记录；二者由同一份上下文渲染，不会漂移。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from app.adapters.base import ModelAdapter
from app.adapters.definition import ModelDefinition
from app.errors import CodegenFailedError

logger = logging.getLogger(__name__)

TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"

# (模板相对路径, 输出相对路径)
RENDER_PLAN: tuple[tuple[str, str], ...] = (
    ("common/include/qforge/generated_config.h.j2", "include/qforge/generated_config.h"),
    ("common/include/qforge/logger.h.j2", "include/qforge/logger.h"),
    ("common/include/qforge/cuda_utils.h.j2", "include/qforge/cuda_utils.h"),
    ("common/include/qforge/engine.h.j2", "include/qforge/engine.h"),
    ("common/src/engine.cpp.j2", "src/engine.cpp"),
    ("common/include/qforge/image_io.h.j2", "include/qforge/image_io.h"),
    ("common/src/image_io.cpp.j2", "src/image_io.cpp"),
    ("common/include/qforge/preprocess.h.j2", "include/qforge/preprocess.h"),
    ("common/src/preprocess.cpp.j2", "src/preprocess.cpp"),
    # 实时输入（摄像头/视频/管道，OpenCV 可选）与结果推送（原生 socket）
    ("common/include/qforge/frame_source.h.j2", "include/qforge/frame_source.h"),
    ("common/src/frame_source.cpp.j2", "src/frame_source.cpp"),
    ("common/include/qforge/push_client.h.j2", "include/qforge/push_client.h"),
    ("common/src/push_client.cpp.j2", "src/push_client.cpp"),
    ("common/CMakeLists.txt.j2", "CMakeLists.txt"),
    ("config/model.yaml.j2", "config/model.yaml"),
    ("config/README.md.j2", "README.md"),
    # 交付产物根目录的"一键启动"脚本（解压后直接跑）
    ("common/start.bat.j2", "start.bat"),
    ("common/start.sh.j2", "start.sh"),
    # 实时模式（摄像头/视频 + 推送检测结果）
    ("common/start_camera.bat.j2", "start_camera.bat"),
    ("common/start_camera.sh.j2", "start_camera.sh"),
    ("config/test_README.md.j2", "test/README.md"),
)


# 任务相关模板：按任务类型二选一（公共模板共用）。
# 为什么要分开：检测与分割的后处理语义完全不同（NMS 出框 vs 逐像素 argmax 出掩膜），
# 把两套源码都生成出来只会让产物里多一堆用不到的代码，也容易让人误读主流程。
_TASK_RENDER_PLAN: dict[str, tuple[tuple[str, str], ...]] = {
    "detection": (
        ("task/detection/src/main.cpp.j2", "src/main.cpp"),
        ("task/detection/include/qforge/detector.h.j2", "include/qforge/detector.h"),
        ("task/detection/src/detector.cpp.j2", "src/detector.cpp"),
        ("model/yolov8/include/qforge/yolov8_decoder.h.j2", "include/qforge/yolov8_decoder.h"),
        ("model/yolov8/src/yolov8_decoder.cpp.j2", "src/yolov8_decoder.cpp"),
    ),
    "segmentation": (
        ("task/segmentation/src/main.cpp.j2", "src/main.cpp"),
        ("task/segmentation/include/qforge/segmenter.h.j2", "include/qforge/segmenter.h"),
        ("task/segmentation/src/segmenter.cpp.j2", "src/segmenter.cpp"),
    ),
}


def render_plan(task: str) -> tuple[tuple[str, str], ...]:
    """按任务类型组装渲染计划：公共模板 + 该任务专属模板。"""
    key = (task or "detection").lower()
    if key not in _TASK_RENDER_PLAN:
        raise CodegenFailedError(
            f"没有该任务类型的代码模板：{task}",
            detail={"task": task, "supported": sorted(_TASK_RENDER_PLAN)},
        )
    return (*RENDER_PLAN, *_TASK_RENDER_PLAN[key])


@dataclass
class GeneratedProject:
    """生成结果。"""

    root: Path
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"root": str(self.root), "files": sorted(self.files), "warnings": self.warnings}


def build_environment() -> Environment:
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATES_ROOT)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
    )
    # 模板中用 `{{ value|cpp }}` 生成 C++ 字面量（字符串加引号、bool 转 true/false、数组展开）
    environment.filters["cpp"] = _format_value
    return environment


def _format_value(value: Any) -> str:
    """把 Python 值渲染成 C++ 字面量（数组只输出元素，由模板补花括号）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # 带 f 后缀：否则 constexpr float 由 double 初始化会触发 C4305 截断告警
        return f"{value!r}f"
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    raise CodegenFailedError(f"无法渲染为 C++ 字面量：{value!r}")


def generate_project(
    *,
    dest: Path,
    definition: ModelDefinition,
    adapter: ModelAdapter,
    task_config: dict[str, Any],
    engine_filename: str,
    backend_context: dict[str, Any],
    task_id: str,
    precision: str,
) -> GeneratedProject:
    """渲染完整 C++ 工程到 dest。"""
    dest.mkdir(parents=True, exist_ok=True)
    environment = build_environment()

    model_params = adapter.template_context(definition)

    context: dict[str, Any] = {
        "task_id": task_id,
        # 精度：小写与 SPEC 5.1 的 TaskConfig 一致；标签用于代码与文档展示
        "precision": precision.lower(),
        "precision_label": precision.upper(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_filename": engine_filename,
        "model": model_params,
        "config": task_config,
        "backend": backend_context,
        "input_shape": list(definition.input.shape),
        "output_shape": list(definition.output.shape),
    }
    context.update(model_params)

    project = GeneratedProject(root=dest)
    try:
        for template_name, output_name in render_plan(definition.task):
            template = environment.get_template(template_name)
            rendered = template.render(**context)
            # 先规范化成 LF：模板文件本身可能被 git 检出成 CRLF（Windows），
            # 不规范化会让生成的 shell 脚本带 CRLF（bad interpreter）。
            rendered = rendered.replace("\r\n", "\n")
            target = dest / output_name
            target.parent.mkdir(parents=True, exist_ok=True)
            # .bat/.cmd 必须 CRLF（cmd.exe 靠它解析标签/goto），其余一律 LF
            # （shell 脚本若带 CRLF 会 bad interpreter，见 AGENTS.md 工程环境注意事项）
            newline = "\r\n" if output_name.lower().endswith((".bat", ".cmd")) else "\n"
            target.write_text(rendered, encoding="utf-8", newline=newline)
            project.files.append(output_name)
    except TemplateError as exc:
        raise CodegenFailedError(
            "代码生成失败（模板渲染错误）", detail={"error": f"{type(exc).__name__}: {exc}"}
        ) from exc

    logger.info("代码生成完成", extra={"task_id": task_id, "files": len(project.files)})
    return project

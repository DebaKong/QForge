"""交付阶段：产物归档与报告汇总（SPEC 11.3 步骤 8 / 12.2 / 13.1）。"""

from __future__ import annotations

import hashlib
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any

from app.db.base import session_scope
from app.models.artifact import Artifact
from app.pipeline.context import PipelineContext
from app.pipeline.stages.prepare import write_json
from app.services import task_service

logger = logging.getLogger("qforge.pipeline")

# 归档时排除的目录（构建中间产物体积大且可复现）
_EXCLUDED_DIRS = {"build", "__pycache__", ".cmake", "CMakeFiles"}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> int:
    """复制目录（跳过构建中间产物），返回复制文件数。"""
    count = 0
    for path in source.rglob("*"):
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def _assemble_artifact_tree(context: PipelineContext) -> Path:
    """按 SPEC 12.2 的 artifact/ 结构组装交付目录。"""
    staging = context.intermediate_dir / "artifact"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    copied: dict[str, int] = {}

    if context.engine_path and context.engine_path.exists():
        target = staging / "model" / context.engine_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(context.engine_path, target)
        metadata = context.engine_dir / "engine_metadata.json"
        if metadata.exists():
            shutil.copy2(metadata, staging / "model" / metadata.name)

    if context.source_dir and context.source_dir.exists():
        copied["source"] = _copy_tree(context.source_dir, staging / "source")
        readme = context.source_dir / "README.md"
        if readme.exists():
            shutil.copy2(readme, staging / "README.md")
        config_file = context.source_dir / "config" / "model.yaml"
        if config_file.exists():
            target = staging / "config" / "model.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_file, target)

    if context.docker_dir.exists():
        target = staging / "docker"
        target.mkdir(parents=True, exist_ok=True)
        for item in context.docker_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)

    if context.source_dir and (context.source_dir / "test").exists():
        copied["test"] = _copy_tree(context.source_dir / "test", staging / "test")

    if context.report_dir.exists():
        copied["report"] = _copy_tree(context.report_dir, staging / "report")

    logger.info("交付目录已组装：%s", copied)
    return staging


def _write_archive(staging: Path, archive_path: Path) -> int:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(staging).as_posix())
                file_count += 1
    return file_count


def run_packaging(context: PipelineContext) -> None:
    """归档产物、写入 Artifact 行、汇总报告（SPEC 12.2 / 13.1）。"""
    staging = _assemble_artifact_tree(context)
    archive_path = context.report_dir / "artifact.zip"
    file_count = _write_archive(staging, archive_path)
    logger.info("产物归档完成：%s 个文件 → %s", file_count, archive_path.name)

    entries: list[dict[str, Any]] = []

    def add(kind: str, path: Path, description: str) -> None:
        if not path.exists():
            return
        relative = path.relative_to(context.storage_root).as_posix()
        entries.append(
            {
                "kind": kind,
                "relative_path": relative,
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path),
                "description": description,
            }
        )

    if context.engine_path:
        add("engine", context.engine_path, "TensorRT Engine")
    add("engine", context.engine_dir / "engine_metadata.json", "Engine 构建环境元数据")
    if context.source_dir:
        add("source", context.source_dir / "CMakeLists.txt", "CMake 构建脚本")
        add("config", context.source_dir / "config" / "model.yaml", "模型与预处理配置")
    add("docker", context.docker_dir / "Dockerfile", "运行镜像 Dockerfile")
    add("report", archive_path, "完整产物归档（SPEC 12.2 artifact 结构）")
    for name in (
        "model_info.json",
        "compatibility.json",
        "preprocessing.json",
        "quantization.json",
        "engine_build.json",
        "codegen.json",
        "cpp_build.json",
        "runtime_verification.json",
        "accuracy.json",
    ):
        add("report", context.report_file(name), f"报告 {name}")

    with session_scope() as session:
        task = task_service.get_task(session, context.task_id)
        archive_row: Artifact | None = None
        for entry in entries:
            row = Artifact(
                task_id=context.task_id,
                kind=entry["kind"],
                relative_path=entry["relative_path"],
                size_bytes=entry["size_bytes"],
                sha256=entry["sha256"],
            )
            session.add(row)
            session.flush()
            if entry["relative_path"].endswith("artifact.zip"):
                archive_row = row
        if archive_row is not None:
            # SPEC 13.1：task.artifact_id 指向最终产物
            task.artifact_id = archive_row.id

    summary = {
        "task_id": context.task_id,
        "precision": context.precision,
        "engine_metadata": context.engine_metadata,
        "verification": context.verification,
        "artifact_entries": len(entries),
        "archive": archive_path.relative_to(context.storage_root).as_posix(),
        "reports": context.reports,
    }
    write_json(context.report_file("summary.json"), summary)
    logger.info("任务交付汇总已写入 report/summary.json")

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


def _copy_tree(
    source: Path, destination: Path, *, exclude_names: frozenset[str] = frozenset()
) -> int:
    """复制目录（跳过构建中间产物），返回复制文件数。"""
    count = 0
    for path in source.rglob("*"):
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.name in exclude_names:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


# 交付根目录自己用的一键启动文件：只放在根目录，不在 source/ 里重复一份
_DELIVERY_ROOT_FILES = (
    "README.md",
    "start.bat",
    "start.sh",
    "start_camera.bat",
    "start_camera.sh",
)


def _copy_binary(context: PipelineContext, staging: Path) -> Path | None:
    """把编译产物放进 bin/，让用户解压后可以直接启动推理（无需自己编译）。

    Windows 产物是 .exe，Linux/容器里编译出来的是无扩展名可执行文件；两者都接受。
    """
    if context.source_dir is None:
        return None
    build_dir = context.source_dir / "build"
    if not build_dir.is_dir():
        return None

    for name in ("qforge_detector.exe", "qforge_detector"):
        candidate = build_dir / name
        if candidate.is_file():
            target = staging / "bin" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            logger.info("已将编译产物放入交付目录：bin/%s", name)
            return target
    logger.info("未找到编译产物（本次可能未编译或编译失败），交付目录不含 bin/")
    return None


# Windows 系统提供的 DLL：不需要（也不应该）打进交付包
_SYSTEM_DLL_PREFIXES = ("api-ms-win-", "ext-ms-")
_SYSTEM_DLL_NAMES = {
    "kernel32.dll",
    "user32.dll",
    "gdi32.dll",
    "advapi32.dll",
    "ole32.dll",
    "oleaut32.dll",
    "shell32.dll",
    "ws2_32.dll",
    "comdlg32.dll",
    "comctl32.dll",
    "setupapi.dll",
    "winmm.dll",
    "ntdll.dll",
    "crypt32.dll",
    "bcrypt.dll",
    "secur32.dll",
    "dbghelp.dll",
    "imm32.dll",
    "version.dll",
    "userenv.dll",
    "dwmapi.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "concrt140.dll",
    "opengl32.dll",
    "uxtheme.dll",
}


def _dumpbin_dependents(executable: Path) -> list[str]:
    """用 dumpbin 读一个 DLL/EXE 直接依赖的模块名（失败返回空表，不阻断交付）。"""
    from app.services import toolchain

    try:
        result = toolchain.run_in_vs_env(
            [f'dumpbin /nologo /dependents "{executable}"'],
            cwd=executable.parent,
            log_path=executable.parent / "dumpbin_dependents.log",
            timeout_seconds=120,
        )
    except Exception:  # pragma: no cover - 工具链异常不应影响交付
        logger.debug("dumpbin 调用失败：%s", executable, exc_info=True)
        return []

    names: list[str] = []
    for line in result.log_text.splitlines():
        candidate = line.strip()
        if not candidate.lower().endswith(".dll"):
            continue
        name = candidate.split()[-1].lower()
        if name.startswith(_SYSTEM_DLL_PREFIXES) or name in _SYSTEM_DLL_NAMES:
            continue
        names.append(name)
    return names


def _resolve_dll_closure(
    roots: list[Path], search_dirs: list[Path], *, limit: int = 200
) -> list[Path]:
    """递归解析 DLL 依赖闭包（只在本机 OpenCV 目录里找得到的才算）。

    实测教训：只把 opencv_*.dll 放进交付包，解压后运行会以 0xC0000135
    （STATUS_DLL_NOT_FOUND）失败——它们还依赖 zlib/libjpeg/ffmpeg 等库。
    """
    resolved: dict[str, Path] = {}
    pending = [item.name.lower() for item in roots]
    while pending and len(resolved) < limit:
        name = pending.pop()
        if name in resolved:
            continue
        for directory in search_dirs:
            candidate = directory / name
            if candidate.is_file():
                resolved[name] = candidate
                pending.extend(_dumpbin_dependents(candidate))
                break
    return list(resolved.values())


def _copy_opencv_runtime(staging: Path, bin_dir: Path | None) -> list[str]:
    """启用实时推理的产物：把 OpenCV 运行库**及其依赖闭包**放进 bin/。

    这样交付 zip 在目标机上仍然「解压即用」（否则解压后运行会因缺少依赖 DLL 直接失败）。
    """
    if bin_dir is None or not Path(bin_dir).is_dir():
        return []
    # 只复制程序真正用到的模块（conda 的 bin 里还有几十个 opencv_* 模块与 Qt6，全带上太大）
    wanted = ("opencv_world", "opencv_core", "opencv_imgproc", "opencv_videoio", "opencv_highgui")
    modules: list[Path] = []
    for item in sorted(Path(bin_dir).iterdir()):
        if not item.is_file():
            continue
        name = item.name.lower()
        if not (name.endswith(".dll") or ".so" in name):
            continue
        if any(module in name for module in wanted):
            modules.append(item)

    # 依赖闭包（zlib/libjpeg/ffmpeg 等）与 OpenCV 在同一目录下
    closure = _resolve_dll_closure(modules, [Path(bin_dir)]) if modules else []

    copied: list[str] = []
    for item in [*modules, *closure]:
        target = staging / "bin" / item.name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied.append(item.name)
    if copied:
        logger.info("已把 OpenCV 运行库放入交付目录：%d 个 DLL", len(copied))
    return copied


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
        # README 与一键启动脚本属于「交付根目录」，不在 source/ 里再放一份（避免用户点错）
        copied["source"] = _copy_tree(
            context.source_dir,
            staging / "source",
            exclude_names=frozenset(_DELIVERY_ROOT_FILES),
        )
        for name in _DELIVERY_ROOT_FILES:
            item = context.source_dir / name
            if item.exists():
                shutil.copy2(item, staging / name)
        config_file = context.source_dir / "config" / "model.yaml"
        if config_file.exists():
            target = staging / "config" / "model.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_file, target)

    # 编译产物放进 bin/：用户解压后可以直接启动推理，不必自己编译
    _copy_binary(context, staging)

    # 启用实时推理的产物：把 OpenCV 运行库一起带上（保持「解压即用」）
    camera = (context.cpp_build_status or {}).get("camera") or {}
    if camera.get("status") == "ENABLED":
        _copy_opencv_runtime(staging, (camera.get("opencv") or {}).get("bin_dir"))

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


def _cuda_include_dirs(context: PipelineContext) -> list[Path]:
    """编译容器内工程所需的 CUDA 头文件目录。

    TensorRT 官方镜像只提供 TRT 头文件与 CUDA **运行库**，不含 cuda_runtime_api.h
    与 crt/*；平台侧已有这些头文件（Windows 编译同样需要），因此随构建上下文一并提供。
    """
    from app.adapters.backends import tensorrt_dev

    files = tensorrt_dev.locate_dev_files(context.settings)
    return [Path(item) for item in files.cuda_include_dirs]


def _run_docker_step(context: PipelineContext) -> None:
    """可选：构建容器镜像（SPEC 12.1：由受控 Worker 执行，日志完整保留）。

    三态汇报，绝不伪造：
    - SKIPPED：任务配置未开启 `build.docker_build`（SPEC 2.1：生成镜像本身是可选项）；
    - BLOCKED：Docker daemon 不可用，如实记录原因；
    - SUCCESS / FAILED：真实执行 `docker build`（失败即任务 FAILED）。
    """
    from app.services import docker_build

    enabled = bool(context.build_options.get("docker_build", context.settings.docker_enabled))
    base_image = context.settings.docker_base_image
    report_path = context.report_file("docker_build.json")

    if not enabled:
        write_json(
            report_path,
            {
                "status": "SKIPPED",
                "reason": "任务配置未开启 build.docker_build（SPEC 2.1 将「生成镜像」列为可选项）",
                "dockerfile": "docker/Dockerfile",
                "base_image": base_image,
            },
        )
        context.reports.append("report/docker_build.json")
        logger.info("未开启镜像构建，跳过（Dockerfile 已生成）")
        return

    available, info = docker_build.docker_available()
    if not available:
        write_json(
            report_path,
            {
                "status": "BLOCKED",
                "reason": info.get("reason"),
                "docker": info,
                "dockerfile": "docker/Dockerfile",
                "base_image": base_image,
            },
        )
        context.reports.append("report/docker_build.json")
        logger.warning("Docker 不可用，镜像构建标记为 BLOCKED：%s", info.get("reason"))
        return

    assert context.engine_path is not None
    context_dir = context.intermediate_dir / "docker-context"
    docker_build.stage_context(
        context_dir=context_dir,
        dockerfile=context.docker_dir / "Dockerfile",
        source_dir=context.source_dir or context.source_root,
        engine_path=context.engine_path,
        engine_metadata_path=context.engine_dir / "engine_metadata.json",
        config_file=context.source_root / "config" / "model.yaml",
        test_dir=context.source_root / "test",
        report_dir=context.report_dir,
        onnx_path=context.onnx_path,
        calibration_cache=context.intermediate_dir
        / context.settings.calibration_cache_filename,
        cuda_include_dirs=_cuda_include_dirs(context),
    )

    image_tag = f"qforge-{context.task_id[:12]}:{context.precision}"
    result = docker_build.build_image(
        context_dir=context_dir,
        image_tag=image_tag,
        log_path=context.log_file("docker_build.log"),
        timeout_seconds=context.settings.docker_build_timeout_seconds,
        base_image=base_image,
        # 默认在容器内重建 Engine：TensorRT 计划文件是平台相关的，
        # Windows 侧构建的 Engine 在 Linux 容器中加载会报 Platform specific tag mismatch（实测）。
        rebuild_engine=bool(context.build_options.get("docker_rebuild_engine", True)),
    )
    payload: dict[str, Any] = {
        "status": "SUCCESS",
        "image_tag": image_tag,
        "base_image": base_image,
        "dockerfile": "docker/Dockerfile",
        "build": result.to_dict(),
    }

    if context.build_options.get("docker_run", False):
        run_result = docker_build.run_container(
            image_tag=image_tag,
            log_path=context.log_file("docker_run.log"),
            timeout_seconds=context.settings.docker_run_timeout_seconds,
            extra_args=["--gpus", "all"],
        )
        payload["container_run"] = run_result.to_dict()
        logger.info(
            "容器运行验证：%s（退出码 %s）",
            "SUCCESS" if run_result.ok else "FAILED",
            run_result.returncode,
        )

    write_json(report_path, payload)
    context.reports.append("report/docker_build.json")


def run_packaging(context: PipelineContext) -> None:
    """归档产物、写入 Artifact 行、汇总报告（SPEC 12.2 / 13.1）。"""
    _run_docker_step(context)
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
    add("docker", context.docker_dir / "entrypoint.sh", "容器启动脚本")
    # 最终交付物：zip 归档（解压后跑 start.bat / start.sh 即可推理）
    camera_enabled = ((context.cpp_build_status or {}).get("camera") or {}).get("status") == "ENABLED"
    archive_note = "完整产物归档（解压后运行 start.bat / start.sh 即可推理"
    archive_note += "；含实时推理 start_camera" if camera_enabled else ""
    archive_note += "）"
    add("archive", archive_path, archive_note)
    # 逐个文件也登记，便于排障（界面上折叠展示）
    if context.source_dir:
        add("source", context.source_dir / "start.bat", "一键启动脚本（Windows）")
        add("source", context.source_dir / "start.sh", "一键启动脚本（Linux/macOS）")
        add("source", context.source_dir / "README.md", "交付说明（怎么跑、输出怎么读）")
        binary = context.source_dir / "build" / "qforge_detector.exe"
        add("engine", binary if binary.exists() else context.source_dir / "build" / "qforge_detector", "编译好的推理程序")
    # 报告按**实际生成**的清单登记（而不是写死名单，否则新增报告会漏登记）
    # 注意：context.reports 中的路径是相对**任务目录**的（如 report/model_info.json）
    for relative in dict.fromkeys(context.reports):
        report_path = context.task_dir / relative
        add("report", report_path, f"报告 {Path(relative).name}")

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
                description=entry["description"],
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

"""容器镜像构建与运行（SPEC 12.1 / 12.2）。

按 SPEC 12.1 的边界：
- 镜像构建由受控 Worker（本模块）执行，不由 API 进程直接做；
- 限制超时；构建日志完整保留到任务日志目录；
- 默认以最小权限运行（生成的 Dockerfile 里使用非 root 用户）。

构建上下文（build context）结构：

    docker/context/
    ├── Dockerfile
    ├── source/            生成的 C++ 工程
    ├── model/model.engine + engine_metadata.json（+ model.onnx，用于容器内重建）
    ├── calibration/       校准缓存与校准图（INT8 时）
    ├── config/ test/ report/
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.errors import BackendUnavailableError, DockerBuildFailedError

logger = logging.getLogger(__name__)

_EXCLUDED_DIRS = {"build", "__pycache__", ".cmake", "CMakeFiles", "context"}


@dataclass
class DockerResult:
    ok: bool
    image_tag: str
    duration_seconds: float
    log_path: Path
    log_text: str
    returncode: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "image_tag": self.image_tag,
            "returncode": self.returncode,
            "duration_seconds": round(self.duration_seconds, 2),
            "log": str(self.log_path),
            "tail": "\n".join(self.log_text.splitlines()[-20:]),
        }


def docker_available() -> tuple[bool, dict[str, Any]]:
    """检查 docker CLI 与 daemon 是否可用。"""
    executable = shutil.which("docker")
    if executable is None:
        return False, {"reason": "未找到 docker 命令"}
    try:
        completed = subprocess.run(
            [executable, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - 环境差异
        return False, {"reason": f"docker 调用失败：{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return False, {
            "reason": "Docker daemon 不可用（是否已启动 Docker Desktop？）",
            "stderr": (completed.stderr or "").strip()[-400:],
        }
    return True, {"server_version": (completed.stdout or "").strip(), "executable": executable}


def stage_context(
    *,
    context_dir: Path,
    dockerfile: Path,
    source_dir: Path,
    engine_path: Path,
    engine_metadata_path: Path | None,
    config_file: Path | None,
    test_dir: Path | None,
    report_dir: Path | None,
    onnx_path: Path | None,
    calibration_cache: Path | None,
    cuda_include_dirs: list[Path] | None = None,
    extra_inputs: dict[str, Path] | None = None,
) -> Path:
    """组装 build context（已存在则先清空，保证可重复构建）。"""
    if context_dir.exists():
        shutil.rmtree(context_dir)
    context_dir.mkdir(parents=True)

    shutil.copy2(dockerfile, context_dir / "Dockerfile")
    # 容器启动脚本（Dockerfile 会 COPY 它）
    entrypoint = dockerfile.parent / "entrypoint.sh"
    if entrypoint.exists():
        shutil.copy2(entrypoint, context_dir / "entrypoint.sh")

    def copy_tree(source: Path, target: Path) -> None:
        for path in source.rglob("*"):
            if any(part in _EXCLUDED_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    if source_dir.exists():
        copy_tree(source_dir, context_dir / "source")

    model_dir = context_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(engine_path, model_dir / engine_path.name)
    if engine_metadata_path and engine_metadata_path.exists():
        shutil.copy2(engine_metadata_path, model_dir / engine_metadata_path.name)
    if onnx_path and onnx_path.exists():
        # 供 --build-arg REBUILD_ENGINE=1 在容器内用 trtexec 重建 Engine
        shutil.copy2(onnx_path, model_dir / onnx_path.name)

    if calibration_cache and calibration_cache.exists():
        calib_dir = context_dir / "calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(calibration_cache, calib_dir / calibration_cache.name)

    # CUDA 头文件：基础镜像只带 CUDA 运行库，编译需要 cuda_runtime_api.h 与 crt/*，
    # 因此把平台已探测到的 CUDA include 目录内容合并进构建上下文（仅头文件，不含库）。
    cuda_target = context_dir / "cuda-include"
    cuda_target.mkdir(parents=True, exist_ok=True)
    for directory in cuda_include_dirs or []:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                destination = cuda_target / path.relative_to(directory)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

    for source, relative in (
        (config_file, "config/model.yaml"),
        (test_dir, "test"),
        (report_dir, "report"),
    ):
        if source is None or not source.exists():
            continue
        target = context_dir / relative
        if source.is_dir():
            copy_tree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    for name, path in (extra_inputs or {}).items():
        if path.exists():
            shutil.copy2(path, context_dir / name)

    logger.info("Docker 构建上下文已就绪：%s", context_dir)
    return context_dir


def _run(
    command: list[str], *, cwd: Path, log_path: Path, timeout: int
) -> tuple[int, float, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            handle.write(f"\n!!! 超时（{timeout}s）\n")
            returncode = 124
    return returncode, time.time() - started, log_path.read_text(encoding="utf-8", errors="replace")


def build_image(
    *,
    context_dir: Path,
    image_tag: str,
    log_path: Path,
    timeout_seconds: int,
    base_image: str | None = None,
    rebuild_engine: bool = False,
    extra_build_args: dict[str, str] | None = None,
) -> DockerResult:
    """执行 docker build（SPEC 12.1：由受控 Worker 执行并保留完整日志）。"""
    available, info = docker_available()
    if not available:
        raise BackendUnavailableError(
            "Docker 不可用，无法构建镜像", detail={"docker": info}
        )
    executable = info["executable"]

    command = [executable, "build", "--tag", image_tag, "--file", str(context_dir / "Dockerfile")]
    if base_image:
        command += ["--build-arg", f"BASE_IMAGE={base_image}"]
    command += ["--build-arg", f"REBUILD_ENGINE={1 if rebuild_engine else 0}"]
    for key, value in (extra_build_args or {}).items():
        command += ["--build-arg", f"{key}={value}"]
    command.append(str(context_dir))

    returncode, duration, text = _run(
        command, cwd=context_dir, log_path=log_path, timeout=timeout_seconds
    )
    result = DockerResult(
        ok=returncode == 0,
        image_tag=image_tag,
        duration_seconds=duration,
        log_path=log_path,
        log_text=text,
        returncode=returncode,
    )
    if not result.ok:
        raise DockerBuildFailedError(
            f"镜像构建失败（退出码 {returncode}）：\n"
            + "\n".join(text.splitlines()[-20:]),
            detail=result.to_dict(),
        )
    logger.info("镜像构建成功：%s（%.1fs）", image_tag, duration)
    return result


def run_container(
    *,
    image_tag: str,
    log_path: Path,
    timeout_seconds: int,
    extra_args: list[str] | None = None,
    name: str | None = None,
) -> DockerResult:
    """运行容器（默认 CMD 会加载 Engine 并做一次真实推理）。

    注意：本机 Docker 为 Linux 容器，容器内**没有** NVIDIA GPU 直通（未配置 nvidia runtime），
    因此容器内的推理会以 CPU 侧失败并在日志中说明；这里如实记录结果，不伪造成功。
    """
    available, info = docker_available()
    if not available:
        raise BackendUnavailableError("Docker 不可用，无法运行容器", detail={"docker": info})
    executable = info["executable"]

    command = [executable, "run", "--rm"]
    if name:
        command += ["--name", name]
    command += extra_args or []
    command.append(image_tag)

    returncode, duration, text = _run(
        command, cwd=log_path.parent, log_path=log_path, timeout=timeout_seconds
    )
    return DockerResult(
        ok=returncode == 0,
        image_tag=image_tag,
        duration_seconds=duration,
        log_path=log_path,
        log_text=text,
        returncode=returncode,
    )

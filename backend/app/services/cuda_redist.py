"""从 NVIDIA 公共分发站获取 CUDA 运行时开发文件（**无需登录**）。

为什么需要它：编译生成的 C++ 工程需要 CUDA 头文件（`cuda_runtime_api.h` 与 `crt/*`）。
TensorRT 开发包必须登录 NVIDIA 账号才能下载，但 **CUDA 这一半是公开的**，
因此「安装即用」的安装脚本可以自动把 CUDA 头文件准备好，把用户的手工步骤降到 0。

清单来源（官方、公开）：
    https://developer.download.nvidia.com/compute/cuda/redist/redistrib_<version>.json

安全约定（AGENTS.md：一律防御路径穿越）：解压前逐个校验成员路径必须落在目标目录内。
"""

from __future__ import annotations

import json
import logging
import platform
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REDIST_BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist"
# 默认版本与 docs/versions.md 的记录保持一致（CUDA 13.0.x）
DEFAULT_CUDA_REDIST_VERSION = "13.0.0"
# 需要的组件：cudart（头文件 + 导入库 + 运行时 DLL）、crt（crt/host_defines.h）、cccl（thrust 等）
DEFAULT_COMPONENTS = ("cuda_cudart", "cuda_crt", "cuda_cccl")

_USER_AGENT = "QForge-installer/0.1 (+https://github.com/DebaKong/QForge)"


@dataclass(frozen=True)
class CudaComponent:
    """一个待下载的 CUDA 分发包。"""

    name: str
    platform_key: str
    relative_path: str
    size_bytes: int | None = None

    @property
    def url(self) -> str:
        return f"{REDIST_BASE_URL}/{self.relative_path}"

    @property
    def filename(self) -> str:
        return self.relative_path.rsplit("/", 1)[-1]

    @property
    def archive_kind(self) -> str:
        if self.filename.endswith(".zip"):
            return "zip"
        if self.filename.endswith((".tar.xz", ".tar.gz", ".tgz")):
            return "tar"
        raise ValueError(f"不认识的压缩格式：{self.filename}")


def platform_key(system: str | None = None) -> str:
    """映射到 NVIDIA 清单里的平台键（例：windows-x86_64 / linux-x86_64）。"""
    system = (system or platform.system()).lower()
    machine = platform.machine().lower()
    arch = "x86_64" if machine in {"amd64", "x86_64"} else machine
    if system == "windows":
        return f"windows-{arch}"
    if system == "linux":
        return f"linux-{arch}"
    return f"{system}-{arch}"


def manifest_url(version: str = DEFAULT_CUDA_REDIST_VERSION) -> str:
    return f"{REDIST_BASE_URL}/redistrib_{version}.json"


def fetch_manifest(url: str | None = None, timeout: float = 60.0) -> dict:
    """取 CUDA redist 清单（JSON）。"""
    request = urllib.request.Request(url or manifest_url(), headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 固定官方域名
        return json.loads(response.read().decode("utf-8"))


def plan_components(
    manifest: dict,
    *,
    key: str | None = None,
    components: tuple[str, ...] = DEFAULT_COMPONENTS,
) -> list[CudaComponent]:
    """从清单里挑出需要的组件（找不到的组件会被跳过，由调用方据此报告）。

    实测（redistrib_13.0.0.json）：平台项是**字典**（`{relative_path, sha256, size}`），
    但历史上也见过列表形式，因此两种都接受。
    """
    key = key or platform_key()
    planned: list[CudaComponent] = []
    for name in components:
        entry = manifest.get(name)
        if not isinstance(entry, dict):
            continue
        raw = entry.get(key)
        if raw is None:
            continue
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if not isinstance(item, dict):
                continue
            relative = item.get("relative_path")
            if not relative:
                continue
            # 实测：清单里的 size 是**字符串**（如 "1048576"），这里统一转成整数
            raw_size = item.get("size")
            size: int | None = None
            if isinstance(raw_size, (int, str)):
                try:
                    size = int(raw_size)
                except (TypeError, ValueError):
                    size = None
            planned.append(
                CudaComponent(
                    name=name,
                    platform_key=key,
                    relative_path=relative,
                    size_bytes=size,
                )
            )
    return planned


def _safe_members(names: list[str], target: Path) -> list[str]:
    target_resolved = target.resolve()
    safe: list[str] = []
    for name in names:
        candidate = (target / name).resolve()
        if candidate == target_resolved or target_resolved in candidate.parents:
            safe.append(name)
        else:
            logger.warning("跳过越界的压缩包成员：%s", name)
    return safe


def download_and_extract(component: CudaComponent, target: Path, *, timeout: float = 600.0) -> Path:
    """下载并解压一个组件到 `target`，返回解压后的根目录。"""
    target.mkdir(parents=True, exist_ok=True)
    archive = target / component.filename
    logger.info("下载 %s（%.1f MB）", component.url, (component.size_bytes or 0) / 1024**2)
    request = urllib.request.Request(component.url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, archive.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)

    extract_root = target / component.filename.rsplit("-archive", 1)[0]
    try:
        if component.archive_kind == "zip":
            with zipfile.ZipFile(archive) as bundle:
                members = _safe_members(bundle.namelist(), extract_root)
                bundle.extractall(extract_root, members=members)
        else:
            with tarfile.open(archive) as bundle:  # noqa: S202 - 成员已过滤
                members = _safe_members(bundle.getnames(), extract_root)
                bundle.extractall(extract_root, members=members)  # noqa: S202
    finally:
        archive.unlink(missing_ok=True)
    return extract_root


def install(
    target: Path,
    *,
    version: str = DEFAULT_CUDA_REDIST_VERSION,
    key: str | None = None,
    components: tuple[str, ...] = DEFAULT_COMPONENTS,
    dry_run: bool = False,
) -> list[CudaComponent]:
    """按清单下载并解压全部组件；`dry_run` 只解析不下载。"""
    planned = plan_components(fetch_manifest(manifest_url(version)), key=key, components=components)
    if dry_run:
        return planned
    for component in planned:
        download_and_extract(component, target)
    return planned

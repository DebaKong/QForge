"""TensorRT / CUDA 开发文件定位（编译生成的 C++ 工程所需）。

背景（阶段 1 实测）：
- `pip install tensorrt` 只提供**运行库 DLL**（tensorrt_libs/*.dll），
  **不包含** `NvInfer.h` 与 `nvinfer_10.lib`；
- CUDA 运行时同理：pip 环境里没有 `cuda_runtime_api.h` 与 `cudart.lib`。

因此编译 C++ 工程需要额外的开发文件。本模块负责在以下位置查找并给出
可以注入 CMake 的 include/lib/DLL 目录；找不到时抛出 BACKEND_UNAVAILABLE
并说明缺什么、去哪里取，而不是让 CMake 报一堆难以理解的错误。

查找顺序：
1. 配置项 `QFORGE_TENSORRT_ROOT` / `QFORGE_CUDA_ROOT`；
2. `toolchain_search_roots`（默认含 D:\\qforge-toolchain 与 CUDA Toolkit 默认安装路径）；
3. pip site-packages（只可能提供 DLL，用于运行期）。
"""

from __future__ import annotations

import logging
import os
import re
import site
from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import Settings
from app.errors import BackendUnavailableError

logger = logging.getLogger(__name__)

TRT_HEADER = "NvInfer.h"
CUDA_HEADER = "cuda_runtime_api.h"
TRT_LIB_NAMES = ("nvinfer", "nvinfer_plugin", "nvonnxparser")
CUDA_LIB_NAMES = ("cudart",)


@dataclass
class DevFiles:
    """编译与运行所需的目录集合。"""

    tensorrt_include_dirs: list[str] = field(default_factory=list)
    tensorrt_lib_dirs: list[str] = field(default_factory=list)
    tensorrt_libs: list[str] = field(default_factory=list)
    cuda_include_dirs: list[str] = field(default_factory=list)
    cuda_lib_dirs: list[str] = field(default_factory=list)
    cuda_libs: list[str] = field(default_factory=list)
    runtime_dll_dirs: list[str] = field(default_factory=list)
    cuda_runtime_version: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, object]:
        return {
            "tensorrt_include_dirs": self.tensorrt_include_dirs,
            "tensorrt_lib_dirs": self.tensorrt_lib_dirs,
            "tensorrt_libs": self.tensorrt_libs,
            "cuda_include_dirs": self.cuda_include_dirs,
            "cuda_lib_dirs": self.cuda_lib_dirs,
            "cuda_libs": self.cuda_libs,
            "runtime_dll_dirs": self.runtime_dll_dirs,
            "cuda_runtime_version": self.cuda_runtime_version,
            "problems": self.problems,
            "complete": self.complete,
        }


def _search_roots(settings: Settings) -> list[Path]:
    roots: list[Path] = []
    for root in settings.toolchain_search_roots:
        if root.exists():
            roots.append(root)
    return roots


def _find_file(roots: list[Path], filename: str, *, max_depth: int = 5) -> Path | None:
    """在若干根目录下按深度受限地查找文件。

    注意：不要用 `glob("*/*/*/" + filename)` 这类固定层数模式——分发包的目录层数
    会随版本变化（如 `tensorrt/TensorRT-10.16.1.11/include/`、`cuda/<archive>/include/`），
    这里用 os.walk 做**深度剪枝**，既准确又不会遍历整棵大树。
    """
    for root in roots:
        direct = root / "include" / filename
        if direct.exists():
            return direct

        root_depth = len(root.parts)
        for current, directories, files in os.walk(root):
            current_depth = len(Path(current).parts) - root_depth
            if current_depth >= max_depth:
                directories[:] = []  # 剪枝：不再往下走
            if filename in files:
                return Path(current) / filename
    return None


def _find_crt_include(roots: list[Path], *, max_depth: int = 5) -> Path | None:
    """定位包含 `crt/host_defines.h` 的 include 目录。

    CUDA 13 把 C 运行时头文件拆到了单独的 `cuda_crt` 分发包：
    `cuda_runtime_api.h` 会 `#include <crt/host_defines.h>`，只装 cudart 包会在编译期报
    「Cannot open include file: 'crt/host_defines.h'」。这里把它找出来一并加入 include 路径。
    """
    for root in roots:
        root_depth = len(root.parts)
        for current, directories, files in os.walk(root):
            current_depth = len(Path(current).parts) - root_depth
            if current_depth >= max_depth:
                directories[:] = []
            if Path(current).name == "crt" and "host_defines.h" in files:
                return Path(current).parent
    return None


def cuda_runtime_version(header: Path | None) -> str | None:
    """从 cuda_runtime_api.h 读取 CUDA 运行时版本（SPEC 4.1 需要记录具体版本）。"""
    if header is None or not header.exists():
        return None
    try:
        content = header.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return None

    match = re.search(r"#define\s+CUDART_VERSION\s+(\d+)", content)
    if not match:
        return None
    value = int(match.group(1))
    return f"{value // 1000}.{(value % 1000) // 10}"


def _lib_dirs_from_header(header: Path) -> list[Path]:
    """由 include 目录反推 lib 目录（分发包装解压后的常见结构）。

    CUDA 分发包的结构是 `<pkg>/lib/x64/`；TensorRT 的是 `<pkg>/lib/`。
    """
    root = header.parent.parent  # <pkg>/include/<header>
    candidates = [
        root / "lib",
        root / "lib" / "x64",
        root / "lib" / "x86_64",
    ]
    return [path for path in candidates if path.is_dir()]


def _bin_dirs_from_header(header: Path) -> list[Path]:
    """由 include 目录反推运行库 DLL 目录（同样存在 bin/ 与 bin/x64/ 两种布局）。"""
    root = header.parent.parent
    candidates = [
        root / "bin",
        root / "bin" / "x64",
        root / "bin" / "x86_64",
        root / "lib",
        root / "lib" / "x64",
    ]
    return [path for path in candidates if path.is_dir()]


def _resolve_lib_stem(lib_dirs: list[Path], base_name: str) -> str | None:
    """在 lib 目录中找到导入库并返回**实际文件名（不含扩展名）**。

    TensorRT 10 的导入库带版本后缀（`nvinfer_10.lib`、`nvinfer_plugin_10.lib`、
    `nvonnxparser_10.lib`），因此不能按 `nvinfer.lib` 硬匹配；链接时应使用真实文件名。
    """
    for directory in lib_dirs:
        exact = directory / f"{base_name}.lib"
        if exact.exists():
            return base_name
    for directory in lib_dirs:
        versioned = sorted(
            path
            for path in directory.glob(f"{base_name}*.lib")
            if re.fullmatch(rf"{re.escape(base_name)}_\d+", path.stem)
        )
        if versioned:
            return versioned[0].stem
    return None


def locate_dev_files(settings: Settings) -> DevFiles:
    """定位 TensorRT / CUDA 开发文件；缺失项记入 problems（不直接抛错）。"""
    result = DevFiles()
    roots = _search_roots(settings)

    # ---------------- TensorRT ----------------
    trt_header: Path | None = None
    if settings.tensorrt_root is not None:
        candidate = Path(settings.tensorrt_root) / "include" / TRT_HEADER
        if candidate.exists():
            trt_header = candidate
    if trt_header is None:
        trt_header = _find_file(roots, TRT_HEADER)

    if trt_header is not None:
        result.tensorrt_include_dirs = [str(trt_header.parent)]
        lib_dirs = _lib_dirs_from_header(trt_header)
        result.tensorrt_lib_dirs = [str(path) for path in lib_dirs]
        resolved: set[str] = set()
        for name in TRT_LIB_NAMES:
            stem = _resolve_lib_stem(lib_dirs, name)
            if stem:
                # 记录**实际文件名**（如 nvinfer_10），供 CMake 链接
                result.tensorrt_libs.append(stem)
                resolved.add(name)
        missing = [name for name in TRT_LIB_NAMES if name not in resolved]
        if missing:
            result.problems.append(
                "TensorRT 导入库缺失："
                + ", ".join(missing)
                + "（需要 nvinfer / nvinfer_plugin / nvonnxparser 的 .lib）"
            )
        # DLL：优先用开发包 bin（与 .lib 版本一致），其次用 pip 包里的运行库
        result.runtime_dll_dirs.extend(str(path) for path in _bin_dirs_from_header(trt_header))
    else:
        result.problems.append(
            f"缺少 TensorRT C++ 开发文件（{TRT_HEADER}）：pip 包只提供运行库 DLL，"
            "需下载 TensorRT Windows 开发包并解压（见 docs/phase-1.md 的环境准备章节）"
        )

    # pip 里的运行库 DLL 作为兜底（放在最后：CMake 复制时后者覆盖前者，
    # 因此开发包的 DLL 优先级更高，保证 .lib 与 DLL 版本一致）
    for site_dir in {Path(site.getsitepackages()[0]) if site.getsitepackages() else None}:
        if site_dir is None:
            continue
        for name in ("tensorrt_libs",):
            candidate = site_dir / name
            if candidate.is_dir():
                result.runtime_dll_dirs.append(str(candidate))

    # ---------------- CUDA ----------------
    cuda_header: Path | None = None
    if settings.cuda_root is not None:
        candidate = Path(settings.cuda_root) / "include" / CUDA_HEADER
        if candidate.exists():
            cuda_header = candidate
    if cuda_header is None:
        cuda_header = _find_file(roots, CUDA_HEADER)

    if cuda_header is not None:
        result.cuda_include_dirs = [str(cuda_header.parent)]
        result.cuda_runtime_version = cuda_runtime_version(cuda_header)
        lib_dirs = _lib_dirs_from_header(cuda_header)
        result.cuda_lib_dirs = [str(path) for path in lib_dirs]
        for name in CUDA_LIB_NAMES:
            stem = _resolve_lib_stem(lib_dirs, name)
            if stem:
                result.cuda_libs.append(stem)
        if not result.cuda_libs:
            result.problems.append("CUDA 导入库缺失：未找到 cudart.lib")
        result.runtime_dll_dirs.extend(str(path) for path in _bin_dirs_from_header(cuda_header))

        crt_include = _find_crt_include(roots)
        if crt_include is not None and str(crt_include) not in result.cuda_include_dirs:
            result.cuda_include_dirs.append(str(crt_include))
        elif not (cuda_header.parent / "crt").is_dir():  # pragma: no cover - 环境缺失时提示
            result.problems.append(
                "缺少 CUDA C 运行时头文件（crt/host_defines.h）：需要 cuda_crt 分发包"
            )
    else:
        result.problems.append(
            f"缺少 CUDA 运行时开发文件（{CUDA_HEADER}）：需 CUDA 分发包中的 cudart 头文件与导入库"
        )

    # 去重并保持稳定顺序
    for attribute in (
        "tensorrt_include_dirs",
        "tensorrt_lib_dirs",
        "cuda_include_dirs",
        "cuda_lib_dirs",
        "runtime_dll_dirs",
    ):
        values = getattr(result, attribute)
        setattr(result, attribute, sorted(dict.fromkeys(values)))
    result.tensorrt_libs = list(dict.fromkeys(result.tensorrt_libs))
    result.cuda_libs = list(dict.fromkeys(result.cuda_libs))

    if result.problems:
        logger.warning("开发文件不完整：%s", result.problems)
    else:
        logger.info("开发文件齐备：%s", result.to_dict())
    return result


def require_dev_files(settings: Settings) -> DevFiles:
    """编译前调用：不完整即抛 BACKEND_UNAVAILABLE（环境问题，必须显式上报）。"""
    files = locate_dev_files(settings)
    if not files.complete:
        raise BackendUnavailableError(
            "编译 C++ 工程所需的开发文件不完整，无法执行编译与运行验证",
            detail={"problems": files.problems, "dev_files": files.to_dict()},
        )
    return files

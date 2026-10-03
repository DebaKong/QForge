"""OpenCV 开发文件探测（**可选能力**：实时摄像头 / 视频文件 / 预览窗口）。

为什么要单独一个模块：
- OpenCV 不是核心流程的依赖（默认构建零第三方依赖），只有「实时推理」需要它；
- 它的安装位置很杂（官方 Windows 包、conda 的 `<prefix>/Library`、vcpkg、Linux 的 /usr），
  因此这里统一探测出 include / lib / bin 三个目录，供编译（-DQFORGE_WITH_OPENCV=ON）与
  交付打包（把运行库 DLL 放进 zip）共用。

约定：`root` 是"含 include/ 与 lib/ 的目录"，Windows 官方包与 conda 都在 root 或 root/Library 下。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# 相对 <搜索根> 可能出现的层级（官方包 build/、conda 的 Library/、vcpkg 等）
_CANDIDATE_SUFFIXES = (
    "",
    "Library",
    "build",
    "opencv",
    "opencv/build",
    "opencv-dev",
    "opencv-dev/Library",
)

_VERSION_PATTERNS = {
    "major": re.compile(r"CV_VERSION_MAJOR\s+(\d+)"),
    "minor": re.compile(r"CV_VERSION_MINOR\s+(\d+)"),
    "patch": re.compile(r"CV_VERSION_REVISION\s+(\d+)"),
}


@dataclass
class OpenCvFiles:
    """OpenCV 开发文件位置。`complete` 为真才可用于编译。"""

    root: Path | None = None
    include_dir: Path | None = None
    lib_dir: Path | None = None
    bin_dir: Path | None = None
    version: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.include_dir is not None and self.lib_dir is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root) if self.root else None,
            "include_dir": str(self.include_dir) if self.include_dir else None,
            "lib_dir": str(self.lib_dir) if self.lib_dir else None,
            "bin_dir": str(self.bin_dir) if self.bin_dir else None,
            "version": self.version,
            "complete": self.complete,
            "problems": self.problems,
        }


def _read_version(include_dir: Path) -> str | None:
    header = include_dir / "opencv2" / "core" / "version.hpp"
    if not header.is_file():
        return None
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - 权限异常
        return None
    parts: list[str] = []
    for key in ("major", "minor", "patch"):
        match = _VERSION_PATTERNS[key].search(text)
        if match is None:
            return None
        parts.append(match.group(1))
    return ".".join(parts)


def _lib_dir(root: Path) -> Path | None:
    """库目录：Windows 官方包在 x64/vc16|vc17/lib，conda 在 lib/。"""
    candidates = [root / "lib", root / "lib64"]
    for pattern in ("x64/vc17/lib", "x64/vc16/lib", "x64/vc15/lib", "x64/mingw/lib", "lib/x64"):
        candidates.append(root / pattern)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        has_library = any(candidate.glob("opencv_*.lib")) or any(
            candidate.glob("libopencv_*.so*")
        )
        if has_library:
            return candidate
    return None


def probe(root: Path, *, with_library_subdir: bool = False) -> OpenCvFiles | None:
    """检查某个目录是否是 OpenCV 根目录；不是就返回 None。"""
    if not root.is_dir():
        return None
    include_dir = root / "include"
    if not (include_dir / "opencv2" / "opencv.hpp").is_file():
        return None
    lib_dir = _lib_dir(root)
    if with_library_subdir and lib_dir is None:
        return None
    return OpenCvFiles(
        root=root,
        include_dir=include_dir,
        lib_dir=lib_dir,
        bin_dir=(root / "bin" if (root / "bin").is_dir() else None),
        version=_read_version(include_dir),
    )


def locate(settings: Settings | None = None, *, max_depth: int = 3) -> OpenCvFiles:
    """定位 OpenCV 开发文件。

    顺序：显式配置 `QFORGE_OPENCV_ROOT` → 工具链搜索根（含数据目录下的 toolchain/）。
    找不到时返回带 problems 的结果（**不是异常**：OpenCV 属可选能力）。
    """
    settings = settings or get_settings()

    explicit = Path(settings.opencv_root) if settings.opencv_root else None
    if explicit is not None:
        probed = probe(explicit) or (probe(explicit / "Library") if True else None)
        if probed is not None and probed.complete:
            logger.info("使用配置的 OpenCV：%s（%s）", probed.root, probed.version)
            return probed
        missing = OpenCvFiles(problems=[f"QFORGE_OPENCV_ROOT={explicit} 下没有找到 OpenCV 开发文件"])
        missing.problems.append("该目录应包含 include/opencv2/opencv.hpp 与 lib/opencv_*.lib")
        return missing

    for search_root in settings.toolchain_search_roots:
        base = Path(search_root)
        if not base.is_dir():
            continue
        for suffix in _CANDIDATE_SUFFIXES:
            candidate = base / suffix if suffix else base
            probed = probe(candidate)
            if probed is not None and probed.complete:
                logger.info("自动探测到 OpenCV：%s（%s）", probed.root, probed.version)
                return probed
        # 再往下找一层（例如 D:\tools\opencv-4.11.0\build）
        for child in sorted(base.glob("*")):
            if not child.is_dir() or max_depth <= 1:
                continue
            for suffix in ("", "Library", "build"):
                candidate = child / suffix if suffix else child
                probed = probe(candidate)
                if probed is not None and probed.complete:
                    logger.info("自动探测到 OpenCV：%s（%s）", probed.root, probed.version)
                    return probed

    return OpenCvFiles(
        problems=[
            "未找到 OpenCV 开发文件（实时摄像头/视频/预览需要）",
            "可任选：1) 设置 QFORGE_OPENCV_ROOT 指向含 include/ 与 lib/ 的目录；"
            "2) 把 OpenCV 解压/安装到工具链搜索根下（如 D:/qforge-toolchain）；"
            "3) 只用 --stdin-frames + ffmpeg 的零依赖方案",
        ]
    )

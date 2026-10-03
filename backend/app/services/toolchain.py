"""本机构建工具链探测与调用（SPEC 11.3 步骤 2~3）。

本机情况（阶段 1 实测）：已安装 Visual Studio Professional 2019（`D:\\vs2019`）与
Windows SDK 10.0.19041，但 `cl.exe` 不在 PATH；CMake 与 Ninja 由 pip 安装在 qforge 环境内。
因此这里统一的做法是：

1. 用 vswhere 或常见路径定位 `vcvars64.bat`；
2. 在**同一个 cmd 进程**里先 call vcvars64.bat 再执行 cmake/ninja，
   这样编译器与系统库环境一次生效，不污染宿主机环境变量；
3. 全过程输出重定向到任务日志文件，失败时保留完整日志（SPEC 15：编译失败必须保留日志）。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import get_settings
from app.errors import BackendUnavailableError, CppBuildFailedError

logger = logging.getLogger(__name__)

_VSWHERE = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
_COMMON_VS_ROOTS = (
    Path(r"D:\vs2019"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"),
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"),
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"),
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"),
)


@dataclass
class ToolchainInfo:
    """构建工具链探测结果。"""

    cmake: str | None = None
    ninja: str | None = None
    vcvars: Path | None = None
    visual_studio: str | None = None
    available: bool = False
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "cmake": self.cmake,
            "ninja": self.ninja,
            "vcvars": str(self.vcvars) if self.vcvars else None,
            "visual_studio": self.visual_studio,
            "available": self.available,
            "problems": self.problems,
        }


def _vswhere_installation() -> tuple[str | None, Path | None]:
    """通过 vswhere 找到带 C++ 工具集的 Visual Studio。"""
    if not _VSWHERE.exists():
        return None, None
    try:
        result = subprocess.run(
            [
                str(_VSWHERE),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - vswhere 异常时降级到常见路径
        logger.warning("vswhere 调用失败：%s", exc)
        return None, None

    path = (result.stdout or "").strip().splitlines()
    if not path:
        return None, None
    root = Path(path[0].strip())
    vcvars = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return (root.name, vcvars if vcvars.exists() else None)


def _find_executable(name: str, override: str | None = None) -> str | None:
    """定位可执行文件：显式配置 > PATH > 当前解释器所在环境的脚本目录。

    第三条很重要：pip 安装的 cmake / ninja 位于 `<env>\\Scripts`（Windows）或 `<env>/bin`，
    而进程未 `conda activate` 时该目录**不在 PATH 上**，只查 PATH 会误判为「工具链缺失」。
    """
    if override:
        candidate = Path(override)
        return str(candidate) if candidate.exists() else None

    found = shutil.which(name)
    if found:
        return found

    interpreter_dir = Path(sys.executable).parent
    suffixes = [f"{name}.exe", name] if os.name == "nt" else [name]
    for base in (interpreter_dir, interpreter_dir / "Scripts", interpreter_dir / "bin"):
        for suffix in suffixes:
            candidate = base / suffix
            if candidate.exists():
                return str(candidate)
    return None


def build_environment(info: ToolchainInfo | None = None) -> dict[str, str]:
    """返回带 CMake/Ninja 与 GPU 运行库目录的子进程环境变量。

    编译与「运行生成的程序」共用它：这样运行期可以直接从 TensorRT/CUDA 的 bin 目录
    加载 DLL，**不需要把 DLL 复制进每个任务目录**（那会白占 2 GB 以上）。
    """
    return _child_env(info or detect_toolchain())


def detect_toolchain() -> ToolchainInfo:
    """探测 CMake / Ninja / MSVC 环境。"""
    settings = get_settings()
    info = ToolchainInfo()

    info.cmake = _find_executable("cmake", settings.cmake_executable)
    if not info.cmake:
        info.problems.append("未找到 cmake（可用 pip install cmake 安装）")

    info.ninja = _find_executable("ninja")
    if not info.ninja:
        info.problems.append("未找到 ninja（可用 pip install ninja 安装）")

    if settings.vcvars_path is not None:
        candidate = Path(settings.vcvars_path)
        if candidate.exists():
            info.vcvars = candidate
            info.visual_studio = candidate.parent.parent.parent.name
        else:
            info.problems.append(f"配置的 vcvars 路径不存在：{candidate}")
    else:
        name, vcvars = _vswhere_installation()
        if vcvars is not None:
            info.visual_studio = name
            info.vcvars = vcvars
        else:
            for root in _COMMON_VS_ROOTS:
                candidate = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if candidate.exists():
                    info.vcvars = candidate
                    info.visual_studio = root.name
                    break

    if info.vcvars is None:
        info.problems.append("未找到 vcvars64.bat（需要 Visual Studio 的 C++ 工具集）")

    info.available = not info.problems
    return info


def require_toolchain() -> ToolchainInfo:
    """构建前的硬性检查；不满足即抛 BACKEND_UNAVAILABLE（环境问题，不是代码缺陷）。"""
    info = detect_toolchain()
    if not info.available:
        raise BackendUnavailableError(
            "本机 C++ 构建工具链不完整，无法编译生成的工程",
            detail={"problems": info.problems, "toolchain": info.to_dict()},
        )
    return info


def build_environment(extra_dll_dirs: list[str] | None = None) -> dict[str, str]:
    """构造子进程环境（编译与运行验证共用）。

    `extra_dll_dirs` 用于追加可选能力的运行库目录（例如实时推理用的 OpenCV bin）。
    """
    info = require_toolchain()
    env = _child_env(info)
    if extra_dll_dirs:
        existing = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([*dict.fromkeys(extra_dll_dirs), existing])
    return env


def _child_env(info: ToolchainInfo) -> dict[str, str]:
    """构造子进程环境：把 cmake/ninja 与 GPU 运行库目录加入 PATH。"""
    env = os.environ.copy()
    extra: list[str] = []
    for executable in (info.cmake, info.ninja):
        if executable:
            extra.append(str(Path(executable).parent))

    # 把 TensorRT / CUDA 运行库目录也加入 PATH：编译期 ninja 与运行期被测程序都可能需要
    try:
        from app.adapters.backends import tensorrt_dev
        from app.config.settings import get_settings

        dev_files = tensorrt_dev.locate_dev_files(get_settings())
        extra.extend(dev_files.runtime_dll_dirs)
    except Exception:  # pragma: no cover - 探测失败不影响构建
        logger.debug("未能收集 GPU 运行库目录", exc_info=True)

    if extra:
        env["PATH"] = os.pathsep.join([*dict.fromkeys(extra), env.get("PATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"
    return env


@dataclass
class CommandResult:
    returncode: int
    duration_seconds: float
    log_path: Path
    log_text: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_in_vs_env(
    commands: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> CommandResult:
    """在 vcvars64 环境中依次执行命令，输出追加写入 log_path。

    实现要点（踩过的坑）：**不要**把带引号的命令串直接传给 `cmd /c`。
    Python 的 subprocess 会按 MSVCRT 规则把内层引号转义成 `\\"`，而 cmd.exe 不认这种转义，
    结果是整条命令以「'\\"...\\"' is not recognized」失败且**没有任何输出**，
    日志里只剩命令行，极难排查。因此这里把命令写成 .bat 文件再执行，彻底绕开引号问题。
    """
    info = require_toolchain()
    assert info.vcvars is not None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = log_path.with_suffix(".bat")
    script_body = "\r\n".join(
        [
            "@echo off",
            "chcp 65001 >nul",
            f'call "{info.vcvars}"',
            *commands,
            "exit /b %ERRORLEVEL%",
        ]
    )
    script_path.write_text(script_body + "\r\n", encoding="utf-8")

    started = time.time()
    logger.info("执行构建脚本：%s", script_path.name)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        for command in commands:
            handle.write(f"$ {command}\n")
        handle.write(f"(脚本：{script_path})\n")
        handle.flush()
        try:
            completed = subprocess.run(
                ["cmd", "/c", str(script_path)],
                cwd=str(cwd),
                env=_child_env(info),
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            handle.write(f"\n!!! 命令超时（{timeout_seconds}s）：{exc}\n")
            returncode = 124

    duration = time.time() - started
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return CommandResult(
        returncode=returncode, duration_seconds=duration, log_path=log_path, log_text=text
    )


def cmake_version(cmake: str) -> str | None:
    try:
        result = subprocess.run(
            [cmake, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
        return (result.stdout or "").strip().splitlines()[0] or None
    except Exception:  # pragma: no cover
        return None


def build_cpp_project(
    *,
    project_dir: Path,
    build_dir: Path,
    log_path: Path,
    timeout_seconds: int,
    jobs: int | None = None,
    extra_defines: dict[str, str] | None = None,
) -> CommandResult:
    """执行 CMake Configure + Build（SPEC 11.3 步骤 2~3）。

    `extra_defines` 用于把可选能力的开关传给 CMake（例如实时推理的
    `QFORGE_WITH_OPENCV=ON` 与 `QFORGE_OPENCV_ROOT=<目录>`）。
    """
    info = require_toolchain()
    assert info.cmake is not None

    defines = " ".join(
        f'-D{key}="{value}"' if " " in value else f"-D{key}={value}"
        for key, value in (extra_defines or {}).items()
    )
    generator = "-G Ninja" if info.ninja else ""
    configure = (
        f'"{info.cmake}" -S "{project_dir}" -B "{build_dir}" {generator} '
        f"-DCMAKE_BUILD_TYPE=Release {defines}"
    ).replace("  ", " ")
    build = f'"{info.cmake}" --build "{build_dir}" --config Release'
    if jobs:
        build += f" --parallel {jobs}"

    result = run_in_vs_env(
        [configure, build], cwd=project_dir, log_path=log_path, timeout_seconds=timeout_seconds
    )
    if not result.ok:
        tail = "\n".join(result.log_text.splitlines()[-25:])
        # 把编译错误尾部直接放进异常消息，便于在任务日志与测试输出中一眼看到原因
        raise CppBuildFailedError(
            f"C++ 工程构建失败（退出码 {result.returncode}）：\n{tail}",
            detail={
                "log": str(result.log_path),
                "duration_seconds": round(result.duration_seconds, 2),
                "tail": tail,
            },
        )
    return result

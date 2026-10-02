"""QForge 命令行入口（AGENTS.md 产品形态要求：安装即用）。

目标：用户只需要记住**一条**启动命令。

    qforge serve --open     启动服务并自动打开浏览器（自动选择执行方式，自动带起 worker）
    qforge doctor           环境体检：缺什么、怎么补，逐条给出可复制的命令
    qforge init             建数据目录与数据库表（首次 `serve` 也会自动完成）
    qforge worker           单独启动 Celery worker（有 Redis 时）
    qforge build-frontend   构建前端界面（需要 Node.js）
    qforge version          版本与关键路径

设计约定：
- 不引入任何新依赖（只用标准库 + 已有依赖），避免给「安装即用」增加负担；
- 所有检查都**如实报告**：缺失就写缺失、给出来源，不伪造可用（AGENTS.md 约束）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.runtime import effective_executor_mode, redis_reachable
from app.config.settings import get_settings
from app.paths import data_root, frontend_dir, project_root

STATUS_PASS = "通过"
STATUS_WARN = "警告"
STATUS_FAIL = "缺失"

PYTHON_REQUIRED = "3.10"
NODE_HINT = "安装 Node.js LTS：https://nodejs.org（或用 winget install OpenJS.NodeJS.LTS）"


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def _child_env() -> dict[str, str]:
    """子进程环境：仓库检出时补上 backend/ 与仓库根，保证 app / workers 可导入。"""
    env = dict(os.environ)
    root = project_root()
    if root is not None:
        parts = [str(root / "backend"), str(root)]
        existing = env.get("PYTHONPATH")
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) != 0


def _storage_writable(root: Path) -> tuple[bool, str]:
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".qforge-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(root)
    except OSError as exc:
        return False, f"{root}（{exc}）"


# --------------------------------------------------------------------------- #
# doctor：环境体检
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    required: bool
    status: str
    detail: str
    hint: str | None = None


def _check_python() -> Check:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if f"{sys.version_info.major}.{sys.version_info.minor}" == PYTHON_REQUIRED:
        return Check("Python", True, STATUS_PASS, version)
    if sys.version_info >= (3, 11):
        return Check(
            "Python",
            True,
            STATUS_WARN,
            f"{version}（本项目锁定 {PYTHON_REQUIRED}，其它版本未经验证）",
            f"建议用 Python {PYTHON_REQUIRED}（conda create -n qforge python={PYTHON_REQUIRED}）",
        )
    return Check(
        "Python",
        True,
        STATUS_FAIL,
        version,
        f"需要 Python {PYTHON_REQUIRED}.x：https://www.python.org/downloads/",
    )


def _check_storage() -> Check:
    settings = get_settings()
    ok, detail = _storage_writable(settings.resolved_storage_root)
    if ok:
        return Check("数据目录", True, STATUS_PASS, detail)
    return Check("数据目录", True, STATUS_FAIL, detail, "检查目录权限，或设置 QFORGE_DATA_DIR 指向可写目录")


def _nvidia_smi_driver_version() -> str | None:
    """用 nvidia-smi 取**显卡驱动**版本号（cuda_driver_info 给的是 CUDA 版本，两者别混）。"""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        output = subprocess.run(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout.strip()
    except OSError:
        return None
    return output.splitlines()[0].strip() if output else None


def _check_gpu() -> Check:
    from app.adapters.backends.cuda_memory import cuda_driver_info

    info = cuda_driver_info()
    if info.get("available"):
        parts = [str(info.get("device_name") or "NVIDIA GPU")]
        driver = _nvidia_smi_driver_version()
        if driver:
            parts.append(f"驱动 {driver}")
        parts.append(f"CUDA {info.get('driver_version')}")
        parts.append(str(info.get("gpu_arch")))
        parts.append(f"{info.get('total_memory_bytes', 0) / 1024**3:.1f} GB")
        return Check("GPU", True, STATUS_PASS, " / ".join(parts))

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            output = subprocess.run(
                [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout.strip()
            if output:
                return Check("GPU", True, STATUS_PASS, output.splitlines()[0])
        except OSError:
            pass
    return Check(
        "GPU",
        True,
        STATUS_FAIL,
        str(info.get("reason") or "未检测到 NVIDIA GPU"),
        "安装 NVIDIA 显卡驱动（Windows 用 GeForce Experience 或官网驱动）：https://www.nvidia.cn/drivers/",
    )


def _check_tensorrt() -> Check:
    try:
        import tensorrt  # noqa: PLC0415

        version = getattr(tensorrt, "__version__", "未知")
        return Check("TensorRT 运行库", True, STATUS_PASS, f"{version}（pip 安装，用于构建 Engine）")
    except Exception as exc:  # noqa: BLE001 - 需要把任何导入失败都如实报告
        return Check(
            "TensorRT 运行库",
            True,
            STATUS_FAIL,
            f"导入失败：{type(exc).__name__}: {exc}",
            "pip install tensorrt（版本需与 CUDA/驱动匹配，见 docs/versions.md）",
        )


def _check_cuda_bindings() -> Check:
    try:
        import cuda  # noqa: F401, PLC0415

        return Check("CUDA Python 绑定", True, STATUS_PASS, "可用（INT8 校准与推理需要）")
    except Exception as exc:  # noqa: BLE001
        return Check(
            "CUDA Python 绑定",
            True,
            STATUS_FAIL,
            f"导入失败：{type(exc).__name__}",
            "pip install cuda-python",
        )


def _check_redis() -> Check:
    settings = get_settings()
    url = settings.celery_broker_url
    if not redis_reachable(url):
        return Check(
            "Redis 任务队列",
            False,
            STATUS_WARN,
            f"未连接（{url}）→ 将使用进程内执行器，功能不受影响",
            "可选：docker compose up -d redis（装上后自动切换为队列模式，支持多任务并行）",
        )
    detail = f"已连接（{url}）"
    try:
        from workers.celery_app import celery_app  # noqa: PLC0415

        replies = celery_app.control.ping(timeout=1.0)
        detail += f"；在线 worker：{len(replies)} 个" if replies else "；⚠ 暂无在线 worker"
        if not replies:
            return Check(
                "Redis 任务队列",
                False,
                STATUS_WARN,
                detail,
                "启动 worker：qforge worker（或让 qforge serve 自动带起）",
            )
    except Exception:  # noqa: BLE001 - 探测失败不影响 Redis 本身可用
        pass
    return Check("Redis 任务队列", False, STATUS_PASS, detail)


def _check_dev_files() -> Check:
    from app.adapters.backends.tensorrt_dev import locate_dev_files

    files = locate_dev_files(get_settings())
    if files.complete:
        return Check(
            "TensorRT/CUDA 开发文件",
            False,
            STATUS_PASS,
            "齐备（用于编译生成的 C++ 工程）",
        )
    problems = "；".join(files.problems) or "未找到 TensorRT 开发包（头文件 + 导入库）"
    return Check(
        "TensorRT/CUDA 开发文件",
        False,
        STATUS_FAIL,
        f"缺失 → 只影响「编译生成的 C++ 工程」，其余流程不受影响。{problems}",
        "TensorRT 开发包需 NVIDIA 账号登录：https://developer.nvidia.com/tensorrt/download "
        "（下载 Windows ZIP 后解压到 QFORGE_TENSORRT_ROOT 指向的目录；CUDA 开发文件可由 "
        "qforge doctor 提示的公共分发站获取，或设置 QFORGE_CUDA_ROOT）",
    )


def _check_build_tools() -> Check:
    from app.services.toolchain import detect_toolchain

    info = detect_toolchain()
    parts = []
    if info.cmake:
        parts.append("cmake")
    if info.ninja:
        parts.append("ninja")
    if info.vcvars:
        parts.append(info.visual_studio or "MSVC")
    detail = "、".join(parts) if parts else "未找到任何 C++ 构建工具"

    if info.available:
        return Check("C++ 构建工具", False, STATUS_PASS, f"{detail}（可编译生成的 C++ 工程）")

    problems = "；".join(info.problems) or "缺少 cmake / ninja / MSVC 之一"
    hint = "pip install cmake ninja"
    if not info.vcvars:
        hint += (
            "；并安装 Visual Studio 生成工具（含「使用 C++ 的桌面开发」）："
            "https://visualstudio.microsoft.com/downloads/"
        )
    return Check(
        "C++ 构建工具",
        False,
        STATUS_FAIL,
        f"{detail} → 缺失项：{problems}（仅影响 C++ 工程编译验证，其余流程不受影响）",
        hint,
    )


def _check_docker() -> Check:
    from app.services import docker_build

    available, info = docker_build.docker_available()
    if available:
        return Check("Docker", False, STATUS_PASS, f"可用（{info.get('server_version', '')}）")
    return Check(
        "Docker",
        False,
        STATUS_WARN,
        f"不可用（{info.get('reason') or '未检测到 daemon'}）→ 仅影响「生成容器镜像」",
        "可选：安装 Docker Desktop 后重试",
    )


def _check_frontend() -> Check:
    built = frontend_dir()
    node = shutil.which("node") or shutil.which("npm")
    if built is not None:
        return Check("前端界面", False, STATUS_PASS, str(built))
    return Check(
        "前端界面",
        False,
        STATUS_WARN,
        "未构建 → API 可用，界面显示内置说明页",
        (f"qforge build-frontend（已检测到 Node）" if node else NODE_HINT),
    )


def collect_checks() -> list[Check]:
    return [
        _check_python(),
        _check_storage(),
        _check_gpu(),
        _check_tensorrt(),
        _check_cuda_bindings(),
        _check_redis(),
        _check_dev_files(),
        _check_build_tools(),
        _check_docker(),
        _check_frontend(),
    ]


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = collect_checks()
    settings = get_settings()
    required_failed = [c for c in checks if c.required and c.status == STATUS_FAIL]
    optional_failed = [c for c in checks if not c.required and c.status == STATUS_FAIL]
    mode = effective_executor_mode(settings)

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "checks": [c.__dict__ for c in checks],
                    "executor_mode": mode,
                    "storage_root": str(settings.resolved_storage_root),
                    "ok": not required_failed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if required_failed else 0

    width = 62
    print()
    print(f"QForge 环境体检（{settings.app_name} {settings.app_version}）")
    print("─" * width)
    for check in checks:
        tag = "必需" if check.required else "可选"
        print(f"[{tag}][{check.status}] {check.name}：{check.detail}")
        if check.hint and check.status != STATUS_PASS:
            print(f"           → {check.hint}")
    print("─" * width)

    warning = [c for c in checks if c.status == STATUS_WARN]
    if not required_failed:
        print("结论：核心流程可用（上传 → 校验 → 量化 → 构建 Engine → 真实推理 → 精度对比 → 打包）")
    else:
        print(f"结论：缺少 {len(required_failed)} 项必需组件，核心流程暂不可用（见上面 → 提示）")
    if optional_failed:
        print(f"      可选增强不可用 {len(optional_failed)} 项（不影响核心流程）")
    if warning:
        print(f"      另有 {len(warning)} 项提示（不影响使用）")

    mode_text = "Redis + Celery（独立 worker）" if mode == "celery" else "进程内后台执行器（无需 Redis）"
    print(f"      执行方式：{mode_text}")
    print(f"      数据目录：{settings.resolved_storage_root}")
    print()
    return 1 if required_failed else 0


# --------------------------------------------------------------------------- #
# init / serve / worker / build-frontend / version
# --------------------------------------------------------------------------- #
def cmd_init(args: argparse.Namespace) -> int:
    settings = get_settings()
    root = settings.resolved_storage_root
    root.mkdir(parents=True, exist_ok=True)

    from app.db.base import create_all

    create_all()
    print(f"数据目录：{root}")
    print(f"数据库  ：{settings.resolved_database_url}")
    print("表结构已就绪（正式 schema 变更以 Alembic 迁移为准：alembic -c backend/alembic.ini upgrade head）")
    if args.build_frontend:
        return cmd_build_frontend(args)
    return 0


def _spawn_worker(concurrency: int | None) -> subprocess.Popen[Any]:
    command = [sys.executable, "-m", "app.cli", "worker"]
    if concurrency:
        command += ["--concurrency", str(concurrency)]
    print(f"启动 worker：{' '.join(command)}")
    return subprocess.Popen(command, env=_child_env())


def cmd_serve(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.resolved_storage_root.mkdir(parents=True, exist_ok=True)
    mode = effective_executor_mode(settings)

    if not _port_free(args.host, args.port):
        print(f"端口被占用：{args.host}:{args.port}。请换 --port，或先停止占用该端口的进程。")
        return 2

    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{browser_host}:{args.port}"

    worker: subprocess.Popen[Any] | None = None
    if mode == "celery" and not args.no_worker:
        worker = _spawn_worker(args.concurrency)

    frontend = settings.resolved_frontend_dir
    print()
    print(f"{settings.app_name} {settings.app_version}")
    print(f"  数据目录 : {settings.resolved_storage_root}")
    print(
        "  执行方式 : "
        + ("Redis + Celery（已自动带起 worker）" if mode == "celery" else "进程内后台执行器（无需 Redis）")
    )
    print(f"  前端界面 : {frontend if frontend else '未构建（显示内置说明页）'}")
    print(f"  访问地址 : {url}")
    print(f"  接口文档 : {url}{settings.api_prefix}/docs")
    print("  停止服务 : Ctrl+C")
    print()

    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        import uvicorn

        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            reload=args.reload,
        )
    except ImportError:
        print("未安装 uvicorn：pip install -r backend/requirements.txt")
        return 1
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                worker.kill()
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    try:
        from workers.celery_app import celery_app
    except ImportError as exc:  # pragma: no cover - 安装形态下必然可导入
        print(f"无法导入 Celery 应用：{exc}（仓库检出时请在仓库根执行）")
        return 1

    pool = args.pool or ("solo" if os.name == "nt" else "prefork")
    argv = ["worker", "--loglevel", args.log_level.lower(), "--pool", pool]
    if args.concurrency:
        argv += ["--concurrency", str(args.concurrency)]
    if args.hostname:
        argv += ["--hostname", args.hostname]
    print(f"启动 Celery worker（pool={pool}）")
    celery_app.worker_main(argv)
    return 0


def cmd_build_frontend(args: argparse.Namespace) -> int:
    root = project_root()
    if root is None:
        print("未找到前端源码（当前不是仓库检出）。")
        print("已安装形态下请把构建产物放到数据目录同级的 web/ 目录：<数据目录>/../web")
        return 1

    frontend = root / "frontend"
    npm = shutil.which("npm")
    if npm is None:
        print(f"未找到 npm。{NODE_HINT}")
        return 1

    install_cmd = ["npm", "ci"] if (frontend / "package-lock.json").exists() else ["npm", "install"]
    for command in (install_cmd, ["npm", "run", "build"]):
        print(f"$ {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=str(frontend),
            env=_child_env(),
            shell=(os.name == "nt"),  # Windows 上 npm 是 .cmd，需要 shell 解析
            check=False,
        )
        if result.returncode != 0:
            print(f"命令失败（退出码 {result.returncode}）：{' '.join(command)}")
            return result.returncode
    print(f"前端已构建：{frontend / 'dist'}")
    print("重新启动服务即可直接打开界面（qforge serve --open）")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    settings = get_settings()
    root = project_root()
    print(f"{settings.app_name} {settings.app_version}")
    print(f"  Python      : {sys.version.split()[0]}（{sys.executable}）")
    print(f"  运行形态    : {'仓库检出' if root else '已安装'}" + (f"（{root}）" if root else ""))
    print(f"  数据目录    : {settings.resolved_storage_root}")
    print(f"  配置文件    : {settings.model_config.get('env_file')}")
    print(f"  执行方式    : {effective_executor_mode(settings)}")
    frontend = settings.resolved_frontend_dir
    print(f"  前端界面    : {frontend if frontend else '未构建'}")
    return 0


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qforge",
        description="QForge —— ONNX 自动化量化部署平台",
        epilog="先跑 `qforge doctor` 体检，再 `qforge serve --open` 启动。",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="环境体检：缺什么、怎么补")
    doctor.add_argument("--json", action="store_true", help="额外输出机器可读的 JSON")
    doctor.set_defaults(func=cmd_doctor)

    init = subparsers.add_parser("init", help="建数据目录与数据库表")
    init.add_argument("--build-frontend", action="store_true", help="顺带构建前端界面")
    init.set_defaults(func=cmd_init)

    serve = subparsers.add_parser("serve", help="启动服务（API + 可选 worker）")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    serve.add_argument("--reload", action="store_true", help="代码变更自动重载（开发用）")
    serve.add_argument("--no-worker", action="store_true", help="不自动带起 Celery worker")
    serve.add_argument("--concurrency", type=int, default=None, help="worker 并发（Celery 模式）")
    serve.add_argument("--log-level", default="INFO")
    serve.set_defaults(func=cmd_serve)

    worker = subparsers.add_parser("worker", help="单独启动 Celery worker")
    worker.add_argument("--pool", default=None, help="默认 Windows=solo，其它=prefork")
    worker.add_argument("--concurrency", type=int, default=None)
    worker.add_argument("--hostname", default=None)
    worker.add_argument("--log-level", default="INFO")
    worker.set_defaults(func=cmd_worker)

    frontend = subparsers.add_parser("build-frontend", help="构建前端界面（需要 Node.js）")
    frontend.set_defaults(func=cmd_build_frontend)

    version = subparsers.add_parser("version", help="版本与关键路径")
    version.set_defaults(func=cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - 供 python -m app.cli 调用
    raise SystemExit(main())

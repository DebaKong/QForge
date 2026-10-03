"""运行期配置。

SPEC 4.1：版本敏感组件必须锁定并记录具体版本。本模块只承载运行期设置，
版本矩阵以 `docs/versions.md` + `backend/requirements.txt` 为准，不在此处写版本区间。

环境变量统一使用 `QFORGE_` 前缀，例如：
    QFORGE_DATABASE_URL=postgresql+psycopg://user:pwd@127.0.0.1:5432/qforge
    QFORGE_CELERY_BROKER_URL=redis://127.0.0.1:6379/0
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.paths import config_file, data_root, frontend_dir

# backend/app/config/settings.py -> config -> app -> backend -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """进程级配置。默认值面向「安装即用」：无需 Redis、无需手工建目录即可启动。"""

    model_config = SettingsConfigDict(
        env_prefix="QFORGE_",
        env_file=str(config_file()),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "QForge"
    app_version: str = "0.1.0"
    environment: Literal["dev", "test", "prod"] = "dev"
    api_prefix: str = "/api"
    log_level: str = "INFO"

    # ---- 存储（SPEC 14.2）----
    # 默认自动适配运行形态：仓库检出用 <repo>/storage，pip 安装用用户数据目录（见 app/paths.py）
    storage_root: Path = Field(default_factory=data_root)

    # ---- 已构建的前端静态文件（安装即用：由 API 直接托管，运行期不需要 Node）----
    frontend_dir: Path | None = None

    # ---- 数据库（SPEC 14.1）----
    # 未显式配置时由 storage_root 推导 SQLite 文件；阶段 1 接 PostgreSQL 只需设置该变量。
    database_url: str | None = None
    database_echo: bool = False
    # 本地开发便捷建表；正式 schema 变更以 Alembic 迁移为准（backend/migrations）
    auto_create_schema: bool = True

    # ---- 任务队列（SPEC 3 / 13）----
    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"
    # 安装即用：默认 **false**（真正走队列，不阻塞 API 请求）。
    # 无 Redis 时执行方式会自动落到 local（见 app/config/runtime.py），该开关不产生影响。
    celery_task_always_eager: bool = False
    celery_task_eager_propagates: bool = True

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # ---- 资源与安全限制（SPEC 15）----
    # 阶段 1 起强制执行（上传校验、ZIP 限额、并发与超时）
    max_upload_mb: int = 2048
    max_zip_uncompressed_mb: int = 8192
    max_zip_entries: int = 20000
    max_zip_depth: int = 8
    task_timeout_seconds: int = 3600
    max_concurrent_tasks: int = 1

    # ---- 任务执行方式（SPEC 3.1：Web API 不直接执行长任务）----
    # auto（默认）：探测到 Redis 就用 celery，否则用进程内后台执行器（见 app/config/runtime.py）
    # local：强制进程内后台线程执行器；celery：强制投递到 broker
    executor_mode: Literal["local", "celery", "auto"] = "auto"

    # ---- 上传与归档（SPEC 8.2 / 15）----
    allowed_model_extensions: list[str] = [".onnx"]
    allowed_image_extensions: list[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    max_calibration_images: int = 5000

    # ---- 构建与运行验证工具链（SPEC 11.3）----
    # None 表示自动探测：cmake 用 PATH（qforge 内 pip 安装的 cmake），
    # vcvars 用 vswhere 与常见安装路径探测（本机为 D:\vs2019）
    cmake_executable: str | None = None
    vcvars_path: Path | None = None
    build_timeout_seconds: int = 900
    run_verify_timeout_seconds: int = 300

    # ---- TensorRT（SPEC 10）----
    trt_workspace_gb: int = 2
    calibration_batch_size: int = 8
    calibration_cache_filename: str = "calibration.cache"

    # ---- C++ 构建所需的开发文件（阶段 1 实测：pip 只提供运行库 DLL）----
    # 留空则自动探测常见位置（含数据目录下的 toolchain/，安装脚本会把 CUDA 头文件放这里）；
    # 指向包含 include/ 与 lib/ 的目录。
    tensorrt_root: Path | None = None
    cuda_root: Path | None = None
    # 可选能力：实时推理（摄像头/视频/预览窗口）需要 OpenCV 开发文件；
    # 留空则自动探测工具链搜索根（含 conda 的 <prefix>/Library 布局）。
    opencv_root: Path | None = None
    toolchain_search_roots: list[Path] = [
        data_root().parent / "toolchain",  # 安装脚本自动下载的 CUDA 文件放在这里
        Path("D:/qforge-toolchain"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
    ]

    # ---- 容器产物（SPEC 12）----
    # 基础镜像 tag 属版本敏感项：该 tag 内的 TensorRT/CUDA 版本已在 docs/versions.md 记录
    # （26.03-py3 对应 TensorRT 10.16.0.72 / CUDA 13.2，与 Engine 构建环境同一 TRT 小版本线）
    docker_base_image: str = "nvcr.io/nvidia/tensorrt:26.03-py3"
    # 镜像构建属 SPEC 2.1 的「可选生成镜像」：默认关闭，任务配置 build.docker_build 打开
    docker_enabled: bool = False
    docker_build_timeout_seconds: int = 3600
    docker_run_timeout_seconds: int = 300

    @property
    def resolved_storage_root(self) -> Path:
        return self.storage_root.resolve()

    @property
    def resolved_frontend_dir(self) -> Path | None:
        """已构建的前端目录：显式配置优先，其次自动探测（见 app/paths.py）。"""
        if self.frontend_dir is not None:
            return self.frontend_dir if self.frontend_dir.is_dir() else None
        return frontend_dir()

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_file = (self.resolved_storage_root / "qforge.db").as_posix()
        return f"sqlite+pysqlite:///{db_file}"


@lru_cache
def get_settings() -> Settings:
    """进程内缓存的配置单例。测试中可用 `get_settings.cache_clear()` 重置。"""
    return Settings()

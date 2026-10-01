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

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config/settings.py -> config -> app -> backend -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """进程级配置。字段默认值面向阶段 0 的本地可运行形态（SQLite + eager Celery）。"""

    model_config = SettingsConfigDict(
        env_prefix="QFORGE_",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "QForge"
    app_version: str = "0.1.0"
    environment: Literal["dev", "test", "prod"] = "dev"
    api_prefix: str = "/api"
    log_level: str = "INFO"

    # ---- 存储（SPEC 14.2）----
    storage_root: Path = REPO_ROOT / "storage"

    # ---- 数据库（SPEC 14.1）----
    # 未显式配置时由 storage_root 推导 SQLite 文件；阶段 1 接 PostgreSQL 只需设置该变量。
    database_url: str | None = None
    database_echo: bool = False
    # 本地开发便捷建表；正式 schema 变更以 Alembic 迁移为准（backend/migrations）
    auto_create_schema: bool = True

    # ---- 任务队列（SPEC 3 / 13）----
    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"
    # 阶段 0 无 Redis 服务，默认 eager（同步执行）；接入真实 broker 时置 false。
    celery_task_always_eager: bool = True
    celery_task_eager_propagates: bool = True

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # ---- 资源与安全限制（SPEC 15）----
    # 阶段 0 只定义配置项与默认值；上传校验、ZIP 限额、并发与超时的强制执行在阶段 1 落地。
    max_upload_mb: int = 2048
    max_zip_uncompressed_mb: int = 8192
    max_zip_entries: int = 20000
    max_zip_depth: int = 8
    task_timeout_seconds: int = 3600
    max_concurrent_tasks: int = 1

    @property
    def resolved_storage_root(self) -> Path:
        return self.storage_root.resolve()

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

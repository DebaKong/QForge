"""Alembic 迁移环境。

要点：
- 数据库 URL 优先级：迁移调用方显式设置 > 应用配置（QFORGE_DATABASE_URL，默认 SQLite），
  保证迁移与运行期始终指向同一个库；
- 导入 `app.models` 以注册全部实体，autogenerate 才能看到完整元数据；
- SQLite 下启用 render_as_batch，使后续 ALTER 类变更可用（阶段 1 切 PostgreSQL 后无影响）。

位置说明：本目录位于**包内**（`app/migrations`），因此既能在仓库检出里用
`alembic -c backend/alembic.ini` 运行，也能在 pip 安装后用 `qforge init` 运行。
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# 仓库检出时把 backend/ 加入 sys.path（`app` 包所在）；安装形态下 `app` 已可直接导入
BACKEND_DIR = Path(__file__).resolve().parents[2]
if (BACKEND_DIR / "app").is_dir() and str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config.settings import get_settings  # noqa: E402
from app.db.base import Base  # noqa: E402
import app.models  # noqa: E402,F401  导入以注册全部实体

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """显式配置优先，其次应用配置（默认 SQLite 文件）。"""
    configured = (config.get_main_option("sqlalchemy.url") or "").strip()
    return configured or get_settings().resolved_database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

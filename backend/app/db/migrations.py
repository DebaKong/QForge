"""数据库迁移的应用入口（安装形态的关键一环）。

为什么需要它：安装形态下用户拿不到仓库的 `alembic.ini`，但**跨版本升级必须能改表结构**。
因此把迁移脚本放进包内（`app/migrations`），这里用代码构造 Alembic 配置并调用它，
不依赖仓库里的 ini 文件。

三种情况都要正确处理（顺序即优先级）：

1. 库里已有我们的表，但没有 `alembic_version` —— 老版本用 `create_all` 建的库
   → `stamp head`（只登记版本，不改结构，避免「表已存在」报错）；
2. `alembic_version` 落后于 head → `upgrade head`；
3. 已经是最新 → 什么也不做。

**不吞异常**：迁移失败会把异常抛给调用方，由启动流程/CLI 明确报错（AGENTS.md：不伪造成功）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import inspect

from app.db.base import get_engine

logger = logging.getLogger(__name__)

VERSION_TABLE = "alembic_version"


def migrations_dir() -> Path:
    """包内迁移脚本目录：<site-packages 或 backend>/app/migrations。"""
    return Path(__file__).resolve().parent.parent / "migrations"


@dataclass(frozen=True)
class MigrationResult:
    """迁移结果：`action` 便于 CLI/日志如实汇报。"""

    action: str  # upgraded | stamped | already-current | created
    revision: str | None
    url: str

    def describe(self) -> str:
        if self.action == "upgraded":
            return f"已应用迁移到最新版本（{self.revision}）"
        if self.action == "stamped":
            return f"已有表结构但未登记迁移版本，已登记为 {self.revision}（未改动表结构）"
        if self.action == "created":
            return f"已按模型创建表结构并登记为 {self.revision}"
        return f"表结构已是最新（{self.revision}）"


def _alembic_config(url: str):
    """构造 Alembic 配置。注意 path_separator=os：Windows 路径必须用它，否则会被当成多路径分隔。"""
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(migrations_dir()))
    config.set_main_option("sqlalchemy.url", url)
    config.set_main_option("path_separator", "os")
    return config


def head_revision() -> str | None:
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_alembic_config(""))
    return script.get_current_head()


def current_revision(url: str) -> str | None:
    """读取库里登记的迁移版本（没有版本表则返回 None）。"""
    from sqlalchemy import create_engine
    from sqlalchemy import text

    engine = create_engine(url, future=True)
    try:
        with engine.connect() as connection:
            names = inspect(connection).get_table_names()
            if VERSION_TABLE not in names:
                return None
            row = connection.execute(text(f"select version_num from {VERSION_TABLE}")).first()
            return row[0] if row else None
    finally:
        engine.dispose()


def _has_application_tables(url: str) -> bool:
    """库里是否已有业务表（用于识别「create_all 建的老库」）。"""
    from sqlalchemy import create_engine

    engine = create_engine(url, future=True)
    try:
        with engine.connect() as connection:
            names = set(inspect(connection).get_table_names())
    finally:
        engine.dispose()
    return bool(names - {VERSION_TABLE})


def apply_migrations(url: str | None = None, *, allow_create_all_fallback: bool = True) -> MigrationResult:
    """把数据库结构调整到最新版本（幂等）。

    只有「Alembic 不可用」才会退化为按模型建表；**迁移本身失败会直接上抛**，
    绝不把失败伪装成成功（AGENTS.md 约束）。
    """
    target_url = url or str(get_engine().url)

    try:
        from alembic import command
        from alembic.script import ScriptDirectory
    except ImportError as exc:  # pragma: no cover - 正常安装不会走到
        if not allow_create_all_fallback:
            raise
        logger.warning("未安装 Alembic（%s），退化为按模型建表", exc)
        from app.db.base import create_all

        create_all()
        return MigrationResult("created", None, target_url)

    head = ScriptDirectory.from_config(_alembic_config(target_url)).get_current_head()
    current = current_revision(target_url)

    if head is not None and current == head:
        return MigrationResult("already-current", current, target_url)

    config = _alembic_config(target_url)
    if current is None and _has_application_tables(target_url):
        # 老库：表已存在（历史上由 create_all 建立），只登记版本，避免重复建表报错
        command.stamp(config, "head")
        logger.info("已有表结构但无迁移版本，已登记为 head：%s", target_url)
        return MigrationResult("stamped", head, target_url)

    command.upgrade(config, "head")
    logger.info("数据库迁移已应用：%s → %s", target_url, head)
    return MigrationResult("upgraded", head, target_url)

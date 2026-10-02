"""数据库迁移入口的用例（安装形态能否正确升级表结构）。

背景：安装形态下用户拿不到仓库里的 `alembic.ini`，因此迁移脚本随包分发，
由 `app.db.migrations` 用代码构造 Alembic 配置来应用。这里覆盖三种真实情况：
全新库、已是最新、以及历史上由 `create_all` 建立的老库。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.cli import main
from app.db import base as db_base
from app.db.migrations import apply_migrations, current_revision, head_revision, migrations_dir

EXPECTED_TABLES = {"projects", "models", "datasets", "tasks", "artifacts", "job_logs"}


def _url(tmp_path: Path) -> str:
    return f"sqlite+pysqlite:///{(tmp_path / 'mig.db').as_posix()}"


def _tables(database: Path) -> set[str]:
    return {
        row[0]
        for row in sqlite3.connect(str(database)).execute(
            "select name from sqlite_master where type='table'"
        )
    }


def test_alembic_ini_stays_ascii() -> None:
    """`backend/alembic.ini` 必须保持 ASCII。

    实测：Alembic/configparser 用**系统 locale**（本机 GBK）读这个文件，
    写入中文会直接抛 UnicodeDecodeError —— 开发路径的 `alembic -c ... upgrade head` 会失败。
    安装形态走代码构造配置，不受影响；但两条路径都必须可用。
    """
    ini = Path(__file__).resolve().parents[1] / "backend" / "alembic.ini"
    raw = ini.read_bytes()
    non_ascii = [index for index, byte in enumerate(raw) if byte > 127]
    assert not non_ascii, f"alembic.ini 含非 ASCII 字节（首个位于 {non_ascii[:1]}），会破坏迁移命令"


def test_migrations_are_packaged_with_the_app() -> None:
    """迁移脚本必须在包内，否则安装后无法升级表结构。"""
    directory = migrations_dir()
    assert (directory / "env.py").is_file(), f"缺少包内迁移环境：{directory}"
    assert list((directory / "versions").glob("*.py")), "至少应有一个迁移版本"
    assert head_revision(), "必须能解析出 head 版本"


def test_apply_migrations_creates_schema_on_fresh_database(tmp_path: Path) -> None:
    url = _url(tmp_path)
    result = apply_migrations(url)

    assert result.action == "upgraded"
    assert result.revision == head_revision()
    tables = _tables(tmp_path / "mig.db")
    assert EXPECTED_TABLES <= tables
    assert "alembic_version" in tables
    assert current_revision(url) == head_revision()


def test_apply_migrations_is_idempotent(tmp_path: Path) -> None:
    url = _url(tmp_path)
    apply_migrations(url)
    assert apply_migrations(url).action == "already-current"


def test_apply_migrations_stamps_legacy_create_all_database(tmp_path: Path) -> None:
    """老库（表已存在但没有迁移版本）必须被登记版本，而不是重复建表报错。"""
    url = _url(tmp_path)
    db_base.reset_engine()
    try:
        db_base.configure_engine(url)
        db_base.create_all()
        assert current_revision(url) is None, "create_all 不应写迁移版本"

        first = apply_migrations(url)
        assert first.action == "stamped", first
        assert EXPECTED_TABLES <= _tables(tmp_path / "mig.db")

        second = apply_migrations(url)
        assert second.action == "already-current"
    finally:
        db_base.reset_engine()


def test_cli_init_applies_migrations(storage_root: Path, capsys) -> None:
    db_base.reset_engine()
    try:
        assert main(["init"]) == 0
    finally:
        db_base.reset_engine()

    output = capsys.readouterr().out
    assert "表结构" in output
    tables = _tables(storage_root / "qforge.db")
    assert EXPECTED_TABLES <= tables
    assert "alembic_version" in tables, "qforge init 必须登记迁移版本，便于后续升级"

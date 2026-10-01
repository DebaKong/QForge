"""SQLAlchemy 2.0 引擎与会话管理。

阶段 0 使用 SQLite（本地可运行）；通过 `QFORGE_DATABASE_URL` 可在阶段 1 切换到
PostgreSQL，模型层与 API 层不需要改动（SPEC 14.1）。

会话边界：
- FastAPI 通过 `get_db` 依赖注入，一个请求一个会话；
- Celery 任务通过 `session_scope()` 获取会话，避免长事务跨任务边界。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config.settings import get_settings


class Base(DeclarativeBase):
    """所有 ORM 实体的基类。"""


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """SQLite 默认不启用外键约束，显式打开以保证测试能发现引用错误。"""

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection: object, _record: object) -> None:  # pragma: no cover
        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()


def configure_engine(url: str | None = None, *, echo: bool | None = None) -> Engine:
    """创建（或重建）全局引擎。测试中用它指向内存库或临时文件库。"""
    global _engine, _session_factory

    settings = get_settings()
    target_url = url or settings.resolved_database_url
    target_echo = settings.database_echo if echo is None else echo

    if _engine is not None:
        _engine.dispose()

    connect_args: dict[str, object] = {}
    if target_url.startswith("sqlite"):
        # API 请求在线程池中执行，SQLite 连接需允许跨线程复用
        connect_args["check_same_thread"] = False

    _engine = create_engine(target_url, echo=target_echo, future=True, connect_args=connect_args)
    if target_url.startswith("sqlite"):
        _enable_sqlite_foreign_keys(_engine)

    _session_factory = sessionmaker(
        bind=_engine, autoflush=False, expire_on_commit=False, class_=Session
    )
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        configure_engine()
    assert _engine is not None
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        configure_engine()
    assert _session_factory is not None
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务化会话上下文：正常退出提交，异常回滚。供 Celery 任务与脚本使用。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：每请求一个会话。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all() -> None:
    """按当前元数据建表。

    阶段 0 用它在测试与本地初始化中建表；对外 schema 变更以 Alembic 迁移为准。
    """
    import app.models  # noqa: F401  导入以注册全部实体

    Base.metadata.create_all(bind=get_engine())


def drop_all() -> None:
    import app.models  # noqa: F401

    Base.metadata.drop_all(bind=get_engine())


def reset_engine() -> None:
    """释放全局引擎（测试隔离用）。"""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None

"""测试夹具。

隔离策略：
- 每个测试用独立临时目录作为 storage_root，并通过 QFORGE_STORAGE_ROOT 注入；
- 每个测试用独立的临时 SQLite 文件库（而非 :memory:），以便暴露真实的连接/锁行为；
- **测试结果不随开发者本机 `.env` 变化**：这里显式覆盖与队列/执行方式/容器相关的开关
  （环境变量优先级高于 .env，见 pydantic-settings 的取值顺序）；
- 测试结束后释放全局引擎，避免用例间互相污染。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

# ---- 必须在导入 app / workers 之前设置：这些模块在导入期就会读取配置 ----
os.environ["QFORGE_ENVIRONMENT"] = "test"
# 队列与执行方式固定为「进程内同步」，避免依赖本机是否装了 Redis
os.environ["QFORGE_EXECUTOR_MODE"] = "local"
os.environ["QFORGE_CELERY_TASK_ALWAYS_EAGER"] = "true"
# 容器构建默认关闭（避免默认测试去拉镜像/构建镜像）
os.environ["QFORGE_DOCKER_ENABLED"] = "false"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for _path in (str(BACKEND_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.config.settings import get_settings  # noqa: E402
from app.db import base as db_base  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_task_executor() -> Iterator[None]:
    """每个用例前后重置进程内任务执行器，避免后台线程跨用例串库。"""
    from app.services.executor import reset_executor

    reset_executor()
    yield
    reset_executor()


@pytest.fixture()
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "storage"
    monkeypatch.setenv("QFORGE_STORAGE_ROOT", str(root))
    monkeypatch.setenv("QFORGE_ENVIRONMENT", "test")
    get_settings.cache_clear()
    return root


@pytest.fixture()
def engine(storage_root: Path, tmp_path: Path) -> Iterator[Engine]:
    db_path = (tmp_path / "qforge-test.db").as_posix()
    configured = db_base.configure_engine(f"sqlite+pysqlite:///{db_path}")
    yield configured
    db_base.reset_engine()


@pytest.fixture()
def session_factory(engine: Engine) -> Iterator[sessionmaker[Session]]:
    db_base.create_all()
    yield db_base.get_session_factory()
    db_base.drop_all()


@pytest.fixture()
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def client(engine: Engine, session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def lenient_client(
    engine: Engine, session_factory: sessionmaker[Session]
) -> Iterator[TestClient]:
    """不向上抛出服务端异常，用于验证 500 错误信封结构。"""
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

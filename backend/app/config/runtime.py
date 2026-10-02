"""运行形态判定：把「自动」执行方式解析成确定值。

默认 `QFORGE_EXECUTOR_MODE=auto`：
- 能连上 Redis → 用 Celery（真正的独立 worker 进程）；
- 连不上 → 用进程内后台执行器。

这样**新机器上不装 Redis 也能直接跑**（少一个必装依赖），装了 Redis 则自动升级为队列模式，
符合 AGENTS.md 的「安装即用优先」。
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from app.config.settings import Settings, get_settings


def redis_reachable(url: str, timeout: float = 0.6) -> bool:
    """只做 TCP 可达性探测（不发 Redis 命令，避免强依赖 redis 客户端）。"""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"redis", "rediss"}:
        return False
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def effective_executor_mode(settings: Settings | None = None) -> str:
    """返回实际生效的执行方式：`local` 或 `celery`。"""
    settings = settings or get_settings()
    if settings.executor_mode in {"local", "celery"}:
        return settings.executor_mode
    return "celery" if redis_reachable(settings.celery_broker_url) else "local"

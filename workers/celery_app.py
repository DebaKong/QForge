"""Celery 应用（SPEC 3 / 13）。

阶段 0 默认 `task_always_eager=True`：本机没有 Redis，任务在 API 进程内同步执行，
用于验证「创建任务 -> 入队 -> Worker 处理 -> 状态与日志」这条链路。
接入真实 Redis 时设置：

    QFORGE_CELERY_TASK_ALWAYS_EAGER=false
    QFORGE_CELERY_BROKER_URL=redis://127.0.0.1:6379/0
    QFORGE_CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/1

并启动：celery -A workers.celery_app:celery_app worker --loglevel=INFO --pool=solo
（Windows 下需使用 solo/threads 池；prefork 不受支持。）
"""

from __future__ import annotations

from workers._bootstrap import ensure_backend_on_path

ensure_backend_on_path()

from app.config.settings import get_settings  # noqa: E402
from celery import Celery  # noqa: E402

TASK_MODULES = ["workers.tasks"]


def create_celery_app() -> Celery:
    settings = get_settings()

    app = Celery(
        "qforge",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=TASK_MODULES,
    )

    app.conf.update(
        task_always_eager=settings.celery_task_always_eager,
        task_eager_propagates=settings.celery_task_eager_propagates,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        enable_utc=True,
        timezone="UTC",
        broker_connection_retry_on_startup=True,
        # SPEC 15：任务超时上限，防止长任务无限占用
        task_time_limit=settings.task_timeout_seconds,
        task_soft_time_limit=max(settings.task_timeout_seconds - 30, 60),
        result_expires=86400,
    )
    return app


celery_app = create_celery_app()

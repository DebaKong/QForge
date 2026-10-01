"""系统健康检查。

只报告数据库方言与存储根目录，不回显可能含口令的连接串（SPEC 13.2 敏感信息）。
"""

from __future__ import annotations

import platform

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import SessionDep
from app.config.settings import get_settings
from app.schemas.common import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
def health(session: SessionDep) -> HealthResponse:
    settings = get_settings()

    database_state = "ok"
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - 仅在数据库不可用时触发
        database_state = f"error:{type(exc).__name__}"

    dialect = session.get_bind().dialect.name

    return HealthResponse(
        status="ok" if database_state == "ok" else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        python_version=platform.python_version(),
        database=f"{dialect}:{database_state}",
        storage_root=str(settings.resolved_storage_root),
        celery_eager=settings.celery_task_always_eager,
        phase="phase-0",
    )

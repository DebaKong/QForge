"""FastAPI 应用入口（SPEC 3 / 16）。

分层边界：本进程只处理 API 与元数据，不执行量化、编译、Docker 构建等长任务；
长任务一律经 Celery 投递给 Worker。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import build_api_router
from app.config.settings import get_settings
from app.db.base import create_all
from app.errors import DomainError, ErrorCode
from app.logging_config import configure_logging
from app.web_static import register_frontend

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.resolved_storage_root.mkdir(parents=True, exist_ok=True)

    if settings.auto_create_schema:
        create_all()
        logger.info(
            "已确保表结构存在（auto_create_schema；正式迁移以 Alembic 为准）",
            extra={"storage_root": str(settings.resolved_storage_root)},
        )
    yield


def register_exception_handlers(app: FastAPI) -> None:
    """统一错误出口：所有错误都带 ErrorCode，便于前端与日志定位（SPEC 13.1 / 15.1）。"""

    @app.exception_handler(DomainError)
    async def _handle_domain_error(_: Request, exc: DomainError) -> JSONResponse:
        log = logger.error if exc.http_status >= 500 else logger.info
        log("领域错误 %s: %s", exc.code.value, exc.message, extra={"error_code": exc.code.value})
        return JSONResponse(status_code=exc.http_status, content=jsonable_encoder(exc.to_payload()))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error_code": ErrorCode.VALIDATION_ERROR.value,
                "message": "请求参数校验失败",
                "detail": {"errors": jsonable_encoder(exc.errors())},
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常", extra={"error_code": ErrorCode.INTERNAL_ERROR.value})
        return JSONResponse(
            status_code=500,
            content={
                "error_code": ErrorCode.INTERNAL_ERROR.value,
                "message": "服务内部错误",
                "detail": {"type": type(exc).__name__},
            },
        )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.environment == "prod")

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.app_version,
        description="ONNX 自动化量化部署平台 API（SPEC.md V1.0；当前实现阶段：阶段 1）",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(build_api_router(), prefix=settings.api_prefix)
    register_exception_handlers(app)
    # 安装即用：已构建的前端由 API 直接托管（运行期不需要 Node / Vite）
    register_frontend(app, settings.resolved_frontend_dir, api_prefix=settings.api_prefix)
    return app


app = create_app()

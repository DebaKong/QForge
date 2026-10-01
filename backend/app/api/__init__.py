"""API 路由装配（SPEC 16.1 最小 API 集合）。"""

from fastapi import APIRouter

from app.api.routes import datasets, health, models, projects, tasks, uploads


def build_api_router() -> APIRouter:
    router = APIRouter()
    router.include_router(health.router)
    router.include_router(projects.router)
    router.include_router(models.router)
    router.include_router(datasets.router)
    router.include_router(tasks.router)
    router.include_router(uploads.router)
    return router

"""Project API（SPEC 16.1）。"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import SessionDep
from app.config.settings import get_settings
from app.schemas.project import (
    ProjectBatchDeleteRequest,
    ProjectBatchDeleteResult,
    ProjectCreate,
    ProjectDeletionPreviewRead,
    ProjectDeletionResult,
    ProjectRead,
)
from app.services import catalog, project_cleanup

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED, summary="创建项目")
def create_project(payload: ProjectCreate, session: SessionDep) -> ProjectRead:
    project = catalog.create_project(session, payload)
    return ProjectRead.model_validate(project)


@router.get("", response_model=list[ProjectRead], summary="项目列表")
def list_projects(session: SessionDep) -> list[ProjectRead]:
    return [ProjectRead.model_validate(item) for item in catalog.list_projects(session)]


@router.post(
    "/delete",
    response_model=ProjectBatchDeleteResult,
    summary="批量删除项目",
    description="逐个删除：某个项目删不掉（例如还有运行中任务）不影响其余项目。",
)
def delete_projects(payload: ProjectBatchDeleteRequest, session: SessionDep) -> ProjectBatchDeleteResult:
    result = project_cleanup.delete_projects(
        session, get_settings().resolved_storage_root, payload.project_ids
    )
    return ProjectBatchDeleteResult.model_validate(result)


@router.get(
    "/{project_id}/deletion-preview",
    response_model=ProjectDeletionPreviewRead,
    summary="删除前的影响面（确认框数据）",
)
def preview_project_deletion(project_id: str, session: SessionDep) -> ProjectDeletionPreviewRead:
    result = project_cleanup.preview(session, get_settings().resolved_storage_root, project_id)
    return ProjectDeletionPreviewRead.model_validate(result.to_dict())


@router.delete(
    "/{project_id}",
    response_model=ProjectDeletionResult,
    summary="删除项目",
    description=(
        "级联删除该项目下的模型、数据集、任务及其日志与产物，并删除磁盘目录。"
        "存在排队/运行中任务时返回 409，需先取消任务。"
    ),
)
def delete_project(project_id: str, session: SessionDep) -> ProjectDeletionResult:
    result = project_cleanup.delete_project(
        session, get_settings().resolved_storage_root, project_id
    )
    return ProjectDeletionResult.model_validate(result)


@router.get("/{project_id}", response_model=ProjectRead, summary="项目详情")
def get_project(project_id: str, session: SessionDep) -> ProjectRead:
    return ProjectRead.model_validate(catalog.get_project(session, project_id))

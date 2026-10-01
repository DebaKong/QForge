"""Project API（SPEC 16.1）。"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import SessionDep
from app.schemas.project import ProjectCreate, ProjectRead
from app.services import catalog

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED, summary="创建项目")
def create_project(payload: ProjectCreate, session: SessionDep) -> ProjectRead:
    project = catalog.create_project(session, payload)
    return ProjectRead.model_validate(project)


@router.get("", response_model=list[ProjectRead], summary="项目列表")
def list_projects(session: SessionDep) -> list[ProjectRead]:
    return [ProjectRead.model_validate(item) for item in catalog.list_projects(session)]


@router.get("/{project_id}", response_model=ProjectRead, summary="项目详情")
def get_project(project_id: str, session: SessionDep) -> ProjectRead:
    return ProjectRead.model_validate(catalog.get_project(session, project_id))

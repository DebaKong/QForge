"""Project / Model / Dataset 元数据读写（SPEC 5.2）。

边界说明：阶段 0 的这些接口只登记元数据，不涉及文件上传与解压；
因此这里不从用户提供的名称派生任何文件系统路径（存储路径一律由 task_id 生成）。
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.errors import ConflictError, NotFoundError
from app.models.dataset import Dataset
from app.models.onnx_model import OnnxModel
from app.models.project import Project
from app.schemas.dataset import DatasetCreate
from app.schemas.model import ModelCreate
from app.schemas.project import ProjectCreate


# ---------------- Project ----------------


def create_project(session: Session, data: ProjectCreate) -> Project:
    existing = session.scalar(select(Project).where(Project.name == data.name))
    if existing is not None:
        raise ConflictError(
            f"项目名已存在：{data.name}", detail={"name": data.name, "project_id": existing.id}
        )
    project = Project(name=data.name, description=data.description)
    session.add(project)
    session.flush()
    return project


def get_project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise NotFoundError("项目不存在", detail={"project_id": project_id})
    return project


def list_projects(session: Session) -> Sequence[Project]:
    return session.scalars(select(Project).order_by(Project.created_at.desc())).all()


# ---------------- OnnxModel ----------------


def create_model(session: Session, data: ModelCreate) -> OnnxModel:
    get_project(session, data.project_id)

    existing = session.scalar(
        select(OnnxModel).where(
            OnnxModel.project_id == data.project_id, OnnxModel.name == data.name
        )
    )
    if existing is not None:
        raise ConflictError(
            f"项目内模型名已存在：{data.name}",
            detail={"name": data.name, "model_id": existing.id},
        )

    payload = data.model_dump(exclude={"project_id", "name"})
    onnx_model = OnnxModel(project_id=data.project_id, name=data.name, **payload)
    session.add(onnx_model)
    session.flush()
    return onnx_model


def get_model(session: Session, model_id: str) -> OnnxModel:
    onnx_model = session.get(OnnxModel, model_id)
    if onnx_model is None:
        raise NotFoundError("模型不存在", detail={"model_id": model_id})
    return onnx_model


def list_models(session: Session, project_id: str | None = None) -> Sequence[OnnxModel]:
    stmt = select(OnnxModel).order_by(OnnxModel.created_at.desc())
    if project_id:
        stmt = stmt.where(OnnxModel.project_id == project_id)
    return session.scalars(stmt).all()


# ---------------- Dataset ----------------


def create_dataset(session: Session, data: DatasetCreate) -> Dataset:
    get_project(session, data.project_id)

    existing = session.scalar(
        select(Dataset).where(Dataset.project_id == data.project_id, Dataset.name == data.name)
    )
    if existing is not None:
        raise ConflictError(
            f"项目内数据集名已存在：{data.name}",
            detail={"name": data.name, "dataset_id": existing.id},
        )

    dataset = Dataset(
        project_id=data.project_id,
        name=data.name,
        kind=data.kind,
        root_path=data.root_path,
    )
    session.add(dataset)
    session.flush()
    return dataset


def get_dataset(session: Session, dataset_id: str) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError("数据集不存在", detail={"dataset_id": dataset_id})
    return dataset


def list_datasets(session: Session, project_id: str | None = None) -> Sequence[Dataset]:
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    if project_id:
        stmt = stmt.where(Dataset.project_id == project_id)
    return session.scalars(stmt).all()

"""Dataset API（SPEC 16.1）。

阶段 0 只登记元数据；calibration.zip 上传、解压安全与图像校验属阶段 1。
"""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from app.api.deps import SessionDep
from app.schemas.dataset import DatasetCreate, DatasetRead
from app.services import catalog

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetRead, status_code=status.HTTP_201_CREATED, summary="登记数据集元数据")
def create_dataset(payload: DatasetCreate, session: SessionDep) -> DatasetRead:
    dataset = catalog.create_dataset(session, payload)
    return DatasetRead.model_validate(dataset)


@router.get("", response_model=list[DatasetRead], summary="数据集列表")
def list_datasets(
    session: SessionDep, project_id: str | None = Query(default=None)
) -> list[DatasetRead]:
    return [
        DatasetRead.model_validate(item) for item in catalog.list_datasets(session, project_id)
    ]


@router.get("/{dataset_id}", response_model=DatasetRead, summary="数据集详情")
def get_dataset(dataset_id: str, session: SessionDep) -> DatasetRead:
    return DatasetRead.model_validate(catalog.get_dataset(session, dataset_id))

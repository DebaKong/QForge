"""OnnxModel API（SPEC 16.1）。

阶段 0 只登记元数据；文件上传与 ONNX 解析属阶段 1。
"""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from app.api.deps import SessionDep
from app.schemas.model import ModelCreate, ModelRead
from app.services import catalog

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=ModelRead, status_code=status.HTTP_201_CREATED, summary="登记模型元数据")
def create_model(payload: ModelCreate, session: SessionDep) -> ModelRead:
    onnx_model = catalog.create_model(session, payload)
    return ModelRead.model_validate(onnx_model)


@router.get("", response_model=list[ModelRead], summary="模型列表")
def list_models(
    session: SessionDep, project_id: str | None = Query(default=None)
) -> list[ModelRead]:
    return [
        ModelRead.model_validate(item) for item in catalog.list_models(session, project_id)
    ]


@router.get("/{model_id}", response_model=ModelRead, summary="模型详情")
def get_model(model_id: str, session: SessionDep) -> ModelRead:
    return ModelRead.model_validate(catalog.get_model(session, model_id))

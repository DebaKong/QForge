"""Dataset API（SPEC 16.1）。

阶段 0 只登记元数据；上传、解压安全与图像校验见 `uploads.py`；
本文件另外提供**数据集管理**：统计清单、抽样预览、重新校验（SPEC 8.3）、批量删除。
"""

from __future__ import annotations

import logging
import mimetypes

from fastapi import APIRouter, Query, status
from fastapi.responses import FileResponse

from app.api.deps import SessionDep
from app.config.settings import get_settings
from app.schemas.dataset import DatasetCreate, DatasetRead
from app.services import catalog, dataset_admin

logger = logging.getLogger("qforge.api")

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


@router.post(
    "/delete",
    summary="批量删除数据集",
    description="逐个删除；正在被排队/运行中任务使用的数据集会被拒绝（409），不影响其余项目。",
)
def delete_datasets(payload: dict, session: SessionDep) -> dict:
    ids = [str(item) for item in (payload.get("dataset_ids") or [])]
    if not ids:
        return {"deleted": [], "failed": [], "deleted_count": 0, "failed_count": 0, "freed_bytes": 0}
    return dataset_admin.delete_datasets(session, get_settings().resolved_storage_root, ids[:200])


@router.get("/{dataset_id}", response_model=DatasetRead, summary="数据集详情")
def get_dataset(dataset_id: str, session: SessionDep) -> DatasetRead:
    return DatasetRead.model_validate(catalog.get_dataset(session, dataset_id))


@router.get(
    "/{dataset_id}/inventory",
    summary="数据集图像清单与统计（SPEC 8.3）",
    description=(
        "返回分页图像清单（尺寸/通道/格式/可读性）、分辨率与通道分布、异常图片列表。"
        "统计默认只扫描前 N 张（大校准集不能把请求拖死），返回里会说明实际扫描数量。"
    ),
)
def get_dataset_inventory(
    dataset_id: str,
    session: SessionDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=dataset_admin.MAX_PAGE_SIZE),
    scan_limit: int = Query(default=dataset_admin.DEFAULT_SCAN_LIMIT, ge=1, le=5000),
) -> dict:
    settings = get_settings()
    dataset = dataset_admin.get_dataset(session, dataset_id)
    return dataset_admin.inventory(
        settings.resolved_storage_root,
        dataset,
        allowed_extensions=settings.allowed_image_extensions,
        offset=offset,
        limit=limit,
        scan_limit=scan_limit,
    )


@router.get(
    "/{dataset_id}/samples",
    summary="读取数据集内的一张样本图（用于界面预览）",
    description="路径受约束：只允许读取该数据集 images/ 目录内的文件（防路径穿越）。",
)
def get_dataset_sample(
    dataset_id: str,
    session: SessionDep,
    path: str = Query(..., description="相对该数据集 images/ 的路径，来自清单接口"),
) -> FileResponse:
    settings = get_settings()
    dataset = dataset_admin.get_dataset(session, dataset_id)
    sample = dataset_admin.resolve_sample(settings.resolved_storage_root, dataset, path)
    media_type = mimetypes.guess_type(sample.name)[0] or "application/octet-stream"
    return FileResponse(sample, media_type=media_type, filename=sample.name)


@router.post(
    "/{dataset_id}/revalidate",
    summary="重新执行数据集校验（SPEC 8.3）",
    description=(
        "重新统计数量/分辨率/通道/格式/异常图片；传入 model_id 时额外做"
        "“抽样预处理检查”（tensor shape/dtype/range），结果写回数据集 meta.validation。"
    ),
)
def revalidate_dataset(
    dataset_id: str,
    session: SessionDep,
    model_id: str | None = Query(default=None),
    sample_size: int = Query(default=8, ge=1, le=64),
) -> dict:
    settings = get_settings()
    dataset = dataset_admin.get_dataset(session, dataset_id)
    return dataset_admin.revalidate(
        session,
        settings.resolved_storage_root,
        dataset,
        allowed_extensions=settings.allowed_image_extensions,
        max_images=settings.max_calibration_images,
        model_id=model_id,
        sample_size=sample_size,
    )

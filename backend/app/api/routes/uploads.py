"""上传与下载 API（SPEC 15 上传安全 / 16.1 产物下载）。

上传一律视为不可信：大小、扩展名、ZIP 条目数/深度/解压后总大小/压缩比全部强制校验，
解压逐条校验并逐块写入，绝不使用 `extractall`。

采用**同步**路由：文件读写与 ONNX 解析都是阻塞操作，FastAPI 会把同步路由放进线程池，
避免阻塞事件循环（异步路由里做这些事反而会拖住整个服务）。
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import FileResponse

from app.adapters.definition import ModelDefinition
from app.adapters.registry import get_model_adapter
from app.api.deps import SessionDep
from app.config.settings import get_settings
from app.errors import NotFoundError, UploadInvalidError, ValidationFailedError
from app.models.artifact import Artifact
from app.models.dataset import Dataset
from app.models.onnx_model import OnnxModel
from app.schemas.dataset import DatasetRead
from app.schemas.model import ModelRead
from app.services import catalog, onnx_inspector, uploads
from app.services.layout import (
    dataset_archive_dir,
    dataset_images_dir,
    dataset_root,
    model_input_dir,
    model_root,
)
from app.services.storage import relative_to_storage, safe_join

logger = logging.getLogger(__name__)

router = APIRouter(tags=["uploads"])


def _max_upload_bytes() -> int:
    return get_settings().max_upload_mb * 1024 * 1024


# ---------------- 模型上传 ----------------


@router.post(
    "/models/upload",
    response_model=ModelRead,
    status_code=status.HTTP_201_CREATED,
    summary="上传 ONNX 模型并执行校验链（SPEC 7.1）",
)
def upload_model(
    session: SessionDep,
    project_id: str = Form(..., description="所属项目 ID"),
    name: str = Form(..., description="模型名（项目内唯一）"),
    architecture: str = Form("yolov8", description="模型架构（决定使用哪个 Model Adapter）"),
    task_type: str = Form("detection"),
    model_definition: str | None = Form(
        None, description="可选：SPEC 6.1 模型定义的 JSON 字符串；不传则按架构默认定义生成"
    ),
    file: UploadFile = File(..., description="ONNX 模型文件"),
) -> ModelRead:
    settings = get_settings()
    storage_root = settings.resolved_storage_root
    catalog.get_project(session, project_id)  # 不存在直接 404
    adapter = get_model_adapter(architecture)

    existing = catalog.list_models(session, project_id)
    if any(item.name == name for item in existing):
        raise ValidationFailedError(
            "项目内模型名已存在", detail={"name": name, "project_id": project_id}
        )

    row = OnnxModel(
        project_id=project_id,
        name=name,
        architecture=architecture,
        task_type=task_type,
    )
    session.add(row)
    session.flush()  # 先拿到 id，目录名只由 id 生成

    try:
        saved = uploads.save_upload(
            file.file,
            file.filename or "model.onnx",
            model_input_dir(storage_root, row.id),
            allowed_extensions=settings.allowed_model_extensions,
            max_bytes=_max_upload_bytes(),
            field_name="ONNX 模型",
            storage_root=storage_root,
        )
        row.file_path = relative_to_storage(storage_root, saved.path)
        row.sha256 = saved.sha256
        row.size_bytes = saved.size_bytes

        inspection = onnx_inspector.inspect(saved.path)
        row.input_spec = inspection.inputs
        row.output_spec = inspection.outputs
        row.opset = inspection.opset

        if model_definition:
            try:
                payload = json.loads(model_definition)
            except json.JSONDecodeError as exc:
                raise UploadInvalidError(
                    "model_definition 不是合法 JSON", detail={"error": str(exc)}
                ) from exc
        else:
            payload = adapter.default_definition(inspection.to_dict())

        definition = ModelDefinition.parse(payload)
        for warning in adapter.validate(definition, inspection.to_dict()):
            logger.warning("模型定义提示（%s）：%s", row.id, warning)
        row.model_definition = definition.to_dict()
        session.flush()
    except Exception:
        uploads.remove_tree(model_root(storage_root, row.id))
        raise

    logger.info(
        "模型上传完成",
        extra={
            "model_id": row.id,
            "size_bytes": row.size_bytes,
            "task_type": task_type,
            "architecture": architecture,
        },
    )
    return ModelRead.model_validate(row)


# ---------------- 校准集上传 ----------------


@router.post(
    "/datasets/upload",
    response_model=DatasetRead,
    status_code=status.HTTP_201_CREATED,
    summary="上传校准集 ZIP 并安全解压（SPEC 8.2 / 15）",
)
def upload_dataset(
    session: SessionDep,
    project_id: str = Form(..., description="所属项目 ID"),
    name: str = Form(..., description="数据集名（项目内唯一）"),
    kind: str = Form("calibration", description="calibration | test"),
    file: UploadFile = File(..., description="calibration.zip（内含 images/ 目录）"),
) -> DatasetRead:
    settings = get_settings()
    storage_root = settings.resolved_storage_root
    catalog.get_project(session, project_id)

    existing = catalog.list_datasets(session, project_id)
    if any(item.name == name for item in existing):
        raise ValidationFailedError(
            "项目内数据集名已存在", detail={"name": name, "project_id": project_id}
        )

    row = Dataset(project_id=project_id, name=name, kind=kind)
    session.add(row)
    session.flush()

    try:
        saved = uploads.save_upload(
            file.file,
            file.filename or "calibration.zip",
            dataset_archive_dir(storage_root, row.id),
            allowed_extensions=[".zip"],
            max_bytes=_max_upload_bytes(),
            field_name="校准集压缩包",
            storage_root=storage_root,
        )
        images_dir = dataset_images_dir(storage_root, row.id)
        report = uploads.extract_image_archive(
            saved.path,
            images_dir,
            storage_root=storage_root,
            allowed_image_extensions=settings.allowed_image_extensions,
            max_entries=settings.max_zip_entries,
            max_total_uncompressed_bytes=settings.max_zip_uncompressed_mb * 1024 * 1024,
            max_depth=settings.max_zip_depth,
            max_images=settings.max_calibration_images,
        )
        row.root_path = relative_to_storage(storage_root, images_dir)
        row.image_count = report.image_count
        row.meta = {
            "archive": saved.filename,
            "archive_sha256": saved.sha256,
            "entries": report.entry_count,
            "images": report.image_count,
            "extracted_bytes": report.total_uncompressed_bytes,
            "skipped": report.skipped[:50],
        }
        session.flush()
    except Exception:
        uploads.remove_tree(dataset_root(storage_root, row.id))
        raise

    logger.info(
        "校准集上传完成",
        extra={"dataset_id": row.id, "images": row.image_count},
    )
    return DatasetRead.model_validate(row)


# ---------------- 产物下载 ----------------


@router.get(
    "/tasks/{task_id}/artifacts/{artifact_id}/download",
    summary="下载任务产物（路径受限在存储根目录内）",
)
def download_artifact(task_id: str, artifact_id: str, session: SessionDep) -> FileResponse:
    artifact = session.get(Artifact, artifact_id)
    if artifact is None or artifact.task_id != task_id:
        raise NotFoundError(
            "产物不存在", detail={"task_id": task_id, "artifact_id": artifact_id}
        )

    settings = get_settings()
    storage_root = settings.resolved_storage_root
    # 用逐段校验的方式还原路径，杜绝 DB 中被写入穿越路径的可能
    segments = [segment for segment in artifact.relative_path.split("/") if segment]
    if not segments:
        raise NotFoundError("产物路径为空", detail={"artifact_id": artifact_id})
    path = safe_join(storage_root, *segments)

    if not path.exists() or not path.is_file():
        raise NotFoundError(
            "产物文件已不在磁盘上（可能已被清理）", detail={"relative_path": artifact.relative_path}
        )

    return FileResponse(
        path,
        filename=path.name,
        media_type="application/octet-stream",
    )

"""数据集管理（SPEC 8.2 / 8.3）：统计、抽样预览、重新校验、批量删除。

设计要点
--------
- **文件访问一律受约束**：目录来自 `layout.dataset_root`，文件来自 `storage.safe_join`，
  解析后必须仍在 `storage_root` 内；单个文件名逐段校验（防路径穿越）。
- **统计有上限**：大校准集不能把请求拖死，扫描默认只看前 N 张，并在返回里说明"基于前 N 张"。
- **删除与项目删除同一策略**：有排队/运行中的任务正在用该数据集时拒绝删除，并说明原因。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.definition import ModelDefinition
from app.errors import ConflictError, NotFoundError, ValidationFailedError
from app.models.dataset import Dataset
from app.models.onnx_model import OnnxModel
from app.models.task import Task
from app.services import calibration, layout, storage, uploads
from app.services.task_state import TERMINAL_STATUSES, TaskStatus

logger = logging.getLogger(__name__)

# 统计扫描上限与预览数量上限（返回里会说明实际扫描了多少张）
DEFAULT_SCAN_LIMIT = 300
DEFAULT_PREVIEW_COUNT = 12
MAX_PAGE_SIZE = 200

_BLOCKING_VALUES = {
    status.value
    for status in TaskStatus
    if status not in TERMINAL_STATUSES and status is not TaskStatus.CREATED
}


def get_dataset(session: Session, dataset_id: str) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError("数据集不存在", detail={"dataset_id": dataset_id})
    return dataset


def images_dir(storage_root: Path, dataset: Dataset) -> Path:
    """数据集的 images 目录（不存在时按空目录处理，便于展示"空数据集"）。"""
    return layout.dataset_root(storage_root, dataset.id) / "images"


def _scan(storage_root: Path, dataset: Dataset, allowed_extensions: list[str], limit: int) -> list[Path]:
    root = images_dir(storage_root, dataset)
    if not root.is_dir():
        return []
    allowed = {item.lower() for item in allowed_extensions}
    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]
    found.sort()
    return found[:limit]


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def inventory(
    storage_root: Path,
    dataset: Dataset,
    *,
    allowed_extensions: list[str],
    offset: int = 0,
    limit: int = 50,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
) -> dict[str, Any]:
    """图像清单 + 统计（分辨率/通道/格式分布、异常图片）。

    返回结构与界面一一对应：`items` 分页给表格，`stats` 给统计卡片，`bad_files` 给异常列表。
    """
    root = images_dir(storage_root, dataset)
    scanned = _scan(storage_root, dataset, allowed_extensions, scan_limit)

    items: list[dict[str, Any]] = []
    bad_files: list[dict[str, str]] = []
    resolution_counter: dict[str, int] = {}
    channel_counter: dict[str, int] = {}
    format_counter: dict[str, int] = {}
    total_bytes = 0

    for path in scanned:
        relative = _relative(root, path)
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0
        total_bytes += size_bytes
        record: dict[str, Any] = {
            "name": path.name,
            "relative_path": relative,
            "size_bytes": size_bytes,
        }
        try:
            with Image.open(path) as image:
                image.load()
                record.update(
                    {
                        "width": int(image.width),
                        "height": int(image.height),
                        "channels": len(image.getbands()),
                        "format": image.format or path.suffix.lstrip(".").upper(),
                        "ok": True,
                    }
                )
                resolution_counter[f"{image.width}x{image.height}"] = (
                    resolution_counter.get(f"{image.width}x{image.height}", 0) + 1
                )
                channel_counter[str(len(image.getbands()))] = (
                    channel_counter.get(str(len(image.getbands())), 0) + 1
                )
                format_counter[record["format"]] = format_counter.get(record["format"], 0) + 1
        except Exception as exc:  # noqa: BLE001 - 坏图登记后继续
            record.update({"ok": False, "reason": f"{type(exc).__name__}: {exc}"})
            bad_files.append({"path": relative, "reason": record["reason"]})
        items.append(record)

    page = items[offset : offset + max(1, min(limit, MAX_PAGE_SIZE))]
    return {
        "dataset_id": dataset.id,
        "image_count": dataset.image_count,
        "scanned": len(scanned),
        "scan_limit": scan_limit,
        "truncated": len(scanned) >= scan_limit,
        "total_bytes": total_bytes,
        "items": page,
        "total_items": len(items),
        "bad_files": bad_files,
        "stats": {
            "resolutions": dict(
                sorted(resolution_counter.items(), key=lambda item: -item[1])
            ),
            "channels": dict(sorted(channel_counter.items(), key=lambda item: -item[1])),
            "formats": dict(sorted(format_counter.items(), key=lambda item: -item[1])),
        },
    }


def resolve_sample(storage_root: Path, dataset: Dataset, relative_path: str) -> Path:
    """把一个相对路径解析成受约束的绝对路径（防路径穿越）。"""
    segments = [segment for segment in relative_path.replace("\\", "/").split("/") if segment]
    if not segments:
        raise ValidationFailedError("样本路径为空", detail={"dataset_id": dataset.id})
    root = layout.dataset_root(storage_root, dataset.id)
    path = storage.safe_join(root, "images", *segments)
    if not path.is_file():
        raise NotFoundError("样本文件不存在", detail={"relative_path": relative_path})
    return path


def _definition_from_model(session: Session, model_id: str | None) -> ModelDefinition | None:
    if not model_id:
        return None
    model = session.get(OnnxModel, model_id)
    if model is None:
        raise NotFoundError("模型不存在", detail={"model_id": model_id})
    payload = model.model_definition
    if not payload:
        raise ValidationFailedError(
            "该模型还没有 Model Definition，无法做预处理抽样检查",
            detail={"model_id": model_id},
        )
    return ModelDefinition.parse(payload)


def revalidate(
    session: Session,
    storage_root: Path,
    dataset: Dataset,
    *,
    allowed_extensions: list[str],
    max_images: int,
    model_id: str | None = None,
    sample_size: int = 8,
) -> dict[str, Any]:
    """重新执行 SPEC 8.3 校验并把结果写回 `dataset.meta["validation"]`。"""
    definition = _definition_from_model(session, model_id)
    root = images_dir(storage_root, dataset)

    if definition is None:
        report = _image_only_report(root, allowed_extensions, max_images)
        report["definition"] = None
        report["warnings"] = [
            *report.get("warnings", []),
            "未提供模型：仅做图像级检查，SPEC 8.3 的“抽样预处理检查”已跳过（可传 model_id 开启）",
        ]
    else:
        calibration_report = calibration.validate_calibration_set(
            root,
            definition,
            allowed_extensions=allowed_extensions,
            max_images=max_images,
            sample_size=sample_size,
        )
        report = calibration_report.to_dict()
        report["definition"] = {
            "model_id": model_id,
            "input_shape": list(definition.input.shape),
        }
        report["root"] = dataset.id  # 不把宿主机绝对路径写进库

    dataset.image_count = int(report.get("image_count") or 0)
    meta = dict(dataset.meta or {})
    meta["validation"] = report
    dataset.meta = meta
    session.flush()
    logger.info(
        "数据集重新校验完成",
        extra={"dataset_id": dataset.id, "images": dataset.image_count},
    )
    return report


def _image_only_report(root: Path, allowed_extensions: list[str], max_images: int) -> dict[str, Any]:
    """没有模型定义时只做图像级检查（数量/分辨率/通道/格式/可读性）。"""
    if not root.is_dir():
        raise ValidationFailedError("数据集目录不存在（可能已被删除）", detail={"root": str(root)})
    try:
        images = calibration.scan_images(root, allowed_extensions, max_images=max_images)
    except Exception as exc:  # noqa: BLE001 - 复用统一的错误语义
        raise ValidationFailedError(f"数据集扫描失败：{exc}") from exc

    resolution_counter: dict[str, int] = {}
    channel_counter: dict[str, int] = {}
    format_counter: dict[str, int] = {}
    bad_files: list[dict[str, str]] = []
    for path in images:
        try:
            with Image.open(path) as image:
                image.load()
                resolution_counter[f"{image.width}x{image.height}"] = (
                    resolution_counter.get(f"{image.width}x{image.height}", 0) + 1
                )
                channel_counter[str(len(image.getbands()))] = (
                    channel_counter.get(str(len(image.getbands())), 0) + 1
                )
                fmt = image.format or path.suffix.lstrip(".").upper()
                format_counter[fmt] = format_counter.get(fmt, 0) + 1
        except Exception as exc:  # noqa: BLE001
            bad_files.append({"path": path.name, "reason": f"{type(exc).__name__}: {exc}"})

    return {
        "root": root.name,
        "image_count": len(images),
        "total_files": len(images),
        "formats": dict(sorted(format_counter.items(), key=lambda item: -item[1])),
        "resolutions": dict(sorted(resolution_counter.items(), key=lambda item: -item[1])),
        "channels": dict(sorted(channel_counter.items(), key=lambda item: -item[1])),
        "bad_files": bad_files[:50],
        "bad_file_count": len(bad_files),
        "sample": {},
        "warnings": (
            [f"存在 {len(bad_files)} 张不可读取的图像，已排除"] if bad_files else []
        ),
    }


def blocking_tasks(session: Session, dataset_ids: list[str]) -> list[dict[str, str]]:
    """正在使用这些数据集的排队/运行中任务（删除前必须为空）。"""
    if not dataset_ids:
        return []
    rows = session.scalars(
        select(Task).where(Task.dataset_id.in_(dataset_ids), Task.status.in_(_BLOCKING_VALUES))
    ).all()
    return [
        {"task_id": task.id, "dataset_id": task.dataset_id or "", "status": task.status.value}
        for task in rows
    ]


def delete_dataset(session: Session, storage_root: Path, dataset_id: str) -> dict[str, Any]:
    """删除单个数据集：先检查占用，再删磁盘、最后删元数据。"""
    dataset = get_dataset(session, dataset_id)
    blockers = blocking_tasks(session, [dataset_id])
    if blockers:
        raise ConflictError(
            "该数据集正被排队或运行中的任务使用，请先取消这些任务再删除",
            detail={"dataset_id": dataset_id, "blocking_tasks": blockers},
        )

    root = layout.dataset_root(storage_root, dataset_id)
    freed_bytes = 0
    removed = False
    if storage.is_within(storage_root, root) and root.exists():
        freed_bytes = sum(
            item.stat().st_size for item in root.rglob("*") if item.is_file()
        )
        uploads.remove_tree(root)
        removed = True

    # 其他任务若引用了该数据集，显式置空，避免留下悬空引用
    session.execute(
        Task.__table__.update().where(Task.dataset_id == dataset_id).values(dataset_id=None)
    )
    session.execute(Dataset.__table__.delete().where(Dataset.id == dataset_id))
    session.flush()

    logger.info("数据集已删除", extra={"dataset_id": dataset_id, "freed_bytes": freed_bytes})
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "removed_directory": removed,
        "freed_bytes": freed_bytes,
    }


def delete_datasets(
    session: Session, storage_root: Path, dataset_ids: list[str]
) -> dict[str, Any]:
    """批量删除：逐个执行并逐个提交，某个失败不影响其余。"""
    deleted: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    seen: set[str] = set()
    for dataset_id in dataset_ids:
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        try:
            deleted.append(delete_dataset(session, storage_root, dataset_id))
            session.commit()
        except (ConflictError, NotFoundError) as error:
            session.rollback()
            failed.append({"dataset_id": dataset_id, "reason": error.message})
            logger.warning("删除数据集失败：%s（%s）", dataset_id, error.message)
    return {
        "deleted": deleted,
        "failed": failed,
        "deleted_count": len(deleted),
        "failed_count": len(failed),
        "freed_bytes": sum(int(item["freed_bytes"]) for item in deleted),
    }

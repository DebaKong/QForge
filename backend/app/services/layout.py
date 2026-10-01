"""存储布局（SPEC 14.2 的任务目录 + 阶段 1 扩展的模型/数据集目录）。

    storage/
    ├── <task_id>/            任务目录（SPEC 14.2 原始定义）
    │   ├── input/            复制/硬链进来的模型与样例输入
    │   ├── calibration/      本任务使用的校准集清单
    │   ├── intermediate/     中间文件（校准缓存、日志）
    │   ├── engine/           model.engine 与 engine_metadata.json
    │   ├── source/           生成的 C++ 工程
    │   ├── docker/           Dockerfile
    │   ├── logs/             流水线与编译日志
    │   └── report/           model_info / compatibility / quantization_report
    ├── models/<model_id>/    上传的原始 ONNX（按模型归档，供多个任务复用）
    └── datasets/<dataset_id>/images/  解压后的校准集

说明：模型与数据集目录是阶段 1 为支撑「上传 → 多个任务复用」而新增的层级，
已在 docs/phase-1.md 记录为对 SPEC 14.2 的扩展；任务目录内仍保留引用副本，
使任务产物自包含（模型用硬链接，跨卷时退化为复制）。
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from app.services.storage import TASK_SUBDIRS, safe_join

logger = logging.getLogger(__name__)

MODELS_DIRNAME = "models"
DATASETS_DIRNAME = "datasets"


# ---------------- 任务目录 ----------------


def task_root(storage_root: Path, task_id: str) -> Path:
    return safe_join(storage_root, task_id)


def task_subdir(storage_root: Path, task_id: str, subdir: str) -> Path:
    if subdir not in TASK_SUBDIRS:
        raise ValueError(f"未知任务子目录：{subdir}")
    return safe_join(storage_root, task_id, subdir)


# ---------------- 模型目录 ----------------


def model_root(storage_root: Path, model_id: str) -> Path:
    return safe_join(storage_root, MODELS_DIRNAME, model_id)


def model_input_dir(storage_root: Path, model_id: str) -> Path:
    directory = model_root(storage_root, model_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------- 数据集目录 ----------------


def dataset_root(storage_root: Path, dataset_id: str) -> Path:
    return safe_join(storage_root, DATASETS_DIRNAME, dataset_id)


def dataset_images_dir(storage_root: Path, dataset_id: str) -> Path:
    directory = dataset_root(storage_root, dataset_id) / "images"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def dataset_archive_dir(storage_root: Path, dataset_id: str) -> Path:
    directory = dataset_root(storage_root, dataset_id) / "archive"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------- 文件安置 ----------------


def place_input_file(source: Path, destination: Path) -> str:
    """把文件放到任务 input 目录：优先硬链接（省空间），失败则复制。

    返回安置方式（"hardlink" / "copy"），用于日志说明。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"

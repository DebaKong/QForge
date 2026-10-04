"""数据集管理（SPEC 8.2/8.3）：统计、预览、重新校验、批量删除、路径防护（纯 CPU）。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.onnx_model import OnnxModel
from app.models.project import Project
from app.models.task import Task
from app.schemas.dataset import DatasetCreate
from app.schemas.project import ProjectCreate
from app.services import catalog, layout
from app.services.task_state import TaskStatus

DEFINITION = {
    "architecture": "yolov8",
    "task": "detection",
    "input": {"name": "images", "shape": [1, 3, 640, 640], "dtype": "float32"},
    "output": {"name": "output0", "shape": [1, 84, 8400], "dtype": "float32"},
    "class_count": 80,
    "preprocess": {"letterbox": True, "color": "RGB", "mean": [0, 0, 0], "std": [255, 255, 255]},
}


def _project(session: Session) -> Project:
    project = catalog.create_project(
        session, ProjectCreate(name=f"ds-{uuid.uuid4().hex[:8]}", description="数据集用例")
    )
    session.commit()
    return project


def _dataset(session: Session, project: Project, *, images: int = 3, bad: bool = False) -> Dataset:
    dataset = catalog.create_dataset(
        session,
        DatasetCreate(
            project_id=project.id, name=f"ds-{uuid.uuid4().hex[:6]}", kind="calibration"
        ),
    )
    session.commit()
    return dataset


def _write_images(root: Path, count: int, *, size: tuple[int, int] = (64, 48), bad: bool = False) -> None:
    images = layout.dataset_images_dir(root.parent.parent, root.parent.name) / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", size, color=(index * 20 % 255, 40, 90))
        image.save(images / f"{index + 1:06d}.png")
    if bad:
        (images / "broken.png").write_bytes(b"not-a-real-png")


def _seed(session: Session, storage_root: Path, *, images: int = 3, bad: bool = False) -> dict:
    project = _project(session)
    dataset = _dataset(session, project)
    image_root = layout.dataset_images_dir(storage_root, dataset.id)
    for index in range(images):
        Image.new("RGB", (64, 48), color=(index * 20 % 255, 40, 90)).save(
            image_root / f"{index + 1:06d}.png"
        )
    if bad:
        (image_root / "broken.png").write_bytes(b"not-a-real-png")
    dataset.image_count = images
    dataset.root_path = layout.dataset_images_dir(storage_root, dataset.id).name
    session.commit()
    return {"project_id": project.id, "project": project, "dataset_id": dataset.id, "dataset": dataset}


def test_inventory_reports_images_stats_and_bad_files(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed(db_session, storage_root, images=3, bad=True)

    response = client.get(f"/api/datasets/{info['dataset_id']}/inventory")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_items"] == 4  # 3 张正常 + 1 张坏图（扩展名仍是图片）
    assert body["stats"]["resolutions"] == {"64x48": 3}
    assert body["stats"]["channels"] == {"3": 3}
    assert body["stats"]["formats"] == {"PNG": 3}
    assert len(body["bad_files"]) == 1 and "broken.png" in body["bad_files"][0]["path"]
    ok_items = [item for item in body["items"] if item["ok"]]
    assert ok_items and ok_items[0]["width"] == 64 and ok_items[0]["height"] == 48


def test_inventory_pagination_and_scan_limit(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed(db_session, storage_root, images=6)

    page = client.get(
        f"/api/datasets/{info['dataset_id']}/inventory", params={"offset": 2, "limit": 2}
    ).json()
    assert len(page["items"]) == 2
    assert page["total_items"] == 6

    limited = client.get(
        f"/api/datasets/{info['dataset_id']}/inventory", params={"scan_limit": 3}
    ).json()
    assert limited["scanned"] == 3
    assert limited["truncated"] is True, "扫描被上限截断时必须如实告知"


def test_sample_endpoint_serves_image(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed(db_session, storage_root, images=2)

    listing = client.get(f"/api/datasets/{info['dataset_id']}/inventory").json()
    relative = listing["items"][0]["relative_path"]

    response = client.get(
        f"/api/datasets/{info['dataset_id']}/samples", params={"path": relative}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_sample_endpoint_rejects_path_traversal(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed(db_session, storage_root, images=1)

    for attempt in ("../../../qforge.db", "..%2F..%2Fqforge.db", "sub/../../../../etc/passwd"):
        response = client.get(
            f"/api/datasets/{info['dataset_id']}/samples", params={"path": attempt}
        )
        assert response.status_code in {400, 404}, f"{attempt} 竟然返回 {response.status_code}"


def test_revalidate_without_model_skips_preprocessing_and_warns(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed(db_session, storage_root, images=2, bad=True)

    response = client.post(f"/api/datasets/{info['dataset_id']}/revalidate")

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["image_count"] == 3
    assert report["definition"] is None
    assert any("跳过" in warning for warning in report["warnings"])

    db_session.expire_all()
    dataset = db_session.get(Dataset, info["dataset_id"])
    assert dataset is not None and dataset.meta is not None
    assert dataset.meta["validation"]["image_count"] == 3, "重新校验结果必须写回 meta"
    assert dataset.image_count == 3


def test_revalidate_with_model_runs_preprocessing_check(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed(db_session, storage_root, images=3)
    model = OnnxModel(
        project_id=info["project_id"],
        name="yolov8n",
        architecture="yolov8",
        task_type="detection",
        model_definition=DEFINITION,
    )
    db_session.add(model)
    db_session.commit()

    response = client.post(
        f"/api/datasets/{info['dataset_id']}/revalidate", params={"model_id": model.id}
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["definition"]["model_id"] == model.id
    assert report["sample"], "带模型时必须做抽样预处理检查（SPEC 8.3 第 3 项）"
    assert report["sample"].get("tensor_shape")


def test_revalidate_with_unknown_model_returns_404(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed(db_session, storage_root, images=1)

    response = client.post(
        f"/api/datasets/{info['dataset_id']}/revalidate", params={"model_id": "nope"}
    )

    assert response.status_code == 404


def test_batch_delete_removes_rows_and_files(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    target = _seed(db_session, storage_root, images=2)
    keeper = _seed(db_session, storage_root, images=2)

    response = client.post(
        "/api/datasets/delete",
        json={"dataset_ids": [target["dataset_id"], "not-exist"]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted_count"] == 1
    assert body["failed_count"] == 1
    assert body["freed_bytes"] > 0

    db_session.expire_all()
    assert db_session.get(Dataset, target["dataset_id"]) is None
    assert not layout.dataset_root(storage_root, target["dataset_id"]).exists()
    # 邻居数据集不受影响
    assert db_session.get(Dataset, keeper["dataset_id"]) is not None
    assert layout.dataset_root(storage_root, keeper["dataset_id"]).exists()


def test_batch_delete_refuses_datasets_used_by_running_task(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    busy = _seed(db_session, storage_root, images=1)
    free = _seed(db_session, storage_root, images=1)
    task = Task(
        project_id=busy["project_id"],
        dataset_id=busy["dataset_id"],
        task_type="detection",
        precision="int8",
        backend_name="tensorrt",
        status=TaskStatus.BUILDING_ENGINE,
    )
    db_session.add(task)
    db_session.commit()

    response = client.post(
        "/api/datasets/delete",
        json={"dataset_ids": [busy["dataset_id"], free["dataset_id"]]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted_count"] == 1, "没被占用的那个应删掉"
    assert body["failed_count"] == 1
    assert "排队或运行中" in body["failed"][0]["reason"]

    db_session.expire_all()
    assert db_session.get(Dataset, busy["dataset_id"]) is not None, "被占用的数据集必须保留"
    assert db_session.get(Dataset, free["dataset_id"]) is None

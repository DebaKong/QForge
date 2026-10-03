"""项目删除：级联清理、运行中任务拒绝、批量部分失败、路径防护（纯 CPU）。"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.errors import PathSecurityError
from app.models.artifact import Artifact, JobLog
from app.models.project import Project
from app.models.task import Task, TaskConfig
from app.schemas.dataset import DatasetCreate
from app.schemas.model import ModelCreate
from app.schemas.project import ProjectCreate
from app.services import catalog, layout, storage
from app.services.task_state import TaskStatus


def _create_project(session: Session, name: str | None = None) -> Project:
    project = catalog.create_project(
        session,
        ProjectCreate(name=name or f"proj-{uuid.uuid4().hex[:8]}", description="删除用例"),
    )
    session.commit()
    return project


def _seed_project(session: Session, storage_root: Path, *, with_data: bool = True) -> dict:
    """造一个带模型/数据集/任务（含产物、日志、磁盘文件）的项目。"""
    project = _create_project(session)
    info: dict = {"project_id": project.id, "project_name": project.name}

    model_id = None
    dataset_id = None
    if with_data:
        model = catalog.create_model(
            session,
            ModelCreate(project_id=project.id, name=f"m-{uuid.uuid4().hex[:6]}", architecture="yolov8"),
        )
        dataset = catalog.create_dataset(
            session,
            DatasetCreate(project_id=project.id, name=f"d-{uuid.uuid4().hex[:6]}", kind="calibration"),
        )
        model_id, dataset_id = model.id, dataset.id
        session.commit()

        model_file = layout.model_input_dir(storage_root, model.id) / "model.onnx"
        model_file.write_bytes(b"onnx-bytes")
        dataset_file = layout.dataset_images_dir(storage_root, dataset.id) / "0001.ppm"
        dataset_file.write_bytes(b"ppm-bytes")

    task_ids = []
    for status in (TaskStatus.SUCCESS, TaskStatus.CREATED):
        task = Task(
            project_id=project.id,
            model_id=model_id,
            dataset_id=dataset_id,
            task_type="detection",
            precision="fp16",
            backend_name="tensorrt",
            status=status,
        )
        session.add(task)
        session.flush()
        session.add(TaskConfig(task_id=task.id, config={"precision": "fp16"}))
        session.add(
            Artifact(
                task_id=task.id,
                kind="archive",
                relative_path=f"{task.id}/report/artifact.zip",
                size_bytes=123,
            )
        )
        session.add(JobLog(task_id=task.id, stage="PACKAGING", level="INFO", message="打包完成"))
        task_dir = storage.task_dir(storage_root, task.id)
        (task_dir / "report").mkdir(parents=True, exist_ok=True)
        (task_dir / "report" / "artifact.zip").write_bytes(b"zip-bytes")
        task_ids.append(task.id)
    session.commit()

    info.update({"model_id": model_id, "dataset_id": dataset_id, "task_ids": task_ids})
    return info


def _path_exists(path: Path) -> bool:
    return path.exists()


def test_preview_reports_counts_and_disk_usage(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed_project(db_session, storage_root)

    response = client.get(f"/api/projects/{info['project_id']}/deletion-preview")

    assert response.status_code == 200
    body = response.json()
    assert body["project_name"] == info["project_name"]
    assert body["models"] == 1
    assert body["datasets"] == 1
    assert body["tasks"] == 2
    assert body["artifacts"] == 2
    assert body["logs"] == 2
    assert body["disk_bytes"] > 0, "要能告诉用户会释放多少磁盘"
    assert body["can_delete"] is True
    assert body["blocking_tasks"] == []


def test_delete_project_removes_metadata_files_and_directories(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed_project(db_session, storage_root)
    other = _seed_project(db_session, storage_root)  # 邻居项目：绝不能受影响

    response = client.delete(f"/api/projects/{info['project_id']}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted"]["tasks"] == 2
    assert body["deleted"]["models"] == 1
    assert body["deleted"]["datasets"] == 1
    assert body["deleted"]["artifacts"] == 2
    assert body["deleted"]["logs"] == 2
    assert body["deleted"]["task_configs"] == 2
    assert body["freed_bytes"] > 0

    # 元数据清干净
    db_session.expire_all()
    assert db_session.get(Project, info["project_id"]) is None
    assert db_session.scalars(
        select(Task).where(Task.project_id == info["project_id"])
    ).all() == []
    assert db_session.scalars(
        select(Artifact).where(Artifact.task_id.in_(info["task_ids"]))
    ).all() == []
    assert db_session.scalars(
        select(JobLog).where(JobLog.task_id.in_(info["task_ids"]))
    ).all() == []
    assert db_session.scalars(
        select(TaskConfig).where(TaskConfig.task_id.in_(info["task_ids"]))
    ).all() == []

    # 磁盘目录清干净
    for task_id in info["task_ids"]:
        assert not _path_exists(storage.task_dir(storage_root, task_id))
    assert not _path_exists(layout.model_root(storage_root, info["model_id"]))
    assert not _path_exists(layout.dataset_root(storage_root, info["dataset_id"]))

    # 邻居项目的任务目录与数据都还在
    assert db_session.get(Project, other["project_id"]) is not None
    for task_id in other["task_ids"]:
        assert _path_exists(storage.task_dir(storage_root, task_id))
    assert _path_exists(layout.model_root(storage_root, other["model_id"]))


def test_delete_project_is_refused_while_task_is_running(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    info = _seed_project(db_session, storage_root)
    task = db_session.get(Task, info["task_ids"][0])
    assert task is not None
    # 直接置成"运行中"状态：本用例只关心删除的准入判断
    task.status = TaskStatus.BUILDING
    task.current_stage = "BUILDING"
    db_session.commit()

    response = client.delete(f"/api/projects/{info['project_id']}")

    assert response.status_code == 409, response.text
    payload = response.json()
    assert payload["error_code"] == "CONFLICT"
    assert "排队或运行中" in payload["message"]
    assert payload["detail"]["blocking_tasks"][0]["status"] == "BUILDING"

    # 拒绝时必须什么都没删
    db_session.expire_all()
    assert db_session.get(Project, info["project_id"]) is not None
    for task_id in info["task_ids"]:
        assert _path_exists(storage.task_dir(storage_root, task_id))

    # 预览也应给出 can_delete=false，界面据此禁用删除按钮
    preview = client.get(f"/api/projects/{info['project_id']}/deletion-preview").json()
    assert preview["can_delete"] is False
    assert preview["blocking_tasks"][0]["id"] == task.id


def test_batch_delete_continues_after_one_failure(
    client: TestClient, db_session: Session, storage_root: Path
) -> None:
    keep = _seed_project(db_session, storage_root)
    blocked = _seed_project(db_session, storage_root)
    task = db_session.get(Task, blocked["task_ids"][0])
    assert task is not None
    task.status = TaskStatus.QUEUED
    task.current_stage = "QUEUED"
    db_session.commit()

    response = client.post(
        "/api/projects/delete",
        json={"project_ids": [keep["project_id"], blocked["project_id"], "not-exist"]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted_count"] == 1
    assert body["failed_count"] == 2
    reasons = {item["project_id"]: item["reason"] for item in body["failed"]}
    assert "排队或运行中" in reasons[blocked["project_id"]]
    assert "不存在" in reasons["not-exist"]

    db_session.expire_all()
    assert db_session.get(Project, keep["project_id"]) is None, "删成功的项目必须真的删掉"
    assert db_session.get(Project, blocked["project_id"]) is not None


def test_delete_missing_project_returns_404(client: TestClient) -> None:
    response = client.delete("/api/projects/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


def test_task_directory_helper_refuses_escaping_id(storage_root: Path) -> None:
    """磁盘删除只走这些受保护的工具函数：越界路径必须被拒绝。"""
    storage_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PathSecurityError):
        storage.task_dir(storage_root, "../outside")
    with pytest.raises(PathSecurityError):
        layout.model_root(storage_root, "..\\..\\windows")

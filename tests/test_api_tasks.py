"""Task API 与流水线接线测试（SPEC 13 / 16.1）。

阶段 0 的语义边界在这里被固定下来：
- 创建任务 = 落状态 + 落配置快照 + 建存储目录，不执行真实流水线；
- 入队后任务停在 QUEUED（占位 Worker 不推进阶段），并且必须留下日志；
- 不存在的阶段产物（报告）返回 501，而不是伪造数据。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.services.storage import TASK_SUBDIRS


def _bootstrap(client: TestClient) -> dict[str, str]:
    project = client.post("/api/projects", json={"name": "det-demo"}).json()
    dataset = client.post(
        "/api/datasets",
        json={"project_id": project["id"], "name": "calib", "kind": "calibration"},
    ).json()
    onnx_model = client.post(
        "/api/models",
        json={
            "project_id": project["id"],
            "name": "yolov8n",
            "architecture": "yolov8",
            "model_definition": {
                "preprocessing": {"resize": "letterbox", "color": "RGB"},
                "postprocessing": {"decoder": "yolov8"},
            },
        },
    ).json()
    return {"project_id": project["id"], "dataset_id": dataset["id"], "model_id": onnx_model["id"]}


def _create_task(client: TestClient, context: dict[str, str], **overrides: object) -> dict:
    payload: dict[str, object] = {
        "project_id": context["project_id"],
        "model_id": context["model_id"],
        "dataset_id": context["dataset_id"],
        "task_type": "detection",
        "precision": "fp16",
        "backend_name": "tensorrt",
    }
    payload.update(overrides)
    response = client.post("/api/tasks", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_create_task_initial_state(client: TestClient) -> None:
    created = _create_task(client, _bootstrap(client))
    assert created["status"] == "CREATED"
    assert created["progress"] == 0
    assert created["current_stage"] == "CREATED"
    assert created["precision"] == "fp16"
    assert created["error_code"] is None
    assert created["started_at"] is None
    assert created["finished_at"] is None


def test_create_task_builds_storage_layout(client: TestClient, storage_root: Path) -> None:
    created = _create_task(client, _bootstrap(client))
    task_dir = storage_root / created["id"]
    assert task_dir.is_dir()
    for name in TASK_SUBDIRS:
        assert (task_dir / name).is_dir(), name


def test_task_config_snapshot_follows_spec_5_1(client: TestClient) -> None:
    context = _bootstrap(client)
    created = _create_task(client, context)

    snapshot = client.get(f"/api/tasks/{created['id']}/config").json()["config"]
    assert snapshot["task_id"] == created["id"]
    assert snapshot["task_type"] == "detection"
    assert snapshot["precision"] == "fp16"
    assert snapshot["model"]["architecture"] == "yolov8"
    assert snapshot["model"]["model_id"] == context["model_id"]
    assert snapshot["backend"] == {"name": "tensorrt", "version": None}
    # 未显式提供的分节从模型定义继承
    assert snapshot["preprocess"] == {"resize": "letterbox", "color": "RGB"}
    assert snapshot["postprocess"] == {"decoder": "yolov8"}
    assert snapshot["calibration"]["dataset_id"] == context["dataset_id"]
    assert snapshot["build"] == {}
    assert snapshot["validation"] == {}


def test_int8_requires_calibration_dataset(client: TestClient) -> None:
    context = _bootstrap(client)
    response = client.post(
        "/api/tasks",
        json={
            "project_id": context["project_id"],
            "model_id": context["model_id"],
            "precision": "int8",
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert body["detail"]["precision"] == "int8"


def test_precision_and_backend_are_validated(client: TestClient) -> None:
    context = _bootstrap(client)
    bad_precision = client.post(
        "/api/tasks", json={"model_id": context["model_id"], "precision": "int4"}
    )
    assert bad_precision.status_code == 422
    assert bad_precision.json()["error_code"] == "VALIDATION_ERROR"

    bad_backend = client.post(
        "/api/tasks", json={"model_id": context["model_id"], "backend_name": "openvino"}
    )
    assert bad_backend.status_code == 422


def test_model_from_other_project_is_rejected(client: TestClient) -> None:
    context = _bootstrap(client)
    other = client.post("/api/projects", json={"name": "other"}).json()
    response = client.post(
        "/api/tasks",
        json={"project_id": other["id"], "model_id": context["model_id"]},
    )
    assert response.status_code == 409
    assert response.json()["error_code"] == "CONFLICT"


def test_enqueue_dispatches_placeholder_worker_and_keeps_queued(
    client: TestClient, storage_root: Path
) -> None:
    created = _create_task(client, _bootstrap(client))

    enqueued = client.post(f"/api/tasks/{created['id']}/enqueue")
    assert enqueued.status_code == 200, enqueued.text
    body = enqueued.json()

    # 阶段 0：占位 Worker 不推进阶段，任务必须停在 QUEUED（不伪造 SUCCESS）
    assert body["status"] == "QUEUED"
    assert body["progress"] == 5
    assert body["current_stage"] == "QUEUED"
    assert body["started_at"] is not None
    assert body["error_code"] is None
    assert body["worker_id"], "占位 Worker 应记录领取者"

    logs = client.get(f"/api/tasks/{created['id']}/logs").json()
    messages = [entry["message"] for entry in logs]
    assert any("已入队" in message for message in messages)
    assert any("占位实现" in message for message in messages)
    assert [entry["stage"] for entry in logs][-1] == "QUEUED"


def test_enqueue_is_rejected_outside_created_state(client: TestClient) -> None:
    created = _create_task(client, _bootstrap(client))
    client.post(f"/api/tasks/{created['id']}/enqueue")

    again = client.post(f"/api/tasks/{created['id']}/enqueue")
    assert again.status_code == 409
    assert again.json()["error_code"] == "CONFLICT"


def test_cancel_then_cancel_again_conflicts(client: TestClient) -> None:
    created = _create_task(client, _bootstrap(client))

    cancelled = client.post(f"/api/tasks/{created['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"
    assert cancelled.json()["finished_at"] is not None

    again = client.post(f"/api/tasks/{created['id']}/cancel")
    assert again.status_code == 409


def test_transitions_endpoint_exposes_state_machine(client: TestClient) -> None:
    created = _create_task(client, _bootstrap(client))

    transitions = client.get(f"/api/tasks/{created['id']}/transitions").json()
    assert transitions["status"] == "CREATED"
    assert transitions["next_status"] == "QUEUED"
    assert transitions["is_terminal"] is False
    assert transitions["allowed_transitions"] == ["CANCELLED", "FAILED", "QUEUED"]

    client.post(f"/api/tasks/{created['id']}/cancel")
    terminal = client.get(f"/api/tasks/{created['id']}/transitions").json()
    assert terminal["is_terminal"] is True
    assert terminal["allowed_transitions"] == []
    assert terminal["next_status"] is None


def test_artifacts_are_empty_in_phase_0(client: TestClient) -> None:
    created = _create_task(client, _bootstrap(client))
    response = client.get(f"/api/tasks/{created['id']}/artifacts")
    assert response.status_code == 200
    assert response.json() == []


def test_report_endpoint_is_explicitly_not_implemented(client: TestClient) -> None:
    created = _create_task(client, _bootstrap(client))
    response = client.get(f"/api/tasks/{created['id']}/report")
    assert response.status_code == 501
    body = response.json()
    assert body["error_code"] == "NOT_IMPLEMENTED"
    assert body["detail"]["planned_phase"] == "phase-2"


def test_unknown_task_endpoints_return_404(client: TestClient) -> None:
    for path in ("", "/config", "/logs", "/artifacts", "/transitions", "/report"):
        response = client.get(f"/api/tasks/missing{path}")
        assert response.status_code == 404, path
        assert response.json()["error_code"] == "NOT_FOUND"


def test_task_list_filters_by_project_and_status(client: TestClient) -> None:
    context = _bootstrap(client)
    first = _create_task(client, context)
    _create_task(client, context, precision="fp32")

    by_project = client.get("/api/tasks", params={"project_id": context["project_id"]}).json()
    assert len(by_project) == 2

    client.post(f"/api/tasks/{first['id']}/cancel")
    cancelled = client.get("/api/tasks", params={"status": "CANCELLED"}).json()
    assert [item["id"] for item in cancelled] == [first["id"]]

    created_only = client.get("/api/tasks", params={"status": "CREATED"}).json()
    assert len(created_only) == 1

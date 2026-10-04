"""Project / Model / Dataset API 测试（SPEC 16.1）。"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _create_project(client: TestClient, name: str = "demo") -> dict:
    response = client.post("/api/projects", json={"name": name, "description": "演示项目"})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_project_returns_201_and_metadata(client: TestClient) -> None:
    payload = _create_project(client, "p1")
    assert payload["id"]
    assert payload["name"] == "p1"
    assert payload["description"] == "演示项目"
    assert payload["created_at"] and payload["updated_at"]


def test_duplicate_project_name_conflicts(client: TestClient) -> None:
    _create_project(client, "p1")
    response = client.post("/api/projects", json={"name": "p1"})
    assert response.status_code == 409
    body = response.json()
    assert body["error_code"] == "CONFLICT"
    assert body["detail"]["name"] == "p1"


def test_project_list_and_detail(client: TestClient) -> None:
    created = _create_project(client, "p1")
    assert [item["id"] for item in client.get("/api/projects").json()] == [created["id"]]

    detail = client.get(f"/api/projects/{created['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == created["id"]


def test_unknown_project_returns_structured_404(client: TestClient) -> None:
    response = client.get("/api/projects/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"
    assert response.json()["detail"]["project_id"] == "does-not-exist"


def test_create_model_requires_existing_project(client: TestClient) -> None:
    response = client.post("/api/models", json={"project_id": "missing", "name": "yolov8n"})
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


def test_create_model_persists_definition_and_rejects_duplicate_name(client: TestClient) -> None:
    project = _create_project(client)
    definition = {
        "input": {"name": "images", "shape": [1, 3, 640, 640], "layout": "NCHW"},
        "preprocessing": {"resize": "letterbox", "color": "RGB"},
        "postprocessing": {"decoder": "yolov8", "nms": {"type": "classwise"}},
    }
    payload = {
        "project_id": project["id"],
        "name": "yolov8n",
        "architecture": "yolov8",
        "task_type": "detection",
        "opset": 17,
        "model_definition": definition,
    }
    response = client.post("/api/models", json=payload)
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["architecture"] == "yolov8"
    assert created["model_definition"] == definition

    duplicate = client.post("/api/models", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json()["error_code"] == "CONFLICT"

    listed = client.get("/api/models", params={"project_id": project["id"]}).json()
    assert [item["id"] for item in listed] == [created["id"]]


def test_create_model_rejects_unsupported_task_type(client: TestClient) -> None:
    project = _create_project(client)
    # 注意：segmentation 自阶段 2 起已受支持（SPEC 2.2），这里改用确实不支持的分类任务
    response = client.post(
        "/api/models",
        json={"project_id": project["id"], "name": "m", "task_type": "classification"},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"


def test_create_dataset_and_filter_by_project(client: TestClient) -> None:
    project = _create_project(client)
    response = client.post(
        "/api/datasets",
        json={"project_id": project["id"], "name": "calib-100", "kind": "calibration"},
    )
    assert response.status_code == 201, response.text
    dataset = response.json()
    assert dataset["kind"] == "calibration"
    assert dataset["image_count"] == 0

    listed = client.get("/api/datasets", params={"project_id": project["id"]}).json()
    assert [item["id"] for item in listed] == [dataset["id"]]
    assert client.get("/api/datasets", params={"project_id": "other"}).json() == []


def test_create_dataset_rejects_unknown_kind(client: TestClient) -> None:
    project = _create_project(client)
    response = client.post(
        "/api/datasets",
        json={"project_id": project["id"], "name": "d", "kind": "training"},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"

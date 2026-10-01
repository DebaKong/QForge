"""Celery 接线与错误信封测试（SPEC 3 / 13.1）。"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from workers.celery_app import celery_app
from workers.tasks import probe


def test_celery_app_is_configured_for_phase_0() -> None:
    assert celery_app.conf.task_always_eager is True, "阶段 0 无 Redis，必须走 eager 执行"
    assert "workers.tasks" in celery_app.conf.include
    assert celery_app.conf.task_time_limit > 0, "SPEC 15：任务必须有超时上限"
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_registered_task_names() -> None:
    registered = set(celery_app.tasks)
    assert "qforge.probe" in registered
    assert "qforge.run_task_pipeline" in registered


def test_probe_task_runs_and_returns_result() -> None:
    result = probe.delay()
    payload: dict[str, Any] = result.get()
    assert payload["status"] == "ok"
    assert payload["task"] == "qforge.probe"


def test_health_reports_phase_and_environment(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["phase"] == "phase-1"
    assert body["celery_eager"] is True
    assert body["database"].startswith("sqlite:ok")
    assert "password" not in response.text.lower()


def test_validation_error_envelope(client: TestClient) -> None:
    response = client.post("/api/projects", json={"name": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert body["detail"]["errors"]


def test_unhandled_exception_envelope(lenient_client: TestClient) -> None:
    from app.main import app

    @app.get("/api/_boom")
    def _boom() -> None:
        raise RuntimeError("intentional failure for envelope test")

    response = lenient_client.get("/api/_boom")
    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "INTERNAL_ERROR"
    assert body["detail"]["type"] == "RuntimeError"


def test_openapi_document_is_available(client: TestClient) -> None:
    document = client.get("/openapi.json").json()
    paths = set(document["paths"])
    for expected in (
        "/api/health",
        "/api/projects",
        "/api/projects/{project_id}",
        "/api/models",
        "/api/datasets",
        "/api/tasks",
        "/api/tasks/{task_id}",
        "/api/tasks/{task_id}/cancel",
        "/api/tasks/{task_id}/logs",
        "/api/tasks/{task_id}/artifacts",
        "/api/tasks/{task_id}/report",
    ):
        assert expected in paths, expected

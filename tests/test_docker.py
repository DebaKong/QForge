"""容器镜像构建与运行验证（SPEC 12 / SPEC 18 验收第 10 条）。

前提：Docker daemon 可用，且基础镜像（默认 `nvcr.io/nvidia/tensorrt:26.03-py3`，约 5.5 GB）
已拉取或可拉取。默认跳过；显式运行：

    pytest tests/test_docker.py -m docker -q -s

验证内容：
1. 任务在 `build.docker_build=true` 下真实执行 `docker build` 并成功；
2. 产物中出现 `report/docker_build.json` 且状态为 SUCCESS；
3. 可选（`build.docker_run=true`）运行容器，容器内加载 Engine 并完成一次真实推理。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_pipeline_e2e import _calibration_zip, _onnx_bytes, _wait_for_terminal

pytestmark = [pytest.mark.docker]

BUILD_TIMEOUT_SECONDS = 3600


def _docker_ready() -> tuple[bool, str]:
    from app.services import docker_build

    available, info = docker_build.docker_available()
    return available, info.get("reason", "")


def test_docker_image_builds_and_runs(client: TestClient, tmp_path: Path) -> None:
    available, reason = _docker_ready()
    if not available:
        pytest.skip(f"Docker 不可用：{reason}")

    project = client.post("/api/projects", json={"name": "docker-check"}).json()

    model_response = client.post(
        "/api/models/upload",
        data={
            "project_id": project["id"],
            "name": "docker-model",
            "architecture": "yolov8",
            "task_type": "detection",
        },
        files={"file": ("synth.onnx", _onnx_bytes(tmp_path), "application/octet-stream")},
    )
    assert model_response.status_code == 201, model_response.text
    model_id = model_response.json()["id"]

    # FP16：容器内不需要校准数据，镜像构建与推理验证路径最短
    created = client.post(
        "/api/tasks",
        json={
            "project_id": project["id"],
            "model_id": model_id,
            "precision": "fp16",
            "backend_name": "tensorrt",
            "build": {"cpp_build": "required", "docker_build": True, "docker_run": True},
        },
    )
    assert created.status_code == 201, created.text
    task_id = created.json()["id"]

    assert client.post(f"/api/tasks/{task_id}/enqueue").status_code == 200

    final = _wait_for_terminal(client, task_id, timeout=BUILD_TIMEOUT_SECONDS)
    logs = client.get(f"/api/tasks/{task_id}/logs").json()
    if final["status"] != "SUCCESS":
        tail = "\n".join(f"[{item['level']}] {item['stage']}: {item['message']}" for item in logs[-12:])
        pytest.fail(f"任务未成功：{final['status']} / {final['error_code']}\n{tail}")

    artifacts = client.get(f"/api/tasks/{task_id}/artifacts").json()
    report_artifact = None
    for artifact in artifacts:
        if artifact["relative_path"].endswith("report/docker_build.json"):
            report_artifact = artifact
            break
    assert report_artifact is not None, "缺少 report/docker_build.json"

    import json

    response = client.get(f"/api/tasks/{task_id}/artifacts/{report_artifact['id']}/download")
    payload = json.loads(response.content.decode("utf-8"))

    assert payload["status"] == "SUCCESS", payload
    assert payload["build"]["image_tag"].startswith("qforge-")
    assert payload["base_image"].startswith("nvcr.io/nvidia/tensorrt:")

    run = payload.get("container_run")
    if run is not None:
        print(f"\n容器运行退出码={run['returncode']} 日志尾部：\n{run.get('tail', '')}")
        assert run["returncode"] == 0, f"容器内推理失败：{run}"

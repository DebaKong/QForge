"""阶段 1 端到端验收测试（需要 GPU + TensorRT + C++ 工具链）。

覆盖 SPEC 18 的 MVP 验收项：
1. 上传合法 ONNX 模型（走真实上传接口，含校验链）；
2. 上传校准图像 ZIP 并完成数据校验；
3. 选择 FP16 / INT8 精度；
4. 构建 Engine 并通过最小推理验证；
5. 生成 C++ 工程并**真实编译**；
6. 运行生成的程序并校验输出结构；
7. 产物打包与报告落盘。

默认跳过（`-m "not gpu"`）；显式运行：
    pytest tests/test_pipeline_e2e.py -m gpu -q -s
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tools.synth_model import build_yolov8_like_onnx

pytestmark = [pytest.mark.gpu, pytest.mark.cpp]

INPUT_SIZE = 64
CLASS_COUNT = 80
ENGINE_TIMEOUT_SECONDS = 900


def _onnx_bytes(tmp_path: Path) -> bytes:
    path = build_yolov8_like_onnx(
        tmp_path / "synth.onnx", input_size=INPUT_SIZE, class_count=CLASS_COUNT
    )
    return path.read_bytes()


def _calibration_zip(image_count: int = 4, size: int = 96) -> bytes:
    """生成校准集压缩包（images/ 目录 + 若干 PNG）。"""
    rng = np.random.default_rng(7)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(image_count):
            array = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            image_buffer = io.BytesIO()
            Image.fromarray(array).save(image_buffer, format="PNG")
            archive.writestr(f"images/{index:06d}.png", image_buffer.getvalue())
    return buffer.getvalue()


def _wait_for_terminal(client: TestClient, task_id: str, *, timeout: float) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = client.get(f"/api/tasks/{task_id}").json()
        if last["status"] in {"SUCCESS", "FAILED", "CANCELLED"}:
            return last
        time.sleep(0.5)
    raise AssertionError(f"任务在 {timeout}s 内未结束：{last}")


def _setup_project(client: TestClient, name: str, tmp_path: Path) -> dict[str, str]:
    project = client.post("/api/projects", json={"name": name}).json()

    model_response = client.post(
        "/api/models/upload",
        data={
            "project_id": project["id"],
            "name": f"{name}-model",
            "architecture": "yolov8",
            "task_type": "detection",
        },
        files={"file": ("synth.onnx", _onnx_bytes(tmp_path), "application/octet-stream")},
    )
    assert model_response.status_code == 201, model_response.text
    model = model_response.json()

    dataset_response = client.post(
        "/api/datasets/upload",
        data={"project_id": project["id"], "name": f"{name}-calib", "kind": "calibration"},
        files={"file": ("calibration.zip", _calibration_zip(), "application/zip")},
    )
    assert dataset_response.status_code == 201, dataset_response.text
    dataset = dataset_response.json()
    assert dataset["image_count"] == 4

    return {"project_id": project["id"], "model_id": model["id"], "dataset_id": dataset["id"]}


def _run_task(client: TestClient, context: dict[str, str], precision: str, tmp_path: Path) -> dict:
    created = client.post(
        "/api/tasks",
        json={
            "project_id": context["project_id"],
            "model_id": context["model_id"],
            "dataset_id": context["dataset_id"] if precision == "int8" else None,
            "precision": precision,
            "backend_name": "tensorrt",
            "build": {"cpp_build": "required"},
        },
    )
    assert created.status_code == 201, created.text
    task_id = created.json()["id"]

    enqueued = client.post(f"/api/tasks/{task_id}/enqueue")
    assert enqueued.status_code == 200, enqueued.text

    final = _wait_for_terminal(client, task_id, timeout=ENGINE_TIMEOUT_SECONDS)
    logs = client.get(f"/api/tasks/{task_id}/logs").json()

    if final["status"] != "SUCCESS":
        tail = "\n".join(f"[{item['level']}] {item['stage']}: {item['message']}" for item in logs[-15:])
        pytest.fail(
            f"任务未成功：status={final['status']} error_code={final['error_code']}\n{tail}"
        )

    return {"task_id": task_id, "task": final, "logs": logs}


def _read_report(client: TestClient, task_id: str, suffix: str) -> dict:
    import json

    artifacts = client.get(f"/api/tasks/{task_id}/artifacts").json()
    for artifact in artifacts:
        if artifact["relative_path"].endswith(suffix):
            response = client.get(
                f"/api/tasks/{task_id}/artifacts/{artifact['id']}/download"
            )
            assert response.status_code == 200, response.text
            return json.loads(response.content.decode("utf-8"))
    raise AssertionError(f"未找到报告文件：{suffix}")


@pytest.mark.parametrize("precision", ["fp16", "int8"])
def test_full_pipeline_end_to_end(
    client: TestClient, tmp_path: Path, precision: str, storage_root: Path
) -> None:
    context = _setup_project(client, f"e2e-{precision}", tmp_path)
    outcome = _run_task(client, context, precision, tmp_path)
    task_id = outcome["task_id"]

    # 1) Engine 真实构建并记录环境元数据（SPEC 4.1 / 10.1）
    engine_build = _read_report(client, task_id, "report/engine_build.json")
    metadata = engine_build["engine_metadata"]
    assert metadata["backend"] == "tensorrt"
    assert metadata["precision"] == precision.upper()
    assert metadata["gpu_arch"] and metadata["gpu_arch"].startswith("sm_")
    assert metadata["engine_size_bytes"] > 0
    assert metadata["input_profile"]["opt"] == [1, 3, INPUT_SIZE, INPUT_SIZE]
    if precision == "int8":
        assert metadata["calibration"]["method"] == "IInt8EntropyCalibrator2"
        assert metadata["calibration"]["images"] == 4

    # 2) 生成的工程被真实编译且程序真实运行（SPEC 11.3）
    cpp_build = _read_report(client, task_id, "report/cpp_build.json")
    assert cpp_build["status"] == "SUCCESS", cpp_build

    runtime_report = _read_report(client, task_id, "report/runtime_verification.json")
    python_side = runtime_report["python_engine_verification"]
    cpp_side = runtime_report["cpp_program_verification"]
    assert python_side["status"] == "SUCCESS"
    assert cpp_side["status"] == "SUCCESS", cpp_side
    assert cpp_side["returncode"] == 0
    assert cpp_side["result"]["output_elements"] == (4 + CLASS_COUNT) * (INPUT_SIZE // 4) ** 2

    # 3) 精度指标以 FP32 为基准（SPEC 9.2）
    accuracy = _read_report(client, task_id, "report/accuracy.json")
    metrics = accuracy["engine_vs_fp32_baseline"]
    assert metrics["baseline"].startswith("onnxruntime")
    assert metrics["elements"] > 0
    assert np.isfinite(metrics["mae"]) and np.isfinite(metrics["rmse"])
    assert metrics["cosine_similarity"] > 0.0
    if precision == "fp32":
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-5)

    # 4) 产物与报告齐备（SPEC 12.2 / 13.1）
    artifacts = client.get(f"/api/tasks/{task_id}/artifacts").json()
    kinds = {item["kind"] for item in artifacts}
    assert {"engine", "report", "source", "config", "docker"} <= kinds
    archive = [item for item in artifacts if item["relative_path"].endswith("artifact.zip")]
    assert archive and archive[0]["size_bytes"] > 0

    task = client.get(f"/api/tasks/{task_id}").json()
    assert task["artifact_id"], "SPEC 13.1：任务应指向最终产物"

    print(
        f"\n[{precision}] Engine {metadata['engine_size_bytes'] / 1024:.1f} KB | "
        f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} "
        f"cosine={metrics['cosine_similarity']:.6f} | "
        f"检测 {cpp_side['result']['detections_count']} 条 | "
        f"耗时 {sum(s['duration_seconds'] for s in []) or ''}"
    )

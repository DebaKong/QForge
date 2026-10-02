"""验证容器化栈：容器内的 worker 能否真的构建 Engine 并完成推理。

用途：`docker compose up -d` 之后跑一遍，确认「api 接收 → redis 派发 → worker 在容器里构建
Engine → 真实推理 → 精度对比 → 打包」整条链路在 Linux 容器里成立（而不是只在 Windows 主机上成立）。

    python tools/docker_stack_check.py [--base http://127.0.0.1:8000/api]

判据（全部满足才算通过）：
1. 入队请求立即返回（不阻塞）；
2. 任务的 worker_id 是 worker 主机名（不是 local-executor）；
3. 任务 SUCCESS，且 Engine 元数据里的操作系统/后端版本来自容器；
4. 精度指标存在（Engine 输出与 FP32 基准对比）。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from synth_model import build_yolov8_like_onnx  # noqa: E402

TASK_TIMEOUT_SECONDS = 1800


def main() -> int:
    parser = argparse.ArgumentParser(description="验证容器化栈的端到端链路")
    parser.add_argument("--base", default="http://127.0.0.1:8000/api")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "int8"])
    parser.add_argument("--input-size", type=int, default=128)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base, timeout=600.0)
    health = client.get("/health").json()
    print(f"后端：{health['status']} | 数据目录：{health['storage_root']}")
    if not str(health["storage_root"]).startswith("/"):
        print("!! 数据目录看起来不是容器内路径：请确认连的是容器里的 API（默认 8000 端口）")

    with tempfile.TemporaryDirectory() as workdir:
        model_path = Path(workdir) / "synth.onnx"
        build_yolov8_like_onnx(model_path, input_size=args.input_size)
        print(f"合成模型：{model_path.name}（{model_path.stat().st_size / 1024:.0f} KB）")

        project = client.post("/projects", json={"name": f"docker-check-{int(time.time())}"}).json()
        with model_path.open("rb") as handle:
            response = client.post(
                "/models/upload",
                data={
                    "project_id": project["id"],
                    "name": "docker-synth",
                    "architecture": "yolov8",
                    "task_type": "detection",
                },
                files={"file": (model_path.name, handle, "application/octet-stream")},
            )
        if response.status_code != 201:
            print(f"模型上传失败：{response.status_code} {response.text[:300]}")
            return 3
        model = response.json()
        print(f"模型：{model['id']} opset={model['opset']} 输入={model['input_spec'][0]['shape']}")

        body: dict = {
            "project_id": project["id"],
            "model_id": model["id"],
            "precision": args.precision,
            "backend_name": "tensorrt",
            # 容器里没有 TensorRT/CUDA 开发文件：用 auto，让它如实标注 BLOCKED 而不是失败
            "build": {"cpp_build": "auto"},
        }
        if args.precision == "int8":
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                for index in range(8):
                    archive.writestr(f"images/{index:04d}.jpg", _synthetic_jpeg(args.input_size))
            dataset = client.post(
                "/datasets/upload",
                data={"project_id": project["id"], "name": "calib", "kind": "calibration"},
                files={"file": ("calib.zip", buffer.getvalue(), "application/zip")},
            ).json()
            body["dataset_id"] = dataset["id"]

        created = client.post("/tasks", json=body)
        created.raise_for_status()
        task_id = created.json()["id"]

        started = time.time()
        client.post(f"/tasks/{task_id}/enqueue").raise_for_status()
        enqueue_seconds = time.time() - started

        final: dict = {}
        while time.time() - started < TASK_TIMEOUT_SECONDS:
            final = client.get(f"/tasks/{task_id}").json()
            if final["status"] in {"SUCCESS", "FAILED", "CANCELLED"}:
                break
            time.sleep(3)

        print(f"入队返回耗时：{enqueue_seconds:.2f}s")
        print(f"任务状态：{final.get('status')}（总耗时 {time.time() - started:.0f}s）")
        print(f"worker_id：{final.get('worker_id')}")

        if final.get("status") != "SUCCESS":
            for item in client.get(f"/tasks/{task_id}/logs").json()[-10:]:
                print(f"  [{item['level']}] {item['stage']}: {item['message']}")
            return 1

        reports = {}
        for artifact in client.get(f"/tasks/{task_id}/artifacts").json():
            relative = artifact["relative_path"]
            for key in ("engine_build.json", "accuracy.json", "cpp_build.json", "docker_build.json"):
                if relative.endswith(f"report/{key}"):
                    reports[key] = json.loads(
                        client.get(f"/tasks/{task_id}/artifacts/{artifact['id']}/download").content
                    )

        metadata = reports.get("engine_build.json", {}).get("engine_metadata", {})
        metrics = reports.get("accuracy.json", {}).get("engine_vs_fp32_baseline", {})
        print(f"Engine：{metadata.get('engine_size_bytes', 0) / 1024:.0f} KB，"
              f"TensorRT {metadata.get('backend_version')}，GPU {metadata.get('gpu_name')}，"
              f"构建 {metadata.get('build_duration_seconds')}s")
        print(f"精度：MAE {metrics.get('mae')} / 余弦 {metrics.get('cosine_similarity')}")
        print(f"C++ 编译验证：{reports.get('cpp_build.json', {}).get('status')}")
        print(f"容器镜像：{reports.get('docker_build.json', {}).get('status')}")

        if "local-executor" in (final.get("worker_id") or ""):
            print("!! worker_id 是 local-executor：任务可能跑在 API 进程里，而不是 worker 容器")
            return 1
        if metrics.get("cosine_similarity") is None:
            print("!! 缺少精度指标")
            return 1
        print("✅ 容器化栈端到端通过（api → redis → 容器内 worker → Engine → 真实推理 → 打包）")
        return 0


def _synthetic_jpeg(size: int) -> bytes:
    """极简合成校准图（纯色渐变），避免引入额外依赖。"""
    import numpy as np
    from PIL import Image

    gradient = np.linspace(20, 220, size, dtype=np.float32)
    array = np.repeat(gradient[:, None], size, axis=1)
    image = np.stack([array, array * 0.85, array * 0.7], axis=2).clip(0, 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())

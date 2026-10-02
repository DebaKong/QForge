"""用**真实** YOLOv8n 模型跑一遍完整流水线（FP16 + INT8），并打印可核对的实测数据。

用法：
    python tools/real_model_check.py <yolov8n.onnx 路径>

说明：校准集是**平台生成的合成图像**（渐变 + 色块 + 噪声），不是真实场景数据。
因此 INT8 的精度指标只能说明「流水线可用」，不能代表真实业务精度——
这一点会在输出中明确标注（SPEC 9.2：精度下降先查校准数据）。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

BASE_URL = "http://127.0.0.1:8000/api"
TASK_TIMEOUT_SECONDS = 1800


def make_calibration_zip(image_count: int, size: int) -> bytes:
    """生成结构化合成校准图（渐变背景 + 随机色块 + 轻噪声），标注为合成数据。"""
    rng = np.random.default_rng(20261002)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(image_count):
            # 纵向渐变 + 若干随机色块：比纯噪声更接近自然图的统计特性
            gradient = np.linspace(40, 200, size, dtype=np.float32)[:, None]
            image = np.repeat(gradient, size, axis=1)
            image = np.stack([image, image * 0.9, image * 0.8], axis=2)

            for _ in range(rng.integers(2, 6)):
                height = int(rng.integers(size // 8, size // 3))
                width = int(rng.integers(size // 8, size // 3))
                top = int(rng.integers(0, size - height))
                left = int(rng.integers(0, size - width))
                colour = rng.integers(0, 255, 3)
                image[top : top + height, left : left + width] = colour

            image += rng.normal(0.0, 6.0, image.shape)
            array = np.clip(image, 0, 255).astype(np.uint8)

            item = io.BytesIO()
            Image.fromarray(array).save(item, format="JPEG", quality=88)
            archive.writestr(f"images/{index:06d}.jpg", item.getvalue())
    return buffer.getvalue()


def wait_for_task(client: httpx.Client, task_id: str) -> dict:
    deadline = time.time() + TASK_TIMEOUT_SECONDS
    last: dict = {}
    while time.time() < deadline:
        last = client.get(f"/tasks/{task_id}").json()
        if last["status"] in {"SUCCESS", "FAILED", "CANCELLED"}:
            return last
        time.sleep(2)
    raise TimeoutError(f"任务 {task_id} 超时：{last}")


def report_payload(client: httpx.Client, task_id: str, suffix: str) -> dict | None:
    artifacts = client.get(f"/tasks/{task_id}/artifacts").json()
    for artifact in artifacts:
        if artifact["relative_path"].endswith(suffix):
            response = client.get(f"/tasks/{task_id}/artifacts/{artifact['id']}/download")
            if response.status_code == 200:
                return json.loads(response.content.decode("utf-8"))
    return None


def run_precision(client: httpx.Client, context: dict, precision: str) -> dict:
    print(f"\n{'=' * 78}\n精度模式 {precision.upper()}\n{'=' * 78}")
    created = client.post(
        "/tasks",
        json={
            "project_id": context["project_id"],
            "model_id": context["model_id"],
            "dataset_id": context["dataset_id"] if precision == "int8" else None,
            "precision": precision,
            "backend_name": "tensorrt",
            # 要求必须编译：编译不过就失败，不允许悄悄跳过
            "build": {"cpp_build": "required"},
        },
    )
    created.raise_for_status()
    task_id = created.json()["id"]
    print(f"任务已创建：{task_id}")

    started = time.time()
    client.post(f"/tasks/{task_id}/enqueue").raise_for_status()
    final = wait_for_task(client, task_id)
    elapsed = time.time() - started
    print(f"任务状态：{final['status']}（总耗时 {elapsed:.0f}s）")

    if final["status"] != "SUCCESS":
        logs = client.get(f"/tasks/{task_id}/logs").json()
        print(f"错误码：{final['error_code']} / {final['message']}")
        for item in logs[-12:]:
            print(f"  [{item['level']}] {item['stage']}: {item['message']}")
        return {"precision": precision, "status": final["status"], "error": final["error_code"]}

    engine_report = report_payload(client, task_id, "report/engine_build.json") or {}
    metadata = engine_report.get("engine_metadata", {})
    runtime = report_payload(client, task_id, "report/runtime_verification.json") or {}
    accuracy = report_payload(client, task_id, "report/accuracy.json") or {}
    cpp_build = report_payload(client, task_id, "report/cpp_build.json") or {}

    python_side = runtime.get("python_engine_verification", {})
    cpp_side = runtime.get("cpp_program_verification", {})
    metrics = accuracy.get("engine_vs_fp32_baseline", {})

    print(
        f"Engine        : {metadata.get('engine_size_bytes', 0) / 1024 / 1024:.1f} MB，"
        f"构建 {metadata.get('build_duration_seconds')}s，精度 {metadata.get('precision')}，"
        f"GPU {metadata.get('gpu_arch')}"
    )
    print(f"C++ 工程编译  : {cpp_build.get('status')}")
    print(
        f"Python 侧推理 : 输出 {python_side.get('output_shapes')}，"
        f"{python_side.get('latency_ms', 0):.1f} ms/次"
    )
    if cpp_side.get("status") == "SUCCESS":
        result = cpp_side.get("result", {})
        print(
            f"生成的程序    : 用 PPM 图像真实推理，输出元素 {result.get('output_elements')}，"
            f"{result.get('latency_ms_avg', 0):.2f} ms/次，检测 {result.get('detections_count')} 条"
        )
    else:
        print(f"生成的程序    : {cpp_side.get('status')} - {cpp_side.get('reason')}")
    print(
        f"精度（对照 FP32 基准）: MAE {metrics.get('mae')} / RMSE {metrics.get('rmse')} / "
        f"余弦 {metrics.get('cosine_similarity')} / 最大绝对误差 {metrics.get('max_abs_error')}"
    )
    if precision == "int8":
        calibration = metadata.get("calibration") or {}
        print(
            f"校准信息      : 方法={calibration.get('method')} 图像数={calibration.get('images')} "
            f"批次数={calibration.get('batches')} → **校准数据为合成图像，指标不代表真实业务精度**"
        )

    return {
        "precision": precision,
        "status": "SUCCESS",
        "engine_mb": round(metadata.get("engine_size_bytes", 0) / 1024 / 1024, 1),
        "build_seconds": metadata.get("build_duration_seconds"),
        "python_latency_ms": python_side.get("latency_ms"),
        "cpp_latency_ms": (cpp_side.get("result") or {}).get("latency_ms_avg"),
        "metrics": metrics,
        "task_id": task_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="用真实 YOLOv8n 跑完整流水线")
    parser.add_argument("model", type=Path, help="yolov8n.onnx 路径")
    parser.add_argument("--images", type=int, default=16, help="合成校准图数量")
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"模型不存在：{args.model}")
        return 2

    client = httpx.Client(base_url=BASE_URL, timeout=600.0)
    health = client.get("/health").json()
    print(f"后端：{health['status']} / {health['app']} {health['version']} / {health['database']}")

    project = client.post(
        "/projects", json={"name": f"real-yolov8n-{int(time.time())}", "description": "真实模型验证"}
    ).json()
    print(f"项目：{project['id']}")

    with args.model.open("rb") as handle:
        model_response = client.post(
            "/models/upload",
            data={
                "project_id": project["id"],
                "name": "yolov8n",
                "architecture": "yolov8",
                "task_type": "detection",
            },
            files={"file": (args.model.name, handle, "application/octet-stream")},
        )
    if model_response.status_code != 201:
        print(f"模型上传失败：{model_response.status_code} {model_response.text[:500]}")
        return 3
    model = model_response.json()
    print(f"模型：{model['id']} opset={model['opset']} 大小={model['size_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  输入：{model['input_spec']}")
    print(f"  输出：{model['output_spec']}")
    print(f"  Model Definition 由适配器给出默认值（可显式覆盖）")

    dataset_response = client.post(
        "/datasets/upload",
        data={"project_id": project["id"], "name": "synthetic-calib", "kind": "calibration"},
        files={
            "file": (
                "calibration.zip",
                make_calibration_zip(args.images, args.image_size),
                "application/zip",
            )
        },
    )
    if dataset_response.status_code != 201:
        print(f"校准集上传失败：{dataset_response.status_code} {dataset_response.text[:500]}")
        return 4
    dataset = dataset_response.json()
    print(f"校准集：{dataset['id']}（合成图像 {dataset['image_count']} 张，{args.image_size}x{args.image_size}）")

    context = {"project_id": project["id"], "model_id": model["id"], "dataset_id": dataset["id"]}
    results = []
    for precision in ("fp16", "int8"):
        results.append(run_precision(client, context, precision))

    print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
    print(f"{'精度':<6}{'Engine':>10}{'构建(s)':>10}{'Python(ms)':>12}{'C++(ms)':>10}{'MAE':>12}{'余弦':>12}")
    for item in results:
        if item["status"] != "SUCCESS":
            print(f"{item['precision']:<6}{item['status']:>10}  ({item.get('error')})")
            continue
        metrics = item["metrics"]
        print(
            f"{item['precision']:<6}{item['engine_mb']:>9.1f}M{item['build_seconds']:>10}"
            f"{item['python_latency_ms']:>12.1f}{item['cpp_latency_ms'] or 0:>10.2f}"
            f"{metrics.get('mae', 0):>12.6f}{metrics.get('cosine_similarity', 0):>12.6f}"
        )
    return 0 if all(item["status"] == "SUCCESS" for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())

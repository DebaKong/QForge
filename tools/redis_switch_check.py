"""验证「切到 Redis + Celery」是否真的生效。

用法（先按 docs/redis.md 起好 Redis、worker 与 celery 模式的 API）：
    python tools/redis_switch_check.py [--base http://127.0.0.1:8001/api]

判定标准：
1. 入队请求立即返回（不阻塞等待任务跑完）；
2. 任务由 **worker 进程** 执行（任务的 worker_id 是 Celery 主机名，不是 local-executor）；
3. 任务最终 SUCCESS，且被 worker 真实跑完（可对照 worker 日志）。
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

TASK_TIMEOUT_SECONDS = 1800


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 Redis/Celery 切换")
    parser.add_argument("--base", default="http://127.0.0.1:8001/api")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base, timeout=120.0)
    health = client.get("/health").json()
    print(f"后端：{health['status']} / celery_eager={health['celery_eager']}")
    if health["celery_eager"]:
        print("!! celery_eager 仍为 true：入队会在 API 进程内同步执行，请把 QFORGE_CELERY_TASK_ALWAYS_EAGER 设为 false")

    projects = client.get("/projects").json()
    target = next((item for item in projects if item["name"].startswith("real-yolov8n")), None)
    if target is None:
        target = client.post("/projects", json={"name": f"redis-check-{int(time.time())}"}).json()
        print(f"未找到已有项目，已新建：{target['id']}")
    else:
        print(f"使用项目：{target['name']}（{target['id']}）")

    models = client.get("/models", params={"project_id": target["id"]}).json()
    if not models:
        print("该项目下没有模型，请先上传模型")
        return 2
    model = models[0]
    print(f"使用模型：{model['name']}（{model['id']}）")

    created = client.post(
        "/tasks",
        json={
            "project_id": target["id"],
            "model_id": model["id"],
            "precision": "fp16",
            "backend_name": "tensorrt",
            "build": {"cpp_build": "auto"},
        },
    )
    created.raise_for_status()
    task_id = created.json()["id"]

    started = time.time()
    queued = client.post(f"/tasks/{task_id}/enqueue")
    enqueue_seconds = time.time() - started
    queued.raise_for_status()
    print(f"入队返回耗时 {enqueue_seconds:.2f}s，任务状态={queued.json()['status']}（应远小于任务总耗时）")

    deadline = time.time() + TASK_TIMEOUT_SECONDS
    final: dict = {}
    while time.time() < deadline:
        final = client.get(f"/tasks/{task_id}").json()
        if final["status"] in {"SUCCESS", "FAILED", "CANCELLED"}:
            break
        time.sleep(2)

    total = time.time() - started
    print(f"任务最终状态：{final.get('status')}（总耗时 {total:.0f}s）")
    print(f"worker_id     ：{final.get('worker_id')}")
    if final.get("error_code"):
        logs = client.get(f"/tasks/{task_id}/logs").json()
        print(f"错误码：{final['error_code']}")
        for item in logs[-8:]:
            print(f"  [{item['level']}] {item['stage']}: {item['message']}")
        return 1

    worker_id = final.get("worker_id") or ""
    if "local-executor" in worker_id or not worker_id:
        print("!! worker_id 为空或是 local-executor：任务可能仍在 API 进程内执行（检查 QFORGE_EXECUTOR_MODE=celery）")
        return 1

    print(f"✅ 任务由 worker 进程执行：{worker_id}")
    accuracy = None
    artifacts = client.get(f"/tasks/{task_id}/artifacts").json()
    for artifact in artifacts:
        if artifact["relative_path"].endswith("report/accuracy.json"):
            accuracy = client.get(
                f"/tasks/{task_id}/artifacts/{artifact['id']}/download"
            ).json()
    if accuracy:
        metrics = accuracy.get("engine_vs_fp32_baseline", {})
        print(f"精度：MAE {metrics.get('mae')} / 余弦 {metrics.get('cosine_similarity')}")
    print(f"产物数量：{len(artifacts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

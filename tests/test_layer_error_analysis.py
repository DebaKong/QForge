"""分层误差分析（SPEC 9.2）：指标、排序、图采集、真实 ONNX 联动、失败降级（纯 CPU）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.models.artifact import Artifact
from app.models.task import Task
from app.services import layer_error_analysis as analysis
from app.services.task_state import TaskStatus
from tools.synth_model import build_yolov8_like_onnx

METRIC_KEYS = ("mae", "mse", "rmse", "max_abs_error", "cosine_similarity", "relative_rmse")


def _synthetic_onnx(tmp_path: Path, size: int = 64) -> Path:
    path = tmp_path / "synthetic.onnx"
    build_yolov8_like_onnx(path, input_size=size)
    return path


def _samples(count: int = 2, size: int = 64) -> list[np.ndarray]:
    rng = np.random.default_rng(seed=7)
    return [rng.random((1, 3, size, size), dtype=np.float32) for _ in range(count)]


def test_compare_arrays_metrics_are_exact() -> None:
    reference = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    actual = np.array([0.0, 1.5, 1.0], dtype=np.float32)

    metrics = analysis.compare_arrays(reference, actual)

    assert metrics["mae"] == pytest.approx((0.0 + 0.5 + 1.0) / 3, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.0 + 0.25 + 1.0) / 3), abs=1e-6)
    assert metrics["max_abs_error"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["cosine_similarity"] == pytest.approx(
        float(np.dot(reference, actual) / (np.linalg.norm(reference) * np.linalg.norm(actual))),
        rel=1e-6,
    )
    assert metrics["baseline_abs_max"] == pytest.approx(2.0)


def test_compare_arrays_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        analysis.compare_arrays(np.zeros(3), np.zeros(4))


def test_relative_rmse_makes_layers_comparable() -> None:
    """相对 RMSE 才能跨层比较：量纲大的层绝对误差大，但不一定更敏感。"""
    big_scale = analysis.compare_arrays(np.full(100, 1000.0), np.full(100, 1002.0))
    small_scale = analysis.compare_arrays(np.full(100, 1.0), np.full(100, 1.2))

    assert big_scale["rmse"] > small_scale["rmse"]  # 绝对误差：大
    assert big_scale["relative_rmse"] < small_scale["relative_rmse"]  # 相对误差：小 → 更不敏感


def test_select_candidate_layers_respects_limits_and_types(tmp_path: Path) -> None:
    import onnx

    model = onnx.load(str(_synthetic_onnx(tmp_path)))
    candidates = analysis.select_candidate_layers(model, max_layers=10)

    assert candidates, "合成模型应至少有一个可观测的 float 中间张量"
    assert len(candidates) <= 10
    assert all(isinstance(name, str) and isinstance(op, str) for name, op, _ in candidates)


def test_instrument_model_appends_outputs_without_changing_graph(tmp_path: Path) -> None:
    import onnx

    model = onnx.load(str(_synthetic_onnx(tmp_path)))
    original_outputs = [item.name for item in model.graph.output]
    original_nodes = len(model.graph.node)
    names = [item[0] for item in analysis.select_candidate_layers(model, max_layers=3)]

    instrumented = analysis._instrument_model(model, names)  # noqa: SLF001

    assert [item.name for item in instrumented.graph.output][: len(original_outputs)] == (
        original_outputs
    ), "原有输出必须保持不变"
    # 新增的输出 = 候选里原本不是图输出的那些（已经是输出的不重复添加）
    expected_added = [name for name in names if name not in original_outputs]
    assert len(instrumented.graph.output) == len(original_outputs) + len(expected_added)
    assert all(
        name in [item.name for item in instrumented.graph.output] for name in names
    ), "所有候选张量都必须在输出里，否则拿不到逐层数据"
    assert len(instrumented.graph.node) == original_nodes, "不能改变原有算子"


def test_no_samples_is_skipped(tmp_path: Path) -> None:
    result = analysis.analyze(_synthetic_onnx(tmp_path), [], input_name="images")
    assert result.status == "SKIPPED"
    assert result.reason


def test_quantization_failure_degrades_to_blocked(tmp_path: Path, monkeypatch) -> None:
    """量化失败不能让任务失败：必须是 BLOCKED + 可读原因。"""
    import onnxruntime.quantization as ort_quant

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("模拟量化失败")

    monkeypatch.setattr(ort_quant, "quantize_static", _boom, raising=True)

    result = analysis.analyze(
        _synthetic_onnx(tmp_path), _samples(), input_name="images", max_layers=5
    )

    assert result.status == "BLOCKED"
    assert "量化失败" in result.reason
    assert result.notes, "BLOCKED 时要说明这不影响任务结论"


def test_real_onnx_layer_analysis_runs(tmp_path: Path) -> None:
    """真实跑一遍：FP32 基线 vs INT8 静态量化，逐层指标 + 敏感层排序。"""
    onnx_path = _synthetic_onnx(tmp_path)

    result = analysis.analyze(onnx_path, _samples(), input_name="images", max_layers=12)

    assert result.status in {"SUCCESS", "BLOCKED"}, result.reason
    if result.status == "BLOCKED":
        # 环境不支持量化时也要给出可读原因（不能静默）
        assert result.reason
        return

    assert result.analyzed_layers > 0
    assert result.requested_layers >= result.analyzed_layers
    assert result.inputs_used == 2
    for layer in result.layers:
        assert layer["operator"] and layer["tensor"]
        for key in METRIC_KEYS:
            assert key in layer, f"缺少指标 {key}"
    # 排序必须是相对 RMSE 降序（SPEC 9.2：误差排序定位敏感层）
    relative = [layer["relative_rmse"] for layer in result.layers]
    assert relative == sorted(relative, reverse=True)
    assert result.ranking[0]["rank"] == 1
    assert len(result.ranking) <= analysis.TOP_SENSITIVE


def _seed_report(session, storage_root: Path) -> tuple[str, str]:
    task = Task(
        task_type="detection",
        precision="int8",
        backend_name="tensorrt",
        status=TaskStatus.SUCCESS,
    )
    session.add(task)
    session.flush()
    report_dir = storage_root / task.id / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    accuracy = {
        "task_id": task.id,
        "precision": "int8",
        # 真实结构（build.py 写入 accuracy.json 的形状）：指标嵌套在两条链路下
        "engine_vs_fp32_baseline": {"baseline": "onnxruntime CPU FP32", "mae": 0.01, "rmse": 0.2},
        "cpp_program_vs_fp32_baseline": {"mae": 0.02, "rmse": 0.3, "cosine_similarity": 0.99},
        "note": "FP32 基准 = onnxruntime CPU 推理",
    }
    layers = {
        "task_id": task.id,
        "status": "SUCCESS",
        "analyzed_layers": 2,
        "ranking": [{"rank": 1, "operator": "Conv", "relative_rmse": 0.2}],
    }
    for name, payload in (("accuracy.json", accuracy), ("layer_error_analysis.json", layers)):
        (report_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        session.add(
            Artifact(
                task_id=task.id,
                kind="report",
                relative_path=f"{task.id}/report/{name}",
                size_bytes=1,
            )
        )
    session.commit()
    return task.id, task.id


def test_precision_route_returns_both_reports(client, db_session, storage_root: Path) -> None:
    task_id, _ = _seed_report(db_session, storage_root)

    response = client.get(f"/api/tasks/{task_id}/precision")

    assert response.status_code == 200, response.text
    body = response.json()
    # 端到端指标在嵌套结构里（与真实 accuracy.json 一致）
    assert body["accuracy"]["engine_vs_fp32_baseline"]["mae"] == pytest.approx(0.01)
    assert body["accuracy"]["cpp_program_vs_fp32_baseline"]["rmse"] == pytest.approx(0.3)
    assert body["layer_error_analysis"]["ranking"][0]["operator"] == "Conv"


def test_precision_route_501_when_missing(client, db_session) -> None:
    task = Task(
        task_type="detection",
        precision="fp16",
        backend_name="tensorrt",
        status=TaskStatus.CREATED,
    )
    db_session.add(task)
    db_session.commit()

    response = client.get(f"/api/tasks/{task.id}/precision")

    assert response.status_code == 501
    assert "精度报告" in response.json()["message"]

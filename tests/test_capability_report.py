"""算子兼容性报告（SPEC 7.1⑥ / 7.2）：能力注册表、字段完整性、真实模型联动（纯 CPU）。"""

from __future__ import annotations

from pathlib import Path

from app.adapters.backends import tensorrt_capabilities
from app.adapters.backends import tensorrt_adapter  # noqa: F401  导入以注册后端适配器
from app.adapters.registry import get_backend_adapter
from app.services import onnx_inspector
from tools.synth_model import build_yolov8_like_onnx

# SPEC 7.2 规定的报告字段
SPEC_FIELDS = ("operator", "domain", "opset", "supported", "condition", "severity", "suggestion")
VALID_SEVERITIES = {"info", "warning", "error"}


def _ops(*pairs: tuple[str, int]) -> list[dict]:
    return [
        {"operator": operator, "domain": "ai.onnx", "count": count} for operator, count in pairs
    ]


def test_unsupported_operator_is_reported_as_error() -> None:
    report = tensorrt_capabilities.evaluate(_ops(("RandomNormal", 1)), opset=17)
    item = report.items[0]

    assert item["supported"] is False
    assert item["severity"] == "error"
    assert item["suggestion"], "不支持时必须给出替换建议，而不是只说不行"
    assert report.summary["verdict"] == "BLOCKED"
    assert report.summary["unsupported"] == ["RandomNormal"]


def test_conditional_operator_is_warning_with_condition() -> None:
    report = tensorrt_capabilities.evaluate(_ops(("TopK", 2)), opset=17)
    item = report.items[0]

    assert item["supported"] is True
    assert item["severity"] == "warning"
    assert "K 必须是常量" in item["condition"]
    assert report.summary["verdict"] == "WARNINGS"
    assert report.summary["conditional"] == ["TopK"]


def test_opset_gate_turns_conditional_operator_into_error() -> None:
    """NonMaxSuppression 需要 opset>=11：低 opset 必须判为不可用，而不是放过。"""
    report = tensorrt_capabilities.evaluate(_ops(("NonMaxSuppression", 1)), opset=10)
    item = report.items[0]

    assert item["supported"] is False
    assert item["severity"] == "error"
    assert "opset" in item["condition"]
    assert report.summary["verdict"] == "BLOCKED"


def test_unregistered_operator_follows_parser_result() -> None:
    """未登记算子不臆断：解析通过按支持，解析失败按错误。"""
    accepted = tensorrt_capabilities.evaluate(_ops(("SomeFutureOp", 1)), parser_accepted=True)
    assert accepted.items[0]["registered"] is False
    assert accepted.items[0]["severity"] == "info"
    assert accepted.summary["verdict"] == "OK"

    failed = tensorrt_capabilities.evaluate(_ops(("SomeFutureOp", 1)), parser_accepted=False)
    assert failed.items[0]["severity"] == "error"
    assert "支持矩阵" in failed.items[0]["suggestion"]
    assert failed.summary["verdict"] == "BLOCKED"


def test_every_item_has_spec_fields() -> None:
    """SPEC 7.2 的七个字段必须在每条记录里都存在（缺字段等于报告不可用）。"""
    report = tensorrt_capabilities.evaluate(
        _ops(("Conv", 10), ("TopK", 1), ("RandomUniform", 1), ("MysteryOp", 1)), opset=17
    )

    assert report.items, "报告不能为空"
    for item in report.items:
        for field in SPEC_FIELDS:
            assert field in item, f"缺少 SPEC 7.2 字段：{field}"
        assert item["severity"] in VALID_SEVERITIES
    # 问题算子排在前面（error → warning → info）
    severities = [item["severity"] for item in report.items]
    assert severities == sorted(
        severities, key=lambda value: tensorrt_capabilities.SEVERITY_RANK[value]
    )
    assert report.summary["operator_instances"] == 13


def test_registry_entries_are_self_consistent() -> None:
    """登记表自身要自洽：不支持必须给建议，有条件必须写清条件。"""
    for operator in tensorrt_capabilities.registered_operators():
        report = tensorrt_capabilities.evaluate(_ops((operator, 1)), opset=17)
        item = report.items[0]
        if not item["supported"]:
            assert item["suggestion"], f"{operator} 标为不支持却没有建议"
        if item["condition"]:
            assert item["severity"] in {"info", "warning", "error"}


def test_real_synthetic_model_is_fully_reported(tmp_path: Path) -> None:
    """对真实生成的 ONNX 跑一遍：每个算子都要有结论字段（不依赖 GPU/TRT）。"""
    onnx_path = tmp_path / "synthetic.onnx"
    build_yolov8_like_onnx(onnx_path, input_size=64)

    inspection = onnx_inspector.inspect(onnx_path).to_dict()
    report = tensorrt_capabilities.evaluate(
        inspection["operators"], opset=inspection.get("opset"), parser_accepted=True
    )

    assert report.summary["operator_kinds"] == len(inspection["operators"])
    assert report.summary["operator_kinds"] >= 2, inspection["operators"]
    assert report.summary["opset"], "必须记录 opset（SPEC 7.2 字段之一）"
    assert report.summary["verdict"] in {"OK", "WARNINGS"}
    for item in report.items:
        assert item["operator"] and item["domain"]
        assert item["opset"] == inspection["opset"]


def test_adapter_exposes_capability_report_without_backend_coupling() -> None:
    """能力表通过 Backend Adapter 暴露，通用流水线里不需要写 if backend == ..."""
    adapter = get_backend_adapter("tensorrt")
    inspection = {
        "opset": 17,
        "operators": [
            {"operator": "Conv", "domain": "ai.onnx", "count": 3},
            {"operator": "TopK", "domain": "ai.onnx", "count": 1},
        ],
    }
    report = adapter.operator_capabilities(
        inspection, {"parsed": True, "backend_version": "10.16.1.11"}
    )

    assert report["summary"]["backend"] == "tensorrt"
    assert report["summary"]["backend_version"] == "10.16.1.11"
    assert report["summary"]["verdict"] == "WARNINGS"
    assert {item["operator"] for item in report["operators"]} == {"Conv", "TopK"}

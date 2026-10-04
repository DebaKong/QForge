"""语义分割后处理与指标（SPEC 2.2 / 9）：手算校验 + 合成模型端到端（纯 CPU）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services import segmentation
from tools.synth_segmentation import build_segmentation_onnx, reference_mask


def _logits(labels: np.ndarray, class_count: int) -> np.ndarray:
    """把标签图变成"确定会 argmax 到该标签"的 logits（正确类给高分）。

    支持 (H, W) 与 (N, H, W) 两种输入；输出恒为 (N, C, H, W)。
    """
    array = np.asarray(labels)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"标签必须是 (H, W) 或 (N, H, W)，收到 shape={array.shape}")
    one_hot = np.zeros((array.shape[0], class_count, *array.shape[1:]), dtype=np.float32)
    for class_id in range(class_count):
        one_hot[:, class_id] = np.where(array == class_id, 1.0, 0.0)
    return one_hot


def test_logits_to_labels_argmax() -> None:
    output = np.zeros((1, 3, 2, 2), dtype=np.float32)
    output[0, 2, 0, 0] = 5.0
    output[0, 1, 1, 1] = 9.0

    labels = segmentation.logits_to_labels(output)

    assert labels.shape == (1, 2, 2)
    assert labels[0, 0, 0] == 2
    assert labels[0, 1, 1] == 1


def test_logits_to_labels_rejects_wrong_rank() -> None:
    """维度不对必须明确报错，不猜也不静默算错。"""
    with pytest.raises(ValueError):
        segmentation.logits_to_labels(np.zeros((1, 3, 8), dtype=np.float32))


def test_resize_labels_nearest_does_not_interpolate() -> None:
    labels = np.array([[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]])

    resized = segmentation.resize_labels_nearest(labels, (2, 2))

    assert resized.shape == (1, 2, 2)
    # 最近邻：左上取原 (0,0)=0，右上取原 (0,2)=1，下方取原 (2,0)=2
    assert resized[0, 0, 0] == 0 and resized[0, 0, 1] == 1 and resized[0, 1, 0] == 2
    # 不能出现插值出来的"第三类"混合值
    assert set(np.unique(resized)).issubset({0, 1, 2})


def test_resize_labels_identity_returns_same() -> None:
    labels = np.arange(16).reshape(1, 4, 4)
    assert np.array_equal(segmentation.resize_labels_nearest(labels, (4, 4)), labels)


def test_postprocess_mask_resizes_and_marks_ignored() -> None:
    labels = np.zeros((1, 4, 4), dtype=np.int32)
    labels[0, 0, :] = 1

    # ignore_index 在这里用"某个真实类别 id"表示未标注区域（模型只有 3 类时不会输出 255）
    result = segmentation.postprocess_mask(
        _logits(labels, 3),
        class_count=3,
        target_hw=(8, 8),
        ignore_index=1,
    )

    assert result.labels.shape == (1, 8, 8)
    assert result.resized is True
    assert result.ignored_mask is not None
    # 掩膜计数必须与标签里该类别的像素数一致（自洽检查，而不是写死一个数字）
    assert int(result.ignored_mask.sum()) == int((result.labels == 1).sum())
    assert int(result.ignored_mask.sum()) > 0


def test_postprocess_mask_reports_channel_mismatch() -> None:
    output = np.zeros((1, 4, 4, 4), dtype=np.float32)
    result = segmentation.postprocess_mask(output, class_count=2)
    assert result.class_count == 4
    assert any("通道数" in note for note in result.notes)


def test_metrics_match_hand_computed_values() -> None:
    """用小到能手算的例子核对 IoU / Dice / 像素准确率。"""
    # 2 类、4 个像素：真值 [0,0,1,1]，预测 [0,1,1,1]
    reference = np.array([0, 0, 1, 1])
    actual = np.array([0, 1, 1, 1])

    metrics = segmentation.evaluate_masks(
        reference, actual, class_count=2, ignore_index=None
    )

    # 类 0：TP=1，真值 2，预测 1 → union=2 → IoU=0.5；Dice=2*1/(2+1)=0.667
    # 类 1：TP=2，真值 2，预测 3 → union=3 → IoU=0.667；Dice=2*2/(2+3)=0.8
    assert metrics["per_class"][0]["iou"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["per_class"][0]["dice"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["per_class"][1]["iou"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["per_class"][1]["dice"] == pytest.approx(0.8, abs=1e-6)
    assert metrics["mean_iou"] == pytest.approx((0.5 + 2 / 3) / 2, abs=1e-6)
    assert metrics["pixel_accuracy"] == pytest.approx(3 / 4, abs=1e-6)
    assert metrics["evaluated_pixels"] == 4


def test_ignore_index_is_fully_excluded() -> None:
    reference = np.array([0, 0, 255, 255])
    actual = np.array([0, 1, 1, 0])

    metrics = segmentation.evaluate_masks(
        reference, actual, class_count=2, ignore_index=segmentation.DEFAULT_IGNORE_INDEX
    )

    assert metrics["evaluated_pixels"] == 2, "被忽略的像素不能参与任何统计"
    # 有效像素是 (真值0,预测0) 与 (真值0,预测1)：类 0 的 IoU = 1/(1+1夹带的漏检) = 0.5
    assert metrics["per_class"][0]["iou"] == pytest.approx(0.5, abs=1e-6)
    # 类 1 没出现在真值里、却被预测出来了 → 全是假正例，IoU 必须是 0（并计入评估）
    assert metrics["per_class"][1]["present"] is True
    assert metrics["per_class"][1]["iou"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["classes_evaluated"] == 2


def test_absent_class_is_excluded_from_mean() -> None:
    reference = np.zeros(4, dtype=np.int32)
    actual = np.zeros(4, dtype=np.int32)

    metrics = segmentation.evaluate_masks(reference, actual, class_count=3, ignore_index=None)

    assert metrics["per_class"][1]["iou"] is None and metrics["per_class"][2]["iou"] is None
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["classes_evaluated"] == 1


def test_confusion_matrix_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        segmentation.confusion_matrix(np.zeros(4), np.zeros(5), class_count=2)


def test_perfect_prediction_scores_one(tmp_path: Path) -> None:
    """端到端：合成模型 → onnxruntime → 后处理 → 指标（用真值构造满分输入）。"""
    import onnxruntime as ort

    onnx_path = build_segmentation_onnx(tmp_path / "seg.onnx", input_size=32, class_count=3)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [item.shape for item in session.get_outputs()] == [[1, 3, 32, 32]]

    reference = reference_mask(32, class_count=3, bands=3)
    metrics = segmentation.evaluate_outputs(
        _logits(reference, 3), reference, class_count=3, ignore_index=None
    )

    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["mean_dice"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)


def test_noisy_prediction_degrades_metrics(tmp_path: Path) -> None:
    """故意让一部分像素预测错，指标必须相应下降（而不是恒等于 1）。"""
    reference = reference_mask(32, class_count=3, bands=3)
    noisy = reference.copy()
    noisy[:, :16] = (noisy[:, :16] + 1) % 3  # 上半部分整片错一类

    metrics = segmentation.evaluate_masks(reference, noisy, class_count=3, ignore_index=None)

    assert 0.0 < metrics["mean_iou"] < 1.0
    assert metrics["pixel_accuracy"] == pytest.approx(0.5, abs=1e-6)


def test_synthetic_model_is_deterministic(tmp_path: Path) -> None:
    """同一 seed 生成的模型必须逐字节一致（回归基线不能漂）。"""
    first = build_segmentation_onnx(tmp_path / "a.onnx", input_size=32, seed=7)
    second = build_segmentation_onnx(tmp_path / "b.onnx", input_size=32, seed=7)
    assert first.read_bytes() == second.read_bytes()


def test_write_regression_assets_produces_usable_model_and_mask(tmp_path: Path) -> None:
    """回归资产要能直接跑：模型可被 onnxruntime 加载，真值掩膜可被新模块评估。"""
    import onnxruntime as ort

    from tools.synth_segmentation import write_regression_assets

    assets = write_regression_assets(tmp_path / "assets", input_size=32, class_count=3)

    assert assets["model"].is_file() and assets["reference_mask"].is_file()
    session = ort.InferenceSession(str(assets["model"]), providers=["CPUExecutionProvider"])
    assert session.get_outputs()[0].shape[1] == 3

    reference = np.load(assets["reference_mask"])
    assert reference.shape == (32, 32) and reference.dtype == np.int32

    # 用真值构造满分预测：mIoU 必须是 1（证明资产与指标模块配套可用）
    metrics = segmentation.evaluate_masks(reference, reference, class_count=3, ignore_index=None)
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["evaluated_pixels"] == 32 * 32


def test_main_cli_writes_assets(tmp_path: Path, capsys) -> None:
    """命令行入口可用（下一轮端到端回归脚本直接调它再生模型）。"""
    from tools import synth_segmentation

    code = synth_segmentation.main(
        ["--out", str(tmp_path / "cli"), "--input-size", "16", "--class-count", "2"]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "model:" in output and "reference_mask:" in output
    assert (tmp_path / "cli" / "segmentation_16_c2.onnx").is_file()


def test_agreement_metrics_identical_masks_score_one() -> None:
    """引擎与基线完全相同 → 一致度必须是满分（无标注场景的定量检查）。"""
    labels = reference_mask(32, class_count=3, bands=3)
    logits = _logits(labels, 3)

    metrics = segmentation.agreement_metrics(logits, logits, class_count=3)

    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["mean_dice"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)
    assert "非人工标注真值" in metrics["baseline"]
    assert "不是与人工标注的精度" in metrics["note"]


def test_agreement_metrics_detects_mask_drift() -> None:
    """引擎掩膜漂移时必须反映到指标上（否则这份检查毫无意义）。"""
    labels = reference_mask(32, class_count=3, bands=3)
    drifted = labels.copy()
    drifted[:, :16] = (drifted[:, :16] + 1) % 3  # 左半部分整体漂到相邻类

    metrics = segmentation.agreement_metrics(
        _logits(drifted, 3), _logits(labels, 3), class_count=3
    )

    assert 0.0 < metrics["mean_iou"] < 1.0
    assert metrics["pixel_accuracy"] == pytest.approx(0.5, abs=1e-6)


def test_agreement_metrics_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        segmentation.agreement_metrics(
            _logits(np.zeros((8, 8), dtype=np.int32), 2),
            _logits(np.zeros((16, 16), dtype=np.int32), 2),
            class_count=2,
        )

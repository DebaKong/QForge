"""存储布局与路径安全测试（SPEC 14.2 / 15）。

这些用例是「上传一律视为不可信」的第一道防线：任何穿越、绝对路径、
保留名或非法片段都必须被拒绝。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import PathSecurityError
from app.services.storage import (
    TASK_SUBDIRS,
    ensure_task_layout,
    is_within,
    relative_to_storage,
    safe_join,
    sanitize_filename,
    task_dir,
    task_file,
    validate_segment,
)


def test_ensure_task_layout_creates_spec_directories(tmp_path: Path) -> None:
    layout = ensure_task_layout(tmp_path, "abc123")
    assert set(layout) == set(TASK_SUBDIRS)
    for name, path in layout.items():
        assert path.is_dir(), name
    assert (tmp_path / "abc123").is_dir()
    # 幂等：重复调用不报错
    assert ensure_task_layout(tmp_path, "abc123")["report"].is_dir()


@pytest.mark.parametrize(
    "segment",
    ["..", ".", "", "a/b", "a\\b", "C:", "a:b", "a*b", "a?b", "a|b", "a\x00b", "-abc", ".hidden"],
)
def test_validate_segment_rejects_unsafe_values(segment: str) -> None:
    with pytest.raises(PathSecurityError):
        validate_segment(segment)


@pytest.mark.parametrize("segment", ["abc", "task_1", "a.b-c", "0", "T" * 128])
def test_validate_segment_accepts_safe_values(segment: str) -> None:
    assert validate_segment(segment) == segment


def test_validate_segment_rejects_reserved_windows_names() -> None:
    for name in ("CON", "con", "NUL", "COM1", "LPT9", "con.txt"):
        with pytest.raises(PathSecurityError):
            validate_segment(name)


def test_safe_join_blocks_escape(tmp_path: Path) -> None:
    with pytest.raises(PathSecurityError):
        safe_join(tmp_path, "..")
    with pytest.raises(PathSecurityError):
        safe_join(tmp_path, "a", "..", "b")  # ".." 在片段校验阶段即被拒绝
    with pytest.raises(PathSecurityError):
        safe_join(tmp_path, "sub/../../outside")


def test_safe_join_keeps_paths_inside_root(tmp_path: Path) -> None:
    target = safe_join(tmp_path, "task1", "input")
    assert is_within(tmp_path, target)
    assert target == (tmp_path / "task1" / "input").resolve()


def test_task_dir_uses_task_id_only(tmp_path: Path) -> None:
    assert task_dir(tmp_path, "deadbeef") == (tmp_path / "deadbeef").resolve()
    with pytest.raises(PathSecurityError):
        task_dir(tmp_path, "../escape")


def test_task_file_rejects_unknown_subdir(tmp_path: Path) -> None:
    with pytest.raises(PathSecurityError):
        task_file(tmp_path, "t1", "not-a-subdir", "a.txt")


def test_task_file_sanitizes_filename(tmp_path: Path) -> None:
    path = task_file(tmp_path, "t1", "input", "../../evil.onnx")
    assert path.name == "evil.onnx"
    assert path.parent == (tmp_path / "t1" / "input").resolve()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("model.onnx", "model.onnx"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\cmd.exe", "cmd.exe"),
        ("a/b/c.txt", "c.txt"),
        ("with space.onnx", "with_space.onnx"),
        ("bad:name?.onnx", "bad_name_.onnx"),
        ("图像-校准 001.jpg", "图像-校准_001.jpg"),
        ("/绝对/路径/model.onnx", "model.onnx"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_keeps_unicode_names() -> None:
    assert sanitize_filename("测试图片.jpg") == "测试图片.jpg"


def test_sanitize_filename_truncates_but_keeps_suffix() -> None:
    result = sanitize_filename("x" * 400 + ".onnx")
    assert len(result) <= 128
    assert result.endswith(".onnx")


def test_sanitize_filename_prefixes_reserved_name() -> None:
    assert sanitize_filename("CON") == "file_CON"
    assert sanitize_filename("con.txt") == "file_con.txt"


def test_sanitize_filename_rejects_empty() -> None:
    for raw in ("", "   ", "...", "/"):
        with pytest.raises(PathSecurityError):
            sanitize_filename(raw)


def test_relative_to_storage_returns_posix_relative_path(tmp_path: Path) -> None:
    nested = tmp_path / "t1" / "engine" / "model.engine"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"x")
    assert relative_to_storage(tmp_path, nested) == "t1/engine/model.engine"


def test_relative_to_storage_rejects_outside_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_bytes(b"x")
    with pytest.raises(PathSecurityError):
        relative_to_storage(tmp_path, outside)

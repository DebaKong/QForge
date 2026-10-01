"""上传与归档安全测试（SPEC 15）。

这些用例是「上传一律视为不可信」的验收：路径穿越、ZIP 炸弹、符号链接、
超大文件、非法扩展名都必须被拒绝，且拒绝原因要能说清楚。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.errors import (
    ArchiveBombSuspectedError,
    ArchiveInvalidError,
    UploadInvalidError,
    UploadTooLargeError,
)
from app.services import uploads
from app.services.uploads import save_upload


def _zip_bytes(entries: dict[str, bytes], *, symlinks: tuple[str, ...] = ()) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        for name in symlinks:
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o120777 << 16)  # 符号链接标志
            archive.writestr(info, b"../../etc/passwd")
    return buffer.getvalue()


# ---------------- save_upload ----------------


def test_save_upload_writes_file_and_hash(tmp_path: Path) -> None:
    saved = save_upload(
        io.BytesIO(b"onnx-bytes"),
        "model.onnx",
        tmp_path / "input",
        allowed_extensions=[".onnx"],
        max_bytes=1024,
        storage_root=tmp_path,
    )
    assert saved.path.read_bytes() == b"onnx-bytes"
    assert saved.size_bytes == 10
    assert len(saved.sha256) == 64
    assert saved.filename == "model.onnx"


def test_save_upload_rejects_wrong_extension(tmp_path: Path) -> None:
    with pytest.raises(UploadInvalidError):
        save_upload(
            io.BytesIO(b"x"),
            "model.pb",
            tmp_path,
            allowed_extensions=[".onnx"],
            max_bytes=1024,
            storage_root=tmp_path,
        )


def test_save_upload_rejects_missing_extension(tmp_path: Path) -> None:
    with pytest.raises(UploadInvalidError):
        save_upload(
            io.BytesIO(b"x"),
            "model",
            tmp_path,
            allowed_extensions=[".onnx"],
            max_bytes=1024,
            storage_root=tmp_path,
        )


def test_save_upload_enforces_size_limit_while_streaming(tmp_path: Path) -> None:
    payload = b"a" * 5000
    with pytest.raises(UploadTooLargeError) as excinfo:
        save_upload(
            io.BytesIO(payload),
            "big.onnx",
            tmp_path,
            allowed_extensions=[".onnx"],
            max_bytes=1000,
            storage_root=tmp_path,
        )
    assert excinfo.value.detail["limit_bytes"] == 1000
    # 半成品必须被删除
    assert list(tmp_path.glob("big*")) == []


def test_save_upload_rejects_empty_file(tmp_path: Path) -> None:
    with pytest.raises(UploadInvalidError):
        save_upload(
            io.BytesIO(b""),
            "empty.onnx",
            tmp_path,
            allowed_extensions=[".onnx"],
            max_bytes=1024,
            storage_root=tmp_path,
        )


def test_save_upload_sanitizes_traversal_filename(tmp_path: Path) -> None:
    saved = save_upload(
        io.BytesIO(b"data"),
        "../../../evil.onnx",
        tmp_path / "input",
        allowed_extensions=[".onnx"],
        max_bytes=1024,
        storage_root=tmp_path,
    )
    assert saved.path.name == "evil.onnx"
    assert saved.path.parent == (tmp_path / "input").resolve()


def test_save_upload_rejects_destination_outside_root(tmp_path: Path) -> None:
    from app.errors import PathSecurityError

    with pytest.raises(PathSecurityError):
        save_upload(
            io.BytesIO(b"data"),
            "model.onnx",
            tmp_path.parent / "outside",
            allowed_extensions=[".onnx"],
            max_bytes=1024,
            storage_root=tmp_path,
        )


# ---------------- extract_image_archive ----------------


def _extract(tmp_path: Path, entries: dict[str, bytes], **overrides: object) -> tuple[uploads.ArchiveReport, Path]:
    archive = tmp_path / "calibration.zip"
    archive.write_bytes(_zip_bytes(entries))
    destination = tmp_path / "images"
    options: dict[str, object] = {
        "storage_root": tmp_path,
        "allowed_image_extensions": [".jpg", ".png"],
        "max_entries": 100,
        "max_total_uncompressed_bytes": 10 * 1024 * 1024,
        "max_depth": 4,
        "max_images": 50,
    }
    options.update(overrides)
    report = uploads.extract_image_archive(archive, destination, **options)  # type: ignore[arg-type]
    return report, destination


def test_extract_archive_keeps_directory_structure(tmp_path: Path) -> None:
    report, destination = _extract(
        tmp_path, {"images/000001.jpg": b"jpeg", "images/sub/000002.png": b"png"}
    )
    assert report.image_count == 2
    assert report.extracted == ["images/000001.jpg", "images/sub/000002.png"]
    assert (destination / "images" / "000001.jpg").read_bytes() == b"jpeg"
    assert (destination / "images" / "sub" / "000002.png").exists()


def test_extract_archive_skips_non_image_entries(tmp_path: Path) -> None:
    report, destination = _extract(
        tmp_path, {"images/a.jpg": b"x", "notes.txt": b"n", "run.exe": b"m"}
    )
    assert report.image_count == 1
    reasons = {item["reason"] for item in report.skipped}
    assert reasons == {"extension"}
    assert not (destination / "run.exe").exists()


def test_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ArchiveInvalidError):
        _extract(tmp_path, {"../evil.jpg": b"x"})


def test_extract_archive_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ArchiveInvalidError):
        _extract(tmp_path, {"/abs/evil.jpg": b"x"})


def test_extract_archive_rejects_backslash_traversal(tmp_path: Path) -> None:
    with pytest.raises(ArchiveInvalidError):
        _extract(tmp_path, {"..\\evil.jpg": b"x"})


def test_extract_archive_rejects_drive_letter(tmp_path: Path) -> None:
    with pytest.raises(ArchiveInvalidError):
        _extract(tmp_path, {"C:/evil.jpg": b"x"})


def test_extract_archive_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "cal.zip"
    archive.write_bytes(_zip_bytes({"images/a.jpg": b"x"}, symlinks=("images/link.jpg",)))
    report = uploads.extract_image_archive(
        archive,
        tmp_path / "images",
        storage_root=tmp_path,
        allowed_image_extensions=[".jpg"],
        max_entries=100,
        max_total_uncompressed_bytes=1024 * 1024,
        max_depth=4,
        max_images=10,
    )
    assert report.image_count == 1
    assert any(item["reason"] == "symlink" for item in report.skipped)


def test_extract_archive_rejects_too_many_entries(tmp_path: Path) -> None:
    entries = {f"images/{index}.jpg": b"x" for index in range(11)}
    with pytest.raises(ArchiveBombSuspectedError):
        _extract(tmp_path, entries, max_entries=10)


def test_extract_archive_rejects_excessive_depth(tmp_path: Path) -> None:
    with pytest.raises(ArchiveBombSuspectedError):
        _extract(tmp_path, {"a/b/c/d/e/f.jpg": b"x"}, max_depth=3)


def test_extract_archive_rejects_declared_total_size(tmp_path: Path) -> None:
    with pytest.raises(ArchiveBombSuspectedError):
        _extract(tmp_path, {"images/a.jpg": b"y" * 4096}, max_total_uncompressed_bytes=1024)


def test_extract_archive_detects_suspicious_compression_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(uploads, "_COMPRESSION_RATIO_MIN_BYTES", 1024)
    monkeypatch.setattr(uploads, "_COMPRESSION_RATIO_LIMIT", 2)
    with pytest.raises(ArchiveBombSuspectedError):
        _extract(tmp_path, {"images/bomb.jpg": b"\0" * 65536})


def test_extract_archive_rejects_image_count_over_limit(tmp_path: Path) -> None:
    entries = {f"images/{index}.jpg": b"x" for index in range(5)}
    with pytest.raises(ArchiveBombSuspectedError):
        _extract(tmp_path, entries, max_images=3)


def test_extract_archive_rejects_non_zip(tmp_path: Path) -> None:
    archive = tmp_path / "not-a-zip.zip"
    archive.write_bytes(b"this is not a zip")
    with pytest.raises(ArchiveInvalidError):
        uploads.extract_image_archive(
            archive,
            tmp_path / "images",
            storage_root=tmp_path,
            allowed_image_extensions=[".jpg"],
            max_entries=10,
            max_total_uncompressed_bytes=1024,
            max_depth=2,
            max_images=5,
        )


def test_extract_archive_rejects_archive_without_images(tmp_path: Path) -> None:
    with pytest.raises(ArchiveInvalidError):
        _extract(tmp_path, {"readme.txt": b"no images here"})


def test_remove_tree_cleans_directory(tmp_path: Path) -> None:
    target = tmp_path / "victim"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "file.txt").write_text("x", encoding="utf-8")
    uploads.remove_tree(target)
    assert not target.exists()

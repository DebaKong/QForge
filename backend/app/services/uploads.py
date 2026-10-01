"""上传与归档处理（SPEC 8.2 / 15：上传一律视为不可信）。

防护要点（每一条都对应 SPEC 15 的风险行）：
1. 大小限制：**边读边计数**，不信任 Content-Length，超限立即中止并删除半成品；
2. 扩展名白名单：模型只收 .onnx，校准集只收图像后缀；
3. 路径穿越：ZIP 条目名逐段校验，禁止绝对路径、盘符、`..`、反斜杠与符号链接；
4. ZIP 炸弹：同时限制条目数、单条解压后大小、总解压后大小、目录深度与压缩比；
5. 绝不调用 `extractall`：逐条读取并按块写入，写入目标必须落在任务目录内。

所有落盘路径都经过 `app.services.storage` 的白名单校验，目录名只由 task_id 生成。
"""

from __future__ import annotations

import hashlib
import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from app.errors import (
    ArchiveBombSuspectedError,
    ArchiveInvalidError,
    UploadInvalidError,
    UploadTooLargeError,
)
from app.services.storage import (
    PathSecurityError,
    is_within,
    sanitize_filename,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB
# 压缩比阈值：解压后 / 压缩后 超过该值且解压后体积可观时判定为可疑炸弹
_COMPRESSION_RATIO_LIMIT = 200
_COMPRESSION_RATIO_MIN_BYTES = 64 * 1024 * 1024


@dataclass
class SavedUpload:
    """一次上传落盘后的结果。"""

    path: Path
    filename: str
    original_name: str
    size_bytes: int
    sha256: str


@dataclass
class ArchiveReport:
    """校准集压缩包处理结果。"""

    archive_path: Path
    extract_root: Path
    entry_count: int = 0
    extracted: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    total_uncompressed_bytes: int = 0

    @property
    def image_count(self) -> int:
        return len(self.extracted)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_extension(filename: str, allowed: list[str], *, field_name: str) -> str:
    suffix = Path(filename).suffix.lower()
    if not suffix:
        raise UploadInvalidError(
            f"{field_name}缺少扩展名", detail={"filename": filename, "allowed": allowed}
        )
    if suffix not in {item.lower() for item in allowed}:
        raise UploadInvalidError(
            f"{field_name}扩展名不被允许",
            detail={"filename": filename, "suffix": suffix, "allowed": allowed},
        )
    return suffix


def save_upload(
    stream: BinaryIO,
    original_name: str,
    dest_dir: Path,
    *,
    allowed_extensions: list[str],
    max_bytes: int,
    field_name: str = "上传文件",
    storage_root: Path | None = None,
) -> SavedUpload:
    """把上传流受限地写入 dest_dir，返回落盘信息。

    dest_dir 必须位于 storage_root 内（传入时校验），文件名经 sanitize_filename 清洗。
    """
    if storage_root is not None and not is_within(storage_root, dest_dir):
        raise PathSecurityError(
            "上传目标目录不在存储根目录内",
            detail={"destination": str(dest_dir), "storage_root": str(storage_root)},
        )

    _check_extension(original_name, allowed_extensions, field_name=field_name)
    safe_name = sanitize_filename(original_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / safe_name

    digest = hashlib.sha256()
    written = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise UploadTooLargeError(
                        f"{field_name}超过上限 {max_bytes // (1024 * 1024)} MB",
                        detail={
                            "filename": original_name,
                            "limit_bytes": max_bytes,
                            "received_bytes": written,
                        },
                    )
                digest.update(chunk)
                handle.write(chunk)
    except UploadTooLargeError:
        target.unlink(missing_ok=True)
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise UploadInvalidError(
            f"{field_name}写入失败：{exc}", detail={"filename": original_name}
        ) from exc

    if written == 0:
        target.unlink(missing_ok=True)
        raise UploadInvalidError(f"{field_name}为空文件", detail={"filename": original_name})

    return SavedUpload(
        path=target,
        filename=safe_name,
        original_name=original_name,
        size_bytes=written,
        sha256=digest.hexdigest(),
    )


def _entry_segments(name: str) -> list[str]:
    """把 ZIP 条目名拆成安全片段；不合法直接抛错。"""
    if not name or name.endswith("/"):
        return []  # 目录条目
    if "\\" in name:
        raise ArchiveInvalidError(
            "压缩包条目包含反斜杠，可能是为绕过检查构造的路径", detail={"entry": name}
        )
    if name.startswith("/") or name.startswith("\\"):
        raise ArchiveInvalidError("压缩包条目使用了绝对路径", detail={"entry": name})
    if len(name) > 1 and name[1] == ":":
        raise ArchiveInvalidError("压缩包条目包含盘符", detail={"entry": name})

    segments = [segment for segment in name.split("/") if segment not in ("", ".")]
    for segment in segments:
        if segment == "..":
            raise ArchiveInvalidError("压缩包条目包含路径穿越（..）", detail={"entry": name})
        try:
            sanitize_filename(segment)
        except PathSecurityError as exc:
            raise ArchiveInvalidError(
                f"压缩包条目名不合法：{segment}", detail={"entry": name}
            ) from exc
    return segments


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    """ZIP 中符号链接的判定：外部属性高位记录了 Unix 文件模式。"""
    mode = info.external_attr >> 16
    return (mode & 0o170000) == 0o120000


def extract_image_archive(
    archive_path: Path,
    dest_dir: Path,
    *,
    storage_root: Path,
    allowed_image_extensions: list[str],
    max_entries: int,
    max_total_uncompressed_bytes: int,
    max_depth: int,
    max_images: int,
) -> ArchiveReport:
    """安全解压校准集压缩包（逐条校验 + 逐块写入，绝不使用 extractall）。"""
    if not is_within(storage_root, dest_dir):
        raise PathSecurityError(
            "解压目标目录不在存储根目录内",
            detail={"destination": str(dest_dir), "storage_root": str(storage_root)},
        )
    if not zipfile.is_zipfile(archive_path):
        raise ArchiveInvalidError(
            "校准集必须是 ZIP 压缩包（SPEC 8.2：calibration.zip -> images/）",
            detail={"archive": archive_path.name},
        )

    allowed_suffixes = {item.lower() for item in allowed_image_extensions}
    report = ArchiveReport(archive_path=archive_path, extract_root=dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        report.entry_count = len(infos)

        if len(infos) > max_entries:
            raise ArchiveBombSuspectedError(
                f"压缩包条目数 {len(infos)} 超过上限 {max_entries}",
                detail={"entries": len(infos), "max_entries": max_entries},
            )

        declared_total = sum(info.file_size for info in infos)
        if declared_total > max_total_uncompressed_bytes:
            raise ArchiveBombSuspectedError(
                "压缩包声明的解压后总大小超过上限",
                detail={
                    "declared_uncompressed_bytes": declared_total,
                    "limit_bytes": max_total_uncompressed_bytes,
                },
            )

        for info in infos:
            if info.is_dir():
                continue
            if _is_symlink(info):
                report.skipped.append({"entry": info.filename, "reason": "symlink"})
                continue

            segments = _entry_segments(info.filename)
            if not segments:
                continue
            if len(segments) > max_depth:
                raise ArchiveBombSuspectedError(
                    f"压缩包目录深度 {len(segments)} 超过上限 {max_depth}",
                    detail={"entry": info.filename, "max_depth": max_depth},
                )

            suffix = Path(segments[-1]).suffix.lower()
            if suffix not in allowed_suffixes:
                report.skipped.append({"entry": info.filename, "reason": "extension"})
                continue

            if len(report.extracted) >= max_images:
                raise ArchiveBombSuspectedError(
                    f"压缩包内图像数超过上限 {max_images}",
                    detail={"max_images": max_images},
                )

            # 压缩比检查：声明的解压后大小远大于压缩后大小
            if (
                info.compress_size > 0
                and info.file_size > _COMPRESSION_RATIO_MIN_BYTES
                and info.file_size / info.compress_size > _COMPRESSION_RATIO_LIMIT
            ):
                raise ArchiveBombSuspectedError(
                    "压缩包压缩比异常，疑似 ZIP 炸弹",
                    detail={
                        "entry": info.filename,
                        "compressed": info.compress_size,
                        "uncompressed": info.file_size,
                    },
                )

            relative = Path(*[sanitize_filename(segment) for segment in segments])
            target = (dest_dir / relative).resolve()
            if not is_within(dest_dir, target):
                raise ArchiveInvalidError(
                    "压缩包条目解压后落在目标目录之外", detail={"entry": info.filename}
                )
            target.parent.mkdir(parents=True, exist_ok=True)

            written = 0
            with archive.open(info) as source, target.open("wb") as sink:
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    report.total_uncompressed_bytes += len(chunk)
                    if report.total_uncompressed_bytes > max_total_uncompressed_bytes:
                        sink.close()
                        target.unlink(missing_ok=True)
                        raise ArchiveBombSuspectedError(
                            "解压后总大小超过上限（实际写入时校验）",
                            detail={
                                "limit_bytes": max_total_uncompressed_bytes,
                                "entry": info.filename,
                            },
                        )
                    sink.write(chunk)

            report.extracted.append(relative.as_posix())

    if not report.extracted:
        raise ArchiveInvalidError(
            "压缩包内没有可用的图像文件",
            detail={
                "entries": report.entry_count,
                "allowed": sorted(allowed_suffixes),
                "skipped": report.skipped[:20],
            },
        )

    report.extracted.sort()
    logger.info(
        "校准集解压完成",
        extra={
            "entries": report.entry_count,
            "images": report.image_count,
            "bytes": report.total_uncompressed_bytes,
            "skipped": len(report.skipped),
        },
    )
    return report


def remove_tree(path: Path) -> None:
    """删除目录树（用于失败回滚）。仅接受目录。"""
    if not path.exists() or not path.is_dir():
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            (Path(root) / name).unlink(missing_ok=True)
        for name in dirs:
            (Path(root) / name).rmdir()
    path.rmdir()

"""存储布局与路径安全（SPEC 14.2 / 15）。

原则：上传与外部输入一律视为不可信。任何来自外部（HTTP 参数、ZIP 条目、文件名）的
路径片段，必须经 `validate_segment` / `sanitize_filename` / `safe_join` 处理，
确保最终路径被限制在 `storage_root` 之内，杜绝路径穿越。

命名规则采用白名单：片段只允许字母、数字（含中文等 Unicode 字母）与 `.` `_` `-`，
且首字符必须是字母或数字。空白、控制字符、路径分隔符、盘符与 shell 元字符一律拒绝。

目录布局（SPEC 14.2）：

    storage/
    └── <task_id>/
        ├── input/
        ├── calibration/
        ├── intermediate/
        ├── engine/
        ├── source/
        ├── docker/
        ├── logs/
        └── report/
"""

from __future__ import annotations

from pathlib import Path

from app.errors import PathSecurityError

TASK_SUBDIRS: tuple[str, ...] = (
    "input",
    "calibration",
    "intermediate",
    "engine",
    "source",
    "docker",
    "logs",
    "report",
)

MAX_SEGMENT_LENGTH = 128

# 除字母数字外额外允许的字符
_EXTRA_ALLOWED_CHARS = frozenset("._-")

# Windows 保留设备名（即使带扩展名也不允许）
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _is_allowed_char(char: str) -> bool:
    """白名单判定：字母/数字（Unicode 感知）或 . _ -。"""
    return char.isalnum() or char in _EXTRA_ALLOWED_CHARS


def validate_segment(segment: str, *, field: str = "路径片段") -> str:
    """校验单个路径片段（不含分隔符），合法则原样返回，否则抛 PathSecurityError。"""
    if not isinstance(segment, str) or not segment:
        raise PathSecurityError(f"{field}不能为空", detail={"field": field, "value": segment})
    if segment in {".", ".."}:
        raise PathSecurityError(f"{field}不允许相对路径标记", detail={"value": segment})
    if len(segment) > MAX_SEGMENT_LENGTH:
        raise PathSecurityError(
            f"{field}过长（上限 {MAX_SEGMENT_LENGTH} 字符）",
            detail={"field": field, "length": len(segment)},
        )
    if not segment[0].isalnum():
        raise PathSecurityError(
            f"{field}必须以字母或数字开头", detail={"field": field, "value": segment}
        )
    if not all(_is_allowed_char(char) for char in segment):
        raise PathSecurityError(
            f"{field}包含不允许的字符（仅允许字母、数字、. _ -）",
            detail={"field": field, "value": segment},
        )
    if segment.split(".")[0].upper() in _RESERVED_NAMES:
        raise PathSecurityError(f"{field}使用了系统保留名", detail={"value": segment})
    return segment


def is_within(root: Path, candidate: Path) -> bool:
    """candidate 解析后是否位于 root 之内（root 自身返回 True）。"""
    root_resolved = Path(root).resolve()
    try:
        Path(candidate).resolve().relative_to(root_resolved)
    except ValueError:
        return False
    return True


def safe_join(root: Path, *segments: str) -> Path:
    """拼接路径并强制约束在 root 之内。任一片段非法或越界即抛异常。"""
    target = Path(root)
    for segment in segments:
        target = target / validate_segment(segment)
    resolved = target.resolve()
    if not is_within(root, resolved):
        raise PathSecurityError(
            "路径越界：目标不在存储根目录内",
            detail={"root": str(Path(root).resolve()), "target": str(resolved)},
        )
    return resolved


def task_dir(storage_root: Path, task_id: str) -> Path:
    """任务专属目录：`storage/<task_id>`。目录名只由 task_id 生成，不使用用户提供的名称。"""
    return safe_join(storage_root, task_id)


def ensure_task_layout(storage_root: Path, task_id: str) -> dict[str, Path]:
    """建立任务目录及全部子目录，返回子目录映射。已存在时不报错。"""
    base = task_dir(storage_root, task_id)
    base.mkdir(parents=True, exist_ok=True)
    layout: dict[str, Path] = {}
    for name in TASK_SUBDIRS:
        sub = base / name
        sub.mkdir(exist_ok=True)
        layout[name] = sub
    return layout


def task_file(storage_root: Path, task_id: str, subdir: str, filename: str) -> Path:
    """任务内文件路径：校验子目录与文件名后返回受限路径。"""
    if subdir not in TASK_SUBDIRS:
        raise PathSecurityError(
            "未授权的任务子目录", detail={"subdir": subdir, "allowed": list(TASK_SUBDIRS)}
        )
    base = task_dir(storage_root, task_id)
    return safe_join(base, subdir, sanitize_filename(filename))


def sanitize_filename(name: str, *, max_length: int = MAX_SEGMENT_LENGTH) -> str:
    """把不受信文件名压缩为安全的单段文件名。

    处理顺序：丢弃目录成分 -> 白名单替换 -> 去除前导分隔符类字符 -> 限长（保留扩展名）
    -> 规避系统保留名 -> 片段校验。
    """
    if not isinstance(name, str) or not name.strip():
        raise PathSecurityError("文件名不能为空", detail={"value": name})

    # 同时处理 / 与 \，只保留最后一段（且绝对路径中的盘符也会被丢弃）
    candidate = name.replace("\\", "/").split("/")[-1]

    cleaned = "".join(
        char if _is_allowed_char(char) else "_" for char in candidate
    ).strip().strip(".")

    # 首字符必须是字母或数字，否则逐个丢弃前缀
    while cleaned and not cleaned[0].isalnum():
        cleaned = cleaned[1:]

    if not cleaned:
        raise PathSecurityError("文件名清洗后为空", detail={"value": name})

    if len(cleaned) > max_length:
        stem, dot, suffix = cleaned.rpartition(".")
        if dot and 0 < len(suffix) <= 16:
            cleaned = f"{stem[: max_length - len(suffix) - 1]}.{suffix}"
        else:
            cleaned = cleaned[:max_length]

    if cleaned.split(".")[0].upper() in _RESERVED_NAMES:
        cleaned = f"file_{cleaned}"

    return validate_segment(cleaned, field="文件名")


def relative_to_storage(storage_root: Path, path: Path) -> str:
    """返回相对于存储根的 POSIX 风格相对路径，用于入库（避免保存宿主机绝对路径）。"""
    resolved = Path(path).resolve()
    if not is_within(storage_root, resolved):
        raise PathSecurityError(
            "目标不在存储根目录内，无法计算相对路径",
            detail={"root": str(Path(storage_root).resolve()), "target": str(resolved)},
        )
    return resolved.relative_to(Path(storage_root).resolve()).as_posix()

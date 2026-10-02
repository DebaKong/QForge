"""路径解析：同一份代码既能以「仓库检出」方式运行，也能以「已安装应用」方式运行。

用户要求（AGENTS.md 产品形态要求）：安装即用。因此数据与配置的默认位置必须自动适配：

- **仓库检出**（开发）：数据在 `<repo>/storage`，配置读 `<repo>/.env`（开发体验不变）；
- **pip 安装**（应用）：数据在用户数据目录
  （Windows `%LOCALAPPDATA%\\QForge`，Linux/macOS `~/.local/share/qforge`），配置读该目录下的 `qforge.env`；
- 两种情况都可以用 `QFORGE_DATA_DIR` 覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/app/paths.py -> app
PACKAGE_DIR = Path(__file__).resolve().parent


def project_root() -> Path | None:
    """当前是否为仓库检出：是则返回仓库根，否则返回 None。

    判定依据是仓库级标志文件（SPEC.md / .git），不依赖目录名，避免安装形态误判。
    """
    for candidate in (PACKAGE_DIR.parent.parent, PACKAGE_DIR.parent, PACKAGE_DIR):
        if (candidate / "SPEC.md").exists() or (candidate / ".git").exists():
            return candidate
    return None


def data_root() -> Path:
    """运行期数据根目录（任务产物、数据库、日志）。"""
    override = os.environ.get("QFORGE_DATA_DIR")
    if override:
        return Path(override).expanduser()

    root = project_root()
    if root is not None:
        return root / "storage"

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "QForge" / "storage"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "qforge" / "storage"


def config_file() -> Path:
    """配置文件路径（允许尚不存在，pydantic-settings 会忽略缺失文件）。"""
    root = project_root()
    if root is not None:
        return root / ".env"
    return data_root().parent / "qforge.env"


def frontend_dir() -> Path | None:
    """已构建的前端静态文件目录（存在才返回，否则由 API 提供降级页面）。"""
    root = project_root()
    if root is not None:
        built = root / "frontend" / "dist"
        if built.is_dir():
            return built
    installed = data_root().parent / "web"
    if installed.is_dir():
        return installed
    return None

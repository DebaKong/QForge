"""导入路径引导。

SPEC 附录 A 的目录结构中 `backend/` 与 `workers/` 同级，而 Worker 需要复用
`app.*`（配置、ORM、服务层）。这里在导入阶段把这两个目录加入 sys.path，
使得 `celery -A workers.celery_app worker` 可在仓库根直接运行，无需手工设置 PYTHONPATH。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"


def ensure_backend_on_path() -> None:
    for path in (str(BACKEND_DIR), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

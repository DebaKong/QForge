"""后端适配器实现集合。

注意：本包**不在导入期**导入 TensorRT 适配器，避免未安装 TensorRT 时
整个应用无法启动；由 `load_backend_adapter()` 按需导入。
"""

from __future__ import annotations

from app.adapters.base import BackendAdapter
from app.adapters.registry import get_backend_adapter


def load_backend_adapter(name: str) -> BackendAdapter:
    """按需导入并实例化后端适配器。"""
    key = (name or "").lower()
    if key == "tensorrt":
        from app.adapters.backends import tensorrt_adapter  # noqa: F401  触发注册

    return get_backend_adapter(key)


__all__ = ["load_backend_adapter"]

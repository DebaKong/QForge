"""适配器注册表（SPEC 19：新增模型/后端只加适配器，不改业务层）。"""

from __future__ import annotations

from app.adapters.base import BackendAdapter, ModelAdapter
from app.errors import ValidationFailedError

_MODEL_ADAPTERS: dict[str, type[ModelAdapter]] = {}
_BACKEND_ADAPTERS: dict[str, type[BackendAdapter]] = {}


def register_model_adapter(adapter_cls: type[ModelAdapter]) -> type[ModelAdapter]:
    key = adapter_cls.architecture.lower()
    if not key:
        raise ValueError("ModelAdapter 必须声明 architecture")
    _MODEL_ADAPTERS[key] = adapter_cls
    return adapter_cls


def register_backend_adapter(adapter_cls: type[BackendAdapter]) -> type[BackendAdapter]:
    key = adapter_cls.name.lower()
    if not key:
        raise ValueError("BackendAdapter 必须声明 name")
    _BACKEND_ADAPTERS[key] = adapter_cls
    return adapter_cls


def get_model_adapter(architecture: str) -> ModelAdapter:
    key = (architecture or "").lower()
    adapter_cls = _MODEL_ADAPTERS.get(key)
    if adapter_cls is None:
        raise ValidationFailedError(
            f"不支持的模型架构：{architecture or '(空)'}",
            detail={"architecture": architecture, "supported": sorted(_MODEL_ADAPTERS)},
        )
    return adapter_cls()


def get_backend_adapter(name: str) -> BackendAdapter:
    key = (name or "").lower()
    adapter_cls = _BACKEND_ADAPTERS.get(key)
    if adapter_cls is None:
        raise ValidationFailedError(
            f"不支持的后端：{name or '(空)'}",
            detail={"backend": name, "supported": sorted(_BACKEND_ADAPTERS)},
        )
    return adapter_cls()


def supported_architectures() -> list[str]:
    return sorted(_MODEL_ADAPTERS)


def supported_backends() -> list[str]:
    return sorted(_BACKEND_ADAPTERS)

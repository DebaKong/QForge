"""CUDA 驱动加载的跨平台用例。

背景：容器化部署（Linux 容器 + `--gpus all`）要求应用在 Linux 下也能加载驱动库，
原先只写了 `ctypes.WinDLL("nvcuda.dll")`，在 Linux 容器里会直接失败。
"""

from __future__ import annotations

import pytest

from app.adapters.backends import cuda_memory


def test_driver_library_name_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_memory.os, "name", "nt")
    assert cuda_memory.cuda_driver_library_name() == "nvcuda.dll"


def test_driver_library_name_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_memory.os, "name", "posix")
    assert cuda_memory.cuda_driver_library_name() == "libcuda.so.1"


def test_missing_driver_reports_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_memory, "cuda_driver_library_name", lambda: "definitely-not-a-real.so")
    info = cuda_memory.cuda_driver_info()
    assert info["available"] is False
    assert "definitely-not-a-real.so" in info["reason"]

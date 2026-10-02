"""CUDA 运行时辅助（显存分配 / 拷贝 / 流）。

TensorRT 的 Python API 需要**设备指针**：INT8 校准器要把每个 batch 拷到显存，
推理时输入输出缓冲区也必须在显存里。这里用 NVIDIA 官方 `cuda-python` 绑定实现，
避免引入需要本地编译的 pycuda。

兼容两种导入路径（不同版本 API 布局不同）：
- 新：`from cuda.bindings import runtime as cudart`
- 旧：`from cuda import cudart`

导入失败时抛出带说明的 BackendUnavailableError，提示安装 `cuda-python`。
"""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.errors import BackendUnavailableError

logger = logging.getLogger(__name__)


def _register_dll_directories() -> list[str]:
    """把工具链中的 CUDA / TensorRT DLL 目录加入 Windows DLL 搜索路径。

    pip 包不会注册这些目录，而 `cuda-python` 与 `nvinfer` 的运行库都在其中，
    因此导入绑定前必须先注册，否则会出现「找不到 cudart64_13.dll」这类错误。
    """
    added: list[str] = []
    registry = getattr(os, "add_dll_directory", None)
    if registry is None:  # pragma: no cover - 非 Windows
        return added
    try:
        from app.adapters.backends import tensorrt_dev
        from app.config.settings import get_settings

        files = tensorrt_dev.locate_dev_files(get_settings())
        for directory in files.runtime_dll_dirs:
            if not Path(directory).is_dir():
                continue
            try:
                registry(directory)
                added.append(directory)
            except OSError:  # pragma: no cover
                continue
    except Exception:  # pragma: no cover - 探测失败不应阻断导入
        logger.debug("注册 DLL 目录失败", exc_info=True)
    return added


def load_cudart() -> Any:
    """导入 cudart 绑定模块。"""
    _register_dll_directories()
    try:  # 新布局
        from cuda.bindings import runtime as cudart  # type: ignore

        return cudart
    except Exception:  # pragma: no cover - 取决于安装版本
        pass
    try:  # 旧布局
        from cuda import cudart  # type: ignore

        return cudart
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError(
            "缺少 CUDA Python 绑定（cuda-python），无法为 TensorRT 分配显存",
            detail={"hint": "pip install cuda-python", "error": f"{type(exc).__name__}: {exc}"},
        ) from exc


def _check(result: Any, action: str) -> Any:
    """把 cudart 的 (err, *values) 返回值转成抛异常或解包。"""
    if isinstance(result, tuple):
        error = result[0]
        values = result[1:]
    else:  # pragma: no cover - 极少数版本直接返回错误码
        error = result
        values = ()

    success = getattr(error, "name", str(error)) in ("cudaSuccess", "0")
    if not success:
        raise BackendUnavailableError(
            f"CUDA 调用失败：{action}", detail={"cuda_error": str(error)}
        )
    if not values:
        return None
    return values[0] if len(values) == 1 else values


@dataclass
class DeviceBuffer:
    """设备端缓冲。"""

    pointer: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: np.dtype

    def free(self, cudart: Any) -> None:
        if self.pointer:
            _check(cudart.cudaFree(self.pointer), "cudaFree")
            self.pointer = 0


def cuda_driver_library_name() -> str:
    """CUDA 驱动库名（跨平台）。

    Windows 用 `nvcuda.dll`；Linux 用 `libcuda.so.1`（由 NVIDIA 驱动/容器运行时提供）。
    容器里构建 Engine 就依赖这里的平台判断（Linux 容器 + `--gpus all`）。
    """
    return "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"


def load_cuda_driver() -> Any:
    """加载 CUDA 驱动库；失败抛 OSError 由调用方给出可读提示。"""
    return ctypes.CDLL(cuda_driver_library_name())


def cuda_driver_info() -> dict[str, Any]:
    """查询驱动与设备信息（SPEC 4.1：Engine 必须记录 GPU 架构与版本）。"""
    try:
        cuda = load_cuda_driver()
    except OSError as exc:  # pragma: no cover - 无驱动/无 GPU 环境
        return {
            "available": False,
            "reason": f"无法加载 {cuda_driver_library_name()}: {exc}",
        }

    def _call(name: str, *args: Any) -> int:
        return int(cuda[name](*args))

    if _call("cuInit", 0) != 0:
        return {"available": False, "reason": "cuInit 失败（无可用 NVIDIA 设备或驱动异常）"}

    count = ctypes.c_int(0)
    if _call("cuDeviceGetCount", ctypes.byref(count)) != 0 or count.value <= 0:
        return {"available": False, "reason": "未检测到 CUDA 设备"}

    driver_version = ctypes.c_int(0)
    _call("cuDriverGetVersion", ctypes.byref(driver_version))

    name_buffer = ctypes.create_string_buffer(128)
    _call("cuDeviceGetName", name_buffer, 128, 0)

    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    _call("cuDeviceComputeCapability", ctypes.byref(major), ctypes.byref(minor), 0)

    total_memory = ctypes.c_size_t(0)
    # CU_DEVICE_ATTRIBUTE_TOTAL_MEMORY = 2
    _call("cuDeviceTotalMem_v2", ctypes.byref(total_memory), 0)

    return {
        "available": True,
        "device_count": count.value,
        "device_name": name_buffer.value.decode("utf-8", errors="replace"),
        "compute_capability": f"{major.value}.{minor.value}",
        "gpu_arch": f"sm_{major.value}{minor.value}",
        "driver_version": f"{driver_version.value // 1000}.{(driver_version.value % 1000) // 10}",
        "total_memory_bytes": int(total_memory.value),
    }


def allocate(cudart: Any, array: np.ndarray) -> DeviceBuffer:
    """按 numpy 数组分配设备缓冲（不拷贝数据）。"""
    nbytes = int(array.nbytes)
    pointer = _check(cudart.cudaMalloc(nbytes), "cudaMalloc")
    return DeviceBuffer(
        pointer=int(pointer), nbytes=nbytes, shape=array.shape, dtype=array.dtype
    )


def _memcpy_kind(cudart: Any, name: str, fallback: int) -> Any:
    """解析 cudaMemcpyKind 枚举（不同 cuda-python 版本对字面量接受度不同）。"""
    kind_enum = getattr(cudart, "cudaMemcpyKind", None)
    if kind_enum is not None and hasattr(kind_enum, name):
        return getattr(kind_enum, name)
    return fallback


def copy_to_device(cudart: Any, buffer: DeviceBuffer, array: np.ndarray) -> None:
    if array.nbytes != buffer.nbytes:  # pragma: no cover - 由调用方保证
        raise BackendUnavailableError(
            "拷贝到显存时大小不一致",
            detail={"host_bytes": int(array.nbytes), "device_bytes": buffer.nbytes},
        )
    contiguous = np.ascontiguousarray(array)
    kind = _memcpy_kind(cudart, "cudaMemcpyHostToDevice", 1)
    _check(
        cudart.cudaMemcpy(buffer.pointer, contiguous.ctypes.data, buffer.nbytes, kind),
        "cudaMemcpy(host->device)",
    )


def copy_to_host(cudart: Any, buffer: DeviceBuffer) -> np.ndarray:
    host = np.empty(buffer.shape, dtype=buffer.dtype)
    kind = _memcpy_kind(cudart, "cudaMemcpyDeviceToHost", 2)
    _check(
        cudart.cudaMemcpy(host.ctypes.data, buffer.pointer, buffer.nbytes, kind),
        "cudaMemcpy(device->host)",
    )
    return host


def create_stream(cudart: Any) -> int:
    return int(_check(cudart.cudaStreamCreate(), "cudaStreamCreate"))


def destroy_stream(cudart: Any, stream: int) -> None:
    if stream:
        _check(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")


def synchronize(cudart: Any, stream: int) -> None:
    _check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

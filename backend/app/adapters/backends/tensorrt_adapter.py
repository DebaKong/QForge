"""TensorRT 后端适配器（SPEC 7.1 / 9 / 10 / 11.3）。

职责：
- `availability()`：环境自检（TensorRT 版本、CUDA 驱动、GPU 架构），不可用时给出可读原因；
- `capability_check()`：用真实 TensorRT ONNX Parser 解析模型，产出算子兼容性报告（SPEC 7.2）；
- `build_engine()`：构建 FP32 / FP16 / **INT8（静态熵校准）** Engine，并记录 engine_metadata；
- `run_inference()`：真实加载 Engine 并推理，用于自动运行验证（SPEC 11.3 步骤 4~6）。

INT8 说明（**已确认的设计变更**）：SPEC 4 / 9 原指定 PPQ，但 PPQ 最新版仍为 0.6.6（2023 年，
仅支持 Python 3.6~3.9、protobuf≤3.20，其 TensorRT 导出针对 TRT 8.4），与本机 RTX 5060（SM 12.0）
所需的 TensorRT 10/11 + CUDA 12.8 + protobuf 4+ 无法共存（实测 `import ppq` 直接失败）。
因此 INT8 静态量化改用 **TensorRT 自带的 INT8 熵校准（IInt8EntropyCalibrator2）**：
同样是「校准集 → 校准 → INT8 Engine」的静态量化流程，且是 NVIDIA 官方路径。
"""

from __future__ import annotations

import json
import logging
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.adapters.base import BackendAdapter, ModelAdapter
from app.adapters.backends import cuda_memory
from app.adapters.definition import ModelDefinition
from app.adapters.registry import register_backend_adapter
from app.errors import (
    BackendUnavailableError,
    EngineBuildFailedError,
    OperatorUnsupportedError,
    RuntimeTestFailedError,
)
from app.services.preprocess import preprocess_batches

logger = logging.getLogger(__name__)

MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2


def import_tensorrt() -> Any:
    """导入 tensorrt；未安装时给出明确错误（BACKEND_UNAVAILABLE）。"""
    try:
        import tensorrt as trt  # type: ignore

        return trt
    except Exception as exc:
        raise BackendUnavailableError(
            "未安装 TensorRT Python 包，无法构建/运行 Engine",
            detail={"hint": "pip install tensorrt", "error": f"{type(exc).__name__}: {exc}"},
        ) from exc


class _FileLogger:
    """把 TensorRT 日志同时写入 Python 日志与构建日志文件（SPEC 10.1：保留完整日志）。"""

    def __init__(self, trt: Any, log_path: Path | None) -> None:
        self._trt = trt
        self._log_path = log_path
        self._level_map = {
            trt.ILogger.INTERNAL_ERROR: logging.ERROR,
            trt.ILogger.ERROR: logging.ERROR,
            trt.ILogger.WARNING: logging.WARNING,
            trt.ILogger.INFO: logging.INFO,
            trt.ILogger.VERBOSE: logging.DEBUG,
        }
        self._severity_names = {
            trt.ILogger.INTERNAL_ERROR: "INTERNAL_ERROR",
            trt.ILogger.ERROR: "ERROR",
            trt.ILogger.WARNING: "WARNING",
            trt.ILogger.INFO: "INFO",
            trt.ILogger.VERBOSE: "VERBOSE",
        }
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = log_path.open("a", encoding="utf-8")
        else:
            self._handle = None

    def log(self, severity: int, msg: str) -> None:
        text = f"[TensorRT][{self._severity_names.get(severity, severity)}] {msg}"
        if self._handle is not None:
            self._handle.write(text + "\n")
            self._handle.flush()
        logging.getLogger("tensorrt").log(self._level_map.get(severity, logging.INFO), msg)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _make_logger_class(trt: Any) -> type:
    class _Logger(trt.ILogger):  # type: ignore[misc]
        def __init__(self, delegate: _FileLogger) -> None:
            trt.ILogger.__init__(self)
            self._delegate = delegate

        def log(self, severity: int, msg: str) -> None:  # noqa: D102
            self._delegate.log(severity, msg)

    return _Logger


@dataclass
class EngineBuildResult:
    metadata: dict[str, Any]
    engine_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {**self.metadata, "engine_size_bytes": self.engine_bytes}


class TensorRTEntropyCalibrator:
    """INT8 熵校准器工厂（运行期才继承 trt.IInt8EntropyCalibrator2）。

    之所以用工厂：模块导入不应要求 TensorRT 存在。
    """

    @staticmethod
    def create(
        trt: Any,
        cudart: Any,
        *,
        batches: Sequence[np.ndarray],
        cache_path: Path,
        input_name: str,
        batch_size: int,
        log: logging.Logger,
    ) -> Any:
        class _Calibrator(trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
            def __init__(self) -> None:
                trt.IInt8EntropyCalibrator2.__init__(self)
                self._batches = list(batches)
                self._index = 0
                self._cache_path = cache_path
                self._input_name = input_name
                self._batch_size = batch_size
                self._device: Any = None
                self._current: np.ndarray | None = None

            # ---- TRT 接口 ----
            def get_batch_size(self) -> int:  # noqa: D102
                return self._batch_size

            def get_batch(self, names: Sequence[str]) -> list[int] | None:  # noqa: D102
                if self._index >= len(self._batches):
                    return None

                batch = np.ascontiguousarray(self._batches[self._index], dtype=np.float32)
                self._index += 1

                if self._device is None or self._device.nbytes != batch.nbytes:
                    if self._device is not None:
                        self._device.free(cudart)
                    self._device = cuda_memory.allocate(cudart, batch)
                cuda_memory.copy_to_device(cudart, self._device, batch)
                self._current = batch
                log.info(
                    "INT8 校准 batch %s/%s", self._index, len(self._batches)
                )
                return [self._device.pointer]

            def read_calibration_cache(self) -> bytes | None:  # noqa: D102
                if self._cache_path.exists():
                    log.info("复用 INT8 校准缓存：%s", self._cache_path.name)
                    return self._cache_path.read_bytes()
                return None

            def write_calibration_cache(self, cache: bytes) -> None:  # noqa: D102
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache_path.write_bytes(cache)
                log.info("已写入 INT8 校准缓存：%s（%s 字节）", self._cache_path.name, len(cache))

        return _Calibrator()


@register_backend_adapter
class TensorRTAdapter(BackendAdapter):
    name = "tensorrt"

    # ---------------- 环境自检 ----------------

    def availability(self) -> tuple[bool, dict[str, Any]]:
        info: dict[str, Any] = {"backend": self.name}
        info.update(cuda_memory.cuda_driver_info())

        try:
            trt = import_tensorrt()
        except BackendUnavailableError as exc:
            info["tensorrt_installed"] = False
            info["reason"] = exc.message
            return False, info

        info["tensorrt_installed"] = True
        info["backend_version"] = trt.__version__
        info["tensorrt_version"] = trt.__version__

        # CUDA 版本：优先记录实际运行时头文件里的版本（SPEC 4.1 需要具体版本），
        # 取不到时退回驱动支持的最高 CUDA 版本，并单独保留 driver_version 字段。
        from app.adapters.backends import tensorrt_dev
        from app.config.settings import get_settings

        dev_files = tensorrt_dev.locate_dev_files(get_settings())
        info["cuda_runtime_version"] = dev_files.cuda_runtime_version
        if dev_files.cuda_runtime_version:
            info["cuda_version"] = dev_files.cuda_runtime_version
        try:
            builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
            info["platform_has_fast_fp16"] = bool(builder.platform_has_fast_fp16)
            info["platform_has_fast_int8"] = bool(builder.platform_has_fast_int8)
        except Exception as exc:  # pragma: no cover
            info["reason"] = f"TensorRT 初始化失败：{type(exc).__name__}: {exc}"
            return False, info

        if not info.get("available"):
            info.setdefault("reason", "未检测到可用的 NVIDIA GPU")
            return False, info

        info.setdefault("cuda_version", info.get("driver_version"))
        return True, info

    # ---------------- 能力检查（SPEC 7.1 步骤 5~6 / 7.2） ----------------

    def capability_check(
        self, onnx_path: Path, *, log_path: Path | None = None
    ) -> dict[str, Any]:
        trt = import_tensorrt()
        delegate = _FileLogger(trt, log_path)
        logger_cls = _make_logger_class(trt)
        try:
            trt_logger = logger_cls(delegate)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, trt_logger)
            parsed = bool(parser.parse(onnx_path.read_bytes()))

            errors: list[str] = []
            for index in range(parser.num_errors):
                errors.append(str(parser.get_error(index)))

            operators: list[dict[str, Any]] = []
            for index in range(network.num_layers):
                layer = network.get_layer(index)
                # 注意：TRT 10 中 num_inputs / num_outputs 是整数，不是序列
                operators.append(
                    {
                        "operator": layer.type.name,
                        "name": layer.name,
                        "inputs": int(layer.num_inputs),
                        "outputs": int(layer.num_outputs),
                    }
                )

            if not parsed or errors:
                raise OperatorUnsupportedError(
                    "TensorRT ONNX Parser 无法解析该模型，存在不支持的算子或结构",
                    detail={"parser_errors": errors[:20], "error_count": len(errors)},
                )

            return {
                "parsed": True,
                "backend": self.name,
                "backend_version": trt.__version__,
                "network_layers": network.num_layers,
                "network_inputs": [
                    {
                        "name": network.get_input(index).name,
                        "shape": list(network.get_input(index).shape),
                        "dtype": str(network.get_input(index).dtype),
                    }
                    for index in range(network.num_inputs)
                ],
                "network_outputs": [
                    {
                        "name": network.get_output(index).name,
                        "shape": list(network.get_output(index).shape),
                        "dtype": str(network.get_output(index).dtype),
                    }
                    for index in range(network.num_outputs)
                ],
                "operators": operators,
                "parser_errors": errors,
            }
        finally:
            delegate.close()

    # ---------------- Engine 构建 ----------------

    def build_engine(
        self,
        *,
        onnx_path: Path,
        engine_path: Path,
        definition: ModelDefinition,
        precision: str,
        calibration_images: list[Path] | None,
        workspace_bytes: int,
        calibration_cache: Path | None,
        log_path: Path,
    ) -> dict[str, Any]:
        trt = import_tensorrt()
        cudart = cuda_memory.load_cudart()
        builder_log = logging.getLogger("qforge.engine")

        delegate = _FileLogger(trt, log_path)
        logger_cls = _make_logger_class(trt)
        started = time.time()
        try:
            trt_logger = logger_cls(delegate)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, trt_logger)

            builder_log.info("开始解析 ONNX：%s", onnx_path.name)
            if not parser.parse(onnx_path.read_bytes()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise EngineBuildFailedError(
                    "TensorRT 解析 ONNX 失败", detail={"parser_errors": errors[:20]}
                )

            # 注意：TRT 10.x 的工厂方法名是 create_builder_config（不是 create_config），
            # 这里两种都兼容，避免因版本差异直接崩掉。
            create_config = getattr(builder, "create_builder_config", None) or getattr(
                builder, "create_config", None
            )
            if create_config is None:  # pragma: no cover
                raise BackendUnavailableError(
                    "当前 TensorRT 版本未提供创建 IBuilderConfig 的方法",
                    detail={"tensorrt_version": trt.__version__},
                )
            config = create_config()
            workspace_flag = getattr(trt.MemoryPoolType, "WORKSPACE", None)
            if workspace_flag is not None:
                config.set_memory_pool_limit(workspace_flag, workspace_bytes)

            precision_upper = precision.upper()
            calibration_info: dict[str, Any] | None = None

            if precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
                builder_log.info("精度模式 FP16：已启用 TF32/FP16 构建标志")
            elif precision == "int8":
                if not builder.platform_has_fast_int8:
                    raise BackendUnavailableError(
                        "当前 GPU 不支持快速 INT8", detail={"gpu": "platform_has_fast_int8=False"}
                    )
                if not calibration_images:
                    raise EngineBuildFailedError(
                        "INT8 静态量化必须提供校准图像（SPEC 9.1 步骤 3）"
                    )

                config.set_flag(trt.BuilderFlag.INT8)
                batches = list(
                    preprocess_batches(
                        calibration_images,
                        definition,
                        batch_size=min(8, max(1, len(calibration_images))),
                    )
                )
                if not batches:
                    raise EngineBuildFailedError("校准批次为空，无法执行 INT8 校准")

                cache_path = calibration_cache or (engine_path.parent / "calibration.cache")
                calibrator = TensorRTEntropyCalibrator.create(
                    trt,
                    cudart,
                    batches=batches,
                    cache_path=cache_path,
                    input_name=definition.input.name,
                    batch_size=int(batches[0].shape[0]),
                    log=builder_log,
                )
                # TRT 10.1 起 IInt8Calibrator 被标记为 deprecated，官方推荐「显式量化」（Q/DQ）。
                # 该路径在 10.16 仍然可用，且是「校准集 → 校准 → INT8 Engine」最直接的实现，
                # 因此这里保留并**显式忽略弃用告警**，同时在报告中记录该事实与升级方向
                # （见 docs/phase-1.md「已知问题」）。不使用 hasattr 探测属性，
                # 因为读取该属性本身就会触发弃用告警。
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    try:
                        config.int8_calibrator = calibrator
                    except Exception as exc:
                        raise BackendUnavailableError(
                            "当前 TensorRT 版本无法设置 INT8 校准器（IInt8Calibrator 可能已移除）",
                            detail={
                                "tensorrt_version": trt.__version__,
                                "error": f"{type(exc).__name__}: {exc}",
                                "hint": "可改用显式量化（Q/DQ ONNX）路径",
                            },
                        ) from exc
                calibration_info = {
                    "method": "IInt8EntropyCalibrator2",
                    "calibrator_api_status": "deprecated since TensorRT 10.1（推荐显式量化）",
                    "images": len(calibration_images),
                    "batches": len(batches),
                    "batch_size": int(batches[0].shape[0]),
                    "cache_file": cache_path.name,
                }
                builder_log.info(
                    "精度模式 INT8：静态熵校准，%s 张图像 / %s 个批次",
                    len(calibration_images),
                    len(batches),
                )
            elif precision == "fp32":
                builder_log.info("精度模式 FP32：基准精度，不启用额外标志")
            else:  # pragma: no cover - 上游已校验
                raise EngineBuildFailedError(f"不支持的精度模式：{precision}")

            builder_log.info("开始构建 Engine（工作区 %s 字节）", workspace_bytes)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise EngineBuildFailedError(
                    "TensorRT 构建 Engine 返回空结果（详见构建日志）",
                    detail={"log": str(log_path)},
                )

            engine_bytes = bytes(serialized)
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            engine_path.write_bytes(engine_bytes)

            duration = time.time() - started
            builder_log.info(
                "Engine 构建成功：%s（%s 字节，耗时 %.1fs）",
                engine_path.name,
                len(engine_bytes),
                duration,
            )

            available, env_info = self.availability()
            input_shape = list(definition.input.shape)
            metadata = {
                "backend": self.name,
                "backend_version": trt.__version__,
                # TensorRT 计划文件是**平台相关**的（Windows 构建的 Engine 不能拿到 Linux 上加载），
                # 因此把构建平台记进元数据，便于日后排查「Engine 不兼容」类问题。
                "platform": platform.system().lower(),
                "python_version": platform.python_version(),
                "cuda_version": env_info.get("driver_version"),
                "cuda_driver_version": env_info.get("driver_version"),
                "gpu_name": env_info.get("device_name"),
                "gpu_arch": env_info.get("gpu_arch"),
                "precision": precision_upper,
                "input_profile": {"min": input_shape, "opt": input_shape, "max": input_shape},
                "build_time": datetime.now(timezone.utc).isoformat(),
                "build_duration_seconds": round(duration, 3),
                "workspace_bytes": workspace_bytes,
                "engine_size_bytes": len(engine_bytes),
                "calibration": calibration_info,
                "environment_available": available,
                "engine_file": engine_path.name,
            }
            (engine_path.parent / "engine_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return metadata
        finally:
            delegate.close()

    # ---------------- 构建上下文（供代码生成注入 CMake） ----------------

    def build_context(self) -> dict[str, Any]:
        """收集编译生成工程所需的 include / lib / DLL 目录（SPEC 11.2）。

        pip 的 tensorrt 包只提供运行库 DLL，编译还需要开发包里的头文件与导入库；
        缺失信息会写进 `dev_files.problems`，由 BUILDING 阶段据此决定
        「编译验证」是继续执行还是标记为 BLOCKED。
        """
        from app.adapters.backends import tensorrt_dev
        from app.config.settings import get_settings

        files = tensorrt_dev.locate_dev_files(get_settings())
        available, environment = self.availability()

        context: dict[str, Any] = {
            "name": self.name,
            "version": environment.get("backend_version"),
            "gpu_arch": environment.get("gpu_arch"),
            "cuda_version": environment.get("cuda_version"),
            "cuda_driver_version": environment.get("driver_version"),
            "environment_available": available,
            "dev_files": files.to_dict(),
            "cmake_complete": files.complete,
        }
        # 模板直接按名字取用（tensorrt_include_dirs / cuda_libs ...）。
        # 注意：必须转成正斜杠——CMake 字符串里 `\q`、`\t` 会被当成转义字符直接报错，
        # 而 Windows 上的 CMake 完全接受正斜杠路径。
        context.update(files.to_dict())
        for key in (
            "tensorrt_include_dirs",
            "tensorrt_lib_dirs",
            "cuda_include_dirs",
            "cuda_lib_dirs",
            "runtime_dll_dirs",
        ):
            context[key] = [Path(item).as_posix() for item in context.get(key) or []]
        return context

    # ---------------- 真实推理（SPEC 11.3 步骤 4~6） ----------------

    def run_inference(
        self, *, engine_path: Path, inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        trt = import_tensorrt()
        cudart = cuda_memory.load_cudart()

        engine_bytes = engine_path.read_bytes()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeTestFailedError(
                "Engine 反序列化失败", detail={"engine": engine_path.name}
            )

        context = engine.create_execution_context()
        if context is None:  # pragma: no cover
            raise RuntimeTestFailedError("创建执行上下文失败")

        stream = cuda_memory.create_stream(cudart)
        device_buffers: list[Any] = []
        host_outputs: list[np.ndarray] = []
        names: list[str] = []
        try:
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                mode = engine.get_tensor_mode(name)
                dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                # 形状优先取执行上下文（动态 shape 时上下文才是权威），回退到引擎
                shape_source = context if hasattr(context, "get_tensor_shape") else engine
                shape = tuple(int(dim) for dim in shape_source.get_tensor_shape(name))

                if mode == trt.TensorIOMode.INPUT:
                    if name not in inputs:
                        raise RuntimeTestFailedError(
                            "缺少推理输入", detail={"expected": name, "provided": list(inputs)}
                        )
                    array = np.ascontiguousarray(inputs[name], dtype=dtype)
                    if array.shape != shape:
                        raise RuntimeTestFailedError(
                            "输入 shape 与 Engine 不一致",
                            detail={"tensor": name, "engine": list(shape), "given": list(array.shape)},
                        )
                else:
                    array = np.empty(shape, dtype=dtype)
                    names.append(name)

                buffer = cuda_memory.allocate(cudart, array)
                device_buffers.append(buffer)
                context.set_tensor_address(name, buffer.pointer)
                if mode == trt.TensorIOMode.INPUT:
                    cuda_memory.copy_to_device(cudart, buffer, array)

            ok = context.execute_async_v3(stream_handle=stream)
            if not ok:
                raise RuntimeTestFailedError("Engine 推理执行失败（execute_async_v3 返回 False）")
            cuda_memory.synchronize(cudart, stream)

            for name, buffer in zip(names, device_buffers[-len(names) :], strict=True):
                host_outputs.append(cuda_memory.copy_to_host(cudart, buffer))
            return host_outputs
        finally:
            for buffer in device_buffers:
                try:
                    buffer.free(cudart)
                except Exception:  # pragma: no cover
                    logger.warning("释放显存失败", exc_info=True)
            cuda_memory.destroy_stream(cudart, stream)

    # ---------------- 便捷入口 ----------------

    def build_and_verify(
        self,
        *,
        model_adapter: ModelAdapter,
        definition: ModelDefinition,
        onnx_path: Path,
        engine_path: Path,
        precision: str,
        calibration_images: list[Path] | None,
        workspace_bytes: int,
        calibration_cache: Path | None,
        build_log: Path,
    ) -> dict[str, Any]:
        metadata = self.build_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            definition=definition,
            precision=precision,
            calibration_images=calibration_images,
            workspace_bytes=workspace_bytes,
            calibration_cache=calibration_cache,
            log_path=build_log,
        )
        return metadata

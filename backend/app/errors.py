"""结构化错误码与领域异常（SPEC 13.1 / 15.1）。

约定：
- 所有对外错误必须携带 `ErrorCode`，便于前端与日志定位失败原因；
- 阶段 0 只实现平台/API 层错误码；SPEC 15.1 中模型、量化、构建类错误码先登记，
  由阶段 1 的具体阶段产生，避免届时改接口。
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """错误码集合。值为对外可见的稳定字符串。"""

    # ---- SPEC 15.1：模型 / 数据 / 量化 / 构建链路（阶段 1+）----
    MODEL_INVALID = "MODEL_INVALID"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    OPERATOR_UNSUPPORTED = "OPERATOR_UNSUPPORTED"
    PREPROCESS_CONFIG_INVALID = "PREPROCESS_CONFIG_INVALID"
    CALIBRATION_DATA_INVALID = "CALIBRATION_DATA_INVALID"
    QUANTIZATION_FAILED = "QUANTIZATION_FAILED"
    TENSORRT_BUILD_FAILED = "TENSORRT_BUILD_FAILED"
    CODEGEN_FAILED = "CODEGEN_FAILED"
    CPP_BUILD_FAILED = "CPP_BUILD_FAILED"
    RUNTIME_TEST_FAILED = "RUNTIME_TEST_FAILED"
    DOCKER_BUILD_FAILED = "DOCKER_BUILD_FAILED"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    TASK_TIMEOUT = "TASK_TIMEOUT"

    # ---- 平台 / API 层（阶段 0）----
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    PATH_TRAVERSAL_DETECTED = "PATH_TRAVERSAL_DETECTED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # ---- 上传与归档安全（阶段 1 新增：把 SPEC 15 风险表变成可执行校验）----
    UPLOAD_INVALID = "UPLOAD_INVALID"
    UPLOAD_TOO_LARGE = "UPLOAD_TOO_LARGE"
    ARCHIVE_INVALID = "ARCHIVE_INVALID"
    ARCHIVE_BOMB_SUSPECTED = "ARCHIVE_BOMB_SUSPECTED"

    # ---- 执行环境（阶段 1 新增）----
    # 目标后端在本机不可用（未安装 TensorRT、无可用 GPU、工具链缺失等）。
    # 这类问题必须显式失败并说明原因，不允许跳过验证后标记成功（AGENTS.md）。
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"


class DomainError(Exception):
    """领域异常基类：携带错误码、HTTP 状态码与可选结构化详情。"""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        http_status: int = 400,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.detail = detail or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error_code": self.code.value,
            "message": self.message,
            "detail": self.detail,
        }


class NotFoundError(DomainError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.NOT_FOUND, message, http_status=404, detail=detail)


class ConflictError(DomainError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.CONFLICT, message, http_status=409, detail=detail)


class ValidationFailedError(DomainError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.VALIDATION_ERROR, message, http_status=422, detail=detail)


class InvalidStateTransitionError(DomainError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(
            ErrorCode.INVALID_STATE_TRANSITION, message, http_status=409, detail=detail
        )


class PathSecurityError(DomainError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(
            ErrorCode.PATH_TRAVERSAL_DETECTED, message, http_status=400, detail=detail
        )


class NotImplementedInPhaseError(DomainError):
    """当前 Phase 未实现的功能：明确返回 501，不返回伪造数据。"""

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.NOT_IMPLEMENTED, message, http_status=501, detail=detail)


class UploadInvalidError(DomainError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.UPLOAD_INVALID, message, http_status=400, detail=detail)


class UploadTooLargeError(DomainError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.UPLOAD_TOO_LARGE, message, http_status=413, detail=detail)


class ArchiveInvalidError(DomainError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.ARCHIVE_INVALID, message, http_status=400, detail=detail)


class ArchiveBombSuspectedError(DomainError):
    """压缩包解压后规模超限（ZIP 炸弹）：拒绝处理并保留证据。"""

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(
            ErrorCode.ARCHIVE_BOMB_SUSPECTED, message, http_status=400, detail=detail
        )


class ResourceLimitError(DomainError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(
            ErrorCode.RESOURCE_LIMIT_EXCEEDED, message, http_status=429, detail=detail
        )


class BackendUnavailableError(DomainError):
    """目标后端在当前环境不可用：环境问题而非代码缺陷，必须如实上报为 BLOCKED。"""

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(
            ErrorCode.BACKEND_UNAVAILABLE, message, http_status=503, detail=detail
        )


# ---------------- 阶段 1：流水线各阶段错误（SPEC 15.1） ----------------
# 这些异常在 Celery/后台线程中抛出，由 pipeline 捕获后写成任务的 error_code 与日志，
# 不直接返回 HTTP 状态码；此处统一 http_status 便于 API 层复用时语义一致。


class _PipelineStageError(DomainError):
    """流水线阶段错误基类：默认 500（任务内部失败），由 pipeline 记录到任务。"""

    default_code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
        http_status: int = 500,
        code: ErrorCode | None = None,
    ) -> None:
        super().__init__(code or self.default_code, message, http_status=http_status, detail=detail)


class ModelInvalidError(_PipelineStageError):
    default_code = ErrorCode.MODEL_INVALID


class ModelLoadFailedError(_PipelineStageError):
    default_code = ErrorCode.MODEL_LOAD_FAILED


class OperatorUnsupportedError(_PipelineStageError):
    default_code = ErrorCode.OPERATOR_UNSUPPORTED


class PreprocessConfigError(_PipelineStageError):
    default_code = ErrorCode.PREPROCESS_CONFIG_INVALID


class CalibrationDataError(_PipelineStageError):
    default_code = ErrorCode.CALIBRATION_DATA_INVALID


class QuantizationFailedError(_PipelineStageError):
    default_code = ErrorCode.QUANTIZATION_FAILED


class EngineBuildFailedError(_PipelineStageError):
    default_code = ErrorCode.TENSORRT_BUILD_FAILED


class CodegenFailedError(_PipelineStageError):
    default_code = ErrorCode.CODEGEN_FAILED


class CppBuildFailedError(_PipelineStageError):
    default_code = ErrorCode.CPP_BUILD_FAILED


class RuntimeTestFailedError(_PipelineStageError):
    default_code = ErrorCode.RUNTIME_TEST_FAILED


class DockerBuildFailedError(_PipelineStageError):
    default_code = ErrorCode.DOCKER_BUILD_FAILED


class TaskTimeoutError(_PipelineStageError):
    default_code = ErrorCode.TASK_TIMEOUT

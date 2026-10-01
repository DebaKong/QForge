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

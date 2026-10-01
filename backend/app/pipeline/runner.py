"""流水线调度器（SPEC 13 状态机的真实执行）。

设计要点：
- 每个阶段先做状态迁移（写库 + 写 job_logs），再执行实际工作，因此前端能实时看到阶段推进；
- 阶段之间检查任务是否被取消，取消即停止（不再推进状态，也不标记 SUCCESS）；
- 任何阶段抛出领域异常 → 任务 FAILED 并带结构化 error_code；非领域异常 → INTERNAL_ERROR + 完整堆栈；
- 全过程同时写入 `storage/<task_id>/logs/pipeline.log`，满足「失败保留完整日志」；
- 无 Redis 时由进程内执行器调用本函数，接入 Redis 后由 Celery 任务调用同一函数。
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from app.adapters.backends import load_backend_adapter
from app.adapters.definition import ModelDefinition
from app.adapters.registry import get_model_adapter
from app.config.settings import get_settings
from app.db.base import session_scope
from app.errors import DomainError, ErrorCode, TaskTimeoutError, ValidationFailedError
from app.models.dataset import Dataset
from app.models.onnx_model import OnnxModel
from app.pipeline.context import PipelineContext
from app.pipeline.stages import (
    run_build,
    run_codegen,
    run_engine_build,
    run_packaging,
    run_preprocessing,
    run_quantization,
    run_testing,
    run_validation,
)
from app.services import task_service
from app.services.layout import task_root
from app.services.storage import ensure_task_layout
from app.services.task_state import TaskStatus, is_terminal

logger = logging.getLogger("qforge.pipeline")

STAGE_SEQUENCE: tuple[TaskStatus, ...] = (
    TaskStatus.VALIDATING,
    TaskStatus.PREPROCESSING,
    TaskStatus.QUANTIZING,
    TaskStatus.BUILDING_ENGINE,
    TaskStatus.GENERATING_CODE,
    TaskStatus.BUILDING,
    TaskStatus.TESTING,
    TaskStatus.PACKAGING,
)

STAGE_MESSAGES: dict[TaskStatus, str] = {
    TaskStatus.VALIDATING: "校验模型与配置",
    TaskStatus.PREPROCESSING: "校验校准集与预处理一致性",
    TaskStatus.QUANTIZING: "量化校准",
    TaskStatus.BUILDING_ENGINE: "构建 TensorRT Engine",
    TaskStatus.GENERATING_CODE: "生成 C++ 推理工程",
    TaskStatus.BUILDING: "编译生成的工程",
    TaskStatus.TESTING: "加载 Engine 并执行样例推理",
    TaskStatus.PACKAGING: "打包产物与报告",
}

STAGE_IMPLS = {
    TaskStatus.VALIDATING: run_validation,
    TaskStatus.PREPROCESSING: run_preprocessing,
    TaskStatus.QUANTIZING: run_quantization,
    TaskStatus.BUILDING_ENGINE: run_engine_build,
    TaskStatus.GENERATING_CODE: run_codegen,
    TaskStatus.BUILDING: run_build,
    TaskStatus.TESTING: run_testing,
    TaskStatus.PACKAGING: run_packaging,
}


class _TaskFileHandler(logging.Handler):
    """把流水线日志同时写入任务目录。"""

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._handle = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            self._handle.write(self.format(record) + "\n")
            self._handle.flush()
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            super().close()


def load_context(task_id: str) -> PipelineContext:
    """从数据库与存储读取任务执行所需的全部输入。"""
    settings = get_settings()
    storage_root = settings.resolved_storage_root
    ensure_task_layout(storage_root, task_id)

    with session_scope() as session:
        task = task_service.get_task(session, task_id)
        config = task_service.get_task_config(session, task_id).config

        model_row = session.get(OnnxModel, task.model_id) if task.model_id else None
        dataset_row = session.get(Dataset, task.dataset_id) if task.dataset_id else None

        model_section = config.get("model") or {}
        definition_payload = model_section.get("definition")
        architecture = (
            model_section.get("architecture")
            or (model_row.architecture if model_row else None)
            or ""
        )

        onnx_relative = model_section.get("path") or (model_row.file_path if model_row else None)
        dataset_relative = dataset_row.root_path if dataset_row else None
        model_id = model_row.id if model_row else None
        dataset_id = dataset_row.id if dataset_row else None
        dataset_kind = dataset_row.kind if dataset_row else None

    definition = ModelDefinition.parse(definition_payload) if definition_payload else None

    onnx_path: Path | None = None
    if onnx_relative:
        candidate = (storage_root / onnx_relative).resolve()
        if candidate.exists():
            onnx_path = candidate
        else:
            logger.warning("配置中的模型路径不存在：%s", candidate)

    context = PipelineContext(
        task_id=task_id,
        settings=settings,
        storage_root=storage_root,
        task_config=config,
        precision=str(config.get("precision") or "fp32").lower(),
        backend_name=str((config.get("backend") or {}).get("name") or "tensorrt").lower(),
        architecture=architecture,
    )
    context.definition = definition
    context.onnx_path = onnx_path
    context.model_id = model_id
    context.dataset_id = dataset_id
    context.dataset_kind = dataset_kind
    context.calibration_dataset_root = (
        (storage_root / dataset_relative).resolve() if dataset_relative else None
    )
    context.started_monotonic = time.monotonic()
    context.model_adapter = get_model_adapter(architecture) if architecture else None
    context.backend_adapter = load_backend_adapter(context.backend_name)
    return context


def _reload_definition(context: PipelineContext) -> None:
    """（保留给外部调用）在缺少模型定义时按架构给出默认定义。

    实际调用点是 VALIDATING 阶段内部（app/pipeline/stages/prepare.py），
    以便「文件不存在」这类更基础的校验先于「架构未知」被报告。
    """
    if context.definition is not None:
        return
    if context.model_adapter is None:
        raise ValidationFailedError(
            "任务缺少模型架构信息，无法确定模型语义",
            detail={"architecture": context.architecture, "task_id": context.task_id},
        )

    inspection = context.inspection
    payload = context.model_adapter.default_definition(inspection)
    context.definition = ModelDefinition.parse(payload)
    logger.warning(
        "任务配置缺少 Model Definition，已按架构默认定义执行（请在模型登记时显式保存）",
        extra={"architecture": context.architecture},
    )


def _check_cancelled(task_id: str) -> bool:
    with session_scope() as session:
        task = task_service.get_task(session, task_id)
        return task.status is TaskStatus.CANCELLED


def _check_timeout(context: PipelineContext, stage: TaskStatus) -> None:
    elapsed = time.monotonic() - context.started_monotonic
    limit = context.settings.task_timeout_seconds
    if elapsed > limit:
        raise TaskTimeoutError(
            f"任务总耗时超过上限 {limit} 秒（当前阶段 {stage.value}）",
            detail={"elapsed_seconds": round(elapsed, 1), "limit_seconds": limit},
        )


def _advance(task_id: str, stage: TaskStatus, *, worker_id: str) -> None:
    with session_scope() as session:
        task = task_service.get_task(session, task_id)
        if is_terminal(task.status):
            return
        task_service.transition(
            session,
            task,
            stage,
            message=STAGE_MESSAGES.get(stage, stage.value),
            worker_id=worker_id,
        )


def _fail(task_id: str, code: ErrorCode | str, message: str, detail: dict) -> None:
    with session_scope() as session:
        task = task_service.get_task(session, task_id)
        if is_terminal(task.status):
            return
        task_service.fail_task(session, task, message=message, error_code=code)
        task_service.append_log(
            session, task, f"失败详情：{detail}", stage=task.status.value, level="ERROR"
        )


def run_pipeline(task_id: str, *, worker_id: str = "local-executor") -> dict:
    """执行一次完整流水线。返回结果摘要（同时写入任务日志与产物报告）。"""
    settings = get_settings()
    storage_root = settings.resolved_storage_root
    log_path = task_root(storage_root, task_id) / "logs" / "pipeline.log"

    handler = _TaskFileHandler(log_path)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    summary: dict = {"task_id": task_id, "status": "RUNNING", "stages": []}
    try:
        context = load_context(task_id)
        logger.info(
            "流水线开始：精度=%s 后端=%s 架构=%s", context.precision, context.backend_name,
            context.architecture,
        )

        for stage in STAGE_SEQUENCE:
            if _check_cancelled(task_id):
                logger.warning("任务已被取消，停止在阶段 %s", stage.value)
                summary["status"] = "CANCELLED"
                return summary

            _check_timeout(context, stage)
            _advance(task_id, stage, worker_id=worker_id)
            logger.info("进入阶段 %s", stage.value)

            started = time.monotonic()
            STAGE_IMPLS[stage](context)
            duration = time.monotonic() - started
            summary["stages"].append(
                {"stage": stage.value, "duration_seconds": round(duration, 3)}
            )
            logger.info("阶段 %s 完成（%.1fs）", stage.value, duration)

        with session_scope() as session:
            task = task_service.get_task(session, task_id)
            _check_timeout(context, TaskStatus.SUCCESS)
            task_service.transition(
                session,
                task,
                TaskStatus.SUCCESS,
                message=f"{context.precision.upper()} 全流程完成，产物已打包",
                worker_id=worker_id,
            )
            summary["status"] = "SUCCESS"
            summary["engine_metadata"] = context.engine_metadata
            summary["verification"] = context.verification
        logger.info("流水线成功完成")
        return summary

    except DomainError as exc:
        logger.error("流水线失败（%s）：%s", exc.code.value, exc.message, exc_info=True)
        _fail(task_id, exc.code, exc.message, exc.detail)
        summary["status"] = "FAILED"
        summary["error_code"] = exc.code.value
        summary["message"] = exc.message
        summary["detail"] = exc.detail
        return summary
    except Exception as exc:  # noqa: BLE001 - 兜底：任何异常都要落库并保留堆栈
        logger.exception("流水线出现未预期异常")
        detail = {"type": type(exc).__name__, "traceback": traceback.format_exc()[-4000:]}
        _fail(task_id, ErrorCode.INTERNAL_ERROR, f"未预期异常：{exc}", detail)
        summary["status"] = "FAILED"
        summary["error_code"] = ErrorCode.INTERNAL_ERROR.value
        summary["message"] = str(exc)
        return summary
    finally:
        logger.removeHandler(handler)
        handler.close()

"""任务状态机（SPEC 13）。

阶段 0 明确以下推进规则，任何超出规则的迁移一律拒绝（抛 InvalidStateTransitionError），
不静默容忍、不自动纠正：

1. 成功路径只能沿 `STAGE_ORDER` 单向、逐级推进（不允许跳级、回退、自环）；
2. `FAILED` 可由任意非终态进入（SPEC：任意可中断阶段 → FAILED）；
3. `CANCELLED` 可由任意非终态进入（SPEC：QUEUED/执行阶段按实际可取消能力实现）；
4. `SUCCESS` / `FAILED` / `CANCELLED` 为终态，不可再迁移。
"""

from __future__ import annotations

from enum import Enum

from app.errors import InvalidStateTransitionError


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    PREPROCESSING = "PREPROCESSING"
    QUANTIZING = "QUANTIZING"
    BUILDING_ENGINE = "BUILDING_ENGINE"
    GENERATING_CODE = "GENERATING_CODE"
    BUILDING = "BUILDING"
    TESTING = "TESTING"
    PACKAGING = "PACKAGING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# 成功路径的顺序（SPEC 13 状态图）
STAGE_ORDER: tuple[TaskStatus, ...] = (
    TaskStatus.CREATED,
    TaskStatus.QUEUED,
    TaskStatus.VALIDATING,
    TaskStatus.PREPROCESSING,
    TaskStatus.QUANTIZING,
    TaskStatus.BUILDING_ENGINE,
    TaskStatus.GENERATING_CODE,
    TaskStatus.BUILDING,
    TaskStatus.TESTING,
    TaskStatus.PACKAGING,
    TaskStatus.SUCCESS,
)

TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

# 进度仅由阶段推导，不允许调用方随意写入（SPEC 13.1: progress 0~100）
PROGRESS_BY_STATUS: dict[TaskStatus, int] = {
    TaskStatus.CREATED: 0,
    TaskStatus.QUEUED: 5,
    TaskStatus.VALIDATING: 10,
    TaskStatus.PREPROCESSING: 20,
    TaskStatus.QUANTIZING: 40,
    TaskStatus.BUILDING_ENGINE: 60,
    TaskStatus.GENERATING_CODE: 70,
    TaskStatus.BUILDING: 80,
    TaskStatus.TESTING: 90,
    TaskStatus.PACKAGING: 95,
    TaskStatus.SUCCESS: 100,
    TaskStatus.FAILED: 100,
    TaskStatus.CANCELLED: 100,
}

_STAGE_INDEX: dict[TaskStatus, int] = {status: i for i, status in enumerate(STAGE_ORDER)}


def is_terminal(status: TaskStatus) -> bool:
    return status in TERMINAL_STATUSES


def is_active(status: TaskStatus) -> bool:
    """是否处于可执行（非终态）阶段。"""
    return status not in TERMINAL_STATUSES


def next_status(status: TaskStatus) -> TaskStatus | None:
    """成功路径上的下一阶段；已是 SUCCESS 或终态时返回 None。"""
    index = _STAGE_INDEX.get(status)
    if index is None or status in TERMINAL_STATUSES:
        return None
    return STAGE_ORDER[index + 1]


def allowed_transitions(status: TaskStatus) -> frozenset[TaskStatus]:
    """当前状态允许迁移到的状态集合（只读，供 API 与前端提示）。"""
    if is_terminal(status):
        return frozenset()

    allowed: set[TaskStatus] = {TaskStatus.FAILED, TaskStatus.CANCELLED}
    nxt = next_status(status)
    if nxt is not None:
        allowed.add(nxt)
    return frozenset(allowed)


def can_transition(current: TaskStatus, target: TaskStatus) -> bool:
    if current is target:
        return False
    return target in allowed_transitions(current)


def ensure_transition(current: TaskStatus, target: TaskStatus) -> None:
    """校验迁移合法性，非法则抛出 InvalidStateTransitionError。"""
    if can_transition(current, target):
        return
    raise InvalidStateTransitionError(
        f"非法状态迁移：{current.value} -> {target.value}",
        detail={
            "from": current.value,
            "to": target.value,
            "allowed": sorted(s.value for s in allowed_transitions(current)),
        },
    )


def progress_for(status: TaskStatus) -> int:
    return PROGRESS_BY_STATUS[status]

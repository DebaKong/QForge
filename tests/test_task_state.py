"""任务状态机单元测试（SPEC 13）。

覆盖：成功路径单向逐级推进、非法迁移拒绝、终态封闭、进度单调性。
"""

from __future__ import annotations

import pytest

from app.errors import InvalidStateTransitionError
from app.services.task_state import (
    STAGE_ORDER,
    TERMINAL_STATUSES,
    TaskStatus,
    allowed_transitions,
    can_transition,
    ensure_transition,
    is_terminal,
    next_status,
    progress_for,
)


def test_stage_order_matches_spec_13() -> None:
    assert [s.value for s in STAGE_ORDER] == [
        "CREATED",
        "QUEUED",
        "VALIDATING",
        "PREPROCESSING",
        "QUANTIZING",
        "BUILDING_ENGINE",
        "GENERATING_CODE",
        "BUILDING",
        "TESTING",
        "PACKAGING",
        "SUCCESS",
    ]


def test_terminal_statuses_are_closed() -> None:
    assert TERMINAL_STATUSES == {TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED}
    for status in TERMINAL_STATUSES:
        assert is_terminal(status)
        assert allowed_transitions(status) == frozenset()
        assert next_status(status) is None


def test_happy_path_advances_one_stage_at_a_time() -> None:
    for current, expected_next in zip(STAGE_ORDER, STAGE_ORDER[1:], strict=False):
        assert next_status(current) is expected_next
        assert can_transition(current, expected_next)
        ensure_transition(current, expected_next)


def test_skipping_stage_is_rejected() -> None:
    with pytest.raises(InvalidStateTransitionError) as excinfo:
        ensure_transition(TaskStatus.CREATED, TaskStatus.VALIDATING)
    assert excinfo.value.code.value == "INVALID_STATE_TRANSITION"
    assert excinfo.value.http_status == 409
    assert excinfo.value.detail["from"] == "CREATED"
    assert excinfo.value.detail["to"] == "VALIDATING"
    assert "QUEUED" in excinfo.value.detail["allowed"]


def test_backward_transition_is_rejected() -> None:
    assert not can_transition(TaskStatus.QUEUED, TaskStatus.CREATED)
    with pytest.raises(InvalidStateTransitionError):
        ensure_transition(TaskStatus.QUEUED, TaskStatus.CREATED)


def test_self_transition_is_rejected() -> None:
    for status in TaskStatus:
        assert not can_transition(status, status)


def test_failed_and_cancelled_reachable_from_every_active_status() -> None:
    active = [s for s in TaskStatus if s not in TERMINAL_STATUSES]
    for status in active:
        assert can_transition(status, TaskStatus.FAILED)
        assert can_transition(status, TaskStatus.CANCELLED)


def test_next_status_only_offers_success_path_step() -> None:
    assert allowed_transitions(TaskStatus.CREATED) == frozenset(
        {TaskStatus.QUEUED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    )
    assert allowed_transitions(TaskStatus.PACKAGING) == frozenset(
        {TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED}
    )


def test_progress_is_monotonic_along_success_path() -> None:
    values = [progress_for(status) for status in STAGE_ORDER]
    assert values == sorted(values)
    assert values[0] == 0
    assert values[-1] == 100
    assert all(0 <= value <= 100 for value in values)

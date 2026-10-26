"""EdgeWorkflowContext: blocking behaviour behind the WorkflowContext interface."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cyberwave.workers import (
    EdgeWorkflowContext,
    WorkflowCanceled,
    WorkflowContext,
    WorkflowSuspend,
)


class _Actions:
    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def wait(self, action_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((action_id, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _Reporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def node_started(self, node_uuid: str, **kwargs: Any) -> None:
        self.events.append(("started", node_uuid, kwargs))

    def node_finished(self, node_uuid: str, **kwargs: Any) -> None:
        self.events.append(("finished", node_uuid, kwargs))

    def node_error(self, node_uuid: str, **kwargs: Any) -> None:
        self.events.append(("error", node_uuid, kwargs))


def _client(
    *, canceled: bool = False, wait_result: dict[str, Any] | Exception | None = None
) -> Any:
    return SimpleNamespace(
        workflow_executions=SimpleNamespace(is_canceled=lambda _uuid: canceled),
        actions=_Actions(
            wait_result if wait_result is not None else {"status": "completed"}
        ),
    )


def test_edge_context_satisfies_the_interface() -> None:
    ctx = EdgeWorkflowContext(_client(), "exec-1", params={"a": 1})
    assert isinstance(ctx, WorkflowContext)
    assert ctx.execution_uuid == "exec-1"
    assert dict(ctx.params) == {"a": 1}


def test_step_runs_the_node_and_reports_start_and_finish() -> None:
    reporter = _Reporter()
    ctx = EdgeWorkflowContext(_client(), "exec-1", reporter=reporter)
    result = {"output": {"x": 1}, "input_data": [], "output_data": [{"x": 1}]}
    assert ctx.step("node-1", lambda: result, activation_index=2) == result
    assert [(kind, node) for kind, node, _ in reporter.events] == [
        ("started", "node-1"),
        ("finished", "node-1"),
    ]
    assert reporter.events[0][2] == {"activation_index": 2}
    assert reporter.events[1][2] == {
        "output_data": [{"x": 1}],
        "activation_index": 2,
    }


def test_step_reports_an_error_and_reraises() -> None:
    reporter = _Reporter()
    ctx = EdgeWorkflowContext(_client(), "exec-1", reporter=reporter)

    def boom() -> None:
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        ctx.step("node-1", boom)
    assert reporter.events[-1][0] == "error"
    assert reporter.events[-1][2]["error"] == "bad input"


def test_step_refuses_to_start_when_canceled() -> None:
    ran: list[int] = []
    ctx = EdgeWorkflowContext(_client(canceled=True), "exec-1")
    with pytest.raises(WorkflowCanceled):
        ctx.step("node-1", lambda: ran.append(1))
    assert ran == []


def test_call_passes_a_stable_idempotency_key() -> None:
    ctx = EdgeWorkflowContext(_client(), "exec-1")
    seen: list[str] = []
    assert ctx.call("dispatch", lambda key: seen.append(key) or "ok") == "ok"
    ctx.call("dispatch", seen.append)
    assert seen == ["exec-1:dispatch", "exec-1:dispatch"]


def test_wait_action_delegates_to_actions_wait() -> None:
    client = _client(wait_result={"status": "completed", "n": 3})
    ctx = EdgeWorkflowContext(client, "exec-1")
    result = ctx.wait_action("nav", "act-1", timeout=30, twin_uuid="twin-1")
    assert result == {"status": "completed", "n": 3}
    assert client.actions.calls == [("act-1", {"twin_uuid": "twin-1", "timeout": 30})]


def test_wait_action_failure_and_timeout_surface_like_actions_wait() -> None:
    failed = EdgeWorkflowContext(
        _client(wait_result=RuntimeError("action a finished with status 'failed'")),
        "exec-1",
    )
    with pytest.raises(RuntimeError, match="failed"):
        failed.wait_action("nav", "a")
    timed_out = EdgeWorkflowContext(
        _client(wait_result=TimeoutError("did not reach a terminal status")), "exec-1"
    )
    with pytest.raises(TimeoutError):
        timed_out.wait_action("nav", "a")


def test_sleep_blocks_for_the_requested_time() -> None:
    slept: list[float] = []
    ctx = EdgeWorkflowContext(_client(), "exec-1", sleep=slept.append)
    ctx.sleep("pause", 4.5)
    assert slept == [4.5]


def test_suspend_is_not_caught_by_except_exception() -> None:
    assert not issubclass(WorkflowSuspend, Exception)
    assert WorkflowSuspend("wait-1").wait_id == "wait-1"

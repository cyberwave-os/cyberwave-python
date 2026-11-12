"""The cloud runner's context and segment runner, over an in-memory journal."""

from __future__ import annotations

import pytest

from cyberwave.workers.cloud_runner.context import (
    CloudWorkflowContext,
    SyncWaits,
    WorkflowWaitTooLong,
)
from cyberwave.workers.cloud_runner.ports import JournalPortError
from cyberwave.workers.cloud_runner.segment import (
    CANCELED,
    COMPLETED,
    FAILED,
    SUSPENDED,
    run_segment,
)
from cyberwave.workers.cloud_runner.wait_results import (
    elapsed_result,
    interpret_action_result,
    timed_out_result,
)
from cyberwave.workers.workflow_context import WorkflowCanceled, WorkflowSuspend


class MemoryPort:
    """A ``JournalPort`` that keeps everything in dicts."""

    def __init__(self) -> None:
        self.steps: dict[str, dict] = {}
        self.waits: dict[str, dict] = {}
        self.canceled = False

    def snapshot(self):
        return {"steps": dict(self.steps), "waits": dict(self.waits)}

    def is_canceled(self):
        return self.canceled

    def start_step(self, node_uuid, activation_index):
        key = f"{node_uuid}:{activation_index}"
        step = self.steps.setdefault(
            key, {"status": "running", "result": None, "error": None, "calls": {}}
        )
        return dict(step)

    def complete_step(self, node_uuid, activation_index, result):
        step = self.steps[f"{node_uuid}:{activation_index}"]
        step.update(status="completed", result=result)
        return {"output_data": result}

    def fail_step(self, node_uuid, activation_index, error):
        self.steps[f"{node_uuid}:{activation_index}"].update(
            status="failed", error=error
        )

    def record_call(self, node_uuid, activation_index, call_id, value):
        calls = self.steps[f"{node_uuid}:{activation_index}"]["calls"]
        return calls.setdefault(call_id, value)

    def register_wait(self, node_uuid, activation_index, wait_id, kind, key, deadline):
        return self.waits.setdefault(
            wait_id,
            {"wait_id": wait_id, "kind": kind, "status": "pending", "result": None},
        )


def test_a_completed_step_is_not_run_again():
    port = MemoryPort()
    runs = []

    def work():
        runs.append(1)
        return {"n": len(runs)}

    first = CloudWorkflowContext(port, "exec").step("a", work)
    # A second segment replays the journal.
    again = CloudWorkflowContext(port, "exec").step("a", work)

    assert first == {"output_data": {"n": 1}}
    assert again == {"n": 1}
    assert runs == [1]


def test_a_call_is_journaled_so_a_crashed_step_does_not_repeat_it():
    port = MemoryPort()
    sent = []

    def crash_after_call(ctx):
        ctx.call("goto", lambda key: sent.append(key) or "action-1")
        raise KeyboardInterrupt  # the segment dies before the step finishes

    with pytest.raises(KeyboardInterrupt):
        ctx = CloudWorkflowContext(port, "exec")
        ctx.step("a", lambda: crash_after_call(ctx))

    ctx = CloudWorkflowContext(port, "exec")
    assert ctx.step("a", lambda: ctx.call("goto", lambda key: sent.append(key))) == {
        "output_data": "action-1"
    }
    assert len(sent) == 1


def test_an_unresolved_wait_suspends_and_a_resolved_one_returns():
    port = MemoryPort()
    ctx = CloudWorkflowContext(port, "exec")
    with pytest.raises(WorkflowSuspend) as suspended:
        ctx.step("a", lambda: ctx.wait_action("w", "action-1"))
    assert suspended.value.wait_id == "a:0:w"

    port.waits["a:0:w"].update(status="resolved", result={"status": "completed"})
    ctx = CloudWorkflowContext(port, "exec")
    assert ctx.step("a", lambda: ctx.wait_action("w", "action-1"))["output_data"] == {
        "status": "completed"
    }


def test_a_canceled_run_does_not_start_a_step():
    port = MemoryPort()
    port.canceled = True
    with pytest.raises(WorkflowCanceled):
        CloudWorkflowContext(port, "exec").step("a", lambda: 1)


def test_a_result_the_journal_cannot_store_fails_the_step():
    port = MemoryPort()

    def refuse(*_args):
        raise JournalPortError("not JSON", "invalid")

    port.complete_step = refuse
    with pytest.raises(JournalPortError):
        CloudWorkflowContext(port, "exec").step("a", lambda: object())
    assert port.steps["a:0"]["status"] == "failed"


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def sync(clock, statuses, **kwargs):
    polled = iter(statuses)
    return SyncWaits(
        action_status=lambda _action, _twin: next(polled),
        poll_interval=3.0,
        sleep=clock.sleep,
        now=clock.now,
        **kwargs,
    )


def test_a_sync_wait_polls_until_the_action_is_terminal():
    clock = Clock()
    port = MemoryPort()
    ctx = CloudWorkflowContext(
        port,
        "exec",
        sync_waits=sync(clock, [{"status": "running"}, {"status": "completed"}]),
    )
    result = ctx.step("a", lambda: ctx.wait_action("w", "action-1", timeout=60))
    assert result["output_data"]["status"] == "completed"
    assert port.waits == {}  # a sync wait registers nothing


def test_a_sync_wait_over_the_cap_fails_up_front():
    ctx = CloudWorkflowContext(
        MemoryPort(), "exec", sync_waits=sync(Clock(), [], max_wait_seconds=10)
    )
    with pytest.raises(WorkflowWaitTooLong):
        ctx.step("a", lambda: ctx.sleep("s", 11))


def test_a_sync_wait_stops_when_the_run_is_canceled_between_polls():
    clock = Clock()
    port = MemoryPort()
    beats = []

    def beat():
        beats.append(clock.t)
        port.canceled = len(beats) == 2

    ctx = CloudWorkflowContext(
        port,
        "exec",
        sync_waits=sync(clock, [{"status": "running"}] * 5, heartbeat=beat),
    )
    with pytest.raises(WorkflowCanceled):
        ctx.step("a", lambda: ctx.wait_action("w", "action-1", timeout=60))
    assert len(beats) == 2


def test_the_waiting_node_sees_what_client_actions_wait_would_raise():
    assert interpret_action_result({"status": "completed"}, "a", 5) == {
        "status": "completed"
    }
    with pytest.raises(RuntimeError, match="finished with status 'failed'"):
        interpret_action_result({"status": "failed", "message": "x"}, "a", 5)
    with pytest.raises(TimeoutError, match="within 5s"):
        interpret_action_result(timed_out_result(), "a", 5)
    assert elapsed_result() == {"elapsed": True}


SOURCE = """
def run(_cw_ctx, client, payload):
    return {"seen": _cw_ctx.step("n", lambda: payload["x"] * 2)}
"""


def test_run_segment_completes_a_run():
    outcome = run_segment(
        SOURCE, port=MemoryPort(), client=None, execution_uuid="e", payload={"x": 2}
    )
    assert outcome.status == COMPLETED
    assert outcome.outputs == {"seen": {"output_data": 4}}


def test_run_segment_reports_suspended_canceled_and_failed():
    waiting = (
        "def run(c, client, payload):\n"
        "    return c.step('n', lambda: c.wait_action('w', 'a'))\n"
    )
    assert (
        run_segment(waiting, port=MemoryPort(), client=None, execution_uuid="e").status
        == SUSPENDED
    )

    canceled = MemoryPort()
    canceled.canceled = True
    assert (
        run_segment(SOURCE, port=canceled, client=None, execution_uuid="e").status
        == CANCELED
    )

    broken = "def run(c, client, payload):\n    raise ValueError('boom')\n"
    outcome = run_segment(broken, port=MemoryPort(), client=None, execution_uuid="e")
    assert (outcome.status, outcome.error) == (FAILED, "boom")


def test_a_reporter_is_told_when_the_run_finishes_but_not_when_it_suspends():
    class Reporter:
        def __init__(self):
            self.finished_calls = []

        def node_started(self, *args, **kwargs): ...
        def node_finished(self, *args, **kwargs): ...
        def node_error(self, *args, **kwargs): ...

        def finished(self, **kwargs):
            self.finished_calls.append(kwargs)

    reporter = Reporter()
    run_segment(
        SOURCE,
        port=MemoryPort(),
        client=None,
        execution_uuid="e",
        payload={"x": 1},
        reporter=reporter,
    )
    assert reporter.finished_calls == [{"status": "success", "error_message": None}]

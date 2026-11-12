"""``run_execution``: a cloud workflow run with no journal, on the SDK alone.

The sources under ``fixtures/cloud_runner`` were compiled by the backend for a
Move Twin node and a Timer node (``/compile?target=cloud``); the ids in them are
random, so they are samples to run, not a snapshot to compare.
"""

from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from cyberwave.actions import ActionsClient
from cyberwave.exceptions import CyberwaveAPIError, CyberwaveError
from cyberwave.workers.cloud_runner import RunnerContext, run_execution
from cyberwave.workers.cloud_runner import runner as runner_module
from cyberwave.workers.cloud_runner.__main__ import (
    EXIT_CANNOT_START,
    EXIT_RUN_ENDED,
    main,
)
from cyberwave.workers.cloud_runner.runner import envelope_from_execution, watch_cancel
from cyberwave.workers.cloud_runner.segment import (
    CANCELED,
    COMPLETED,
    FAILED,
    run_with_context,
)
from cyberwave.workers.workflow_context import WorkflowCanceled
from cyberwave.workflow_executions import WorkflowExecutionManager

FIXTURES = Path(__file__).parent / "fixtures" / "cloud_runner"
MOVE_TWIN = (FIXTURES / "move_twin.py.txt").read_text()
TIMER = (FIXTURES / "timer.py.txt").read_text()


class Reporter:
    def __init__(self):
        self.events = []
        self.errors = []

    def node_started(self, node_uuid, **kwargs):
        self.events.append(("started", node_uuid))

    def node_finished(self, node_uuid, **kwargs):
        self.events.append(("finished", node_uuid))

    def node_error(self, node_uuid, **kwargs):
        self.events.append(("error", node_uuid))
        self.errors.append(kwargs.get("error"))

    def finished(self, **kwargs):
        self.events.append(("run", kwargs["status"]))


class FakeNavigation:
    def __init__(self):
        self.dispatched = []

    def follow_path(self, plan, **kwargs):
        self.dispatched.append(plan)
        return {"action_id": "act-1"}


class FakeClient:
    """Enough of ``Cyberwave`` for the Move Twin sample."""

    def __init__(self, statuses):
        self.navigation = FakeNavigation()
        self.workflow_executions = WorkflowExecutionManager(None)
        self.api = SimpleNamespace(api_client=object())
        self.environments = SimpleNamespace()
        self.actions = ActionsClient(api_client=None)
        self._statuses = iter(statuses)
        self.actions.get_status = lambda *_a, **_k: next(self._statuses)

    def twin(self, **_kwargs):
        return SimpleNamespace(navigation=self.navigation)


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr("cyberwave.actions.time.sleep", lambda _seconds: None)


RUNNING = {"status": "running"}


def never_canceled():
    return False


def test_a_move_twin_run_dispatches_once_waits_and_completes():
    client = FakeClient([{"status": "running"}, {"status": "completed"}])
    reporter = Reporter()

    outcome = run_execution(
        client,
        workflow_uuid="wf",
        execution_uuid="ex",
        source=MOVE_TWIN,
        reporter=reporter,
        cancel_check=never_canceled,
        execution=RUNNING,
    )

    assert outcome.status == COMPLETED
    assert len(client.navigation.dispatched) == 1
    assert [event[0] for event in reporter.events] == [
        "started",
        "finished",
        "started",
        "finished",
        "run",
    ]
    assert reporter.events[-1] == ("run", "success")


def test_a_failed_action_fails_the_run():
    client = FakeClient([{"status": "failed", "message": "blocked by an obstacle"}])
    reporter = Reporter()

    outcome = run_execution(
        client,
        workflow_uuid="wf",
        execution_uuid="ex",
        source=MOVE_TWIN,
        reporter=reporter,
        cancel_check=never_canceled,
        execution=RUNNING,
    )

    assert outcome.status == FAILED
    assert "blocked by an obstacle" in outcome.error
    assert reporter.events[-2:] == [("error", reporter.events[-2][1]), ("run", "error")]


def test_a_cancel_command_during_an_action_wait_ends_the_run_at_the_next_poll():
    client = FakeClient([])
    reporter = Reporter()
    polls = []

    def get_status(*_args, **_kwargs):
        polls.append(1)
        if len(polls) == 2:  # the backend's cancel command arrives over MQTT
            client.workflow_executions.mark_canceled("ex")
        return {"status": "running"}

    client.actions.get_status = get_status

    outcome = run_execution(
        client,
        workflow_uuid="wf",
        execution_uuid="ex",
        source=MOVE_TWIN,
        reporter=reporter,
        cancel_check=never_canceled,
        execution=RUNNING,
    )

    assert outcome.status == CANCELED
    assert reporter.events[-1] == ("run", "canceled")
    assert len(polls) == 2  # the cancel is seen before the third poll
    # The node that was waiting says why it ended, not just which run it was.
    assert [e for e in reporter.errors if e] == [
        "execution ex was canceled while this node was running"
    ]


def test_the_status_read_ends_a_wait_when_the_command_was_missed():
    client = FakeClient([])
    clock = {"t": 0.0}
    polls = []

    def get_status(*_args, **_kwargs):
        polls.append(1)
        clock["t"] += 3.0  # each poll is 3 s later
        return {"status": "running"}

    client.actions.get_status = get_status

    def cancel_check():
        return len(polls) >= 3  # the backend has it canceled from the third poll on

    context = RunnerContext(
        client,
        "ex",
        cancel_check=cancel_check,
        poll_seconds=3.0,
        fallback_seconds=15.0,
        monotonic=lambda: clock["t"],
    )
    with pytest.raises(WorkflowCanceled):
        context.wait_action("nav", "act-1", timeout=600)
    # The status was read at 0 s (not canceled yet) and next after 15 s, which is
    # five polls of 3 s later: the cancel ends the wait before the sixth.
    assert len(polls) == 5


def test_a_sleep_is_interrupted_by_a_cancel():
    client = FakeClient([])
    sleeps = []
    clock = {"t": 0.0}

    def sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds
        client.workflow_executions.mark_canceled("ex")  # the command arrives

    context = RunnerContext(
        client,
        "ex",
        cancel_check=never_canceled,
        poll_seconds=3.0,
        sleep=sleep,
        monotonic=lambda: clock["t"],
    )
    with pytest.raises(WorkflowCanceled):
        context.sleep("nap", 60)
    assert sleeps == [3.0]


def test_the_timer_sample_sleeps_in_poll_sized_naps():
    clock = {"t": 0.0}
    naps = []

    def sleep(seconds):
        naps.append(seconds)
        clock["t"] += seconds

    context = RunnerContext(
        FakeClient([]),
        "ex",
        cancel_check=never_canceled,
        poll_seconds=3.0,
        sleep=sleep,
        monotonic=lambda: clock["t"],
    )
    run_source = run_with_context(
        TIMER, context, client=FakeClient([]), execution_uuid="ex", payload={}
    )
    assert run_source.status == COMPLETED
    assert naps == [3.0, 2.0]  # the sample's 5 second timer


def test_the_status_is_read_at_most_once_per_fallback_interval():
    client = FakeClient([])
    clock = {"t": 0.0}
    asked = []
    context = RunnerContext(
        client,
        "ex",
        cancel_check=lambda: asked.append(1) or False,
        fallback_seconds=15.0,
        monotonic=lambda: clock["t"],
    )
    for _ in range(5):  # quick steps in a row
        context.is_canceled()
    assert len(asked) == 1
    clock["t"] = 14.0
    context.is_canceled()
    assert len(asked) == 1
    clock["t"] = 15.5
    context.is_canceled()
    assert len(asked) == 2


def test_a_status_read_that_finds_the_run_canceled_marks_it_like_the_command():
    client = FakeClient([])
    context = RunnerContext(client, "ex", cancel_check=lambda: True)
    assert context.is_canceled() is True
    assert client.workflow_executions.is_canceled("ex") is True


def test_a_cancel_check_that_fails_does_not_end_the_run():
    def broken():
        raise ConnectionError("backend unreachable")

    context = RunnerContext(FakeClient([]), "ex", cancel_check=broken)
    assert context.is_canceled() is False


def _status_reads(*outcomes):
    """A ``cancel_check`` that returns or raises each outcome in turn."""
    queue = iter(outcomes)

    def check():
        outcome = next(queue)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return check


def _refused(status):
    return CyberwaveAPIError("refused", status_code=status)


@pytest.mark.parametrize("status", [401, 403])
def test_a_run_whose_token_is_refused_twice_in_a_row_is_treated_as_ended(status):
    client = FakeClient([])
    context = RunnerContext(
        client,
        "ex",
        cancel_check=_status_reads(_refused(status), _refused(status)),
        fallback_seconds=0.0,
    )
    assert context.is_canceled() is False  # one refusal can be a blip
    assert context.is_canceled() is True
    assert client.workflow_executions.is_canceled("ex") is True


@pytest.mark.parametrize(
    "between",
    [False, ConnectionError("backend unreachable"), _refused(500), _refused(404)],
)
def test_anything_but_a_refusal_between_two_refusals_resets_the_count(between):
    context = RunnerContext(
        FakeClient([]),
        "ex",
        cancel_check=_status_reads(_refused(401), between, _refused(401)),
        fallback_seconds=0.0,
    )
    assert [context.is_canceled() for _ in range(3)] == [False, False, False]


@pytest.mark.parametrize("failure", [ConnectionError("down"), _refused(500)])
def test_a_backend_that_keeps_failing_for_other_reasons_does_not_end_the_run(failure):
    context = RunnerContext(
        FakeClient([]),
        "ex",
        cancel_check=_status_reads(*[failure] * 6),
        fallback_seconds=0.0,
    )
    assert [context.is_canceled() for _ in range(6)] == [False] * 6


def test_a_wait_ends_when_the_runs_token_is_refused_and_the_command_was_missed():
    client = FakeClient([])
    clock = {"t": 0.0}
    polls = []

    def get_status(*_args, **_kwargs):
        polls.append(1)
        clock["t"] += 3.0
        return {"status": "running"}

    client.actions.get_status = get_status

    def refused():
        raise _refused(401)

    context = RunnerContext(
        client,
        "ex",
        cancel_check=refused,
        poll_seconds=3.0,
        fallback_seconds=3.0,
        monotonic=lambda: clock["t"],
    )
    with pytest.raises(WorkflowCanceled):
        context.wait_action("nav", "act-1", timeout=600)
    # The status is read on every poll here: the second refusal ends the wait,
    # where a run that only swallowed the errors would wait out its 600 s.
    assert len(polls) <= 3


def test_once_canceled_it_stays_canceled_without_asking_again():
    client = FakeClient([])
    asked = []
    context = RunnerContext(
        client, "ex", cancel_check=lambda: asked.append(1) or True, fallback_seconds=0.0
    )
    assert context.is_canceled() is True
    assert context.is_canceled() is True
    assert len(asked) == 1


def test_the_source_and_the_cancel_state_come_from_the_backend(monkeypatch):
    calls = []

    def fake_request(api_client, method, path, **kwargs):
        calls.append((method, path, kwargs.get("query")))
        if path.endswith("/compile"):
            return 200, {
                "worker_source": "def run(*a): return {}",
                "source_sha256": "h",
            }
        return 200, {"status": "canceled"}

    monkeypatch.setattr(runner_module, "request", fake_request)
    client = SimpleNamespace(api=SimpleNamespace(api_client=object()))

    assert runner_module.fetch_cloud_source(client, "wf") == (
        "def run(*a): return {}",
        "h",
    )
    assert runner_module.execution_is_canceled(client, "ex") is True
    assert calls[0] == ("GET", "/api/v1/workflows/{uuid}/compile", {"target": "cloud"})


def test_a_run_nothing_else_created_waits_for_the_backend_to_record_it(monkeypatch):
    """Node events that beat the ``started`` event to the backend are dropped."""
    answers = iter([CyberwaveError("404"), CyberwaveError("404"), (200, {})])

    def fake_request(api_client, method, path, **kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(runner_module, "request", fake_request)
    monkeypatch.setattr(runner_module, "RECORDED_POLL_SECONDS", 0)
    client = SimpleNamespace(api=SimpleNamespace(api_client=object()))

    runner_module._wait_until_recorded(client, "ex")

    assert next(answers, None) is None  # asked three times, then stopped


def test_waiting_for_the_record_gives_up_instead_of_blocking_the_run(monkeypatch):
    def fake_request(api_client, method, path, **kwargs):
        raise CyberwaveError("404")

    monkeypatch.setattr(runner_module, "request", fake_request)
    monkeypatch.setattr(runner_module, "RECORDED_POLL_SECONDS", 0)
    client = SimpleNamespace(api=SimpleNamespace(api_client=object()))

    runner_module._wait_until_recorded(client, "ex", seconds=0)  # returns


def test_the_entry_point_refuses_to_start_without_its_environment(monkeypatch):
    for name in ("CYBERWAVE_API_KEY", "CYBERWAVE_WORKFLOW_UUID"):
        monkeypatch.delenv(name, raising=False)
    assert main([]) == EXIT_CANNOT_START


def test_a_job_told_to_stop_reports_its_run_failed_instead_of_dying(
    monkeypatch, tmp_path
):
    """The launcher ends a job at its limit with SIGTERM; the run must be reported."""
    import cyberwave

    client = StartingClient()
    client.mqtt = SimpleNamespace(
        connected=False, connect=lambda: None, disconnect=lambda: None
    )
    monkeypatch.setattr(cyberwave, "Cyberwave", lambda api_key: client)
    monkeypatch.setattr(runner_module, "_wait_until_recorded", lambda *a: None)
    monkeypatch.setenv("CYBERWAVE_API_KEY", "key")
    monkeypatch.setenv("CYBERWAVE_WORKFLOW_UUID", "wf")
    source = tmp_path / "run.py"
    source.write_text(
        "import os\nimport signal\n\n"
        "def run(_cw_ctx, client, payload):\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    return {}\n"
    )
    before = signal.getsignal(signal.SIGTERM)

    code = main(["--source-file", str(source), "--new-execution"])

    assert code == EXIT_RUN_ENDED
    assert client.reporter.events[-1] == ("run", "error")
    assert signal.getsignal(signal.SIGTERM) is before  # not left installed


# ---------------------------------------------------------------------------
# What a run reads from the backend: the execution and the pinned source
# ---------------------------------------------------------------------------

ECHO = "def run(_cw_ctx, client, payload):\n    return {'payload': payload}\n"


class StartingClient:
    """A client whose ``workflow_executions.start`` hands out a recording reporter."""

    def __init__(self):
        self.reporter = Reporter()
        self.started = []
        self.api = SimpleNamespace(api_client=object())
        self.workflow_executions = WorkflowExecutionManager(None)
        self.workflow_executions.start = self._start

    def _start(self, **kwargs):
        self.started.append(kwargs)
        return self.reporter


def backend(monkeypatch, execution, compile_body=None):
    """Answer the runner's two reads; returns the list of paths it asked for."""
    asked = []

    def fake_request(api_client, method, path, **kwargs):
        asked.append(path)
        if path.endswith("/compile"):
            return 200, compile_body
        return 200, execution

    monkeypatch.setattr(runner_module, "request", fake_request)
    return asked


def compiled(source=ECHO, digest="h1"):
    return {"worker_source": source, "source_sha256": digest}


def test_the_envelope_is_built_from_the_executions_trigger_data():
    envelope = envelope_from_execution(
        {
            "trigger_data": {"speed": 3, "params": {"mode": "fast"}},
            "started_at": "2026-10-28T08:00:00Z",
            "triggered_by": "user-1",
        }
    )
    assert envelope == {
        "inputs": {"speed": 3, "params": {"mode": "fast"}},
        "params": {"mode": "fast"},
        "requested_at": "2026-10-28T08:00:00Z",
        "requested_by_uuid": "user-1",
    }


def test_a_requested_execution_gets_started_and_runs_with_its_trigger_data(
    monkeypatch,
):
    client = StartingClient()
    asked = backend(
        monkeypatch,
        {"status": "requested", "trigger_data": {"params": {"mode": "fast"}}},
        compiled(),
    )

    outcome = run_execution(
        client, workflow_uuid="wf", execution_uuid="ex", cancel_check=never_canceled
    )

    assert outcome.status == COMPLETED
    assert client.started == [
        {"workflow_uuid": "wf", "execution_uuid": "ex", "source_type": None}
    ]
    assert outcome.outputs["payload"]["params"] == {"mode": "fast"}
    assert asked == [
        "/api/v1/workflows/executions/{uuid}",
        "/api/v1/workflows/{uuid}/compile",
    ]


def test_a_running_execution_is_not_started_again(monkeypatch):
    client = StartingClient()
    backend(monkeypatch, {"status": "running"}, compiled())

    outcome = run_execution(
        client, workflow_uuid="wf", execution_uuid="ex", cancel_check=never_canceled
    )

    assert outcome.status == COMPLETED
    assert client.started == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [("success", COMPLETED), ("error", FAILED), ("canceled", CANCELED)],
)
def test_a_run_that_is_already_over_is_not_run_again(monkeypatch, status, expected):
    client = StartingClient()
    asked = backend(monkeypatch, {"status": status}, compiled())

    outcome = run_execution(client, workflow_uuid="wf", execution_uuid="ex")

    assert outcome.status == expected
    assert f"already {status}" in outcome.error
    assert client.started == []
    assert asked == ["/api/v1/workflows/executions/{uuid}"]  # no source fetched


def test_a_source_that_differs_from_the_pin_is_refused_and_reported(monkeypatch):
    client = StartingClient()
    backend(
        monkeypatch,
        {
            "status": "requested",
            "metadata": {"runner": {"source_sha256": "pinned-hash"}},
        },
        compiled(digest="other-hash"),
    )

    outcome = run_execution(client, workflow_uuid="wf", execution_uuid="ex")

    assert outcome.status == FAILED
    assert "changed after this run was triggered" in outcome.error
    assert client.reporter.events == [("run", "error")]


def test_a_source_that_matches_the_pin_runs(monkeypatch):
    client = StartingClient()
    backend(
        monkeypatch,
        {"status": "requested", "metadata": {"runner": {"source_sha256": "h1"}}},
        compiled(digest="h1"),
    )

    outcome = run_execution(
        client, workflow_uuid="wf", execution_uuid="ex", cancel_check=never_canceled
    )

    assert outcome.status == COMPLETED


def test_a_run_with_no_pin_runs_whatever_the_workflow_compiles_to(monkeypatch):
    client = StartingClient()
    backend(monkeypatch, {"status": "requested"}, compiled(digest="anything"))

    outcome = run_execution(
        client, workflow_uuid="wf", execution_uuid="ex", cancel_check=never_canceled
    )

    assert outcome.status == COMPLETED


def test_a_source_that_cannot_be_fetched_fails_the_run_and_reports_it(monkeypatch):
    client = StartingClient()

    def fake_request(api_client, method, path, **kwargs):
        if path.endswith("/compile"):
            raise ConnectionError("backend unreachable")
        return 200, {"status": "requested"}

    monkeypatch.setattr(runner_module, "request", fake_request)

    outcome = run_execution(client, workflow_uuid="wf", execution_uuid="ex")

    assert outcome.status == FAILED
    assert client.reporter.events == [("run", "error")]


def test_a_source_passed_in_is_used_as_is_without_the_pin_check(monkeypatch):
    client = StartingClient()
    asked = backend(
        monkeypatch,
        {"status": "requested", "metadata": {"runner": {"source_sha256": "pinned"}}},
    )

    outcome = run_execution(
        client,
        workflow_uuid="wf",
        execution_uuid="ex",
        source=ECHO,
        cancel_check=never_canceled,
    )

    assert outcome.status == COMPLETED
    assert asked == ["/api/v1/workflows/executions/{uuid}"]


def test_an_explicit_payload_overrides_the_trigger_data(monkeypatch):
    client = StartingClient()
    backend(
        monkeypatch,
        {"status": "running", "trigger_data": {"params": {"mode": "fast"}}},
        compiled(),
    )

    outcome = run_execution(
        client,
        workflow_uuid="wf",
        execution_uuid="ex",
        payload={"params": {"mode": "slow"}},
        cancel_check=never_canceled,
    )

    assert outcome.outputs["payload"]["params"] == {"mode": "slow"}


# ---------------------------------------------------------------------------
# The cancel command (the edge mechanism): MQTT marks the execution canceled
# ---------------------------------------------------------------------------


class FakeMqtt:
    def __init__(self, *, connected=True, prefix=""):
        self.connected = connected
        self.topic_prefix = prefix
        self.subscribed = []
        self.unsubscribed = []
        self.handler = None
        self.events = []

    def subscribe(self, topic, handler, qos=0, *, subscriber_key=None):
        self.subscribed.append((topic, qos, subscriber_key))
        self.handler = handler
        self.events.append("subscribe")

    def unsubscribe(self, topic, subscriber_key=None):
        self.unsubscribed.append((topic, subscriber_key))
        self.events.append("unsubscribe")


def client_with_mqtt(mqtt):
    client = FakeClient([])
    client.mqtt = mqtt
    return client


def test_watch_cancel_listens_on_the_workflows_cancel_topic():
    mqtt = FakeMqtt(prefix="dev/")
    client = client_with_mqtt(mqtt)

    with watch_cancel(client, "wf", "ex"):
        topic, qos, key = mqtt.subscribed[0]
        assert (topic, qos) == ("dev/cyberwave/workflow/wf/cancel", 1)
        assert mqtt.unsubscribed == []
    assert mqtt.unsubscribed == [(topic, key)]


def test_watch_cancel_marks_only_its_own_execution():
    mqtt = FakeMqtt()
    client = client_with_mqtt(mqtt)
    with watch_cancel(client, "wf", "ex"):
        mqtt.handler({"execution_uuid": "another-run"})
        mqtt.handler("not a dict")
        assert client.workflow_executions.is_canceled("ex") is False
        assert client.workflow_executions.is_canceled("another-run") is False

        mqtt.handler({"execution_uuid": "ex", "requested_at": "now"})
        assert client.workflow_executions.is_canceled("ex") is True


@pytest.mark.parametrize("mqtt", [None, FakeMqtt(connected=False)])
def test_watch_cancel_does_nothing_without_a_connection(mqtt):
    client = FakeClient([])
    if mqtt is not None:
        client.mqtt = mqtt
    with watch_cancel(client, "wf", "ex"):  # a no-op, not an error
        pass
    if mqtt is not None:
        assert mqtt.subscribed == []


def test_a_run_listens_before_it_reads_the_execution_and_stops_after(monkeypatch):
    mqtt = FakeMqtt()
    client = client_with_mqtt(mqtt)

    def fake_request(api_client, method, path, **kwargs):
        mqtt.events.append("read")
        return 200, {"status": "running"}

    monkeypatch.setattr(runner_module, "request", fake_request)
    run_execution(
        client,
        workflow_uuid="wf",
        execution_uuid="ex",
        source=ECHO,
        cancel_check=never_canceled,
    )
    assert mqtt.events[0] == "subscribe"
    assert mqtt.events[-1] == "unsubscribe"
    assert mqtt.events.index("read") > mqtt.events.index("subscribe")


def test_a_cancel_command_that_arrives_while_the_run_starts_ends_it(monkeypatch):
    mqtt = FakeMqtt()
    client = client_with_mqtt(mqtt)
    one_step = (
        "def run(c, client, payload):\n"
        "    return c.step('n', lambda: {'output': 1, 'input_data': [], 'output_data': [1]})\n"
    )

    def fake_request(api_client, method, path, **kwargs):
        if path.endswith("/compile"):
            mqtt.handler({"execution_uuid": "ex"})  # canceled while fetching the source
            return 200, compiled(one_step)
        return 200, {"status": "running"}

    monkeypatch.setattr(runner_module, "request", fake_request)
    outcome = run_execution(
        client, workflow_uuid="wf", execution_uuid="ex", cancel_check=never_canceled
    )
    assert outcome.status == CANCELED

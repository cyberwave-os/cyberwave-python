"""Run a compiled cloud workflow start to finish, on the SDK alone.

No journal: the runner holds the run for its whole life, blocks on waits like an
edge worker, and a runner that dies ends the run (the launcher's timeout bounds
it). That is :class:`~cyberwave.workers.workflow_context.EdgeWorkflowContext`, and
it learns of a cancel the way an edge worker does: the backend's cancel command on
``cyberwave/workflow/<uuid>/cancel`` marks the execution canceled
(``workflow_executions.mark_canceled``) and ``is_canceled()`` reads that mark. Two
things are added. A message can be missed (no retry, and a runner may connect late),
so the execution's status is also read now and then (``fallback_seconds``) and marks
the same way. And a wait or a sleep checks the mark on every poll (``poll_seconds``),
which the edge context does not, so a cancel ends a navigation wait at once instead
of after it.

What a run needs comes from two reads on routes that already exist: the
execution (``GET /workflows/executions/{uuid}``: its status, the trigger data the
run was started with, and the source hash pinned at trigger time under
``metadata["runner"]["source_sha256"]``) and the workflow's cloud source
(``GET /workflows/{uuid}/compile``). A source whose hash differs from the pin is
refused: the workflow changed after the run was triggered. The run itself and its
outcome are the same ``run_with_context`` the journaling runner uses, so a run is
reported finished the same way.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from typing import Any

from ..._rest import request
from ...actions import ActionsClient
from ...exceptions import CyberwaveAPIError, CyberwaveError
from ...workflow_executions import WorkflowExecutionReporter
from ...workflows import TERMINAL_STATUSES
from ..hooks import workflow_cancel_topic
from ..workflow_context import (
    DEFAULT_ACTION_TIMEOUT_SECONDS,
    EdgeWorkflowContext,
    WorkflowCanceled,
)
from .segment import (
    CANCELED,
    COMPLETED,
    FAILED,
    SegmentOutcome,
    fail_before_start,
    run_with_context,
)

logger = logging.getLogger(__name__)

# How often a blocked run asks whether it was canceled. A navigation takes tens of
# seconds, so three seconds costs a cancel at most that much extra and a third of
# the requests of the edge worker's one second.
DEFAULT_CANCEL_POLL_SECONDS = 3.0
# How often the backend is asked for the execution's status, as the durable second
# source of a cancel next to the MQTT command.
DEFAULT_CANCEL_FALLBACK_SECONDS = 15.0
# The backend revokes a run's token when the run ends (a cancel included), so a status
# read that the backend refuses for credentials means the run is over, not that the
# backend hiccuped. One refusal can still be a blip: this many in a row end the run.
AUTH_REJECTED_STATUSES = (401, 403)
MAX_AUTH_REJECTIONS = 2

#: How long a run nothing else created waits for the backend to record it, and how
#: often it asks.
RECORDED_WAIT_SECONDS = 20.0
RECORDED_POLL_SECONDS = 0.5

#: ``execution.metadata[RUNNER_METADATA_KEY]["source_sha256"]``: what the backend
#: records when it triggers a run on the runner.
RUNNER_METADATA_KEY = "runner"
#: Where the run's params sit inside ``trigger_data`` (the backend's
#: ``TRIGGER_DATA_PARAMS_KEY``).
TRIGGER_PARAMS_KEY = "params"


def fetch_cloud_source(client: Any, workflow_uuid: str) -> tuple[str, str]:
    """``(source, sha256)`` of the workflow's cloud source, compiled by the backend."""
    _, body = request(
        client.api.api_client,
        "GET",
        "/api/v1/workflows/{uuid}/compile",
        path_params={"uuid": workflow_uuid},
        query={"target": "cloud"},
    )
    source = body.get("worker_source") if isinstance(body, dict) else None
    if not isinstance(source, str) or not source:
        raise CyberwaveError(f"workflow {workflow_uuid} has no cloud source")
    return source, str(body.get("source_sha256") or "")


def fetch_execution(client: Any, execution_uuid: str) -> dict[str, Any]:
    """The execution as the backend has it (status, trigger data, metadata)."""
    _, body = request(
        client.api.api_client,
        "GET",
        "/api/v1/workflows/executions/{uuid}",
        path_params={"uuid": execution_uuid},
    )
    if not isinstance(body, dict):
        raise CyberwaveError(f"execution {execution_uuid} could not be read")
    return body


def execution_is_canceled(client: Any, execution_uuid: str) -> bool:
    """Whether the backend has the execution canceled."""
    return fetch_execution(client, execution_uuid).get("status") == "canceled"


def envelope_from_execution(execution: Mapping[str, Any]) -> dict[str, Any]:
    """The run envelope the generated code reads (``inputs``, ``params``,
    ``requested_at``), the same shape an edge worker gets in its run command."""
    trigger_data = dict(execution.get("trigger_data") or {})
    return {
        "inputs": trigger_data,
        "params": trigger_data.get(TRIGGER_PARAMS_KEY) or {},
        "requested_at": execution.get("started_at"),
        "requested_by_uuid": execution.get("triggered_by"),
    }


def pinned_source_hash(execution: Mapping[str, Any]) -> str:
    pinned = (execution.get("metadata") or {}).get(RUNNER_METADATA_KEY) or {}
    return str(pinned.get("source_sha256") or "")


@contextmanager
def watch_cancel(
    client: Any, workflow_uuid: str, execution_uuid: str
) -> Iterator[None]:
    """Mark ``execution_uuid`` canceled while the block runs, when the backend's cancel
    command for it arrives on the workflow's cancel topic, like an edge worker.

    Without a connected MQTT client it does nothing (the status read still catches a
    cancel). The topic carries the cancels of every run of the workflow, so other
    executions' are ignored.
    """
    mqtt = getattr(client, "mqtt", None)
    if mqtt is None or not getattr(mqtt, "connected", False):
        logger.info("no MQTT connection: a cancel will be seen by the status read only")
        yield
        return
    topic = f"{getattr(mqtt, 'topic_prefix', '') or ''}{workflow_cancel_topic(workflow_uuid)}"
    key = ("cloud-runner-cancel", execution_uuid)
    manager = client.workflow_executions

    def on_cancel(payload: Any) -> None:
        if (
            isinstance(payload, Mapping)
            and str(payload.get("execution_uuid")) == execution_uuid
        ):
            manager.mark_canceled(execution_uuid)

    mqtt.subscribe(topic, on_cancel, qos=1, subscriber_key=key)
    try:
        yield
    finally:
        mqtt.unsubscribe(topic, key)


class RunnerContext(EdgeWorkflowContext):
    """The blocking edge context, made to end a wait when the run is canceled.

    ``is_canceled`` is the edge's (the mark on ``client.workflow_executions``), which
    ``watch_cancel`` sets from the cancel command. ``cancel_check`` returns whether
    the backend has the execution canceled; it is asked at most every
    ``fallback_seconds`` and marks the same way, and a check that fails is logged
    and counts as "not canceled", so a backend hiccup does not end a run holding a
    robot in motion. The exception is a refusal of the run's own credential
    (``AUTH_REJECTED_STATUSES``) ``MAX_AUTH_REJECTIONS`` times in a row: the backend
    revokes the token the moment the run ends, a cancel included, so neither this read
    nor the broker can still deliver the cancel and the run is treated as ended.
    """

    def __init__(
        self,
        client: Any,
        execution_uuid: str,
        *,
        cancel_check: Callable[[], bool] | None = None,
        params: Mapping[str, Any] | None = None,
        reporter: Any = None,
        poll_seconds: float = DEFAULT_CANCEL_POLL_SECONDS,
        fallback_seconds: float = DEFAULT_CANCEL_FALLBACK_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            client, execution_uuid, params=params, reporter=reporter, sleep=sleep
        )
        self._cancel_check = cancel_check or partial(
            execution_is_canceled, client, execution_uuid
        )
        self._poll_seconds = poll_seconds
        self._fallback_seconds = fallback_seconds
        self._monotonic = monotonic
        self._checked_at: float | None = None
        self._auth_rejections = 0

    def _read_status(self) -> None:
        now = self._monotonic()
        if (
            self._checked_at is not None
            and now - self._checked_at < self._fallback_seconds
        ):
            return
        self._checked_at = now
        try:
            canceled = self._cancel_check()
        except CyberwaveAPIError as exc:
            if exc.status_code not in AUTH_REJECTED_STATUSES:
                self._auth_rejections = 0
                logger.warning("could not check for cancel", exc_info=True)
                return
            self._auth_rejections += 1
            if self._auth_rejections < MAX_AUTH_REJECTIONS:
                logger.warning(
                    "could not check for cancel: the run's token was refused"
                )
                return
            logger.warning(
                "the run's token was refused %d times in a row: the backend ended "
                "execution %s",
                self._auth_rejections,
                self.execution_uuid,
            )
            self._client.workflow_executions.mark_canceled(self.execution_uuid)
            return
        except Exception:
            self._auth_rejections = 0
            logger.warning("could not check for cancel", exc_info=True)
            return
        self._auth_rejections = 0
        if canceled:
            self._client.workflow_executions.mark_canceled(self.execution_uuid)

    def is_canceled(self) -> bool:
        if super().is_canceled():
            return True
        self._read_status()
        return super().is_canceled()

    def _raise_if_canceled(self) -> None:
        if self.is_canceled():
            # Raised inside a node, so the step reports this text as the node's
            # error: say what happened rather than which run it was.
            raise WorkflowCanceled(
                f"execution {self.execution_uuid} was canceled while this node was running"
            )

    def wait_action(
        self,
        local_id: str,
        action_id: str,
        *,
        timeout: float = DEFAULT_ACTION_TIMEOUT_SECONDS,
        twin_uuid: str | None = None,
    ) -> dict[str, Any]:
        actions: ActionsClient = self._client.actions
        return actions.wait(
            action_id,
            twin_uuid=twin_uuid,
            timeout=timeout,
            poll_interval=self._poll_seconds,
            on_poll=self._raise_if_canceled,
        )

    def sleep(self, local_id: str, seconds: float) -> None:
        deadline = self._monotonic() + seconds
        while (left := deadline - self._monotonic()) > 0:
            self._raise_if_canceled()
            self._sleep(min(self._poll_seconds, left))


def _start_reporter(
    client: Any, status: str, workflow_uuid: str, execution_uuid: str
) -> Any:
    """The execution's reporter. A row the backend left ``REQUESTED`` is started,
    which moves it to ``RUNNING``."""
    if status == "requested":
        return client.workflow_executions.start(
            workflow_uuid=workflow_uuid,
            execution_uuid=execution_uuid,
            source_type=None,
        )
    return WorkflowExecutionReporter(
        client,
        workflow_uuid=workflow_uuid,
        execution_uuid=execution_uuid,
        source_type=None,
    )


def _wait_until_recorded(
    client: Any, execution_uuid: str, seconds: float = RECORDED_WAIT_SECONDS
) -> None:
    """Wait for the backend to have the execution a ``started`` event creates.

    Node events that reach the backend before its row exists are dropped, and events
    are ingested out of order, so a run nothing else created waits for its row.
    """
    deadline = time.monotonic() + seconds
    while True:
        try:
            fetch_execution(client, execution_uuid)
            return
        except Exception:  # noqa: BLE001 - not there yet, or a blip: try again
            if time.monotonic() >= deadline:
                logger.warning("execution %s was not recorded in time", execution_uuid)
                return
            time.sleep(RECORDED_POLL_SECONDS)


def _pinned_source(
    client: Any, workflow_uuid: str, execution: Mapping[str, Any]
) -> str:
    """The workflow's cloud source, refused when it no longer hashes to the pin."""
    source, digest = fetch_cloud_source(client, workflow_uuid)
    pinned = pinned_source_hash(execution)
    if pinned and digest != pinned:
        raise CyberwaveError(
            "the workflow changed after this run was triggered "
            f"(pinned source {pinned[:12]}, now {digest[:12]})"
        )
    return source


def run_execution(
    client: Any,
    *,
    workflow_uuid: str,
    execution_uuid: str,
    payload: Mapping[str, Any] | None = None,
    source: str | None = None,
    reporter: Any = None,
    cancel_check: Callable[[], bool] | None = None,
    poll_seconds: float = DEFAULT_CANCEL_POLL_SECONDS,
    fallback_seconds: float = DEFAULT_CANCEL_FALLBACK_SECONDS,
    execution: Mapping[str, Any] | None = None,
    wait_until_recorded: bool = False,
) -> SegmentOutcome:
    """Run one execution of a cloud workflow and report how it ended.

    The execution is read once (pass ``execution`` to skip the read). A run that
    is already over is not run again. Unless given, the payload is built from the
    execution's trigger data, the reporter is the execution's (sending ``started``
    when the backend left the row ``REQUESTED``, which moves it to ``RUNNING``),
    and the source is the workflow's cloud source, checked against the hash pinned
    at trigger time when there is one. A source passed in is used as is.
    ``wait_until_recorded`` holds the run until the backend has the execution, for a
    run whose ``started`` event is what creates it.
    """
    # Listen before reading the status: a cancel that lands in between is then seen
    # by one or the other.
    with watch_cancel(client, workflow_uuid, execution_uuid):
        if execution is None:
            execution = fetch_execution(client, execution_uuid)
        status = str(execution.get("status") or "")
        if status in TERMINAL_STATUSES:
            return SegmentOutcome(
                {"success": COMPLETED, "canceled": CANCELED}.get(status, FAILED),
                error=f"execution {execution_uuid} is already {status}",
            )
        if reporter is None:
            reporter = _start_reporter(client, status, workflow_uuid, execution_uuid)
            if wait_until_recorded:
                _wait_until_recorded(client, execution_uuid)
        if source is None:
            try:
                source = _pinned_source(client, workflow_uuid, execution)
            except Exception as exc:  # noqa: BLE001 - reported as the run's failure
                logger.warning(
                    "could not get the source of %s: %s", execution_uuid, exc
                )
                return fail_before_start(reporter, str(exc))
        envelope = {**envelope_from_execution(execution), **dict(payload or {})}
        context = RunnerContext(
            client,
            execution_uuid,
            cancel_check=cancel_check,
            params=envelope.get("params") or {},
            reporter=reporter,
            poll_seconds=poll_seconds,
            fallback_seconds=fallback_seconds,
        )
        return run_with_context(
            source,
            context,
            client=client,
            execution_uuid=execution_uuid,
            payload=envelope,
            reporter=reporter,
        )

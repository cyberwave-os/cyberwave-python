"""The cloud ``WorkflowContext``: journaled steps and durable waits.

Generated cloud code (``workflow_cloud_renderer``) runs as ``run(ctx, client,
payload)`` and calls this object instead of sleeping or polling. A run is a
series of *segments*: each one executes ``run`` from the top, a wait that is not
resolved yet ends the segment, and a later segment replays the run from the
journal and continues.

* A step that already completed returns its journaled result without running its
  function again, so replay never repeats a node's work.
* A side-effecting call made inside a step (:meth:`CloudWorkflowContext.call`) is
  journaled the moment it returns, so a segment that dies after the call and
  before the step finished does not repeat it. A crash between the call and its
  journal write still repeats it, which is why the call receives an idempotency
  key for APIs that honor one.
* :meth:`wait_action` and :meth:`sleep` wait in one of two ways. By default they
  register a wait and raise :class:`WorkflowSuspend`; whoever resolves the wait
  (the backend, on an action event or the deadline) stores its result, and the
  next segment returns or raises from it exactly as the blocking edge context
  would. With :class:`SyncWaits` they block in the segment instead, like the edge
  worker, up to a cap on how long one wait may take, and register nothing.

What a resolved wait stores, and what the node then sees, is defined in
``wait_results`` so the code that resolves waits and this module agree.

It talks to the journal through a ``JournalPort`` (a database in the backend, REST
in a runner), and uses the SDK's own ``WorkflowSuspend`` and ``WorkflowCanceled``;
the generated code never refers to them, and the segment runner is the only place
that catches them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..workflow_context import (
    DEFAULT_ACTION_TIMEOUT_SECONDS,
    WorkflowCanceled,
    WorkflowSuspend,
)
from .ports import JournalPort, JournalPortError
from .wait_results import (
    ACTION_WAIT,
    TIMER_WAIT,
    interpret_action_result,
    is_terminal_action,
    timed_out_result,
)

logger = logging.getLogger(__name__)

# The longest one wait may block a synchronous segment. Longer waits need to
# suspend, so the run fails with a clear message instead of holding a runner.
DEFAULT_MAX_SYNC_WAIT_SECONDS = 600.0

# How often a blocked segment polls an action, checks for cancel and heartbeats. The
# edge worker polls every second; a cloud runner pays an HTTP round trip for each
# beat, and a navigation takes tens of seconds, so three seconds costs a wait at
# most that much extra and a third of the requests.
DEFAULT_SYNC_POLL_SECONDS = 3.0


class WorkflowWaitTooLong(RuntimeError):
    """A wait is longer than a synchronous segment is allowed to block."""


@dataclass(frozen=True)
class SyncWaits:
    """Block in the segment on waits instead of suspending the run.

    ``action_status(action_id, twin_uuid)`` returns the action's current status
    payload (``None`` when unknown); a failed poll does not end the wait, as with
    the SDK's ``actions.wait``. ``heartbeat`` runs on every poll and while
    sleeping, for whoever holds the segment's lease to extend it. ``sleep`` and
    ``now`` are injectable for tests.
    """

    action_status: Callable[[str, str | None], Mapping[str, Any] | None]
    max_wait_seconds: float = DEFAULT_MAX_SYNC_WAIT_SECONDS
    poll_interval: float = DEFAULT_SYNC_POLL_SECONDS
    heartbeat: Callable[[], None] | None = None
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic

    def check_fits(self, what: str, seconds: float) -> None:
        if seconds > self.max_wait_seconds:
            raise WorkflowWaitTooLong(
                f"{what} of {seconds:g}s is longer than the {self.max_wait_seconds:g}s "
                "a cloud run can wait without suspending"
            )


@dataclass
class _OpenStep:
    node_uuid: str
    activation_index: int
    calls: dict[str, Any] = field(default_factory=dict)


class CloudWorkflowContext:
    """Replaying implementation of ``WorkflowContext`` over a ``JournalPort``.

    ``reporter`` is optional and has the shape of the SDK's
    ``WorkflowExecutionReporter`` (``node_started``, ``node_finished``,
    ``node_error``). A replayed step reports nothing: it was reported when it ran.
    """

    def __init__(
        self,
        port: JournalPort,
        execution_uuid: str,
        *,
        params: Mapping[str, Any] | None = None,
        reporter: Any = None,
        now: Callable[[], float] = time.time,
        sync_waits: SyncWaits | None = None,
    ) -> None:
        self._port = port
        self._sync = sync_waits
        self._reporter = reporter
        self._now = now
        self.execution_uuid = execution_uuid
        self.params: Mapping[str, Any] = dict(params or {})
        snapshot = port.snapshot()
        self._steps: dict[str, Any] = dict(snapshot.get("steps") or {})
        self._waits: dict[str, Any] = dict(snapshot.get("waits") or {})
        self._open: _OpenStep | None = None

    # -- cancel ----------------------------------------------------------------
    def is_canceled(self) -> bool:
        return self._port.is_canceled()

    # -- steps -----------------------------------------------------------------
    def step(
        self,
        node_uuid: str,
        fn: Callable[[], Any],
        *,
        activation_index: int = 0,
    ) -> Any:
        key = f"{node_uuid}:{activation_index}"
        done = self._steps.get(key)
        if done is not None and done.get("status") == "completed":
            return done["result"]
        if self.is_canceled():
            raise WorkflowCanceled(self.execution_uuid)

        opened = self._port.start_step(node_uuid, activation_index)
        if opened.get("status") == "completed":
            return opened["result"]
        if opened.get("status") == "failed":
            raise RuntimeError(opened.get("error") or f"step {key} failed earlier")

        self._open = _OpenStep(
            node_uuid, activation_index, dict(opened.get("calls") or {})
        )
        self._report("node_started", node_uuid, activation_index)
        try:
            result = fn()
        except (WorkflowSuspend, WorkflowCanceled):
            raise
        except Exception as exc:
            self._fail(node_uuid, activation_index, exc)
            raise
        finally:
            self._open = None

        try:
            stored = self._port.complete_step(node_uuid, activation_index, result)
        except Exception as exc:
            # A result the journal cannot store will not become storable on a
            # retry, so the step has failed; anything else (a lost connection)
            # leaves the step open for the next segment to finish.
            if isinstance(exc, JournalPortError) and exc.code == "invalid":
                self._fail(node_uuid, activation_index, exc)
            raise
        self._report(
            "node_finished",
            node_uuid,
            activation_index,
            output_data=stored.get("output_data"),
        )
        return stored

    def _fail(self, node_uuid: str, activation_index: int, exc: Exception) -> None:
        self._port.fail_step(node_uuid, activation_index, str(exc))
        self._report("node_error", node_uuid, activation_index, error=str(exc))

    def _report(
        self, method: str, node_uuid: str, activation_index: int, **kwargs: Any
    ) -> None:
        if self._reporter is None:
            return
        try:
            getattr(self._reporter, method)(
                node_uuid, activation_index=activation_index, **kwargs
            )
        except Exception:  # reporting is best effort, like the SDK's reporter
            logger.warning(
                "could not report %s for %s", method, node_uuid, exc_info=True
            )

    def _current(self, what: str) -> _OpenStep:
        if self._open is None:
            raise RuntimeError(f"{what} is only valid inside a step")
        return self._open

    # -- side effects ----------------------------------------------------------
    def call(self, call_id: str, fn: Callable[[str], Any]) -> Any:
        step = self._current("call()")
        if call_id in step.calls:
            return step.calls[call_id]
        idempotency_key = (
            f"{self.execution_uuid}:{step.node_uuid}:{step.activation_index}:{call_id}"
        )
        value = fn(idempotency_key)
        stored = self._port.record_call(
            step.node_uuid, step.activation_index, call_id, value
        )
        step.calls[call_id] = stored
        return stored

    # -- waits -----------------------------------------------------------------
    def _await(
        self, local_id: str, kind: str, key: str | None, deadline: float
    ) -> dict[str, Any]:
        """The resolved wait for this call site, registered on first use.

        Raises :class:`WorkflowSuspend` while the wait has not resolved.
        """
        step = self._current("a wait")
        wait_id = f"{step.node_uuid}:{step.activation_index}:{local_id}"
        wait = self._waits.get(wait_id) or self._port.register_wait(
            step.node_uuid,
            step.activation_index,
            wait_id,
            kind,
            key or wait_id,
            deadline,
        )
        if wait.get("status") != "resolved":
            raise WorkflowSuspend(wait_id)
        return wait

    def wait_action(
        self,
        local_id: str,
        action_id: str,
        *,
        timeout: float = DEFAULT_ACTION_TIMEOUT_SECONDS,
        twin_uuid: str | None = None,  # the SDK signature; the wait is keyed by action
    ) -> dict[str, Any]:
        if self._sync is not None:
            return self._block_on_action(self._sync, action_id, timeout, twin_uuid)
        wait = self._await(local_id, ACTION_WAIT, action_id, self._now() + timeout)
        return interpret_action_result(wait.get("result"), action_id, timeout)

    def sleep(self, local_id: str, seconds: float) -> None:
        self._current("sleep()")
        if seconds <= 0:
            return
        if self._sync is not None:
            self._block_for(self._sync, seconds)
        else:
            self._await(local_id, TIMER_WAIT, None, self._now() + seconds)

    # -- synchronous waits -------------------------------------------------------
    def _tick(self, sync: SyncWaits) -> None:
        """One beat of a long wait: stay alive, and stop if the run was canceled."""
        if sync.heartbeat is not None:
            sync.heartbeat()
        if self.is_canceled():
            raise WorkflowCanceled(self.execution_uuid)

    def _block_for(self, sync: SyncWaits, seconds: float) -> None:
        sync.check_fits("sleep", seconds)
        deadline = sync.now() + seconds
        while (left := deadline - sync.now()) > 0:
            self._tick(sync)
            sync.sleep(min(sync.poll_interval, left))

    def _block_on_action(
        self, sync: SyncWaits, action_id: str, timeout: float, twin_uuid: str | None
    ) -> dict[str, Any]:
        sync.check_fits("action wait", timeout)
        deadline = sync.now() + timeout
        while sync.now() < deadline:
            self._tick(sync)
            try:
                status = sync.action_status(action_id, twin_uuid)
            except Exception:  # one failed poll must not end the wait
                logger.warning("could not poll action %s", action_id, exc_info=True)
                status = None
            if status is not None and is_terminal_action(status):
                return interpret_action_result(status, action_id, timeout)
            sync.sleep(sync.poll_interval)
        return interpret_action_result(timed_out_result(), action_id, timeout)

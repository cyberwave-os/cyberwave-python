"""Run one segment of a cloud workflow.

``run_segment`` loads the compiled module, runs its ``run(ctx, client, payload)``
once against a journal and says how the segment ended:

* ``completed``: ``run`` returned; the outputs are in the outcome;
* ``suspended``: a wait is pending (``wait_id``); the segment is over and a later
  one continues once the wait resolves (never with ``sync_waits``, which blocks
  on waits instead);
* ``canceled``: the execution was canceled before or between steps;
* ``failed``: a node raised, or the run could not start; ``error`` says why.

A segment that ends the run also reports it finished through ``reporter`` (the
SDK's ``WorkflowExecutionReporter``). That report is what persists the run's
terminal status, through the same ingress every runtime uses
(``record_workflow_finished``, terminal-wins), so a caller that wants the status
stored passes a reporter: with none, the outcome is returned and nothing else
happens. The lease, the wake-up flag and the timers belong to whoever launches
segments, not to the segment: this function runs, replays and stops, nothing
more. A crash is not caught here on purpose: the process simply ends, the
journal holds what finished, and the next segment replays it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..workflow_context import WorkflowCanceled, WorkflowSuspend
from .context import CloudWorkflowContext, SyncWaits
from .ports import JournalPort

logger = logging.getLogger(__name__)

COMPLETED = "completed"
SUSPENDED = "suspended"
CANCELED = "canceled"
FAILED = "failed"

_REPORTED_STATUS = {COMPLETED: "success", CANCELED: "canceled", FAILED: "error"}


@dataclass(frozen=True)
class SegmentOutcome:
    status: str
    wait_id: str | None = None
    error: str | None = None
    outputs: Mapping[str, Any] | None = None


def load_cloud_workflow(source: str) -> Callable[..., Any]:
    """The ``run`` function of a compiled cloud worker."""
    namespace: dict[str, Any] = {"__name__": "wf_generated"}
    exec(compile(source, "<cloud-worker>", "exec"), namespace)  # noqa: S102
    run = namespace.get("run")
    if not callable(run):
        raise ValueError(  # noqa: TRY004
            "a compiled cloud worker must define run(ctx, client, payload)"
        )
    return run


def run_with_context(
    source: str,
    context: Any,
    *,
    client: Any,
    execution_uuid: str,
    payload: Mapping[str, Any] | None = None,
    reporter: Any = None,
) -> SegmentOutcome:
    """Run the compiled module once against ``context`` and report how it ended.

    ``run_segment`` is this over the journaling context; a runner with no journal
    passes a blocking one (see ``runner.RunnerContext``).
    """
    outcome = _call(source, context, client, execution_uuid, dict(payload or {}))
    if outcome.status != SUSPENDED:
        _report_finished(reporter, outcome)
    return outcome


def _call(
    source: str,
    context: Any,
    client: Any,
    execution_uuid: str,
    payload: dict[str, Any],
) -> SegmentOutcome:
    try:
        run = load_cloud_workflow(source)
        outputs = run(context, client, payload)
    except WorkflowSuspend as suspended:
        return SegmentOutcome(SUSPENDED, wait_id=suspended.wait_id)
    except WorkflowCanceled:
        return SegmentOutcome(CANCELED)
    except Exception as exc:  # noqa: BLE001 - any node error ends the run, not the process
        logger.info("run %s failed: %s", execution_uuid, exc)
        return SegmentOutcome(FAILED, error=str(exc) or type(exc).__name__)
    return SegmentOutcome(COMPLETED, outputs=outputs)


def fail_before_start(reporter: Any, error: str) -> SegmentOutcome:
    """End a run that could not begin (no usable source) and report it."""
    outcome = SegmentOutcome(FAILED, error=error)
    _report_finished(reporter, outcome)
    return outcome


def run_segment(
    source: str,
    *,
    port: JournalPort,
    client: Any,
    execution_uuid: str,
    payload: Mapping[str, Any] | None = None,
    reporter: Any = None,
    now: Callable[[], float] = time.time,
    sync_waits: SyncWaits | None = None,
) -> SegmentOutcome:
    envelope = dict(payload or {})
    outcome = _execute(
        source, port, client, execution_uuid, envelope, reporter, now, sync_waits
    )
    if outcome.status != SUSPENDED:
        _report_finished(reporter, outcome)
    return outcome


def _execute(
    source: str,
    port: JournalPort,
    client: Any,
    execution_uuid: str,
    payload: dict[str, Any],
    reporter: Any,
    now: Callable[[], float],
    sync_waits: SyncWaits | None,
) -> SegmentOutcome:
    if port.is_canceled():
        return SegmentOutcome(CANCELED)
    try:
        context = CloudWorkflowContext(
            port,
            execution_uuid,
            params=payload.get("params") or {},
            reporter=reporter,
            now=now,
            sync_waits=sync_waits,
        )
    except Exception as exc:  # noqa: BLE001 - the journal could not be read
        logger.info("segment of %s failed: %s", execution_uuid, exc)
        return SegmentOutcome(FAILED, error=str(exc) or type(exc).__name__)
    return _call(source, context, client, execution_uuid, payload)


def _report_finished(reporter: Any, outcome: SegmentOutcome) -> None:
    if reporter is None:
        return
    try:
        reporter.finished(
            status=_REPORTED_STATUS[outcome.status], error_message=outcome.error
        )
    except Exception:  # best effort, like every report
        logger.warning("could not report the run finished", exc_info=True)

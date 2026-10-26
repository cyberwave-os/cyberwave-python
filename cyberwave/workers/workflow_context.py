"""``WorkflowContext``: what generated workflow code calls instead of blocking.

A compiled workflow reaches the outside world through one small interface so
the same generated code can run in two places that wait differently:

* **Edge** (:class:`EdgeWorkflowContext`): a resident process on a device.
  Waits block, calls run inline. This is the behaviour edge workers have
  always had.
* **Cloud**: a short-lived job. A wait ends the process and a later resume
  replays the run from the top, so every method here has to be replay-safe:
  completed steps return their recorded values, and a pending wait raises
  :class:`WorkflowSuspend`. The cloud implementation lives in the Cyberwave
  backend, not in this SDK.

Wait semantics are part of the contract, not of an implementation:
:meth:`WorkflowContext.wait_action` returns the terminal status of the action,
raises ``RuntimeError`` when the action finished unsuccessfully and raises
``TimeoutError`` when the deadline passes first, exactly like
``client.actions.wait``. Generated code can therefore catch the same
exceptions on both targets.

The ids passed to :meth:`call`, :meth:`wait_action` and :meth:`sleep` must be
stable for a given call site: a cloud resume matches them against what it
recorded the first time.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..client import Cyberwave

DEFAULT_ACTION_TIMEOUT_SECONDS = 120.0


class WorkflowSuspend(BaseException):
    """Raised by a context to end the current run segment while a wait is pending.

    A ``BaseException`` on purpose: node code that catches ``Exception`` must
    not turn a suspension into a node failure.
    """

    def __init__(self, wait_id: str) -> None:
        super().__init__(wait_id)
        self.wait_id = wait_id


class WorkflowCanceled(Exception):
    """Raised when the execution was canceled before or between steps."""


@runtime_checkable
class WorkflowContext(Protocol):
    """The surface generated workflow code uses. See the module docstring."""

    execution_uuid: str
    params: Mapping[str, Any]

    def is_canceled(self) -> bool:
        """Whether the execution has been canceled."""
        ...

    def step(
        self,
        node_uuid: str,
        fn: Callable[[], Any],
        *,
        activation_index: int = 0,
    ) -> Any:
        """Run one node and return what ``fn`` returned.

        ``fn`` returns a JSON-serialisable dict, the *step result*:
        ``{"output": <what the node wrote to its output slot>,
        "input_data": [...], "output_data": [...]}``. ``input_data`` and
        ``output_data`` are what the node reports to the platform; ``output`` is
        what downstream nodes read, and generated code restores it from the
        returned dict. On a replaying context a node that already completed
        returns its recorded result without running ``fn`` again.
        """
        ...

    def call(self, call_id: str, fn: Callable[[str], Any]) -> Any:
        """Run a side-effecting call made inside a node, once per execution.

        ``fn`` receives an idempotency key derived from the execution and
        ``call_id``; an API that honors it performs the effect once even when
        a crash forces ``fn`` to run again.
        """
        ...

    def wait_action(
        self,
        local_id: str,
        action_id: str,
        *,
        timeout: float = DEFAULT_ACTION_TIMEOUT_SECONDS,
        twin_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Wait for an action to reach a terminal status (see module docstring)."""
        ...

    def sleep(self, local_id: str, seconds: float) -> None:
        """Pause the run for ``seconds``."""
        ...


class NodeReporter(Protocol):
    """The subset of ``WorkflowExecutionReporter`` that ``step`` reports through."""

    def node_started(self, node_uuid: str, **kwargs: Any) -> None: ...

    def node_finished(self, node_uuid: str, **kwargs: Any) -> None: ...

    def node_error(self, node_uuid: str, **kwargs: Any) -> None: ...


class EdgeWorkflowContext:
    """Blocking implementation for a resident edge worker."""

    def __init__(
        self,
        client: Cyberwave,
        execution_uuid: str,
        *,
        params: Mapping[str, Any] | None = None,
        reporter: NodeReporter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._reporter = reporter
        self._sleep = sleep
        self.execution_uuid = execution_uuid
        self.params: Mapping[str, Any] = dict(params or {})

    def is_canceled(self) -> bool:
        return bool(self._client.workflow_executions.is_canceled(self.execution_uuid))

    def step(
        self,
        node_uuid: str,
        fn: Callable[[], Any],
        *,
        activation_index: int = 0,
    ) -> Any:
        if self.is_canceled():
            raise WorkflowCanceled(self.execution_uuid)
        reporter = self._reporter
        if reporter is not None:
            reporter.node_started(node_uuid, activation_index=activation_index)
        try:
            output = fn()
        except Exception as exc:
            if reporter is not None:
                reporter.node_error(
                    node_uuid, error=str(exc), activation_index=activation_index
                )
            raise
        if reporter is not None:
            output_data = (
                output.get("output_data") if isinstance(output, dict) else None
            )
            reporter.node_finished(
                node_uuid, output_data=output_data, activation_index=activation_index
            )
        return output

    def call(self, call_id: str, fn: Callable[[str], Any]) -> Any:
        return fn(f"{self.execution_uuid}:{call_id}")

    def wait_action(
        self,
        local_id: str,
        action_id: str,
        *,
        timeout: float = DEFAULT_ACTION_TIMEOUT_SECONDS,
        twin_uuid: str | None = None,
    ) -> dict[str, Any]:
        return self._client.actions.wait(
            action_id, twin_uuid=twin_uuid, timeout=timeout
        )

    def sleep(self, local_id: str, seconds: float) -> None:
        self._sleep(seconds)

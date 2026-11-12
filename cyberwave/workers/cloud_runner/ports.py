"""The journal as the cloud runner sees it.

A runner never touches the database: it reads and writes the run journal through
a ``JournalPort``. The REST journal API (``api/workflow_execution_journal.py``)
is one implementation, reached over HTTP with the run token; ``InProcessJournalPort``
in ``inprocess.py`` is the other, for tests and for deployments that run the
runner next to the backend.

Every method returns the JSON shapes the REST API returns, so the two
implementations are interchangeable and the context never sees an ORM object.
A port reports a refusal by raising ``JournalPortError``, whose ``code`` an HTTP
implementation derives from the response status (``InProcessJournalPort`` translates
the backend's ``JournalError`` into it).
"""

from __future__ import annotations

from typing import Any, Protocol


class JournalPortError(Exception):
    """A port refused what was asked. ``code`` says why: ``not_found``,
    ``invalid`` (a malformed request, or a result the journal cannot store),
    ``conflict`` or ``finished``. An HTTP implementation derives it from the
    response status."""

    def __init__(self, message: str, code: str = "invalid") -> None:
        super().__init__(message)
        self.code = code


class JournalPort(Protocol):
    """One run's journal. The execution is bound when the port is built."""

    def snapshot(self) -> dict[str, Any]:
        """``{"status", "steps", "waits", "pending_wait"}``; steps are keyed
        ``"<node uuid>:<activation index>"``."""
        ...

    def is_canceled(self) -> bool: ...

    def start_step(self, node_uuid: str, activation_index: int) -> dict[str, Any]:
        """``{"status", "result", "error", "calls"}``. Idempotent."""
        ...

    def complete_step(
        self, node_uuid: str, activation_index: int, result: Any
    ) -> dict[str, Any]:
        """Journal a step result; returns its stored form."""
        ...

    def fail_step(self, node_uuid: str, activation_index: int, error: str) -> None: ...

    def record_call(
        self, node_uuid: str, activation_index: int, call_id: str, value: Any
    ) -> Any:
        """Journal a side-effecting call. The first value recorded wins and is
        what every later call returns."""
        ...

    def register_wait(
        self,
        node_uuid: str,
        activation_index: int,
        wait_id: str,
        kind: str,
        key: str,
        deadline: float,
    ) -> dict[str, Any]:
        """Register a wait (``deadline`` is epoch seconds). The first
        registration of a wait id pins its deadline."""
        ...

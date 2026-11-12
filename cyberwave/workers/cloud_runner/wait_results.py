"""What a resolved wait stores, and what the waiting node sees.

The code that resolves a wait (the backend, on an action event or at the
deadline) and the code that reads it (``CloudWorkflowContext``) are different
modules, and the stored result is the contract between them, so both sides use the
builders and the reader here.

* action, finished: the action's status payload; ``status == "completed"``
  returns it, any other terminal ``status`` raises ``RuntimeError``;
* action, deadline passed: :func:`timed_out_result`, which raises ``TimeoutError``;
* timer: :func:`elapsed_result` (the context ignores what a timer stores).

The exceptions and their messages are the ones ``client.actions.wait`` raises, so
generated code catches the same things on both targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...actions import TERMINAL_STATUSES

ACTION_WAIT = "action"
TIMER_WAIT = "timer"


def is_terminal_action(payload: Mapping[str, Any] | None) -> bool:
    return str((payload or {}).get("status") or "").lower() in TERMINAL_STATUSES


def timed_out_result() -> dict[str, Any]:
    return {"timed_out": True}


def elapsed_result() -> dict[str, Any]:
    return {"elapsed": True}


def interpret_action_result(
    result: Mapping[str, Any] | None, action_id: str, timeout: float
) -> dict[str, Any]:
    """The value ``wait_action`` returns for a resolved action wait, or the
    exception it raises."""
    result = result or {}
    if result.get("timed_out"):
        raise TimeoutError(
            f"action {action_id} did not reach a terminal status within {timeout}s"
        )
    status = str(result.get("status") or "").lower()
    if status != "completed":
        raise RuntimeError(
            f"action {action_id} finished with status {status!r}: "
            f"{result.get('message')!r}"
        )
    return dict(result)

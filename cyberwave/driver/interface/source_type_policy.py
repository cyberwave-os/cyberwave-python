"""Source-type policy for driver interfaces.

Convention (see docs ``base-driver-class.mdx`` → *Source-type convention*):

- Drivers **publish** state as ``edge`` (physical feedback) or ``sim`` (simulator).
- Drivers **listen** for teleop commands: ``tele``, ``edit``, ``sim_tele``.
- A driver must **never** act on ``edge*`` messages inbound — that is its own
  feedback echoing back, and actuating on it makes the robot fight itself.
- Inbound messages **without** a ``source_type`` are accepted leniently (treated
  as commands), because not every producer stamps the field. The ``edge*``
  self-echo guard still applies even in this relaxed case.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from cyberwave.constants import (
    EDGE_STATE_SOURCE_TYPES,
    SOURCE_TYPE_EDGE,
    SOURCE_TYPE_EDIT,
    SOURCE_TYPE_SIM,
    SOURCE_TYPE_SIM_TELE,
    SOURCE_TYPE_TELE,
)

#: Inbound command source types a teleop listener accepts by default.
COMMAND_SOURCE_TYPES: frozenset[str] = frozenset(
    {SOURCE_TYPE_TELE, SOURCE_TYPE_EDIT, SOURCE_TYPE_SIM_TELE}
)

#: Default outbound source type for a driver's published state.
DEFAULT_PUBLISH_SOURCE_TYPE: str = SOURCE_TYPE_EDGE

#: Default outbound source type when the driver runs against a simulator.
DEFAULT_SIM_PUBLISH_SOURCE_TYPE: str = SOURCE_TYPE_SIM


def accepts_inbound(allowed: frozenset[str], source_type: str | None) -> bool:
    """Return True if a command listener should process this inbound message.

    Lenient on absence, strict on presence, and the ``edge*`` self-echo guard is
    non-overridable (even if ``edge`` appears in *allowed*).
    """
    if source_type in EDGE_STATE_SOURCE_TYPES:
        return False
    if source_type is None:
        return True
    return source_type in allowed


def filtered_listener(
    callback: Callable[[dict[str, Any]], Any],
    allowed: Iterable[str] | None,
) -> Callable[[dict[str, Any]], Any]:
    """Wrap *callback* so inbound messages failing the source-type policy are dropped.

    ``allowed=None`` means no source-type filtering — the callback is returned
    unchanged so legacy listeners keep their current behavior. Otherwise the
    wrapper applies :func:`accepts_inbound`; dropped messages return ``None`` so
    an awaiting dispatcher no-ops cleanly, accepted messages return the callback's
    own result (which may be a coroutine).
    """
    if allowed is None:
        return callback
    allowed_set = frozenset(allowed)

    def wrapper(envelope: dict[str, Any]) -> Any:
        if not accepts_inbound(allowed_set, envelope.get("source_type")):
            return None
        return callback(envelope)

    return wrapper

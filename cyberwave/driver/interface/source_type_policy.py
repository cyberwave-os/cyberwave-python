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

import functools
import logging
import threading
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

logger = logging.getLogger(__name__)

#: Inbound command source types a teleop listener accepts by default.
COMMAND_SOURCE_TYPES: frozenset[str] = frozenset(
    {SOURCE_TYPE_TELE, SOURCE_TYPE_EDIT, SOURCE_TYPE_SIM_TELE}
)

#: "May this runtime, in live mode, act on this navigation command?" —
#: ``substrate == hardware && role == absent``, deliberately *not*
#: ``direction == command``: ``edge`` is a state value, admitted because
#: navigation is a server-owned channel (see :func:`accepts_navigation_command`).
#: That exception is why this needs a name distinct from the backend's broader
#: ``HARDWARE_SUBSTRATE_SOURCE_TYPES``, which admits both role variants.
LIVE_NAVIGATION_COMMAND_SOURCE_TYPES: frozenset[str] = frozenset(
    {SOURCE_TYPE_EDGE, SOURCE_TYPE_TELE}
)
#: Its simulation counterpart: exactly ``substrate == sim``, both directions.
SIMULATION_NAVIGATION_SOURCE_TYPES: frozenset[str] = frozenset(
    {SOURCE_TYPE_SIM, SOURCE_TYPE_SIM_TELE}
)

#: Default outbound source type for a driver's published state.
DEFAULT_PUBLISH_SOURCE_TYPE: str = SOURCE_TYPE_EDGE

#: Default outbound source type when the driver runs against a simulator.
DEFAULT_SIM_PUBLISH_SOURCE_TYPE: str = SOURCE_TYPE_SIM


#: ``(guard, predicate, source_type, topic)`` -> count. Diagnostics only; nothing
#: gates on it. It exists to size a later move to fail-closed defaults, which
#: cannot be judged while a guard drops silently.
_PROVENANCE_DROPS: dict[tuple[str, str, str, str], int] = {}

#: Listeners already reported as having no filter, so each is logged once.
_UNFILTERED_LISTENERS: set[str] = set()
_DROP_LOCK = threading.Lock()

#: Cap on distinct counter keys. ``source_type`` is read straight off the wire, so
#: a publisher emitting a fresh value per message would otherwise grow this map for
#: the life of the driver and earn a WARNING each time. Kept identical to the
#: backend's ``src/lib/provenance_drops.py``.
_MAX_DROP_KEYS = 2048

#: Stands in for a dimension folded away once the counter is at capacity.
_OVERFLOW = "<over-cap>"

_capped = False


def _publisher_of(envelope: dict[str, Any] | None) -> str:
    """Best available publisher identity — no single field carries it."""
    if not envelope:
        return "unknown"
    for field in ("source_subtype", "sender", "workload_uuid"):
        value = envelope.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _is_log_milestone(count: int) -> bool:
    """Whether this running total earns a WARNING: 1, 10, 100, 1000, ...

    Logging only the first drop caps flood but hides scale, and DEBUG is off in
    most deployments — a guard rejecting continuously would then be worth exactly
    one line for the life of the process. Decades keep the volume logarithmic while
    carrying the running total into the logs. Kept identical to the backend's
    ``src/lib/provenance_drops.py``.
    """
    if count <= 0:
        return False
    while count % 10 == 0:
        count //= 10
    return count == 1


def record_provenance_drop(
    *,
    guard: str,
    predicate: str,
    source_type: str | None,
    topic: str | None = None,
    envelope: dict[str, Any] | None = None,
) -> None:
    """Log and count one rejected message.

    *predicate* is the rule that rejected it, so a log line names the gate that
    fired. Milestone drops log at WARNING and the rest at DEBUG — these arrive at
    telemetry rates — while the counter keeps the true total.
    """
    global _capped
    key = (guard, predicate, str(source_type), topic or "unknown")
    with _DROP_LOCK:
        count = _PROVENANCE_DROPS.get(key)
        if count is None and len(_PROVENANCE_DROPS) >= _MAX_DROP_KEYS:
            # A key already present is always incremented, so a flood of one-shot
            # source types cannot displace the recurring drops worth measuring.
            key = (guard, predicate, _OVERFLOW, _OVERFLOW)
            count = _PROVENANCE_DROPS.get(key)
            first_fold, _capped = not _capped, True
        else:
            first_fold = False
        count = (count or 0) + 1
        _PROVENANCE_DROPS[key] = count
    if first_fold:
        logger.warning(
            "provenance counter at capacity: %d distinct keys; keys not already "
            "seen are folded into one %r bucket, so totals stay correct but stop "
            "being attributable",
            _MAX_DROP_KEYS,
            _OVERFLOW,
        )
    milestone = _is_log_milestone(count)
    if not milestone and not logger.isEnabledFor(logging.DEBUG):
        # Skip resolving the publisher for a line nobody will see.
        return
    log = logger.warning if milestone else logger.debug
    log(
        "provenance drop: guard=%s predicate=%r source_type=%r topic=%s "
        "publisher=%s count=%d",
        guard,
        predicate,
        source_type,
        topic or "unknown",
        _publisher_of(envelope),
        count,
    )


def provenance_drop_counts() -> dict[tuple[str, str, str, str], int]:
    """Snapshot of the rejection counters, for diagnostics and tests."""
    with _DROP_LOCK:
        return dict(_PROVENANCE_DROPS)


def reset_provenance_drop_counts() -> None:
    """Clear the counters and the unfiltered-listener log memory."""
    global _capped
    with _DROP_LOCK:
        _PROVENANCE_DROPS.clear()
        _UNFILTERED_LISTENERS.clear()
        _capped = False


@functools.lru_cache(maxsize=64)
def _not_in_allowed(allowed: frozenset[str]) -> str:
    """The drop predicate for *allowed*, built once per set rather than per drop."""
    return f"source_type not in allowed={sorted(allowed)}"


def accepts_inbound(
    allowed: frozenset[str],
    source_type: str | None,
    *,
    topic: str | None = None,
    envelope: dict[str, Any] | None = None,
) -> bool:
    """Return True if a command listener should process this inbound message.

    Lenient on absence, strict on presence, and the ``edge*`` self-echo guard is
    non-overridable (even if ``edge`` appears in *allowed*).

    *topic* and *envelope* only sharpen the drop log; the decision ignores them, so
    existing two-argument callers keep working.
    """
    if source_type in EDGE_STATE_SOURCE_TYPES:
        record_provenance_drop(
            guard="accepts_inbound",
            predicate="self-echo: direction == state && substrate == hardware",
            source_type=source_type,
            topic=topic,
            envelope=envelope,
        )
        return False
    if source_type is None:
        return True
    if source_type not in allowed:
        record_provenance_drop(
            guard="accepts_inbound",
            predicate=_not_in_allowed(frozenset(allowed)),
            source_type=source_type,
            topic=topic,
            envelope=envelope,
        )
        return False
    return True


def accepts_navigation_command(
    envelope: dict[str, Any],
    runtime_mode: str | None,
    *,
    topic: str | None = None,
) -> bool:
    """Return whether this runtime owns a navigation command.

    Navigation is a server-owned command channel, so ``edge`` is a valid live
    source. Missing and unknown sources fail closed. ``stop`` remains
    cross-runtime because it can only reduce motion.

    *topic* only sharpens the drop log.
    """

    if str(envelope.get("command") or "").strip().lower() == "stop":
        return True
    mode = str(runtime_mode or "").strip().lower()
    if mode == "simulation":
        allowed = SIMULATION_NAVIGATION_SOURCE_TYPES
    elif mode == "live":
        allowed = LIVE_NAVIGATION_COMMAND_SOURCE_TYPES
    else:
        record_provenance_drop(
            guard="accepts_navigation_command",
            predicate=f"unknown runtime_mode={runtime_mode!r} fails closed",
            source_type=envelope.get("source_type"),
            topic=topic,
            envelope=envelope,
        )
        return False
    source_type = str(envelope.get("source_type") or "").strip().lower()
    if source_type not in allowed:
        record_provenance_drop(
            guard="accepts_navigation_command",
            predicate=f"mode={mode} requires source_type in {sorted(allowed)}",
            source_type=envelope.get("source_type"),
            topic=topic,
            envelope=envelope,
        )
        return False
    return True


def filtered_listener(
    callback: Callable[[dict[str, Any]], Any],
    allowed: Iterable[str] | None,
    *,
    topic: str | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """Wrap *callback* so inbound messages failing the source-type policy are dropped.

    ``allowed=None`` means no source-type filtering — the callback is returned
    unchanged so legacy listeners keep their current behavior. Otherwise the
    wrapper applies :func:`accepts_inbound`; dropped messages return ``None`` so
    an awaiting dispatcher no-ops cleanly, accepted messages return the callback's
    own result (which may be a coroutine).

    A listener declaring no ``source_types`` has **no** filter, not a lenient one:
    the unwrapped return skips :func:`accepts_inbound` entirely, so even the
    non-overridable ``edge*`` self-echo guard never runs. That is the default for
    nearly every listener in the tree.

    Whether it should be deny is open, and deliberately not changed here — flipping
    it would silently stop undeclared drivers from responding. The log below is the
    measurement that has to come first; grep ``no provenance filter``.
    """
    if allowed is None:
        name = getattr(callback, "__qualname__", None) or repr(callback)
        key = f"{name}@{topic or 'unknown'}"
        with _DROP_LOCK:
            first = key not in _UNFILTERED_LISTENERS
            _UNFILTERED_LISTENERS.add(key)
        if first:
            logger.warning(
                "no provenance filter: listener=%s topic=%s declares no "
                "source_types, so every inbound message reaches it — including "
                "edge* self-echo, which accepts_inbound would have dropped",
                name,
                topic or "unknown",
            )
        return callback
    allowed_set = frozenset(allowed)

    def wrapper(envelope: dict[str, Any]) -> Any:
        if not accepts_inbound(
            allowed_set,
            envelope.get("source_type"),
            topic=topic,
            envelope=envelope,
        ):
            return None
        return callback(envelope)

    return wrapper

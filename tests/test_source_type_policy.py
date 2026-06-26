"""Tests for the driver source-type policy (publish edge/sim, listen tele; relaxed)."""

from cyberwave.constants import (
    SOURCE_TYPE_EDGE,
    SOURCE_TYPE_EDGE_LEADER,
    SOURCE_TYPE_EDIT,
    SOURCE_TYPE_SIM_TELE,
    SOURCE_TYPE_TELE,
)
from cyberwave.driver.interface.source_type_policy import (
    COMMAND_SOURCE_TYPES,
    accepts_inbound,
    filtered_listener,
)


def test_command_source_types_are_the_teleop_set():
    assert COMMAND_SOURCE_TYPES == frozenset(
        {SOURCE_TYPE_TELE, SOURCE_TYPE_EDIT, SOURCE_TYPE_SIM_TELE}
    )


def test_accepts_allowed_command_source():
    assert accepts_inbound(COMMAND_SOURCE_TYPES, SOURCE_TYPE_TELE) is True


def test_accepts_missing_source_type_relaxed():
    # Not every producer stamps source_type; untagged is treated as a command.
    assert accepts_inbound(COMMAND_SOURCE_TYPES, None) is True


def test_rejects_edge_self_echo_even_when_relaxed():
    assert accepts_inbound(COMMAND_SOURCE_TYPES, SOURCE_TYPE_EDGE) is False
    assert accepts_inbound(COMMAND_SOURCE_TYPES, SOURCE_TYPE_EDGE_LEADER) is False


def test_rejects_present_but_not_allowed():
    assert accepts_inbound(frozenset({SOURCE_TYPE_TELE}), SOURCE_TYPE_EDIT) is False


def test_edge_guard_holds_even_if_caller_lists_edge_as_allowed():
    # The self-echo guard is non-overridable: a driver can never actuate on edge*.
    assert accepts_inbound(frozenset({SOURCE_TYPE_EDGE}), SOURCE_TYPE_EDGE) is False


def _recording_callback():
    seen = []
    return seen, lambda envelope: seen.append(envelope)


def test_filtered_listener_none_allowed_is_passthrough():
    # Legacy listeners (no declared source_types) keep current behavior.
    seen, cb = _recording_callback()
    wrapped = filtered_listener(cb, None)
    assert wrapped is cb
    wrapped({"source_type": SOURCE_TYPE_EDGE})
    assert len(seen) == 1


def test_filtered_listener_drops_disallowed_and_edge():
    seen, cb = _recording_callback()
    wrapped = filtered_listener(cb, COMMAND_SOURCE_TYPES)
    wrapped({"source_type": SOURCE_TYPE_EDGE, "joint_1": 0.1})  # self-echo → dropped
    wrapped({"source_type": "bogus"})  # unknown → dropped
    assert seen == []


def test_filtered_listener_passes_tele_and_untagged():
    seen, cb = _recording_callback()
    wrapped = filtered_listener(cb, COMMAND_SOURCE_TYPES)
    wrapped({"source_type": SOURCE_TYPE_TELE, "joint_1": 0.1})
    wrapped({"joint_1": 0.2})  # untagged → relaxed accept
    assert len(seen) == 2


def test_filtered_listener_preserves_async_callback_result():
    async def acb(envelope):
        return "ok"

    import asyncio

    wrapped = filtered_listener(acb, COMMAND_SOURCE_TYPES)
    coro = wrapped({"source_type": SOURCE_TYPE_TELE})
    assert asyncio.run(coro) == "ok"
    # Dropped messages return None (not a coroutine) so the dispatcher no-ops cleanly.
    assert wrapped({"source_type": SOURCE_TYPE_EDGE}) is None

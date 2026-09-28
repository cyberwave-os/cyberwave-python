"""Tests for the driver source-type policy (publish edge/sim, listen tele; relaxed)."""

import pytest

from cyberwave.constants import (
    SOURCE_TYPE_EDGE,
    SOURCE_TYPE_EDGE_LEADER,
    SOURCE_TYPE_EDIT,
    SOURCE_TYPE_SIM,
    SOURCE_TYPE_SIM_TELE,
    SOURCE_TYPE_TELE,
)
from cyberwave.driver.interface.source_type_policy import (
    COMMAND_SOURCE_TYPES,
    LIVE_NAVIGATION_COMMAND_SOURCE_TYPES,
    SIMULATION_NAVIGATION_SOURCE_TYPES,
    _is_log_milestone,
    accepts_navigation_command,
    accepts_inbound,
    filtered_listener,
    provenance_drop_counts,
    reset_provenance_drop_counts,
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


def test_navigation_command_is_owned_by_exact_runtime():
    assert accepts_navigation_command(
        {"command": "path", "source_type": SOURCE_TYPE_EDGE}, "live"
    )
    assert accepts_navigation_command(
        {"command": "path", "source_type": SOURCE_TYPE_TELE}, "live"
    )
    assert not accepts_navigation_command(
        {"command": "path", "source_type": SOURCE_TYPE_SIM}, "live"
    )
    assert not accepts_navigation_command({"command": "path"}, "live")

    assert accepts_navigation_command(
        {"command": "path", "source_type": SOURCE_TYPE_SIM}, "simulation"
    )
    assert accepts_navigation_command(
        {"command": "path", "source_type": SOURCE_TYPE_SIM_TELE}, "simulation"
    )
    assert not accepts_navigation_command(
        {"command": "path", "source_type": SOURCE_TYPE_TELE}, "simulation"
    )


def test_navigation_stop_is_cross_runtime_fail_safe():
    assert accepts_navigation_command(
        {"command": "stop", "source_type": SOURCE_TYPE_SIM}, "live"
    )
    assert accepts_navigation_command({"command": "stop"}, "live")


@pytest.mark.parametrize("runtime_mode", [None, "", "sim", "typo"])
def test_navigation_command_rejects_unknown_runtime_mode(
    runtime_mode: str | None,
) -> None:
    assert not accepts_navigation_command(
        {"command": "path", "source_type": SOURCE_TYPE_EDGE}, runtime_mode
    )
    assert not accepts_navigation_command(
        {"command": "path", "source_type": SOURCE_TYPE_SIM}, runtime_mode
    )


# ── the two navigation sets ask different questions ──────────────────────────


def test_live_navigation_set_is_hardware_role_absent():
    """``substrate == hardware && role == absent`` — and ``edge`` is in it.

    ``edge`` is a state value, so this is not a command allow-list: navigation is
    a server-owned channel. That exception is why the set has its own name.
    """
    assert LIVE_NAVIGATION_COMMAND_SOURCE_TYPES == frozenset(
        {SOURCE_TYPE_EDGE, SOURCE_TYPE_TELE}
    )
    assert SOURCE_TYPE_EDGE_LEADER not in LIVE_NAVIGATION_COMMAND_SOURCE_TYPES
    assert SIMULATION_NAVIGATION_SOURCE_TYPES == frozenset(
        {SOURCE_TYPE_SIM, SOURCE_TYPE_SIM_TELE}
    )


# ── rejection logging + counters ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_drop_counters():
    reset_provenance_drop_counts()
    yield
    reset_provenance_drop_counts()


def test_self_echo_drop_is_counted_with_its_predicate():
    assert accepts_inbound(COMMAND_SOURCE_TYPES, SOURCE_TYPE_EDGE) is False
    counts = provenance_drop_counts()
    assert len(counts) == 1
    (guard, predicate, source_type, _topic), count = next(iter(counts.items()))
    assert guard == "accepts_inbound"
    assert "self-echo" in predicate
    assert source_type == SOURCE_TYPE_EDGE
    assert count == 1


def test_repeated_drops_accumulate_on_one_key():
    for _ in range(4):
        accepts_inbound(COMMAND_SOURCE_TYPES, SOURCE_TYPE_EDGE_LEADER, topic="t/1")
    assert list(provenance_drop_counts().values()) == [4]


def test_allowed_and_absent_source_types_are_not_counted():
    assert accepts_inbound(COMMAND_SOURCE_TYPES, SOURCE_TYPE_TELE) is True
    assert accepts_inbound(COMMAND_SOURCE_TYPES, None) is True
    assert provenance_drop_counts() == {}


def test_drop_log_names_topic_and_publisher(caplog):
    with caplog.at_level("WARNING"):
        accepts_inbound(
            COMMAND_SOURCE_TYPES,
            SOURCE_TYPE_SIM,
            topic="cyberwave/twin/abc/joint/update",
            envelope={"source_type": SOURCE_TYPE_SIM, "source_subtype": "mock-vla"},
        )
    assert "cyberwave/twin/abc/joint/update" in caplog.text
    assert "mock-vla" in caplog.text


def test_navigation_drop_is_counted():
    assert not accepts_navigation_command(
        {"command": "path", "source_type": SOURCE_TYPE_SIM}, "live", topic="nav/1"
    )
    counts = provenance_drop_counts()
    assert len(counts) == 1
    (guard, _predicate, _source, topic), _count = next(iter(counts.items()))
    assert guard == "accepts_navigation_command"
    assert topic == "nav/1"


def test_listener_without_declared_sources_is_reported_as_unfiltered(caplog):
    """The default is no filter at all, and that is now visible.

    Behavior is unchanged: still returned unwrapped, still receiving everything
    including self-echo. Only the log is new.
    """
    seen: list[dict] = []

    def listener(envelope: dict) -> None:
        seen.append(envelope)

    with caplog.at_level("WARNING"):
        wrapped = filtered_listener(listener, None, topic="cyberwave/twin/abc/pose")
    assert wrapped is listener
    assert "no provenance filter" in caplog.text
    assert "cyberwave/twin/abc/pose" in caplog.text

    # Unwrapped means the edge* self-echo guard never runs for this listener.
    wrapped({"source_type": SOURCE_TYPE_EDGE})
    assert seen == [{"source_type": SOURCE_TYPE_EDGE}]
    assert provenance_drop_counts() == {}


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, True),
        (2, False),
        (9, False),
        (10, True),
        (11, False),
        (100, True),
        (200, False),
        (1000, True),
        (0, False),
    ],
)
def test_log_milestones_are_the_decades(count, expected):
    """Kept identical to the backend's ``src/lib/provenance_drops.py``."""
    assert _is_log_milestone(count) is expected


def test_persistent_drop_keeps_reporting_its_running_total(caplog):
    """A once-only WARNING caps flood but hides scale, and DEBUG is usually off.

    A driver whose plant floods a command topic should be able to tell "one stray
    message" from "everything is being rejected" without turning on DEBUG.
    """
    with caplog.at_level("WARNING"):
        for _ in range(100):
            accepts_inbound(COMMAND_SOURCE_TYPES, SOURCE_TYPE_EDGE, topic="t/1")
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert [r.args[-1] for r in warnings] == [1, 10, 100]
    # The counter still holds the true total, milestones or not.
    assert list(provenance_drop_counts().values()) == [100]


# --- key cap -----------------------------------------------------------------


def test_unseen_source_types_fold_into_one_bucket_at_the_cap():
    """`source_type` is read straight off the wire, so the map needs a ceiling."""
    from cyberwave.driver.interface.source_type_policy import (
        _MAX_DROP_KEYS,
        _OVERFLOW,
    )

    allowed = frozenset({"tele"})
    for index in range(_MAX_DROP_KEYS + 200):
        accepts_inbound(allowed, f"junk-{index}", topic="t")

    counts = provenance_drop_counts()
    assert len(counts) == _MAX_DROP_KEYS + 1
    folded = [value for key, value in counts.items() if key[2] == _OVERFLOW]
    assert folded == [200]


def test_a_recurring_drop_keeps_counting_past_the_cap():
    from cyberwave.driver.interface.source_type_policy import _MAX_DROP_KEYS

    allowed = frozenset({"tele"})
    accepts_inbound(allowed, "edge_follower", topic="t")
    for index in range(_MAX_DROP_KEYS + 50):
        accepts_inbound(allowed, f"junk-{index}", topic="t")
    for _ in range(9):
        accepts_inbound(allowed, "edge_follower", topic="t")

    self_echo = [
        value
        for key, value in provenance_drop_counts().items()
        if key[2] == "edge_follower"
    ]
    assert self_echo == [10]

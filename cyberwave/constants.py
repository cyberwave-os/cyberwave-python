"""
Constants used across the Cyberwave platform.

This module defines local constants so the SDK is self-contained.
"""
SOURCE_TYPE_EDGE_LEADER = "edge_leader"
SOURCE_TYPE_EDGE_FOLLOWER = "edge_follower"
SOURCE_TYPE_EDGE = "edge"
SOURCE_TYPE_TELE = "tele"
SOURCE_TYPE_EDIT = "edit"
SOURCE_TYPE_SIM = "sim"
SOURCE_TYPE_SIM_TELE = "sim_tele"
SOURCE_TYPES = (
    SOURCE_TYPE_EDGE_LEADER,
    SOURCE_TYPE_EDGE_FOLLOWER,
    SOURCE_TYPE_EDGE,
    SOURCE_TYPE_TELE,
    SOURCE_TYPE_EDIT,
    SOURCE_TYPE_SIM,
    SOURCE_TYPE_SIM_TELE,
)

# Physical edge feedback — not teleop commands (drivers publish, UI/teleop consumes).
# The self-echo guard: "is this my own feedback echoing back?" ``edge_leader`` counts,
# because a leader arm's stream is still state. The controller's
# ``MEASURED_PLANT_SOURCES`` asks "is this the plant I observe?" and drops it — a
# leader is an input device. Both are right; don't normalise them together.
EDGE_STATE_SOURCE_TYPES = frozenset(
    {
        SOURCE_TYPE_EDGE,
        SOURCE_TYPE_EDGE_LEADER,
        SOURCE_TYPE_EDGE_FOLLOWER,
    }
)

__all__ = [
    "SOURCE_TYPE_EDGE_LEADER",
    "SOURCE_TYPE_EDGE_FOLLOWER",
    "SOURCE_TYPE_EDGE",
    "SOURCE_TYPE_TELE",
    "SOURCE_TYPE_EDIT",
    "SOURCE_TYPE_SIM",
    "SOURCE_TYPE_SIM_TELE",
    "SOURCE_TYPES",
    "EDGE_STATE_SOURCE_TYPES",
]

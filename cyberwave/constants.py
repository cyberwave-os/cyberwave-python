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

__all__ = [
    "SOURCE_TYPE_EDGE_LEADER",
    "SOURCE_TYPE_EDGE_FOLLOWER",
    "SOURCE_TYPE_EDGE",
    "SOURCE_TYPE_TELE",
    "SOURCE_TYPE_EDIT",
    "SOURCE_TYPE_SIM",
    "SOURCE_TYPE_SIM_TELE",
    "SOURCE_TYPES",
]

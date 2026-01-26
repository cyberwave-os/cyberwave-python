"""
Constants used across the Cyberwave platform.

This module defines local constants so the SDK is self-contained.
"""

SOURCE_TYPE_EDGE = "edge"
SOURCE_TYPE_TELE = "tele"
SOURCE_TYPE_EDIT = "edit"
SOURCE_TYPE_SIM = "sim"
SOURCE_TYPES = [
    SOURCE_TYPE_EDGE,
    SOURCE_TYPE_TELE,
    SOURCE_TYPE_EDIT,
    SOURCE_TYPE_SIM,
]

__all__ = [
    "SOURCE_TYPE_EDGE",
    "SOURCE_TYPE_TELE",
    "SOURCE_TYPE_EDIT",
    "SOURCE_TYPE_SIM",
    "SOURCE_TYPES",
]

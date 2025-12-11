"""
Constants used across the Cyberwave platform.

This module re-exports constants from the shared_constants module at the monorepo root.
This ensures the SDK uses the same constants as all other projects.

For projects that don't use the SDK, import directly from shared_constants.
"""

import sys
from pathlib import Path

# Import from shared constants at monorepo root
# Walk up directory tree to find shared_constants.py
_root_path = Path(__file__).parent
_shared_constants_loaded = False
while _root_path != _root_path.parent:
    _shared_constants_path = _root_path / "shared_constants.py"
    if _shared_constants_path.exists():
        sys.path.insert(0, str(_root_path))
        try:
            from shared_constants import (
                SOURCE_TYPE_EDGE,
                SOURCE_TYPE_TELE,
                SOURCE_TYPE_EDIT,
                SOURCE_TYPE_SIM,
                SOURCE_TYPES,
            )
            _shared_constants_loaded = True
            break
        except ImportError:
            pass
    _root_path = _root_path.parent

if not _shared_constants_loaded:
    # Fallback if shared constants not available
    SOURCE_TYPE_EDGE = "edge"
    SOURCE_TYPE_TELE = "tele"
    SOURCE_TYPE_EDIT = "edit"
    SOURCE_TYPE_SIM = "sim"
    SOURCE_TYPES = ["edit", "edge", "tele", "sim"]

__all__ = [
    "SOURCE_TYPE_EDGE",
    "SOURCE_TYPE_TELE",
    "SOURCE_TYPE_EDIT",
    "SOURCE_TYPE_SIM",
    "SOURCE_TYPES",
]

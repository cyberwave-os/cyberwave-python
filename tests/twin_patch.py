"""Helpers for patching ``cyberwave.twin.*`` submodules in unit tests.

``unittest.mock.patch`` string targets like ``cyberwave.twin.transport.time.sleep``
fail when ``cyberwave.twin`` on the parent package is the compact-API function
(``from cyberwave import twin``), not the twin subpackage.  Resolve the real
submodule via ``importlib`` instead.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import patch


def patch_twin(target: str, *args: Any, **kwargs: Any):
    """Return a ``patch.object`` context manager for a ``cyberwave.twin.*`` target.

    *target* is relative to the twin package, e.g. ``"transport.time.sleep"``
    or ``"capabilities.joints.controllable_joint_names"``.
    """
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        mod_path = "cyberwave.twin." + ".".join(parts[:i])
        try:
            mod = importlib.import_module(mod_path)
            tail = parts[i:]
            break
        except ModuleNotFoundError:
            continue
    else:
        raise ModuleNotFoundError(f"No module for twin patch target {target!r}")

    obj = mod
    for part in tail[:-1]:
        obj = getattr(obj, part)
    return patch.object(obj, tail[-1], *args, **kwargs)

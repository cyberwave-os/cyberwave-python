"""Worker module loader — imports .py files and collects registered hooks."""

from __future__ import annotations

import builtins
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_workers(
    workers_dir: str | Path,
    *,
    cw_instance: object,
) -> int:
    """Import all ``.py`` files from *workers_dir*.

    Worker modules use the bare ``cw`` variable (the :class:`Cyberwave`
    client injected into ``builtins``).  This function:

    1. Injects *cw_instance* as ``builtins.cw`` so worker code can use
       ``cw.on_frame(...)`` without an import.
    2. Imports each ``.py`` file as a standalone module (sorted
       alphabetically for deterministic load order).
    3. Skips files whose name starts with ``_``.
    4. Logs and continues on import failures so one broken worker does
       not crash the runtime.

    Returns:
        Number of worker modules successfully loaded.
    """
    workers_dir = Path(workers_dir)

    if not workers_dir.is_dir():
        logger.warning("Workers directory does not exist: %s", workers_dir)
        return 0

    builtins.cw = cw_instance  # type: ignore[attr-defined]

    loaded = 0
    for py_file in sorted(workers_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"cyberwave_worker_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning("Cannot create module spec for: %s", py_file)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            loaded += 1
            logger.info("Loaded worker: %s", py_file.name)
        except Exception:
            logger.exception("Failed to load worker: %s", py_file.name)

    return loaded

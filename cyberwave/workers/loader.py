"""Worker module loader — imports .py files and collects registered hooks."""

from __future__ import annotations

import builtins
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_loadable(path: Path) -> bool:
    """Return True if *path* is a ``.py`` file that should be auto-loaded.

    Files whose names start with ``_`` (e.g. ``__init__.py``, ``_helpers.py``)
    are considered private and are skipped.
    """
    return not path.name.startswith("_")


def load_workers(
    workers_path: str | Path,
    *,
    cw_instance: object,
    loaded_modules: list[object] | None = None,
) -> int:
    """Import ``.py`` worker modules from *workers_path*.

    *workers_path* may be either a **directory** (all ``.py`` files in it are
    loaded) or a **single ``.py`` file** (only that file is loaded).

    Worker modules use the bare ``cw`` variable (the :class:`Cyberwave`
    client injected into ``builtins``).  This function:

    1. Injects *cw_instance* as ``builtins.cw`` so worker code can use
       ``cw.on_frame(...)`` without an import.
    2. Imports each ``.py`` file as a standalone module (sorted
       alphabetically when loading a directory, for deterministic order).
    3. Skips files whose name starts with ``_``.
    4. Logs and continues on import failures so one broken worker does
       not crash the runtime.

    .. note::

        ``builtins.cw`` is process-global.  Only one Cyberwave client may
        be active per process; subsequent calls overwrite the previous
        binding.  Test suites should clean up via ``del builtins.cw`` or
        use :class:`HookContext` to isolate state.

    Returns:
        Number of worker modules successfully loaded.
    """
    workers_path = Path(workers_path)

    if workers_path.is_file():
        py_files = [workers_path] if _is_loadable(workers_path) else []
    elif workers_path.is_dir():
        py_files = [f for f in sorted(workers_path.glob("*.py")) if _is_loadable(f)]
    else:
        logger.warning("Workers path does not exist: %s", workers_path)
        return 0

    builtins.cw = cw_instance  # type: ignore[attr-defined]

    loaded = 0
    for py_file in py_files:
        module_name = f"cyberwave_worker_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning("Cannot create module spec for: %s", py_file)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            if loaded_modules is not None:
                loaded_modules.append(module)
            loaded += 1
            logger.info("Loaded worker: %s", py_file.name)
        except Exception:
            logger.exception("Failed to load worker: %s", py_file.name)

    return loaded

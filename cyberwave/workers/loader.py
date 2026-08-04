"""Worker module loader — imports .py files and collects registered hooks."""

from __future__ import annotations

import builtins
import importlib.util
import logging
import re
import sys
from pathlib import Path

from cyberwave._error_metadata import format_error_metadata

logger = logging.getLogger(__name__)

# Generated ``wf_*.py`` workers declare these top-level constants; scanned
# from source on a load failure so the alert can be attributed even when the
# module never finished importing (e.g. a SyntaxError).
_TWIN_UUID_RE = re.compile(r'^TWIN_UUID\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
_WORKFLOW_UUID_RE = re.compile(
    r'^WORKFLOW_UUID\s*=\s*["\']([^"\']+)["\']', re.MULTILINE
)


def _extract_worker_identity(
    py_file: Path, module: object | None
) -> tuple[str | None, str | None]:
    """Best-effort ``(twin_uuid, workflow_uuid)`` for a failed worker load.

    Prefers the (possibly partially) executed module namespace, then falls
    back to scanning the source text so a worker that fails to even parse
    still attributes its alert to a twin.
    """
    twin_uuid = getattr(module, "TWIN_UUID", None) if module is not None else None
    workflow_uuid = (
        getattr(module, "WORKFLOW_UUID", None) if module is not None else None
    )
    if twin_uuid and workflow_uuid:
        return twin_uuid, workflow_uuid
    try:
        text = py_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # An invalid byte sequence must not abort load_workers() for every other file.
        return twin_uuid, workflow_uuid
    if not twin_uuid:
        match = _TWIN_UUID_RE.search(text)
        twin_uuid = match.group(1) if match else None
    if not workflow_uuid:
        match = _WORKFLOW_UUID_RE.search(text)
        workflow_uuid = match.group(1) if match else None
    return twin_uuid, workflow_uuid


def _report_worker_load_failure(
    py_file: Path, module: object | None, cw_instance: object, error: BaseException
) -> None:
    """Publish a ``worker_load_error`` alert when a worker fails to import
    (it registers no hooks and runs nothing — otherwise a silent failure).
    Best-effort: never raise from here.
    """
    try:
        publish_alert = getattr(cw_instance, "publish_alert", None)
        if not callable(publish_alert):
            return
        twin_uuid, workflow_uuid = _extract_worker_identity(py_file, module)
        if not twin_uuid:
            return  # nothing to attach to; the log line already captured it
        error_code, technical_detail = format_error_metadata(error, limit=400)
        publish_alert(
            twin_uuid,
            "Workflow worker failed to load",
            description=(
                f"The workflow worker '{py_file.name}' could not be loaded "
                "and will not run."
            ),
            alert_type="worker_load_error",
            severity="error",
            category="technical",
            source_type="edge",
            workflow_uuid=workflow_uuid,
            metadata={
                "error_code": error_code,
                "technical_detail": technical_detail,
            },
        )
    except Exception:
        logger.warning(
            "Could not publish worker_load_error alert for %s",
            py_file.name,
            exc_info=True,
        )


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
        module: object | None = None
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
        except Exception as exc:
            logger.exception("Failed to load worker: %s", py_file.name)
            _report_worker_load_failure(py_file, module, cw_instance, exc)

    return loaded

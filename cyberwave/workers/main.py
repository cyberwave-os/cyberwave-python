"""Edge worker main -- run as: python -m cyberwave.workers.main

Intended to be the CMD of the ``cyberwaveos/edge-ml-worker`` Docker image.
Environment variables consumed:

  CYBERWAVE_API_KEY           (required) API token for SDK authentication.
  CYBERWAVE_WORKERS_DIR       Path to worker scripts (default: /app/workers).
  CYBERWAVE_WORKER_LOG_LEVEL  Worker logging verbosity (default: INFO).
                              Takes precedence over ``CYBERWAVE_EDGE_LOG_LEVEL``.
  CYBERWAVE_EDGE_LOG_LEVEL    Fallback log level (kept for backwards compat).

Workers and edge drivers live in different containers; tuning them
independently is the common case (e.g. raising the driver to DEBUG to
inspect a hardware issue while leaving workers at INFO so the per-frame
trace isn't a wall of noise). Hence the dedicated ``CYBERWAVE_WORKER_LOG_LEVEL``
on top of the legacy ``CYBERWAVE_EDGE_LOG_LEVEL``.
"""

from __future__ import annotations

import logging
import os
import sys

_VALID_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def _resolve_log_level() -> tuple[int, str, str]:
    """Resolve the worker log level and the env var that supplied it.

    Returns ``(level_int, level_name, source_var)``. Falls back to ``INFO``
    when neither variable is set or the configured value is unparseable.
    The source-var string is logged on startup so operators can tell at a
    glance which knob is in effect.
    """
    for var in ("CYBERWAVE_WORKER_LOG_LEVEL", "CYBERWAVE_EDGE_LOG_LEVEL"):
        raw = os.environ.get(var)
        if not raw:
            continue
        candidate = raw.strip().upper()
        if candidate in _VALID_LEVELS:
            return logging.getLevelName(candidate), candidate, var
        # Unrecognised value: warn but don't crash — we log this
        # explicitly after basicConfig so the message actually reaches
        # the configured handler.
        return logging.INFO, "INFO", f"{var}=invalid({raw!r}); defaulted"
    return logging.INFO, "INFO", "default"


def main() -> None:
    level_int, level_name, level_source = _resolve_log_level()
    logging.basicConfig(
        level=level_int,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Emit the startup banner at ``max(level_int, INFO)`` so it's *always*
    # visible: at INFO/DEBUG the banner stays at INFO; at WARNING/ERROR it
    # rides up to the configured level so the operator can still see which
    # knob is in effect when they've intentionally muted INFO. Without this,
    # the very setting most likely to need verification (high-suppression
    # configs) is also the one that hides the verification line.
    logging.getLogger("cyberwave.workers").log(
        max(level_int, logging.INFO),
        "Worker runtime starting with log level=%s (source=%s)",
        level_name,
        level_source,
    )

    api_key = os.environ.get("CYBERWAVE_API_KEY")
    if not api_key:
        logging.error("CYBERWAVE_API_KEY environment variable is required")
        sys.exit(1)

    from cyberwave import Cyberwave

    cw = Cyberwave(api_key=api_key)
    workers_dir = os.environ.get("CYBERWAVE_WORKERS_DIR", "/app/workers")
    cw.run_edge_workers(workers_dir=workers_dir)


if __name__ == "__main__":
    main()

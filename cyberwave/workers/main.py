"""Edge worker main -- run as: python -m cyberwave.workers.main

Intended to be the CMD of the ``cyberwaveos/edge-ml-worker`` Docker image.
Environment variables consumed:

  CYBERWAVE_API_KEY         (required) API token for SDK authentication.
  CYBERWAVE_WORKERS_DIR     Path to worker scripts (default: /app/workers).
  CYBERWAVE_EDGE_LOG_LEVEL  Logging verbosity (default: INFO).
"""

from __future__ import annotations

import logging
import os
import sys


def main() -> None:
    log_level = os.environ.get("CYBERWAVE_EDGE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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

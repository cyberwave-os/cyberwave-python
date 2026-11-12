"""``python -m cyberwave.workers.cloud_runner``: run one cloud workflow execution.

Configuration is the environment, like the edge worker's:

  CYBERWAVE_API_KEY         (required) the run token, or an API key when testing.
  CYBERWAVE_BASE_URL        Backend URL (the SDK's default when unset).
  CYBERWAVE_WORKFLOW_UUID   (required) the workflow.
  CYBERWAVE_EXECUTION_UUID  (required unless --new-execution) the execution; with
                            --new-execution it names the new one (else a random uuid).
  CYBERWAVE_RUN_PAYLOAD     Optional JSON run envelope, e.g. ``{"params": {}}``; its
                            keys override what the execution's trigger data gives.

``--source-file PATH`` runs that source instead of fetching the workflow's.
``--new-execution`` sends the ``started`` event first, for a run nothing else
created. The exit code is 0 for a run that completed, 1 for one that failed or
was canceled, 2 for a runner that could not start.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import uuid
from pathlib import Path

from .segment import COMPLETED

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_RUN_ENDED, EXIT_CANNOT_START = 0, 1, 2


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required")
    return value


class JobTimeLimit(Exception):
    """The job was told to stop (SIGTERM) because it reached its time limit."""


def _at_time_limit(signum: int, frame: object) -> None:
    # The launcher ends a job at its limit with SIGTERM (`timeout`), and Python answers
    # SIGTERM by dying at once, reporting nothing and leaving the run `RUNNING`. Raise
    # instead, so the run is reported failed and the connection closed inside the
    # launcher's kill-after grace.
    raise JobTimeLimit("the job reached its time limit")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cyberwave.workers.cloud_runner")
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--new-execution", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )
    try:
        api_key = _required("CYBERWAVE_API_KEY")
        workflow_uuid = _required("CYBERWAVE_WORKFLOW_UUID")
        execution_uuid = (
            os.environ.get("CYBERWAVE_EXECUTION_UUID", "").strip() or str(uuid.uuid4())
            if args.new_execution
            else _required("CYBERWAVE_EXECUTION_UUID")
        )
        payload = json.loads(os.environ.get("CYBERWAVE_RUN_PAYLOAD") or "{}")
        source = args.source_file.read_text() if args.source_file else None
    except (SystemExit, ValueError, OSError) as exc:
        logger.error("cannot start: %s", exc)
        return EXIT_CANNOT_START

    from cyberwave import Cyberwave

    from .runner import run_execution

    client = Cyberwave(api_key=api_key)
    try:
        client.mqtt.connect()  # the reporter publishes node events over MQTT
    except Exception as exc:  # noqa: BLE001 - nothing has run yet
        logger.error("cannot start: could not connect to MQTT: %s", exc)
        return EXIT_CANNOT_START
    previous_handler = signal.signal(signal.SIGTERM, _at_time_limit)
    try:
        # `started` is an MQTT event, so the row may not exist yet when the run
        # would read it: describe the run (a REQUESTED one, which the runner starts)
        # instead of reading it.
        outcome = run_execution(
            client,
            workflow_uuid=workflow_uuid,
            execution_uuid=execution_uuid,
            payload=payload,
            source=source,
            execution={"status": "requested"} if args.new_execution else None,
            wait_until_recorded=args.new_execution,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        client.mqtt.disconnect()
    logger.info(
        "execution %s ended: %s%s",
        execution_uuid,
        outcome.status,
        f" ({outcome.error})" if outcome.error else "",
    )
    return EXIT_OK if outcome.status == COMPLETED else EXIT_RUN_ENDED


if __name__ == "__main__":
    sys.exit(main())

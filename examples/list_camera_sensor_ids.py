#!/usr/bin/env python3
"""
Example: list camera sensor ids from a twin's live universal schema (Twin.list_camera_sensor_ids).

Configuration:
    Required: CYBERWAVE_API_KEY
    Required (choose one):
        CYBERWAVE_TWIN_REGISTRY   catalog path, e.g. cyberwave/generic-camera
        CYBERWAVE_TWIN_UUID       existing twin UUID (``cw.twin(twin_id=...)``)
    Optional: CYBERWAVE_CAPTURE_FRAME=1   fetch one latest-frame JPEG after listing

Set via ``.env`` file in the SDK repo root or export as environment variables
(see ``examples/ur7-santas-little-helper.py`` for the same ``.env`` pattern).

Run from repo root:
  poetry run python examples/list_camera_sensor_ids.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parent.parent
    _env_file = _repo_root / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

from cyberwave import Cyberwave


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    api_key = os.getenv("CYBERWAVE_API_KEY")
    if not (api_key or "").strip():
        print(
            "ERROR: CYBERWAVE_API_KEY is required",
            file=sys.stderr,
        )
        return 1

    registry = (os.getenv("CYBERWAVE_TWIN_REGISTRY") or "").strip()
    twin_uuid = (os.getenv("CYBERWAVE_TWIN_UUID") or "").strip()
    if not registry and not twin_uuid:
        print(
            "ERROR: set CYBERWAVE_TWIN_REGISTRY or CYBERWAVE_TWIN_UUID",
            file=sys.stderr,
        )
        return 1
    if registry and twin_uuid:
        print(
            "ERROR: set only one of CYBERWAVE_TWIN_REGISTRY or CYBERWAVE_TWIN_UUID",
            file=sys.stderr,
        )
        return 1

    cw = Cyberwave()
    try:
        if twin_uuid:
            twin = cw.twin(twin_id=twin_uuid)
        else:
            twin = cw.twin(registry)

        print("twin.uuid:", twin.uuid)
        print("twin.name:", getattr(twin, "name", ""))

        ids = twin.list_camera_sensor_ids()
        print("list_camera_sensor_ids():", ids)

        if _env_truthy("CYBERWAVE_CAPTURE_FRAME"):
            if ids:
                sid = ids[0]
                print("capture_frame(format='bytes', sensor_id=%r) ..." % (sid,))
                blob = twin.capture_frame("bytes", sensor_id=sid)
            else:
                print("capture_frame(format='bytes') (no sensor ids in schema) ...")
                blob = twin.capture_frame("bytes")
            n = len(blob) if blob is not None else 0
            head = blob[:4] if isinstance(blob, (bytes, bytearray)) and n >= 2 else b""
            jpeg = head[:2] == b"\xff\xd8"
            print("latest-frame bytes: len=%d jpeg_magic=%s" % (n, jpeg))
    finally:
        cw.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

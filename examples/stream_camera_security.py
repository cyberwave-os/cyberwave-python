from __future__ import annotations

"""
Example: Register a camera sensor and stream frames with a simple analyzer in backend.

Requirements:
- CYBERWAVE_BASE_URL, CYBERWAVE_TOKEN, CYBERWAVE_ENV_UUID
- A JPEG image file to send as a test frame

Notes:
- Backend has /api/v1/sensors and /api/v1/sensors/{uuid}/video endpoints.
- Analyzer hooks in backend (ai_tasks.py) can process frames and emit events.
"""

import os
from pathlib import Path
from cyberwave import Cyberwave
import time


def main():
    base = os.getenv("CYBERWAVE_BASE_URL", "http://localhost:8000")
    token = os.getenv("CYBERWAVE_TOKEN", "")
    env_uuid = os.getenv("CYBERWAVE_ENV_UUID", "")
    if not (token and env_uuid):
        raise SystemExit("Set CYBERWAVE_TOKEN and CYBERWAVE_ENV_UUID")

    # Test image path (simulate a camera frame)
    img_path = os.getenv("CYBERWAVE_TEST_IMAGE", "")
    if not img_path or not Path(img_path).exists():
        raise SystemExit("Set CYBERWAVE_TEST_IMAGE to a JPG/PNG file path")

    cw = Cyberwave(base, token)

    # Create a camera sensor (standalone or attach to a twin by twin_uuid)
    sensor = cw.sensors.create(environment_uuid=env_uuid, name="security_cam_1", sensor_type="camera", description="Security cam")
    sensor_uuid = sensor["uuid"]
    print("Sensor created:", sensor_uuid)

    # Send a single frame (backend analyzer can process and append events)
    data = Path(img_path).read_bytes()
    resp = cw.sensors.send_frame(sensor_uuid, data, content_type="image/jpeg")
    print("Frame sent. Backend response:", resp)

    # Poll recent events to show detection
    # Note: this example uses the sensor events endpoint added for quick polling
    time.sleep(1)
    events = cw._http.get(f"/api/v1/sensors/{sensor_uuid}/events")  # type: ignore[attr-defined]
    print("Recent events:", events)


if __name__ == "__main__":
    main()



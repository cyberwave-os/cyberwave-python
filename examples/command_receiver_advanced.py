"""
Command Receiver (Advanced) — handle commands with system metrics.

Responds to "take_snapshot" commands with CPU and memory usage.

Env vars:
    CYBERWAVE_API_KEY    API key
    TWIN_UUID            Twin UUID

Requirements:
    pip install cyberwave psutil
"""

import asyncio
import os
from datetime import datetime

import psutil

from cyberwave import Cyberwave

cw = Cyberwave()
twin_uuid = os.environ["TWIN_UUID"]

cw.mqtt.connect()


def command_handler(data):
    if "status" in data:
        return

    if data.get("command") == "take_snapshot":
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
        }
        print(f"Snapshot: {snapshot}")
        cw.mqtt.publish_command_message(twin_uuid, "ok")
    else:
        cw.mqtt.publish_command_message(twin_uuid, "error")


cw.mqtt.subscribe_command_message(twin_uuid, command_handler)
print("Listening for commands… Press Ctrl+C to stop.")

try:
    asyncio.run(asyncio.sleep(float("inf")))
except KeyboardInterrupt:
    cw.disconnect()

"""
Command Receiver — subscribe to and respond to commands for a twin.

Env vars:
    CYBERWAVE_API_KEY    API key
    TWIN_UUID            Twin UUID

Requirements:
    pip install cyberwave
"""

import asyncio
import os

from cyberwave import Cyberwave

cw = Cyberwave()
twin_uuid = os.environ["TWIN_UUID"]

cw.mqtt.connect()


def command_handler(data):
    if "status" in data:
        return

    command = data.get("command")
    if command == "greetings":
        print("Hello World!")
        cw.mqtt.publish_command_message(twin_uuid, "ok")
    else:
        cw.mqtt.publish_command_message(twin_uuid, "error")


cw.mqtt.subscribe_command_message(twin_uuid, command_handler)
print("Listening for commands… Press Ctrl+C to stop.")

try:
    asyncio.run(asyncio.sleep(float("inf")))
except KeyboardInterrupt:
    cw.disconnect()

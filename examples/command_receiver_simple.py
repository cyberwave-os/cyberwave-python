"""
Simple Command Receiver Example

This is the simplest example demonstrating how to receive and respond to commands
for a digital twin using the subscribe_command_message and publish_command_message APIs.

Quick Start:
    1. Set environment variables:
       export CYBERWAVE_TOKEN="your-token"
       export TWIN_UUID="your-twin-uuid"
    
    2. Run the receiver:
       python3 examples/command_receiver_simple.py
    
    3. Send command to topic:
       {prefix}cyberwave/twin/{twin_uuid}/command
       
       Example command:
       {"command": "greetings"}

Requirements:
    pip install cyberwave
"""

import asyncio
import os
import sys
from cyberwave import Cyberwave


async def main():
    token = os.getenv("CYBERWAVE_TOKEN")
    if not token:
        print("Please set CYBERWAVE_TOKEN environment variable")
        return

    host = os.getenv("CYBERWAVE_MQTT_HOST")
    port_str = os.getenv("CYBERWAVE_MQTT_PORT")
    mqtt_username = os.getenv("CYBERWAVE_MQTT_USERNAME")
    mqtt_password = os.getenv("CYBERWAVE_MQTT_PASSWORD")
    
    port = int(port_str) if port_str else None
    
    client = Cyberwave(
        token=token,
        mqtt_host=host,
        mqtt_port=port,
        mqtt_username=mqtt_username,
        mqtt_password=mqtt_password,
    )

    twin_uuid = os.getenv("TWIN_UUID")
    if not twin_uuid:
        print("Please set TWIN_UUID environment variable")
        return
    
    if not client.mqtt.connected:
        client.mqtt.connect()
    
    # All the command receiver logic and API usage is here
    # Just handle commands and respond with client.mqtt.publish_command_message()
    def command_handler(data):
        if "status" in data:
            return
        
        command_type = data.get("command")
        if not command_type:
            client.mqtt.publish_command_message(twin_uuid, "error")
            return
        
        try:
            if command_type == "greetings":
                print("Hello World!")
                client.mqtt.publish_command_message(twin_uuid, "ok")
            else:
                client.mqtt.publish_command_message(twin_uuid, "error")
        except Exception:
            client.mqtt.publish_command_message(twin_uuid, "error")
    
    client.mqtt.subscribe_command_message(twin_uuid, command_handler)
    print("✅ Subscribed to command messages")
    print("Send command: {\"command\": \"greetings\"}")
    print("Press Ctrl+C to stop...\n")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        print("\nCommand receiver stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)


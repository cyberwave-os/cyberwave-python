"""
Command Receiver Example

This example demonstrates how to receive and respond to commands for a digital twin
using the subscribe_command_message and publish_command_message APIs.

The example subscribes to commands for a specific twin, processes them, and sends
back a response with either "ok" or "error" status.

Designed for Linux/Unix devices with enhanced system metrics.

Quick Start:
    1. Set environment variables:
       export CYBERWAVE_TOKEN="your-token"
       export TWIN_UUID="your-twin-uuid"
    
    2. Run the receiver:
       python3 examples/command_receiver.py
    
    3. Send commands to topic:
       {prefix}cyberwave/twin/{twin_uuid}/command
       
       Example command:
       {"command": "take_snapshot", "timestamp": 1704067200.0}

For complete setup instructions, see: COMMAND_RECEIVER_GUIDE.md

Requirements:
    pip install cyberwave
    pip install psutil  # Required for performance metrics on Linux/Unix
"""

import asyncio
import json
import os
import logging
import platform
import sys
import time
from datetime import datetime
from cyberwave import Cyberwave

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import psutil for performance metrics, required on Linux/Unix
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Install with 'pip install psutil' for performance metrics.")
    logger.warning("On Linux/Unix systems, psutil is required for detailed system metrics.")

# Detect platform
IS_UNIX = platform.system() in ('Linux', 'Darwin', 'FreeBSD', 'OpenBSD', 'NetBSD')
IS_LINUX = platform.system() == 'Linux'


async def main():
    token = os.getenv("CYBERWAVE_TOKEN")
    if not token:
        print("Please set CYBERWAVE_TOKEN environment variable")
        return

    host = os.getenv("CYBERWAVE_MQTT_HOST")
    port_str = os.getenv("CYBERWAVE_MQTT_PORT")
    mqtt_username = os.getenv("CYBERWAVE_MQTT_USERNAME")
    mqtt_password = os.getenv("CYBERWAVE_MQTT_PASSWORD")
    
    # Convert port to int if provided
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
            if command_type == "take_snapshot":
                take_performance_snapshot()
                client.mqtt.publish_command_message(twin_uuid, "ok")
            else:
                client.mqtt.publish_command_message(twin_uuid, "error")
        except Exception:
            client.mqtt.publish_command_message(twin_uuid, "error")
    
    
    def take_performance_snapshot():
        if not IS_UNIX:
            logger.warning(f"Running on {platform.system()}. This example is optimized for Linux/Unix systems.")
        
        snapshot = {"timestamp": datetime.now().isoformat()}
        
        try:
            if not PSUTIL_AVAILABLE:
                snapshot["error"] = "psutil not available"
                logger.info(f"Performance snapshot: {json.dumps(snapshot, indent=2)}")
                return snapshot
            
            snapshot["cpu_percent"] = psutil.cpu_percent(interval=1)
            snapshot["memory_percent"] = psutil.virtual_memory().percent
            
            try:
                snapshot["disk_percent"] = psutil.disk_usage('/').percent
            except Exception:
                pass
            
            if IS_UNIX:
                try:
                    boot_time = psutil.boot_time()
                    uptime_days = (time.time() - boot_time) / 86400
                    snapshot["uptime_days"] = round(uptime_days, 2)
                except Exception:
                    pass
                
        except Exception as e:
            snapshot["error"] = str(e)
        
        logger.info(f"Performance snapshot: {json.dumps(snapshot, indent=2)}")
        return snapshot
    
    client.mqtt.subscribe_command_message(twin_uuid, command_handler)
    print("✅ Subscribed to command messages")
    print("Commands will be processed and responses ('ok' or 'error') will be sent automatically.")
    print("Press Ctrl+C to stop...\n")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Error during operation: {e}", exc_info=True)
    finally:
        client.disconnect()
        print("\nCommand receiver stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)

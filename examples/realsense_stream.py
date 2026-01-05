"""
Camera Streaming Example

This example demonstrates how to stream camera feed to a digital twin
using WebRTC with the improved developer experience.

Requirements:
    pip install cyberwave[camera]
"""

import asyncio
import logging
import sys
import os
from cyberwave import Cyberwave
from cyberwave.camera import RealSenseStreamer
from cyberwave.utils import TimeReference

time_reference = TimeReference()

async def main():
    token = os.getenv("CYBERWAVE_TOKEN")

    # Configure logging to see SDK logs
    logging.basicConfig(
        level=os.getenv("CYBERWAVE_LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cw = Cyberwave(
        token=token,
    )

    twin_uuid = os.getenv("TWIN_UUID")
    print("Twin UUID: ", twin_uuid)
    if not twin_uuid:
        print("Creating a new twin for streaming...")
        realsense = cw.twin("intel/realsensed455")
        twin_uuid = realsense.uuid
        print(f"Created camera twin: {twin_uuid}")
    
    # Connect MQTT client before using the streamer
    streamer = realsense.start_streaming()
    await streamer.start()
    print("Started realsense streamer")

    try:
        while True:
            await asyncio.sleep(1)
            print("Streaming...")
    except KeyboardInterrupt:
        print("Stopping realsense streamer...")
    except Exception as e:
        print(f"Error streaming realsense: {e}")
    finally:
        await streamer.stop()
        cw.disconnect()
        print("Disconnected")

if __name__ == "__main__":
    asyncio.run(main())
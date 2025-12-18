"""
Camera Streaming Example

This example demonstrates how to stream camera feed to a digital twin
using WebRTC with the improved developer experience.

Requirements:
    pip install cyberwave[camera]
"""

import asyncio
import logging

import os
from cyberwave import Cyberwave


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
        camera = cw.twin("cyberwave/standard-cam")
        twin_uuid = camera.uuid
        print(f"Created camera twin: {twin_uuid}")

    print(f"Starting camera stream to twin {twin_uuid}...")
    try:
        camera.start_streaming()
        print("Camera streaming started successfully!")
        print("Stream is active. Press Ctrl+C to stop...")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        camera.stop_streaming()
        print("\nStopping stream...")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        camera.stop_streaming()
        print("Stream stopped and resources cleaned up")


if __name__ == "__main__":
    asyncio.run(main())

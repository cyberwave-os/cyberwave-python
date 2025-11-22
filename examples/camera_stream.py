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

    client = Cyberwave(
        token=token,
    )

    twin_uuid = os.getenv("TWIN_UUID")
    print("Twin UUID: ", twin_uuid)
    if not twin_uuid:
        print("Creating a new twin for streaming...")
        robot = client.twin("cyberwave/standard-cam")
        twin_uuid = robot.uuid
        print(f"Created twin: {twin_uuid}")

    # Create camera streamer
    streamer = client.video_stream(
        twin_uuid=twin_uuid,
        camera_id=0,
        fps=10,
    )

    print(f"Starting camera stream to twin {twin_uuid}...")
    try:
        await streamer.start()
        print("Camera streaming started successfully!")
        print("Stream is active. Press Ctrl+C to stop...")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        client.mqtt.disconnect()
        print("\nStopping stream...")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        await streamer.stop()
        client.disconnect()
        print("Stream stopped and resources cleaned up")


if __name__ == "__main__":
    asyncio.run(main())

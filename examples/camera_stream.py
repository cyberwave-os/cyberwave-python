"""
Camera Streaming Example

Stream camera feed to a digital twin. Press Ctrl+C to stop.

Requirements:
    pip install cyberwave[camera]
"""

import asyncio
import os
from cyberwave import Cyberwave


async def main():
    cw = Cyberwave(token=os.getenv("CYBERWAVE_TOKEN"))
    camera = cw.twin("cyberwave/standard-cam")

    try:
        print(f"Streaming to twin {camera.uuid}... (Ctrl+C to stop)")
        await camera.start_streaming()

        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopping...")
    finally:
        await camera.stop_streaming()
        cw.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

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


async def read_keyboard_input(stop_event: asyncio.Event):
    """
    Read keyboard input asynchronously.
    Runs in a separate task and sets stop_event when 'q' is pressed.
    """
    # Skip keyboard input if stdin is not a TTY (e.g., running in background)
    if not sys.stdin.isatty():
        return
    
    loop = asyncio.get_event_loop()
    
    def read_stdin():
        """Read from stdin in a thread-safe way."""
        try:
            # Read a line from stdin (blocks until Enter is pressed)
            line = sys.stdin.readline().strip().lower()
            if line == 'q':
                return True
        except (EOFError, OSError, KeyboardInterrupt):
            # stdin closed, not available, or interrupted
            pass
        return False
    
    while not stop_event.is_set():
        try:
            # Run stdin read in executor to avoid blocking the event loop
            # This will block until user presses Enter, but that's acceptable
            should_stop = await loop.run_in_executor(None, read_stdin)
            if should_stop:
                stop_event.set()
                break
        except Exception:
            # If reading fails, exit the loop
            break


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

    streamer = None
    print(f"Starting camera stream to twin {twin_uuid}...")
    try:
        camera.start_streaming()
        print("Camera streaming started successfully!")
        print("Stream is active. Press 'q' and Enter to stop, or Ctrl+C...")

        # Create stop event for keyboard input
        stop_event = asyncio.Event()
        
        # Start keyboard input reader task
        keyboard_task = asyncio.create_task(read_keyboard_input(stop_event))

        # Main loop - wait for stop event or exception
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
        finally:
            # Cancel keyboard task if still running
            if not keyboard_task.done():
                keyboard_task.cancel()
                try:
                    await keyboard_task
                except asyncio.CancelledError:
                    pass
            
            if stop_event.is_set():
                print("\n'q' pressed. Stopping stream...")

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Stopping stream...")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        # Properly stop the streamer and disconnect
        if streamer is not None:
            try:
                print("Stopping camera streamer...")
                await streamer.stop()
            except Exception as e:
                print(f"Error stopping streamer: {e}")

        # Stop streaming on the twin (clears reference)
        try:
            camera.stop_streaming()
        except Exception as e:
            print(f"Error in stop_streaming: {e}")

        # Disconnect from MQTT and clean up client connections
        try:
            print("Disconnecting from MQTT...")
            cw.disconnect()
        except Exception as e:
            print(f"Error disconnecting: {e}")

        print("Stream stopped and resources cleaned up")


if __name__ == "__main__":
    asyncio.run(main())

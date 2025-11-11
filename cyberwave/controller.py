"""Edge controller functionality for Cyberwave SDK."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from .mqtt_client import CyberwaveMQTTClient

logger = logging.getLogger(__name__)

class EdgeController:
    """
    Manages Edge commands to Cyberwave platform.

    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        twin_uuid: Optional[str] = None,
    ):
        """
        Initialize the edge controller.

        Args:
            client: Cyberwave SDK client instance

            twin_uuid: Optional UUID of the digital twin
        """
        self.client = client
        self.twin_uuid: Optional[str] = twin_uuid
        self._pending_responses: Dict[str, asyncio.Future] = {}

    async def start(self, twin_uuid: Optional[str] = None):
        """
        Start streaming command controller to Cyberwave.

        Args:
            twin_uuid: UUID of the digital twin (uses instance twin_uuid if not provided)
        """
        # Use provided twin_uuid or fall back to instance twin_uuid
        if twin_uuid is not None:
            self.twin_uuid = twin_uuid
        elif self.twin_uuid is None:
            raise ValueError(
                "twin_uuid must be provided either during initialization or when calling start()"
            )

        logger.info(f"Starting controller for twin {self.twin_uuid}")

        if not self.client.connected:
            self.client.connect()

        self._subscribe_to_controller()

        await asyncio.sleep(2.5)

        logger.info(f"Controller ready for twin {self.twin_uuid}")


    def _subscribe_to_controller(self):
        """Subscribe to Edge command topic."""
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before subscribing")

        prefix = self.client.topic_prefix
        topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/command"

        def on_command_message(data):
            """Callback for command messages."""
            logger.debug(f"Received command message: {data}")
            try:
                payload = data if isinstance(data, dict) else json.loads(data)
                
                if "status" in payload:
                    status = payload.get("status")
                    
                    if status not in ["ok", "error"]:
                        logger.warning(f"Unexpected status value: {status}")
                        return
                    
                    if "start_video" in self._pending_responses:
                        self._resolve_pending_response("start_video", payload)
                    elif "stop_video" in self._pending_responses:
                        self._resolve_pending_response("stop_video", payload)
                    return
                
                command_type = payload.get("command")
                if command_type:
                    if command_type not in ["start_video", "stop_video"]:
                        logger.warning(f"Unknown command type: {command_type}")
            except Exception as e:
                logger.error(f"Error processing command message: {e}", exc_info=True)

        self.client.subscribe(topic, on_command_message, qos=2)

    def _resolve_pending_response(self, command: str, payload: Dict[str, Any]):
        """Resolve a pending response future."""
        if command in self._pending_responses:
            future = self._pending_responses.pop(command)
            if not future.done():
                future.set_result(payload)

    async def start_video(self):
        """Send start_video command and wait for response."""
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before sending commands")
        
        if not self.client.connected:
            self.client.connect()
        
        future = asyncio.Future()
        self._pending_responses["start_video"] = future
        
        prefix = self.client.topic_prefix
        topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/command"
        
        message = {
            "command": "start_video",
            "timestamp": time.time()
        }
        self.client.publish(topic, message, qos=2)

        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for start_video response")
            self._pending_responses.pop("start_video", None)
            return None
        except Exception as e:
            logger.error(f"Error waiting for start_video response: {e}", exc_info=True)
            self._pending_responses.pop("start_video", None)
            return None

    async def stop_video(self):
        """Send stop_video command and wait for response."""
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before sending commands")
        
        if not self.client.connected:
            self.client.connect()
        
        future = asyncio.Future()
        self._pending_responses["stop_video"] = future
        
        prefix = self.client.topic_prefix
        topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/command"
        
        message = {
            "command": "stop_video",
            "timestamp": time.time()
        }
        self.client.publish(topic, message, qos=2)

        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for stop_video response")
            self._pending_responses.pop("stop_video", None)
            return None
        except Exception as e:
            logger.error(f"Error waiting for stop_video response: {e}", exc_info=True)
            self._pending_responses.pop("stop_video", None)
            return None

    async def stop(self):
        """Stop controller and cleanup resources."""
        for command, future in self._pending_responses.items():
            if not future.done():
                future.cancel()
        self._pending_responses.clear()

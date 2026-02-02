"""Lightweight health check for edge data streaming."""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EdgeHealthCheck:
    """Publishes edge health status on a background thread.
    
    Simple, self-contained health monitoring without requiring BaseEdgeNode.
    Designed for data streaming scripts that need basic health reporting.
    
    Example:
        health = EdgeHealthCheck(
            mqtt_client=cw.mqtt,
            twin_uuids=["camera_uuid", "robot_uuid"],
        )
        health.start()
        
        # Update stats when frames are sent
        health.update_frame_count()
        
        health.stop()
    """
    
    def __init__(
        self,
        mqtt_client: Any,
        twin_uuids: List[str],
        edge_id: Optional[str] = None,
        stale_timeout: int = 30,
        interval: int = 5,
    ):
        """Initialize health publisher.
        
        Args:
            mqtt_client: MQTT client with publish() and topic_prefix
            twin_uuids: List of twin UUIDs to publish health to
            edge_id: Unique edge device ID (default: first twin UUID)
            stale_timeout: Seconds before stream is considered stale
            interval: Publish interval in seconds
        """
        self.mqtt_client = mqtt_client
        self.twin_uuids = twin_uuids
        self.edge_id = edge_id or twin_uuids[0]  # Default to first twin UUID
        self.stale_timeout = stale_timeout
        self.interval = interval
        
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    

    def update_frame_count(self):
        """Update frame count and timestamp (call this when sending frames)."""
        self.frame_count += 1
        self.last_frame_time = time.time()
    

    def start(self):
        """Start publishing health in background thread."""
        if self._thread and self._thread.is_alive():
            return  # Already running
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()
        logger.info(f"🟢 Health publisher started (interval={self.interval}s)")
    

    def stop(self):
        """Stop publishing health."""
        if not self._thread:
            return
        
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("🛑 Health publisher stopped")
    

    def get_health_data(self) -> Dict[str, Any]:
        """Build health data for current stream state."""
        now = time.time()
        uptime = now - self.start_time
        fps = self.frame_count / uptime if uptime > 0 else 0.0
        time_since_last = now - self.last_frame_time
        is_stale = time_since_last > self.stale_timeout
        
        return {
            "streams": {
                "stream": {
                    "camera_id": "stream",
                    "connection_state": "disconnected" if is_stale else "connected",
                    "ice_connection_state": "connected" if self.frame_count > 0 else "new",
                    "frames_sent": self.frame_count,
                    "last_frame_ts": self.last_frame_time,
                    "fps": round(fps, 2),
                    "uptime_seconds": round(uptime, 1),
                    "restart_count": 0,
                    "is_stale": is_stale,
                    "is_healthy": not is_stale,
                }
            },
            "stream_count": 1,
            "healthy_streams": 0 if is_stale else 1,
            "camera_config": None,
        }
    

    def _publish_loop(self):
        """Background thread loop for publishing health."""
        while not self._stop_event.is_set():
            try:
                # Build health data
                health_data = self.get_health_data()
                
                # Build complete payload
                now = time.time()
                base_payload = {
                    "type": "edge_health",
                    "timestamp": now,
                    "edge_id": self.edge_id,
                    "uptime_seconds": round(now - self.start_time, 1),
                    **health_data,  # Include streams, stream_count, etc.
                }
                
                # Publish to each twin UUID
                prefix = getattr(self.mqtt_client, "topic_prefix", "")
                for twin_uuid in self.twin_uuids:
                    if not twin_uuid:
                        continue
                    
                    payload = dict(base_payload, twin_uuid=twin_uuid)
                    topic = f"{prefix}cyberwave/twin/{twin_uuid}/edge_health"
                    
                    try:
                        self.mqtt_client.publish(topic, json.dumps(payload))
                    except Exception as e:
                        # Silently ignore publish errors to avoid spam
                        pass
            
            except Exception as e:
                logger.warning(f"⚠️ Health publish error: {e}")
            
            # Wait for next interval
            self._stop_event.wait(self.interval)

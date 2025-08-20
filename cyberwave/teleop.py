from __future__ import annotations

from typing import Any, Dict, List, Optional

from .http import HttpClient


class TeleopAPI:
    def __init__(self, http: HttpClient):
        self._h = http

    def start(self, twin_uuid: str, sensors: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._h.post(f"/api/v1/twins/{twin_uuid}/teleop/start", {"sensors": sensors or [], "metadata": metadata or {}})

    def stop(self, twin_uuid: str) -> Dict[str, Any]:
        return self._h.post(f"/api/v1/twins/{twin_uuid}/teleop/stop", {})

    def mark_outcome(self, twin_uuid: str, outcome: str) -> Dict[str, Any]:
        return self._h.post(f"/api/v1/twins/{twin_uuid}/teleop/mark", {"outcome": outcome})

    def session(self, twin_uuid: str, sensors: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for a teleop session."""
        api = self

        class _Ctx:
            def __enter__(self_inner):
                api.start(twin_uuid, sensors=sensors, metadata=metadata)
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                api.stop(twin_uuid)

        return _Ctx()



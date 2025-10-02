from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient
from .http import HttpClient


class TeleopAPI:
    """Teleoperation API following proper segregation of competence"""

    def __init__(self, http: AsyncHttpClient, sync_http: Optional[HttpClient] = None):
        self._h = http
        self._sync_http = sync_http

    @staticmethod
    def _with_leading_slash(path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    def _run_sync(self, coro):
        """Run an async coroutine in a synchronous context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            if loop.is_running():
                raise RuntimeError("Cannot run synchronous teleop call inside a running event loop")
            return loop.run_until_complete(coro)

    async def start(self, twin_uuid: str, sensors: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start teleoperation session"""
        return await self._h.post(f"twins/{twin_uuid}/teleop/start", {"sensors": sensors or [], "metadata": metadata or {}})

    def start_sync(self, twin_uuid: str, sensors: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"sensors": sensors or [], "metadata": metadata or {}}
        if self._sync_http:
            return self._sync_http.post(self._with_leading_slash(f"twins/{twin_uuid}/teleop/start"), json=payload)
        return self._run_sync(self.start(twin_uuid, sensors=sensors, metadata=metadata))

    async def stop(self, twin_uuid: str) -> Dict[str, Any]:
        """Stop teleoperation session"""
        return await self._h.post(f"twins/{twin_uuid}/teleop/stop", {})

    def stop_sync(self, twin_uuid: str) -> Dict[str, Any]:
        if self._sync_http:
            return self._sync_http.post(self._with_leading_slash(f"twins/{twin_uuid}/teleop/stop"), json={})
        return self._run_sync(self.stop(twin_uuid))

    async def mark_outcome(self, twin_uuid: str, outcome: str) -> Dict[str, Any]:
        """Mark teleoperation outcome"""
        return await self._h.post(f"twins/{twin_uuid}/teleop/mark", {"outcome": outcome})

    def mark_outcome_sync(self, twin_uuid: str, outcome: str) -> Dict[str, Any]:
        payload = {"outcome": outcome}
        if self._sync_http:
            return self._sync_http.post(self._with_leading_slash(f"twins/{twin_uuid}/teleop/mark"), json=payload)
        return self._run_sync(self.mark_outcome(twin_uuid, outcome))

    def session(self, twin_uuid: str, sensors: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for a teleop session."""
        api = self

        class _Ctx:
            def __enter__(self_inner):
                api.start_sync(twin_uuid, sensors=sensors, metadata=metadata)
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                api.stop_sync(twin_uuid)

            async def __aenter__(self_inner):
                await api.start(twin_uuid, sensors=sensors, metadata=metadata)
                return self_inner

            async def __aexit__(self_inner, exc_type, exc, tb):
                await api.stop(twin_uuid)

        return _Ctx()



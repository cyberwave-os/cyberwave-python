from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class RunsAPI:
    """Runs management API following proper segregation of competence"""
    
    def __init__(self, http: AsyncHttpClient):
        self._h = http

    async def start(
        self,
        *,
        environment_uuid: str,
        mission_key: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        mode: str = "virtual",
        mission_version: Optional[int] = None,
        mission_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start a mission run"""
        if not mission_key and not mission_spec:
            raise ValueError("Provide mission_key or mission_spec")
        payload = {
            "environment_uuid": environment_uuid,
            "mission_key": mission_key,
            "parameters": parameters or {},
            "mode": mode,
            "mission_version": mission_version,
            "mission_spec": mission_spec,
        }
        return await self._h.post("runs", payload)

    async def get(self, run_uuid: str) -> Dict[str, Any]:
        """Get run status"""
        return await self._h.get(f"runs/{run_uuid}")

    async def list(self) -> List[Dict[str, Any]]:
        """List all runs"""
        return await self._h.get("runs")

    async def stop(self, run_uuid: str) -> Dict[str, Any]:
        """Stop a running mission"""
        return await self._h.post(f"runs/{run_uuid}/stop", {})

    async def wait_until_complete(self, run_uuid: str, timeout_s: float = 120, poll_s: float = 1.0) -> Dict[str, Any]:
        """Wait for run completion with polling"""
        import asyncio
        end = time.time() + timeout_s
        while time.time() < end:
            info = await self.get(run_uuid)
            if info.get("status") in ("succeeded", "failed", "stopped"):
                return info
            await asyncio.sleep(poll_s)
        return await self.stop(run_uuid)



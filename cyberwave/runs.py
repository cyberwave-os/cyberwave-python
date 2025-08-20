from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .http import HttpClient


class RunsAPI:
    def __init__(self, http: HttpClient):
        self._h = http

    def start(
        self,
        *,
        environment_uuid: str,
        mission_key: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        mode: str = "virtual",
        mission_version: Optional[int] = None,
        mission_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        return self._h.post("/api/v1/runs", payload)

    def get(self, run_uuid: str) -> Dict[str, Any]:
        return self._h.get(f"/api/v1/runs/{run_uuid}")

    def list(self) -> List[Dict[str, Any]]:
        return self._h.get("/api/v1/runs")

    def stop(self, run_uuid: str) -> Dict[str, Any]:
        return self._h.post(f"/api/v1/runs/{run_uuid}/stop")

    def wait_until_complete(self, run_uuid: str, timeout_s: float = 120, poll_s: float = 1.0) -> Dict[str, Any]:
        end = time.time() + timeout_s
        while time.time() < end:
            info = self.get(run_uuid)
            if info.get("status") in ("succeeded", "failed", "stopped"):
                return info
            time.sleep(poll_s)
        return self.stop(run_uuid)



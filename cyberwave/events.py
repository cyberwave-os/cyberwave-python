"""Events API client for Cyberwave SDK"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class EventsAPI:
    """High level helper for interacting with the universal events API."""

    def __init__(self, http: AsyncHttpClient):
        self._http = http

    async def ingest(
        self,
        *,
        event_type: str,
        payload: Dict[str, Any],
        severity: str = "info",
        source: str = "sdk",
        environment_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Send an event into the universal event system."""
        body: Dict[str, Any] = {
            "event_type": event_type,
            "severity_level": severity,
            "source_kind": source,
            "payload": payload,
        }
        if environment_id:
            body["environment_id"] = environment_id
        if metadata:
            body["metadata"] = metadata
        if tags:
            body["tags"] = tags
        return await self._http.post("events/ingest", body)

    async def query(
        self,
        *,
        source: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        device_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        limit: int = 50,
        since_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query cached events using the backend filters."""
        params: Dict[str, Any] = {"limit": limit}
        if source:
            params["source"] = source
        if event_type:
            params["event_type"] = event_type
        if severity:
            params["severity"] = severity
        if device_id:
            params["device_id"] = device_id
        if environment_id:
            params["environment_id"] = environment_id
        if since_minutes is not None:
            params["since_minutes"] = since_minutes
        return await self._http.get("events/query", params=params)

    async def live(self, *, limit: int = 20) -> Dict[str, Any]:
        """Return the most recent events from the live endpoint."""
        return await self._http.get("events/live", params={"limit": limit})


async def ingest_event_async(
    api: EventsAPI,
    *,
    event_type: str,
    payload: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Coroutine helper used by the compact API to ingest events synchronously."""
    return await api.ingest(event_type=event_type, payload=payload, **kwargs)


def ingest_event(api: EventsAPI, **kwargs: Any) -> Dict[str, Any]:
    """Synchronous helper that bridges to :func:`ingest_event_async`."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return loop.create_task(ingest_event_async(api, **kwargs))  # type: ignore[return-value]

    return asyncio.run(ingest_event_async(api, **kwargs))

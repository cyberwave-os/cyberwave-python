"""Asset management API client."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class AssetsAPI:
    """High-level helpers for working with catalog assets."""

    def __init__(self, http: AsyncHttpClient):
        self._h = http

    async def list(
        self,
        *,
        registry_id: Optional[str] = None,
        asset_type: Optional[str] = None,
        project_uuid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List assets with optional filters."""

        params: Dict[str, Any] = {}
        if registry_id:
            params["registry_id"] = registry_id
        if asset_type:
            params["asset_type"] = asset_type
        if project_uuid:
            params["project_uuid"] = project_uuid

        result = await self._h.get("assets", params=params or None)
        if isinstance(result, list):
            return result

        # Some endpoints wrap results in a dictionary
        if isinstance(result, dict):
            items = result.get("results")
            if isinstance(items, list):
                return items

        return []

    async def get(self, asset_uuid: str) -> Dict[str, Any]:
        """Fetch a single asset by UUID."""

        return await self._h.get(f"assets/{asset_uuid}")

    async def find_by_registry_id(self, registry_id: str) -> Optional[Dict[str, Any]]:
        """Return the first asset that matches a registry identifier."""

        assets = await self.list(registry_id=registry_id)
        if assets:
            return assets[0]
        return None

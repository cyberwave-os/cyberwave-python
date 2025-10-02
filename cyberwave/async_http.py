"""
Unified AsyncHttpClient for all specialized APIs
Replaces the mixed async/sync HttpClient patterns
"""

from typing import Dict, Any, Optional
import httpx
import logging

logger = logging.getLogger("cyberwave.http")


class AsyncHttpClient:
    """Unified async HTTP client for all specialized APIs"""
    
    def __init__(self, base_url: str, access_token_getter=None, timeout: float = 10.0):
        """
        Initialize async HTTP client
        
        Args:
            base_url: The base URL for API requests
            access_token_getter: Function that returns current access token
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self._get_token = access_token_getter or (lambda: None)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        
    async def __aenter__(self) -> "AsyncHttpClient":
        return self
        
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
        
    async def aclose(self):
        """Close the HTTP client"""
        await self._client.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers including authorization if token available"""
        headers = {"Content-Type": "application/json"}
        token = self._get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
    
    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET request"""
        try:
            response = await self._client.get(
                path, 
                headers=self._get_headers(),
                params=params
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} GET {path}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"GET {path} failed: {e}")
            raise
    
    async def post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST request"""
        try:
            response = await self._client.post(
                path,
                json=data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} POST {path}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"POST {path} failed: {e}")
            raise
    
    async def put(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """PUT request"""
        try:
            response = await self._client.put(
                path,
                json=data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} PUT {path}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"PUT {path} failed: {e}")
            raise
    
    async def patch(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """PATCH request"""
        try:
            response = await self._client.patch(
                path,
                json=data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} PATCH {path}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"PATCH {path} failed: {e}")
            raise
    
    async def delete(self, path: str) -> Dict[str, Any]:
        """DELETE request"""
        try:
            response = await self._client.delete(
                path,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} DELETE {path}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"DELETE {path} failed: {e}")
            raise

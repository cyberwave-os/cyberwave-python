import sys
import os
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from unittest.mock import AsyncMock, MagicMock
from cyberwave import APIError
from tests.test_assets import MockResponse

@pytest.mark.asyncio
async def test_get_workspace_by_slug_success(mock_client):
    """Test getting a workspace by slug successfully."""
    mock_response = {"id": 1, "name": "Test Workspace", "slug": "test-workspace"}
    mock_client._client.request.return_value = MockResponse(mock_response, 200)
    test_slug = "test-workspace"

    result = await mock_client.get_workspace_by_slug(slug=test_slug)
    
    # Check the result
    assert result == mock_response
    # Check that the correct URL was called
    mock_client._client.request.assert_called_once_with(
        "GET",
        f"/workspaces/slug/{test_slug}",
        headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
    )

@pytest.mark.asyncio
async def test_get_workspace_by_slug_not_found(mock_client):
    """Test getting a workspace by slug when it doesn't exist (404)."""
    mock_client._client.request.side_effect = APIError(status_code=404, detail="Workspace not found")
    test_slug = "non-existent-workspace"

    result = await mock_client.get_workspace_by_slug(slug=test_slug)
    
    # Assert that None is returned on 404
    assert result is None
    mock_client._client.request.assert_called_once_with(
        "GET",
        f"/workspaces/slug/{test_slug}",
        headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
    )

@pytest.mark.asyncio
async def test_get_asset_catalog_success(mock_client):
    """Test getting an asset catalog by UUID successfully."""
    mock_response = {"uuid": "1", "name": "Cat"}
    mock_client._client.request.return_value = MockResponse(mock_response, 200)
    catalog_uuid = "1"

    result = await mock_client.get_asset_catalog(catalog_uuid)

    assert result == mock_response
    mock_client._client.request.assert_called_once_with(
        "GET",
        f"/asset-catalogs/{catalog_uuid}",
        headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
    )


@pytest.mark.asyncio
async def test_client_async_context_manager():
    """Ensure the client can be used as an async context manager."""
    from cyberwave import Client
    client = Client(use_token_cache=False)
    mock_aclose = AsyncMock()
    client._client = MagicMock()
    client._client.aclose = mock_aclose

    async with client as c:
        assert c is client

    mock_aclose.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_asset_catalog_not_found(mock_client):
    """Test getting an asset catalog when it doesn't exist (404)."""
    mock_client._client.request.side_effect = APIError(status_code=404, detail="Asset catalog not found")
    catalog_uuid = "missing"

    result = await mock_client.get_asset_catalog(catalog_uuid)

    assert result is None
    mock_client._client.request.assert_called_once_with(
        "GET",
        f"/asset-catalogs/{catalog_uuid}",
        headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
    )

@pytest.mark.asyncio
async def test_list_asset_catalogs_success(mock_client):
    """Test listing asset catalogs successfully."""
    mock_response = [{"uuid": "1", "name": "Cat"}]
    mock_client._client.request.return_value = MockResponse(mock_response, 200)

    result = await mock_client.list_asset_catalogs()

    assert result == mock_response
    mock_client._client.request.assert_called_once_with(
        "GET",
        "/asset-catalogs",
        headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
    )

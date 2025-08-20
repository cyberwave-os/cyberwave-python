import unittest
from unittest.mock import AsyncMock, MagicMock
import sys
import types
import importlib.util
from pathlib import Path

# Stub out the optional httpx dependency if it's missing
if 'httpx' not in sys.modules:
    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def aclose(self):
            pass
        async def request(self, *args, **kwargs):
            pass
    sys.modules['httpx'] = types.SimpleNamespace(
        AsyncClient=DummyAsyncClient,
        Response=object,
        HTTPStatusError=Exception,
    )
if 'numpy' not in sys.modules:
    sys.modules['numpy'] = types.ModuleType('numpy')
if 'rerun' not in sys.modules:
    sys.modules['rerun'] = types.ModuleType('rerun')
if 'aiofiles' not in sys.modules:
    sys.modules['aiofiles'] = types.ModuleType('aiofiles')
if 'pydantic' not in sys.modules:
    pydantic_stub = types.ModuleType('pydantic')
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None
    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    pydantic_stub.ValidationError = Exception
    sys.modules['pydantic'] = pydantic_stub
if 'yaml' not in sys.modules:
    yaml_stub = types.ModuleType('yaml')
    yaml_stub.add_representer = lambda *args, **kwargs: None
    sys.modules['yaml'] = yaml_stub
if 'jsonschema' not in sys.modules:
    sys.modules['jsonschema'] = types.ModuleType('jsonschema')

geometry_stub = types.ModuleType('cyberwave.geometry')
for name in ['Mesh', 'Skeleton', 'Joint', 'FloorPlan', 'Wall', 'Point3D', 'Sensor', 'Zone']:
    setattr(geometry_stub, name, type(name, (), {}))
def noop(*args, **kwargs):
    pass
geometry_stub.log_mesh_rr = noop
geometry_stub.log_skeleton_rr = noop
sys.modules['cyberwave.geometry'] = geometry_stub

# Use the actual Cyberwave Client class from the SDK
from cyberwave.client import Client


class MockResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestAssetClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = Client(base_url="http://testserver", use_token_cache=False)
        self.client._access_token = "test-token"
        self.mock_request = AsyncMock()
        # Replace internal HTTP client's request method
        self.client._client = MagicMock()
        self.client._client.request = self.mock_request

    async def test_list_asset_catalogs(self):
        expected = [{"uuid": "1", "name": "Cat"}]
        self.mock_request.return_value = MockResponse(expected, 200)

        result = await self.client.list_asset_catalogs()

        self.mock_request.assert_called_once_with(
            "GET",
            "/asset-catalogs",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
        )
        self.assertEqual(result, expected)

    async def test_get_asset_catalog(self):
        expected = {"uuid": "1", "name": "Cat"}
        self.mock_request.return_value = MockResponse(expected, 200)

        result = await self.client.get_asset_catalog("1")

        self.mock_request.assert_called_once_with(
            "GET",
            "/asset-catalogs/1",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
        )
        self.assertEqual(result, expected)

    async def test_create_asset_catalog(self):
        expected = {"uuid": "1", "name": "Cat"}
        self.mock_request.return_value = MockResponse(expected, 201)

        result = await self.client.create_asset_catalog("Cat", "desc", public=True)

        self.mock_request.assert_called_once_with(
            "POST",
            "/asset-catalogs",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
            json={"name": "Cat", "description": "desc", "public": True},
        )
        self.assertEqual(result, expected)

    async def test_update_asset_catalog(self):
        expected = {"uuid": "1", "name": "Cat"}
        self.mock_request.return_value = MockResponse(expected, 200)

        result = await self.client.update_asset_catalog("1", "Cat", "desc", public=True)

        self.mock_request.assert_called_once_with(
            "PUT",
            "/asset-catalogs/1",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
            json={"name": "Cat", "description": "desc", "public": True},
        )
        self.assertEqual(result, expected)

    async def test_delete_asset_catalog(self):
        expected = {"success": True}
        self.mock_request.return_value = MockResponse(expected, 200)

        result = await self.client.delete_asset_catalog("1")

        self.mock_request.assert_called_once_with(
            "DELETE",
            "/asset-catalogs/1",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
        )
        self.assertEqual(result, expected)

    async def test_list_assets(self):
        expected = [{"uuid": "a1", "name": "Asset"}]
        self.mock_request.return_value = MockResponse(expected, 200)

        result = await self.client.list_assets()

        self.mock_request.assert_called_once_with(
            "GET",
            "/assets",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
        )
        self.assertEqual(result, expected)

    async def test_get_asset(self):
        expected = {"uuid": "a1", "name": "Asset"}
        self.mock_request.return_value = MockResponse(expected, 200)

        result = await self.client.get_asset("a1")

        self.mock_request.assert_called_once_with(
            "GET",
            "/assets/a1",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
        )
        self.assertEqual(result, expected)

    async def test_create_asset(self):
        expected = {"uuid": "a1", "name": "Asset"}
        self.mock_request.return_value = MockResponse(expected, 201)

        result = await self.client.create_asset("Asset", "desc", 1, 2, registry_id="so/100")

        self.mock_request.assert_called_once_with(
            "POST",
            "/assets",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
            json={
                "name": "Asset",
                "description": "desc",
                "asset_catalog_id": 1,
                "level_id": 2,
                "registry_id": "so/100",
            },
        )
        self.assertEqual(result, expected)

    async def test_update_asset(self):
        expected = {"uuid": "a1", "name": "Asset"}
        self.mock_request.return_value = MockResponse(expected, 200)

        result = await self.client.update_asset("a1", "Asset", "desc", 1, 2, registry_id="so/100")

        self.mock_request.assert_called_once_with(
            "PUT",
            "/assets/a1",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
            json={
                "name": "Asset",
                "description": "desc",
                "asset_catalog_id": 1,
                "level_id": 2,
                "registry_id": "so/100",
            },
        )
        self.assertEqual(result, expected)

    async def test_delete_asset(self):
        expected = {"success": True}
        self.mock_request.return_value = MockResponse(expected, 200)

        result = await self.client.delete_asset("a1")

        self.mock_request.assert_called_once_with(
            "DELETE",
            "/assets/a1",
            headers={"Accept": "application/json", "Authorization": "Bearer test-token"},
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

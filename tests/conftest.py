import pytest
from unittest.mock import AsyncMock, MagicMock
import sys
import types
import httpx

# Provide lightweight stubs if optional deps are missing
if 'httpx' not in sys.modules:
    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def request(self, *args, **kwargs):
            pass
        async def aclose(self):
            pass

    class DummyResponse:
        def __init__(self, status_code: int = 200):
            self.status_code = status_code
            self.request = None
        def json(self):
            return {}

    class DummyHTTPStatusError(Exception):
        def __init__(self, message: str = "", request=None, response=None):
            super().__init__(message)
            self.request = request
            self.response = response

    sys.modules['httpx'] = types.SimpleNamespace(
        AsyncClient=DummyAsyncClient,
        Response=DummyResponse,
        HTTPStatusError=DummyHTTPStatusError,
    )

if 'numpy' not in sys.modules:
    sys.modules['numpy'] = types.ModuleType('numpy')

if 'typer' not in sys.modules:
    typer_stub = types.ModuleType('typer')
    typer_stub.Typer = lambda *a, **k: None
    typer_testing = types.ModuleType('typer.testing')
    class DummyRunner:
        def invoke(self, *args, **kwargs):
            return types.SimpleNamespace(exit_code=0, stdout="")
    typer_testing.CliRunner = DummyRunner
    typer_stub.testing = typer_testing
    sys.modules['typer'] = typer_stub
    sys.modules['typer.testing'] = typer_testing

if 'django' not in sys.modules:
    sys.modules['django'] = types.ModuleType('django')

if 'rerun' not in sys.modules:
    sys.modules['rerun'] = types.ModuleType('rerun')

import cyberwave


@pytest.fixture
def mock_client():
    """Provides a CyberWave Client with the HTTP layer mocked."""
    client = cyberwave.Client(use_token_cache=False)
    client._access_token = "test-token"
    mock_request = AsyncMock()
    client._client = MagicMock()
    client._client.request = mock_request
    yield client

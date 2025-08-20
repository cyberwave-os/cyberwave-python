import sys
import types

# Provide minimal stub modules for optional dependencies used by cyberwave
if 'numpy' not in sys.modules:
    sys.modules['numpy'] = types.ModuleType('numpy')
if 'httpx' not in sys.modules:
    httpx_stub = types.ModuleType('httpx')
    class DummyClient:
        pass
    httpx_stub.AsyncClient = DummyClient
    sys.modules['httpx'] = httpx_stub
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

import cyberwave


def test_robot_in_all():
    assert hasattr(cyberwave, 'Robot')

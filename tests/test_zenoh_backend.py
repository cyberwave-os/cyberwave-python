"""Tests specific to the ZenohBackend implementation.

These are skipped when ``eclipse-zenoh`` is not installed.
"""

from unittest.mock import patch

import pytest

from cyberwave.data.exceptions import BackendUnavailableError

try:
    import zenoh  # noqa: F401

    _has_zenoh = True
except ImportError:
    _has_zenoh = False

pytestmark = pytest.mark.skipif(not _has_zenoh, reason="eclipse-zenoh not installed")


@pytest.fixture
def backend():
    from cyberwave.data.zenoh_backend import ZenohBackend

    be = ZenohBackend()
    yield be
    be.close()


class TestZenohSession:
    def test_session_opens(self, backend):
        assert backend._session is not None

    def test_close_idempotent(self, backend):
        backend.close()
        backend.close()


class TestSharedMemoryConfig:
    def test_shared_memory_flag_accepted(self):
        from cyberwave.data.zenoh_backend import ZenohBackend

        try:
            be = ZenohBackend(shared_memory=True)
        except BackendUnavailableError:
            pytest.skip("POSIX shared memory not available on this platform")
        assert be._session is not None
        be.close()


class TestConnectEndpoints:
    def test_custom_endpoints_accepted(self):
        from cyberwave.data.zenoh_backend import ZenohBackend

        be = ZenohBackend(connect=["tcp/127.0.0.1:7447"])
        assert be._session is not None
        be.close()


class TestImportError:
    def test_missing_zenoh_gives_clear_error(self):
        with patch.dict("sys.modules", {"zenoh": None}):
            import importlib

            from cyberwave.data import zenoh_backend

            orig = zenoh_backend._has_zenoh
            zenoh_backend._has_zenoh = False
            try:
                with pytest.raises(BackendUnavailableError, match="eclipse-zenoh"):
                    zenoh_backend.ZenohBackend()
            finally:
                zenoh_backend._has_zenoh = orig

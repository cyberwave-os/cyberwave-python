"""Tests for data-layer configuration and the get_backend() factory."""

import os
from unittest.mock import patch

import pytest

from cyberwave.data.config import BackendConfig, _parse_int_env, get_backend
from cyberwave.data.exceptions import BackendConfigError, BackendUnavailableError
from cyberwave.data.filesystem_backend import FilesystemBackend

try:
    import zenoh  # noqa: F401

    _has_zenoh = True
except ImportError:
    _has_zenoh = False


class TestBackendConfig:
    def test_default_backend_is_zenoh(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CYBERWAVE_DATA_BACKEND", None)
            cfg = BackendConfig()
            assert cfg.backend == "zenoh"

    def test_env_var_selects_filesystem(self):
        with patch.dict(os.environ, {"CYBERWAVE_DATA_BACKEND": "filesystem"}):
            cfg = BackendConfig()
            assert cfg.backend == "filesystem"

    def test_zenoh_connect_from_env(self):
        with patch.dict(
            os.environ, {"ZENOH_CONNECT": "tcp/localhost:7447, tcp/10.0.0.1:7447"}
        ):
            cfg = BackendConfig()
            assert cfg.zenoh_connect == ["tcp/localhost:7447", "tcp/10.0.0.1:7447"]

    def test_zenoh_connect_empty_env(self):
        with patch.dict(os.environ, {"ZENOH_CONNECT": ""}):
            cfg = BackendConfig()
            assert cfg.zenoh_connect == []

    def test_shared_memory_from_env(self):
        with patch.dict(os.environ, {"ZENOH_SHARED_MEMORY": "true"}):
            cfg = BackendConfig()
            assert cfg.zenoh_shared_memory is True

    def test_shared_memory_false_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ZENOH_SHARED_MEMORY", None)
            cfg = BackendConfig()
            assert cfg.zenoh_shared_memory is False

    def test_filesystem_base_dir_from_env(self):
        with patch.dict(os.environ, {"CYBERWAVE_DATA_DIR": "/custom/path"}):
            cfg = BackendConfig()
            assert cfg.filesystem_base_dir == "/custom/path"

    def test_explicit_config_overrides_env(self):
        with patch.dict(os.environ, {"CYBERWAVE_DATA_BACKEND": "zenoh"}):
            cfg = BackendConfig(backend="filesystem")
            assert cfg.backend == "filesystem"

    def test_key_prefix_default(self):
        cfg = BackendConfig()
        assert cfg.key_prefix == "cw"


class TestParseIntEnv:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("42", 42),
            ("  7 ", 7),
            ("0", 0),
            ("-5", -5),
            (None, None),
            ("", None),
            ("   ", None),
            ("notanint", None),
            ("1.5", None),
        ],
    )
    def test_parse(self, value, expected):
        assert _parse_int_env(value) == expected


class TestShmConfig:
    def test_pool_bytes_from_env(self):
        with patch.dict(os.environ, {"ZENOH_SHM_POOL_BYTES": "134217728"}):
            assert BackendConfig().zenoh_shm_pool_bytes == 134217728

    def test_pool_bytes_invalid_is_none(self):
        with patch.dict(os.environ, {"ZENOH_SHM_POOL_BYTES": "huge"}):
            assert BackendConfig().zenoh_shm_pool_bytes is None

    def test_pool_bytes_unset_is_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert BackendConfig().zenoh_shm_pool_bytes is None

    def test_min_bytes_from_env(self):
        with patch.dict(os.environ, {"ZENOH_SHM_MIN_BYTES": "8192"}):
            assert BackendConfig().zenoh_shm_min_bytes == 8192

    def test_on_exhaustion_from_env(self):
        with patch.dict(os.environ, {"ZENOH_SHM_ON_EXHAUSTION": "drop"}):
            assert BackendConfig().zenoh_shm_on_exhaustion == "drop"

    def test_on_exhaustion_defaults_to_copy(self):
        with patch.dict(os.environ, {}, clear=True):
            assert BackendConfig().zenoh_shm_on_exhaustion == "copy"

    def test_on_exhaustion_invalid_warns_and_defaults(self):
        with patch.dict(os.environ, {"ZENOH_SHM_ON_EXHAUSTION": "bogus"}):
            with pytest.warns(UserWarning, match="ZENOH_SHM_ON_EXHAUSTION"):
                cfg = BackendConfig()
            assert cfg.zenoh_shm_on_exhaustion == "copy"

    def test_explicit_values_override_env(self):
        with patch.dict(
            os.environ,
            {"ZENOH_SHM_ON_EXHAUSTION": "drop", "ZENOH_SHM_POOL_BYTES": "999"},
        ):
            cfg = BackendConfig(
                zenoh_shm_on_exhaustion="copy", zenoh_shm_pool_bytes=123
            )
            assert cfg.zenoh_shm_on_exhaustion == "copy"
            assert cfg.zenoh_shm_pool_bytes == 123


class TestGetBackendFactory:
    def test_filesystem_backend_created(self, tmp_path):
        cfg = BackendConfig(
            backend="filesystem",
            filesystem_base_dir=str(tmp_path / "factory_test"),
        )
        be = get_backend(cfg)
        assert isinstance(be, FilesystemBackend)
        be.close()

    @pytest.mark.skipif(not _has_zenoh, reason="eclipse-zenoh not installed")
    def test_zenoh_backend_created(self):
        from cyberwave.data.zenoh_backend import ZenohBackend

        cfg = BackendConfig(backend="zenoh")
        be = get_backend(cfg)
        assert isinstance(be, ZenohBackend)
        be.close()

    def test_unknown_backend_raises(self):
        cfg = BackendConfig(backend="redis")
        with pytest.raises(BackendConfigError, match="Unknown data backend.*redis"):
            get_backend(cfg)

    def test_unknown_backend_lists_supported(self):
        cfg = BackendConfig(backend="nope")
        with pytest.raises(BackendConfigError, match="zenoh.*filesystem"):
            get_backend(cfg)

    def test_zenoh_unavailable_raises(self):
        from cyberwave.data import zenoh_backend

        orig = zenoh_backend._has_zenoh
        zenoh_backend._has_zenoh = False
        try:
            cfg = BackendConfig(backend="zenoh")
            with pytest.raises(BackendUnavailableError, match="eclipse-zenoh"):
                get_backend(cfg)
        finally:
            zenoh_backend._has_zenoh = orig

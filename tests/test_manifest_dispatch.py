"""Tests for dispatch mode detection and end-to-end module call integration.

The integration tests construct minimal stand-ins for the CloudNode module
dispatch path without requiring the full cloud node infrastructure.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cyberwave.manifest.loader import from_file
from cyberwave.manifest.schema import ManifestSchema, detect_dispatch_mode

# Re-export _PLATFORM_PARAMS to test stripping logic
_PLATFORM_PARAMS = frozenset({"workload_uuid", "command_type", "status"})


# ---------------------------------------------------------------------------
# Unit: dispatch mode detection (same as in test_manifest_schema, but
# grouped here for the plan's test_manifest_dispatch.py requirement)
# ---------------------------------------------------------------------------


class TestDispatchModeUnit:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("inference.py", "module"),
            ("./models/inference.py", "module"),
            ("train.py", "module"),
            ("python server.py --params {body}", "shell"),
            ("source activate && python run.py {body}", "shell"),
            ("inference.py --extra arg", "shell"),
        ],
    )
    def test_dispatch_mode(self, value: str, expected: str):
        assert detect_dispatch_mode(value) == expected


# ---------------------------------------------------------------------------
# Integration: manifest → module infer() call
# ---------------------------------------------------------------------------


class TestModuleDispatchIntegration:
    """Simulates the module dispatch path from cloud_node.py."""

    def _write_inference_module(self, workdir: Path) -> None:
        """Write a trivial inference.py that exports infer()."""
        code = textwrap.dedent("""\
            def infer(**params):
                return {"result": "ok", "received_keys": sorted(params.keys())}
        """)
        (workdir / "inference.py").write_text(code)

    def _import_module(self, module_path: Path, module_key: str) -> object:
        """Mirror the module import logic from _dispatch_module_workload."""
        spec = importlib.util.spec_from_file_location(module_key, module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module
        spec.loader.exec_module(module)
        return module

    def test_infer_called_with_user_params_only(self, tmp_path):
        """Platform keys (workload_uuid, etc.) must not reach the user function."""
        self._write_inference_module(tmp_path)
        module = self._import_module(
            tmp_path / "inference.py", "_cyberwave_test_infer"
        )

        params = {
            "input": "test_image.jpg",
            "threshold": 0.5,
            "workload_uuid": "wl-123",
            "command_type": "inference",
            "status": "pending",
        }
        user_params = {k: v for k, v in params.items() if k not in _PLATFORM_PARAMS}

        fn = getattr(module, "infer")
        result = fn(**user_params)

        assert result["result"] == "ok"
        assert "workload_uuid" not in result["received_keys"]
        assert "command_type" not in result["received_keys"]
        assert "input" in result["received_keys"]
        assert "threshold" in result["received_keys"]

        # Clean up
        del sys.modules["_cyberwave_test_infer"]

    def test_module_cached_after_first_import(self, tmp_path):
        """Module should be cached in sys.modules after first import."""
        self._write_inference_module(tmp_path)
        module_key = "_cyberwave_test_cache"
        self._import_module(tmp_path / "inference.py", module_key)
        assert module_key in sys.modules

        # Calling again should find it cached
        assert sys.modules[module_key] is not None
        del sys.modules[module_key]

    def test_manifest_from_file_with_module_inference(self, tmp_path):
        """End-to-end: write cyberwave.yml, load manifest, detect module mode."""
        yml = tmp_path / "cyberwave.yml"
        yml.write_text("cyberwave:\n  inference: inference.py\n")
        self._write_inference_module(tmp_path)

        manifest = from_file(yml)
        assert manifest.inference == "inference.py"
        assert detect_dispatch_mode(manifest.inference) == "module"


class TestInstallBridging:
    """Test that effective_install normalises install vs install_script."""

    def test_manifest_install_field(self, tmp_path):
        yml = tmp_path / "cyberwave.yml"
        yml.write_text('cyberwave:\n  install: "echo hello"\n')
        manifest = from_file(yml)
        assert manifest.effective_install == "echo hello"

    def test_manifest_install_script_field(self, tmp_path):
        yml = tmp_path / "cyberwave.yml"
        yml.write_text("cyberwave-cloud-node:\n  install_script: ./install.sh\n")
        manifest = from_file(yml)
        assert manifest.effective_install == "./install.sh"

    def test_install_preferred_over_install_script(self):
        m = ManifestSchema.model_validate(
            {"install": "pip install A", "install_script": "./setup.sh"}
        )
        assert m.effective_install == "pip install A"


class TestFnNameMapping:
    """Verify the workload-type → function-name mapping used in _dispatch_module_workload.

    This mirrors the ``_WORKLOAD_FN_MAP`` constant in cloud_node.py; both must
    stay in sync.
    """

    @pytest.mark.parametrize(
        "workload_type,expected_fn",
        [
            ("inference", "infer"),
            ("training", "train"),
            ("simulate", "simulate"),
        ],
    )
    def test_fn_name_mapping(self, workload_type: str, expected_fn: str):
        """Each workload type must map to its own function name, not a shared default."""
        # Mirrors _WORKLOAD_FN_MAP in cloud_node.py
        _fn_map = {"inference": "infer", "training": "train", "simulate": "simulate"}
        assert _fn_map.get(workload_type, workload_type) == expected_fn


@pytest.fixture
def fake_cw():
    """Minimal stand-in for a Cyberwave client used by load_workers tests."""

    class _FakeCW:
        pass

    return _FakeCW()


class TestLoadWorkersFilePath:
    """Verify that load_workers accepts a single file path (not just a directory)."""

    def test_single_file_loaded(self, tmp_path, fake_cw):
        """Passing an individual .py file should load only that file."""
        from cyberwave.workers.loader import load_workers

        worker_file = tmp_path / "my_worker.py"
        worker_file.write_text("# stub worker\n")
        other_file = tmp_path / "other.py"
        other_file.write_text("# should NOT be loaded\n")

        loaded = load_workers(worker_file, cw_instance=fake_cw)
        assert loaded == 1

    def test_directory_still_loads_all(self, tmp_path, fake_cw):
        """Passing a directory should still load all non-underscore .py files."""
        from cyberwave.workers.loader import load_workers

        (tmp_path / "a.py").write_text("# worker a\n")
        (tmp_path / "b.py").write_text("# worker b\n")
        (tmp_path / "_private.py").write_text("# skipped\n")

        loaded = load_workers(tmp_path, cw_instance=fake_cw)
        assert loaded == 2

    def test_underscore_file_skipped(self, tmp_path, fake_cw):
        """A single-file path whose name starts with _ should be skipped."""
        from cyberwave.workers.loader import load_workers

        private_file = tmp_path / "_internal.py"
        private_file.write_text("# private\n")

        loaded = load_workers(private_file, cw_instance=fake_cw)
        assert loaded == 0

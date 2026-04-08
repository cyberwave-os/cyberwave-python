"""Tests for cyberwave.manifest.schema — positive/negative validation matrix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cyberwave.manifest.schema import ManifestSchema, detect_dispatch_mode


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


class TestManifestSchemaPositive:
    def test_empty_block_is_valid(self):
        m = ManifestSchema.model_validate({})
        assert m.version == "1"
        assert m.inference is None
        assert m.gpu is False
        assert m.profile_slug == "default"

    def test_full_readme_example(self):
        data = {
            "name": "my-model",
            "install": "pip install ultralytics",
            "inference": "inference.py",
            "training": "python train.py {body}",
            "workers": ["detector.py"],
            "input": ["image", "depth"],
            "gpu": True,
            "requirements": ["ultralytics>=8.0"],
            "models": ["yolov8n"],
            "resources": {"memory": "4g", "cpus": 2.0},
        }
        m = ManifestSchema.model_validate(data)
        assert m.name == "my-model"
        assert m.gpu is True
        assert m.workers == ["detector.py"]
        assert m.resources is not None
        assert m.resources.memory == "4g"

    def test_install_field_accepted(self):
        m = ManifestSchema.model_validate({"install": "pip install X"})
        assert m.effective_install == "pip install X"

    def test_install_script_field_accepted(self):
        m = ManifestSchema.model_validate({"install_script": "./install.sh"})
        assert m.effective_install == "./install.sh"

    def test_install_takes_priority_over_install_script(self):
        m = ManifestSchema.model_validate(
            {"install": "pip install A", "install_script": "./install.sh"}
        )
        assert m.effective_install == "pip install A"

    def test_input_string_normalised(self):
        m = ManifestSchema.model_validate({"input": "image"})
        assert m.input == ["image"]

    def test_input_list_preserved(self):
        m = ManifestSchema.model_validate({"input": ["image", "depth"]})
        assert m.input == ["image", "depth"]

    def test_version_1(self):
        m = ManifestSchema.model_validate({"version": "1"})
        assert m.version == "1"

    def test_bash_inference_valid(self):
        m = ManifestSchema.model_validate({"inference": "python server.py {body}"})
        assert detect_dispatch_mode(m.inference) == "shell"  # type: ignore[arg-type]

    def test_module_inference_valid(self):
        m = ManifestSchema.model_validate({"inference": "inference.py"})
        assert detect_dispatch_mode(m.inference) == "module"  # type: ignore[arg-type]

    def test_runtime_with_model_valid(self):
        m = ManifestSchema.model_validate({"runtime": "ultralytics", "model": "yolov8n.pt"})
        assert m.runtime == "ultralytics"
        assert m.model == "yolov8n.pt"

    def test_simulate_field_valid(self):
        m = ManifestSchema.model_validate({"simulate": "./sim.sh"})
        assert m.simulate == "./sim.sh"

    def test_requirements_stored(self):
        m = ManifestSchema.model_validate({"requirements": ["ultralytics>=8.0"]})
        assert m.requirements == ["ultralytics>=8.0"]

    def test_models_stored(self):
        m = ManifestSchema.model_validate({"models": ["yolov8n"]})
        assert m.models == ["yolov8n"]

    def test_effective_install_none_when_both_absent(self):
        m = ManifestSchema.model_validate({})
        assert m.effective_install is None


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


class TestManifestSchemaNegative:
    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="inference_timeout"):
            ManifestSchema.model_validate({"inference_timeout": 30})

    def test_unsupported_version(self):
        with pytest.raises(ValidationError, match="Unsupported manifest version"):
            ManifestSchema.model_validate({"version": "99"})

    def test_runtime_without_model_or_inference(self):
        with pytest.raises(ValidationError, match="'runtime' is set"):
            ManifestSchema.model_validate({"runtime": "ultralytics"})

    def test_heartbeat_interval_not_string(self):
        with pytest.raises(ValidationError, match="heartbeat_interval"):
            ManifestSchema.model_validate({"heartbeat_interval": "fast"})

    def test_workers_must_be_list(self):
        with pytest.raises(ValidationError, match="workers"):
            ManifestSchema.model_validate({"workers": "detector.py"})

    def test_requirements_must_be_list(self):
        with pytest.raises(ValidationError, match="requirements"):
            ManifestSchema.model_validate({"requirements": "ultralytics"})


# ---------------------------------------------------------------------------
# dispatch mode detection
# ---------------------------------------------------------------------------


class TestDetectDispatchMode:
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

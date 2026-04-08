"""Tests for cyberwave.manifest.loader — key normalisation and from_file()."""

from __future__ import annotations

import pytest
import yaml

from cyberwave.manifest.loader import from_dict, from_file


class TestFromDict:
    def test_cyberwave_key_loaded(self):
        m = from_dict({"cyberwave": {"inference": "model.py"}})
        assert m.inference == "model.py"

    def test_cloud_node_key_loaded(self):
        m = from_dict({"cyberwave-cloud-node": {"install_script": "./setup.sh"}})
        assert m.install_script == "./setup.sh"

    def test_cyberwave_key_takes_priority(self):
        m = from_dict(
            {
                "cyberwave": {"inference": "a.py"},
                "cyberwave-cloud-node": {"inference": "b.py"},
            }
        )
        assert m.inference == "a.py"

    def test_flat_format_all_known_fields_loaded(self):
        m = from_dict({"inference": "model.py", "gpu": True})
        assert m.inference == "model.py"
        assert m.gpu is True

    def test_flat_format_with_unknown_fields_raises(self):
        with pytest.raises(ValueError, match="No 'cyberwave:'"):
            from_dict({"services": {"web": {}}, "volumes": {}})

    def test_null_block_returns_defaults(self):
        m = from_dict({"cyberwave": None})
        assert m.inference is None
        assert m.version == "1"


class TestFromFile:
    def test_from_file_basic(self, tmp_path):
        yml = tmp_path / "cyberwave.yml"
        yml.write_text(yaml.dump({"cyberwave": {"inference": "model.py"}}))
        m = from_file(yml)
        assert m.inference == "model.py"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            from_file(tmp_path / "cyberwave.yml")

    def test_readme_front_matter_fallback(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("---\ncyberwave:\n  inference: readme_model.py\n---\n# Hello\n")
        m = from_file(tmp_path / "cyberwave.yml")
        assert m.inference == "readme_model.py"

    def test_malformed_yaml_raises(self, tmp_path):
        yml = tmp_path / "cyberwave.yml"
        yml.write_text("cyberwave:\n  - broken: [\n")
        with pytest.raises(yaml.YAMLError):
            from_file(yml)

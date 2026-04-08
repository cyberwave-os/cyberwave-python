"""Tests for cyberwave.manifest.validator — warnings, lenient mode, legacy key detection."""

from __future__ import annotations

import yaml

from cyberwave.manifest.validator import validate_manifest


class TestLegacyKeyWarning:
    def test_legacy_key_warning_from_dict(self):
        result = validate_manifest(data={"cyberwave-cloud-node": {"inference": "m.py"}})
        assert result.valid
        assert any("cyberwave-cloud-node" in w for w in result.warnings)

    def test_legacy_key_warning_from_file(self, tmp_path):
        yml = tmp_path / "cyberwave.yml"
        yml.write_text(yaml.dump({"cyberwave-cloud-node": {"inference": "m.py"}}))
        result = validate_manifest(path=yml)
        assert result.valid
        assert any("cyberwave-cloud-node" in w for w in result.warnings)

    def test_no_warning_for_cyberwave_key(self):
        result = validate_manifest(data={"cyberwave": {"inference": "m.py"}})
        assert result.valid
        assert not result.warnings


class TestLenientMode:
    def test_lenient_mode_unknown_field(self):
        result = validate_manifest(
            data={"cyberwave": {"foo": 1, "inference": "m.py"}},
            lenient=True,
        )
        assert result.valid
        assert any("foo" in w for w in result.warnings)
        assert result.manifest is not None
        assert result.manifest.inference == "m.py"

    def test_strict_mode_unknown_field(self):
        result = validate_manifest(data={"cyberwave": {"foo": 1}})
        assert not result.valid
        assert any("foo" in e.field_path or "foo" in e.message for e in result.errors)


class TestErrorCapture:
    def test_yaml_error_captured(self, tmp_path):
        yml = tmp_path / "cyberwave.yml"
        yml.write_text("bad: yaml: [\n")
        result = validate_manifest(path=yml)
        assert not result.valid
        assert result.errors[0].field_path == "yaml"

    def test_missing_file_captured(self, tmp_path):
        result = validate_manifest(path=tmp_path / "nonexistent.yml")
        assert not result.valid
        assert result.errors[0].field_path == "path"

    def test_format_errors_output(self):
        result = validate_manifest(data={"cyberwave": {"inference_timeout": 30}})
        assert not result.valid
        text = result.format_errors()
        assert "Manifest validation failed" in text

    def test_structure_error_for_bad_keys(self):
        result = validate_manifest(data={"services": {"web": {}}, "volumes": {}})
        assert not result.valid
        assert result.errors[0].field_path == "structure"

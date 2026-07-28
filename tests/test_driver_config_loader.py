"""DriverConfigLoader: env > yaml > default precedence per type."""

from __future__ import annotations

from pathlib import Path

from cyberwave.driver.support.config_loader import DriverConfigLoader


def _loader(tmp_path: Path, yaml_text: str | None) -> DriverConfigLoader:
    paths: tuple[Path, ...] = ()
    if yaml_text is not None:
        cfg = tmp_path / "config.yml"
        cfg.write_text(yaml_text)
        paths = (cfg,)
    return DriverConfigLoader(env_prefix="CW_TESTBOT", search_paths=paths)


def test_default_when_no_file_no_env(tmp_path):
    section = _loader(tmp_path, None).section("can")
    assert section.get_str("interface", "can0") == "can0"
    assert section.get_int("bitrate", 500) == 500
    assert section.get_bool("setup", True) is True
    assert section.get_optional_float("velocity") is None
    assert section.get_optional_str("usb_address") is None


def test_yaml_overrides_default(tmp_path):
    section = _loader(
        tmp_path, "can:\n  interface: can5\n  bitrate: 250000\n  setup: false\n"
    ).section("can")
    assert section.get_str("interface", "can0") == "can5"
    assert section.get_int("bitrate", 500) == 250000
    assert section.get_bool("setup", True) is False


def test_env_overrides_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("CW_TESTBOT_INTERFACE", "can9")
    monkeypatch.setenv("CW_TESTBOT_BITRATE", "125000")
    section = _loader(tmp_path, "can:\n  interface: can5\n  bitrate: 250000\n").section("can")
    assert section.get_str("interface", "can0") == "can9"
    assert section.get_int("bitrate", 500) == 125000


def test_blank_env_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("CW_TESTBOT_INTERFACE", "   ")
    section = _loader(tmp_path, "can:\n  interface: can5\n").section("can")
    assert section.get_str("interface", "can0") == "can5"


def test_invalid_env_int_falls_through_to_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("CW_TESTBOT_BITRATE", "not-a-number")
    section = _loader(tmp_path, "can:\n  bitrate: 250000\n").section("can")
    assert section.get_int("bitrate", 500) == 250000


def test_bool_env_truthy_forms(tmp_path, monkeypatch):
    section_yaml = "can:\n  setup: false\n"
    for raw, expected in (("1", True), ("true", True), ("YES", True), ("off", False)):
        monkeypatch.setenv("CW_TESTBOT_SETUP", raw)
        assert _loader(tmp_path, section_yaml).section("can").get_bool("setup", True) is expected


def test_optional_float_from_yaml_and_env(tmp_path, monkeypatch):
    section = _loader(tmp_path, "motion:\n  velocity: 2.5\n").section("motion")
    assert section.get_optional_float("velocity") == 2.5
    monkeypatch.setenv("CW_TESTBOT_VELOCITY", "7.5")
    section = _loader(tmp_path, "motion:\n  velocity: 2.5\n").section("motion")
    assert section.get_optional_float("velocity") == 7.5


def test_custom_env_name(tmp_path, monkeypatch):
    monkeypatch.setenv("CW_SPECIAL_URDF", "/x.urdf")
    section = _loader(tmp_path, None).section("kinematics")
    assert section.get_optional_str("urdf_path", env="CW_SPECIAL_URDF") == "/x.urdf"


def test_path_env_selects_explicit_file(tmp_path, monkeypatch):
    explicit = tmp_path / "elsewhere.yml"
    explicit.write_text("can:\n  interface: canX\n")
    monkeypatch.setenv("CW_TESTBOT_CONFIG", str(explicit))
    loader = DriverConfigLoader(env_prefix="CW_TESTBOT", path_env="CW_TESTBOT_CONFIG")
    assert loader.section("can").get_str("interface", "can0") == "canX"


def test_unreadable_file_falls_back(tmp_path):
    bad = tmp_path / "config.yml"
    bad.write_text("can: [unclosed")
    loader = DriverConfigLoader(env_prefix="CW_TESTBOT", search_paths=(bad,))
    assert loader.section("can").get_str("interface", "can0") == "can0"

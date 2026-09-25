from __future__ import annotations

import textwrap
from pathlib import Path

from cyberwave.driver.ros2.manifest import load_manifest


def test_load_manifest_parses_managed_launch(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        """
        node_name: piper_bridge
        managed_launch:
          package: piper
          launch_file: start_single_piper.launch.py
          launch_args:
            can_port: can0
            auto_enable: false
          readiness:
            kind: service
            name: /enable_srv
            timeout_s: 60
        """
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml_text)
    m = load_manifest(str(path))
    assert m.managed_launch is not None
    assert m.managed_launch.package == "piper"
    assert m.managed_launch.launch_file == "start_single_piper.launch.py"
    assert m.managed_launch.launch_args["can_port"] == "can0"
    assert m.managed_launch.readiness.name == "/enable_srv"
    assert m.managed_launch.readiness.timeout_s == 60.0


def test_load_manifest_without_managed_launch_is_none(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("node_name: foo\n")
    m = load_manifest(str(path))
    assert m.managed_launch is None

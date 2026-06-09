"""cw-driver.yml compile helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from cyberwave.manifest.cw_driver import (
    compile_cw_driver_file,
    load_cw_driver_yml,
    resolve_mqtt_bundle_from_driver_config,
)
from cyberwave.manifest.driver_config import TWIN_COMMAND_TOPIC_SLUG

_REPO_SO101 = (
    Path(__file__).resolve().parents[3]
    / "cyberwave-edge-nodes"
    / "cyberwave-edge-so101"
    / "cw-driver.yml"
)


def test_compile_minimal_cw_driver_yml(tmp_path: Path) -> None:
    yml = tmp_path / "cw-driver.yml"
    yml.write_text(
        """
registry_ids:
  - acme/test-arm
mqtt:
  schema_version: 1
  driver_family: python
  twin:
    command:
      direction: both
      payload_schema_ref: TwinCommandPayload
      description: Command ingress.
  commands:
    supported:
      - stop
      - name: nudge
        continuous: true
        rate_hz: 10
""",
        encoding="utf-8",
    )
    bundle = compile_cw_driver_file(yml)
    assert bundle["driver_family"] == "python"
    assert TWIN_COMMAND_TOPIC_SLUG in bundle["topics"]
    assert bundle["commands"]["supported"] == ["stop", "nudge"]
    assert bundle["commands"]["specs"]["nudge"]["continuous"] is True
    assert bundle["asset_registry_id"] == "acme/test-arm"


@pytest.mark.skipif(not _REPO_SO101.is_file(), reason="monorepo edge-nodes not present")
def test_compile_so101_reference_manifest() -> None:
    raw = load_cw_driver_yml(_REPO_SO101)
    bundle = resolve_mqtt_bundle_from_driver_config(raw)
    assert "remoteoperate" in bundle["commands"]["supported"]
    assert "cyberwave/joint/{twin_uuid}/update" in bundle["topics"]
    assert "cyberwave/twin/{twin_uuid}/webrtc-offer" not in bundle["topics"]

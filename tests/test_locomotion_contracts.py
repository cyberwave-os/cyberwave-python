import json
import math
from pathlib import Path

import pytest

import cyberwave
from cyberwave.locomotion_contracts import (
    LOCOMOTION_VELOCITY_COMMAND_CONTRACT,
    LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS,
    build_locomotion_velocity_command,
    stop_locomotion_velocity_command,
)


def test_locomotion_contract_helpers_are_exported_from_package() -> None:
    assert (
        cyberwave.LOCOMOTION_VELOCITY_COMMAND_CONTRACT
        == LOCOMOTION_VELOCITY_COMMAND_CONTRACT
    )
    assert (
        cyberwave.LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS
        == LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS
    )
    assert (
        cyberwave.stop_locomotion_velocity_command()
        == stop_locomotion_velocity_command()
    )


def test_build_locomotion_velocity_command_matches_repo_schema_required_fields() -> None:
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "contracts"
        / "locomotion.velocity_command.v1.schema.json"
    )
    if not schema_path.exists():
        pytest.skip("Repository contract schema is not available in this test context")
    schema = json.loads(schema_path.read_text())
    payload = build_locomotion_velocity_command(
        linear_x=0.2,
        angular_z=0.1,
        duration_ms=500,
        gait="walk",
        origin="teleop",
    ).to_payload()

    assert schema["properties"]["contract"]["const"] == (
        LOCOMOTION_VELOCITY_COMMAND_CONTRACT
    )
    assert tuple(schema["required"]) == LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS
    assert schema["x-cyberwave-adapter-capabilities"]["stop"] == (
        stop_locomotion_velocity_command()
    )
    assert payload["contract"] == LOCOMOTION_VELOCITY_COMMAND_CONTRACT
    assert set(LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS).issubset(payload)


def test_build_locomotion_velocity_command_allows_schema_stop_duration() -> None:
    command = build_locomotion_velocity_command(
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0,
        duration_ms=0,
        gait="stand",
        origin="workflow",
    )

    assert command.to_payload() == {
        "linear_x": 0.0,
        "linear_y": 0.0,
        "angular_z": 0.0,
        "duration_ms": 0,
        "gait": "stand",
        "origin": "workflow",
        "contract": LOCOMOTION_VELOCITY_COMMAND_CONTRACT,
    }


def test_build_locomotion_velocity_command_rejects_short_active_duration() -> None:
    with pytest.raises(ValueError, match="0 or at least 50"):
        build_locomotion_velocity_command(duration_ms=1)


def test_build_locomotion_velocity_command_rejects_fractional_duration() -> None:
    with pytest.raises(ValueError, match="integer duration_ms"):
        build_locomotion_velocity_command(duration_ms=500.5)


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    (
        ("linear_x", {"linear_x": math.nan}),
        ("angular_z", {"angular_z": math.inf}),
        ("duration_ms", {"duration_ms": math.inf}),
    ),
)
def test_build_locomotion_velocity_command_rejects_non_finite_numbers(
    field_name: str,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match=f"finite numeric {field_name}"):
        build_locomotion_velocity_command(**kwargs)


def test_stop_locomotion_velocity_command_uses_canonical_contract() -> None:
    payload = stop_locomotion_velocity_command({"origin": "workflow"})

    assert payload["contract"] == LOCOMOTION_VELOCITY_COMMAND_CONTRACT
    assert payload["duration_ms"] == 0
    assert payload["gait"] == "stand"
    assert payload["origin"] == "workflow"

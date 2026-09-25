"""B3: a topic registered as both listener and publisher must merge source_types,
not clobber them, in the exported cw-driver manifest."""

from __future__ import annotations

from cyberwave.driver import (
    CallbackGroup,
    DriverInterfaceRegistry,
    ProtocolArgs,
    TopicSpec,
)


def _joint_update_spec() -> TopicSpec:
    return TopicSpec(
        namespace="joint",
        leaf="update",
        payload_schema_ref="JointStatesPayload",
    )


def test_listener_then_publisher_source_types_are_unioned():
    reg = DriverInterfaceRegistry()
    reg.add_listener(
        _joint_update_spec(),
        CallbackGroup(callback=lambda e: None),
        protocol=ProtocolArgs(source_types=["tele", "edit", "sim_tele"]),
    )
    reg.add_publisher(
        _joint_update_spec(),
        CallbackGroup(),
        protocol=ProtocolArgs(source_types=["edge"]),
    )

    manifest = reg.to_cw_driver_dict(registry_id="acme/arm")
    entry = manifest["mqtt"]["joint"]["update"]

    assert entry["direction"] == "both"
    assert set(entry["source_types"]) == {"tele", "edit", "sim_tele", "edge"}
    # Deterministic order: listener (registered first) values precede publisher's.
    assert entry["source_types"] == ["tele", "edit", "sim_tele", "edge"]

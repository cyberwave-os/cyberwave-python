"""Locomotion / flight / catalog movement commands are PLAYGROUND-compatible.

They previously raised ``NotSimulatedError`` in any simulation mode (live/driver
-only). The browser playground now renders these commands directly (see the
frontend's ``usePlaygroundLocomotionCommandFollower``), so they no longer
require a running MuJoCo-style simulation instance — ``SimLevel.PLAYGROUND``
is a no-op preflight check, same as an undecorated method. They still work
normally in live mode.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.exceptions import NotSimulatedError
from cyberwave.twin.base import Twin
from cyberwave.twin.capabilities.flight import FlightHandle
from cyberwave.twin.capabilities.locomotion import LocomotionHandle
from cyberwave.twin.command_factory import _make_catalog_command_method
from cyberwave.twin.commands import TwinCommandsHandle
from cyberwave.twin.simulation_support import SimLevel


def _twin(runtime_mode):
    twin = SimpleNamespace(
        client=SimpleNamespace(config=SimpleNamespace(runtime_mode=runtime_mode)),
        publish_command=lambda *a, **k: None,
        publish_command_burst=lambda *a, **k: None,
        _resolve_topic_and_payload=lambda **k: SimpleNamespace(
            topic="t", payload={}, source_type=k.get("source_type")
        ),
        _publish_resolved=lambda resolved: None,
    )
    twin._ensure_simulation_support = Twin._ensure_simulation_support.__get__(twin)
    return twin


def test_locomotion_methods_are_playground_level() -> None:
    for name in ("move_forward", "move_backward", "turn_left", "turn_right", "stop", "move"):
        method = getattr(LocomotionHandle, name)
        assert method.__cw_sim_level__ == SimLevel.PLAYGROUND, name


def test_flight_send_is_playground_level() -> None:
    assert FlightHandle._send.__cw_sim_level__ == SimLevel.PLAYGROUND


def test_flying_twin_direct_send_is_playground_level() -> None:
    # The twin.takeoff/land/hover/... shortcuts funnel through _send_drone_command,
    # so this chokepoint gates the direct-method surface too.
    from cyberwave.twin.classes import FlyingTwin

    assert FlyingTwin._send_drone_command.__cw_sim_level__ == SimLevel.PLAYGROUND


def test_locomotion_move_forward_ok_in_sim_mode() -> None:
    LocomotionHandle(_twin("simulation")).move_forward(0.3)  # publishes, no raise


def test_locomotion_move_forward_ok_in_live_mode() -> None:
    LocomotionHandle(_twin("live")).move_forward(0.3)  # publishes, no raise


def test_flight_takeoff_ok_in_sim_mode() -> None:
    FlightHandle(_twin("simulation")).takeoff()  # publishes, no raise


def test_catalog_move_forward_ok_in_sim_mode() -> None:
    # twin.commands.move_forward() — the catalog-dispatch path, which resolves
    # its preflight level from the delegate it routes to (locomotion here),
    # not a blanket UNSUPPORTED. `resolve_command_delegate` looks up the
    # `locomotion` property on the twin's CLASS (`hasattr(type(twin), prop)`),
    # so a plain instance attribute (as on SimpleNamespace) wouldn't be found
    # — mirror the real `LocomotionCapableMixin.locomotion` class property.
    class _LocomotionTwin:
        def __init__(self, client, driver):
            self.client = client
            self.driver = driver
            self._ensure_simulation_support = Twin._ensure_simulation_support.__get__(self)
            self.publish_command = lambda *a, **k: None
            self.publish_command_burst = lambda *a, **k: None

        @property
        def locomotion(self):
            return LocomotionHandle(self)

    twin = _LocomotionTwin(
        client=SimpleNamespace(config=SimpleNamespace(runtime_mode="simulation")),
        driver=SimpleNamespace(
            get_mqtt_schema=lambda: {
                "commands": {
                    "supported": ["move_forward"],
                    "specs": {"move_forward": {"continuous": True}},
                }
            }
        ),
    )
    handle = TwinCommandsHandle.__new__(TwinCommandsHandle)
    handle._twin = twin
    handle._bound_catalog_commands = []
    handle._command_routing = {}
    method = _make_catalog_command_method("move_forward")
    method(handle)  # publishes via the continuous burst branch, no raise


def test_catalog_flight_takeoff_ok_in_sim_mode() -> None:
    # twin.commands.takeoff() -- the catalog-dispatch path for a *flight* command.
    # Regression test: FlightHandle's public shorthands (takeoff/land/hover/
    # ascend/descend/gimbal_rotate) publish through the internal `_send`, but
    # `resolve_command_delegate` returns the shorthand itself as the delegate, so
    # the catalog dispatch's preflight check inspects the shorthand's own
    # __cw_sim_level__ -- it must carry @simulation_level directly, not rely on
    # `_send`'s decorator alone, or this raises NotSimulatedError before `_send`
    # (and its correctly-configured decorator) is ever reached.
    class _FlightTwin:
        def __init__(self, client, driver):
            self.client = client
            self.driver = driver
            self._ensure_simulation_support = Twin._ensure_simulation_support.__get__(self)
            self.publish_command = lambda *a, **k: None
            self.publish_command_burst = lambda *a, **k: None
            self._resolve_topic_and_payload = lambda **k: SimpleNamespace(
                topic="t", payload={}, source_type=k.get("source_type")
            )
            self._publish_resolved = lambda resolved: None

        @property
        def flight(self):
            return FlightHandle(self)

    twin = _FlightTwin(
        client=SimpleNamespace(config=SimpleNamespace(runtime_mode="simulation")),
        driver=SimpleNamespace(
            get_mqtt_schema=lambda: {
                "commands": {"supported": ["takeoff"], "specs": {"takeoff": {}}},
            }
        ),
    )
    handle = TwinCommandsHandle.__new__(TwinCommandsHandle)
    handle._twin = twin
    handle._bound_catalog_commands = []
    handle._command_routing = {}
    method = _make_catalog_command_method("takeoff")
    method(handle)  # publishes via the flight delegate branch, no raise


def test_catalog_command_allowed_via_controller_policy_playground_binding() -> None:
    # A raw MQTT passthrough command with no capability delegate (same shape as
    # test_catalog_unknown_command_still_unsupported_in_sim_mode's led_toggle)
    # becomes PLAYGROUND-compatible once the twin's *attached controller policy*
    # exposes it with a `playground` keyboard binding -- the same data
    # PlaygroundLocomotionCommandDrivers reads on the frontend. This lets a
    # specific asset's controller opt a custom command into playground preview
    # without the SDK hardcoding it into a capability handle.
    class _FakePolicyHandle:
        def playground_actuations(self):
            return frozenset({"led_toggle"})

    class _TwinWithPolicy:
        def __init__(self, client, driver):
            self.client = client
            self.driver = driver
            self._ensure_simulation_support = Twin._ensure_simulation_support.__get__(self)
            self.publish_command = lambda *a, **k: None
            self._prepare_outbound_command = lambda: None

        @property
        def policy(self):
            return _FakePolicyHandle()

    twin = _TwinWithPolicy(
        client=SimpleNamespace(config=SimpleNamespace(runtime_mode="simulation")),
        driver=SimpleNamespace(
            get_mqtt_schema=lambda: {
                "commands": {"supported": ["led_toggle"], "specs": {"led_toggle": {}}},
            }
        ),
    )
    handle = TwinCommandsHandle.__new__(TwinCommandsHandle)
    handle._twin = twin
    handle._bound_catalog_commands = []
    handle._command_routing = {}
    method = _make_catalog_command_method("led_toggle")
    method(handle)  # publishes via the raw-publish branch, no raise


def test_catalog_dispatch_in_live_mode_never_touches_controller_policy() -> None:
    # Regression: `twin._ensure_simulation_support(_catalog_command_sim_level(...))`
    # evaluates its argument eagerly, so `_catalog_command_sim_level` -- which can
    # now trigger a network fetch of the attached controller policy -- used to run
    # on *every* commands.<name>() call regardless of runtime mode, even though
    # `_ensure_simulation_support` is a no-op in live mode. That added a blocking
    # HTTP round trip to every live-mode command dispatch; from an async caller
    # (e.g. a script under `asyncio.run(...)`) it stalled the event loop long
    # enough to starve concurrent MQTT/WebRTC coroutines, breaking live command
    # delivery entirely. `commands.<name>()` in live mode must never touch
    # `twin.policy` at all.
    class _ExplodingPolicyHandle:
        def playground_actuations(self):
            raise AssertionError("must not be called in live runtime mode")

    class _TwinWithPolicy:
        def __init__(self, client, driver):
            self.client = client
            self.driver = driver
            self._ensure_simulation_support = Twin._ensure_simulation_support.__get__(self)
            self.publish_command_burst = lambda *a, **k: None
            self._prepare_outbound_command = lambda: None

        @property
        def policy(self):
            return _ExplodingPolicyHandle()

    twin = _TwinWithPolicy(
        client=SimpleNamespace(config=SimpleNamespace(runtime_mode="live")),
        driver=SimpleNamespace(
            get_mqtt_schema=lambda: {
                "commands": {
                    "supported": ["move_forward"],
                    "specs": {"move_forward": {"continuous": True}},
                }
            }
        ),
    )
    handle = TwinCommandsHandle.__new__(TwinCommandsHandle)
    handle._twin = twin
    handle._bound_catalog_commands = []
    handle._command_routing = {}
    method = _make_catalog_command_method("move_forward")
    method(handle)  # publishes via the burst branch; must not raise or block


def test_catalog_unknown_command_still_unsupported_in_sim_mode() -> None:
    # A catalog command with no capability delegate (e.g. a raw mqtt_publish
    # passthrough) is unaffected by this change — still blocked in sim mode.
    twin = _twin("simulation")
    twin.driver = SimpleNamespace(
        get_mqtt_schema=lambda: {
            "commands": {
                "supported": ["led_toggle"],
                "specs": {"led_toggle": {}},
            }
        }
    )
    handle = TwinCommandsHandle.__new__(TwinCommandsHandle)
    handle._twin = twin
    handle._bound_catalog_commands = []
    handle._command_routing = {}
    method = _make_catalog_command_method("led_toggle")
    with pytest.raises(NotSimulatedError):
        method(handle)

"""Bind MQTT catalog commands as methods on :class:`~cyberwave.twin.commands.TwinCommandsHandle`."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any, Callable

from ..manifest.driver_config import command_spec, supported_mqtt_commands
from ._helpers import motion_outbound_requires_policy
from .transport import DEFAULT_BURST_DURATION_S, DEFAULT_BURST_RATE_HZ

if TYPE_CHECKING:
    from .base import Twin
    from .commands import TwinCommandsHandle
    from .simulation_support import SimLevel

_CAPABILITY_PROPERTIES: tuple[str, ...] = (
    "locomotion",
    "flight",
    "gripper",
    "joints",
)

_LOCOMOTION_DIRECTIONAL = frozenset(
    {"move_forward", "move_backward", "turn_left", "turn_right"}
)

_RESERVED_COMMAND_HANDLE_ATTRS = frozenset(
    {
        "_twin",
        "_bound_catalog_commands",
        "_command_routing",
        "get_schema",
        "get_supported_commands",
    }
)


def catalog_command_names(schema: dict[str, Any]) -> list[str]:
    """Return command names from a compiled MQTT catalog schema."""
    return supported_mqtt_commands(schema)


def resolve_command_delegate(
    twin: Twin,
    command: str,
) -> tuple[str, Callable[..., Any]] | None:
    """Return ``(capability_property, handle_method)`` when the twin implements *command*."""
    if command.startswith("_") or not command.isidentifier():
        return None
    twin_type = type(twin)
    for prop in _CAPABILITY_PROPERTIES:
        if not hasattr(twin_type, prop):
            continue
        handle = getattr(twin, prop)
        fn = getattr(handle, command, None)
        if callable(fn):
            return (prop, fn)
    return None


def _merge_catalog_payload(
    data: dict[str, Any] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = dict(data or {})
    if kwargs:
        payload.update(kwargs)
    return payload


def _command_arg_names(spec: dict[str, Any]) -> list[str]:
    raw = spec.get("args")
    if not isinstance(raw, list):
        return []
    return [a["name"] for a in raw if isinstance(a, dict) and a.get("name")]


def _command_arg_defaults(spec: dict[str, Any]) -> dict[str, Any]:
    raw = spec.get("args")
    if not isinstance(raw, list):
        return {}
    return {
        a["name"]: a["default"]
        for a in raw
        if isinstance(a, dict) and a.get("name") and "default" in a
    }


def _build_arg_payload(
    spec: dict[str, Any],
    data: dict[str, Any] | Any | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Map a positional primary arg + kwargs onto declared args (relaxed)."""
    arg_names = _command_arg_names(spec)
    if arg_names and not isinstance(data, dict):
        payload = _command_arg_defaults(spec)
        if data is not None:
            payload[arg_names[0]] = data
        payload.update(kwargs)
        return payload
    payload = _merge_catalog_payload(data, **kwargs)
    for name, default in _command_arg_defaults(spec).items():
        payload.setdefault(name, default)
    return payload


def _invoke_locomotion_delegate(
    delegate: Callable[..., Any],
    *,
    payload: dict[str, Any],
    source_type: str | None,
) -> None:
    """Map catalog ``data`` / kwargs onto :class:`~cyberwave.twin.capabilities.locomotion.LocomotionHandle`."""
    name = delegate.__name__
    params = dict(payload)

    if name == "stop":
        delegate(source_type=source_type)
        return

    if name in _LOCOMOTION_DIRECTIONAL:
        if name in {"move_forward", "move_backward"}:
            speed = float(params.pop("distance", params.pop("linear_x", 0.3)))
        else:
            speed = float(params.pop("angle", params.pop("angular_z", 0.5)))
        duration = float(params.pop("duration", DEFAULT_BURST_DURATION_S))
        rate_hz = float(params.pop("rate_hz", DEFAULT_BURST_RATE_HZ))
        if params:
            raise TypeError(
                f"Unexpected keyword(s) for locomotion.{name}(): {sorted(params)}"
            )
        delegate(
            speed,
            duration=duration,
            rate_hz=rate_hz,
            source_type=source_type,
        )
        return

    if name == "move":
        delegate(
            distance=params.pop("distance", None),
            angle=params.pop("angle", None),
            linear_x=params.pop("linear_x", None),
            angular_z=params.pop("angular_z", None),
            source_type=source_type,
            command=str(params.pop("command", "move")),
            duration=float(params.pop("duration", 0.0)),
            rate_hz=float(params.pop("rate_hz", DEFAULT_BURST_RATE_HZ)),
            **params,
        )
        return

    if params:
        delegate(**params, source_type=source_type)
    else:
        delegate(source_type=source_type)


def _invoke_capability_delegate(
    delegate: Callable[..., Any],
    *,
    payload: dict[str, Any],
    source_type: str | None,
) -> None:
    """Forward merged catalog payload to a flight/gripper/joints handle method."""
    if payload:
        delegate(**payload, source_type=source_type)
    else:
        delegate(source_type=source_type)


def _burst_timing(
    payload: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    params = dict(payload)
    duration_s = float(
        params.pop("duration", spec.get("default_duration_s", DEFAULT_BURST_DURATION_S))
    )
    rate_hz = float(params.pop("rate_hz", spec.get("rate_hz", DEFAULT_BURST_RATE_HZ)))
    return duration_s, rate_hz, params


def command_routing_entry(
    *,
    via: str,
    continuous: bool = False,
) -> dict[str, Any]:
    return {"via": via, "continuous": continuous}


def _catalog_command_sim_level(twin: Twin, command: str) -> "SimLevel":
    """Resolve the preflight simulation level for a catalog-dispatched command.

    Two independent signals can grant ``PLAYGROUND``:

    1. **Capability delegate.** Catalog commands (``twin.commands.<name>()``)
       route to whatever capability handle declares a same-named method
       (locomotion, flight, gripper, joints); that handle's own
       ``@simulation_level`` annotation is authoritative, so e.g. a locomotion
       command inherits ``LocomotionHandle.move_forward``'s ``PLAYGROUND`` level.
    2. **Controller policy binding.** The twin's attached controller policy may
       expose *this* command with a ``playground`` keyboard binding (see
       ``seed_controllers.py``'s ``generate_locomotion_controller_from_asset`` /
       ``_drone_runtime_metadata``) — the same data the frontend's
       ``PlaygroundLocomotionCommandDrivers`` reads to render it. That makes the
       command playground-safe for this specific asset even if it has no
       capability delegate at all (a raw MQTT passthrough command that only the
       controller policy, not the SDK, knows is drivable in the playground).

    A command with neither signal stays ``UNSUPPORTED`` in simulation mode,
    unchanged from prior behavior.
    """
    from .simulation_support import SimLevel

    policy_handle = getattr(twin, "policy", None)
    if policy_handle is not None and command in policy_handle.playground_actuations():
        return SimLevel.PLAYGROUND

    delegate_info = resolve_command_delegate(twin, command)
    if delegate_info is None:
        return SimLevel.UNSUPPORTED
    _, delegate_fn = delegate_info
    return getattr(delegate_fn, "__cw_sim_level__", SimLevel.UNSUPPORTED)


def _make_catalog_command_method(command: str) -> Callable[..., None]:
    """Build a bound catalog method (continuous burst, delegate, or single publish)."""

    def method(
        self: TwinCommandsHandle,
        data: dict[str, Any] | None = None,
        *,
        source_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        twin = self._twin

        # Resolve the preflight level only in simulation mode: unlike the
        # decorator-only path this once was, `_catalog_command_sim_level` can now
        # make a network call (attached controller policy lookup), and this
        # argument is evaluated eagerly regardless of what
        # `_ensure_simulation_support` does with it. Computing it unconditionally
        # would add a blocking HTTP round trip to every live-mode command
        # dispatch — including from async callers, where it can stall the event
        # loop and starve concurrent MQTT/WebRTC coroutines.
        from .runtime_state import RUNTIME_MODE_SIMULATION, active_runtime_mode

        if active_runtime_mode(twin.client) == RUNTIME_MODE_SIMULATION:
            twin._ensure_simulation_support(
                _catalog_command_sim_level(twin, command), method=f"commands.{command}"
            )
        schema = twin.driver.get_mqtt_schema()
        spec = command_spec(schema, command)
        payload = _build_arg_payload(spec, data, kwargs)

        if spec.get("continuous"):
            duration_s, rate_hz, burst_data = _burst_timing(payload, spec)
            twin.publish_command_burst(
                command,
                burst_data,
                duration_s=duration_s,
                rate_hz=rate_hz,
                source_type=source_type,
            )
            return

        delegate_info = resolve_command_delegate(twin, command)
        if delegate_info is not None:
            capability, delegate_fn = delegate_info
            if capability == "locomotion":
                _invoke_locomotion_delegate(
                    delegate_fn,
                    payload=payload,
                    source_type=source_type,
                )
            else:
                _invoke_capability_delegate(
                    delegate_fn,
                    payload=payload,
                    source_type=source_type,
                )
            return

        if motion_outbound_requires_policy(command):
            twin._prepare_outbound_command()
        twin.publish_command(command, payload, source_type=source_type)

    method.__name__ = command
    method.__qualname__ = f"TwinCommandsHandle.{command}"
    method.__doc__ = f"Publish MQTT catalog command {command!r}."
    return method


def bind_catalog_commands(handle: TwinCommandsHandle) -> list[str]:
    """Attach one callable per ``commands.supported`` entry on *handle*.

    Continuous commands (``commands.specs[name].continuous``) use
    :meth:`~cyberwave.twin.transport.TwinTransportMixin.publish_command_burst`.
    Otherwise delegates to capability handles when present, else a single publish.

    Returns the list of command names that were bound (for ``describe()`` / ``dir()``).
    """
    schema = handle._twin.driver.get_mqtt_schema()
    bound: list[str] = []
    routing: dict[str, dict[str, Any]] = {}
    twin = handle._twin
    for command in catalog_command_names(schema):
        if command in _RESERVED_COMMAND_HANDLE_ATTRS:
            continue
        if not command.isidentifier():
            continue
        if hasattr(handle, command):
            continue

        spec = command_spec(schema, command)
        fn = _make_catalog_command_method(command)

        if spec.get("continuous"):
            routing[command] = command_routing_entry(via="burst", continuous=True)
        else:
            delegate_info = resolve_command_delegate(twin, command)
            if delegate_info is not None:
                capability, _ = delegate_info
                routing[command] = command_routing_entry(
                    via=f"{capability}.{command}",
                    continuous=False,
                )
            else:
                routing[command] = command_routing_entry(
                    via="mqtt_publish",
                    continuous=False,
                )

        setattr(handle, command, MethodType(fn, handle))
        bound.append(command)

    setattr(handle, "_bound_catalog_commands", bound)
    setattr(handle, "_command_routing", routing)
    return bound


def unbind_catalog_commands(handle: TwinCommandsHandle) -> None:
    """Remove dynamically bound catalog command methods from *handle*."""
    for command in list(getattr(handle, "_bound_catalog_commands", ())):
        if command in _RESERVED_COMMAND_HANDLE_ATTRS:
            continue
        if hasattr(handle, command):
            delattr(handle, command)
    handle._bound_catalog_commands = []
    handle._command_routing = {}


def rebind_catalog_commands(handle: TwinCommandsHandle) -> list[str]:
    """Drop prior catalog bindings, clear the twin schema cache, and bind again."""
    unbind_catalog_commands(handle)
    twin = handle._twin
    if hasattr(twin, "_mqtt_catalog_cache"):
        twin._mqtt_catalog_cache = None
    if hasattr(twin, "_driver_catalog_cache"):
        twin._driver_catalog_cache = None
    return bind_catalog_commands(handle)

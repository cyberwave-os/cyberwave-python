"""env_params.py — Environment variable handling for Cyberwave ROS 2 Python drivers.

All Cyberwave ROS 2 overrides use the CW_ROS2_<PARAM_NAME_UPPER> convention,
so the same env file works regardless of driver language.
"""

from __future__ import annotations

import os
from typing import Any

from .manifest import ManifestManagedLaunch, NodeManifest

_PREFIX = "CW_ROS2_"
_SYSTEM_SUFFIXES = frozenset({"NODE_NAME", "NAMESPACE", "DOMAIN_ID", "AUTO_ACTIVATE"})


def node_name_from_env(default: str) -> str:
    """Return the node name, overridden by CW_ROS2_NODE_NAME if set."""
    return os.environ.get("CW_ROS2_NODE_NAME", default)


def node_namespace_from_env() -> str:
    """Return the node namespace.

    Resolution order:
    1. CW_ROS2_NAMESPACE if set (including to ""); use as-is.
    2. If unset: CYBERWAVE_TWIN_UUID → /twin_<uuid>, dashes→underscores,
       lowercase.
    3. If no twin: empty string (global ROS 2 namespace).

    ``twin_`` is the prefix every Cyberwave ROS image already uses: the go2,
    kinova, nav2, rtabmap-slam, moveit, elevation-mapping and object-pose
    entrypoints all export ``ROS_NAMESPACE=/twin_<uuid>``, the sim proxy is
    started with ``-r __ns:=/twin_<uuid>``, and edge-core substitutes the same
    string into a declared Compose graph as ``${CW_TWIN_NS}``. A prefix is needed
    at all only because a ROS 2 name segment may not start with a digit and a
    UUID usually does.

    It used to be ``/CW_<UUID>``. That was not a second convention so much as a
    default nothing exercised: kinova and moveit both export
    ``CW_ROS2_NAMESPACE`` from ``ROS_NAMESPACE`` specifically so their
    ``Ros2DriverBase`` nodes land with the rest of the graph, and the
    intelligence services are launched from a launch file with an explicit
    ``namespace=``. The one driver with neither -- the Piper -- was therefore the
    only one that ever ran at ``/CW_<UUID>``, which put it one namespace away
    from any sibling in its own graph. ``move_group`` beside it would have waited
    on a ``robot_description`` that never arrived, and nothing would have logged
    an error.

    Kept in step with the other copy of this function, ``env_params.cpp``, and
    edge-core's ``CW_TWIN_NS``; see
    ``drivers/tests/base/test_ros2_namespace_prefix.py``, which holds all four to
    one answer.
    """
    if "CW_ROS2_NAMESPACE" in os.environ:
        return os.environ["CW_ROS2_NAMESPACE"]
    twin_uuid = os.environ.get("CYBERWAVE_TWIN_UUID", "")
    if twin_uuid:
        return "/twin_" + twin_uuid.replace("-", "_").lower()
    return ""


def collect_env_param_overrides(manifest: NodeManifest) -> dict[str, Any]:
    """Return {param_name: typed_value} for all CW_ROS2_<NAME> env vars that
    match a declared parameter (by case-insensitive suffix comparison).

    System keys (NODE_NAME, NAMESPACE, DOMAIN_ID, AUTO_ACTIVATE) are excluded.
    Values are converted to the type declared in the manifest.
    """
    # Build lookup: lower-case param name → type string
    type_map: dict[str, str] = {}
    for p in manifest.params:
        type_map[p.name.lower()] = p.type
    for t in manifest.topics:
        type_map[t.name.lower()] = "string"
    for s in manifest.services:
        type_map[s.name.lower()] = "string"

    overrides: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(_PREFIX):
            continue
        suffix = key[len(_PREFIX):]
        if suffix in _SYSTEM_SUFFIXES:
            continue
        param_name = suffix.lower()
        if param_name not in type_map:
            continue
        try:
            overrides[param_name] = _convert(type_map[param_name], raw)
        except (ValueError, TypeError):
            pass  # leave the manifest default; rclpy will log a warning
    return overrides


def _coerce_like(example: object, raw: str) -> Any:
    """Coerce *raw* to the same kind as *example* (launch-arg env overrides)."""
    if isinstance(example, bool):
        return raw.lower() in ("true", "1", "yes")
    if isinstance(example, int) and not isinstance(example, bool):
        return int(raw)
    if isinstance(example, float):
        return float(raw)
    return raw


def resolve_managed_launch_args(
    manifest: NodeManifest,
    managed: ManifestManagedLaunch,
) -> dict[str, str | int | float | bool]:
    """Merge ``CW_ROS2_*`` / ``CW_MANAGED_LAUNCH_*`` env into manifest launch args."""
    args: dict[str, str | int | float | bool] = dict(managed.launch_args)
    key_by_lower = {str(k).lower(): k for k in args}

    for pname, value in collect_env_param_overrides(manifest).items():
        launch_key = key_by_lower.get(pname.lower())
        if launch_key is not None:
            args[launch_key] = value

    prefix = "CW_MANAGED_LAUNCH_"
    for env_key, raw in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        arg_name = env_key[len(prefix) :].lower()
        launch_key = key_by_lower.get(arg_name)
        if launch_key is None:
            continue
        args[launch_key] = _coerce_like(args[launch_key], raw)

    return args


def _convert(type_str: str, raw: str) -> Any:
    if type_str == "bool":
        return raw.lower() in ("true", "1", "yes")
    if type_str == "int":
        return int(raw)
    if type_str == "double":
        return float(raw)
    return raw  # string / fallback

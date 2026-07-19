"""Read driver/MQTT interface catalogs from platform metadata (API payloads).

Structured driver config is compiled server-side from ``cw-driver.yml`` and stored
on catalog assets. Twins receive it via asset metadata sync. This module only
**reads** those payloads — it does not load or compile on-disk ``cw-driver.yml``
files.

Canonical lookup order (first match wins):

1. ``metadata["mqtt"]`` — original seed shape
2. ``metadata["driver"]["config"]`` — target twin/asset shape (bundle or ``{"mqtt": ...}``)
3. ``metadata["driver_config"]["mqtt"]`` — legacy nested shape
"""

from __future__ import annotations

import copy
from typing import Any, Literal

MQTT_BUNDLE_SCHEMA_VERSION = 1
JOINT_UPDATE_TOPIC_SLUG = "cyberwave/joint/{twin_uuid}/update"
JOINT_SUBSCRIBE_TOPIC_SLUG = "cyberwave/joint/{twin_uuid}/+"
TWIN_COMMAND_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/command"
TWIN_POSITION_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/position"
TWIN_ROTATION_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/rotation"
TWIN_KINEMATICS_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/state/kinematics"
TWIN_TELEMETRY_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/telemetry"
TWIN_CAMERA_PHOTO_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/camera/photo"
TWIN_IMU_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/imu"
TWIN_GPS_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/gps"
TWIN_DEPTH_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/depth"
TWIN_POINTCLOUD_TOPIC_SLUG = "cyberwave/twin/{twin_uuid}/pointcloud"

InboundStream = Literal[
    "pose", "joints", "power", "camera", "imu", "gps", "depth", "pointcloud"
]

INBOUND_STREAM_SLUGS: dict[InboundStream, tuple[str, ...]] = {
    "pose": (
        TWIN_KINEMATICS_TOPIC_SLUG,
        TWIN_POSITION_TOPIC_SLUG,
        TWIN_ROTATION_TOPIC_SLUG,
    ),
    "joints": (JOINT_UPDATE_TOPIC_SLUG,),
    "power": (),  # resolved by battery slug pattern below
    "camera": (TWIN_CAMERA_PHOTO_TOPIC_SLUG,),
    "imu": (TWIN_IMU_TOPIC_SLUG,),
    "gps": (TWIN_GPS_TOPIC_SLUG,),
    "depth": (TWIN_DEPTH_TOPIC_SLUG,),
    "pointcloud": (TWIN_POINTCLOUD_TOPIC_SLUG,),
}

_LISTEN_FILTER_NAMES = frozenset(
    {"pose", "joints", "power", "camera", "imu", "depth", "pointcloud"}
)

_LOCOMOTION_COMMANDS = frozenset(
    {
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "stop",
    }
)

OutboundTopicChannel = Literal["twin_command", "joint_update", "sensor_actuation"]

_CANONICAL_SLUG_BY_CHANNEL: dict[OutboundTopicChannel, str] = {
    "twin_command": TWIN_COMMAND_TOPIC_SLUG,
    "joint_update": JOINT_UPDATE_TOPIC_SLUG,
    "sensor_actuation": TWIN_COMMAND_TOPIC_SLUG,
}


def _is_mqtt_bundle(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    topics = value.get("topics")
    return isinstance(topics, dict) and bool(topics)


def _coerce_mqtt_bundle(candidate: Any) -> dict[str, Any] | None:
    if _is_mqtt_bundle(candidate):
        return copy.deepcopy(candidate)
    if isinstance(candidate, dict):
        nested = candidate.get("mqtt")
        if _is_mqtt_bundle(nested):
            return copy.deepcopy(nested)
    return None


def extract_mqtt_bundle_from_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the compiled MQTT interface catalog from asset or twin metadata."""
    if not isinstance(metadata, dict):
        return None

    direct = _coerce_mqtt_bundle(metadata.get("mqtt"))
    if direct is not None:
        return direct

    driver = metadata.get("driver")
    if isinstance(driver, dict):
        from_driver = _coerce_mqtt_bundle(driver.get("config"))
        if from_driver is not None:
            return from_driver

    legacy = metadata.get("driver_config")
    if isinstance(legacy, dict):
        from_legacy = _coerce_mqtt_bundle(legacy)
        if from_legacy is not None:
            return from_legacy

    return None


def extract_mqtt_from_asset_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Alias for :func:`extract_mqtt_bundle_from_metadata` (asset/twin metadata dict)."""
    return extract_mqtt_bundle_from_metadata(metadata)


def _is_zenoh_bundle(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    channels = value.get("channels")
    return isinstance(channels, dict) and bool(channels)


def extract_zenoh_bundle_from_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the compiled Zenoh interface catalog from asset or twin metadata."""
    if not isinstance(metadata, dict):
        return None
    candidate = metadata.get("zenoh")
    if _is_zenoh_bundle(candidate):
        return copy.deepcopy(candidate)
    return None


def extract_driver_config_from_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the structured ``driver.config`` object when present.

    Falls back to the MQTT bundle under ``metadata["mqtt"]`` when ``driver.config``
    is not set (original seed layout).
    """
    if not isinstance(metadata, dict):
        return None

    driver = metadata.get("driver")
    if isinstance(driver, dict):
        config = driver.get("config")
        if isinstance(config, dict):
            return copy.deepcopy(config)

    bundle = extract_mqtt_bundle_from_metadata(metadata)
    if bundle is not None:
        return copy.deepcopy(bundle)
    return None


def _command_name_from_supported_entry(entry: Any) -> str | None:
    if isinstance(entry, str):
        name = entry.strip()
        return name or None
    if isinstance(entry, dict):
        raw_name = entry.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()
    return None


def supported_mqtt_commands(bundle: dict[str, Any] | None) -> list[str]:
    """Return sorted unique command names from a compiled MQTT bundle."""
    if not isinstance(bundle, dict):
        return []
    commands = bundle.get("commands")
    if not isinstance(commands, dict):
        return []
    raw = commands.get("supported")
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for entry in raw:
        name = _command_name_from_supported_entry(entry)
        if name:
            names.append(name)
    return sorted(set(names))


def command_specs(bundle: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return per-command specs from a compiled MQTT bundle (may be empty)."""
    if not isinstance(bundle, dict):
        return {}
    commands = bundle.get("commands")
    if not isinstance(commands, dict):
        return {}
    raw_specs = commands.get("specs")
    if isinstance(raw_specs, dict):
        return {
            str(key): copy.deepcopy(value)
            if isinstance(value, dict)
            else {}
            for key, value in raw_specs.items()
        }
    return {}


def command_spec(bundle: dict[str, Any] | None, command: str) -> dict[str, Any]:
    """Return the spec dict for *command* (empty when discrete or unknown)."""
    return copy.deepcopy(command_specs(bundle).get(command, {}))


def command_args(bundle: dict[str, Any] | None, command: str) -> list[dict[str, Any]]:
    """Return declared argument descriptors for *command* (empty when none)."""
    spec = command_spec(bundle, command)
    raw = spec.get("args")
    if not isinstance(raw, list):
        return []
    return [dict(a) for a in raw if isinstance(a, dict) and a.get("name")]


def mqtt_topic_slugs(bundle: dict[str, Any] | None) -> list[str]:
    """Return topic slug keys from a compiled MQTT bundle."""
    if not isinstance(bundle, dict):
        return []
    topics = bundle.get("topics")
    if not isinstance(topics, dict):
        return []
    return sorted(str(k) for k in topics.keys())


def zenoh_channel_names(bundle: dict[str, Any] | None) -> list[str]:
    """Return Zenoh channel keys from a compiled edge catalog bundle."""
    if not isinstance(bundle, dict):
        return []
    channels = bundle.get("channels")
    if not isinstance(channels, dict):
        return []
    return sorted(str(k) for k in channels.keys())


def supported_transports_from_metadata(
    metadata: dict[str, Any] | None,
) -> list[str]:
    """Return transport names present on twin/asset metadata (``mqtt``, ``zenoh``)."""
    transports: list[str] = []
    mqtt = extract_mqtt_bundle_from_metadata(metadata)
    if mqtt and mqtt_topic_slugs(mqtt):
        transports.append("mqtt")
    zenoh = extract_zenoh_bundle_from_metadata(metadata)
    if zenoh and zenoh_channel_names(zenoh):
        transports.append("zenoh")
    return transports


def has_locomotion_commands(bundle: dict[str, Any] | None) -> bool:
    """True when the bundle advertises standard locomotion MQTT commands."""
    supported = set(supported_mqtt_commands(bundle))
    return bool(supported & _LOCOMOTION_COMMANDS)


def has_joint_update_topic(bundle: dict[str, Any] | None) -> bool:
    """True when the bundle includes the joint update topic slug."""
    return JOINT_UPDATE_TOPIC_SLUG in set(mqtt_topic_slugs(bundle))


def resolve_outbound_topic_slug(
    *,
    channel: OutboundTopicChannel,
    bundle: dict[str, Any] | None,
) -> str:
    """Pick topic slug from catalog ``topics``; fall back to canonical slug when empty."""
    canonical = _CANONICAL_SLUG_BY_CHANNEL[channel]
    slugs = set(mqtt_topic_slugs(bundle))
    if not slugs:
        return canonical
    if canonical in slugs:
        return canonical
    raise ValueError(
        f"No MQTT topic slug for channel {channel!r}. "
        f"Expected {canonical!r} in catalog topics. Available: {sorted(slugs)}"
    )


def _topic_entry(bundle: dict[str, Any], slug: str) -> dict[str, Any]:
    topics = bundle.get("topics")
    if not isinstance(topics, dict):
        return {}
    entry = topics.get(slug)
    return entry if isinstance(entry, dict) else {}


def _slug_readable(slug: str, entry: dict[str, Any], *, stream: InboundStream) -> bool:
    direction = str(entry.get("direction", "both")).lower()
    if direction in {"subscribe", "both"}:
        return True
    # Platform publishes position/rotation for viz; locomotion twins still read them.
    if stream == "pose" and slug in {TWIN_POSITION_TOPIC_SLUG, TWIN_ROTATION_TOPIC_SLUG}:
        return slug.endswith("/position") or slug.endswith("/rotation")
    return False


def _battery_slugs(slugs: set[str]) -> list[str]:
    return sorted(
        s
        for s in slugs
        if "battery" in s.lower()
        and not s.endswith("/command")
    )


def resolve_inbound_topics(
    stream: InboundStream,
    bundle: dict[str, Any] | None,
    *,
    twin_uuid: str,
    topic_prefix: str = "",
) -> list[tuple[str, str]]:
    """Return ``(catalog_slug, resolved_topic)`` pairs for inbound subscribe."""
    if not isinstance(bundle, dict):
        bundle = {}

    catalog_slugs = set(mqtt_topic_slugs(bundle))
    prefix = topic_prefix or ""
    candidates = INBOUND_STREAM_SLUGS.get(stream, ())
    resolved: list[tuple[str, str]] = []

    if stream == "power":
        for slug in _battery_slugs(catalog_slugs):
            entry = _topic_entry(bundle, slug)
            if _slug_readable(slug, entry, stream=stream) or "battery" in slug:
                resolved.append((slug, f"{prefix}{slug.format(twin_uuid=twin_uuid)}"))
        if not resolved:
            raise NotImplementedError(
                f"No battery/status subscribe slug in catalog for twin {twin_uuid}. "
                f"Available: {sorted(catalog_slugs)}"
            )
        return resolved

    if stream == "joints":
        if JOINT_UPDATE_TOPIC_SLUG in catalog_slugs:
            resolved.append(
                (
                    JOINT_UPDATE_TOPIC_SLUG,
                    f"{prefix}{JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
                )
            )
    elif stream == "imu":
        if TWIN_IMU_TOPIC_SLUG in catalog_slugs:
            resolved.append(
                (
                    TWIN_IMU_TOPIC_SLUG,
                    f"{prefix}{TWIN_IMU_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
                )
            )
    elif stream == "gps":
        if TWIN_GPS_TOPIC_SLUG in catalog_slugs:
            resolved.append(
                (
                    TWIN_GPS_TOPIC_SLUG,
                    f"{prefix}{TWIN_GPS_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
                )
            )
    elif stream == "depth":
        if TWIN_DEPTH_TOPIC_SLUG in catalog_slugs:
            resolved.append(
                (
                    TWIN_DEPTH_TOPIC_SLUG,
                    f"{prefix}{TWIN_DEPTH_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
                )
            )
    elif stream == "pointcloud":
        if TWIN_POINTCLOUD_TOPIC_SLUG in catalog_slugs:
            resolved.append(
                (
                    TWIN_POINTCLOUD_TOPIC_SLUG,
                    f"{prefix}{TWIN_POINTCLOUD_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
                )
            )
    else:
        for slug in candidates:
            if slug not in catalog_slugs and stream != "pose":
                continue
            if slug not in catalog_slugs and stream == "pose":
                if slug not in {
                    TWIN_POSITION_TOPIC_SLUG,
                    TWIN_ROTATION_TOPIC_SLUG,
                    TWIN_KINEMATICS_TOPIC_SLUG,
                }:
                    continue
            entry = _topic_entry(bundle, slug)
            if slug in catalog_slugs and not _slug_readable(slug, entry, stream=stream):
                if stream == "pose" and slug in {
                    TWIN_POSITION_TOPIC_SLUG,
                    TWIN_ROTATION_TOPIC_SLUG,
                }:
                    pass
                elif stream != "pose":
                    continue
            resolved.append((slug, f"{prefix}{slug.format(twin_uuid=twin_uuid)}"))

    if stream == "pose":
        has_kinematics = any(TWIN_KINEMATICS_TOPIC_SLUG == s for s, _ in resolved)
        has_legacy = any(
            s in {TWIN_POSITION_TOPIC_SLUG, TWIN_ROTATION_TOPIC_SLUG} for s, _ in resolved
        )
        if not has_kinematics and not has_legacy:
            # Fallback: canonical position+rotation when catalog is command-only.
            resolved = [
                (TWIN_POSITION_TOPIC_SLUG, f"{prefix}{TWIN_POSITION_TOPIC_SLUG.format(twin_uuid=twin_uuid)}"),
                (TWIN_ROTATION_TOPIC_SLUG, f"{prefix}{TWIN_ROTATION_TOPIC_SLUG.format(twin_uuid=twin_uuid)}"),
            ]
    if not resolved and stream == "joints":
        # Fallback: joint state is published and read on /update only.
        resolved = [
            (
                JOINT_UPDATE_TOPIC_SLUG,
                f"{prefix}{JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
            ),
        ]
    if not resolved and stream == "imu":
        resolved = [
            (
                TWIN_IMU_TOPIC_SLUG,
                f"{prefix}{TWIN_IMU_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
            ),
        ]
    if not resolved and stream == "gps":
        resolved = [
            (
                TWIN_GPS_TOPIC_SLUG,
                f"{prefix}{TWIN_GPS_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
            ),
        ]
    if not resolved and stream == "depth":
        resolved = [
            (
                TWIN_DEPTH_TOPIC_SLUG,
                f"{prefix}{TWIN_DEPTH_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
            ),
        ]
    if not resolved and stream == "pointcloud":
        resolved = [
            (
                TWIN_POINTCLOUD_TOPIC_SLUG,
                f"{prefix}{TWIN_POINTCLOUD_TOPIC_SLUG.format(twin_uuid=twin_uuid)}",
            ),
        ]

    if not resolved:
        raise NotImplementedError(
            f"No inbound topics for stream {stream!r} on twin {twin_uuid}. "
            f"Available slugs: {sorted(catalog_slugs)}"
        )
    return resolved


def select_listen_slugs(
    bundle: dict[str, Any] | None,
    *,
    filters: list[str] | None = None,
    include_telemetry: bool = False,
) -> list[str]:
    """Catalog slugs to subscribe for ``twin.listen()``."""
    if not isinstance(bundle, dict):
        bundle = {}
    topics = bundle.get("topics")
    if not isinstance(topics, dict):
        topics = {}

    if filters is not None:
        unknown = set(filters) - _LISTEN_FILTER_NAMES
        if unknown:
            raise ValueError(
                f"Unknown listen filter(s): {sorted(unknown)}. "
                f"Use: {sorted(_LISTEN_FILTER_NAMES)}"
            )
        selected: set[str] = set()
        for name in filters:
            selected.update(INBOUND_STREAM_SLUGS.get(name, ()))  # type: ignore[arg-type]
            if name == "power":
                selected.update(_battery_slugs(set(topics.keys())))
    else:
        selected = set(topics.keys())

    result: list[str] = []
    for slug, entry in topics.items():
        if slug not in selected and filters is not None:
            continue
        if "/telemetry" in slug:
            if include_telemetry:
                result.append(str(slug))
            continue
        if filters is None:
            direction = str(entry.get("direction", "both")).lower() if isinstance(entry, dict) else "both"
            if direction not in {"subscribe", "both"}:
                if not (slug.endswith("/position") or slug.endswith("/rotation")):
                    continue
        result.append(str(slug))

    if filters is not None:
        for name in filters:
            for slug in INBOUND_STREAM_SLUGS.get(name, ()):  # type: ignore[arg-type]
                if slug and slug not in result:
                    result.append(str(slug))

    return sorted(result)

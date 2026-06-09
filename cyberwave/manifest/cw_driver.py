"""Compile ``cw-driver.yml`` into ``metadata[\"mqtt\"]`` MQTT interface catalogs.

Logic mirrors ``cyberwave-backend/src/lib/cw_driver_catalog.py`` (compile path only).
Used by :meth:`cyberwave.twin.commands.TwinCommandsHandle.set_schema`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from .driver_config import JOINT_UPDATE_TOPIC_SLUG

CW_DRIVER_FILE_NAME = "cw-driver.yml"
MQTT_BUNDLE_SCHEMA_VERSION = 1

_NAMESPACE_PREFIXES: dict[str, str] = {
    "joint": "cyberwave/joint/{twin_uuid}",
    "twin": "cyberwave/twin/{twin_uuid}",
    "webrtc": "cyberwave/twin/{twin_uuid}",
    "pose": "cyberwave/twin/{twin_uuid}",
    "environment": "cyberwave/environment/{environment_uuid}",
}

_WEBRTC_LEAF_ALIASES: dict[str, str] = {
    "offer": "webrtc-offer",
    "answer": "webrtc-answer",
    "candidate": "webrtc-candidate",
}

_LOCOMOTION_COMMANDS = frozenset(
    {
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
        "stop",
    }
)

_COMMAND_SPEC_KEYS = frozenset({"continuous", "rate_hz", "default_duration_s"})


class MqttTopicEntrySchema(BaseModel):
    """One logical MQTT topic in the compiled catalog."""

    description: str
    direction: str = Field(...)
    payload_schema_ref: str | None = None
    json_schema: dict[str, Any] | None = None
    units: dict[str, str] | None = None
    source_types: list[str] | None = None
    related_topics: list[str] | None = None
    direction_notes: str | None = None

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"publish", "subscribe", "both"}:
            raise ValueError(
                f"direction must be publish, subscribe, or both (got {value!r})"
            )
        return normalized


def _is_mqtt_bundle(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    topics = value.get("topics")
    return isinstance(topics, dict) and bool(topics)


def load_cw_driver_yml(path: Path | str) -> dict[str, Any]:
    """Load a ``cw-driver.yml`` file and return the parsed dict."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"cw-driver.yml not found: {file_path}")
    with file_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"cw-driver.yml root must be a mapping: {file_path}")
    return raw


def _is_topic_entry(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        "description" in value or "direction" in value or "payload_schema_ref" in value
    )


def _leaf_to_slug(namespace: str, leaf: str) -> str:
    prefix = _NAMESPACE_PREFIXES.get(namespace)
    if prefix is None:
        raise ValueError(f"Unknown mqtt namespace {namespace!r}")
    if namespace == "webrtc":
        segment = _WEBRTC_LEAF_ALIASES.get(leaf, leaf.replace("_", "-"))
        return f"{prefix}/{segment}"
    if leaf == "+":
        return f"{prefix}/+"
    return f"{prefix}/{leaf}"


def _flatten_mqtt_tree(mqtt_raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reserved = frozenset({"schema_version", "driver_family", "commands", "constraints"})
    topics: dict[str, dict[str, Any]] = {}
    for namespace, ns_body in mqtt_raw.items():
        if namespace in reserved or not isinstance(ns_body, dict):
            continue
        if namespace not in _NAMESPACE_PREFIXES:
            continue
        for leaf_key, leaf_value in ns_body.items():
            if not _is_topic_entry(leaf_value):
                continue
            slug = _leaf_to_slug(namespace, str(leaf_key))
            topics[slug] = MqttTopicEntrySchema.model_validate(leaf_value).model_dump(
                exclude_none=True
            )
    return topics


def normalize_supported_commands(
    raw_list: list[Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    names: list[str] = []
    specs: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    for entry in raw_list:
        if isinstance(entry, str):
            name = entry.strip()
            if not name:
                raise ValueError("supported command name must not be empty")
            spec: dict[str, Any] = {}
        elif isinstance(entry, dict):
            raw_name = entry.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(
                    "supported command object must include a non-empty 'name'"
                )
            name = raw_name.strip()
            unknown = set(entry.keys()) - {"name"} - _COMMAND_SPEC_KEYS
            if unknown:
                raise ValueError(f"unknown keys on command {name!r}: {sorted(unknown)}")
            spec = {}
            if entry.get("continuous"):
                spec["continuous"] = True
            if "rate_hz" in entry:
                spec["rate_hz"] = float(entry["rate_hz"])
            if "default_duration_s" in entry:
                spec["default_duration_s"] = float(entry["default_duration_s"])
        else:
            raise ValueError(
                "each supported entry must be a string or mapping with 'name'"
            )

        if name in seen:
            raise ValueError(f"duplicate supported command: {name!r}")
        seen.add(name)
        names.append(name)
        specs[name] = spec

    return names, specs


def _normalize_commands_block(commands_raw: Any) -> dict[str, Any]:
    if isinstance(commands_raw, list):
        names, specs = normalize_supported_commands(commands_raw)
        return {"supported": names, "specs": specs}

    if not isinstance(commands_raw, dict):
        return {}

    result: dict[str, Any] = {}
    for key, value in commands_raw.items():
        if key == "supported" and isinstance(value, list):
            names, specs = normalize_supported_commands(value)
            result["supported"] = names
            result["specs"] = specs
        else:
            result[key] = value
    return result


def _first_registry_id(raw: dict[str, Any]) -> str | None:
    registry_ids = raw.get("registry_ids")
    if not isinstance(registry_ids, list):
        return None
    for rid in registry_ids:
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    return None


def compile_driver_mqtt_bundle(
    raw: dict[str, Any],
    *,
    cw_driver_yml_path: str | None = None,
    registry_id: str | None = None,
) -> dict[str, Any]:
    """Compile a loaded cw-driver.yml dict into ``metadata[\"mqtt\"]`` shape."""
    mqtt_raw = raw.get("mqtt")
    if not isinstance(mqtt_raw, dict):
        raise ValueError("cw-driver.yml must contain a top-level 'mqtt' mapping")

    schema_version = int(mqtt_raw.get("schema_version", MQTT_BUNDLE_SCHEMA_VERSION))
    driver_family = mqtt_raw.get("driver_family")
    topics = _flatten_mqtt_tree(mqtt_raw)

    commands_block: dict[str, Any] = {}
    if isinstance(mqtt_raw.get("commands"), dict):
        commands_block = _normalize_commands_block(mqtt_raw["commands"])
    constraints = mqtt_raw.get("constraints")
    if isinstance(constraints, list):
        commands_block.setdefault("constraints", [str(c) for c in constraints])

    provenance: dict[str, Any] = {
        "source": "cw-driver.yml",
        "capabilities_derived": False,
    }
    if cw_driver_yml_path:
        provenance["cw_driver_yml"] = cw_driver_yml_path

    bundle: dict[str, Any] = {
        "schema_version": schema_version,
        "topics": topics,
        "commands": commands_block,
        "provenance": provenance,
    }
    if driver_family:
        bundle["driver_family"] = str(driver_family)
    rid = registry_id or _first_registry_id(raw)
    if rid:
        bundle["asset_registry_id"] = rid
        provenance["registry_id"] = rid
    return enrich_mqtt_bundle_for_agents(bundle)


def compile_cw_driver_file(
    path: Path | str,
    *,
    registry_id: str | None = None,
) -> dict[str, Any]:
    """Load and compile a single ``cw-driver.yml`` file."""
    file_path = Path(path)
    raw = load_cw_driver_yml(file_path)
    return compile_driver_mqtt_bundle(
        raw,
        cw_driver_yml_path=str(file_path.resolve()),
        registry_id=registry_id,
    )


def enrich_mqtt_bundle_for_agents(bundle: dict[str, Any]) -> dict[str, Any]:
    """Add recommended_sdk_methods and example_payloads for agents/MCP."""
    result = copy.deepcopy(bundle)
    commands = result.get("commands") or {}
    supported: list[str] = []
    command_specs: dict[str, dict[str, Any]] = {}
    if isinstance(commands, dict):
        raw_supported = commands.get("supported")
        if isinstance(raw_supported, list):
            supported = [
                str(c.get("name") if isinstance(c, dict) else c)
                for c in raw_supported
                if c
            ]
        raw_specs = commands.get("specs")
        if isinstance(raw_specs, dict):
            command_specs = {
                str(k): dict(v) if isinstance(v, dict) else {}
                for k, v in raw_specs.items()
            }

    recommended: list[dict[str, str | None]] = []
    examples: list[dict[str, Any]] = []

    if any(c in _LOCOMOTION_COMMANDS for c in supported):
        recommended.extend(
            [
                {
                    "intent": "locomote_forward",
                    "method": "twin.move_forward(distance_m, duration=..., rate_hz=...)",
                    "mqtt_command": "move_forward",
                },
                {
                    "intent": "stop",
                    "method": "publish command 'stop' on twin command topic",
                    "mqtt_command": "stop",
                },
            ]
        )
        examples.append(
            {
                "topic_slug": "cyberwave/twin/{twin_uuid}/command",
                "payload": {
                    "command": "move_forward",
                    "data": {"linear_x": 1.0, "angular_z": 0.0},
                    "source_type": "tele",
                },
            }
        )

    if JOINT_UPDATE_TOPIC_SLUG in (result.get("topics") or {}):
        recommended.append(
            {
                "intent": "set_joint",
                "method": "twin.joints[joint_name] = angle_rad",
                "mqtt_command": None,
            }
        )

    result["recommended_sdk_methods"] = recommended
    result["example_payloads"] = examples
    return result


def merge_mqtt_into_metadata(
    metadata: dict[str, Any],
    bundle: dict[str, Any],
    *,
    merge: bool,
) -> dict[str, Any]:
    """Return a copy of *metadata* with ``mqtt`` set or merged from *bundle*."""
    meta = copy.deepcopy(metadata)
    if not merge:
        meta["mqtt"] = copy.deepcopy(bundle)
        return meta

    prior = meta.get("mqtt")
    if not isinstance(prior, dict):
        meta["mqtt"] = copy.deepcopy(bundle)
        return meta

    merged = copy.deepcopy(prior)
    for key, value in bundle.items():
        if key == "topics" and isinstance(value, dict):
            topics = merged.setdefault("topics", {})
            if isinstance(topics, dict):
                topics.update(copy.deepcopy(value))
            continue
        if key == "commands" and isinstance(value, dict):
            commands = merged.setdefault("commands", {})
            if isinstance(commands, dict):
                for ck, cv in value.items():
                    if ck == "supported" and isinstance(cv, list):
                        existing = commands.get("supported") or []
                        commands["supported"] = sorted(
                            {*(existing if isinstance(existing, list) else []), *cv}
                        )
                    else:
                        commands[ck] = copy.deepcopy(cv)
            continue
        merged[key] = copy.deepcopy(value)
    meta["mqtt"] = merged
    return meta


def resolve_mqtt_bundle_from_driver_config(
    driver_config: Union[str, Path, dict[str, Any]],
    *,
    registry_id: str | None = None,
) -> dict[str, Any]:
    """Compile a path, cw-driver dict, or pre-built MQTT bundle."""
    if isinstance(driver_config, (str, Path)):
        return compile_cw_driver_file(driver_config, registry_id=registry_id)

    if not isinstance(driver_config, dict):
        raise TypeError(
            "driver_config must be a path, cw-driver.yml dict, or mqtt bundle dict"
        )

    if _is_mqtt_bundle(driver_config):
        return enrich_mqtt_bundle_for_agents(copy.deepcopy(driver_config))

    nested = driver_config.get("mqtt")
    if isinstance(nested, dict) and _is_mqtt_bundle(nested):
        return enrich_mqtt_bundle_for_agents(copy.deepcopy(nested))

    if isinstance(nested, dict) and "topics" not in nested:
        return compile_driver_mqtt_bundle(
            driver_config,
            registry_id=registry_id or _first_registry_id(driver_config),
        )

    if "mqtt" in driver_config:
        return compile_driver_mqtt_bundle(
            driver_config,
            registry_id=registry_id or _first_registry_id(driver_config),
        )

    raise ValueError(
        "driver_config dict must be a cw-driver.yml root (with 'mqtt' mapping) "
        "or a compiled metadata['mqtt'] bundle (with 'topics')"
    )

# !! GENERATED from common.controller_ref.v1.schema.json, common.runtime_target.v1.schema.json -- do not edit.
# Run `make -C cyberwave-contracts models` in the monorepo to regenerate.

from typing import Any, Dict, Literal

from typing_extensions import NotRequired, Required, TypedDict


class PolicyRefPayload(TypedDict, total=False):
    """Cyberwave Controller Reference

    A typed reference to a controller policy: how it is looked up, and the value to look it up by. Used wherever a payload or a stored section names a controller (policy_ref, manual_controller_ref, default_runtime_routes values). `catalog_key` is the current spelling of a catalog reference; `catalog_seed_id` is its legacy spelling, accepted on read for one version and still what the backend emits until writers move to `catalog_key`.
    """

    kind: Required[Literal["uuid", "catalog_key", "catalog_seed_id", "slug"]]
    value: Required[str]


class ControlRuntimeTargetPayload(TypedDict, total=False):
    """Cyberwave Controller Runtime Target

    Where and how a controller policy runs for one runtime kind: whether it is enabled, the backend that executes it, and the contracts it reads and writes. Resolved by the backend from a policy's metadata and returned in controller setup views.
    """

    backend: Required[str]
    enabled: Required[bool]
    runtime_kind: Required[Literal["physical", "simulation"]]
    adapter: NotRequired[str | None]
    input_contract: NotRequired[str | None]
    metadata: NotRequired[Dict[str, Any]]
    output_contract: NotRequired[str | None]
    safety_level: NotRequired[str | None]
    source_type: NotRequired[str | None]

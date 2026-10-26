"""
Marconi100 rack scene — one equipment rack and three server twins bound to recorded channels.

Creates the rack (cyberwave/equipment_rack procedural primitive) and the twins
r48-s00-bottom / r48-s10-middle / r48-s19-top (cyberwave/generic_box) with the
sensor slots of contract section 6 and bindings in the v2 shape of the
replay-feed contract (section 1) with "Use recorded data" switched on.
Re-running updates the existing rack and twins instead of duplicating them
and rewrites legacy bindings in the v2 shape.

Usage:
    python make_scene.py --environment <uuid> --dataset <uuid>
        [--base-url http://localhost:8000] [--api-key test-api-key]

Requirements:
    pip install -r requirements.txt
"""

import argparse

from cyberwave import Cyberwave
from cyberwave.rest.models.environment_procedural_primitive_create_schema import (
    EnvironmentProceduralPrimitiveCreateSchema,
)
from cyberwave.rest.models.environment_procedural_primitive_patch_schema import (
    EnvironmentProceduralPrimitivePatchSchema,
)
from cyberwave.rest.models.position import Position

RACK_ID = "rack-48"
RACK = {
    "name": "Rack 48",
    "template_key": "cyberwave/equipment_rack",
    "parameters": {"width": 0.7, "depth": 0.8, "height": 1.9, "levels": 4},
}
# A primitive's position is its bounding-box centre: base on the floor at the origin.
RACK_CENTER = {"x": 0.0, "y": 0.0, "z": 0.95}

SERVER_ASSET = "cyberwave/generic_box"
SIMULATION_BACKENDS = ["mujoco", "playground"]
# (twin name, Marconi node, z of the box in the rack)
SERVERS = [("r48-s00-bottom", 960, 0.15), ("r48-s10-middle", 970, 1.0), ("r48-s19-top", 979, 1.75)]

SENSORS = [
    {"id": "inlet_temperature", "name": "Inlet air temperature", "type": "temperature", "parameters": {"unit": "°C"}},
    {"id": "power", "name": "Server power", "type": "power", "parameters": {"unit": "W"}},
    {"id": "fan_speed", "name": "Fan speed", "type": "fan_speed", "parameters": {"unit": "rpm"}},
]


def data_sources(dataset: str, node: int) -> dict:
    """Bindings v2: a key only for slots with a channel attached; fan_speed stays with the environment."""
    return {
        "inlet_temperature": {
            "dataset_uuid": dataset,
            "channel_id": f"ipmi/ambient/node={node}",
            "use_recorded": True,
            "label": "Inlet air temperature",
            "unit": "°C",
            "type": "temperature",
        },
        "power": {
            "dataset_uuid": dataset,
            "channel_id": f"ipmi/total_power/node={node}",
            "use_recorded": True,
            "label": "Server total power",
            "unit": "W",
            "type": "power",
        },
    }


def ensure_rack(cw: Cyberwave, environment: str) -> None:
    try:
        cw.api.src_app_api_environments_create_environment_procedural_primitive(
            environment,
            EnvironmentProceduralPrimitiveCreateSchema(
                id=RACK_ID, position=Position(RACK_CENTER), fixed_base=True, **RACK
            ),
        )
        print(f"rack {RACK_ID}: created")
    except Exception as exc:
        if getattr(exc, "status", None) != 409:
            raise
        # Same id already in the environment: refresh its parameters and pose.
        cw.api.src_app_api_environments_update_environment_procedural_primitive(
            environment,
            RACK_ID,
            EnvironmentProceduralPrimitivePatchSchema(
                position=Position(RACK_CENTER), fixed_base=True, **RACK
            ),
        )
        print(f"rack {RACK_ID}: updated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--environment", required=True, help="environment uuid")
    parser.add_argument("--dataset", required=True, help="time-series dataset uuid the twins replay")
    parser.add_argument("--base-url", help="backend URL (default: CYBERWAVE_BASE_URL)")
    parser.add_argument("--api-key", help="API key (default: CYBERWAVE_API_KEY)")
    args = parser.parse_args()

    cw = Cyberwave(base_url=args.base_url, api_key=args.api_key, environment_id=args.environment)
    ensure_rack(cw, args.environment)

    scene = cw.get_scene(args.environment)
    existing = {twin.name: twin for twin in cw.twins.list(environment_id=args.environment)}
    for name, node, z in SERVERS:
        twin = existing.get(name)
        if twin is None:
            twin = scene.add_twin(SERVER_ASSET, name=name, position=[0.0, 0.0, z], fixed_base=True)
        # PUT merges top-level metadata keys: send the whole sensors list and data_sources map.
        cw.twins.update(
            twin.uuid,
            position_x=0.0,
            position_y=0.0,
            position_z=z,
            fixed_base=True,
            metadata={"sensors": SENSORS, "data_sources": data_sources(args.dataset, node)},
            supported_simulation_backends=SIMULATION_BACKENDS,
        )
        print(f"{name}: {twin.uuid}  (node={node})")


if __name__ == "__main__":
    main()

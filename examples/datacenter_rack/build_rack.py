"""
Rack build — one 42U rack with twenty docked 2U servers bound to recorded data.

Creates, idempotently by twin name, the rack twin rack-<n> from
cyberwave/server-rack-42u and the servers rack-<n>-s00 .. s19 from
cyberwave/server-2u, docked to the rack link server_rack_42u_link bottom to
top, each with its identifier in datasets (thing_id node=<20 n + i>, the
Marconi100 numbering). Then auto-binds the servers' sensor slots to the
dataset's channels with a slot template and prints the report. --as-kit also
registers the assembly as the catalog kit cyberwave/server-rack-42u-20x2u.
--dry-run prints the plan and calls no API.

Usage:
    python build_rack.py --environment <uuid> --rack 48 --dataset <uuid>
        [--template marconi100|<json file>] [--use-recorded | --no-use-recorded]
        [--position x y z] [--alerts | --no-alerts] [--as-kit] [--dry-run]
        [--base-url http://localhost:8000] [--api-key test-api-key]

Requirements:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cyberwave import Cyberwave

RACK_ASSET = "cyberwave/server-rack-42u"
RACK_LINK = "server_rack_42u_link"
SERVER_ASSET = "cyberwave/server-2u"
SERVERS_PER_RACK = 20
KIT_REGISTRY_ID = "cyberwave/server-rack-42u-20x2u"
KIT_NAME = "Server rack 42U with 20 x 2U servers"
KIT_DESCRIPTION = "One cyberwave/server-rack-42u with twenty cyberwave/server-2u docked bottom to top at 2U pitch."
# Dock offsets on the rack link, in metres: the servers sit at x = 0.105 in front
# of the rails, one 2U pitch (2 x 44.45 mm) apart, the first 0.10 m above the link.
SERVER_X = 0.105
SERVER_Y = 0.0
U_HEIGHT = 0.04445
BOTTOM_Z = 0.10
IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)  # scene.add_twin / scene.dock
# Scene visuals while a sensor is out of range (replay-feed README §16.5):
# servers turn red, the rack shows a pulsing "!" when any docked server alerts.
SERVER_ALERT_ACTION = {"kind": "color", "color": "#ef4444"}
RACK_ALERT_ACTION = {"kind": "marker"}
IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)  # asset kits
HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"


def rack_name(rack: int) -> str:
    return f"rack-{rack}"


def server_name(rack: int, index: int) -> str:
    return f"{rack_name(rack)}-s{index:02d}"


def node_number(rack: int, index: int) -> int:
    """Marconi100 numbering: rack n holds nodes 20n .. 20n + 19, bottom to top."""
    return SERVERS_PER_RACK * rack + index


def server_offset(index: int) -> tuple[float, float, float]:
    """Dock offset of server i (0 = bottom, 19 = top) on the rack link."""
    return (SERVER_X, SERVER_Y, round(BOTTOM_Z + U_HEIGHT * (2 * index + 1), 5))


def plan_servers(rack: int) -> list[dict[str, Any]]:
    """The twenty servers of a rack: name, node, thing_id and dock offset."""
    return [
        {
            "index": i,
            "name": server_name(rack, i),
            "node": node_number(rack, i),
            "thing_id": f"node={node_number(rack, i)}",
            "offset": server_offset(i),
        }
        for i in range(SERVERS_PER_RACK)
    ]


def load_template(spec: str) -> dict[str, str]:
    """A slot template (slot id -> signal): a bundled name or a JSON file path."""
    candidates = [Path(spec), HERE / spec, TEMPLATES_DIR / f"{spec}.json"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        bundled = ", ".join(sorted(p.stem for p in TEMPLATES_DIR.glob("*.json")))
        raise SystemExit(f"template {spec!r} not found (bundled: {bundled})")
    with open(path, encoding="utf-8") as fh:
        template = json.load(fh)
    if not isinstance(template, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v
        for k, v in template.items()
    ):
        raise SystemExit(f"{path}: a template maps slot ids to signal names")
    return template


def kit_components(
    rack_asset_uuid: str, server_asset_uuid: str
) -> list[dict[str, Any]]:
    """Kit definition: the rack as root, the servers docked with the build offsets."""
    components: list[dict[str, Any]] = [
        {"key": "rack", "asset_uuid": rack_asset_uuid, "name": "rack"}
    ]
    for i in range(SERVERS_PER_RACK):
        key = f"s{i:02d}"
        components.append(
            {
                "key": key,
                "asset_uuid": server_asset_uuid,
                "name": f"server-{key}",
                "parent_key": "rack",
                "attach_to_link": RACK_LINK,
                "position": list(server_offset(i)),
                "rotation": list(IDENTITY_WXYZ),
            }
        )
    return components


def print_plan(
    args: argparse.Namespace, template: dict[str, str], servers: list[dict[str, Any]]
) -> None:
    x, y, z = args.position
    print(f"environment {args.environment}, dataset {args.dataset}")
    print(f"{rack_name(args.rack)}: {RACK_ASSET} at ({x}, {y}, {z}), fixed base")
    for server in servers:
        ox, oy, oz = server["offset"]
        print(
            f"{server['name']}: {SERVER_ASSET} docked to {RACK_LINK} at ({ox}, {oy}, {oz:.5f}), "
            f"thing_id {server['thing_id']}"
        )
    print(
        f"template ({len(template)} slots): "
        + ", ".join(f"{slot} <- {signal}" for slot, signal in template.items())
    )
    print(f"auto-bind: use_recorded {'on' if args.use_recorded else 'off'}")
    print(f"alerts: {describe_alerts(args.alerts)}")
    if args.as_kit:
        print(
            f"kit {KIT_REGISTRY_ID}: rack + {SERVERS_PER_RACK} servers with the same offsets"
        )


def describe_alerts(alerts: bool) -> str:
    if not alerts:
        return "off (alert_action untouched)"
    return (
        f"servers {SERVER_ALERT_ACTION['kind']} {SERVER_ALERT_ACTION['color']}, "
        f"rack {RACK_ALERT_ACTION['kind']}"
    )


def server_metadata(server: dict[str, Any], alerts: bool) -> dict[str, Any]:
    """The top-level metadata keys written to a server in one PUT."""
    metadata: dict[str, Any] = {"thing_id": server["thing_id"]}
    if alerts:
        metadata["alert_action"] = dict(SERVER_ALERT_ACTION)
    return metadata


def require_asset(cw: Cyberwave, registry_id: str) -> Any:
    asset = cw.assets.get_by_registry_id(registry_id)
    if asset is None:
        raise SystemExit(
            f"asset {registry_id} not found in the catalog; seed it first (see README)"
        )
    return asset


def ensure_rack(
    cw: Cyberwave, scene: Any, args: argparse.Namespace, existing: dict[str, Any]
) -> Any:
    name = rack_name(args.rack)
    x, y, z = args.position
    twin = existing.get(name)
    alerts = getattr(args, "alerts", True)
    if twin is None:
        twin = scene.add_twin(
            RACK_ASSET, name=name, position=[x, y, z], fixed_base=True
        )
        if alerts:
            # POST /twins keeps the asset's metadata only, so the marker is a
            # second write on a new rack; on an existing one it rides along.
            cw.twins.update(twin.uuid, metadata={"alert_action": RACK_ALERT_ACTION})
        print(f"{name}: created {twin.uuid}")
    else:
        fields: dict[str, Any] = {
            "position_x": x,
            "position_y": y,
            "position_z": z,
            "fixed_base": True,
        }
        if alerts:
            fields["metadata"] = {"alert_action": RACK_ALERT_ACTION}
        cw.twins.update(twin.uuid, **fields)
        print(f"{name}: exists {twin.uuid}, position updated")
    return twin


def ensure_servers(
    cw: Cyberwave,
    scene: Any,
    rack_twin: Any,
    servers: list[dict[str, Any]],
    existing: dict[str, Any],
    alerts: bool = True,
) -> list[Any]:
    twins = []
    for server in servers:
        twin = existing.get(server["name"])
        if twin is None:
            twin = scene.add_twin(
                SERVER_ASSET,
                name=server["name"],
                fixed_base=True,
                dock_to=rack_twin,
                dock_link=RACK_LINK,
                dock_offset=server["offset"],
                dock_rotation=IDENTITY_XYZW,
            )
            state = "created"
        else:
            # Re-dock with the build offsets so a moved server returns to its slot.
            scene.dock(
                twin,
                rack_twin,
                RACK_LINK,
                offset_position=list(server["offset"]),
                offset_rotation=list(IDENTITY_XYZW),
            )
            state = "exists"
        # One PUT per server: the identifier and the alert visual together.
        cw.twins.update(twin.uuid, metadata=server_metadata(server, alerts))
        print(
            f"{server['name']}: {state} {twin.uuid}, {server['thing_id']}, z {server['offset'][2]:.5f}"
        )
        twins.append(twin)
    return twins


def print_report(report: dict[str, Any]) -> None:
    for twin in report.get("twins", []):
        bound = twin.get("bound", [])
        skipped = twin.get("skipped", [])
        print(
            f"{twin.get('twin_name')} ({twin.get('thing_id')}): bound {len(bound)}, skipped {len(skipped)}"
        )
        if bound:
            print(
                "  bound: "
                + ", ".join(f"{b['slot']} <- {b['channel_id']}" for b in bound)
            )
        if skipped:
            print(
                "  skipped: "
                + "; ".join(f"{s['slot']}: {s['reason']}" for s in skipped)
            )
    print(f"total: bound {report.get('bound', 0)}, skipped {report.get('skipped', 0)}")


def register_kit(cw: Cyberwave, rack_asset: Any, server_asset: Any) -> None:
    components = kit_components(rack_asset.uuid, server_asset.uuid)
    kit = cw.assets.get_by_registry_id(KIT_REGISTRY_ID)
    if kit is None:
        created = cw.asset_kits.create(
            KIT_NAME, KIT_REGISTRY_ID, components, description=KIT_DESCRIPTION
        )
        print(
            f"kit {KIT_REGISTRY_ID}: created {created['asset']['uuid']} ({len(components)} components)"
        )
    else:
        cw.assets.update_kit(
            kit.uuid, name=KIT_NAME, components=components, description=KIT_DESCRIPTION
        )
        print(
            f"kit {KIT_REGISTRY_ID}: updated {kit.uuid} ({len(components)} components)"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--environment", required=True, help="environment uuid")
    parser.add_argument(
        "--rack",
        type=int,
        required=True,
        help="rack number n: twins rack-<n>, nodes 20n .. 20n+19",
    )
    parser.add_argument(
        "--dataset", required=True, help="time-series dataset uuid to bind"
    )
    parser.add_argument(
        "--template",
        default="marconi100",
        help="slot template: bundled name or JSON file (default marconi100)",
    )
    parser.add_argument(
        "--use-recorded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="switch 'Use recorded data' on for the new bindings (default on)",
    )
    parser.add_argument(
        "--position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="rack position in metres (default 0 0 0)",
    )
    parser.add_argument(
        "--alerts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "write metadata.alert_action: servers turn red, the rack shows a "
            "marker while a sensor is out of range (default on; --no-alerts skips)"
        ),
    )
    parser.add_argument(
        "--as-kit", action="store_true", help=f"also register the kit {KIT_REGISTRY_ID}"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and call no API"
    )
    parser.add_argument("--base-url", help="backend URL (default: CYBERWAVE_BASE_URL)")
    parser.add_argument("--api-key", help="API key (default: CYBERWAVE_API_KEY)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    template = load_template(args.template)
    servers = plan_servers(args.rack)
    print_plan(args, template, servers)
    if args.dry_run:
        print("dry run: nothing was sent")
        return

    cw = Cyberwave(
        base_url=args.base_url, api_key=args.api_key, environment_id=args.environment
    )
    rack_asset = require_asset(cw, RACK_ASSET)
    server_asset = require_asset(cw, SERVER_ASSET)
    scene = cw.get_scene(args.environment)
    existing = {
        twin.name: twin for twin in cw.twins.list(environment_id=args.environment)
    }

    print("\n== twins ==")
    rack_twin = ensure_rack(cw, scene, args, existing)
    server_twins = ensure_servers(
        cw, scene, rack_twin, servers, existing, alerts=args.alerts
    )

    print("\n== bindings ==")
    report = cw.environments.auto_bind(
        args.environment,
        args.dataset,
        twin_ids=[twin.uuid for twin in server_twins],
        template=template,
        use_recorded=args.use_recorded,
    )
    print_report(report)

    if args.as_kit:
        print("\n== kit ==")
        register_kit(cw, rack_asset, server_asset)


if __name__ == "__main__":
    main()

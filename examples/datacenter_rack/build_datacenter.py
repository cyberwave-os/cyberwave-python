"""
Data-center build — a whole Marconi100 room of racks with docked servers bound
to recorded data, placed from a layout file.

Reuses build_rack.py: for every row of the layout table one rack twin rack-<n>
from cyberwave/server-rack-42u, placed at (X * tile, Y * tile, 0) and turned by
a yaw of +/-90 degrees so the racks of a row stand side by side along the
room's X axis, then its twenty servers rack-<n>-s00 .. s19 docked bottom to top
with thing_id node=<id> from the layout. One auto_bind call covers all servers
of the run and a compact report follows. Idempotent by twin name, together
with build_rack.py. --dry-run prints the plan and calls no API.

Layout: a markdown table with the columns Rack | X | Y | Nodes (identifiers);
X and Y are floor-tile grid coordinates, the nodes are listed bottom to top.
The file has no front orientation: --facing says which way the glass doors
look (default -y, written --facing=-y because of the leading minus).

Usage:
    python build_datacenter.py --environment <uuid> --dataset <uuid>
        [--layout layouts/marconi100.md] [--racks all|48|46-48|0,17,48]
        [--tile 0.6] [--facing=-y|+y] [--template marconi100|<json file>]
        [--use-recorded | --no-use-recorded] [--alerts | --no-alerts] [--full-report] [--dry-run]
        [--base-url http://localhost:8000] [--api-key test-api-key]

Requirements:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cyberwave import Cyberwave

import build_rack
from build_rack import (
    RACK_ALERT_ACTION,
    RACK_ASSET,
    SERVER_ASSET,
    SERVERS_PER_RACK,
)

HERE = Path(__file__).resolve().parent
DEFAULT_LAYOUT = HERE / "layouts" / "marconi100.md"
DEFAULT_TILE = 0.6  # metres per grid unit: one floor tile per rack width
# The rack asset faces +x and is 0.6 m wide along its y axis. A yaw about z
# turns the front to (cos yaw, sin yaw): -90 degrees for -Y, +90 for +Y.
FACING_YAW = {"-y": -90.0, "+y": 90.0}
MAX_REASONS = 5
EXAMPLES_PER_REASON = 3


@dataclass(frozen=True)
class LayoutRack:
    number: int
    x: float
    y: float
    nodes: tuple[int, ...]


def _cells(line: str) -> list[str]:
    cells = [c.strip() for c in line.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(c and set(c) <= set("-:") for c in cells)


def parse_layout(path: Path) -> list[LayoutRack]:
    """The racks of a layout file: rows of its markdown table Rack | X | Y | Nodes."""
    racks: list[LayoutRack] = []
    seen: dict[int, int] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if "|" not in line:
                continue
            cells = _cells(line)
            if not cells or cells[0].lower() == "rack" or _is_separator(cells):
                continue
            if len(cells) != 4:
                raise SystemExit(
                    f"{path}:{lineno}: expected 4 columns (Rack | X | Y | Nodes), got {len(cells)}"
                )
            try:
                number = int(cells[0])
                x, y = float(cells[1]), float(cells[2])
                nodes = tuple(int(n) for n in cells[3].split(",") if n.strip())
            except ValueError as exc:
                raise SystemExit(f"{path}:{lineno}: {exc}") from None
            if number in seen:
                raise SystemExit(
                    f"{path}:{lineno}: rack {number} already defined on line {seen[number]}"
                )
            seen[number] = lineno
            racks.append(LayoutRack(number, x, y, nodes))
    if not racks:
        raise SystemExit(f"{path}: no layout table (Rack | X | Y | Nodes) found")
    return racks


def _span(nodes: tuple[int, ...]) -> str:
    return f"{nodes[0]}..{nodes[-1]}" if nodes else "none"


def layout_problems(racks: list[LayoutRack]) -> list[str]:
    """Nodes that disagree with build_rack.node_number, shared nodes, shared tiles."""
    problems = []
    owner: dict[int, int] = {}
    tiles: dict[tuple[float, float], int] = {}
    for rack in racks:
        expected = tuple(
            build_rack.node_number(rack.number, i) for i in range(SERVERS_PER_RACK)
        )
        if rack.nodes != expected:
            problems.append(
                f"rack {rack.number}: layout lists nodes {_span(rack.nodes)} "
                f"({len(rack.nodes)}), build_rack expects {_span(expected)} ({len(expected)})"
            )
        for node in rack.nodes:
            if node in owner and owner[node] != rack.number:
                problems.append(
                    f"rack {rack.number}: node {node} also listed for rack {owner[node]}"
                )
            owner.setdefault(node, rack.number)
        tile = (rack.x, rack.y)
        if tile in tiles:
            problems.append(
                f"rack {rack.number}: tile ({rack.x:g}, {rack.y:g}) also used by rack {tiles[tile]}"
            )
        tiles.setdefault(tile, rack.number)
    return problems


def parse_racks(spec: str, available: list[int]) -> list[int]:
    """Rack numbers of 'all', '48', '46-48' or '0,17,48', in layout order."""
    if spec.strip().lower() == "all":
        return list(available)
    wanted: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        try:
            if "-" in token:
                low, high = (int(part) for part in token.split("-", 1))
                if low > high:
                    raise SystemExit(f"--racks: empty range {token!r}")
                wanted.update(range(low, high + 1))
            else:
                wanted.add(int(token))
        except ValueError:
            raise SystemExit(f"--racks: bad token {token!r} (use 48, 46-48 or 0,17,48)") from None
    missing = sorted(wanted - set(available))
    if missing:
        raise SystemExit(f"--racks: not in the layout: {', '.join(map(str, missing))}")
    return [number for number in available if number in wanted]


def rack_position(rack: LayoutRack, tile: float) -> tuple[float, float, float]:
    """World position in metres: grid coordinates scaled by the tile size, on the floor."""
    return (round(rack.x * tile, 6), round(rack.y * tile, 6), 0.0)


def facing_quaternion(facing: str) -> tuple[float, float, float, float]:
    """Rotation (w, x, y, z) of a rack whose front looks along -Y or +Y."""
    half = math.radians(FACING_YAW[facing]) / 2
    return (round(math.cos(half), 6), 0.0, 0.0, round(math.sin(half), 6))


def plan_room(
    racks: list[LayoutRack], numbers: list[int], tile: float, facing: str
) -> list[dict[str, Any]]:
    """One entry per selected rack: name, position, rotation and its servers."""
    by_number = {rack.number: rack for rack in racks}
    rotation = facing_quaternion(facing)
    return [
        {
            "number": number,
            "name": build_rack.rack_name(number),
            "position": rack_position(by_number[number], tile),
            "yaw": FACING_YAW[facing],
            "rotation": rotation,
            "servers": build_rack.plan_servers(number),
        }
        for number in numbers
    ]


def print_plan(
    args: argparse.Namespace, template: dict[str, str], plan: list[dict[str, Any]]
) -> None:
    print(f"environment {args.environment}, dataset {args.dataset}")
    print(
        f"layout {args.layout}: tile {args.tile:g} m, fronts facing {args.facing.upper()} "
        f"(yaw {FACING_YAW[args.facing]:g} deg)"
    )
    for entry in plan:
        x, y, z = entry["position"]
        nodes = [server["node"] for server in entry["servers"]]
        print(
            f"{entry['name']}: {RACK_ASSET} at ({x:g}, {y:g}, {z:g}), yaw {entry['yaw']:g} deg, "
            f"fixed base, {len(nodes)} x {SERVER_ASSET} nodes {nodes[0]}..{nodes[-1]}"
        )
    servers = sum(len(entry["servers"]) for entry in plan)
    print(f"{len(plan)} racks, {servers} servers")
    print(
        f"template ({len(template)} slots): "
        + ", ".join(f"{slot} <- {signal}" for slot, signal in template.items())
    )
    print(f"auto-bind: use_recorded {'on' if args.use_recorded else 'off'}")
    print(f"alerts: {build_rack.describe_alerts(args.alerts)}")


def ensure_rack(
    cw: Cyberwave,
    scene: Any,
    entry: dict[str, Any],
    existing: dict[str, Any],
    alerts: bool = True,
) -> tuple[Any, str]:
    """Create or re-place the rack twin; returns (twin, 'created' | 'updated')."""
    x, y, z = entry["position"]
    w, qx, qy, qz = entry["rotation"]
    twin = existing.get(entry["name"])
    if twin is None:
        twin = scene.add_twin(
            RACK_ASSET,
            name=entry["name"],
            position=[x, y, z],
            orientation=[qx, qy, qz, w],
            fixed_base=True,
        )
        if alerts:
            # POST /twins keeps the asset's metadata only: one more write per
            # new rack; on an existing rack it rides along with the pose.
            cw.twins.update(twin.uuid, metadata={"alert_action": RACK_ALERT_ACTION})
        return twin, "created"
    fields: dict[str, Any] = {
        "position_x": x,
        "position_y": y,
        "position_z": z,
        "rotation_w": w,
        "rotation_x": qx,
        "rotation_y": qy,
        "rotation_z": qz,
        "fixed_base": True,
    }
    if alerts:
        fields["metadata"] = {"alert_action": RACK_ALERT_ACTION}
    cw.twins.update(twin.uuid, **fields)
    return twin, "updated"


def build_room(
    cw: Cyberwave,
    scene: Any,
    plan: list[dict[str, Any]],
    existing: dict[str, Any],
    alerts: bool = True,
) -> list[Any]:
    """Ensure every rack and its servers; returns the server twins in plan order."""
    server_twins: list[Any] = []
    for entry in plan:
        rack_twin, state = ensure_rack(cw, scene, entry, existing, alerts=alerts)
        new = sum(1 for server in entry["servers"] if server["name"] not in existing)
        # ensure_servers prints one line per server; the room reports per rack.
        with contextlib.redirect_stdout(io.StringIO()):
            twins = build_rack.ensure_servers(
                cw, scene, rack_twin, entry["servers"], existing, alerts=alerts
            )
        server_twins.extend(twins)
        x, y, z = entry["position"]
        print(
            f"{entry['name']}: {state} {rack_twin.uuid} at ({x:g}, {y:g}, {z:g}), "
            f"yaw {entry['yaw']:g} deg; servers {new} created, {len(twins) - new} re-docked"
        )
    return server_twins


def print_summary(report: dict[str, Any]) -> None:
    """Totals and the most frequent skipped reasons with a few examples each."""
    twins = report.get("twins", [])
    unbound = sum(1 for twin in twins if not twin.get("bound"))
    print(
        f"total: {len(twins)} servers, bound {report.get('bound', 0)}, "
        f"skipped {report.get('skipped', 0)}, servers without a bound slot {unbound}"
    )
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for twin in twins:
        for skipped in twin.get("skipped", []):
            reason = str(skipped.get("reason", "?"))
            counts[reason] += 1
            if len(examples.setdefault(reason, [])) < EXAMPLES_PER_REASON:
                examples[reason].append(f"{twin.get('twin_name')}/{skipped.get('slot')}")
    if not counts:
        return
    print(f"skipped by reason ({len(counts)} reasons):")
    for reason, count in counts.most_common(MAX_REASONS):
        print(f"  {count} x {reason}: {', '.join(examples[reason])}")
    if len(counts) > MAX_REASONS:
        print(f"  ... and {len(counts) - MAX_REASONS} more reasons (--full-report lists all)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--environment", required=True, help="environment uuid")
    parser.add_argument("--dataset", required=True, help="time-series dataset uuid to bind")
    parser.add_argument(
        "--layout",
        type=Path,
        default=DEFAULT_LAYOUT,
        help="markdown layout table Rack | X | Y | Nodes (default layouts/marconi100.md)",
    )
    parser.add_argument(
        "--racks",
        default="all",
        help="racks to build: all, 48, 46-48 or 0,17,48 (default all)",
    )
    parser.add_argument(
        "--tile",
        type=float,
        default=DEFAULT_TILE,
        help="metres per grid unit (default 0.6, one floor tile per rack width)",
    )
    parser.add_argument(
        "--facing",
        choices=sorted(FACING_YAW),
        default="-y",
        help="which way the rack fronts look (default -y; write --facing=-y)",
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
        "--alerts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "write metadata.alert_action: servers turn red, racks show a "
            "marker while a sensor is out of range (default on; --no-alerts skips)"
        ),
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="also print the per-server binding report of build_rack",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and call no API"
    )
    parser.add_argument("--base-url", help="backend URL (default: CYBERWAVE_BASE_URL)")
    parser.add_argument("--api-key", help="API key (default: CYBERWAVE_API_KEY)")
    args = parser.parse_args(argv)
    if args.tile <= 0:
        parser.error("--tile must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    template = build_rack.load_template(args.template)
    if not args.layout.is_file():
        raise SystemExit(f"layout {args.layout} not found")
    racks = parse_layout(args.layout)
    problems = layout_problems(racks)
    if problems:
        raise SystemExit(
            f"{args.layout}: layout disagrees with build_rack:\n  " + "\n  ".join(problems)
        )
    numbers = parse_racks(args.racks, [rack.number for rack in racks])
    plan = plan_room(racks, numbers, args.tile, args.facing)
    print_plan(args, template, plan)
    if args.dry_run:
        print("dry run: nothing was sent")
        return

    cw = Cyberwave(
        base_url=args.base_url, api_key=args.api_key, environment_id=args.environment
    )
    build_rack.require_asset(cw, RACK_ASSET)
    build_rack.require_asset(cw, SERVER_ASSET)
    scene = cw.get_scene(args.environment)
    existing = {
        twin.name: twin for twin in cw.twins.list(environment_id=args.environment)
    }

    print("\n== twins ==")
    server_twins = build_room(cw, scene, plan, existing, alerts=args.alerts)

    print("\n== bindings ==")
    report = cw.environments.auto_bind(
        args.environment,
        args.dataset,
        twin_ids=[twin.uuid for twin in server_twins],
        template=template,
        use_recorded=args.use_recorded,
    )
    if args.full_report:
        build_rack.print_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()

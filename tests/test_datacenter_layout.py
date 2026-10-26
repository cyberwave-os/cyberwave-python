"""build_datacenter.py: layout parsing, placement maths, rack selection, dry run
and the build flow against a fake client (replay-feed README §14)."""

import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "datacenter_rack"
LAYOUT = EXAMPLE_DIR / "layouts" / "marconi100.md"
ENV = "ba7e2341-b3a9-4fdf-97e9-808a234443a2"
DATASET = "a1985343-53c2-45b2-b366-09a0606dc649"
COS45 = round(math.cos(math.pi / 4), 6)


@pytest.fixture(scope="module")
def build_datacenter():
    # The script imports build_rack from its own folder; no __pycache__ there.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        for name in ("build_rack", "build_datacenter"):
            sys.modules.pop(name, None)
        module = importlib.import_module("build_datacenter")
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.dont_write_bytecode = previous
    yield module
    for name in ("build_rack", "build_datacenter"):
        sys.modules.pop(name, None)


def rotate(quaternion, vector):
    """Apply the unit quaternion (w, x, y, z) to a vector."""
    w, x, y, z = quaternion
    vx, vy, vz = vector
    # v' = v + 2 w (q x v) + 2 q x (q x v)
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return (vx + 2 * (w * cx + dx), vy + 2 * (w * cy + dy), vz + 2 * (w * cz + dz))


# --- layout ------------------------------------------------------------------


def test_bundled_layout_has_49_racks_in_three_rows(build_datacenter):
    racks = build_datacenter.parse_layout(LAYOUT)
    assert [rack.number for rack in racks] == list(range(49))
    assert sum(len(rack.nodes) for rack in racks) == 980
    for rack in racks:
        assert rack.nodes == tuple(range(20 * rack.number, 20 * rack.number + 20))
    assert {rack.y for rack in racks} == {2, 6, 10}
    assert (racks[0].x, racks[0].y) == (21, 2)
    assert (racks[17].x, racks[17].y) == (4, 2)
    assert (racks[18].x, racks[18].y) == (20, 6)
    assert (racks[48].x, racks[48].y) == (4, 10)
    assert build_datacenter.layout_problems(racks) == []


def test_layout_accepts_rows_with_and_without_edge_pipes(build_datacenter, tmp_path):
    layout = tmp_path / "layout.md"
    layout.write_text(
        "# Room\n\n|Rack|X|Y|Nodes (identifiers)|\n|---|-|-|---|\n"
        "| 0 | 3 | 2 | " + ", ".join(map(str, range(20))) + " |\n"
        "1|2|2|" + ", ".join(map(str, range(20, 40))) + "\n",
        encoding="utf-8",
    )
    racks = build_datacenter.parse_layout(layout)
    assert [(r.number, r.x, r.y) for r in racks] == [(0, 3, 2), (1, 2, 2)]
    assert racks[1].nodes == tuple(range(20, 40))


def test_layout_errors_are_reported_with_the_line(build_datacenter, tmp_path):
    layout = tmp_path / "layout.md"
    layout.write_text("|Rack|X|Y|Nodes|\n|-|-|-|-|\n0|3|2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="layout.md:3: expected 4 columns"):
        build_datacenter.parse_layout(layout)
    layout.write_text("|Rack|X|Y|Nodes|\n|-|-|-|-|\n0|a|2|1, 2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="layout.md:3: "):
        build_datacenter.parse_layout(layout)
    layout.write_text("|Rack|X|Y|Nodes|\n|-|-|-|-|\n0|3|2|1\n0|4|2|2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="rack 0 already defined on line 3"):
        build_datacenter.parse_layout(layout)
    layout.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no layout table"):
        build_datacenter.parse_layout(layout)


def test_nodes_that_disagree_with_build_rack_are_reported(build_datacenter):
    LayoutRack = build_datacenter.LayoutRack
    racks = [
        LayoutRack(0, 21, 2, tuple(range(20))),
        LayoutRack(1, 20, 2, tuple(range(20, 39)) + (40,)),
        LayoutRack(2, 20, 2, tuple(range(19, 39))),
    ]
    problems = build_datacenter.layout_problems(racks)
    assert problems == [
        "rack 1: layout lists nodes 20..40 (20), build_rack expects 20..39 (20)",
        "rack 2: layout lists nodes 19..38 (20), build_rack expects 40..59 (20)",
        "rack 2: node 19 also listed for rack 0",
        "rack 2: node 20 also listed for rack 1",
        "rack 2: node 21 also listed for rack 1",
        "rack 2: node 22 also listed for rack 1",
        "rack 2: node 23 also listed for rack 1",
        "rack 2: node 24 also listed for rack 1",
        "rack 2: node 25 also listed for rack 1",
        "rack 2: node 26 also listed for rack 1",
        "rack 2: node 27 also listed for rack 1",
        "rack 2: node 28 also listed for rack 1",
        "rack 2: node 29 also listed for rack 1",
        "rack 2: node 30 also listed for rack 1",
        "rack 2: node 31 also listed for rack 1",
        "rack 2: node 32 also listed for rack 1",
        "rack 2: node 33 also listed for rack 1",
        "rack 2: node 34 also listed for rack 1",
        "rack 2: node 35 also listed for rack 1",
        "rack 2: node 36 also listed for rack 1",
        "rack 2: node 37 also listed for rack 1",
        "rack 2: node 38 also listed for rack 1",
        "rack 2: tile (20, 2) also used by rack 1",
    ]


# --- placement ---------------------------------------------------------------


def test_rack_position_scales_grid_units_by_the_tile(build_datacenter):
    rack = build_datacenter.LayoutRack(48, 4, 10, tuple(range(960, 980)))
    assert build_datacenter.rack_position(rack, 0.6) == (2.4, 6.0, 0.0)
    assert build_datacenter.rack_position(rack, 1.0) == (4.0, 10.0, 0.0)
    assert build_datacenter.rack_position(rack, 0.5) == (2.0, 5.0, 0.0)


def test_facing_quaternion_turns_the_rack_front_to_the_requested_side(build_datacenter):
    minus_y = build_datacenter.facing_quaternion("-y")
    plus_y = build_datacenter.facing_quaternion("+y")
    assert minus_y == (COS45, 0.0, 0.0, -COS45)
    assert plus_y == (COS45, 0.0, 0.0, COS45)
    # The asset's front is its +x, its width runs along its y.
    assert rotate(minus_y, (1, 0, 0)) == pytest.approx((0, -1, 0), abs=1e-5)
    assert rotate(plus_y, (1, 0, 0)) == pytest.approx((0, 1, 0), abs=1e-5)
    assert rotate(minus_y, (0, 1, 0)) == pytest.approx((1, 0, 0), abs=1e-5)


def test_plan_room_uses_layout_order_names_and_build_rack_servers(build_datacenter):
    racks = build_datacenter.parse_layout(LAYOUT)
    plan = build_datacenter.plan_room(racks, [48, 0], 0.6, "-y")
    assert [entry["name"] for entry in plan] == ["rack-48", "rack-0"]
    assert plan[0]["position"] == (2.4, 6.0, 0.0)
    assert plan[0]["yaw"] == -90.0
    assert plan[0]["rotation"] == (COS45, 0.0, 0.0, -COS45)
    assert plan[1]["position"] == (12.6, 1.2, 0.0)
    assert [s["thing_id"] for s in plan[0]["servers"]][:2] == ["node=960", "node=961"]
    assert plan[0]["servers"][19]["name"] == "rack-48-s19"


# --- --racks -----------------------------------------------------------------


def test_parse_racks_accepts_numbers_ranges_and_lists(build_datacenter):
    available = list(range(49))
    assert build_datacenter.parse_racks("all", available) == available
    assert build_datacenter.parse_racks("48", available) == [48]
    assert build_datacenter.parse_racks("46-48", available) == [46, 47, 48]
    assert build_datacenter.parse_racks("48, 0,17", available) == [0, 17, 48]
    assert build_datacenter.parse_racks("1-2,2-3", available) == [1, 2, 3]
    with pytest.raises(SystemExit, match="not in the layout: 49, 50"):
        build_datacenter.parse_racks("48-50", available)
    with pytest.raises(SystemExit, match="empty range '3-1'"):
        build_datacenter.parse_racks("3-1", available)
    with pytest.raises(SystemExit, match="bad token 'x'"):
        build_datacenter.parse_racks("x", available)


# --- dry run -----------------------------------------------------------------


def test_dry_run_prints_the_plan_without_the_api(build_datacenter, capsys, monkeypatch):
    def no_client(*args, **kwargs):
        raise AssertionError("dry run must not build a client")

    monkeypatch.setattr(build_datacenter, "Cyberwave", no_client)

    build_datacenter.main(
        ["--environment", ENV, "--dataset", DATASET, "--racks", "46-48", "--dry-run"]
    )

    out = capsys.readouterr().out
    assert "tile 0.6 m, fronts facing -Y (yaw -90 deg)" in out
    assert (
        "rack-46: cyberwave/server-rack-42u at (3.6, 6, 0), yaw -90 deg, fixed base, "
        "20 x cyberwave/server-2u nodes 920..939" in out
    )
    assert "rack-48: cyberwave/server-rack-42u at (2.4, 6, 0), yaw -90 deg" in out
    assert "3 racks, 60 servers" in out
    assert "template (13 slots): inlet_temperature <- ipmi/ambient" in out
    assert "auto-bind: use_recorded on" in out
    assert out.rstrip().endswith("dry run: nothing was sent")


def test_dry_run_of_the_whole_room_with_other_tile_and_facing(
    build_datacenter, capsys, monkeypatch
):
    monkeypatch.setattr(build_datacenter, "Cyberwave", Mock())
    build_datacenter.main(
        ["--environment", ENV, "--dataset", DATASET, "--tile", "1", "--facing=+y", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert "49 racks, 980 servers" in out
    assert "rack-0: cyberwave/server-rack-42u at (21, 2, 0), yaw 90 deg" in out
    assert out.count("yaw 90 deg") == 49 + 1
    build_datacenter.Cyberwave.assert_not_called()


def test_layout_mismatch_stops_before_anything_is_sent(
    build_datacenter, tmp_path, monkeypatch
):
    layout = tmp_path / "layout.md"
    layout.write_text(
        "|Rack|X|Y|Nodes|\n|-|-|-|-|\n0|3|2|" + ", ".join(map(str, range(1, 21))) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_datacenter, "Cyberwave", Mock())
    with pytest.raises(SystemExit, match="rack 0: layout lists nodes 1..20"):
        build_datacenter.main(
            ["--environment", ENV, "--dataset", DATASET, "--layout", str(layout)]
        )
    build_datacenter.Cyberwave.assert_not_called()
    with pytest.raises(SystemExit, match="not found"):
        build_datacenter.main(
            ["--environment", ENV, "--dataset", DATASET, "--layout", str(tmp_path / "x.md")]
        )


# --- build flow against a fake client ----------------------------------------


def fake_client(existing_names):
    """A Cyberwave stand-in: twins by name, a scene that hands out twins."""
    cw = Mock()
    cw.assets.get_by_registry_id.side_effect = lambda rid: SimpleNamespace(uuid=rid)
    cw.twins.list.return_value = [
        SimpleNamespace(uuid=f"uuid-{name}", name=name) for name in existing_names
    ]
    scene = Mock()
    scene.add_twin.side_effect = lambda asset, name, **kw: SimpleNamespace(
        uuid=f"uuid-{name}", name=name
    )
    cw.get_scene.return_value = scene
    return cw, scene


def test_build_reuses_twins_rotates_racks_and_binds_once(
    build_datacenter, capsys, monkeypatch
):
    cw, scene = fake_client(["rack-48", "rack-48-s00", "rack-48-s19"])
    cw.environments.auto_bind.return_value = {
        "twins": [
            {"twin_name": "rack-47-s00", "thing_id": "node=940",
             "bound": [{"slot": "inlet_temperature", "channel_id": "ipmi/ambient/node=940"}],
             "skipped": [{"slot": "fan_speed", "reason": "channel not in dataset"}]},
            {"twin_name": "rack-48-s00", "thing_id": "node=960", "bound": [],
             "skipped": [{"slot": "inlet_temperature", "reason": "already bound"},
                         {"slot": "fan_speed", "reason": "channel not in dataset"}]},
        ],
        "bound": 1,
        "skipped": 3,
    }
    monkeypatch.setattr(build_datacenter, "Cyberwave", Mock(return_value=cw))

    build_datacenter.main(
        ["--environment", ENV, "--dataset", DATASET, "--racks", "47-48",
         "--base-url", "http://localhost:8000", "--api-key", "test-api-key"]
    )

    # Rack 47 is new: created with position and (x, y, z, w) orientation.
    rack_calls = [c for c in scene.add_twin.call_args_list if c.args[0] == "cyberwave/server-rack-42u"]
    assert len(rack_calls) == 1
    assert rack_calls[0].kwargs == {
        "name": "rack-47",
        "position": [3.0, 6.0, 0.0],
        "orientation": [0.0, 0.0, -COS45, COS45],
        "fixed_base": True,
    }
    # Rack 48 exists: position and rotation updated in place, the alert marker
    # riding in the same call; the new rack 47 gets the marker in its own call.
    cw.twins.update.assert_any_call(
        "uuid-rack-48",
        position_x=2.4, position_y=6.0, position_z=0.0,
        rotation_w=COS45, rotation_x=0.0, rotation_y=0.0, rotation_z=-COS45,
        fixed_base=True,
        metadata={"alert_action": {"kind": "marker"}},
    )
    cw.twins.update.assert_any_call("uuid-rack-47", metadata={"alert_action": {"kind": "marker"}})
    # 40 servers: 38 created, the two existing ones re-docked; all get a thing_id
    # and the red alert colour in one metadata write each.
    server_calls = [c for c in scene.add_twin.call_args_list if c.args[0] == "cyberwave/server-2u"]
    assert len(server_calls) == 38
    assert scene.dock.call_count == 2
    server_updates = [
        c for c in cw.twins.update.call_args_list if "thing_id" in (c.kwargs.get("metadata") or {})
    ]
    assert len(server_updates) == 40
    assert cw.twins.update.call_count == 42
    assert all(
        c.kwargs["metadata"]["alert_action"] == {"kind": "color", "color": "#ef4444"} for c in server_updates
    )
    thing_ids = {(c.args[0], c.kwargs["metadata"]["thing_id"]) for c in server_updates}
    assert ("uuid-rack-48-s00", "node=960") in thing_ids
    assert ("uuid-rack-47-s19", "node=959") in thing_ids
    # One auto_bind over every server of the run.
    cw.environments.auto_bind.assert_called_once()
    bind = cw.environments.auto_bind.call_args
    assert bind.args == (ENV, DATASET)
    assert len(bind.kwargs["twin_ids"]) == 40
    assert bind.kwargs["twin_ids"][0] == "uuid-rack-47-s00"
    assert bind.kwargs["template"]["inlet_temperature"] == "ipmi/ambient"
    assert bind.kwargs["use_recorded"] is True

    out = capsys.readouterr().out
    assert "rack-47: created uuid-rack-47 at (3, 6, 0), yaw -90 deg; servers 20 created, 0 re-docked" in out
    assert "rack-48: updated uuid-rack-48 at (2.4, 6, 0), yaw -90 deg; servers 18 created, 2 re-docked" in out
    assert "rack-47-s00: created" not in out  # no per-server lines
    assert "total: 2 servers, bound 1, skipped 3, servers without a bound slot 1" in out
    assert "skipped by reason (2 reasons):" in out
    assert "  2 x channel not in dataset: rack-47-s00/fan_speed, rack-48-s00/fan_speed" in out
    assert "  1 x already bound: rack-48-s00/inlet_temperature" in out


def test_full_report_adds_the_per_server_lines(build_datacenter, capsys, monkeypatch):
    cw, _ = fake_client([])
    cw.environments.auto_bind.return_value = {
        "twins": [{"twin_name": "rack-48-s00", "thing_id": "node=960", "bound": [], "skipped": []}],
        "bound": 0,
        "skipped": 0,
    }
    monkeypatch.setattr(build_datacenter, "Cyberwave", Mock(return_value=cw))
    build_datacenter.main(
        ["--environment", ENV, "--dataset", DATASET, "--racks", "48", "--full-report"]
    )
    out = capsys.readouterr().out
    assert "rack-48-s00 (node=960): bound 0, skipped 0" in out
    assert "total: 1 servers, bound 0, skipped 0" in out
    assert "skipped by reason" not in out


def test_missing_asset_stops_before_twins(build_datacenter, monkeypatch):
    cw, scene = fake_client([])
    cw.assets.get_by_registry_id.side_effect = lambda rid: None
    monkeypatch.setattr(build_datacenter, "Cyberwave", Mock(return_value=cw))
    with pytest.raises(SystemExit, match="cyberwave/server-rack-42u not found"):
        build_datacenter.main(["--environment", ENV, "--dataset", DATASET, "--racks", "48"])
    scene.add_twin.assert_not_called()

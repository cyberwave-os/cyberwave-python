# Data-center rack

`build_rack.py` builds one rack of the data-center demo from the catalog assets
(`cyberwave/server-rack-42u`, `cyberwave/server-2u`) and binds its servers to a
time-series dataset (contract `cyberwave-backend/src/lib/replay_feed/README.md` §12–§14).

It is idempotent by twin name. Running it again re-docks the servers, rewrites their
`thing_id` and re-runs the binding (slots already bound are reported as skipped).

| Step | What happens |
|---|---|
| twins | `rack-<n>` from `cyberwave/server-rack-42u` at `--position` (default origin), fixed base. `rack-<n>-s00` … `s19` from `cyberwave/server-2u`, docked to `server_rack_42u_link` at x = 0.105, y = 0, z = 0.10 + 0.04445 × (2 i + 1) (bottom to top), fixed base. Each server gets `metadata.thing_id = node=<20 n + i>`, the Marconi100 numbering (rack 48 holds nodes 960 … 979). |
| bindings | `cw.environments.auto_bind(...)` with the slot template: for every server and every slot of the template the channel `<signal>/<thing_id>` of the dataset is attached when its catalog has it (`use_recorded` on by default). The report lists bound and skipped slots per server. |
| kit (`--as-kit`) | Registers `cyberwave/server-rack-42u-20x2u`: the rack as root and the twenty servers with the same offsets, so the assembly can be dropped from the catalog (`cw.asset_kits.instantiate(...)` or the GUI). Re-running updates the kit. |

Templates map sensor slots of `cyberwave/server-2u` to dataset signals. `templates/marconi100.json`
is the bundled one (`--template marconi100`; a file path works too) and is the same mapping as
`slots` in `../timeseries_replay/recipes/marconi100.json`, which the import stores as the dataset's
`slot_template`:

| Slot | Signal | Slot | Signal |
|---|---|---|---|
| inlet_temperature | ipmi/ambient | gpu0_temperature | ipmi/gpu0_core_temp |
| total_power | ipmi/total_power | gpu1_temperature | ipmi/gpu1_core_temp |
| cpu0_power | ipmi/p0_power | gpu2_temperature | ipmi/gpu3_core_temp |
| cpu1_power | ipmi/p1_power | gpu3_temperature | ipmi/gpu4_core_temp |
| cpu0_temperature | ipmi/p0_vdd_temp | fan_speed | ipmi/fan0_0 |
| cpu1_temperature | ipmi/p1_vdd_temp | psu0_input_power | ipmi/ps0_input_power |
| | | psu1_input_power | ipmi/ps1_input_power |

Marconi numbers its four GPUs 0, 1, 3, 4; `outlet_temperature` stays unbound.

Install: `pip install -r requirements.txt` (the SDK from this repo: `pip install -e ../..`).
The script falls back to `CYBERWAVE_BASE_URL` and `CYBERWAVE_API_KEY`; the flags override them.

## Local stack

Both assets must exist in the catalog: `cyberwave/server-2u` is seeded with
`docker compose -f local.yml run --rm django python manage.py seed_asset_datacenter --assets server_2u`
(in `cyberwave-backend`), `cyberwave/server-rack-42u` comes from the catalog. The dataset is a
time-series import (see `../timeseries_replay/README.md`).

Print the plan without touching the API:

```bash
python build_rack.py --environment ba7e2341-b3a9-4fdf-97e9-808a234443a2 --rack 48 \
  --dataset a1985343-53c2-45b2-b366-09a0606dc649 --dry-run
```

Build rack 48 against the Marconi February dataset:

```bash
python build_rack.py --environment ba7e2341-b3a9-4fdf-97e9-808a234443a2 --rack 48 \
  --dataset a1985343-53c2-45b2-b366-09a0606dc649 --template templates/marconi100.json \
  --base-url http://localhost:8000 --api-key test-api-key
```

Add `--as-kit` to register the kit as well, `--no-use-recorded` to attach the channels for
Replay mode only (the devices keep feeding Live and Simulate), `--position 2 0 0` to place the rack.

Then open the environment in **Live** mode and start the **Recorded data feed** (or from Python:
`cw.replay_feeds.start(env, "2022-02-08T05:00:00Z", end="2022-02-08T06:00:00Z", speed=60)`);
`cw.environments.sensor_sources(env)` lists what is bound.

## Whole room: `build_datacenter.py`

`build_datacenter.py` builds every rack of a room from a layout file with the same
twins, offsets, `thing_id`s and binding as `build_rack.py` (it imports that module), so
the two scripts stay idempotent together: `rack-<n>` and `rack-<n>-s00` … `s19` are reused
by name, servers are re-docked, slots already bound are reported as skipped.

| Step | What happens |
|---|---|
| layout | `--layout` (default `layouts/marconi100.md`, a copy of the Marconi100 spatial distribution) is a markdown table `Rack \| X \| Y \| Nodes (identifiers)`. X and Y are floor-tile grid coordinates; the node list is the servers bottom to top. The list must equal `build_rack.node_number` (20 n … 20 n + 19); a mismatch, a node in two racks or two racks on one tile stops the run before anything is sent. |
| racks | One `rack-<n>` per selected row at `(X × tile, Y × tile, 0)`, fixed base, turned by a yaw about z so the racks of a row stand side by side along the room's X axis. Existing racks get their position and rotation updated (`cw.twins.update` with `position_*`/`rotation_*`). One progress line per rack. |
| servers | The twenty servers of each rack through `build_rack.ensure_servers` (same dock offsets, `thing_id = node=<id>`). |
| bindings | One `cw.environments.auto_bind(...)` over all servers of the run, then totals and the most frequent skipped reasons with a few examples each (`--full-report` adds the per-server lines). |

Assumptions, both flags:

- `--tile` (default `0.6`): metres per grid unit. The rack is 0.6 m wide, so one tile per
  rack puts the racks of a row shoulder to shoulder; rows at Y = 2, 6, 10 end up 2.4 m apart.
- `--facing` (default `-y`): the layout has no front orientation. The rack asset's front
  (glass door) is its +x and its width runs along its y, so a yaw of -90° about z
  (quaternion w, x, y, z = cos 45°, 0, 0, -sin 45°) makes the fronts look along -Y; `+y`
  uses +90°. Write it as `--facing=-y`: a bare `-y` reads as a flag.

`--racks` selects a subset while testing (`48`, `46-48`, `0,17,48`; default `all`),
`--dry-run` prints one line per rack (position, yaw, nodes), the counts and the template
without calling the API.

Local stack, the three racks of the last row first, then the whole room:

```bash
cd cyberwave-sdks/cyberwave-python/examples/datacenter_rack
python -B build_datacenter.py --environment <uuid> --dataset <uuid> \
  --template templates/marconi100.json --base-url http://localhost:8000 --api-key test-api-key \
  --racks 46-48 --dry-run
python -B build_datacenter.py --environment <uuid> --dataset <uuid> \
  --template templates/marconi100.json --base-url http://localhost:8000 --api-key test-api-key
```

`-B` keeps the examples folder free of `__pycache__`. A full run creates 49 + 980 twins and
one auto-bind of 980 × 13 slots; re-running only re-docks and reports the slots as skipped.

# Time-series replay

Tools around the time-series dataset contract
(`cyberwave-backend/src/lib/data/dataset/timeseries/README.md`): pack a recorded
dataset for upload, build a demo scene whose twins are bound to recorded channels,
and play those channels into the twins as live metrics.

For normal use the script player is not needed any more: switch on **Use recorded
data** on a twin's bindings (Edit mode, *Data sources* section) and start the
**Recorded data feed** panel in Live or Simulate. The feed runs on the backend
(replay-feed contract `cyberwave-backend/src/lib/replay_feed/README.md` §3–§4, user
docs `docs-mintlify/feature-reference/environment-editor/replay-feed.mdx`) and
publishes the same messages as `player.py`. The script stays for offline runs
(`--from-folder`, `--dry-run`) and for playing from a machine of your own.

| Tool | What it does |
|---|---|
| `make_zip.py` | Packs a data folder plus a recipe into an upload zip: recipe at the root as `cyberwave-timeseries.json`, the folder stored under its own name, Parquet files stored without recompression. |
| `make_scene.py` | Creates the Marconi100 rack-48 demo scene: one `cyberwave/equipment_rack` procedural primitive and three `cyberwave/generic_box` server twins with sensor slots, v2 bindings (`use_recorded: true`) and `supported_simulation_backends`. Idempotent: re-running updates the existing rack and twins and rewrites legacy bindings in the v2 shape. |
| `player.py` | Script player: reads the twins' bindings, fetches rows window by window and publishes `cyberwave/twin/{uuid}/metrics` messages at `speed ×` real time. `--from-folder` plays a local Parquet folder without the API. |
| `local_reader.py` | Helper for `--from-folder`: long-layout Parquet + recipe reader. |
| `recipes/marconi100.json` | Reference recipe for the CINECA Marconi100 ExaMon export. |
| `bindings/marconi100_rack48.json` | Bindings file for the three demo twins (used by the offline dry run). |

A whole rack (one `cyberwave/server-rack-42u` with twenty docked `cyberwave/server-2u` twins,
bound through the recipe's `slots` template and optionally registered as a catalog kit) is
built by `../datacenter_rack/build_rack.py`, see `../datacenter_rack/README.md`.

Install: `pip install -r requirements.txt` (the SDK itself from this repo: `pip install -e ../..`).
`pyarrow` and `pandas` are only needed for `--from-folder`; `requests` only for playback from the API.

All tools fall back to `CYBERWAVE_BASE_URL`, `CYBERWAVE_API_KEY`, `CYBERWAVE_MQTT_HOST`,
`CYBERWAVE_MQTT_PORT` and `CYBERWAVE_MQTT_TOPIC_PREFIX`; the flags below override them.

## Marconi100 demo on the local stack

`<data-folder>` is a folder with the ExaMon layout (the rack-48 subset or a full month):

```
year_month=22-02/plugin=ipmi_pub/metric=ambient/a_0.parquet
year_month=22-02/plugin=ipmi_pub/metric=total_power/a_0.parquet
...
```

1. Build the upload zip:

   ```bash
   python make_zip.py <data-folder> recipes/marconi100.json marconi100_rack48_2022-02_timeseries.zip
   ```

2. Upload it in the UI: *Datasets → Upload Dataset → Upload zip*. Note the dataset uuid.

3. Run the import task by hand (no local worker consumes the `tfds` queue, contract §10):

   ```bash
   cd cyberwave-backend && docker compose -f local.yml exec django python manage.py shell -c \
     "from src.lib.data.dataset.ingest.tasks import process_imported_dataset_task as t; t.run(dataset_uuid='<dataset-uuid>')"
   ```

   The dataset card then shows the kind "Time series"; the dataset's Playground tab shows the
   time range, the counts and the first channels.

4. Create the scene. The three twins get bindings for `ipmi/ambient` (inlet temperature) and
   `ipmi/total_power` (power) of nodes 960 / 970 / 979 with **Use recorded data** switched on;
   `fan_speed` has no binding and stays with the environment:

   ```bash
   python make_scene.py --environment <env-uuid> --dataset <dataset-uuid> \
     --base-url http://localhost:8000 --api-key test-api-key
   ```

   It prints the twin uuids. A fresh local stack needs the box asset first:
   `docker compose -f local.yml run --rm django python manage.py seed_asset_cyberwave_catalog --assets box`.

5. Play the data. Open the environment in **Live** mode: the right panel shows the
   **Recorded data feed** as soon as one sensor uses recorded data. Set the start to
   `2022-02-08 05:00:00` (UTC) and the speed to 60×, press **Play**, then select a server
   twin: the **Live values** section shows inlet temperature and power with the replay
   badge. The feed stops by itself after two hours of wall-clock time. On the local stack
   it needs the Celery worker (the `replay_feed` queue) running.

   The script plays the same hour at 60× (one message per twin per second):

   ```bash
   python player.py --environment <env-uuid> \
     --start 2022-02-08T05:00:00Z --end 2022-02-08T06:00:00Z --speed 60 \
     --base-url http://localhost:8000 --api-key test-api-key \
     --mqtt-host localhost --mqtt-port 1883 --topic-prefix local
   ```

   `--source-type sim` marks the messages as simulation provenance, `--loop` restarts at
   `--start`, `--run-id` names the run. The script's messages carry the recorded-origin tag,
   so they pass the backend's hold-back like the feed's own (a sensor with "Use recorded data"
   on never takes device values). Do not run the script while a feed is playing: two recorded
   sources would then write the same sensors.

## Player details

- Bindings: every twin of the environment whose `metadata.data_sources[slot]` has a channel
  attached and `use_recorded: true` (legacy entries with `source: "replay"` read the same),
  grouped by dataset. `--all-attached` plays every attached binding regardless of the switch;
  `--dataset` keeps only the bindings of that dataset.
- Rows: `GET /api/v1/datasets/{uuid}/timeseries/rows` (contract §5) in windows of 60 ticks
  of playback (`tick × 60 × speed` data seconds). The next window is fetched while the
  current one plays; a `truncated` response halves the window.
- Messages (contract §7): per tick one message per twin with only the slots that got a new
  reading, a wall-clock float `timestamp`, `metrics.readings` and `metrics.replay`. Never
  more than one message per second per twin (`--tick` is at least 1). The messages carry no
  feed id, so Cyberwave stores them as injected values of the run without a feed.

## Offline dry run

Plays straight from a local folder and prints the messages instead of publishing. No
backend, broker or API key is needed:

```bash
python player.py --from-folder <data-folder> --recipe recipes/marconi100.json \
  --bindings bindings/marconi100_rack48.json \
  --start 2022-02-08T05:00:00Z --end 2022-02-08T05:03:00Z --speed 600 --dry-run
```

A bindings file maps twin names (or uuids) to slot → channel:

```json
{"r48-s00-bottom": {"inlet_temperature": "ipmi/ambient/node=960", "power": "ipmi/total_power/node=960"}}
```

With `--environment` the names are resolved to twin uuids and, without `--dry-run`, the
messages are published; `--from-folder` also works with bindings read from the twins. The
local reader supports Parquet files, the `long` layout and signal templates over path fields;
`scale` and `offset` from the recipe are applied and units come from its `signals` annotations.

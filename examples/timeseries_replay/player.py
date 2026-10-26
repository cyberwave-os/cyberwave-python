"""
Time-series replay — plays recorded channels into twins as live metrics over MQTT.

For normal use the backend's recorded data feed (the "Recorded data feed"
panel in Live and Simulate) replaces this script; it stays for offline runs
and --from-folder. Bindings come from the environment's twins
(metadata.data_sources entries whose "Use recorded data" switch is on, or
every attached entry with --all-attached) or from a --bindings JSON file.
Rows come from the dataset API window by window, or with --from-folder
straight from a local Parquet folder. Each tick publishes one metrics message
per twin (contract section 7).

Usage:
    python player.py --environment <uuid> --start 2022-02-08T05:00:00Z --end 2022-02-08T06:00:00Z
    python player.py --from-folder <folder> --recipe recipes/marconi100.json --bindings b.json
        --start ... --end ... --speed 600 --dry-run

Requirements:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cyberwave import Cyberwave

try:
    import requests  # only needed when rows come from the API
except ImportError:
    requests = None

ROWS_LIMIT = 100_000
CHANNELS_PER_REQUEST = 200


def parse_time(text: str) -> float:
    """ISO-8601 to epoch seconds; naive values are UTC."""
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def is_uuid(text: str) -> bool:
    try:
        uuid.UUID(text)
    except ValueError:
        return False
    return True


@dataclass
class Binding:
    twin: str  # twin uuid (the bindings-file key when no environment is given)
    twin_name: str
    slot: str
    channel: str
    dataset: str | None
    unit: str = ""


class ApiRows:
    """Rows from GET /api/v1/datasets/{uuid}/timeseries/rows (contract section 5)."""

    def __init__(self, base_url: str, api_key: str, dataset: str, channels, workspace_id=None) -> None:
        if requests is None:
            raise SystemExit("playing from the API needs the requests package: pip install requests")
        self.dataset = dataset
        self.url = f"{base_url.rstrip('/')}/api/v1/datasets/{dataset}/timeseries/rows"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.channels = sorted(channels)
        self.workspace_id = workspace_id

    def fetch(self, start: float, end: float) -> tuple[list[tuple[float, str, float]], bool]:
        rows: list[tuple[float, str, float]] = []
        truncated = False
        for i in range(0, len(self.channels), CHANNELS_PER_REQUEST):
            params = [("channel", c) for c in self.channels[i : i + CHANNELS_PER_REQUEST]]
            params += [("start", iso(start)), ("end", iso(end)), ("limit", ROWS_LIMIT)]
            if self.workspace_id:
                params.append(("workspace_uuid", self.workspace_id))
            response = requests.get(self.url, params=params, headers=self.headers, timeout=120)
            if not response.ok:
                raise RuntimeError(
                    f"rows request for dataset {self.dataset} failed: HTTP {response.status_code} {response.text[:200]}"
                )
            body = response.json()
            rows += [(parse_time(r["time"]), r["channel"], float(r["value"])) for r in body["rows"]]
            truncated = truncated or bool(body.get("truncated"))
        rows.sort()
        return rows, truncated


class WindowBuffer:
    """Rows of one source in window order, with the next window fetched in the background."""

    def __init__(self, source, start: float, end: float, window: float, min_window: float) -> None:
        self.source = source
        self.end = end
        self.window = window
        self.min_window = min_window
        self.rows: list[tuple[float, str, float]] = []
        self.loaded_to = start
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = self._submit(start)

    def _submit(self, w_start: float):
        if w_start >= self.end:
            return None
        w_end = min(w_start + self.window, self.end)
        return w_start, w_end, self.pool.submit(self.source.fetch, w_start, w_end)

    def _finish_window(self) -> None:
        w_start, w_end, future = self.pending
        rows, truncated = future.result()
        while truncated and (w_end - w_start) > self.min_window:
            # The API hit its row limit: halve the window and refetch.
            self.window = (w_end - w_start) / 2
            w_end = w_start + self.window
            rows, truncated = self.source.fetch(w_start, w_end)
        if truncated:
            print(f"warning: {iso(w_start)}..{iso(w_end)} still truncated at {len(rows)} rows (too many channels per tick)")
        self.rows += rows
        self.loaded_to = w_end
        self.pending = self._submit(w_end)

    def take(self, until: float) -> list[tuple[float, str, float]]:
        """Rows with time < until, in order; waits for windows still loading."""
        while self.loaded_to < until and self.pending is not None:
            self._finish_window()
        n = 0
        while n < len(self.rows) and self.rows[n][0] < until:
            n += 1
        taken, self.rows = self.rows[:n], self.rows[n:]
        return taken

    def close(self) -> None:
        self.pool.shutdown(wait=False)


def uses_recorded(entry: dict) -> bool:
    """The binding's "Use recorded data" switch; legacy entries carry source "replay" instead."""
    use_recorded = entry.get("use_recorded")
    if isinstance(use_recorded, bool):
        return use_recorded
    return entry.get("source") == "replay"


def bindings_from_twins(cw: Cyberwave, environment: str, dataset: str | None, all_attached: bool = False) -> list[Binding]:
    """Attached bindings with the switch on; with all_attached every attached binding."""
    bindings = []
    for twin in cw.twins.list(environment_id=environment):
        for slot, entry in ((twin.metadata or {}).get("data_sources") or {}).items():
            if not isinstance(entry, dict) or not entry.get("channel_id"):
                continue  # no channel attached: the environment feeds the slot
            if not all_attached and not uses_recorded(entry):
                continue
            if dataset and entry.get("dataset_uuid") != dataset:
                continue
            bindings.append(
                Binding(twin.uuid, twin.name, slot, entry["channel_id"], entry.get("dataset_uuid"), entry.get("unit") or "")
            )
    return bindings


def bindings_from_file(path: str, cw: Cyberwave | None, environment: str | None, dataset: str | None) -> list[Binding]:
    """{"<twin uuid or name>": {"<slot>": "<channel_id>"}}; names resolve through the environment."""
    spec = json.loads(Path(path).read_text())
    uuids = {}
    if cw is not None and environment:
        for twin in cw.twins.list(environment_id=environment):
            uuids[twin.name] = uuids[twin.uuid] = twin.uuid
    bindings = []
    for key, slots in spec.items():
        if uuids and key not in uuids:
            raise SystemExit(f"bindings: twin {key!r} is not in environment {environment}")
        for slot, channel in slots.items():
            bindings.append(Binding(uuids.get(key, key), key, slot, channel, dataset))
    return bindings


def drop_unusable_datasets(cw: Cyberwave, bindings: list[Binding]) -> list[Binding]:
    """Skip, with a warning, bindings whose dataset the API cannot serve.

    A binding keeps pointing at its dataset after that dataset was deleted or
    while its import has not finished; those would otherwise abort the run.
    """
    if requests is None:
        return bindings
    base_url = cw.config.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {cw.config.api_key}"}
    params = {"workspace_uuid": cw.config.workspace_id} if cw.config.workspace_id else None
    problems: dict[str, str | None] = {}
    for dataset in sorted({b.dataset for b in bindings if b.dataset}):
        response = requests.get(f"{base_url}/api/v1/datasets/{dataset}", params=params, headers=headers, timeout=60)
        if response.status_code == 404:
            problems[dataset] = "not found (deleted?)"
        elif not response.ok:
            problems[dataset] = f"HTTP {response.status_code}"
        elif not ((response.json().get("metadata") or {}).get("timeseries") or {}).get("catalog_path"):
            problems[dataset] = "has no time-series data (import not finished?)"
        else:
            problems[dataset] = None
    kept = []
    for b in bindings:
        problem = problems.get(b.dataset) if b.dataset else None
        if problem:
            print(f"warning: {b.twin_name}.{b.slot} <- {b.channel}: dataset {b.dataset} {problem}; skipped")
        else:
            kept.append(b)
    return kept


def build_sources(args, cw: Cyberwave | None, bindings: list[Binding]) -> dict:
    """Row sources keyed by dataset (one local source under None with --from-folder)."""
    if args.from_folder:
        from local_reader import LocalRows  # needs pyarrow and pandas; the API path does not

        source = LocalRows(args.from_folder, args.recipe, {b.channel for b in bindings})
        for b in bindings:
            b.unit = b.unit or source.units[b.channel]
        return {None: source}
    channels: dict[str, set] = {}
    for b in bindings:
        if not b.dataset:
            raise SystemExit(f"{b.twin_name}/{b.slot}: binding has no dataset_uuid (pass --dataset)")
        channels.setdefault(b.dataset, set()).add(b.channel)
    return {
        dataset: ApiRows(cw.config.base_url, cw.config.api_key, dataset, chans, cw.config.workspace_id)
        for dataset, chans in channels.items()
    }


def play(args, bindings: list[Binding], sources: dict, publish, prefix: str) -> None:
    start, end = parse_time(args.start), parse_time(args.end)
    tick_seconds = args.tick * args.speed  # data seconds per tick
    window = tick_seconds * 60  # each fetch covers 60 ticks of playback
    run_id = args.run_id or time.strftime("replay-%Y%m%d-%H%M%S")
    by_twin: dict[str, list[Binding]] = {}
    by_channel: dict[tuple, list[Binding]] = {}
    for b in bindings:
        by_twin.setdefault(b.twin, []).append(b)
        by_channel.setdefault((None if args.from_folder else b.dataset, b.channel), []).append(b)

    while True:
        buffers = {key: WindowBuffer(source, start, end, window, tick_seconds) for key, source in sources.items()}
        last_at: dict[tuple[str, str], float] = {}
        clock, tick_no, next_wall = start, 0, time.monotonic()
        while clock < end:
            tick_end = min(clock + tick_seconds, end)
            latest: dict[tuple[str, str], tuple[float, float]] = {}
            for key, buffer in buffers.items():
                for at, channel, value in buffer.take(tick_end):
                    for b in by_channel.get((key, channel), ()):
                        latest[(b.twin, b.slot)] = (at, value)  # rows are time ordered: last wins
            tick_no += 1
            sent = readings = 0
            for twin, twin_bindings in by_twin.items():
                new = {}
                for b in twin_bindings:
                    item = latest.get((twin, b.slot))
                    if item is None or last_at.get((twin, b.slot)) == item[0]:
                        continue
                    last_at[(twin, b.slot)] = item[0]
                    new[b.slot] = {"value": item[1], "unit": b.unit, "at": item[0], "channel_id": b.channel}
                if not new:
                    continue
                message = {
                    "source_type": args.source_type,
                    "timestamp": time.time(),
                    "metrics": {
                        "readings": new,
                        "replay": {
                            "run_id": run_id,
                            "dataset_uuid": twin_bindings[0].dataset,
                            "data_time": iso(tick_end),
                            "speed": float(args.speed),
                        },
                    },
                }
                if publish(f"{prefix}cyberwave/twin/{twin}/metrics", message):
                    sent += 1
                    readings += len(new)
            published_at = time.monotonic()
            print(f"tick {tick_no}  data {iso(tick_end)}  {sent} messages / {readings} readings")
            clock = tick_end
            if clock >= end:
                break
            next_wall += args.tick
            # Keep at least one tick between messages, even after a slow fetch.
            time.sleep(max(0.0, max(next_wall, published_at + args.tick) - time.monotonic()))
        for buffer in buffers.values():
            buffer.close()
        if not args.loop:
            break
        print(f"end of range, restarting at {iso(start)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--environment", help="environment uuid; bindings come from its twins")
    parser.add_argument("--dataset", help="only play bindings of this dataset (the dataset of a --bindings file)")
    parser.add_argument(
        "--all-attached",
        action="store_true",
        help='play every attached binding, not only those with "Use recorded data" switched on',
    )
    parser.add_argument("--start", required=True, help="data time to start at (ISO-8601)")
    parser.add_argument("--end", required=True, help="data time to stop at (ISO-8601, exclusive)")
    parser.add_argument("--speed", type=float, default=60.0, help="data seconds per wall second (default 60)")
    parser.add_argument("--tick", type=float, default=1.0, help="wall seconds between messages (default 1, min 1)")
    parser.add_argument("--source-type", choices=["edge", "sim"], default="edge")
    parser.add_argument("--run-id", help="replay run id (default: generated)")
    parser.add_argument("--loop", action="store_true", help="restart at --start when --end is reached")
    parser.add_argument("--from-folder", help="read rows from this local Parquet folder instead of the API")
    parser.add_argument("--recipe", help="recipe JSON for --from-folder")
    parser.add_argument("--bindings", help='JSON file {"<twin uuid or name>": {"<slot>": "<channel_id>"}}')
    parser.add_argument("--dry-run", action="store_true", help="print the messages instead of publishing")
    parser.add_argument("--base-url", help="backend URL (default: CYBERWAVE_BASE_URL)")
    parser.add_argument("--api-key", help="API key (default: CYBERWAVE_API_KEY)")
    parser.add_argument("--mqtt-host", help="MQTT broker host (default: CYBERWAVE_MQTT_HOST)")
    parser.add_argument("--mqtt-port", type=int, help="MQTT broker port (default: CYBERWAVE_MQTT_PORT)")
    parser.add_argument("--mqtt-tls", action="store_true", help="use TLS for MQTT (implied by port 8883)")
    parser.add_argument("--topic-prefix", help="MQTT topic prefix, e.g. local (default: CYBERWAVE_MQTT_TOPIC_PREFIX)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # tick lines show up promptly when redirected to a file
    if args.from_folder and not args.recipe:
        parser.error("--from-folder needs --recipe")
    if not args.environment and not args.bindings:
        parser.error("--environment or --bindings is required")
    if args.tick < 1 or args.speed <= 0:
        parser.error("--tick must be >= 1 second and --speed > 0")
    if parse_time(args.end) <= parse_time(args.start):
        parser.error("--end must be after --start")

    # A fully offline dry run (bindings file, local folder, no environment) needs no client.
    offline = args.dry_run and args.bindings and args.from_folder and not args.environment
    cw = None
    if not offline:
        cw = Cyberwave(
            base_url=args.base_url,
            api_key=args.api_key,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            mqtt_use_tls=args.mqtt_tls,
            topic_prefix=args.topic_prefix,
            source_type=args.source_type,
            environment_id=args.environment,
        )

    if args.bindings:
        bindings = bindings_from_file(args.bindings, cw, args.environment, args.dataset)
    else:
        bindings = bindings_from_twins(cw, args.environment, args.dataset, args.all_attached)
    if cw is not None and not args.from_folder:
        bindings = drop_unusable_datasets(cw, bindings)
    if not bindings:
        hint = ""
        if not args.bindings and not args.all_attached:
            hint = ' with "Use recorded data" switched on (--all-attached plays every attached binding)'
        raise SystemExit(f"no replay bindings found{hint}")
    if not args.dry_run and not args.environment:
        # Topics need twin uuids; names only resolve through --environment.
        names = sorted({b.twin for b in bindings if not is_uuid(b.twin)})
        if names:
            raise SystemExit(f"bindings: pass --environment to resolve twin names to uuids: {', '.join(names)}")
    sources = build_sources(args, cw, bindings)
    for b in bindings:
        print(f"{b.twin_name}.{b.slot} <- {b.channel}" + (f" ({b.unit})" if b.unit else ""))

    if args.dry_run:
        prefix = cw.mqtt.topic_prefix if cw else (args.topic_prefix or "")

        def publish(topic: str, message: dict) -> bool:
            print(topic, json.dumps(message, ensure_ascii=False))
            return True

    else:
        cw.mqtt.connect()
        prefix = cw.mqtt.topic_prefix
        publish = cw.mqtt.publish

    try:
        play(args, bindings, sources, publish, prefix)
    except KeyboardInterrupt:
        print("stopped")
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc
    finally:
        if cw is not None:
            cw.disconnect()


if __name__ == "__main__":
    main()

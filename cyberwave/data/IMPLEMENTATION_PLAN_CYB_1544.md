# CYB-1544: Implementation Plan for SDK Zenoh Data Layer

## Overview

CYB-1544 is a sub-epic of CYB-1498 (Edge Models). It delivers the complete `cw.data` SDK module — a transport-agnostic pub/sub API for edge data. Developers and drivers call `cw.data.publish/subscribe/latest`; the transport (Zenoh or filesystem) is selected by config.

This sub-epic has five sub-issues:

```
CYB-1552  ──▶  CYB-1553  ──▶  CYB-1554  ──▶  CYB-1555
transport      wire format     public API      debug tools
(DONE - PR#1597)              (incl. Phase 1
                               staleness)
                                   │
                                   └──▶  CYB-1584
                                         time-aware fusion
                                         (data.at, data.window)
```

CYB-1552 → CYB-1553 → CYB-1554 → CYB-1555 is the strict sequential path. CYB-1584 depends on CYB-1554 but is independent of CYB-1555 — they can be developed in parallel after CYB-1554 merges.

All code lives in `cyberwave-sdks/cyberwave-python/cyberwave/data/`, except CYB-1554 which also touches `cyberwave/client.py`.

---

## Current State (after CYB-1552)

Implemented and in review (PR #1597):

```
cyberwave/data/
├── __init__.py            # Re-exports DataBackend, get_backend, exceptions
├── backend.py             # DataBackend ABC, Sample, Subscription, VALID_POLICIES
├── zenoh_backend.py       # ZenohBackend (RingChannel-based latest, queryables)
├── filesystem_backend.py  # FilesystemBackend (ring buffer, atomic writes, polling)
├── config.py              # BackendConfig, get_backend() factory
└── exceptions.py          # DataBackendError hierarchy
```

What exists: raw `bytes` pub/sub over two backends, env-var config, 58 tests.
What's missing: wire format, typed API, key expressions, client wiring, record/replay.

---

## CYB-1553: Wire Format, SDK Header, and Key Expression Convention

### Goal

A canonical on-wire contract so Python SDK publishers, C++ native publishers, and any Zenoh client can interoperate. Every sample on the bus carries a small header followed by the payload. The design must support high-throughput hot paths (60fps 1080p video, 1kHz force/torque) without per-sample JSON serialization.

### New files

```
cyberwave/data/
├── header.py              # HeaderMeta, encode/decode, HeaderTemplate (cached encoder)
└── keys.py                # build_key / parse_key / validate_key / ParsedKey
```

### Design: Wire format

Every Zenoh sample is structured as:

```
[total_header_length: 4 bytes LE uint32][ts: 8 bytes LE float64][seq: 8 bytes LE int64][header_json: UTF-8][payload_bytes]
```

- **`total_header_length`** (4 bytes): covers `ts` + `seq` + `header_json`. Lets the decoder split header from payload in a single `struct.unpack` + slice.
- **`ts`** (8 bytes): acquisition timestamp as float64 (Unix epoch seconds). Binary-packed for O(1) encode/decode — no JSON parsing needed to read the timestamp.
- **`seq`** (8 bytes): per-channel sequence number as int64. Binary-packed for the same reason.
- **`header_json`** (variable): UTF-8 JSON with static channel metadata (content_type, shape, dtype, channel-specific fields). This portion is identical across samples on the same channel — enabling the `HeaderTemplate` optimization.
- **`payload_bytes`** (variable): raw payload (numpy buffer, JSON bytes, or opaque binary).

The split between binary-packed per-sample fields (`ts`, `seq`) and JSON-encoded static fields (everything else) is the key design decision: it avoids any JSON serialization on the hot path while keeping the metadata human-readable and extensible.

### Design: `header.py`

#### Core types

```python
@dataclass(slots=True)
class HeaderMeta:
    content_type: str          # "numpy/ndarray", "application/json", "application/octet-stream"
    ts: float                  # Unix epoch seconds (float64), acquisition time
    seq: int = 0               # Per-channel sequence number for drop detection
    shape: tuple[int, ...] | None = None
    dtype: str | None = None   # numpy dtype string, e.g. "uint8", "float32"
    metadata: dict[str, Any] | None = None
```

`HeaderMeta` is deliberately generic. Channel-specific fields from the README (e.g., `fps`, `unit`, `sample_rate`, `n_points`, `resolution_m`, `origin`) go into `metadata`. The per-channel schema contract defines which keys are required in `metadata` for each channel type:

| Channel | Required `metadata` keys | Example |
|---|---|---|
| `frames` | `width`, `height`, `channels`, `fps` | `{"width": 1920, "height": 1080, "channels": 3, "fps": 30}` |
| `depth` | `width`, `height`, `unit` | `{"width": 640, "height": 480, "unit": "mm"}` |
| `audio` | `sample_rate`, `channels` | `{"sample_rate": 48000, "channels": 1}` |
| `pointcloud` | `n_points`, `fields` | `{"n_points": 65536, "fields": ["x","y","z"]}` |
| `map` | `width`, `height`, `resolution_m`, `origin` | `{"width": 200, "height": 200, "resolution_m": 0.05, "origin": {"x":0,"y":0,"theta":0}}` |
| JSON channels | (none required beyond `content_type`) | `ts` is in the payload itself for JSON channels |

The `metadata` dict is serialized **flat** into the JSON header (not nested under a `"metadata"` key). This matches the README spec where header fields like `width`, `height`, `fps` are top-level. On the wire, a frames header JSON looks like: `{"content_type": "numpy/ndarray", "shape": [1080,1920,3], "dtype": "uint8", "width": 1920, "height": 1080, "channels": 3, "fps": 30}`.

#### Low-level encode/decode (stateless, for one-off or cross-language use)

```python
def encode(header: HeaderMeta, payload: bytes) -> bytes:
    """Encode a header + payload into the wire format.
    
    Packs ts/seq as binary, serializes the rest as JSON.
    """

def decode(raw: bytes) -> tuple[HeaderMeta, bytes]:
    """Split a wire sample into (header, payload_bytes).
    
    Unpacks ts/seq from the binary prefix, parses the JSON remainder.
    """
```

#### HeaderTemplate (cached encoder for hot paths)

On a 60fps camera stream, everything except `ts` and `seq` is identical across frames. `HeaderTemplate` pre-encodes the JSON portion once and combines it with per-sample binary fields on each `pack()` call.

```python
class HeaderTemplate:
    """Pre-compiled header for repeated publishing on the same channel.

    Encodes the static JSON (content_type, shape, dtype, metadata) once
    during __init__. On each pack() call, only ts and seq are packed
    as binary — no JSON serialization, no string formatting.
    """

    def __init__(
        self,
        content_type: str,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._seq: int = 0
        # Pre-encode the JSON portion (everything except ts/seq)
        json_dict = {"content_type": content_type}
        if shape is not None:
            json_dict["shape"] = list(shape)
        if dtype is not None:
            json_dict["dtype"] = dtype
        if metadata:
            json_dict.update(metadata)  # flat merge
        self._cached_json_bytes: bytes = json.dumps(json_dict, separators=(",", ":")).encode()
        self._cached_header_len: int = 16 + len(self._cached_json_bytes)  # ts(8) + seq(8) + json

    def pack(self, payload: bytes, *, ts: float | None = None) -> bytes:
        """Combine cached header with payload.
        
        Per-sample cost: one struct.pack (16 bytes) + one bytes join.
        No JSON serialization.
        """
        if ts is None:
            ts = time.time()
        seq = self._seq
        self._seq += 1
        header_len_bytes = struct.pack("<I", self._cached_header_len)
        ts_seq_bytes = struct.pack("<dq", ts, seq)
        return header_len_bytes + ts_seq_bytes + self._cached_json_bytes + payload
```

**Performance of `pack()`:**

| Operation | Cost |
|---|---|
| `struct.pack("<I", header_len)` | ~50ns |
| `struct.pack("<dq", ts, seq)` | ~50ns |
| `bytes` join (4 + 16 + ~200 + payload) | ~200ns for small payloads; dominated by payload size for large ones |
| **Total header overhead** | **<500ns** (excluding payload copy) |

For a 1080p frame (~6MB), total `pack()` time is ~1-2ms, dominated entirely by the payload memcpy — the header overhead is negligible. For small JSON payloads (joint states, IMU), total `pack()` is <1μs.

**When to use what:**

| Scenario | Use | Header cost per sample |
|---|---|---|
| Driver publishing frames at 60fps | `HeaderTemplate.pack()` | <500ns |
| Driver publishing joint states at 100Hz | `HeaderTemplate.pack()` | <500ns |
| One-off publish (config, alert) | `encode()` | ~3-5μs (full JSON encode) |
| Decode (subscriber side) | `decode()` | ~2-3μs (struct.unpack + JSON parse) |
| C++ / Rust native publisher | Follow wire format spec manually | N/A |

The `DataBus` (CYB-1554) creates a `HeaderTemplate` per channel on first publish and reuses it for subsequent calls. This is transparent to the user.

#### Sequence numbers

Each `HeaderTemplate` maintains an auto-incrementing `_seq: int` counter. This enables:
- **Drop detection** on the subscriber side (gap in sequence = dropped samples)
- **Replay ordering** validation (monotonic within a channel)

`seq` is per-channel only. Cross-channel temporal correlation uses `ts` (acquisition timestamp), not `seq` — two channels' sequence numbers have no relation to each other.

#### Content types

| Content type | When used | Payload |
|---|---|---|
| `numpy/ndarray` | Frames, depth maps, point clouds, audio | Raw buffer from `ndarray.tobytes()` |
| `application/json` | Joint states, IMU, GPS, telemetry, etc. | UTF-8 JSON |
| `application/octet-stream` | Opaque binary blobs | Raw bytes passed through |

#### Decode path performance

Decoding is per-sample. The binary prefix is O(1) (`struct.unpack("<dq", ...)`), the JSON portion is `json.loads()` on ~200 bytes (~2-3μs). Total decode: ~2-3μs. This is acceptable because:
- Subscribers typically process at lower rates than publishers (`"latest"` policy drops intermediate frames).
- The decode cost is dwarfed by actual processing (model inference, etc.).
- If decode becomes a bottleneck, `decode()` can add a fast path for known content types using `struct.unpack` instead of `json.loads`, without changing the wire format.

### Design: `keys.py`

Key expression pattern from README: `cw/{twin_uuid}/data/{channel}/{sensor_name}`

Keys always have exactly **5 segments** separated by `/`. The `sensor_name` is always present (defaults to `"default"` when not specified). This makes parsing unambiguous — split on `/`, expect 5 parts, done.

Examples:
- `cw/550e8400-e29b-41d4-a716-446655440000/data/frames/default`
- `cw/550e8400-e29b-41d4-a716-446655440000/data/depth/realsense_left`
- `cw/550e8400-e29b-41d4-a716-446655440000/data/joint_states/default`

Zenoh wildcards work naturally: `cw/{uuid}/data/frames/*` matches all cameras, `cw/{uuid}/data/**` matches all channels and sensors.

```python
@dataclass(slots=True)
class ParsedKey:
    """Result of parsing a Zenoh key expression."""
    twin_uuid: str
    channel: str
    sensor_name: str


def build_key(
    twin_uuid: str,
    channel: str,
    sensor_name: str = "default",
    *,
    prefix: str = "cw",
) -> str:
    """Build a Zenoh key expression. Validates components.
    
    Raises ChannelError if twin_uuid is not a valid UUID or
    channel/sensor_name contain slashes.
    """

def parse_key(key: str) -> ParsedKey:
    """Parse a key expression into its components.
    
    Expects exactly 5 slash-separated segments.
    Raises ChannelError if the key is malformed.
    """

def validate_key(key: str) -> bool:
    """Return True if key matches the canonical 5-segment pattern."""
```

Validation rules:
- `twin_uuid` must be a valid UUID string
- `channel` must be a single path segment (no slashes) — known channel name or custom
- `sensor_name` must be a single path segment (no slashes), defaults to `"default"`
- Publish keys must not contain wildcards (`*`, `**`)

### Tests

- Golden fixture tests: encode a known header + payload → assert exact bytes; decode → assert roundtrip.
- Wire format structure: verify the 4-byte length prefix, 8-byte ts, 8-byte seq, JSON, payload layout.
- `HeaderTemplate` hot-path test: pack 10,000 samples, verify all decode correctly, assert total time <5ms (i.e., <500ns per pack).
- `HeaderTemplate` seq auto-increment: verify monotonically increasing per template; verify independent counters across templates.
- `metadata` flat merge: verify channel-specific fields (e.g., `fps`, `unit`) appear as top-level keys in the JSON, not nested.
- Key builder: valid UUIDs, default sensor name, custom prefix.
- Key parser: valid 5-segment keys → `ParsedKey`; malformed keys (4 segments, 6 segments, invalid UUID, slashes in channel) → `ChannelError`.
- Cross-language contract: a fixture file with known encoded bytes that C++ tests can also verify (future-proofing).

### Integration with CYB-1552

`header.py` and `keys.py` do NOT depend on `DataBackend`. They are pure functions (and one stateful class) operating on bytes and strings. The public API (CYB-1554) composes them: `HeaderTemplate.pack()` before `backend.publish()`, `decode()` after `backend.latest()`.

---

## CYB-1554: Public API — `publish()`, `subscribe()`, `latest()`

### Goal

The user-facing `cw.data.*` facade that accepts typed payloads (numpy arrays, dicts, raw bytes), handles encoding/decoding, and delegates to the backend.

### New files

```
cyberwave/data/
└── api.py                 # DataBus class (the cw.data.* facade)
```

### Modified files

```
cyberwave/client.py        # Add `data` property returning DataBus
cyberwave/__init__.py      # (optional) re-export for convenience
```

### Design: `api.py`

```python
class DataBus:
    """Public API for the data layer. Accessed as `cw.data`."""

    def __init__(
        self,
        backend: DataBackend,
        twin_uuid: str,
        *,
        sensor_name: str = "default",
        key_prefix: str = "cw",
    ) -> None:
        self._templates: dict[str, HeaderTemplate] = {}  # per-channel cached encoders
        ...

    def publish(
        self,
        channel: str,
        sample: np.ndarray | dict | bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish a typed sample to a channel.

        - numpy arrays → content_type="numpy/ndarray", payload=array.tobytes()
        - dicts → content_type="application/json", payload=json.dumps(dict).encode()
        - bytes → content_type="application/octet-stream", payload as-is

        On the first call per channel, a HeaderTemplate is created and cached.
        Subsequent calls reuse it — only ts and seq are updated per sample.
        """

    def subscribe(
        self,
        channel: str,
        callback: Callable,
        *,
        policy: str = "latest",
        raw: bool = False,
    ) -> Subscription:
        """Subscribe to a channel.

        callback receives decoded data (numpy array, dict, or bytes)
        unless raw=True, in which case it receives Sample with raw bytes.
        """

    def latest(
        self,
        channel: str,
        *,
        timeout_s: float = 1.0,
        max_age_ms: float | None = None,
        raw: bool = False,
    ) -> np.ndarray | dict | bytes | None:
        """Get the latest value on a channel, decoded.

        If max_age_ms is set, returns None when the sample's acquisition
        timestamp (header.ts) is older than max_age_ms milliseconds.
        This is the Phase 1 temporal synchronization primitive from the
        README — the lowest-effort upgrade for sensor fusion code.
        """

    def close(self) -> None:
        """Close the underlying backend."""
```

### Wiring into `Cyberwave` client

```python
# In cyberwave/client.py, class Cyberwave:

@property
def data(self) -> DataBus:
    if self._data_bus is None:
        backend = get_backend()  # uses CYBERWAVE_DATA_BACKEND env
        twin_uuid = self.config.twin_uuid or os.getenv("CYBERWAVE_TWIN_UUID", "")
        self._data_bus = DataBus(backend, twin_uuid)
    return self._data_bus
```

This enables the README pattern: `cw.data.publish("frames/default", frame)`.

### Call chain (hot path — repeated publishes)

```
User                     DataBus (api.py)           HeaderTemplate         DataBackend
────                     ────────────────           ──────────────         ───────────
                         (first call to channel)
cw.data.publish(         infer content_type →       __init__(): encode     
  "depth", depth_arr)    create HeaderTemplate      static JSON once       
                         cache in _templates        

                         (subsequent calls)
cw.data.publish(         lookup cached template →   pack(payload, ts) →    backend.publish(
  "depth", depth_arr)    arr.tobytes()              splice ts + seq        key, wire_bytes)
                                                    ~100ns, no JSON
```

### Call chain (decode — subscribe / latest)

```
backend.latest(key)  →   decode(raw)  →   _decode_sample(header, payload)  →  user
                         JSON parse        np.frombuffer / json.loads           typed result
                         ~2-3μs            depends on content type
```

### Encoding dispatch (first publish per channel)

On the first `publish()` call for a given channel, `DataBus` inspects the sample type, creates a `HeaderTemplate`, and caches it:

```python
def _get_or_create_template(self, channel: str, sample) -> HeaderTemplate:
    if channel in self._templates:
        return self._templates[channel]

    if isinstance(sample, np.ndarray):
        tmpl = HeaderTemplate(
            content_type="numpy/ndarray",
            shape=sample.shape,
            dtype=str(sample.dtype),
        )
    elif isinstance(sample, dict):
        tmpl = HeaderTemplate(content_type="application/json")
    elif isinstance(sample, bytes):
        tmpl = HeaderTemplate(content_type="application/octet-stream")
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    self._templates[channel] = tmpl
    return tmpl
```

Subsequent publishes to the same channel call `tmpl.pack(payload_bytes)` — no type checking, no JSON encoding, no allocation beyond the final wire bytes.

**Shape change detection:** If a numpy array's shape or dtype changes between publishes on the same channel (e.g., camera resolution switch), the cached template is invalidated and recreated. This is checked with a cheap comparison (`sample.shape != tmpl.shape`), not on every call — only when the first publish created a numpy template.

### Decoding dispatch

```python
def _decode_sample(header: HeaderMeta, payload: bytes):
    if header.content_type == "numpy/ndarray":
        return np.frombuffer(payload, dtype=header.dtype).reshape(header.shape)
    elif header.content_type == "application/json":
        return json.loads(payload)
    else:
        return payload
```

### Performance budget per sample (publish hot path)

| Step | Cost | Notes |
|---|---|---|
| `arr.tobytes()` | ~1-2ms (1080p) | Dominated by memcpy; zero-copy with shared memory |
| `tmpl.pack()` | ~100ns | Fixed-size splice of ts/seq into cached bytes |
| `backend.publish()` | ~10-50μs | Zenoh `session.put()` |
| **Total** | ~1-2ms | Bottleneck is payload copy, not header |

### Tests

- Publish numpy array → latest returns identical array (shape, dtype, values).
- Publish dict → latest returns identical dict.
- Publish bytes → latest returns identical bytes.
- Subscribe with callback receives decoded data.
- Subscribe with `raw=True` receives raw `Sample`.
- `DataBus.close()` tears down the backend.
- Integration: `cw = Cyberwave(...)` → `cw.data.publish(...)` → `cw.data.latest(...)` roundtrip.
- HeaderTemplate caching: second publish on same channel does NOT re-encode JSON.
- Shape change: publish array with new shape → template invalidated, new header encoded.
- Seq numbers: verify monotonically increasing across publishes on same channel.
- `latest(max_age_ms=50)` with fresh sample → returns data.
- `latest(max_age_ms=50)` with stale sample → returns `None`.
- `latest(max_age_ms=None)` (default) → always returns data regardless of age.
- Error: unsupported sample type raises `TypeError`.
- Error: corrupt header raises `DataBackendError`.

### Temporal synchronization (Phase 1)

The README defines a phased approach to sensor fusion primitives. CYB-1554 implements **Phase 1 only**: the `max_age_ms` parameter on `latest()`.

```python
depth = cw.data.latest("depth", max_age_ms=50)
if depth is None:
    return  # depth sample is older than 50ms — too stale to fuse
```

**Implementation:** After decoding the sample from the backend, compare `time.time() - header.ts` against `max_age_ms / 1000`. If stale, return `None`. Cost: one float subtraction and comparison — negligible.

**What is NOT in CYB-1554:**

| Primitive | README Phase | Owner | Why not here |
|---|---|---|---|
| `@cw.on_synchronized(channels, slop_ms)` | Phase 2 | CYB-1545 (worker runtime) | Requires hook registration and multi-channel dispatching — worker runtime concern |
| `cw.data.at(channel, t=, interpolation=)` | Phase 3 | **CYB-1584** (new sub-issue) | Requires per-channel `TimeIndexedRingBuffer` — new infrastructure |
| `cw.data.window(channel, from_t=, to_t=)` | Phase 4 | **CYB-1584** (new sub-issue) | Shares ring buffer with Phase 3 |
| Clock status queryable | Phase 5 | Edge Core | Not SDK scope |

Phases 3 and 4 are tracked in a new sub-issue **CYB-1584: Time-aware fusion primitives (`data.at()`, `data.window()`)** under CYB-1544.

### Lazy initialization

`DataBus` is created lazily on first access to `cw.data`. This avoids opening a Zenoh session for users who never use the data layer. It also means the backend is only initialized when `CYBERWAVE_DATA_BACKEND` is actually needed.

---

## CYB-1555: Debug Utilities — `record()` and `replay()`

### Goal

Let developers capture live samples to disk and replay them later — deterministic debugging, regression testing, offline analysis.

### New files

```
cyberwave/data/
├── recording.py           # record() context manager + replay() function
└── recording_format.py    # File format: header + sample entries (versioned)
```

### Design: Recording format

A recording is a directory:

```
my_recording/
├── manifest.json          # version, channels, start_ts, end_ts, sample_count
├── frames__default.bin    # binary stream: [ts:f64][header_len:u32][header_json][payload_len:u32][payload]...
├── depth__default.bin
└── joint_states.bin
```

One file per channel. Each entry is self-describing (timestamp + header + payload). This allows random access by channel and streaming reads for replay.

`manifest.json`:
```json
{
  "version": 1,
  "channels": ["frames/default", "depth/default", "joint_states"],
  "start_ts": 1711234567.123,
  "end_ts": 1711234597.456,
  "sample_count": 1500
}
```

### Design: `recording.py`

```python
def record(
    data_bus: DataBus,
    channels: list[str],
    path: str | Path,
    *,
    max_samples: int | None = None,
    max_duration_s: float | None = None,
) -> RecordingSession:
    """Start recording samples from channels to disk.

    Returns a RecordingSession context manager.
    """

class RecordingSession:
    """Active recording. Use as context manager or call stop() manually."""

    def stop(self) -> RecordingManifest: ...
    def __enter__(self) -> RecordingSession: ...
    def __exit__(self, *exc) -> None: ...


def replay(
    data_bus: DataBus,
    path: str | Path,
    *,
    speed: float = 1.0,
    loop: bool = False,
    channels: list[str] | None = None,  # None = all recorded channels
) -> None:
    """Replay recorded samples by publishing them to the data bus.

    Preserves inter-sample timing scaled by `speed`.
    Triggers the same hooks as live data.
    """
```

### How `record()` works

1. Subscribes to each requested channel via `data_bus.subscribe(ch, _on_sample, policy="fifo", raw=True)`.
2. Writes each `Sample` to the per-channel `.bin` file with timestamp prefix.
3. Stops when `max_samples` / `max_duration_s` is reached, or `stop()` / context-manager exit.
4. Writes `manifest.json` on stop.

### How `replay()` works

1. Reads `manifest.json` to discover channels.
2. Opens each `.bin` file, reads entries into a merged timeline sorted by timestamp.
3. Iterates the timeline, sleeping for the inter-sample delta (scaled by `speed`).
4. Calls `data_bus.publish(channel, payload)` for each entry — re-triggering any active hooks.
5. If `loop=True`, repeats from the start.

### Tests

- Record 10 samples → verify file structure and manifest.
- Record → replay → verify callback sequence matches original order.
- Replay with `speed=0` (instantaneous) → all samples delivered.
- Replay with `channels` filter → only specified channels replayed.
- Replay with `loop=True` → verify second pass.
- Cross-backend: record on Zenoh, replay on filesystem (portable recordings).

---

## Dependency Graph

```
CYB-1552 (DONE)                    CYB-1553                  CYB-1554                CYB-1555
DataBackend + backends     ──▶     header.py + keys.py  ──▶  DataBus (cw.data.*)  ──▶ record/replay
backend.py                         encode/decode              publish(typed)          recording.py
zenoh_backend.py                   build_key/parse_key        subscribe(decoded)      recording_format.py
filesystem_backend.py              HeaderMeta                 latest(decoded +
config.py                          HeaderTemplate               max_age_ms)
exceptions.py                                                 client.py wiring
                                                                    │
                                                                    └──▶  CYB-1584
                                                                          data.at()
                                                                          data.window()
                                                                          TimeIndexedRingBuffer
```

### Why this order

- CYB-1553 is pure functions (no runtime deps) and must exist before CYB-1554 can encode/decode.
- CYB-1554 composes CYB-1552 (backend) + CYB-1553 (header) into the user API. Includes Phase 1 temporal sync (`max_age_ms`).
- CYB-1555 depends on CYB-1554 because `record()` subscribes and `replay()` publishes via `DataBus`.
- CYB-1584 depends on CYB-1554 (needs the decoded `ts` from `HeaderMeta`). Independent of CYB-1555.

### Can anything be parallelized?

- CYB-1553 and the "wiring" part of CYB-1554 (adding `data` property to `Cyberwave`) are independent and could be developed concurrently. However, the encoding/decoding logic in CYB-1554 directly uses CYB-1553's `encode()`/`decode()`, so the merge order must remain sequential.
- After CYB-1554 merges, **CYB-1555 and CYB-1584 can be developed in parallel** — record/replay and time-aware fusion have no dependency on each other.

---

## File Layout After All Five Issues

```
cyberwave/data/
├── __init__.py              # CYB-1552 (updated in CYB-1554 to re-export DataBus)
├── backend.py               # CYB-1552
├── zenoh_backend.py         # CYB-1552
├── filesystem_backend.py    # CYB-1552
├── config.py                # CYB-1552
├── exceptions.py            # CYB-1552
├── header.py                # CYB-1553
├── keys.py                  # CYB-1553
├── api.py                   # CYB-1554 (includes max_age_ms staleness)
├── recording.py             # CYB-1555
├── recording_format.py      # CYB-1555
├── ring_buffer.py           # CYB-1584 (TimeIndexedRingBuffer)
└── fusion.py                # CYB-1584 (data.at, data.window, interpolation)

tests/
├── test_data_backend_contract.py  # CYB-1552
├── test_filesystem_backend.py     # CYB-1552
├── test_zenoh_backend.py          # CYB-1552
├── test_data_config.py            # CYB-1552
├── test_data_header.py            # CYB-1553
├── test_data_keys.py              # CYB-1553
├── test_data_api.py               # CYB-1554
├── test_data_api_integration.py   # CYB-1554
├── test_data_staleness.py         # CYB-1554 (max_age_ms tests)
├── test_data_recording.py         # CYB-1555
├── test_data_ring_buffer.py       # CYB-1584
├── test_data_fusion.py            # CYB-1584
└── conftest.py                    # CYB-1552 (REST stubs)
```

---

## Risks and Open Questions

| # | Item | Resolution |
|---|---|---|
| 1 | **Header serialization on hot path** | `HeaderTemplate` caches the static JSON once; per-sample cost is ~100ns (byte splice), not ~3-5μs (full `json.dumps`). If this ever becomes a bottleneck, a binary `struct.pack` path can be added behind the same `pack()` interface without changing the wire format (version the header format to negotiate). |
| 2 | **Decode cost on subscriber side** | `json.loads` on ~200 byte header is ~2-3μs. Acceptable because subscribers typically process at lower rates than publishers (`"latest"` policy drops intermediate samples), and decode is dwarfed by actual processing (model inference). A C extension or `struct.unpack` fast path can be added later for known content types. |
| 3 | **Fixed-width timestamp in HeaderTemplate** | The `ts` field is formatted to a constant byte width so byte offsets don't shift. This wastes a few bytes of JSON padding but keeps the splice O(1). The decoder strips padding transparently. |
| 4 | **numpy dependency in header decode** | numpy is already a core dependency (`>=1.26.0`). No new deps. |
| 5 | **Recording file size for long captures** | `max_samples` and `max_duration_s` parameters. No automatic rotation in v1 — add later if needed. |
| 6 | **Key expression validation strictness** | Publish keys must be exact (no wildcards). Subscribe keys may use Zenoh wildcards (`*`, `**`) for multi-sensor listening. |
| 7 | **Thread safety of DataBus** | `DataBus` delegates to `DataBackend` which is already thread-safe. `HeaderTemplate.pack()` is thread-safe (reads cached bytes, writes to new buffer). The `_templates` dict is written on first publish per channel and read-only thereafter — safe under CPython's GIL. |
| 8 | **Lazy backend init in client.py** | If `CYBERWAVE_DATA_BACKEND=zenoh` but Zenoh isn't installed, the error only surfaces on first `cw.data` access, not at `Cyberwave()` construction. Intentional — avoid penalizing users who don't use the data layer. |
| 9 | **Shape change invalidation** | If a numpy array's shape/dtype changes mid-stream (rare — e.g., camera resolution switch), the cached `HeaderTemplate` is invalidated and recreated. This adds one `json.dumps` call at the switch point. Detected via cheap tuple comparison, not on every publish. |

---

## Acceptance Criteria (sub-epic-level)

From the CYB-1544 issue description:

- [x] Same client code works across both backends — **CYB-1552** (tested with parametrized contract tests)
- [ ] Publish/subscribe/latest semantics are equivalent across backends — **CYB-1554** (API-level parity tests)
- [ ] Key-expression mapping is validated in tests — **CYB-1553** (key builder/parser tests)
- [ ] No consumer needs direct Zenoh calls — **CYB-1554** (DataBus facade hides backend)
- [ ] Backend switch requires config only — **CYB-1552** (`CYBERWAVE_DATA_BACKEND` env var)
- [ ] Staleness-aware `latest()` supports sensor fusion — **CYB-1554** (`max_age_ms` parameter, Phase 1)
- [ ] Time-aware fusion primitives (`data.at()`, `data.window()`) — **CYB-1584** (Phases 3+4, after CYB-1554)

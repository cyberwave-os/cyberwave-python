"""Recording retrieval + local inspection.

Entry points:
    twin.recordings.list()/.get()             (twin-scoped)
    cw.environments.recordings.list(environment_id=...)/.get()  (env-scoped)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Union
from urllib.parse import urlparse

import urllib3

from cyberwave.rest.models.recording_materializing_schema import (
    RecordingMaterializingSchema,
)

logger = logging.getLogger(__name__)

DEFAULT_MATERIALIZING_RETRY_SECONDS = 15

#: Page size requested from the catalog endpoint. The deployed frontend uses
#: the same conservative size: environments with large per-recording metadata
#: can complete a 50-item page that fails at the server/proxy boundary when
#: asked to render 100. Every read remains paged, preventing an unbounded
#: history response from overflowing the HTTP body ceiling.
CATALOG_PAGE_SIZE = 50

#: How many recordings ``list()`` returns when the caller does not say. ``0``
#: means "keep following pages until the catalog is exhausted".
DEFAULT_LIST_LIMIT = 200


def _response_header(error: Exception, name: str) -> str | None:
    """Read an exception response header case-insensitively."""
    headers = getattr(error, "headers", None)
    if not headers:
        return None
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return str(value)
    return None


def _is_probable_recording_payload_too_large(error: Exception) -> bool:
    """Recognize Cloud Run's platform-generated oversized-response failure.

    Cloud Run does not expose a dedicated status or response header for this
    case. It discards the application response and emits the narrow signature
    observed in its request logs: an empty HTML 500 from Google Frontend with a
    zero content length. Requiring the whole signature avoids relabeling normal
    JSON 500 responses from the Cyberwave backend.
    """
    content_type = (_response_header(error, "content-type") or "").lower()
    return (
        getattr(error, "status", None) == 500
        and getattr(error, "body", None) in (None, "")
        and (_response_header(error, "server") or "").lower() == "google frontend"
        and _response_header(error, "content-length") == "0"
        and content_type.startswith("text/html")
    )


def _materializing_message(body: RecordingMaterializingSchema) -> str:
    """Render a 202 body, preserving its reason and retry interval.

    ``retry_after_seconds`` is a top-level field on the 202 body, not a key
    inside ``playback_readiness``; read both so neither is dropped.
    """
    readiness = body.playback_readiness or {}
    reason = readiness.get("reason") or body.detail or "unknown reason"
    retry = (
        body.retry_after_seconds
        or readiness.get("retry_after_seconds")
        or DEFAULT_MATERIALIZING_RETRY_SECONDS
    )
    return (
        f"Recording assets are still materializing ({reason}). Retry in ~{retry}s."
    )

#: Artifacts downloaded concurrently by one ``get()``. A recording is segmented
#: (60s video parts, chunked pointclouds), so a serial fetch pays a connect +
#: TLS + first-byte round trip per part; that latency, not bandwidth, is what
#: dominates a many-part recording. Sizing beyond a handful buys little — the
#: transfers then compete for the same link.
DEFAULT_DOWNLOAD_WORKERS = 8

#: Hard ceiling on download concurrency, whatever a caller asks for. Each worker
#: costs a thread, a socket, a streaming buffer, and a slot in the connection
#: pool, and the gains flatten as soon as the link saturates — while a long
#: recording can hold hundreds of parts, so ``len(sources)`` alone is no bound.
MAX_DOWNLOAD_WORKERS = 32

# Process-wide connection pool for artifact downloads, created lazily on first
# use. Shared (rather than one-per-download) so keep-alive sockets are recycled
# instead of leaking until GC.
_http_pool: urllib3.PoolManager | None = None
_http_pool_maxsize = 0
# Guards the two globals above: concurrent get() calls (or a caller driving
# get() from its own threads) would otherwise both pass the resize check and
# build a pool, orphaning one along with its established connections.
_http_pool_lock = threading.Lock()


def _get_http_pool(maxsize: int) -> urllib3.PoolManager:
    """The shared pool, holding at least ``maxsize`` connections per host.

    ``maxsize`` is the per-host keep-alive budget (PoolManager keys pools by host,
    and signed URLs may span storage backends). It MUST cover the download
    concurrency: at urllib3's default of 1, with ``block=False``, the surplus
    connections are opened and then discarded on return, so every artifact past
    the pool size re-pays the handshake that this parallelism exists to avoid.
    A caller asking for more workers than the pool holds therefore has to grow
    it — hence the rebuild rather than a fixed size set on first use.

    ``maxsize`` is required rather than defaulted, because the pool only ever
    grows: a caller that guesses high cannot be walked back by the next one, and a
    caller that inherits a default it does not need will grow the pool past the
    concurrency the fetch actually planned for.
    """
    global _http_pool, _http_pool_maxsize
    with _http_pool_lock:
        if _http_pool is None or maxsize > _http_pool_maxsize:
            # Close the pool being replaced. Its established keep-alive sockets
            # are otherwise orphaned until GC, which is the very leak the shared
            # pool exists to prevent, and the next fetch re-pays a handshake for
            # each one lost. A connection checked out by an in-flight download is
            # unaffected: urllib3 closes it when it is released into the cleared
            # pool instead of returning it to the queue.
            outgrown, _http_pool = _http_pool, urllib3.PoolManager(maxsize=maxsize)
            _http_pool_maxsize = maxsize
            if outgrown is not None:
                outgrown.clear()
        return _http_pool


class RecordingType(str, Enum):
    CAMERA = "camera"
    ROBOT = "robot"
    # POINTCLOUD = colored/lidar points (the ``colored_pointcloud`` stream);
    # DEPTH = raw depth maps (the legacy-named ``pointcloud`` stream). They map to
    # the ``read_pointcloud()`` and ``read_depth()`` accessors respectively.
    POINTCLOUD = "pointcloud"
    DEPTH = "depth"
    AUDIO = "audio"


_SOURCE_CAMERA = "camera"
_SOURCE_ACTUATION = "actuation"
_SOURCE_POINTCLOUD = "pointcloud"
_SOURCE_COLORED_POINTCLOUD = "colored_pointcloud"
_SOURCE_AUDIO = "audio"

_SOURCE_TO_TYPE: dict[str, RecordingType] = {
    _SOURCE_CAMERA: RecordingType.CAMERA,
    _SOURCE_ACTUATION: RecordingType.ROBOT,
    # The ``pointcloud`` response key carries raw depth maps -> DEPTH; the
    # ``colored_pointcloud`` key carries colored/lidar points -> POINTCLOUD.
    _SOURCE_POINTCLOUD: RecordingType.DEPTH,
    _SOURCE_COLORED_POINTCLOUD: RecordingType.POINTCLOUD,
    _SOURCE_AUDIO: RecordingType.AUDIO,
}


def _filename_from(url: str, default_ext: str) -> str:
    """Derive a base filename from a signed URL path, ensuring an extension."""
    base = Path(urlparse(url).path).name or f"artifact.{default_ext}"
    if "." not in base:
        base = f"{base}.{default_ext}"
    return base


def _first_int(part: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    """The first of ``keys`` present on ``part`` as an int, else ``None``."""
    for key in keys:
        value = part.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


#: Per-stream ordering keys, most authoritative first. Camera parts label their
#: window ``first_timestamp_us`` while actuation parts say ``start_timestamp_us``,
#: hence two spellings per group. The timestamp tier is not always usable: the
#: server coerces a missing window to ``0`` when it records the part and again
#: when it builds the envelope, so a zero-filled tier answers for every part
#: while carrying no order at all — see ``_ordered_parts``.
_PART_ORDER_KEYS: tuple[tuple[str, ...], ...] = (
    ("first_timestamp_us", "start_timestamp_us"),
    ("chunk_index", "part_index"),
)


def _ordered_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort one segmented stream's parts into playback order.

    Envelope order is not treated as a contract. Both server paths do sort today
    — the finalized path by ``chunk_index``, the progressive path when it rebuilds
    its manifest from the stored part rows — so this is defense in depth rather
    than a repair for a live defect. Out-of-order parts would play a recording's
    segments out of sequence and concatenate robot parquet rows against time, so
    order them from the position each part reports rather than from the list.

    A tier is used only when every part answers it AND the answers are distinct.
    Completeness alone is not a usable test: because the server zero-fills a
    missing timestamp, an absent window is indistinguishable from epoch zero, so
    an "everyone answered" check passes vacuously and would claim the sort while
    ordering nothing — leaving the index tier below it permanently unreachable.
    Requiring a tier to discriminate is what keeps that fallback live, and when
    no tier can order the parts, arrival order is left untouched.
    """
    if len(parts) < 2:
        return parts
    for keys in _PART_ORDER_KEYS:
        values = [_first_int(p, keys) for p in parts]
        if all(v is not None for v in values) and len(set(values)) == len(values):
            # Position sits in the key to keep the dicts themselves out of the
            # comparison; the check above already rules out ties.
            return [
                part
                for _value, _position, part in sorted(
                    zip(values, range(len(parts)), parts), key=lambda t: (t[0], t[1])
                )
            ]
    return parts


def _require(module: str) -> Any:
    """Import an optional viewer dependency or raise a friendly install error."""
    import importlib

    from ..exceptions import CyberwaveError

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise CyberwaveError(
            f"{module!r} is required for this viewer. "
            "Install the data extras with: pip install 'cyberwave[data]'"
        ) from e


def _classify(metadata: dict[str, Any]) -> frozenset[RecordingType]:
    """Derive stream types from list metadata for servers without readiness.

    This fallback must stay in lockstep with the backend's
    ``src/lib/recordings/streams.py``. Newer servers provide the authoritative
    stream list in ``playback_readiness`` instead.
    """
    types: set[RecordingType] = set()
    rec_type = str(metadata.get("recording_type") or "")

    # Camera: modern metadata, camera-prefixed recording types, an active
    # manifest carrying video parts, or a legacy row identified only by video.
    if (
        metadata.get("metadata_type") == "CameraRecordingMetadata"
        or rec_type.startswith("camera")
        or metadata.get("video_parts")
        or (not rec_type and metadata.get("mp4_path"))
    ):
        types.add(RecordingType.CAMERA)

    # `path`/`num_rows` are intentionally not robot evidence: camera rows carry
    # both too. Legacy robot rows instead carry twin_type or joint_names.
    if (
        metadata.get("metadata_type") == "TwinRecordingMetadata"
        or rec_type == "robot"
        or metadata.get("robot_parts")
        or (
            not rec_type
            and (
                metadata.get("twin_type") == "robot"
                or ("joint_names" in metadata and not metadata.get("mp4_path"))
            )
        )
    ):
        types.add(RecordingType.ROBOT)

    additional = metadata.get("additional_camera_data")
    legacy = additional if isinstance(additional, dict) else {}
    if metadata.get("pointcloud") or legacy.get("pointcloud"):
        types.add(RecordingType.DEPTH)
    if metadata.get("colored_pointcloud") or legacy.get("colored_pointcloud"):
        types.add(RecordingType.POINTCLOUD)
    if metadata.get("audio_parts"):
        types.add(RecordingType.AUDIO)
    return frozenset(types)


def _normalize_types(
    types: "Union[RecordingType, str, Iterable[Union[RecordingType, str]]]",
) -> frozenset[RecordingType]:
    if isinstance(types, (RecordingType, str)):
        candidates: Iterable[Union[RecordingType, str]] = [types]
    else:
        candidates = types
    out: set[RecordingType] = set()
    for t in candidates:
        out.add(t if isinstance(t, RecordingType) else RecordingType(str(t)))
    return frozenset(out)


def _parse_date_filter(value: "date | datetime | str | None") -> "date | None":
    """Normalize a ``start``/``end`` recordings filter value to a calendar date.

    Accepts a ``date``, a ``datetime`` (the time-of-day is dropped — the REST
    endpoint filters by calendar day), or an ISO 8601 string such as
    ``"2026-07-01"`` or ``"2026-07-01T10:30:00Z"``.
    """
    from ..exceptions import CyberwaveError

    if value is None:
        return None
    if isinstance(value, datetime):
        # The backend filters by UTC calendar day. For an aware datetime,
        # normalize to UTC before dropping the time-of-day so the day matches
        # the server's; a naive datetime is taken as-is.
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as e:
            raise CyberwaveError(
                f"Invalid ISO date/datetime string for recordings filter: {value!r}"
            ) from e
        # Same UTC normalization as the aware-datetime path above.
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.date()
    raise CyberwaveError(
        f"Unsupported type for recordings date filter: {type(value).__name__}"
    )


@dataclass(frozen=True)
class RecordingListItem:
    """Lightweight view of one recording from the list endpoint (no network)."""

    uuid: str
    twin_uuid: str | None
    environment_uuid: str
    metadata: dict[str, Any]
    #: Server-computed playback readiness; absent on servers predating it.
    readiness: dict[str, Any] | None = None

    @property
    def types(self) -> frozenset[RecordingType]:
        streams = (self.readiness or {}).get("streams")
        if isinstance(streams, list):
            resolved: set[RecordingType] = set()
            for stream in streams:
                if not isinstance(stream, dict):
                    continue
                try:
                    resolved.add(RecordingType(str(stream.get("type"))))
                except ValueError:
                    continue
            return frozenset(resolved)
        return _classify(self.metadata)

    @property
    def is_final(self) -> bool:
        """False while the recording is still being written (progressive/active).

        Active manifests carry ``is_final: False`` / ``recording_type: "active"``;
        finalized rows omit the flag, so absence means final.
        """
        return (
            self.metadata.get("is_final") is not False
            and str(self.metadata.get("recording_type") or "") != "active"
        )

    @property
    def processing_status(self) -> str:
        """``"ready"`` when derived artifacts are available; ``"processing"``
        while the recording is active or its derivatives (e.g. the full-session
        MP4) are still being built — downloads then contain the unprocessed
        segments available so far. Prefers the server-stamped field."""
        status = str(self.metadata.get("processing_status") or "")
        if status in ("ready", "processing"):
            return status
        return "ready" if self.is_final else "processing"

    @property
    def is_playback_ready(self) -> bool:
        """Whether the server reports artifacts ready for playback.

        Older servers omit readiness, so retain their historical optimistic
        behavior rather than treating every recording as unavailable.
        """
        state = (self.readiness or {}).get("state")
        return state is None or state == "ready"

    @classmethod
    def _from_rest(cls, obj: Any) -> "RecordingListItem":
        twin_uuid = getattr(obj, "twin_uuid", None)
        readiness = getattr(obj, "playback_readiness", None)
        return cls(
            uuid=str(obj.uuid),
            twin_uuid=str(twin_uuid) if twin_uuid else None,
            environment_uuid=str(obj.environment_uuid),
            metadata=dict(getattr(obj, "metadata", None) or {}),
            readiness=dict(readiness) if isinstance(readiness, dict) else None,
        )

    def get(
        self, *, path: str | None = None, max_workers: int | None = None
    ) -> Any:
        """Fetch this recording's artifacts (shortcut for ``manager.get(item)``)."""
        from ..exceptions import CyberwaveError

        manager = getattr(self, "_manager", None)
        if manager is None:
            raise CyberwaveError(
                "This RecordingListItem is not attached to a manager; use "
                "twin.recordings.get(item) or cw.environments.recordings.get(item) "
                "instead."
            )
        env = getattr(self, "_environment_id", None) or self.environment_uuid
        # Forward this item's owning twin so a twin-scoped item from
        # twin.recordings.list() only downloads THAT twin's artifacts on a
        # shared multi-twin recording. Env-scoped items may lack twin_uuid; in
        # that case pass None to keep the full-envelope (all-twins) behavior.
        return manager.get(
            self,
            environment_id=env,
            path=path,
            twin_uuid=self.twin_uuid or None,
            max_workers=max_workers,
        )


class RecordingList(list):
    """A ``list`` of :class:`RecordingListItem` with an extra ``filter``."""

    def filter(
        self,
        types: "Union[RecordingType, str, Iterable[Union[RecordingType, str]]]",
    ) -> "RecordingList":
        wanted = _normalize_types(types)
        return RecordingList(item for item in self if item.types & wanted)


class Recording:
    """A fetched recording with artifacts downloaded to a local temp dir."""

    def __init__(
        self,
        *,
        uuid: str,
        twin_uuid: str | None,
        environment_uuid: str,
        types: frozenset[RecordingType],
        signed_urls: Any,
        local_paths: dict[str, list[Path]],
        tempdir: str,
    ) -> None:
        self.uuid = uuid
        self.twin_uuid = twin_uuid
        self.environment_uuid = environment_uuid
        self.types = types
        self.signed_urls = signed_urls
        self.local_paths = local_paths
        self._tempdir = tempdir
        # Reap the temp dir even if the caller never calls close()/__exit__ —
        # e.g. ``twin.recordings.get(item).read_robot()`` drops the Recording
        # reference immediately. weakref.finalize runs at most once (on the
        # first of close()/__exit__/gc), and is preferred over __del__.
        self._finalizer = weakref.finalize(
            self, shutil.rmtree, tempdir, ignore_errors=True
        )

    def close(self) -> None:
        """Delete the downloaded temp files."""
        # Idempotent: finalize() is a no-op after the first call.
        self._finalizer()

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        kinds = ",".join(sorted(t.value for t in self.types))
        return f"<Recording {self.uuid} types=[{kinds}]>"

    def _paths_with_ext(self, ext: str) -> list[Path]:
        return [
            p
            for paths in self.local_paths.values()
            for p in paths
            if p.suffix.lower() == ext
        ]

    # --- Contextual methods -------------------------------------------------
    # We surface only the accessors that make sense for THIS recording, so a
    # robot recording exposes ``read_robot()``, a depth recording exposes
    # ``read_depth()``, a colored/lidar recording exposes ``read_pointcloud()``,
    # and a recording with video exposes ``show_video()`` — never inapplicable,
    # non-working methods. Presence is driven by ``__getattr__``/``__dir__``
    # against the checks below. ``info()`` is always available (a plain method).
    #   name -> (predicate method, implementation method)
    _CONTEXT_METHODS: "dict[str, tuple[str, str]]" = {
        "read_robot": ("_has_robot", "_read_robot"),
        "read_depth": ("_has_depth", "_read_depth"),
        "read_pointcloud": ("_has_colored", "_read_pointcloud"),
        "show_video": ("_has_video", "_show_video_ctx"),
    }

    def _has_robot(self) -> bool:
        return bool(self._first_source(_SOURCE_ACTUATION))

    def _has_depth(self) -> bool:
        return bool(self._first_source(_SOURCE_POINTCLOUD))

    def _has_colored(self) -> bool:
        return bool(self._first_source(_SOURCE_COLORED_POINTCLOUD))

    def _has_video(self) -> bool:
        return bool(self._paths_with_ext(".mp4")) or bool(
            self._first_source(_SOURCE_CAMERA)
        )

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails, so the real methods
        # (``_read_depth`` etc.) and fields resolve normally and never recurse.
        spec = type(self).__dict__.get("_CONTEXT_METHODS", {}).get(name)
        if spec is None:
            raise AttributeError(name)
        check, impl = spec
        if getattr(self, check)():
            return getattr(self, impl)
        raise AttributeError(
            f"{name!r} is not available for this recording "
            f"(no matching stream); available: {self._available_context_methods()}"
        )

    def __dir__(self) -> list[str]:
        base = [d for d in super().__dir__() if d not in self._CONTEXT_METHODS]
        return sorted(base + self._available_context_methods())

    def _available_context_methods(self) -> list[str]:
        return [
            name
            for name, (check, _impl) in self._CONTEXT_METHODS.items()
            if getattr(self, check)()
        ]

    def _read_pc_parquet(self, source_key: str) -> list[dict[str, Any]]:
        """Read the downloaded point-cloud parquet(s) into per-frame numpy arrays.

        A stream may span several parquet parts (multi-session / segmented
        recording); all downloaded parts are joined and the frames returned in
        ``timestamp_us`` order.
        """
        from ..exceptions import CyberwaveError

        paths = self.local_paths.get(source_key) or []
        parquets = sorted(p for p in paths if str(p).endswith(".parquet"))
        if not parquets:
            raise CyberwaveError(
                "No parquet downloaded for this stream. Either this fetch used a "
                "path filter that excluded it (get(path=...)), or the server is "
                "still materializing it — in which case call get() again in a "
                "few minutes."
            )
        pq = _require("pyarrow.parquet")
        pa = _require("pyarrow")
        np = _require("numpy")
        tables = [pq.read_table(str(p)) for p in parquets]
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        frames = [
            {
                "timestamp_us": int(ts),
                "frame": np.frombuffer(data, dtype=str(dt)).reshape(int(r), int(c)),
            }
            for ts, r, c, dt, data in zip(
                table.column("timestamp_us").to_pylist(),
                table.column("rows").to_pylist(),
                table.column("cols").to_pylist(),
                table.column("dtype").to_pylist(),
                table.column("data").to_pylist(),
                strict=True,
            )
        ]
        frames.sort(key=lambda f: f["timestamp_us"])
        return frames

    def _read_robot(self) -> Any:
        """Return the robot actuation parquet as a ``pyarrow.Table``.

        This is the recording's own table (joint action/observation columns). A
        segmented recording is downloaded as several parquet parts; they share
        one schema (so joint column order is preserved) and are concatenated
        into a single table. Point-cloud streams have their own readers — this
        never touches them.
        """
        from ..exceptions import CyberwaveError

        paths = self.local_paths.get(_SOURCE_ACTUATION) or []
        parquets = sorted(p for p in paths if p.suffix.lower() == ".parquet")
        if not parquets:
            raise CyberwaveError(
                "No robot parquet downloaded for this recording. Either this fetch "
                "used a path filter that excluded it (get(path=...)), or the server "
                "is still materializing it — in which case call get() again in a "
                "few minutes."
            )
        pq = _require("pyarrow.parquet")
        pa = _require("pyarrow")
        tables = [pq.read_table(str(p)) for p in parquets]
        return tables[0] if len(tables) == 1 else pa.concat_tables(tables)

    def _read_depth(self) -> list[dict[str, Any]]:
        """Return depth frames as ``{timestamp_us, frame}`` (uint16 ``H×W``)."""
        return self._read_pc_parquet(_SOURCE_POINTCLOUD)

    def _read_pointcloud(self) -> list[dict[str, Any]]:
        """Return point frames as ``{timestamp_us, frame}`` (float32 ``N×cols``)."""
        return self._read_pc_parquet(_SOURCE_COLORED_POINTCLOUD)

    def _show_video_ctx(self) -> None:
        """Play the recording's video in a window (``cv2``)."""
        from ..exceptions import CyberwaveError

        videos = self._paths_with_ext(".mp4")
        if not videos:
            raise CyberwaveError(
                "This recording has a video stream but no .mp4 was downloaded "
                "locally. Either this fetch used a path filter that excluded it "
                "(get(path=...)), or the server is still materializing it — in "
                "which case call get() again in a few minutes."
            )
        return self._show_video(videos[0])

    def info(self) -> dict[str, Any]:
        """Summarize the recording: identity, stream types, and the accessors
        that apply to it (``read_robot``/``read_depth``/``read_pointcloud``/
        ``show_video``)."""
        return {
            "uuid": self.uuid,
            "twin_uuid": self.twin_uuid,
            "environment_uuid": self.environment_uuid,
            "types": sorted(t.value for t in self.types),
            "accessors": self._available_context_methods(),
        }

    def _twin_data(self) -> dict[str, Any]:
        items = getattr(self.signed_urls, "items", None)
        return getattr(items, "twin_data", None) or {}

    def _first_source(self, key: str) -> dict[str, Any]:
        for data in self._twin_data().values():
            if isinstance(data, dict) and isinstance(data.get(key), dict):
                return data[key]
        return {}

    def _show_video(self, path: Path) -> None:
        from ..exceptions import CyberwaveError

        cv2 = _require("cv2")
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            delay = max(1, int(1000 / fps))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                try:
                    cv2.imshow(f"recording {self.uuid}", frame)
                    key = cv2.waitKey(delay)
                except cv2.error as e:
                    # The ``data`` extra ships headless OpenCV (no GUI), so
                    # imshow/waitKey are unavailable. Surface an actionable hint
                    # instead of a cryptic "function not implemented" error.
                    raise CyberwaveError(
                        "GUI video playback requires a non-headless OpenCV build. "
                        "Install it with: pip install opencv-python "
                        "(the 'cyberwave[data]' extra ships opencv-python-headless "
                        "for decoding only)."
                    ) from e
                if key & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # headless build: no windows to destroy
        return None


class RecordingManager:
    """List and fetch recordings for an environment."""

    types = RecordingType

    def __init__(self, api: Any) -> None:
        self.api = api

    def _latest_available_date(
        self,
        environment_id: str,
        *,
        include_unready: bool,
        twin_uuids: "list[str] | None",
    ) -> "date | None":
        """Return the most recent calendar day (UTC) that has recordings.

        Mirrors the replay calendar picker: ask the storage-free availability
        endpoint which days exist before paging any of them. ``None`` means the
        environment (under these filters) has no recordings at all.
        """
        from ..exceptions import CyberwaveError

        try:
            availability = self.api.src_app_api_environments_recordings_get_environment_recordings_availability(
                environment_id,
                include_unready=include_unready,
                twin_uuid=twin_uuids,
            )
        except Exception as e:  # noqa: BLE001
            raise CyberwaveError(
                "Failed to read recording availability for environment "
                f"{environment_id}: {e}"
            ) from e
        # The availability contract types its bounds as ISO strings, while the
        # catalog endpoint wants real dates — normalize rather than relying on
        # incidental coercion.
        return _parse_date_filter(getattr(availability, "last_date", None))

    def list(
        self,
        environment_id: str,
        *,
        filter: "Union[RecordingType, str, Iterable[Union[RecordingType, str]], None]" = None,  # noqa: A002
        start: "date | datetime | str | None" = None,
        end: "date | datetime | str | None" = None,
        include_unready: bool = False,
        limit: int = DEFAULT_LIST_LIMIT,
        twin_uuids: "list[str] | None" = None,
    ) -> RecordingList:
        """List recordings for an environment, optionally filtered by type.

        ``start``/``end`` accept a ``date``, a ``datetime``, or an ISO 8601
        string (e.g. ``"2026-07-01"`` or ``"2026-07-01T10:30:00Z"``) and are
        inclusive calendar-day bounds. With neither given, this asks the
        availability endpoint which days have recordings and lists the most
        recent one — the same narrow-then-fetch flow the replay calendar picker
        uses, and the reason listing a busy environment no longer tries to
        return its whole history in one response.

        ``limit`` caps how many recordings are returned; pages are fetched
        ``CATALOG_PAGE_SIZE`` at a time until the cap is reached or the catalog
        runs out. ``limit=0`` follows every page. ``filter`` is applied to the
        fetched rows afterwards, so it narrows the result rather than widening
        the search.

        ``include_unready`` defaults to ``False``, matching the replay picker;
        set it to ``True`` to include materializing and failed rows.
        ``twin_uuids`` restricts both the availability lookup and the pages to
        those twins server-side.

        Both narrowings log a warning (visible without any logging setup) so a
        partial result never looks like the whole catalog: one naming the day
        that was listed, one when ``limit`` stopped the walk while the server
        still had pages.

        Defaults: ``start=None`` and ``end=None`` select the latest available
        UTC day; ``include_unready=False`` selects ready rows only; and
        ``limit=200`` returns at most 200 rows. Each HTTP page is 50 rows.
        ``limit=0`` follows every page in the selected day/range. The public
        ``filter`` is applied after pagination, so it does not alter the server
        search window or make the result complete beyond ``limit``.
        """
        from ..exceptions import CyberwaveError

        if limit < 0:
            raise CyberwaveError(
                "limit must be 0 (fetch every page) or a positive number of "
                f"recordings, got {limit}"
            )

        start_date = _parse_date_filter(start)
        end_date = _parse_date_filter(end)
        if (start_date is None) != (end_date is None):
            raise CyberwaveError(
                "Both 'start' and 'end' must be provided together to filter "
                "recordings by date — a one-sided date window is ignored."
            )
        if start_date is None:
            start_date = end_date = self._latest_available_date(
                environment_id,
                include_unready=include_unready,
                twin_uuids=twin_uuids,
            )
            if start_date is None:
                logger.warning(
                    "recordings.list(): environment %s has no recordings under "
                    "these filters.",
                    environment_id,
                )
                return RecordingList()
            # Warn rather than log quietly: a caller who expects the whole
            # history gets one day, and nothing else in the result says so.
            logger.warning(
                "recordings.list(): no start/end given, so only %s is listed — "
                "the most recent day with recordings. Pass start=/end= to list "
                "a wider window.",
                start_date.isoformat(),
            )

        window = (
            start_date.isoformat()
            if start_date == end_date
            else f"{start_date.isoformat()}..{end_date.isoformat()}"  # type: ignore[union-attr]
        )
        items = RecordingList()
        cursor: str | None = None
        truncated_by_limit = False
        while True:
            remaining = (limit - len(items)) if limit else CATALOG_PAGE_SIZE
            page_limit = min(remaining, CATALOG_PAGE_SIZE)
            try:
                resp = self.api.src_app_api_environments_recordings_get_environment_recordings(
                    environment_id,
                    start_date=start_date,
                    end_date=end_date,
                    include_unready=include_unready,
                    limit=page_limit,
                    cursor=cursor,
                    twin_uuid=twin_uuids,
                )
            except Exception as e:  # noqa: BLE001
                if _is_probable_recording_payload_too_large(e):
                    from ..exceptions import RecordingPayloadTooLargeError

                    trace_id = _response_header(e, "x-cloud-trace-context")
                    trace_hint = f" Cloud trace: {trace_id}." if trace_id else ""
                    raise RecordingPayloadTooLargeError(
                        "Recording catalog payload is too large for the API "
                        f"gateway for {window}. Use a shorter start/end interval "
                        f"or set limit below {page_limit}, then retry.{trace_hint}",
                        environment_id=environment_id,
                        start_date=start_date,
                        end_date=end_date,
                        page_limit=page_limit,
                        trace_id=trace_id,
                    ) from e
                raise CyberwaveError(
                    f"Failed to list recordings for environment {environment_id}: {e}"
                ) from e
            page = [
                RecordingListItem._from_rest(obj)
                for obj in (getattr(resp, "items", None) or [])
            ]
            for item in page:
                # Attach so item.get() works without the caller having to hold
                # onto the manager separately.
                object.__setattr__(item, "_manager", self)
                object.__setattr__(item, "_environment_id", environment_id)
            items.extend(page)

            next_cursor = getattr(resp, "next_cursor", None)
            server_has_more = bool(getattr(resp, "has_more", False))
            # An empty page, one that advertises more without handing back a
            # cursor, or one that hands back the cursor we just used cannot be
            # advanced past — stop instead of re-requesting it forever.
            can_advance = bool(page and next_cursor and next_cursor != cursor)
            if not server_has_more or not can_advance:
                if server_has_more:
                    logger.warning(
                        "recordings.list(): the server reports more recordings "
                        "for %s but returned no usable next cursor; stopping "
                        "with %d.",
                        window,
                        len(items),
                    )
                break
            if limit and len(items) >= limit:
                truncated_by_limit = True
                break
            cursor = next_cursor

        if truncated_by_limit:
            logger.warning(
                "recordings.list(): stopped at limit=%d and more recordings are "
                "available for %s. Raise limit= or pass limit=0 to fetch every "
                "page.",
                limit,
                window,
            )

        if filter is not None:
            items = items.filter(filter)
        return items

    def get(
        self,
        recording: "RecordingListItem | str",
        *,
        environment_id: str | None = None,
        path: str | None = None,
        twin_uuid: str | None = None,
        max_workers: int | None = None,
    ) -> Recording:
        """Fetch a recording's signed URLs and download all artifacts locally.

        ``twin_uuid``, when given, restricts the downloaded artifacts to that
        twin's entry in the (possibly multi-twin) recording envelope — used by
        ``TwinRecordingsHandle.get()`` so a twin-scoped fetch never mixes in
        another twin's files from the same shared recording.

        ``max_workers`` bounds how many artifacts download concurrently
        (default :data:`DEFAULT_DOWNLOAD_WORKERS`); ``1`` restores a strictly
        serial fetch. Concurrency helps in proportion to how many parts a
        recording has — a single large parquet is one object on one connection
        and gains nothing.
        """
        from ..exceptions import CyberwaveError

        if isinstance(recording, RecordingListItem):
            env = environment_id or recording.environment_uuid
            rec_uuid = recording.uuid
            base_types = set(recording.types)
        else:
            rec_uuid = str(recording)
            env = environment_id
            base_types = set()
        if not env:
            raise CyberwaveError(
                "environment_id is required when passing a recording uuid"
            )
        # Reject rather than coerce: ``max_workers=0`` reads as "no concurrency"
        # but would fall through the ``or`` below to the default and start eight
        # threads — the opposite of what the caller asked for.
        if max_workers is not None and max_workers < 1:
            raise CyberwaveError(
                f"max_workers must be at least 1 (got {max_workers}); "
                "pass 1 for a serial download or None for the default"
            )

        try:
            result = self.api.src_app_api_environments_recordings_get_recording_data(
                env, rec_uuid, return_flatbuffers=False
            )
        except Exception as e:  # noqa: BLE001
            raise CyberwaveError(f"Failed to fetch recording {rec_uuid}: {e}") from e

        # The endpoint answers 200 with a playback envelope or 202 with a
        # materializing body, and the two are distinct types. The 200 envelope
        # carries no readiness field at all, so this must be a type check —
        # duck-typing on ``playback_readiness`` silently never fires.
        if isinstance(result, RecordingMaterializingSchema):
            raise CyberwaveError(_materializing_message(result))
        if result is None:
            raise CyberwaveError(
                "The server returned no playback envelope for this recording. "
                "Its assets may still be materializing; retry shortly."
            )
        envelope = result

        sources = self._collect_sources(envelope, twin_uuid=twin_uuid)
        self._warn_pointcloud_pending(envelope, twin_uuid=twin_uuid)
        tempdir = tempfile.mkdtemp(prefix="cw-recording-")
        local_paths: dict[str, list[Path]] = {}
        types = set(base_types)
        # Plan every download up front, in the main thread: the path filter is
        # applied here, and ``idx`` still enumerates ALL sources so a filtered
        # fetch produces the same filenames as an unfiltered one. Results are
        # derived from this plan rather than from completion order, which keeps
        # ``local_paths`` deterministic no matter how the threads interleave —
        # ``_paths_with_ext`` does not sort, so completion order would otherwise
        # decide which video ``show_video()`` plays.
        # The index prefix is what makes filename order equal playback order, and
        # the readers lean on it: ``_read_robot`` concatenates parquet parts in
        # ``sorted()`` filename order with no timestamp re-sort. Three digits stop
        # sorting lexically at a thousand artifacts ("1000_" < "999_"), which a
        # long segmented recording can reach, so widen the pad to fit the count —
        # while keeping the usual 3 digits so filenames stay stable for callers
        # that cache them.
        pad = max(3, len(str(max(len(sources) - 1, 0))))
        plan = [
            (source, url, Path(tempdir) / f"{idx:0{pad}d}_{name}")
            for idx, (source, url, name) in enumerate(sources)
            if path is None or path in url or path in name
        ]
        if max_workers and max_workers > MAX_DOWNLOAD_WORKERS:
            logger.warning(
                "max_workers=%d exceeds the %d-worker ceiling; using %d",
                max_workers,
                MAX_DOWNLOAD_WORKERS,
                MAX_DOWNLOAD_WORKERS,
            )
        # min() also collapses an empty plan to the serial branch (max() floors it).
        workers = max(
            1,
            min(
                max_workers or DEFAULT_DOWNLOAD_WORKERS,
                len(plan),
                MAX_DOWNLOAD_WORKERS,
            ),
        )
        try:
            if workers == 1:
                for _source, url, dest in plan:
                    self._download(url, dest)
            else:
                logger.debug(
                    "Downloading %d artifact(s) for recording %s with %d workers",
                    len(plan),
                    rec_uuid,
                    workers,
                )
                # Size the shared pool BEFORE the workers start, so a caller who
                # raised ``max_workers`` past the default does not spend the win
                # on rebuilt connections.
                _get_http_pool(workers)
                cancel = threading.Event()
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="cw-artifact"
                ) as pool:
                    futures = [
                        pool.submit(self._download, url, dest, cancel=cancel)
                        for _source, url, dest in plan
                    ]
                    try:
                        # Completion order, not plan order: it surfaces a failure
                        # as early as possible so fewer queued parts get started.
                        # The cost is that with several failures, which one is
                        # reported depends on timing.
                        for future in as_completed(futures):
                            future.result()
                    except BaseException:
                        # Drop what has not started, tell what IS running to stop
                        # at its next chunk, then wait for it. The rmtree below must
                        # not race a worker mid-write: it would either blow up inside
                        # the worker or let the worker recreate the directory after
                        # the tree was removed, leaking exactly the orphan this
                        # cleanup exists to prevent.
                        #
                        # The cancel flag is what keeps that wait short. urllib3's
                        # read timeout would NOT bound it: that timer measures
                        # inactivity and resets on every chunk, so a healthy
                        # transfer of a large artifact would hold the wait open for
                        # its full remaining duration — minutes on a big parquet,
                        # which reads as a hung Ctrl-C.
                        cancel.set()
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
            for source, _url, dest in plan:
                local_paths.setdefault(source, []).append(dest)
                if source in _SOURCE_TO_TYPE:
                    types.add(_SOURCE_TO_TYPE[source])
        except BaseException:
            # No Recording is returned on failure, so nothing will ever call
            # ``.close()`` to reap ``tempdir`` (with its partial artifacts).
            # Clean it up here before re-raising so the download is all-or-nothing.
            shutil.rmtree(tempdir, ignore_errors=True)
            raise

        # A camera recording whose segment MP4 derivatives are not built yet
        # yields no video file. Surface that instead of returning a silently
        # video-less Recording. A ``path`` filter legitimately excludes streams,
        # so only warn when the caller did not filter.
        if (
            path is None
            and RecordingType.CAMERA in base_types
            and not local_paths.get(_SOURCE_CAMERA)
        ):
            logger.warning(
                "Recording %s has a camera stream but no downloadable video yet "
                "(segments may still be converting to MP4); call get() again "
                "in a few minutes.",
                rec_uuid,
            )

        # Prefer the list item's owning twin; otherwise keep the caller-supplied
        # twin_uuid (e.g. from TwinRecordingsHandle.get) instead of dropping it.
        resolved_twin_uuid = (
            recording.twin_uuid
            if isinstance(recording, RecordingListItem)
            else twin_uuid
        )
        # Scope the envelope handed to the Recording to the same twin used when
        # collecting/downloading sources. Otherwise the presence predicates
        # (_has_robot/_has_depth/_has_colored) would scan ALL twins and surface
        # accessors for another twin's stream that was never downloaded — then
        # fail with a misleading "still materializing, call get() again" hint.
        signed_urls = self._scope_envelope_to_twin(envelope, twin_uuid)
        return Recording(
            uuid=rec_uuid,
            twin_uuid=resolved_twin_uuid,
            environment_uuid=env,
            types=frozenset(types),
            signed_urls=signed_urls,
            local_paths=local_paths,
            tempdir=tempdir,
        )

    @staticmethod
    def _scope_envelope_to_twin(envelope: Any, twin_uuid: str | None) -> Any:
        """Return an envelope whose ``items.twin_data`` is narrowed to ``twin_uuid``.

        When ``twin_uuid`` is ``None`` the envelope is returned unchanged (the
        env-scoped, all-twins case). The Recording only reads
        ``signed_urls.items.twin_data``, so a lightweight namespace suffices.
        """
        if twin_uuid is None:
            return envelope
        items = getattr(envelope, "items", None)
        twin_data = getattr(items, "twin_data", None) or {}
        scoped = {t: d for t, d in twin_data.items() if str(t) == str(twin_uuid)}
        from types import SimpleNamespace

        return SimpleNamespace(items=SimpleNamespace(twin_data=scoped))

    @staticmethod
    def _collect_sources(
        envelope: Any, *, twin_uuid: str | None = None
    ) -> list[tuple[str, str, str]]:
        """Flatten the recording-data envelope into (source, url, filename).

        With ``twin_uuid`` set, only that twin's entry is flattened — otherwise
        every twin in the (possibly multi-twin) envelope is included.
        """
        out: list[tuple[str, str, str]] = []
        items = getattr(envelope, "items", None)
        twin_data = getattr(items, "twin_data", None) or {}
        if twin_uuid is not None:
            twin_data = {
                t: d for t, d in twin_data.items() if str(t) == str(twin_uuid)
            }
        for _twin, data in twin_data.items():
            if not isinstance(data, dict):
                continue
            cam = data.get(_SOURCE_CAMERA)
            if isinstance(cam, dict):
                videos = cam.get("videos") or [{"signed_url": cam.get("signed_url")}]
                for v in _ordered_parts(videos):
                    url = v.get("signed_url")
                    if url:
                        out.append((_SOURCE_CAMERA, url, _filename_from(url, "mp4")))
            act = data.get(_SOURCE_ACTUATION)
            if isinstance(act, dict):
                parts = act.get("parts") or [{"signed_url": act.get("signed_url")}]
                for p in _ordered_parts(parts):
                    url = p.get("signed_url")
                    if url:
                        out.append(
                            (_SOURCE_ACTUATION, url, _filename_from(url, "parquet"))
                        )
            for pc_key in (_SOURCE_POINTCLOUD, _SOURCE_COLORED_POINTCLOUD):
                pc = data.get(pc_key)
                if not isinstance(pc, dict):
                    continue
                if pc.get("format") == "parquet":
                    status = pc.get("status")
                    url = pc.get("signed_url")
                    if status == "generating" or not url:
                        # Not ready yet; nothing to download this call.
                        continue
                    out.append((pc_key, url, _filename_from(url, "parquet")))
                else:
                    for url in pc.get("signed_urls") or []:
                        if url:
                            out.append((pc_key, url, _filename_from(url, "fb")))
            aud = data.get(_SOURCE_AUDIO)
            if isinstance(aud, dict):
                clips = aud.get("clips") or [{"signed_url": aud.get("signed_url")}]
                for c in _ordered_parts(clips):
                    url = c.get("signed_url")
                    if url:
                        out.append((_SOURCE_AUDIO, url, _filename_from(url, "mp3")))
        return out

    @staticmethod
    def _warn_pointcloud_pending(envelope: Any, *, twin_uuid: str | None = None) -> None:
        """Log any generating/updating parquet streams so callers know to retry."""
        items = getattr(envelope, "items", None)
        twin_data = getattr(items, "twin_data", None) or {}
        for t, data in twin_data.items():
            if twin_uuid is not None and str(t) != str(twin_uuid):
                continue
            if not isinstance(data, dict):
                continue
            for pc_key in (_SOURCE_POINTCLOUD, _SOURCE_COLORED_POINTCLOUD):
                pc = data.get(pc_key)
                if isinstance(pc, dict) and pc.get("status") in (
                    "generating",
                    "updating",
                ):
                    logger.warning(
                        "Recording %s pointcloud (%s) not final: %s",
                        t,
                        pc_key,
                        pc.get("message", ""),
                    )

    @staticmethod
    def _download(
        url: str, dest: Path, *, cancel: threading.Event | None = None
    ) -> None:
        """Stream one artifact to ``dest``.

        ``cancel``, when a concurrent fetch has already failed elsewhere, lets a
        transfer abandon itself at the next chunk boundary instead of running to
        completion into a temp dir that is about to be deleted.
        """
        from ..exceptions import CyberwaveError

        if cancel is not None and cancel.is_set():
            raise CyberwaveError(f"Download abandoned before starting: {url}")

        # Reuse one process-wide pool. A fresh PoolManager per artifact never gets
        # closed, so its keep-alive socket lingers until GC — downloading many
        # streams/parts (or repeated get() calls) would accumulate open fds. The
        # shared pool bounds connections and lets release_conn() recycle them.
        #
        # Ask for the minimum this transfer needs — one connection. Only the
        # planner in ``get()`` knows a fetch's concurrency, and it sizes the pool
        # before the workers start; asking for the default here would grow the pool
        # straight back to 8 on the first chunk of any fetch planned with fewer
        # workers than that, discarding the pool that call had just built.
        http = _get_http_pool(1)
        response = http.request(
            "GET",
            url,
            preload_content=False,
            timeout=urllib3.Timeout(connect=10.0, read=60.0),
        )
        try:
            if response.status >= 400:
                raise CyberwaveError(
                    f"Recording artifact download failed with HTTP {response.status}"
                )
            with open(dest, "wb") as fh:
                for chunk in response.stream(1024 * 1024):
                    if cancel is not None and cancel.is_set():
                        raise CyberwaveError(f"Download cancelled mid-stream: {url}")
                    fh.write(chunk)
        finally:
            response.release_conn()


class TwinRecordingsHandle:
    """``twin.recordings`` — recordings scoped to a single twin."""

    def __init__(self, twin: Any) -> None:
        self._twin = twin

    @property
    def types(self) -> type[RecordingType]:
        return RecordingType

    def _manager(self) -> RecordingManager:
        return self._twin.client.environments.recordings

    def list(
        self,
        *,
        filter: "Union[RecordingType, str, Iterable[Union[RecordingType, str]], None]" = None,  # noqa: A002
        start: "date | datetime | str | None" = None,
        end: "date | datetime | str | None" = None,
        include_unready: bool = False,
        limit: int = DEFAULT_LIST_LIMIT,
    ) -> RecordingList:
        """List this twin's recordings using :meth:`RecordingManager.list` defaults.

        No date range selects this twin's latest available UTC day;
        ``include_unready`` defaults to ``False``; and ``limit`` defaults to
        200 (or uses ``0`` for every page in the selected day/range). The twin
        restriction is applied server-side before availability and pagination.
        """
        # Narrow server-side: a page of environment-wide rows can legitimately
        # contain none of this twin's, so filtering only after the fetch would
        # make a bounded read look empty.
        all_items = self._manager().list(
            self._twin.environment_id,
            start=start,
            end=end,
            include_unready=include_unready,
            limit=limit,
            twin_uuids=[str(self._twin.uuid)],
        )
        mine = RecordingList(
            item for item in all_items if item.twin_uuid == str(self._twin.uuid)
        )
        if filter is not None:
            mine = mine.filter(filter)
        return mine

    def get(
        self,
        recording: "RecordingListItem | str",
        *,
        path: str | None = None,
        max_workers: int | None = None,
    ) -> Recording:
        return self._manager().get(
            recording,
            environment_id=self._twin.environment_id,
            path=path,
            twin_uuid=str(self._twin.uuid),
            max_workers=max_workers,
        )

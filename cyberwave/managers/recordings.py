"""Recording retrieval + local inspection.

Entry points:
    twin.recordings.list()/.get()             (twin-scoped)
    cw.environments.recordings.list(environment_id=...)/.get()  (env-scoped)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import weakref
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Union
from urllib.parse import urlparse

import urllib3

logger = logging.getLogger(__name__)

# Process-wide connection pool for artifact downloads, created lazily on first
# use. Shared (rather than one-per-download) so keep-alive sockets are recycled
# instead of leaking until GC.
_http_pool: urllib3.PoolManager | None = None


def _get_http_pool() -> urllib3.PoolManager:
    global _http_pool
    if _http_pool is None:
        _http_pool = urllib3.PoolManager()
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
    """Derive the set of source types a recording carries from list metadata."""
    types: set[RecordingType] = set()
    rec_type = str(metadata.get("recording_type") or "")
    # Camera: finalized recordings expose "camera*"/mp4_path; active (in-progress)
    # manifests instead carry a non-empty video_parts list on a
    # "recording_type": "active" item — classify those as CAMERA too so that
    # filtering does not silently drop recordings that are still in progress.
    if rec_type.startswith("camera") or metadata.get("mp4_path") or metadata.get("video_parts"):
        types.add(RecordingType.CAMERA)
    # Robot: finalized robot metadata, or an active manifest carrying robot_parts.
    if (
        metadata.get("metadata_type") == "TwinRecordingMetadata"
        or rec_type == "robot"
        or metadata.get("robot_parts")
    ):
        types.add(RecordingType.ROBOT)
    if metadata.get("pointcloud"):
        types.add(RecordingType.DEPTH)
    if metadata.get("colored_pointcloud"):
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

    @property
    def types(self) -> frozenset[RecordingType]:
        return _classify(self.metadata)

    @classmethod
    def _from_rest(cls, obj: Any) -> "RecordingListItem":
        twin_uuid = getattr(obj, "twin_uuid", None)
        return cls(
            uuid=str(obj.uuid),
            twin_uuid=str(twin_uuid) if twin_uuid else None,
            environment_uuid=str(obj.environment_uuid),
            metadata=dict(getattr(obj, "metadata", None) or {}),
        )

    def get(self, *, path: str | None = None) -> Any:
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
                "No parquet downloaded for this stream — the server may still be "
                "materializing it; call get() again in a few minutes."
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
                "No robot parquet downloaded for this recording — the server may "
                "still be materializing it; call get() again in a few minutes."
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
                "Video stream is present but not downloaded locally for this recording."
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

    def list(
        self,
        environment_id: str,
        *,
        filter: "Union[RecordingType, str, Iterable[Union[RecordingType, str]], None]" = None,  # noqa: A002
        start: "date | datetime | str | None" = None,
        end: "date | datetime | str | None" = None,
    ) -> RecordingList:
        """List recordings for an environment, optionally filtered by type.

        ``start``/``end`` accept a ``date``, a ``datetime``, or an ISO 8601
        string (e.g. ``"2026-07-01"`` or ``"2026-07-01T10:30:00Z"``) and are
        inclusive calendar-day bounds.
        """
        from ..exceptions import CyberwaveError

        start_date = _parse_date_filter(start)
        end_date = _parse_date_filter(end)
        if (start_date is None) != (end_date is None):
            raise CyberwaveError(
                "Both 'start' and 'end' must be provided together to filter "
                "recordings by date — a one-sided date window is ignored."
            )
        try:
            resp = self.api.src_app_api_environments_recordings_get_environment_recordings(
                environment_id, start_date, end_date
            )
        except Exception as e:  # noqa: BLE001
            raise CyberwaveError(
                f"Failed to list recordings for environment {environment_id}: {e}"
            ) from e
        items = RecordingList(
            RecordingListItem._from_rest(obj)
            for obj in (getattr(resp, "items", None) or [])
        )
        for item in items:
            # Attach so item.get() works without the caller having to hold onto
            # the manager separately.
            object.__setattr__(item, "_manager", self)
            object.__setattr__(item, "_environment_id", environment_id)
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
    ) -> Recording:
        """Fetch a recording's signed URLs and download all artifacts locally.

        ``twin_uuid``, when given, restricts the downloaded artifacts to that
        twin's entry in the (possibly multi-twin) recording envelope — used by
        ``TwinRecordingsHandle.get()`` so a twin-scoped fetch never mixes in
        another twin's files from the same shared recording.
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

        try:
            envelope = (
                self.api.src_app_api_environments_recordings_get_recording_data(
                    env, rec_uuid, return_flatbuffers=False
                )
            )
        except Exception as e:  # noqa: BLE001
            raise CyberwaveError(f"Failed to fetch recording {rec_uuid}: {e}") from e

        sources = self._collect_sources(envelope, twin_uuid=twin_uuid)
        self._warn_pointcloud_pending(envelope, twin_uuid=twin_uuid)
        tempdir = tempfile.mkdtemp(prefix="cw-recording-")
        local_paths: dict[str, list[Path]] = {}
        types = set(base_types)
        try:
            for idx, (source, url, name) in enumerate(sources):
                if path is not None and path not in url and path not in name:
                    continue
                dest = Path(tempdir) / f"{idx:03d}_{name}"
                self._download(url, dest)
                local_paths.setdefault(source, []).append(dest)
                if source in _SOURCE_TO_TYPE:
                    types.add(_SOURCE_TO_TYPE[source])
        except BaseException:
            # No Recording is returned on failure, so nothing will ever call
            # ``.close()`` to reap ``tempdir`` (with its partial artifacts).
            # Clean it up here before re-raising so the download is all-or-nothing.
            shutil.rmtree(tempdir, ignore_errors=True)
            raise

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
                for v in videos:
                    url = v.get("signed_url")
                    if url:
                        out.append((_SOURCE_CAMERA, url, _filename_from(url, "mp4")))
            act = data.get(_SOURCE_ACTUATION)
            if isinstance(act, dict):
                parts = act.get("parts") or [{"signed_url": act.get("signed_url")}]
                for p in parts:
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
                for c in clips:
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
    def _download(url: str, dest: Path) -> None:
        from ..exceptions import CyberwaveError

        # Reuse one process-wide pool. A fresh PoolManager per artifact never gets
        # closed, so its keep-alive socket lingers until GC — downloading many
        # streams/parts (or repeated get() calls) would accumulate open fds. The
        # shared pool bounds connections and lets release_conn() recycle them.
        http = _get_http_pool()
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
    ) -> RecordingList:
        all_items = self._manager().list(self._twin.environment_id, start=start, end=end)
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
    ) -> Recording:
        return self._manager().get(
            recording,
            environment_id=self._twin.environment_id,
            path=path,
            twin_uuid=str(self._twin.uuid),
        )

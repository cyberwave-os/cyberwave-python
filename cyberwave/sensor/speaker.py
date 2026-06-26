"""Speaker (audio output) streaming — consumer-side counterpart to ``microphone.py``.

Three sources feed the same :class:`HostSpeakerCapture`:

* File (any PyAV container)
* WebRTC remote audio track (via :class:`SpeakerAudioStreamer`)
* Zenoh PCM published by a peer microphone driver

The base WebRTC/MQTT signalling classes are imported from ``microphone.py``
unchanged.

Example::

    speaker = HostSpeakerCapture()
    speaker.start()
    speaker.play_file("/path/to/cue.mp3")

    streamer = SpeakerAudioStreamer(cw.mqtt, twin_uuid="...")
    streamer.set_zenoh_source(
        data_bus=cw.data, channel="audio/default",
        source_twin_uuid="mic-twin-uuid",
    )
    await streamer.start()
"""

from __future__ import annotations

import asyncio
import contextlib
import fractions
import logging
import platform
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

import numpy as np
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from .audio_resample import InboundAudioAdapter, infer_webrtc_frame_channels
from .microphone import (
    AUDIO_PTIME,
    DEFAULT_AUDIO_RECORDING,
    DEFAULT_AUDIO_SENSOR_ID,
    DEFAULT_AUTO_RECONNECT,
    DEFAULT_LAYOUT,
    DEFAULT_SAMPLE_RATE,
    BaseAudioStreamer,
    BaseAudioTrack,
)

if TYPE_CHECKING:
    from ..data.api import DataBus
    from ..mqtt_client import CyberwaveMQTTClient

logger = logging.getLogger(__name__)


DEFAULT_SPEAKER_NAME = DEFAULT_AUDIO_SENSOR_ID
DEFAULT_SPEAKER_FRONTEND_TYPE = "speaker"
DEFAULT_SPEAKER_BIT_DEPTH = 16

# Passive speaker twin sensor *types* — edge/frontend consumers only.
# Keep in sync with ``SPEAKER_SENSOR_TYPES`` in ``audio_sensor.rs``.
SPEAKER_SENSOR_TYPES = frozenset(
    {"speaker", "loudspeaker", "speakerphone", "audio_out"}
)


def _get_sounddevice_module() -> Any | None:
    try:
        import sounddevice as sd  # type: ignore[import]

        return sd
    except Exception:
        return None


def list_host_sound_devices(
    *, kind: str | None = None
) -> tuple[list[dict[str, Any]], int | None]:
    """Return ``(devices, default_index)`` for host sound devices.

    ``kind`` filters to ``"input"`` or ``"output"``; ``None`` returns both.
    The default index is for the requested kind (output when ``kind is None``).
    """
    sd = _get_sounddevice_module()
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed; install with: pip install 'cyberwave[speaker]'"
        )

    raw_devices = sd.query_devices()
    default_device = sd.default.device
    default_output_candidate = (
        default_device[1] if isinstance(default_device, (list, tuple)) and len(default_device) > 1 else default_device
    )
    default_input_candidate = (
        default_device[0] if isinstance(default_device, (list, tuple)) and default_device else default_device
    )
    default_output_index = (
        int(default_output_candidate)
        if isinstance(default_output_candidate, int) and default_output_candidate >= 0
        else None
    )
    default_input_index = (
        int(default_input_candidate)
        if isinstance(default_input_candidate, int) and default_input_candidate >= 0
        else None
    )

    devices: list[dict[str, Any]] = []
    for index, device in enumerate(raw_devices if isinstance(raw_devices, (list, tuple)) else [raw_devices]):
        if not isinstance(device, dict):
            continue
        max_in = int(device.get("max_input_channels", 0) or 0)
        max_out = int(device.get("max_output_channels", 0) or 0)
        if kind == "input" and max_in <= 0:
            continue
        if kind == "output" and max_out <= 0:
            continue
        if kind is None and max_in <= 0 and max_out <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": str(device.get("name", f"device-{index}")),
                "max_input_channels": max_in,
                "max_output_channels": max_out,
                "default_samplerate": float(device.get("default_samplerate", 0.0) or 0.0),
                "hostapi": int(device.get("hostapi", -1) or -1),
            }
        )

    default_index = default_output_index if kind != "input" else default_input_index
    if default_index is not None and not any(d["index"] == default_index for d in devices):
        default_index = devices[0]["index"] if devices else None
    return devices, default_index


def check_host_speaker_settings(
    *,
    device: int | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
) -> None:
    """Validate that the host speaker supports the given output settings."""
    sd = _get_sounddevice_module()
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed; install with: pip install 'cyberwave[speaker]'"
        )
    sd.check_output_settings(
        device=device,
        channels=channels,
        samplerate=sample_rate,
        dtype="int16",
    )


def query_supported_output_sample_rates(
    *,
    device: int | None,
    channels: int,
    candidate_rates: list[int] | None = None,
) -> list[int]:
    """Probe a speaker device for accepted output sample rates."""
    sd = _get_sounddevice_module()
    if sd is None:
        raise RuntimeError("sounddevice is not installed; cannot query output rates")
    if candidate_rates is None:
        candidate_rates = [8_000, 16_000, 22_050, 32_000, 44_100, 48_000, 96_000]
    supported: list[int] = []
    for rate in candidate_rates:
        try:
            sd.check_output_settings(
                device=device,
                channels=channels,
                samplerate=rate,
                dtype="int16",
            )
            supported.append(rate)
        except Exception:
            pass
    return supported


def create_linux_speaker_monitor() -> Any | None:
    """Return a pyudev monitor for sound-subsystem hotplug events (Linux only)."""
    if platform.system().lower() != "linux":
        return None
    try:
        import pyudev  # type: ignore[import]
    except Exception:
        return None
    context = pyudev.Context()
    monitor = pyudev.Monitor.from_netlink(context)
    monitor.filter_by(subsystem="sound")
    return monitor


class DSPState:
    """Thread-safe DSP controls: volume, per-channel gain, routing matrix.

    :meth:`process` returns ``float32`` in ``[-1.0, 1.0]`` so the caller can
    scale to any integer bit depth without losing headroom.
    """

    __slots__ = ("_lock", "_channels", "_volume", "_channel_gains", "_routing_matrix")

    def __init__(self, channels: int) -> None:
        self._lock = threading.Lock()
        self._channels = max(1, int(channels))
        self._volume: float = 1.0
        self._channel_gains: np.ndarray = np.ones(self._channels, dtype=np.float32)
        self._routing_matrix: np.ndarray = np.eye(self._channels, dtype=np.float32)

    @property
    def channels(self) -> int:
        return self._channels

    def set_volume(self, volume: float) -> None:
        v = float(max(0.0, min(1.0, volume)))
        with self._lock:
            self._volume = v

    def get_volume(self) -> float:
        with self._lock:
            return self._volume

    def set_channel_gain(self, channel_idx: int, gain: float) -> None:
        g = float(max(0.0, gain))
        with self._lock:
            if 0 <= channel_idx < self._channel_gains.shape[0]:
                self._channel_gains[channel_idx] = g

    def get_channel_gains(self) -> np.ndarray:
        with self._lock:
            return self._channel_gains.copy()

    def configure_routing(self, matrix: Iterable[Iterable[float]]) -> None:
        m = np.asarray(matrix, dtype=np.float32)
        if m.ndim != 2 or m.shape[0] != self._channels:
            raise ValueError(
                f"Routing matrix must be 2D with first dimension == {self._channels}"
            )
        with self._lock:
            self._routing_matrix = m

    def get_routing(self) -> np.ndarray:
        with self._lock:
            return self._routing_matrix.copy()

    def _snapshot(self) -> tuple[float, np.ndarray, np.ndarray]:
        with self._lock:
            return self._volume, self._channel_gains, self._routing_matrix

    def process(self, frames: np.ndarray) -> np.ndarray:
        """Gain → routing → volume → clip. Returns float32 in ``[-1.0, 1.0]``."""
        if frames.size == 0:
            return np.empty((0, self._channels), dtype=np.float32)
        volume, gains, routing = self._snapshot()
        audio = frames.astype(np.float32) * (1.0 / 32768.0)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        if audio.shape[1] != gains.shape[0]:
            in_ch = audio.shape[1]
            if in_ch == 1 and gains.shape[0] > 1:
                audio = np.repeat(audio, gains.shape[0], axis=1)
            elif in_ch > 1 and gains.shape[0] == 1:
                audio = audio.mean(axis=1, keepdims=True)
            else:
                audio = audio[:, : gains.shape[0]]
        np.multiply(audio, gains[np.newaxis, :], out=audio)
        audio = audio @ routing
        if volume != 1.0:
            audio *= volume
        np.clip(audio, -1.0, 1.0, out=audio)
        return audio


class _AudioSource:
    """Pulls 20 ms s16 chunks. ``read()`` runs on the PortAudio realtime thread
    and must be **non-blocking** (return ``None`` for "no data right now").

    :attr:`eof` is set permanently once the source is exhausted or closed.
    """

    def __init__(self) -> None:
        self.eof: threading.Event = threading.Event()

    def read(self) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        self.eof.set()


class _FileAudioSource(_AudioSource):
    """PyAV-decoded file source. Decode runs on a producer thread; the realtime
    callback only pops from a bounded queue.
    """

    _RING_CHUNKS = 16  # ~320 ms look-ahead at 20 ms/chunk

    def __init__(
        self,
        path: str,
        *,
        target_sample_rate: int,
        target_channels: int,
        loop: bool = False,
    ) -> None:
        super().__init__()
        try:
            import av  # type: ignore[import]
        except Exception as exc:  # pragma: no cover - depends on optional native libs
            raise RuntimeError(
                "PyAV is required to decode audio files; install with 'cyberwave[speaker]'"
            ) from exc

        self._av = av
        self._path = path
        self._target_sample_rate = target_sample_rate
        self._target_channels = target_channels
        self._loop = loop
        self._frames_per_chunk = int(AUDIO_PTIME * target_sample_rate)
        self._buffer = np.empty((0, target_channels), dtype=np.int16)
        self._chunks: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self._RING_CHUNKS)
        self._stop = threading.Event()
        self._open()
        self._producer = threading.Thread(
            target=self._produce,
            name=f"speaker-file-decode:{path}",
            daemon=True,
        )
        self._producer.start()

    @property
    def file_path(self) -> str:
        return self._path

    def _open(self) -> None:
        from av.audio.resampler import AudioResampler as PyAVAudioResampler

        self._container = self._av.open(self._path)
        layout = "stereo" if self._target_channels == 2 else "mono"
        self._resampler = PyAVAudioResampler(
            format="s16",
            layout=layout,
            rate=self._target_sample_rate,
        )
        self._stream = next(
            (s for s in self._container.streams if s.type == "audio"), None
        )
        if self._stream is None:
            raise RuntimeError(f"No audio stream found in {self._path!r}")
        self._iter = self._container.decode(self._stream)

    @staticmethod
    def _normalize_frame(arr: np.ndarray, target_channels: int) -> np.ndarray:
        """Coerce a PyAV ``to_ndarray()`` result into ``(samples, target_channels)`` int16.

        PyAV returns ``(samples,)`` for 1-D, ``(channels, samples)`` for planar
        formats, ``(1, samples * channels)`` for packed.
        """
        if arr.dtype != np.int16:
            arr = arr.astype(np.int16, copy=False)
        if arr.ndim == 1:
            return arr.reshape(-1, target_channels)
        if arr.shape[0] == target_channels and arr.shape[0] != arr.shape[1]:
            return np.ascontiguousarray(arr.T)
        return arr.reshape(-1, target_channels)

    def _decode_one(self) -> np.ndarray | None:
        try:
            frame = next(self._iter)
        except StopIteration:
            return None
        out_frames = self._resampler.resample(frame)
        if not out_frames:
            return np.empty((0, self._target_channels), dtype=np.int16)
        chunks: list[np.ndarray] = [
            self._normalize_frame(np.asarray(f.to_ndarray()), self._target_channels)
            for f in out_frames
        ]
        return np.concatenate(chunks, axis=0) if chunks else np.empty(
            (0, self._target_channels), dtype=np.int16
        )

    def _next_chunk(self) -> np.ndarray | None:
        """Produce one ``frames_per_chunk`` chunk; ``None`` when exhausted."""
        while self._buffer.shape[0] < self._frames_per_chunk:
            decoded = self._decode_one()
            if decoded is None:
                if self._loop:
                    with contextlib.suppress(Exception):
                        self._container.close()
                    self._open()
                    continue
                if self._buffer.shape[0] == 0:
                    return None
                pad = np.zeros(
                    (self._frames_per_chunk - self._buffer.shape[0], self._target_channels),
                    dtype=np.int16,
                )
                chunk = np.concatenate([self._buffer, pad], axis=0)
                self._buffer = np.empty((0, self._target_channels), dtype=np.int16)
                return chunk
            if decoded.size:
                self._buffer = np.concatenate([self._buffer, decoded], axis=0)
        chunk = self._buffer[: self._frames_per_chunk]
        self._buffer = self._buffer[self._frames_per_chunk :]
        return chunk

    def _produce(self) -> None:
        # Container + iterator are owned by this thread for their entire
        # lifetime: closing them from another thread while PyAV is mid-
        # ``next(self._iter)`` segfaults on some PyAV / Python builds.
        try:
            while not self._stop.is_set():
                try:
                    chunk = self._next_chunk()
                except Exception:
                    logger.exception("File decode failed for %r; closing source", self._path)
                    break
                if chunk is None:
                    break
                while not self._stop.is_set():
                    try:
                        self._chunks.put(chunk, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            # Poison pill — consumer sets ``eof`` only when this is popped so
            # blocking callers don't return before the last chunk has played.
            while not self._stop.is_set():
                try:
                    self._chunks.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            with contextlib.suppress(Exception):
                self._container.close()
            if self._stop.is_set():
                self.eof.set()

    def read(self) -> np.ndarray | None:
        try:
            chunk = self._chunks.get_nowait()
        except queue.Empty:
            return None
        if chunk is None:
            self.eof.set()
            return None
        return chunk

    def close(self) -> None:
        # Signal stop and drain the chunk ring so the producer can unblock
        # from ``put()``.  Container teardown is deferred to the producer
        # thread itself (see ``_produce``) — closing it here would race
        # against ``next(self._iter)`` and can segfault.
        self._stop.set()
        with contextlib.suppress(queue.Empty):
            while True:
                self._chunks.get_nowait()
        if self._producer.is_alive() and self._producer is not threading.current_thread():
            self._producer.join(timeout=2.0)
        self.eof.set()


class _QueueAudioSource(_AudioSource):
    """Bounded-queue source fed by WebRTC or Zenoh subscribers. Non-blocking read."""

    def __init__(self, *, target_channels: int, max_chunks: int = 64) -> None:
        super().__init__()
        self._target_channels = target_channels
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(1, max_chunks))

    def push(self, chunk: np.ndarray) -> None:
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Drop oldest to keep latency bounded.
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(chunk)

    def read(self) -> np.ndarray | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        with contextlib.suppress(queue.Empty):
            while True:
                self._queue.get_nowait()
        self.eof.set()


class _MixingAudioSource(_AudioSource):
    """Sum-mixing fan-in: each producer feeds an independent bounded queue;
    ``read()`` pops one chunk per producer, aligns lengths, and sums in int32
    before clipping back to int16.

    Producers with nothing ready in a slot contribute silence — no stalling.
    """

    def __init__(
        self,
        *,
        target_channels: int,
        frames_per_chunk: int,
        max_chunks_per_input: int = 32,
    ) -> None:
        super().__init__()
        self._target_channels = target_channels
        self._frames_per_chunk = frames_per_chunk
        self._max_chunks = max(1, max_chunks_per_input)
        self._inputs: dict[Any, queue.Queue[np.ndarray]] = {}
        self._lock = threading.Lock()

    def add_input(self, key: Any) -> Callable[[np.ndarray], None]:
        """Register a producer; returns a ``push(chunk)`` callable."""
        with self._lock:
            q: queue.Queue[np.ndarray] = queue.Queue(maxsize=self._max_chunks)
            self._inputs[key] = q

        def _push(chunk: np.ndarray) -> None:
            try:
                q.put_nowait(chunk)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    q.get_nowait()
                with contextlib.suppress(queue.Full):
                    q.put_nowait(chunk)

        return _push

    def remove_input(self, key: Any) -> None:
        with self._lock:
            self._inputs.pop(key, None)

    def read(self) -> np.ndarray | None:
        with self._lock:
            queues = list(self._inputs.values())
        if not queues:
            return None
        acc: np.ndarray | None = None
        contributors = 0
        for q in queues:
            try:
                chunk = q.get_nowait()
            except queue.Empty:
                continue
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, self._target_channels)
            if chunk.shape[1] != self._target_channels:
                if chunk.shape[1] == 1 and self._target_channels > 1:
                    chunk = np.repeat(chunk, self._target_channels, axis=1)
                elif chunk.shape[1] > 1 and self._target_channels == 1:
                    chunk = np.rint(chunk.mean(axis=1, keepdims=True)).astype(np.int16)
                else:
                    chunk = chunk[:, : self._target_channels]
            if chunk.shape[0] > self._frames_per_chunk:
                chunk = chunk[: self._frames_per_chunk]
            elif chunk.shape[0] < self._frames_per_chunk:
                pad = np.zeros(
                    (self._frames_per_chunk - chunk.shape[0], self._target_channels),
                    dtype=np.int16,
                )
                chunk = np.concatenate([chunk, pad], axis=0)
            if acc is None:
                acc = chunk.astype(np.int32, copy=True)
            else:
                acc += chunk.astype(np.int32, copy=False)
            contributors += 1
        if acc is None or contributors == 0:
            return None
        np.clip(acc, -32768, 32767, out=acc)
        return acc.astype(np.int16, copy=False)

    def close(self) -> None:
        with self._lock:
            self._inputs.clear()
        self.eof.set()


class HostSpeakerCapture:
    """Handle on a host speaker (output) device.

    Single funnel for file/WebRTC/Zenoh sources; DSP is applied here so the
    host sees one canonical layout. Cross-platform via PortAudio. macOS
    bare-metal requires speaker access for the running process.
    """

    # bit_depth → (numpy dtype handed to sounddevice, unused legacy shift).
    # 24-bit lives in an int32 container because numpy has no int24.
    _BIT_DEPTH_TARGETS: dict[int, tuple[str, int]] = {
        16: ("int16", 0),
        24: ("int32", 8),
        32: ("int32", 16),
    }

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        frames_per_chunk: int | None = None,
        device_index: int | None = None,
        bit_depth: int = DEFAULT_SPEAKER_BIT_DEPTH,
    ) -> None:
        if bit_depth not in self._BIT_DEPTH_TARGETS:
            raise ValueError("bit_depth must be 16, 24, or 32")
        if int(sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if int(channels) < 1:
            raise ValueError("channels must be >= 1")
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frames_per_chunk = int(frames_per_chunk or AUDIO_PTIME * self.sample_rate)
        self.device_index = device_index
        self.bit_depth = bit_depth
        self.dsp = DSPState(channels=self.channels)

        self._source: _AudioSource | None = None
        self._source_lock = threading.Lock()
        self._stream: Any | None = None
        self._lock = threading.Lock()
        self._muted: bool = False

        out_dtype_name, self._bit_shift = self._BIT_DEPTH_TARGETS[bit_depth]
        self._out_dtype = np.dtype(out_dtype_name)
        # Preallocated scratch — avoids allocation inside the realtime callback.
        self._scratch = np.empty(
            (self.frames_per_chunk, self.channels), dtype=self._out_dtype
        )
        # Scale float DSP output to the actual signed integer full-scale of
        # the requested bit depth (not just int16 widened into int32).
        self._sample_full_scale: float = float((1 << (bit_depth - 1)) - 1)

    @property
    def is_running(self) -> bool:
        stream = self._stream
        if stream is None:
            return False
        # ``active`` flips False after a PortAudio underrun-driven stop.
        return bool(getattr(stream, "active", True))

    def start(self) -> None:
        """Start host speaker capture. Idempotent."""
        with self._lock:
            stream = self._stream
            if stream is not None:
                if bool(getattr(stream, "active", True)):
                    return
                # PortAudio can leave a stale handle after underrun/stop while
                # ``_stream`` is still set — recreate instead of no-op'ing.
                self._stream = None
                with contextlib.suppress(Exception):
                    stream.stop()
                    stream.close()
            sd = _get_sounddevice_module()
            if sd is None:
                raise RuntimeError(
                    "sounddevice is not installed; install with: pip install 'cyberwave[speaker]'"
                )
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self._out_dtype.name,
                blocksize=self.frames_per_chunk,
                device=self.device_index,
                callback=self._on_audio_callback,
                latency="low",
            )
            try:
                stream.start()
            except Exception:
                with contextlib.suppress(Exception):
                    stream.close()
                raise
            self._stream = stream
            logger.info(
                "Speaker started (device=%s, rate=%dHz, channels=%d, bit_depth=%d, chunk=%d frames)",
                self.device_index if self.device_index is not None else "default",
                self.sample_rate,
                self.channels,
                self.bit_depth,
                self.frames_per_chunk,
            )

    def stop(self) -> None:
        """Stop host speaker capture and clear the active source."""
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
                stream.close()
        self.clear_source()

    def mute(self) -> None:
        """Emit silence without releasing the device or its source."""
        self._muted = True

    def unmute(self) -> None:
        self._muted = False

    @property
    def is_muted(self) -> bool:
        return self._muted

    def pause(self) -> None:
        """Deprecated alias for :meth:`mute` — kept for backwards compatibility."""
        self.mute()

    def resume(self) -> None:
        """Deprecated alias for :meth:`unmute`."""
        self.unmute()

    def set_source(self, source: _AudioSource | None) -> None:
        with self._source_lock:
            previous = self._source
            self._source = source
        # Close outside the lock so a slow ``close()`` can't stall the audio
        # callback (which only holds the lock to swap a pointer).
        if previous is not None and previous is not source:
            with contextlib.suppress(Exception):
                previous.close()

    def clear_source(self) -> None:
        self.set_source(None)

    def _get_active_source(self) -> _AudioSource | None:
        """Locked pointer load — prevents a concurrent ``set_source`` from
        closing the returned source mid-``read()``."""
        with self._source_lock:
            return self._source

    def set_file_source(self, path: str, *, loop: bool = False) -> None:
        self.set_source(
            _FileAudioSource(
                path,
                target_sample_rate=self.sample_rate,
                target_channels=self.channels,
                loop=loop,
            )
        )

    def set_queue_source(self, max_chunks: int = 64) -> _QueueAudioSource:
        src = _QueueAudioSource(target_channels=self.channels, max_chunks=max_chunks)
        self.set_source(src)
        return src

    def play_file(self, path: str, *, loop: bool = False, blocking: bool = False) -> None:
        """Play *path*. Auto-starts the device. ``blocking=True`` waits for EOF
        (driven by the file source's :class:`threading.Event`, no polling)."""
        if not self.is_running:
            self.start()
        source = _FileAudioSource(
            path,
            target_sample_rate=self.sample_rate,
            target_channels=self.channels,
            loop=loop,
        )
        self.set_source(source)
        if not blocking or loop:
            return
        source.eof.wait()
        # Brief drain for PortAudio's internal buffer.
        drain_s = (self.frames_per_chunk / self.sample_rate) * 2
        time.sleep(max(drain_s, 0.02))
        if self._get_active_source() is source:
            self.clear_source()

    def stop_file(self, path: str) -> bool:
        """Stop playback of *path* when it is the active file source.

        Returns ``True`` when the active file source matched *path* and was
        cleared. Returns ``False`` when another source is active or nothing is
        playing.
        """
        src = self._get_active_source()
        if not isinstance(src, _FileAudioSource):
            return False
        try:
            matches = Path(src.file_path).resolve() == Path(path).resolve()
        except OSError:
            matches = src.file_path == path
        if not matches:
            return False
        self.clear_source()
        return True

    def play_chunk(self, chunk: np.ndarray) -> None:
        """Enqueue a one-off PCM chunk. Switches to a queue source if needed."""
        if not self.is_running:
            self.start()
        src = self._source
        if not isinstance(src, _QueueAudioSource):
            src = self.set_queue_source()
        src.push(chunk)

    def set_volume(self, volume: float) -> None:
        self.dsp.set_volume(volume)

    def get_volume(self) -> float:
        return self.dsp.get_volume()

    def set_channel_gain(self, channel_idx: int, gain: float) -> None:
        self.dsp.set_channel_gain(channel_idx, gain)

    def configure_routing(self, matrix: Iterable[Iterable[float]]) -> None:
        self.dsp.configure_routing(matrix)

    def get_physical_info(self) -> dict[str, Any]:
        """Return device + DSP runtime info."""
        info: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "volume": self.dsp.get_volume(),
            "channel_gains": self.dsp.get_channel_gains().tolist(),
            "routing": self.dsp.get_routing().tolist(),
            "device_index": self.device_index,
            "is_running": self.is_running,
        }
        sd = _get_sounddevice_module()
        if sd is not None and self.device_index is not None:
            try:
                device_info = sd.query_devices(self.device_index)
                if isinstance(device_info, dict):
                    info["name"] = device_info.get("name")
                    info["hostapi"] = device_info.get("hostapi")
                    info["max_output_channels"] = device_info.get("max_output_channels")
                    info["default_samplerate"] = device_info.get("default_samplerate")
            except Exception:
                logger.debug("query_devices failed for index=%s", self.device_index, exc_info=True)
        return info

    def _on_audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        if status:
            logger.debug("sounddevice output status: %s", status)
        if self._muted:
            outdata.fill(0)
            return
        src = self._get_active_source()
        chunk: np.ndarray | None = None
        source_eof = False
        if src is not None:
            try:
                chunk = src.read()
            except Exception:
                logger.exception("Speaker source read failed; emitting silence")
                chunk = None
            if chunk is None and src.eof.is_set():
                source_eof = True
        if chunk is None or chunk.size == 0:
            outdata.fill(0)
            if source_eof:
                # Detach the exhausted source so blocking callers can return.
                with self._source_lock:
                    if self._source is src:
                        self._source = None
            return
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        if chunk.shape[1] != self.channels:
            if chunk.shape[1] == 1 and self.channels > 1:
                chunk = np.repeat(chunk, self.channels, axis=1)
            elif chunk.shape[1] > 1 and self.channels == 1:
                chunk = np.rint(chunk.mean(axis=1, keepdims=True)).astype(np.int16)
            else:
                chunk = chunk[:, : self.channels]
        processed = self.dsp.process(chunk)
        n = min(processed.shape[0], frames)
        if n:
            np.multiply(
                processed[:n], self._sample_full_scale, out=self._scratch[:n, :], casting="unsafe"
            )
            outdata[:n] = self._scratch[:n]
        if n < frames:
            outdata[n:] = 0


class SpeakerAudioTrack(BaseAudioTrack):
    """WebRTC audio track for the speaker side.

    Emits 20 ms silence frames to satisfy the SFU's ``m=audio`` expectation;
    the audible audio arrives on the *remote* track and is routed into
    :class:`HostSpeakerCapture` by the streamer's ``on("track")`` handler.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        layout: str | None = None,
        channels: int = 1,
        bit_depth: int = DEFAULT_SPEAKER_BIT_DEPTH,
    ) -> None:
        super().__init__()
        if channels not in (1, 2):
            raise ValueError("SpeakerAudioTrack only supports mono or stereo")
        # Layout is fully determined by ``channels``; ignore mismatched
        # ``layout`` rather than emit malformed frames.
        derived_layout = "stereo" if channels == 2 else "mono"
        if layout is not None and layout != derived_layout:
            logger.debug(
                "SpeakerAudioTrack: ignoring layout=%r in favour of channel-derived %r",
                layout,
                derived_layout,
            )
        self.sample_rate = sample_rate
        self.layout = derived_layout
        self.channels = channels
        self.bit_depth = bit_depth
        self._samples_per_frame = int(AUDIO_PTIME * sample_rate)
        self._bytes_per_frame = self._samples_per_frame * 2 * channels
        self._start: float | None = None
        self._pts: int = 0

    def get_stream_attributes(self) -> dict[str, Any]:
        return {
            "audio_type": "speaker",
            "direction": "output",
            "sample_rate": self.sample_rate,
            "layout": self.layout,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "ptime_ms": int(AUDIO_PTIME * 1000),
        }

    def get_stream_config(self) -> dict[str, Any] | None:
        return {
            "kind": "audio",
            "direction": "output",
            "sample_rate_hz": self.sample_rate,
            "channels": self.channels,
            "codec": "opus",
            "bit_depth": self.bit_depth,
        }

    async def recv(self) -> AudioFrame:
        if self._closed:
            raise MediaStreamError("Track is closed")
        now = time.monotonic()
        if self._start is None:
            self._start = now
        next_pts_time = self._start + (self._pts / self.sample_rate)
        wait = next_pts_time - now
        if wait > 0:
            await asyncio.sleep(wait)
        raw = bytes(self._bytes_per_frame)
        frame = AudioFrame(format="s16", layout=self.layout, samples=self._samples_per_frame)
        frame.pts = self._pts
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        frame.planes[0].update(raw)
        self._pts += self._samples_per_frame
        self.frame_count += 1
        return frame


class SpeakerAudioStreamer(BaseAudioStreamer):
    """Consumer-side audio streamer.

    Drives the same WebRTC + MQTT signalling lifecycle as
    :class:`MicrophoneAudioStreamer`, and routes received audio into a
    :class:`HostSpeakerCapture`. File / WebRTC / Zenoh sources can coexist —
    they are sum-mixed through a shared :class:`_MixingAudioSource`.
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        *,
        twin_uuid: str | None = None,
        turn_servers: list | None = None,
        auto_reconnect: bool = DEFAULT_AUTO_RECONNECT,
        speaker_name: Optional[str] = DEFAULT_SPEAKER_NAME,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        layout: str = DEFAULT_LAYOUT,
        channels: int = 1,
        bit_depth: int = DEFAULT_SPEAKER_BIT_DEPTH,
        device_index: int | None = None,
        recording: bool = DEFAULT_AUDIO_RECORDING,
        frontend_type: str = DEFAULT_SPEAKER_FRONTEND_TYPE,
        stream_source: Optional[str] = None,
        stream_instance_id: Optional[str] = None,
        enable_health_check: bool = True,
        playback: HostSpeakerCapture | None = None,
    ) -> None:
        super().__init__(
            client,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            auto_reconnect=auto_reconnect,
            mic_name=speaker_name,
            recording=recording,
            frontend_type=frontend_type,
            stream_source=stream_source,
            stream_instance_id=stream_instance_id,
            enable_health_check=enable_health_check,
        )
        if playback is not None:
            # When the caller supplies a playback, its settings win and we
            # warn about any conflicting kwargs.
            overridden: list[str] = []
            if sample_rate != playback.sample_rate:
                overridden.append(f"sample_rate ({sample_rate} vs playback={playback.sample_rate})")
            if channels != playback.channels:
                overridden.append(f"channels ({channels} vs playback={playback.channels})")
            if bit_depth != playback.bit_depth:
                overridden.append(f"bit_depth ({bit_depth} vs playback={playback.bit_depth})")
            if device_index is not None and device_index != playback.device_index:
                overridden.append(
                    f"device_index ({device_index} vs playback={playback.device_index})"
                )
            if overridden:
                logger.warning(
                    "SpeakerAudioStreamer: ignoring constructor args because playback "
                    "is supplied: %s",
                    ", ".join(overridden),
                )
            self._sample_rate = playback.sample_rate
            self._channels = playback.channels
            self._bit_depth = playback.bit_depth
            self.playback = playback
            self._owns_playback = False
        else:
            self._owns_playback = True
            self._sample_rate = sample_rate
            self._channels = channels
            self._bit_depth = bit_depth
            self.playback = HostSpeakerCapture(
                sample_rate=sample_rate,
                channels=channels,
                frames_per_chunk=int(AUDIO_PTIME * sample_rate),
                device_index=device_index,
                bit_depth=bit_depth,
            )
        self._layout = layout
        self._zenoh_subscriptions: list[Any] = []
        self._zenoh_push_callbacks: list[Callable[[np.ndarray], None]] = []
        self._zenoh_mix_keys: list[Any] = []
        self._zenoh_routes: dict[str, dict[str, Any]] = {}
        self._webrtc_consumer_tasks: list[asyncio.Task[None]] = []
        # Mixer is created lazily so single-source callers don't pay for it.
        self._mix_source: _MixingAudioSource | None = None
        self._webrtc_mix_key: object = object()
        self._webrtc_adapter: InboundAudioAdapter | None = None

    @property
    def webrtc_active(self) -> bool:
        """True when a WebRTC consumer background task or peer connection is up."""
        return self._run_task is not None or self.pc is not None

    def has_zenoh_source(self, source_twin_uuid: str) -> bool:
        return source_twin_uuid in self._zenoh_routes

    def _playback_output_format(self) -> tuple[int, int]:
        return int(self.playback.sample_rate), int(self.playback.channels)

    def _get_webrtc_adapter(self) -> InboundAudioAdapter:
        out_rate, out_channels = self._playback_output_format()
        if (
            self._webrtc_adapter is None
            or self._webrtc_adapter._output_sample_rate != out_rate
            or self._webrtc_adapter._output_channels != out_channels
        ):
            self._webrtc_adapter = InboundAudioAdapter(
                output_sample_rate=out_rate,
                output_channels=out_channels,
            )
        return self._webrtc_adapter

    def _send_offer(self, sdp: str) -> None:
        """Build the MQTT offer payload with ``sensor_type="speaker"`` /
        ``role="consumer"`` so the SFU routes this edge offer to its consumer
        handler instead of the default producer path."""
        prefix = self.client.topic_prefix
        offer_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-offer"
        stream_attributes = self.streamer.get_stream_attributes() if self.streamer else {}
        offer_payload = {
            "target": "backend",
            "sender": "edge",
            "type": self.pc.localDescription.type,
            "sdp": sdp,
            "timestamp": time.time(),
            "recording": self.recording,
            "stream_attributes": stream_attributes,
            "sensor": self.mic_name,
            "track_id": self.streamer.id if self.streamer else None,
            "frontend_type": self.frontend_type,
            "session_id": f"{self.client.client_id}_{self.mic_name}",
            # ``sensor_type`` matches catalog speaker metadata (``type: "speaker"``).
            "sensor_type": "speaker",
            "role": "consumer",
        }
        if self.stream_source:
            offer_payload["stream_source"] = self.stream_source
        if self.stream_instance_id:
            offer_payload["stream_instance_id"] = self.stream_instance_id
        self._publish_message(offer_topic, offer_payload)

    def initialize_track(self) -> BaseAudioTrack:
        return SpeakerAudioTrack(
            sample_rate=self._sample_rate,
            layout=self._layout,
            channels=self._channels,
            bit_depth=self._bit_depth,
        )

    async def _setup_webrtc(self) -> None:  # type: ignore[override]
        await super()._setup_webrtc()
        pc = self.pc
        if pc is None:
            return
        speaker = self.playback

        @pc.on("track")
        def _on_remote_track(track: Any) -> None:
            if track.kind != "audio":
                return
            logger.info("Speaker received remote audio track id=%s", getattr(track, "id", None))
            push = self._ensure_mixer().add_input(self._webrtc_mix_key)
            if not speaker.is_running:
                try:
                    speaker.start()
                except Exception:
                    logger.exception("Failed to start host speaker for remote track")
                    return

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = self._event_loop
            if loop is None:
                logger.error(
                    "No running event loop available for remote track consumer; "
                    "dropping incoming WebRTC audio"
                )
                if self._mix_source is not None:
                    self._mix_source.remove_input(self._webrtc_mix_key)
                return

            task = loop.create_task(self._drain_remote_track(track, push))
            self._webrtc_consumer_tasks.append(task)

    async def _drain_remote_track(
        self, track: Any, push: Callable[[np.ndarray], None]
    ) -> None:
        try:
            while True:
                frame = await track.recv()
                if frame is None:
                    return
                try:
                    in_channels = infer_webrtc_frame_channels(frame)
                    arr = np.asarray(frame.to_ndarray(), dtype=np.int16)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, in_channels)
                    elif arr.shape[0] == in_channels and arr.shape[0] != arr.shape[1]:
                        arr = np.ascontiguousarray(arr.T, dtype=np.int16)
                    in_rate = int(getattr(frame, "sample_rate", None) or self._sample_rate)
                    adapted = self._get_webrtc_adapter().convert(
                        arr,
                        input_sample_rate=in_rate,
                        input_channels=in_channels,
                    )
                except Exception:
                    logger.debug("Could not adapt WebRTC audio frame", exc_info=True)
                    continue
                if adapted.size:
                    push(adapted)
        except MediaStreamError:
            logger.info("Remote speaker track ended")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Speaker remote track consumer failed")
        finally:
            if self._mix_source is not None:
                self._mix_source.remove_input(self._webrtc_mix_key)

    def _ensure_mixer(self) -> _MixingAudioSource:
        """Install the shared mixer as the playback source if not already there."""
        if self._mix_source is None:
            out_rate, out_channels = self._playback_output_format()
            self._mix_source = _MixingAudioSource(
                target_channels=out_channels,
                frames_per_chunk=int(AUDIO_PTIME * out_rate),
            )
        if self.playback._get_active_source() is not self._mix_source:
            self.playback.set_source(self._mix_source)
        return self._mix_source

    async def stop_webrtc_consumer(self) -> None:
        """Tear down the WebRTC consumer without closing Zenoh routes or playback."""
        for task in self._webrtc_consumer_tasks:
            task.cancel()
        for task in self._webrtc_consumer_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("WebRTC consumer task raised during stop_webrtc_consumer()")
        self._webrtc_consumer_tasks.clear()
        if self._mix_source is not None:
            self._mix_source.remove_input(self._webrtc_mix_key)
        await super().stop()

    def stop_zenoh_source(self, source_twin_uuid: str) -> bool:
        """Close the Zenoh subscription for *source_twin_uuid*. Returns whether it was active."""
        route = self._zenoh_routes.pop(source_twin_uuid, None)
        if route is None:
            return False

        subscription = route["subscription"]
        mix_key = route["mix_key"]
        with contextlib.suppress(Exception):
            subscription.close()
        with contextlib.suppress(ValueError):
            self._zenoh_subscriptions.remove(subscription)
        with contextlib.suppress(ValueError):
            self._zenoh_mix_keys.remove(mix_key)
        push = route.get("push")
        if push is not None:
            with contextlib.suppress(ValueError):
                self._zenoh_push_callbacks.remove(push)
        if self._mix_source is not None:
            self._mix_source.remove_input(mix_key)
        return True

    def _detach_mixer_from_playback(self) -> None:
        """Drop the mixer source without closing a caller-owned playback device."""
        if self._mix_source is None:
            return
        if self.playback._get_active_source() is self._mix_source:
            self.playback.clear_source()
        self._mix_source = None

    async def stop(self) -> None:  # type: ignore[override]
        await self.stop_webrtc_consumer()
        for twin_uuid in list(self._zenoh_routes):
            self.stop_zenoh_source(twin_uuid)
        if self._owns_playback:
            self._mix_source = None
            try:
                self.playback.stop()
            except Exception:
                logger.exception("Error stopping host speaker")
        else:
            self._detach_mixer_from_playback()

    async def start_zenoh_only(self) -> None:
        """Start the host speaker without bringing up a WebRTC peer connection."""
        if not self.playback.is_running:
            self.playback.start()

    def set_file_source(self, path: str, *, loop: bool = False) -> None:
        if not self.playback.is_running:
            self.playback.start()
        self.playback.set_file_source(path, loop=loop)

    def set_webrtc_source(self) -> None:
        """Prepare the host speaker for incoming WebRTC audio. Idempotent and
        source-preserving — does not tear down active Zenoh subscriptions."""
        if not self.playback.is_running:
            self.playback.start()
        self._ensure_mixer()

    def set_zenoh_source(
        self,
        *,
        data_bus: "DataBus",
        channel: str,
        source_twin_uuid: str | None = None,
        input_sample_rate: int | None = None,
        input_channels: int | None = None,
    ) -> Any:
        """Subscribe to *channel* on *source_twin_uuid* (or the bus's twin) and
        pipe PCM into the host speaker.

        Multiple calls fan into a shared mixer — audio is sum-mixed, not
        replaced. Returns the underlying ``Subscription``; the streamer also
        tracks it for cleanup on :meth:`stop`.
        """
        target_twin = source_twin_uuid or data_bus.twin_uuid
        existing = self._zenoh_routes.get(target_twin)
        if existing is not None:
            return existing["subscription"]

        out_rate, out_channels = self._playback_output_format()
        zenoh_in_rate = int(input_sample_rate or self._sample_rate)
        zenoh_in_channels = int(input_channels or self._channels)
        adapter = InboundAudioAdapter(
            output_sample_rate=out_rate,
            output_channels=out_channels,
        )

        if not self.playback.is_running:
            self.playback.start()
        mixer = self._ensure_mixer()
        mix_key = object()
        push = mixer.add_input(mix_key)
        self._zenoh_mix_keys.append(mix_key)
        self._zenoh_push_callbacks.append(push)

        def _on_pcm(payload: Any) -> None:
            arr: np.ndarray
            if isinstance(payload, np.ndarray):
                arr = payload.astype(np.int16, copy=False)
            elif isinstance(payload, (bytes, bytearray)):
                arr = np.frombuffer(payload, dtype=np.int16)
            else:
                return
            if arr.ndim == 1:
                arr = arr.reshape(-1, zenoh_in_channels)
            try:
                adapted = adapter.convert(
                    arr,
                    input_sample_rate=zenoh_in_rate,
                    input_channels=zenoh_in_channels,
                )
            except Exception:
                logger.debug("Could not adapt Zenoh PCM chunk", exc_info=True)
                return
            if adapted.size:
                push(adapted)

        subscription = data_bus.subscribe(
            channel,
            _on_pcm,
            policy="latest",
            twin_uuid=target_twin if target_twin != data_bus.twin_uuid else None,
        )
        self._zenoh_subscriptions.append(subscription)
        self._zenoh_routes[target_twin] = {
            "subscription": subscription,
            "mix_key": mix_key,
            "push": push,
            "channel": channel,
            "input_sample_rate": zenoh_in_rate,
            "input_channels": zenoh_in_channels,
        }
        return subscription

    def subscribe_zenoh_sources(
        self,
        *,
        data_bus: "DataBus",
        sources: Iterable[tuple[str, str]],
    ) -> list[Any]:
        """Subscribe to multiple ``(twin_uuid, channel)`` Zenoh sources; all are
        sum-mixed into the same playback path. Returns the subscriptions."""
        return [
            self.set_zenoh_source(
                data_bus=data_bus,
                channel=channel,
                source_twin_uuid=twin_uuid,
            )
            for twin_uuid, channel in sources
        ]

    def set_volume(self, volume: float) -> None:
        self.playback.set_volume(volume)

    def set_channel_gain(self, channel_idx: int, gain: float) -> None:
        self.playback.set_channel_gain(channel_idx, gain)

    def set_gain(self, channel_idx: int, gain: float) -> None:
        """Deprecated alias for :meth:`set_channel_gain`."""
        self.set_channel_gain(channel_idx, gain)

    def get_physical_info(self) -> dict[str, Any]:
        return self.playback.get_physical_info()


def play_file(
    path: str,
    *,
    device_index: int | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    bit_depth: int = DEFAULT_SPEAKER_BIT_DEPTH,
    volume: float | None = None,
    gain: float | None = None,
    loop: bool = False,
    blocking: bool = True,
) -> HostSpeakerCapture:
    """One-shot ``play_file("/tmp/cue.mp3")``. Returns the :class:`HostSpeakerCapture`
    so the caller can switch sources, adjust volume, or stop the device."""
    speaker = HostSpeakerCapture(
        sample_rate=sample_rate,
        channels=channels,
        device_index=device_index,
        bit_depth=bit_depth,
    )
    if volume is not None:
        speaker.set_volume(volume)
    if gain is not None:
        speaker.set_channel_gain(0, gain)
    speaker.start()
    speaker.play_file(path, loop=loop, blocking=blocking)
    return speaker


def associate_speaker_to_microphone(
    streamer: SpeakerAudioStreamer,
    *,
    data_bus: "DataBus",
    microphone_twin_uuid: str,
    channel: str = "audio/default",
    transport: str = "zenoh",
) -> Any:
    """Wire a speaker streamer to play a single microphone twin.

    ``transport="zenoh"`` subscribes to the mic's ``audio/<sensor>`` data-bus
    channel; ``"webrtc"`` is a placeholder for the SFU relay path.
    """
    if transport == "zenoh":
        return streamer.set_zenoh_source(
            data_bus=data_bus,
            channel=channel,
            source_twin_uuid=microphone_twin_uuid,
        )
    if transport == "webrtc":
        streamer.set_webrtc_source()
        return None
    raise ValueError(f"Unsupported transport {transport!r}")


def associate_speaker_to_microphones(
    streamer: SpeakerAudioStreamer,
    *,
    data_bus: "DataBus",
    microphone_twin_uuids: Iterable[str],
    channel: str = "audio/default",
    transport: str = "zenoh",
) -> list[Any]:
    """Wire a speaker streamer to multiple microphone twins (sum-mixed)."""
    if transport == "zenoh":
        return streamer.subscribe_zenoh_sources(
            data_bus=data_bus,
            sources=[(twin, channel) for twin in microphone_twin_uuids],
        )
    if transport == "webrtc":
        streamer.set_webrtc_source()
        return []
    raise ValueError(f"Unsupported transport {transport!r}")


__all__ = [
    "DEFAULT_SPEAKER_NAME",
    "DEFAULT_SPEAKER_FRONTEND_TYPE",
    "DEFAULT_SPEAKER_BIT_DEPTH",
    "SPEAKER_SENSOR_TYPES",
    "DSPState",
    "HostSpeakerCapture",
    "SpeakerAudioStreamer",
    "SpeakerAudioTrack",
    "associate_speaker_to_microphone",
    "associate_speaker_to_microphones",
    "check_host_speaker_settings",
    "create_linux_speaker_monitor",
    "list_host_sound_devices",
    "play_file",
    "query_supported_output_sample_rates",
]

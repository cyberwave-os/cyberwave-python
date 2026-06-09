"""
Listen to microphone audio on the local Zenoh data bus.

The generic-microphone driver publishes raw PCM int16 chunks on a Zenoh channel
like ``audio/audio`` (twin sensor id) or ``audio/default``. This script
subscribes and prints per-chunk stats; use ``--play`` to hear audio on the
host speakers.

On edge devices the driver usually uses **peer-to-peer Zenoh** (no
``ZENOH_CONNECT``). Do not point this script at ``tcp/127.0.0.1:7447`` unless
the driver also has ``ZENOH_CONNECT`` set to that router.

Prerequisites (Raspberry Pi / Debian use a venv — system pip is blocked):

    cd cyberwave-sdks/cyberwave-python
    python3 -m venv .venv
    .venv/bin/pip install -e '.[zenoh]'

With the microphone driver already running and audio streaming on the edge:

    # Twin UUID from the running driver, e.g.:
    # docker inspect cyberwave-driver-XXXX --format '{{range .Config.Env}}{{println .}}{{end}}' | grep CYBERWAVE_TWIN_UUID
    export CYBERWAVE_TWIN_UUID="<your-twin-uuid>"
    export CYBERWAVE_DATA_BACKEND=zenoh
    .venv/bin/python examples/listen_zenoh_audio.py --sensor audio

Or use the helper script (creates .venv, auto-detects twin UUID):

    ./examples/listen_zenoh_audio.sh

Record 4 seconds to a WAV file then exit:

    unset ZENOH_CONNECT
    .venv/bin/python examples/listen_zenoh_audio.py --sensor audio \\
        --record /tmp/zenoh_mic.wav --record-seconds 4

Optional speaker playback:

    .venv/bin/pip install sounddevice
    .venv/bin/python examples/listen_zenoh_audio.py --play

Alternative — run inside the driver container (SDK + zenoh already installed):

    docker cp examples/listen_zenoh_audio.py cyberwave-driver-XXXX:/tmp/
    docker exec -e CYBERWAVE_TWIN_UUID=<uuid> cyberwave-driver-XXXX \\
        python3 /tmp/listen_zenoh_audio.py --sensor audio
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from cyberwave.data.api import DataBus
from cyberwave.data.backend import Sample
from cyberwave.data.config import BackendConfig, get_backend
from cyberwave.workers.decode import decode_sample_payload, extract_wire_metadata


def _parse_connect(value: str | None, *, use_router: bool) -> list[str]:
    if use_router:
        return ["tcp/127.0.0.1:7447"]
    if value:
        return [e.strip() for e in value.split(",") if e.strip()]
    env = os.environ.get("ZENOH_CONNECT", "").strip()
    if env:
        return [e.strip() for e in env.split(",") if e.strip()]
    return []


def _resolve_channel(channel: str | None, sensor: str | None) -> str:
    if channel:
        return channel
    if sensor:
        return f"audio/{sensor}"
    return "audio/default"


def _rms_int16(audio: np.ndarray) -> float:
    flat = np.asarray(audio, dtype=np.int16).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(flat.astype(np.float64) ** 2)))


class _AudioListener:
    def __init__(
        self,
        *,
        play: bool,
        print_every: int,
        record_path: Path | None = None,
        record_seconds: float | None = None,
    ) -> None:
        self._play = play
        self._print_every = max(1, print_every)
        self._record_path = record_path
        self._record_seconds = record_seconds
        self._chunk_count = 0
        self._sample_rate_hz: int | None = None
        self._channels: int = 1
        self._stream: Any = None
        self._record_chunks: list[np.ndarray] = []
        self._recorded_frames = 0
        self.done = threading.Event()

    def _ensure_playback(self, sample_rate_hz: int, channels: int) -> None:
        if not self._play or self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise SystemExit(
                "Playback requires sounddevice: pip install sounddevice"
            ) from exc

        self._stream = sd.OutputStream(
            samplerate=sample_rate_hz,
            channels=channels,
            dtype="int16",
        )
        self._stream.start()

    def __call__(self, sample: Sample) -> None:
        audio, _ts = decode_sample_payload(sample, content_hint="numpy")
        if not isinstance(audio, np.ndarray):
            return

        if self._sample_rate_hz is None:
            meta = extract_wire_metadata(sample)
            rate = meta.get("sample_rate_hz")
            if isinstance(rate, (int, float)) and rate > 0:
                self._sample_rate_hz = int(rate)
            ch = meta.get("channels")
            if isinstance(ch, (int, float)) and ch >= 1:
                self._channels = int(ch)

        flat = np.asarray(audio, dtype=np.int16)
        if flat.ndim == 2 and flat.shape[1] > 1:
            samples = flat.shape[0]
        else:
            flat = flat.reshape(-1)
            samples = flat.size

        rate = self._sample_rate_hz or 16000
        duration_s = samples / float(rate)
        self._chunk_count += 1

        if self._chunk_count == 1 or self._chunk_count % self._print_every == 0:
            rms = _rms_int16(flat)
            print(
                f"chunk={self._chunk_count}  "
                f"samples={samples}  "
                f"duration={duration_s * 1000:.1f} ms  "
                f"rms={rms:.0f}  "
                f"rate={rate} Hz  "
                f"ch={self._channels}  "
                f"ts={_ts:.3f}"
            )

        if self._play:
            self._ensure_playback(rate, self._channels)
            out = flat if flat.ndim == 2 else flat.reshape(-1, 1)
            self._stream.write(out)

        if self._record_path is not None and self._record_seconds is not None:
            chunk = np.asarray(audio, dtype=np.int16)
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)
            self._record_chunks.append(chunk.copy())
            self._recorded_frames += chunk.shape[0]
            target_frames = int(rate * self._record_seconds)
            if self._recorded_frames >= target_frames:
                self.done.set()

    def write_wav(self) -> int:
        """Write accumulated PCM to ``self._record_path``. Returns bytes written."""
        if self._record_path is None or not self._record_chunks:
            return 0

        rate = self._sample_rate_hz or 16000
        channels = self._channels
        audio = np.concatenate(self._record_chunks, axis=0)
        target_frames = (
            int(rate * self._record_seconds)
            if self._record_seconds is not None
            else audio.shape[0]
        )
        audio = audio[:target_frames]
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        channels = max(1, int(audio.shape[1]))

        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self._record_path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(audio.astype(np.int16, copy=False).tobytes())

        return self._record_path.stat().st_size

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subscribe to microphone audio on the Zenoh data bus",
    )
    parser.add_argument(
        "--twin-uuid",
        default=os.environ.get("CYBERWAVE_TWIN_UUID", ""),
        help="Twin UUID (default: CYBERWAVE_TWIN_UUID env)",
    )
    parser.add_argument(
        "--connect",
        default=None,
        help="Zenoh router endpoint(s), comma-separated (overrides P2P default)",
    )
    parser.add_argument(
        "--router",
        action="store_true",
        help="Connect to Zenoh router at tcp/127.0.0.1:7447 instead of P2P discovery",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Full data channel (e.g. audio/audio). Overrides --sensor.",
    )
    parser.add_argument(
        "--sensor",
        default="audio",
        help="Audio sensor id when --channel is omitted (default: audio)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play received PCM through the host speakers (requires sounddevice)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        metavar="N",
        help="Log every Nth chunk (default: 25)",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        default=None,
        help="Write received audio to a WAV file and exit when done",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=4.0,
        metavar="SEC",
        help="Duration to record when --record is set (default: 4)",
    )
    args = parser.parse_args()

    twin_uuid = args.twin_uuid.strip()
    if not twin_uuid:
        print("error: pass --twin-uuid or set CYBERWAVE_TWIN_UUID", file=sys.stderr)
        sys.exit(1)

    channel = _resolve_channel(args.channel, args.sensor)
    connect = _parse_connect(args.connect, use_router=args.router)

    os.environ.setdefault("CYBERWAVE_DATA_BACKEND", "zenoh")

    record_path = Path(args.record).expanduser() if args.record else None
    record_seconds = args.record_seconds if record_path else None

    listener = _AudioListener(
        play=args.play,
        print_every=args.print_every,
        record_path=record_path,
        record_seconds=record_seconds,
    )
    backend = get_backend(BackendConfig(zenoh_connect=connect))
    bus = DataBus(backend, twin_uuid)

    mode = f"router ({', '.join(connect)})" if connect else "peer-to-peer (multicast)"
    print(f"Listening on {channel!r} for twin {twin_uuid}")
    print(f"Zenoh mode: {mode}")
    if record_path is not None:
        print(f"Recording {record_seconds}s to {record_path} then exiting.\n")
    else:
        print("Press Ctrl+C to stop.\n")

    sub = bus.subscribe(channel, listener, policy="fifo", raw=True)
    start = time.monotonic()

    try:
        idle_reported = False
        while not listener.done.is_set():
            time.sleep(0.05)
            if record_path is None and not idle_reported and listener._chunk_count == 0:
                if time.monotonic() - start > 5.0:
                    idle_reported = True
                    print(
                        "No audio chunks yet. Check:\n"
                        f"  - channel matches driver logs (often audio/audio, not audio/default)\n"
                        f"  - microphone driver is active: docker logs cyberwave-driver-* | grep ZENOH\n"
                        f"  - try P2P (default) not --router unless driver uses ZENOH_CONNECT\n"
                        f"  - restart driver if it was running for a long time: docker restart cyberwave-driver-*",
                        file=sys.stderr,
                    )
    except KeyboardInterrupt:
        print(f"\nStopping. Received {listener._chunk_count} chunk(s).")
    finally:
        sub.close()
        if record_path is not None and listener._record_chunks:
            nbytes = listener.write_wav()
            print(
                f"Wrote {record_path} "
                f"({listener._recorded_frames} frames, "
                f"{listener._sample_rate_hz or 16000} Hz, "
                f"{listener._channels} ch, {nbytes} bytes)"
            )
        elif record_path is not None:
            print("No audio captured; WAV file not written.", file=sys.stderr)
        listener.close()
        bus.close()
        backend.close()


if __name__ == "__main__":
    main()

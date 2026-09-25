"""Smoke test 3/4 — Mistral Voxtral STT (direct HTTP).

Records ~5 seconds of audio from the default input device, encodes it as a
16 kHz mono WAV, posts it to the Mistral audio-transcriptions endpoint via
plain `httpx`, and prints the transcript.

Diagnostics built in:
  - lists every input device so you can spot the wrong default
  - 1-second countdown before recording
  - live "🟩🟩⬛⬛⬛" RMS meter while recording
  - peak / mean amplitude after recording (warns if silent)
  - saves the WAV to /tmp so you can play it back with `afplay`

Why direct HTTP instead of the `mistralai` Python SDK: the SDK switched from
v1.x to v2.x with a different import surface, and the v2.x package on PyPI
currently doesn't re-export `Mistral` cleanly. The REST endpoint is stable.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

DURATION = 5.0
SAMPLE_RATE = 16000
ENDPOINT = "https://api.mistral.ai/v1/audio/transcriptions"

SILENT_THRESHOLD = 0.005


def _print_devices(sd) -> None:
    print("  available input devices:")
    try:
        default_in = sd.default.device[0] if sd.default.device else None
    except Exception:
        default_in = None
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            marker = " ← default" if idx == default_in else ""
            print(f"    [{idx:>2}] {dev['name']}{marker}")


def _record_with_meter(sd, duration: float) -> np.ndarray:
    frames_per_chunk = SAMPLE_RATE // 10
    total_chunks = int(duration * 10)
    chunks: list[np.ndarray] = []

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=frames_per_chunk,
    )
    stream.start()
    try:
        for _ in range(total_chunks):
            data, _overflow = stream.read(frames_per_chunk)
            chunks.append(data.copy())
            rms = float(np.sqrt(np.mean(data**2)))
            bars = min(20, int(rms * 400))
            meter = "🟩" * bars + "⬛" * (20 - bars)
            print(f"\r    {meter}  rms={rms:.4f}", end="", flush=True)
    finally:
        stream.stop()
        stream.close()
    print()
    return np.concatenate(chunks, axis=0)


def main() -> None:
    if not os.environ.get("MISTRAL_API_KEY"):
        print("❌ MISTRAL_API_KEY not set")
        sys.exit(1)

    try:
        import httpx
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        print(f"❌ import failed: {exc}")
        sys.exit(1)

    model = os.environ.get("MISTRAL_STT_MODEL", "voxtral-mini-latest")

    _print_devices(sd)
    print()

    for n in (3, 2, 1):
        print(f"  starting in {n}...", end="\r", flush=True)
        time.sleep(1)
    print(f"🎙  Recording {DURATION:.0f}s — speak NOW (e.g. 'move right for two seconds')")

    audio = _record_with_meter(sd, DURATION)

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    print(f"    peak={peak:.4f}  rms={rms:.4f}")

    wav_path = Path(tempfile.gettempdir()) / "voxtral_smoke.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"    saved → {wav_path}   (play back: afplay {wav_path})")

    if rms < SILENT_THRESHOLD:
        print("❌ Recording is effectively silent.")
        print("   Likely causes:")
        print("   • Terminal lacks Microphone permission")
        print("       System Settings → Privacy & Security → Microphone → enable for your terminal")
        print("       (then fully QUIT and re-open the terminal app)")
        print("   • Wrong default input device (see list above)")
        print("       fix:  python -c \"import sounddevice as sd; sd.default.device = (<idx>, None)\"")
        print("   • Mic muted at the hardware/menu-bar level")
        sys.exit(1)

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    print(f"    encoded {len(wav_bytes):,} bytes → POST {ENDPOINT} (model={model})...")

    resp = httpx.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"},
        data={"model": model},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        timeout=30.0,
    )

    if resp.status_code != 200:
        print(f"❌ HTTP {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    body = resp.json()
    transcript = (body.get("text") or "").strip()
    if not transcript:
        print(f"❌ Voxtral returned no text. Full body: {body}")
        print(f"   Audio captured ok (rms={rms:.4f}) but Voxtral heard nothing intelligible.")
        print(f"   Play back with:  afplay {wav_path}")
        sys.exit(1)

    print(f"  transcript: {transcript!r}")
    print(f"✅ Mistral STT OK — model: {model}")


if __name__ == "__main__":
    main()

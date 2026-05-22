"""Voice input — spacebar push-to-talk + Mistral Voxtral transcription.

Public API:
  record_via_spacebar() → (wav_bytes, stats) | None
  transcribe(wav_bytes) → (transcript, error)
  capture_utterance()   → (utterance, error)   # convenience: record + transcribe

Why direct HTTP to Voxtral (not the `mistralai` SDK): the v2.x SDK on PyPI
currently doesn't re-export `Mistral` cleanly. The REST endpoint is stable.

UX rules baked in:
  * minimum recording duration (0.3 s) — guards against accidental space taps
  * silent-recording rejection (rms < 0.005) — saves a Voxtral round-trip
  * live RMS meter while recording — instant "is the mic hot?" signal
  * Esc cancels the *current* recording; the agent loop owns Ctrl+C for exit

macOS prerequisite: the terminal app must already have BOTH Microphone and
Accessibility permissions (the latter for `pynput` to see the spacebar).
"""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import numpy as np


VOXTRAL_ENDPOINT = "https://api.mistral.ai/v1/audio/transcriptions"
SAMPLE_RATE = 16000

MIN_DURATION_S = 0.3
MAX_DURATION_S = 12.0
SILENT_RMS_THRESHOLD = 0.005


def _draw_meter(rms: float, elapsed: float) -> str:
    bars = min(20, int(rms * 400))
    meter = "🟩" * bars + "⬛" * (20 - bars)
    return f"\r  🔴 {meter}  {elapsed:4.1f}s  rms={rms:.3f}"


def record_via_spacebar(
    *,
    max_duration_s: float = MAX_DURATION_S,
    min_duration_s: float = MIN_DURATION_S,
) -> tuple[bytes, dict] | None:
    """Block until SPACE is pressed, record while held, return WAV bytes.

    Returns:
        (wav_bytes, stats) on success, where stats has keys
          {"duration": s, "peak": float, "rms": float, "wav_path": str}.
        None if cancelled (Esc), too short, or silent.
    """
    import sounddevice as sd
    import soundfile as sf
    from pynput import keyboard

    print("\n  🎤 hold SPACE to talk  (Esc to cancel this turn)")

    started = threading.Event()
    recording = threading.Event()
    cancelled = threading.Event()

    def on_press(key):
        if key == keyboard.Key.space and not started.is_set():
            started.set()
            recording.set()
        elif key == keyboard.Key.esc:
            cancelled.set()
            return False
        return None

    def on_release(key):
        if key == keyboard.Key.space and started.is_set():
            recording.clear()
            return False
        return None

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        while not started.is_set() and not cancelled.is_set():
            time.sleep(0.03)

        if cancelled.is_set():
            print("  (cancelled)")
            return None

        chunks: list[np.ndarray] = []
        frames_per_chunk = SAMPLE_RATE // 10  # 100 ms

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=frames_per_chunk,
        )
        stream.start()
        t0 = time.monotonic()
        try:
            while recording.is_set() and (time.monotonic() - t0) < max_duration_s:
                data, _overflow = stream.read(frames_per_chunk)
                chunks.append(data.copy())
                rms = float(np.sqrt(np.mean(data**2)))
                print(_draw_meter(rms, time.monotonic() - t0), end="", flush=True)
        finally:
            stream.stop()
            stream.close()
        print()
    finally:
        listener.stop()

    if not chunks:
        print("  ⚠️  no audio captured")
        return None

    audio = np.concatenate(chunks, axis=0)
    duration = len(audio) / SAMPLE_RATE
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    if duration < min_duration_s:
        print(f"  ⚠️  recording too short ({duration:.2f}s < {min_duration_s}s) — try again")
        return None

    if rms < SILENT_RMS_THRESHOLD:
        print(f"  ⚠️  recording is silent (rms={rms:.4f}) — check the mic and try again")
        return None

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    wav_path = Path("/tmp") / "nl_arm_last.wav"
    try:
        sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
    except Exception:
        wav_path = Path("")

    return wav_bytes, {
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "wav_path": str(wav_path),
    }


def transcribe(
    wav_bytes: bytes,
    *,
    model: str | None = None,
    timeout_s: float = 30.0,
) -> tuple[str | None, str | None]:
    """POST WAV bytes to Voxtral, return (transcript, error)."""
    import httpx

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return None, "MISTRAL_API_KEY not set"

    chosen_model = model or os.environ.get("MISTRAL_STT_MODEL", "voxtral-mini-latest")

    try:
        resp = httpx.post(
            VOXTRAL_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": chosen_model},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            timeout=timeout_s,
        )
    except Exception as exc:
        return None, f"network error: {exc}"

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"

    try:
        body = resp.json()
    except ValueError as exc:
        return None, f"invalid JSON from Voxtral: {exc}"

    text = (body.get("text") or "").strip()
    if not text:
        return None, "Voxtral returned no text (audio may be unintelligible)"
    return text, None


def capture_utterance() -> tuple[str | None, str | None]:
    """Convenience: record + transcribe in one call.

    Returns:
        (utterance, None) on success.
        (None, reason)    on cancel/silent/short/STT failure — caller should re-prompt.
    """
    rec = record_via_spacebar()
    if rec is None:
        return None, "no usable recording"

    wav_bytes, stats = rec
    print(
        f"  📡 transcribing {len(wav_bytes):,} bytes  "
        f"(dur={stats['duration']:.1f}s, peak={stats['peak']:.2f})..."
    )

    t0 = time.monotonic()
    transcript, err = transcribe(wav_bytes)
    dt_ms = (time.monotonic() - t0) * 1000

    if err:
        print(f"  ❌ STT: {err}  ({dt_ms:.0f} ms)")
        return None, err

    print(f"  📝 ({dt_ms:.0f} ms) {transcript!r}")
    return transcript, None

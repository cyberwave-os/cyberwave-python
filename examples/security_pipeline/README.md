# Security Pipeline (Multi-camera, Privacy-preserving)

Two cameras stream into a single edge worker. The worker runs YOLOv8
person detection, **pixelates every person** in each frame, and feeds
the anonymised frame back to the driver. The driver substitutes it
into the WebRTC stream **before** the bytes leave the edge.

```
┌────────┐                      ┌─────────┐                  ┌──────────┐
│ Camera │ ── raw frame ──►     │ Driver  │ ── frames/      │  Worker  │
│  1, 2  │                      │ (cam N) │    default ──►  │          │
└────────┘                      │         │                  │  YOLOv8  │
                                │         │ ◄── frames/     │ + mosaic │
                                │         │   filtered  ──  │          │
                                │         │                  └──────────┘
                                │         │ ── WebRTC ──► Frontend (pixelated persons only)
                                │         │ ── MQTT   ──► Cloud (events only)
                                └─────────┘
```

## Why pixelate and not stick figures?

- A pixelated person still reads as *a person doing something* to a
  human reviewer — useful for retail floors, warehouses, and hospitals
  where presence and motion matter.
- Stick figures are visually fragile: occlusion, low light, or an
  unusual pose drops keypoints and the operator sees a floating hand
  or nothing at all.
- Pixelate doesn't need a pose model, so per-frame cost on CPU is
  roughly ~2× lower than pose + skeleton overlay.

For privacy-sensitive deployments raw video must never leave the edge.
This pipeline keeps the **policy** in a worker (easy to swap) while
the **stream lifecycle** stays in the driver (no WebRTC code in
user-facing workers).

See [Frame Filters](https://docs.cyberwave.com/edge/drivers/frame-filters)
for the full driver contract.

> **Privacy caveat.** Pixelation is deliberate *visual obscuring*, not
> cryptographic de-identification. Low-density mosaics can be partially
> reversed by public depixelation models; for GDPR-grade requirements
> combine pixelate with redact-style masking or drop frames entirely
> and publish only events.

## Layout

```
security_pipeline/
├── worker.py        ← edge worker (this directory is mounted as CYBERWAVE_WORKERS_DIR)
├── .env.example     ← copy to .env and fill in twin UUIDs + API key
└── README.md
```

## Run

1. Provision two camera twins on the same edge device.
2. Enable the frame filter on each twin by setting the per-twin
   metadata flag on its generic-camera driver container:
   ```bash
   CYBERWAVE_METADATA_FRAME_FILTER_ENABLED=true
   ```
   The channel name (`frames/filtered`) and fail-closed blank
   fallback are hard-coded in the driver. The freshness window
   defaults to 200 ms (tuned for ≥ 5 Hz GPU workers) and can be
   widened for CPU-bound workers via
   `CYBERWAVE_METADATA_FRAME_FILTER_FRESHNESS_MS` (e.g. `400`–`500`)
   at the cost of keeping visibly-stale anonymised frames on screen
   longer. `0` is a valid "force blank" fail-close test mode.
3. Copy `.env.example` to `.env` and fill in `CAMERA_1_TWIN`, `CAMERA_2_TWIN`, `CYBERWAVE_API_KEY`.
4. Mount this directory as `/app/workers` in your worker container, or set
   `CYBERWAVE_WORKERS_DIR=/path/to/security_pipeline` and start the runtime:
   ```bash
   cyberwave worker start
   ```

The worker subscribes to both cameras' `frames/default` channels, runs
inference, and publishes anonymised frames back on the SDK constant
`FILTERED_FRAME_CHANNEL` (`frames/filtered`), scoped per-twin. No
additional configuration is required.

## Verify

- Open each camera twin in the frontend — you should see pixelated
  people, **or a black frame** when no person is detected, never raw
  faces or bodies. The black-frame fallback is the worker's
  privacy fail-closed gate (`np.zeros_like(frame)` when
  `result.detections` contains no `person` matches), mirrored
  driver-side for stale and shape-mismatched frames. Without that
  gate a single missed detection would leak the raw frame through a
  fresh, well-formed publish.
- Watch for `person_too_close` events on `cyberwave/twin/{uuid}/event` (MQTT).
- Confirm the privacy boundary holds:
  ```bash
  mosquitto_sub -h <broker> -t 'cyberwave/twin/+/frames/#' -v
  ```
  Should print **nothing** — `frames/*` are local to the edge.

## Tuning

| Knob | Where | Default | Notes |
|---|---|---|---|
| Confidence threshold | `worker.py` (`model.predict(..., confidence=)`) | `0.4` | Lower = more sensitive |
| Anonymisation mode | `worker.py` (`anonymize_frame(..., mode=)`) | `"pixelate"` | `"blur"` / `"redact"` / `"bbox"` also available |
| Mosaic block size | `worker.py` (`anonymize_frame(..., pixel_size=)`) | adaptive (~24 blocks across short side) | Larger = more privacy, less silhouette detail |
| Skeleton overlay | `worker.py` (`anonymize_frame(..., draw_skeleton=)`) | `False` | Requires swapping to `yolov8n-pose-onnx` |
| Freshness window | driver env `CYBERWAVE_METADATA_FRAME_FILTER_FRESHNESS_MS` | `200` ms | Widen to `400`–`500` for CPU-only workers; `0` forces blank (test mode). Higher values leave visibly-stale frames on screen longer. |
| Stale fallback | driver internal constant | `blank` | Fail-closed by design — to see raw frames, disable the filter entirely. |
| Detection overlays | driver env `CYBERWAVE_DETECTION_OVERLAYS` | `true` | When enabled alongside the filter, bounding boxes and labels are drawn **on top** of the anonymised frame and reveal each person's location. Set to `false` if that defeats your anonymisation requirement. |

## Performance notes

- `yolov8n` (plain detector) is used instead of `yolov8n-pose-onnx`
  since pixelation doesn't need keypoints. This is ~2× cheaper per
  frame on CPU.
- A single worker can comfortably handle two 720p streams at 15 fps on
  a modern x86 CPU. For higher fps or more cameras, set
  `CYBERWAVE_MODEL_DEVICE=cuda:0` and use a GPU-enabled worker container.

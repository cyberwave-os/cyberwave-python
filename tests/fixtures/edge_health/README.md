# `edge_health` wire fixtures

JSON fixtures pinning the on-the-wire shape of `edge_health` payloads.
The Python SDK validates against these directly via
`tests/test_edge_health_wire_fixtures.py`.  The C++ SDK (added in
CYB-2004 PR 2) reads from the same files via the monorepo path and is
expected to vendor an equivalent copy under its own `tests/` when
`cpp-sdk-gen.sh` learns to mirror to a standalone repo — the same
in-tree-ownership pattern this directory follows for Python.  The two
publishers cannot silently drift on field names, casing, or types so
long as both sides keep validating against fixtures with identical
content.

If you change the wire schema, update the fixtures here first; the SDK
tests will then refuse to build until the publisher is updated to match.
This is the deliberately-blunt cross-language contract enforcement that
saves us from a class of "frontend renders the wrong resolution for
RealSense cameras because the C++ side spells it `Resolution` instead
of `resolution`" bugs.

The fixtures live inside `cyberwave-python/tests/` rather than at a
shared monorepo root so they ship with the SDK when it's published to
`cyberwave-os/cyberwave-python`.  Reaching outside the SDK tree for any
asset — fixtures, schemas, anything — breaks the standalone publish; see
the run that exposed it in [CYB-2005 review fallout][run-fallout] and
the in-tree principle written up in `cyberwave-sdks/README.md`.

[run-fallout]: https://github.com/cyberwave-os/cyberwave-python/actions/runs/26058310626

## Fixtures

- `camera_minimal.json` — a single-camera publisher (cv2 webcam).  The
  canonical example: one `streams["stream"]` entry with a
  `stream_config` block of `kind: "camera"`, and the deprecated legacy
  `camera_config` slot mirroring the same data.
- `multi_stream_realsense.json` — a RealSense d455-style publisher
  registering both `rgb-0` and `depth-0` from one driver.  Pins the
  multi-stream emission contract (one `streams[id]` entry per
  registered stream config) plus the deterministic
  `camera_config` shim winner (lexicographic on `stream_id`, so
  `depth-0` wins over `rgb-0`).

The fixtures intentionally omit publisher-time-of-day fields
(`timestamp`, `uptime_seconds`, `last_frame_ts`, `frames_sent`, `fps`,
`is_stale`, `is_healthy`, `connection_state`, `ice_connection_state`,
`restart_count`) so they can be diffed against synthesised payloads
without timing dependencies.  Tests assert structural equality on
everything that survives the redaction.

<p align="center">
  <a href="https://cyberwave.com">
    <img src="assets/cyberwave-logo.png" alt="Cyberwave logo" width="320" />
  </a>
</p>

<h1 align="center">Cyberwave Python SDK</h1>

<p align="center">
  <b>Making the physical world programmable.</b><br/>
  Connect, control, and simulate any robot. The same code runs in simulation and on real hardware.
</p>

<p align="center">
  <img src="assets/hero-montage.gif" alt="Cyberwave SDK — deploy any robot as a digital twin, in simulation and on real hardware" width="800" />
</p>

<p align="center">
  <a href="https://github.com/cyberwave-os/cyberwave-python/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-orange.svg" alt="License"></a>
  <a href="https://docs.cyberwave.com"><img src="https://img.shields.io/badge/Documentation-docs.cyberwave.com-orange" alt="Documentation"></a>
  <a href="https://discord.gg/dfGhNrawyF"><img src="https://badgen.net/badge/icon/discord?icon=discord&label&color=orange" alt="Discord"></a>
  <a href="https://pypi.org/project/cyberwave/"><img src="https://img.shields.io/pypi/v/cyberwave.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/cyberwave/"><img src="https://img.shields.io/pypi/pyversions/cyberwave.svg" alt="PyPI Python versions"></a>
  <a href="https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml"><img src="https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml/badge.svg" alt="Build"></a>
</p>

---

Cyberwave is an all-in-one platform for building and deploying intelligent physical AI agents. 
Connect a physical robot or sensor, test it in simulation, and run AI models; all through one Python SDK.
This package is the official client.

## Installation

```bash
pip install cyberwave
```

Optional features install via extras, e.g. `cyberwave[camera]` (video streaming),
`cyberwave[ml]` (vision models), or `cyberwave[zenoh]` (edge data bus). See the
[installation docs](https://docs.cyberwave.com/overview) for the full list.

## Quick Start

Get an API key from your Cyberwave instance (**Profile → API Tokens**) and export it:

```bash
export CYBERWAVE_API_KEY="your_api_key_here"
```

Then create and control your first digital twin:

```python
from cyberwave import Cyberwave

cw = Cyberwave()  # reads CYBERWAVE_API_KEY from the environment

# Create a digital twin from a catalog asset.
# Pin it to a specific twin and environment by passing their IDs (UUID or slug).
# Omit both and Cyberwave creates a "Quickstart Environment" for you automatically.
arm = cw.twin(
    "the-robot-studio/so101",
    twin_id="your-twin-uuid",                 # e.g. "acme/twins/arm-station-1"
    environment_id="your-environment-uuid",   # e.g. "acme/envs/production-floor"
)

# Place it in the scene (editor layout)
arm.edit_position(x=1.0, y=0.0, z=0.5)
arm.edit_rotation(yaw=90)  # degrees

# Move a joint by name
joint_names = arm.joints.list()
if joint_names:
    arm.set_joints({joint_names[0]: -0.2})  # radians
    print(arm.get_joints())

# Drive a locomotion twin in simulation
cw.affect("simulation")          # or cw.affect("live") for the real robot
rover = cw.twin("unitree/go2")
rover.move_forward(0.3)

cw.disconnect()
```

The same script targets real hardware by switching `cw.affect("live")`, no other changes.

## Core Concepts

- **Twins** — virtual representations of robots and sensors. You develop and test against a twin, then deploy to hardware with identical code. Instantiate any catalog asset with `cw.twin("vendor/slug")`.
- **Environments** — scenes your twins live in. Validate quickly in the browser-based Playground, or use MuJoCo for high-fidelity physics and RL.
- **Simulation vs. live** — `cw.affect("simulation")` and `cw.affect("live")` switch where commands and state go. The same code drives both.
- **Edge & cloud** — stream camera/sensor data and run AI models on the edge or in the cloud, without managing the infrastructure in between.

## Demos

Watch the SDK in action with our demos.

<table>
  <tr>
    <td width="50%">
      <a href="https://youtu.be/kUxSxCMCQgc">
        <img src="https://img.youtube.com/vi/kUxSxCMCQgc/hqdefault.jpg" alt="Build a natural language voice agent on SO101" width="100%" />
      </a>
      <br/>
      <b><a href="https://youtu.be/kUxSxCMCQgc">Build a natural language voice agent on SO101</a></b>
    </td>
    <td width="50%">
      <a href="https://youtu.be/ITz9zMf0ObA">
        <img src="https://img.youtube.com/vi/ITz9zMf0ObA/hqdefault.jpg" alt="Controlling a DJI Mini 4 Pro with the Cyberwave Python SDK" width="100%" />
      </a>
      <br/>
      <b><a href="https://youtu.be/ITz9zMf0ObA">Controlling a DJI Mini 4 Pro with the Cyberwave Python SDK</a></b>
    </td>
  </tr>
</table>

## Examples

Runnable scripts live in [examples/](examples) and see the [examples index](examples/README.md) for the full list.

| Example | Shows |
| --- | --- |
| [quickstart.py](examples/quickstart.py) | Create a twin, scene layout, joints, locomotion |
| [joints.py](examples/joints.py) | Read and write joint positions by name |
| [locomotion.py](examples/locomotion.py) | Velocity-style locomotion commands |
| [capture_frame.py](examples/capture_frame.py) | Grab a single camera frame from a twin |
| [camera_stream.py](examples/camera_stream.py) | Stream a camera feed over WebRTC |
| [drone_hovering.py](examples/drone_hovering.py) | Takeoff, hover, and land a flying twin |
| [workflows.py](examples/workflows.py) | List, trigger, and monitor workflows |
| [ai/yolo.ipynb](examples/ai/yolo.ipynb) | Run YOLO vision models (Colab) |

## Documentation

Full guides and the complete API reference are at **[docs.cyberwave.com](https://docs.cyberwave.com)**
([overview](https://docs.cyberwave.com/overview) ·
[API reference](https://docs.cyberwave.com/api-reference/overview)).

## Listing recordings

`recordings.list()` is paged. With no `start`/`end` it lists the most recent day
that has recordings instead of the environment's whole history, and it returns
at most `limit` rows (default `200`, fetched 50 per request). Pass `limit=0` to
follow every page.

Like the replay picker, listing excludes materializing or failed recordings by
default. Pass `include_unready=True` only when a caller needs those rows.

```python
items = cw.environments.recordings.list(environment_id="acme/envs/floor")
items = cw.environments.recordings.list(
    environment_id="acme/envs/floor",
    start="2026-07-01",
    end="2026-07-05",
    limit=0,
)
```

Whenever the result is partial — scoped to one day, or cut short by `limit`
while the server still had pages — `list()` logs a warning naming the window and
telling you which argument widens it, so a truncated list never looks complete.

If Cloud Run rejects a catalog response at its payload-size boundary, the SDK
raises `RecordingPayloadTooLargeError` with the affected window, cloud trace,
and a concrete retry hint. Restrict `start`/`end` or lower `limit` and retry.

## Recording readiness

Recording list items expose the server's playback assessment when it is
available. Use `is_playback_ready` before downloading artifacts, or request
unready entries explicitly while building a retry UI:

```python
items = cw.environments.recordings.list(
    environment_id="acme/envs/floor",
    include_unready=True,
)

for item in items:
    print(item.uuid, item.readiness, item.is_playback_ready)
```

`item.readiness` is `None` when connected to a server that predates this
feature. If `get()` receives a materializing response, it raises `CyberwaveError`
with the server's suggested retry interval instead of returning an empty
recording.

## Contributing

Contributions are welcome. Please open an
[issue](https://github.com/cyberwave-os/cyberwave-python/issues) or a pull request.

## Support

- **Documentation**: [docs.cyberwave.com](https://docs.cyberwave.com)
- **Issues**: [GitHub Issues](https://github.com/cyberwave-os/cyberwave-python/issues)
- **Community**: [Discord](https://discord.gg/dfGhNrawyF)

## License

Released under the [MIT License](LICENSE).

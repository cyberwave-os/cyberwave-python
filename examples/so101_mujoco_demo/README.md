# SO-101 MuJoCo Demo

Simulates a [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm in MuJoCo,
streaming joint state and camera feeds to the Cyberwave frontend.

## Quick start

```bash
cp .env.example .env   # fill in API key + email
just setup             # install deps
just create            # create Cyberwave environment → out/env.json
just export            # download MuJoCo scene → out/mujoco_scene/
just run               # launch viewer (sine-wave control)
```

## Prerequisites

- Python 3.10–3.12
- [`uv`](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [`just`](https://just.systems) — `brew install just` or `cargo install just`

## Environment setup

Copy `.env.example` to `.env` and fill in your credentials:

- `CYBERWAVE_API_KEY` — your API key (required)
- `CYBERWAVE_MQTT_USERNAME` — the email of the account that owns the key (required for MQTT auth)
- `CYBERWAVE_BASE_URL` — uncomment and set to target dev or a local instance

## Cyberwave environment

`just create` provisions a fresh environment on Cyberwave (SO-101 arm + observer camera)
and writes metadata to `out/env.json`. Re-run this only when you want a new environment.

`just export` downloads the universal schema and the MuJoCo scene ZIP from that
environment into `out/mujoco_scene/`. Re-run after `create` or whenever the
asset patches change.

## Control

| Command | Behaviour |
|---|---|
| `just run` | Sine-wave — joints sweep slowly through their range |
| `just run-manual` | No motion — drag joints in the MuJoCo slider panel (Ctrl+M) |

To customise motion, edit `sine_control()` in `so101_mujoco_control.py`.
Write target positions to `data.ctrl[i]` for actuator `i`.

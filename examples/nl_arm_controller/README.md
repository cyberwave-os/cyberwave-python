# NL Arm Controller — AgileX PiPER workshop demo

Voice/text-driven AgileX PiPER controller. Speak or type a natural-language
command, an LLM translates it into a structured joint-motion plan, and the
Cyberwave SDK executes it on the digital twin (and optionally the physical
PiPER).

## Tech stack

- **STT** — Mistral Voxtral (`voxtral-mini-latest`), spacebar push-to-talk.
- **Planner** — Anthropic Claude (text or vision) → constrained JSON action plan.
- **Robot control** — Cyberwave Python SDK → MQTT → PiPER.
- **TTS** — macOS `say`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../..            # the cyberwave SDK from this repo

cp .env.example .env            # then paste your keys + twin UUIDs

python nl_arm_controller.py --check    # env + joint-model self-check
python nl_arm_controller.py            # text REPL
```

## Modes

```bash
python nl_arm_controller.py                    # text REPL, drives the twin
python nl_arm_controller.py --voice            # voice REPL (hold SPACE)
python nl_arm_controller.py --vision           # text + webcam scene awareness
python nl_arm_controller.py --voice --vision   # full demo
python nl_arm_controller.py --keys             # keyboard teleop only
python nl_arm_controller.py --dry-run          # plan only, no robot motion
```

`CW_MODE=simulation` drives only the 3D twin in the browser viewer.
`CW_MODE=live` also drives the physical PiPER — requires the PiPER edge driver
running on the edge device wired to the arm.

## Joint model

The PiPER exposes seven platform joints. `motion.py` is the single source of
truth for them: the LLM prompts in `planner.py` are generated from the same
table, and `teleop.py` builds its key map from the same specs, so the three
can't drift apart.

| Joint | Function | Demo envelope | Keys |
|---|---|---|---|
| `joint1` | base rotation | ±90° | `1` / `2` |
| `joint2` | shoulder pitch | 0° → +90° | `3` / `4` |
| `joint3` | elbow | −90° → 0° | `5` / `6` |
| `joint4` | forearm roll | ±60° | `7` / `8` |
| `joint5` | wrist pitch | ±60° | `9` / `0` |
| `joint6` | wrist roll | ±60° | `Q` / `W` |
| `joint7` | gripper | 0–100% open | `E` / `R` |

The key bindings mirror the **Keyboard (PiPER)** controller in the Cyberwave
environment view one-for-one (odd digit / `Q` / `E` increases, even digit /
`W` / `R` decreases), so muscle memory transfers between the browser panel and
the terminal. `--keys`, or typing `keys` at the text prompt, opens teleop;
`H` homes the arm, `X` returns to the prompt.

Three things about this arm that the SO-101 version of this demo did not have
to model:

- **`joint2` and `joint3` are one-sided.** The shoulder only sweeps forward
  from zero and the elbow only folds back, so "reach forward" is `joint2`
  positive *together with* `joint3` negative. The prompts say so explicitly.
- **`joint7` is the gripper, and it is prismatic.** Plans address it in percent
  open via a dedicated `set_gripper` action; the executor converts to the
  twin's native units and publishes with `degrees=False`. Putting `joint7` in
  a `set_joint` or `set_pose` action is a validation error, because
  `joints.set(..., degrees=True)` would silently scale a metre value by π/180.
- **`joint8` mimics `joint7` at −1.0.** The platform drives it; never command
  it directly.

### Safety envelope

Every commanded angle is clamped before it is sent. The envelope is the
*intersection* of the conservative demo limits in `motion.ARM_JOINTS` and the
twin's own URDF limits, which the executor fetches from `robot.get_schema()`
on startup and prints. So a hallucinated angle can never exceed either bound,
and the printed table is what is actually being enforced.

Plans are also capped at 8 actions and 5 s per action, and joint moves are
linearly ramped at 20 Hz rather than snapped. Each ramp tick only publishes
the joints that actually moved, so a single-joint wave costs ~20 msg/s instead
of ~140.

## Smoke tests

Run these individually to isolate a layer:

```bash
python smoke_tests/01_sdk_and_arm.py         # SDK + twin + one joint + gripper
python smoke_tests/02_anthropic.py           # Claude reachable
python smoke_tests/03_mistral_stt.py         # Voxtral reachable
python smoke_tests/04_spacebar.py            # push-to-talk key capture
python smoke_tests/05_executor.py            # motion executor on the twin
python smoke_tests/06_planner.py             # Claude → plan  (--execute to run it)
python smoke_tests/07_camera_data.py         # webcam frames
python smoke_tests/08_vision_planner.py      # Claude Vision → plan
python smoke_tests/09_keyboard_bindings.py   # bindings + clamps, fully offline
```

`09` needs no keys and no network — run it after touching the joint model to
confirm the prompt table, key map, and executor clamps are still in sync.

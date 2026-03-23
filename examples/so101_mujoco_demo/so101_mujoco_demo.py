#!/usr/bin/env python3
"""
SO-101 MuJoCo Demo
==================

Build a simulated SO-101 arm scene in Cyberwave, export it to MuJoCo, and run
it locally.  Joint positions and the observer camera are
always streamed to the Cyberwave frontend.

Commands
--------
  create   — create the Cyberwave environment (arm + observer camera)
  export   — download MuJoCo scene ZIP from Cyberwave into out/
  run      — launch the MuJoCo viewer

Control modes (set via CONTROL_MODE env var)
---------------------------------------------
  sine     — default; sweeps joints through a sine wave (edit so101_mujoco_control.py)
  manual   — no automated motion; use the MuJoCo viewer slider panel (Ctrl+M)

Environment variables (copy .env.example → .env)
-------------------------------------------------
  CYBERWAVE_API_KEY   API token
  CYBERWAVE_BASE_URL  API base URL — MQTT broker derived automatically
  CONTROL_MODE        sine | manual (default: sine)
"""

import argparse, json, os, shutil, sys, time, zipfile
import ctypes
from pathlib import Path

# Select a sensible default OpenGL backend per OS before importing mujoco.
# - Linux: EGL works well for headless rendering.
# - macOS: use CGL (EGL is not supported by MuJoCo there).
# Override by setting MUJOCO_GL in the environment before running.
if "MUJOCO_GL" not in os.environ:
    if sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    elif sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "egl"

# NVIDIA GLX fix — must happen before any mujoco/glfw import.
#
# On GLVND systems the GLFW OpenGL function pointer loader (glad) can fail with
# "gladLoadGL error" because the NVIDIA vendor library has not been brought into
# the process symbol table yet when glfwGetProcAddress is called.
# Force-loading libGLX_nvidia.so.0 via ctypes RTLD_GLOBAL before importing
# mujoco makes the NVIDIA symbols available globally and prevents this error.
_NVIDIA_GLX = "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0"
if Path("/dev/nvidia0").exists():
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    if Path(_NVIDIA_GLX).exists():
        try:
            ctypes.CDLL(_NVIDIA_GLX, ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # non-fatal: fall back to env-var-only fix

import mujoco
import mujoco.viewer
import numpy as np
from cyberwave import Cyberwave
from cyberwave.constants import SOURCE_TYPE_EDGE
from cyberwave.sensor.camera_sim import CyberwaveSimStreaming
from cyberwave.sensor.config import cameras_from_schema

# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_joint_prefix(name: str, uuid_hex: str) -> str:
    """Strip the Cyberwave UUID hex prefix from a MuJoCo joint name.

    MuJoCo export names joints as ``{uuid_hex}__{schema_name}`` (e.g.
    ``a6f99dd2a72745c6b3799f12e0fd8a6d___1``).  The frontend expects the bare
    schema name (``_1``, ``_2``, …) for its index-based URDF joint mapping.
    """
    prefix = uuid_hex + "__"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
OUT_DIR     = SCRIPT_DIR / "out"

ENV_NAME        = "so101_mujoco_demo"
ARM_ASSET_KEY   = "the-robot-studio/so101"
CAM_ASSET_KEY   = "cyberwave/standard-cam"

# MuJoCo viewer look-at point and initial camera angle
CAMERA_LOOKAT = [0.2, 0.0, 0.3]

# Home pose: arm slightly extended for visibility.
# Keys are joint-name fragments; MuJoCo joints whose name contains the key are
# set to the mapped value (radians).
HOME_POSE = {
    "_1": 0.0,   # shoulder yaw
    "_2": 0.5,   # shoulder pitch — tilt forward
    "_3": 0.8,   # elbow — bent
    "_4": 0.0,   # wrist pitch
    "_5": 0.0,   # wrist roll
    "_6": 0.0,   # jaw — open
}

# ── create ────────────────────────────────────────────────────────────────────

def cmd_create(_args):
    """Create workspace/project/env, add arm + observer camera, write out/env.json."""
    client = Cyberwave()

    # Workspace — find or create "SDK Demo Workspace"
    workspaces = client.workspaces.list()
    workspace  = next((w for w in workspaces if w.name == "SDK Demo Workspace"), None)
    if not workspace:
        workspace = client.workspaces.create(name="SDK Demo Workspace")
    print(f"Workspace : {workspace.name} ({workspace.uuid})")

    # Project — find or create "SDK Demo Project" inside that workspace
    projects = client.projects.list(workspace_id=workspace.uuid)
    project  = next((p for p in projects if p.name == "SDK Demo Project"), None)
    if not project:
        project = client.projects.create(name="SDK Demo Project", workspace_id=workspace.uuid)
    print(f"Project   : {project.name} ({project.uuid})")

    # Remove stale env with the same name
    for env in client.environments.list(project_id=project.uuid):
        if env.name == ENV_NAME:
            print(f"Removing old environment: {env.uuid}")
            try:
                client.environments.delete(env.uuid, project_id=project.uuid)
            except Exception as e:
                print(f"  [WARN] delete failed: {e}")

    # Environment
    env = client.environments.create(
        name=ENV_NAME,
        description="SO-101 arm with observer camera",
        project_id=project.uuid,
    )
    print(f"Environment UUID : {env.uuid}")

    # Scene: SO-101 arm
    scene = client.get_scene(env.uuid)
    arm = scene.add_twin(
        asset_key=ARM_ASSET_KEY,
        name="so101_robot",
        position=[0, 0, 0],
        fixed_base=True,
    )
    print(f"SO-101 twin      : {arm.uuid}")

    # Scene: observer camera — directly above the arm, 1.0 m up, pointing straight down.
    cam = scene.add_twin(
        asset_key=CAM_ASSET_KEY,
        name="observer_camera",
        position=[0.0, 0.0, 1.5],
        orientation=[0.0, 0.7071068, 0.0, 0.7071068],
        fixed_base=True,
    )
    print(f"Camera twin      : {cam.uuid}")

    # Persist metadata
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env_data = {
        "environment_uuid": str(env.uuid),
        "environment_name": ENV_NAME,
        "arm_twin_uuid":    str(arm.uuid),
        "camera_twin_uuid": str(cam.uuid),
        "base_url":         client.config.base_url,
    }
    out_path = OUT_DIR / "env.json"
    out_path.write_text(json.dumps(env_data, indent=2))
    print(f"\nWritten: {out_path}")
    print(f"\n{'='*60}\nSUCCESS — next step: just export\n{'='*60}")

# ── export ────────────────────────────────────────────────────────────────────

def cmd_export(_args):
    """Export universal schema + MuJoCo scene ZIP from Cyberwave into out/."""
    env_json = OUT_DIR / "env.json"
    if not env_json.exists():
        sys.exit(f"ERROR: {env_json} not found.\nRun `just create` first.")

    env_uuid = json.loads(env_json.read_text())["environment_uuid"]
    client   = Cyberwave()

    # Universal schema
    print("1. Exporting universal schema...")
    schema      = client.environments.get_universal_schema_json(env_uuid)
    schema_path = OUT_DIR / "universal_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    print(f"   Saved: {schema_path}")

    # MuJoCo ZIP
    print("\n2. Downloading MuJoCo scene ZIP...")
    zip_path = OUT_DIR / "mujoco_scene.zip"
    client.environments.export_mujoco_scene(env_uuid, str(zip_path))
    print(f"   Saved: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

    # Extract
    print("\n3. Extracting...")
    extract_dir = OUT_DIR / "mujoco_scene"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    xml_path = extract_dir / "mujoco_scene.xml"
    print(f"   Scene XML: {xml_path}" if xml_path.exists() else
          f"   [WARN] mujoco_scene.xml not found in {extract_dir}")

    print(f"\n{'='*60}\nSUCCESS — next step: just run\n{'='*60}")

# ── run ───────────────────────────────────────────────────────────────────────

def _setup_viewer_camera(viewer, lookat: list):
    viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance  = 1.2
    viewer.cam.azimuth   = -135.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat[:] = lookat


def _make_centering_fn(home_pose: dict):
    """Return a center_fn(model, data) that sets joints to the home pose."""
    def center_fn(model, data):
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            qadr     = model.jnt_qposadr[i]
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            home_val = next((v for k, v in home_pose.items() if k in jnt_name), None)
            if home_val is not None:
                lo, hi = (model.jnt_range[i] if model.jnt_limited[i] else (-3.14, 3.14))
                data.qpos[qadr] = float(np.clip(home_val, lo, hi))
            elif model.jnt_limited[i]:
                lo, hi = model.jnt_range[i]
                data.qpos[qadr] = 0.5 * (lo + hi)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    return center_fn


def cmd_run(args):
    """Load the exported MuJoCo scene and run with viewer."""
    xml_path = OUT_DIR / "mujoco_scene" / "mujoco_scene.xml"
    if not xml_path.exists():
        sys.exit(f"ERROR: {xml_path} not found.\nRun `just export` first.")

    env_json = OUT_DIR / "env.json"
    if not env_json.exists():
        sys.exit(f"ERROR: {env_json} not found.\nRun `just create` first.")
    env_data = json.loads(env_json.read_text())
    arm_twin_uuid = env_data["arm_twin_uuid"]
    arm_uuid_hex  = arm_twin_uuid.replace("-", "")

    control_mode = os.getenv("CONTROL_MODE", "sine").lower()
    headless     = args.headless

    # Resolve controller
    control_fn = None
    if control_mode == "sine":
        try:
            from so101_mujoco_control import sine_control
            control_fn = sine_control
        except ImportError:
            print("[WARN] so101_mujoco_control.py not found — defaulting to manual mode.")
            control_mode = "manual"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    center_fn = _make_centering_fn(HOME_POSE)
    center_fn(model, data)

    print(f"Model  : {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
    print(f"Control: {control_mode}")

    # One MQTT connection shared by camera streaming and joint publishing.
    cw = Cyberwave()
    cw.mqtt.connect()

    # Camera streaming.  MUJOCO_GL=egl must be set before the process starts
    # (see justfile) — mujoco reads it at import time, not at renderer creation.
    schema_cameras = cameras_from_schema(OUT_DIR / "universal_schema.json")
    max_w = max((c.get("width",  640) for c in schema_cameras), default=640)
    max_h = max((c.get("height", 480) for c in schema_cameras), default=480)
    if model.vis.global_.offwidth  < max_w:
        model.vis.global_.offwidth  = max_w
    if model.vis.global_.offheight < max_h:
        model.vis.global_.offheight = max_h
    streaming = CyberwaveSimStreaming(
        client=cw.mqtt,
        schema_cameras=schema_cameras,
    )
    streaming.start(model)
    print("Camera streaming started.")

    if headless:
        print("\nRunning headless (200 steps)...")
        for _ in range(200):
            if control_fn:
                control_fn(model, data, data.time)
            mujoco.mj_step(model, data)
            streaming.capture(model, data)
            cw.mqtt.update_joints_state(
                twin_uuid=arm_twin_uuid,
                joint_positions={
                    _strip_joint_prefix(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i), arm_uuid_hex): float(data.qpos[model.jnt_qposadr[i]])
                    for i in range(model.njnt)
                    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
                },
                source_type=SOURCE_TYPE_EDGE,
                timestamp=data.time,
            )
        print(f"Done. t={data.time:.3f}s")
        return

    print("\nLaunching MuJoCo viewer — close window to exit.")
    if control_mode == "manual":
        print("\n[MANUAL MODE] No automated motion.")
        print("  Use Ctrl+M to open the slider panel and set joint targets manually.")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            _setup_viewer_camera(viewer, CAMERA_LOOKAT)

            last_t = data.time
            while viewer.is_running():
                t0 = time.perf_counter()

                # Re-center if the simulation was reset
                if data.time < last_t or data.time == 0.0:
                    center_fn(model, data)
                last_t = data.time

                if control_fn and control_mode != "manual":
                    control_fn(model, data, data.time)

                mujoco.mj_step(model, data)

                streaming.capture(model, data)

                cw.mqtt.update_joints_state(
                    twin_uuid=arm_twin_uuid,
                    joint_positions={
                        _strip_joint_prefix(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i), arm_uuid_hex): float(data.qpos[model.jnt_qposadr[i]])
                        for i in range(model.njnt)
                        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
                    },
                    source_type=SOURCE_TYPE_EDGE,
                    timestamp=data.time,
                )

                viewer.sync()

                rem = model.opt.timestep - (time.perf_counter() - t0)
                if rem > 0:
                    time.sleep(rem)
    finally:
        try:
            streaming.stop()
        except Exception:
            pass
        try:
            cw.disconnect()
        except Exception:
            pass

    print(f"\nViewer closed. t={data.time:.3f}s")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create", help="Create the Cyberwave environment")
    sub.add_parser("export", help="Export MuJoCo scene from Cyberwave into out/")
    run_p = sub.add_parser("run",    help="Launch the MuJoCo simulation")
    run_p.add_argument("--headless", action="store_true",
                       help="Run without viewer (CI / smoke-test)")

    args = parser.parse_args()
    {"create": cmd_create, "export": cmd_export, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()

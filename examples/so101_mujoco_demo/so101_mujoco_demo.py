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

import argparse
import fractions
import json
import os
import shutil
import sys
import time
import zipfile
import ctypes
from datetime import datetime, timezone as _tz
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

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from cyberwave import Cyberwave  # noqa: E402
from cyberwave.constants import SOURCE_TYPE_EDGE  # noqa: E402
from cyberwave.sensor.camera_sim import CyberwaveSimStreaming  # noqa: E402
from cyberwave.sensor.config import cameras_from_schema  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _strip_joint_prefix(name: str, uuid_hex: str) -> str:
    """Strip the Cyberwave UUID hex prefix from a MuJoCo joint name.

    MuJoCo export names joints as ``{uuid_hex}__{schema_name}`` (e.g.
    ``a6f99dd2a72745c6b3799f12e0fd8a6d___1``).  The frontend expects the bare
    schema name (``_1``, ``_2``, …) for its index-based URDF joint mapping.
    """
    prefix = uuid_hex + "__"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "out"

ENV_NAME = "so101_mujoco_demo"
ARM_ASSET_KEY = "the-robot-studio/so101"
CAM_ASSET_KEY = "cyberwave/standard-cam"

# Path to Django backend media directory (where telemetry files are stored locally).
# The FileSystemStorage backend uses {media_root}/{relative_path} for telemetry.
# Override via BACKEND_MEDIA_DIR env var when using a non-standard layout.
BACKEND_MEDIA_DIR = Path(
    os.getenv(
        "BACKEND_MEDIA_DIR",
        str(SCRIPT_DIR / "../../../../cyberwave-backend/media"),
    )
).resolve()

# MuJoCo viewer look-at point and initial camera angle
CAMERA_LOOKAT = [0.2, 0.0, 0.3]

# ── Alert schedule ────────────────────────────────────────────────────────────
#
# Each entry fires once when sim-time crosses the threshold (seconds).
_ALERT_SCHEDULE = [
    # (threshold_s, name, severity, description)
    (
        5.0,
        "Joint velocity approaching limit",
        "warning",
        "Left shoulder joint nearing velocity cap during sine sweep",
    ),
    (
        15.0,
        "Torque spike detected",
        "critical",
        "Shoulder pitch joint: 18.7 Nm transient torque peak",
    ),
    (
        35.0,
        "Motion resumed after protection",
        "warning",
        "Arm returned to safe operating range after torque protection",
    ),
]

# ── Camera alert schedule ─────────────────────────────────────────────────────
#
# Same structure as _ALERT_SCHEDULE but fires on the *camera* twin.
_CAM_ALERT_SCHEDULE = [
    # (threshold_s, name, severity, description)
    (
        8.0,
        "Low lighting detected",
        "warning",
        "Scene illumination below recommended threshold for observation camera",
    ),
    (
        20.0,
        "Camera frame drop event",
        "warning",
        "Transient GPU load caused brief frame-rate reduction in camera pipeline",
    ),
    (
        38.0,
        "Vision pipeline nominal",
        "info",
        "Camera auto-exposure settled; image quality stable",
    ),
]


# ── Camera frame recorder ─────────────────────────────────────────────────────


class SimFrameRecorder:
    """Record camera frames from a MuJoCo scene to an H.264/Matroska (.mkv) file.

    Uses a dedicated ``mujoco.Renderer`` so it never contends with the WebRTC
    streaming renderer.  Frames are throttled to *fps* to keep file size small.

    Args:
        output_path: Destination ``.mkv`` file (created on :meth:`start`).
        cam_id: MuJoCo camera index.
        model: ``mujoco.MjModel`` — used to create the internal Renderer.
        width: Render width in pixels (default 320).
        height: Render height in pixels (default 240).
        fps: Target recording frame-rate (default 15).
    """

    def __init__(
        self,
        output_path: Path,
        cam_id: int,
        model,
        width: int = 320,
        height: int = 240,
        fps: int = 15,
    ) -> None:
        self._path = output_path
        self._cam_id = cam_id
        self._fps = fps
        self._interval = 1.0 / fps
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._container = None
        self._video_stream = None
        self._frame_count = 0
        self._last_write_time = 0.0

    def start(self) -> None:
        """Open the MKV container and add an H.264 video stream."""
        import av as _av

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._container = _av.open(str(self._path), "w", format="matroska")
        self._video_stream = self._container.add_stream("libx264", rate=self._fps)
        self._video_stream.width = self._renderer.width
        self._video_stream.height = self._renderer.height
        self._video_stream.pix_fmt = "yuv420p"
        self._video_stream.options = {"preset": "ultrafast", "crf": "28"}

    def capture(self, data) -> None:
        """Render and write one frame (throttled to *fps*)."""
        import av as _av

        now = time.monotonic()
        if now - self._last_write_time < self._interval:
            return
        self._last_write_time = now
        self._renderer.update_scene(data, camera=self._cam_id)
        frame_rgb = self._renderer.render()  # H×W×3 uint8 RGB
        frame = _av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame = frame.reformat(format="yuv420p")
        frame.pts = self._frame_count
        frame.time_base = fractions.Fraction(1, self._fps)
        for packet in self._video_stream.encode(frame):
            self._container.mux(packet)
        self._frame_count += 1

    def stop(self) -> int:
        """Flush encoder and close the file.  Returns number of frames written."""
        if self._video_stream and self._container:
            for packet in self._video_stream.encode():
                self._container.mux(packet)
            self._container.close()
            self._container = None
        self._renderer.close()
        return self._frame_count


def _publish_camera_stored(
    cw,
    cam_twin_uuid: str,
    env_uuid: str,
    storage_path: str,
    ts: float,
) -> None:
    """Publish a ``camera_stored`` MQTT message so the backend creates a
    ``CAMERA_STORED`` TwinTelemetry event and triggers the MKV→MP4→parquet
    pipeline automatically.
    """
    topic = f"{cw.mqtt.topic_prefix}cyberwave/twin/{cam_twin_uuid}/telemetry"
    payload = {
        "type": "camera_stored",
        "timestamp": ts,
        "environment_uuid": env_uuid,
        "paths": [storage_path],
    }
    cw.mqtt.publish(topic, payload)
    print(f"[CAMERA] Published camera_stored → {storage_path}")


# Home pose: arm slightly extended for visibility.
# Keys are joint-name fragments; MuJoCo joints whose name contains the key are
# set to the mapped value (radians).
HOME_POSE = {
    "_1": 0.0,  # shoulder yaw
    "_2": 0.5,  # shoulder pitch — tilt forward
    "_3": 0.8,  # elbow — bent
    "_4": 0.0,  # wrist pitch
    "_5": 0.0,  # wrist roll
    "_6": 0.0,  # jaw — open
}

# ── create ────────────────────────────────────────────────────────────────────


def cmd_create(_args):
    """Create workspace/project/env, add arm + observer camera, write out/env.json."""
    client = Cyberwave()

    # Workspace — find or create "SDK Demo Workspace"
    workspaces = client.workspaces.list()
    workspace = next((w for w in workspaces if w.name == "SDK Demo Workspace"), None)
    if not workspace:
        workspace = client.workspaces.create(name="SDK Demo Workspace")
    print(f"Workspace : {workspace.name} ({workspace.uuid})")

    # Project — find or create "SDK Demo Project" inside that workspace
    projects = client.projects.list(workspace_id=workspace.uuid)
    project = next((p for p in projects if p.name == "SDK Demo Project"), None)
    if not project:
        project = client.projects.create(
            name="SDK Demo Project", workspace_id=workspace.uuid
        )
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
        "arm_twin_uuid": str(arm.uuid),
        "camera_twin_uuid": str(cam.uuid),
        "base_url": client.config.base_url,
    }
    out_path = OUT_DIR / "env.json"
    out_path.write_text(json.dumps(env_data, indent=2))
    print(f"\nWritten: {out_path}")
    print(f"\n{'=' * 60}\nSUCCESS — next step: just export\n{'=' * 60}")


# ── export ────────────────────────────────────────────────────────────────────


def cmd_export(_args):
    """Export universal schema + MuJoCo scene ZIP from Cyberwave into out/."""
    env_json = OUT_DIR / "env.json"
    if not env_json.exists():
        sys.exit(f"ERROR: {env_json} not found.\nRun `just create` first.")

    env_uuid = json.loads(env_json.read_text())["environment_uuid"]
    client = Cyberwave()

    # Universal schema
    print("1. Exporting universal schema...")
    schema = client.environments.get_universal_schema_json(env_uuid)
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
    print(
        f"   Scene XML: {xml_path}"
        if xml_path.exists()
        else f"   [WARN] mujoco_scene.xml not found in {extract_dir}"
    )

    print(f"\n{'=' * 60}\nSUCCESS — next step: just run\n{'=' * 60}")


# ── run ───────────────────────────────────────────────────────────────────────


def _setup_viewer_camera(viewer, lookat: list):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 1.2
    viewer.cam.azimuth = -135.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat[:] = lookat


def _make_centering_fn(home_pose: dict):
    """Return a center_fn(model, data) that sets joints to the home pose."""

    def center_fn(model, data):
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            qadr = model.jnt_qposadr[i]
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            home_val = next((v for k, v in home_pose.items() if k in jnt_name), None)
            if home_val is not None:
                lo, hi = model.jnt_range[i] if model.jnt_limited[i] else (-3.14, 3.14)
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
    arm_uuid_hex = arm_twin_uuid.replace("-", "")
    cam_twin_uuid = env_data["camera_twin_uuid"]
    env_uuid = env_data["environment_uuid"]

    control_mode = os.getenv("CONTROL_MODE", "sine").lower()
    headless = args.headless

    # Resolve controller
    control_fn = None
    if control_mode == "sine":
        try:
            from so101_mujoco_control import sine_control

            control_fn = sine_control
        except ImportError:
            print(
                "[WARN] so101_mujoco_control.py not found — defaulting to manual mode."
            )
            control_mode = "manual"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    center_fn = _make_centering_fn(HOME_POSE)
    center_fn(model, data)

    print(f"Model  : {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
    print(f"Control: {control_mode}")

    # One MQTT connection shared by camera streaming and joint publishing.
    cw = Cyberwave()
    cw.mqtt.connect()

    # Twin objects used for alert emission.
    arm_twin = cw.twin(twin_id=arm_twin_uuid)
    cam_twin = cw.twin(twin_id=cam_twin_uuid)
    print(f"Arm alert target  : {arm_twin.name} ({arm_twin_uuid})")
    print(f"Cam alert target  : {cam_twin.name} ({cam_twin_uuid})")
    _alert_fired: set[float] = set()
    _cam_alert_fired: set[float] = set()

    # Camera streaming.  MUJOCO_GL=egl must be set before the process starts
    # (see justfile) — mujoco reads it at import time, not at renderer creation.
    schema_cameras = cameras_from_schema(OUT_DIR / "universal_schema.json")
    max_w = max((c.get("width", 640) for c in schema_cameras), default=640)
    max_h = max((c.get("height", 480) for c in schema_cameras), default=480)
    if model.vis.global_.offwidth < max_w:
        model.vis.global_.offwidth = max_w
    if model.vis.global_.offheight < max_h:
        model.vis.global_.offheight = max_h
    streaming = CyberwaveSimStreaming(
        client=cw.mqtt,
        schema_cameras=schema_cameras,
    )
    streaming.start(model)
    print("Camera streaming started.")

    # Locate the observer camera in the MuJoCo model for the frame recorder.
    cam_uuid_hex = cam_twin_uuid.replace("-", "")
    _rec_cam_id = -1
    for _ci in range(model.ncam):
        _cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, _ci) or ""
        if cam_uuid_hex in _cname or "color_camera" in _cname:
            _rec_cam_id = _ci
            print(f"Recording camera  : {_cname!r} (id={_ci})")
            break

    # MKV path is built here but the timestamp will be snapped to loop-start below.
    _run_ts_us = 0  # placeholder — set just before the headless/viewer loop starts
    _run_date = datetime.now(_tz.utc).strftime("%Y-%m-%d")
    # MKV path and recorder are finalised just before the headless/viewer loop starts.
    _mkv_storage_path: str = ""
    _mkv_host_path: Path = Path()
    recorder: "SimFrameRecorder | None" = None
    if _rec_cam_id < 0:
        print("[WARN] Could not find observer camera in model — skipping recording.")

    if headless:
        # Snap MKV timestamp to actual loop-start wall time so the camera session
        # window aligns with the alerts that will be created during the run.
        _run_ts_us = int(time.time() * 1_000_000)
        _run_date = datetime.now(_tz.utc).strftime("%Y-%m-%d")
        if _rec_cam_id >= 0:
            _mkv_storage_path = (
                f"video_telemetry/{cam_twin_uuid}/color_camera"
                f"/{_run_date}/{_run_ts_us}/chunk_0.mkv"
            )
            _mkv_host_path = BACKEND_MEDIA_DIR / _mkv_storage_path
            recorder = SimFrameRecorder(
                output_path=_mkv_host_path,
                cam_id=_rec_cam_id,
                model=model,
                width=320,
                height=240,
                fps=15,
            )
            recorder.start()
            print(f"Frame recorder    : {_mkv_host_path}")

        # Announce camera twin telemetry so the backend records it as a session.
        cw.mqtt.publish_telemetry_start(cam_twin_uuid)

        # Run 20 000 steps = 40 s of sim time to fire every alert in the schedule.
        print("\nRunning headless (20 000 steps / ~40 s sim-time)...")
        for step_i in range(20_000):
            if control_fn:
                control_fn(model, data, data.time)
            mujoco.mj_step(model, data)
            streaming.capture(model, data)
            if recorder is not None:
                recorder.capture(data)
            cw.mqtt.update_joints_state(
                twin_uuid=arm_twin_uuid,
                joint_positions={
                    _strip_joint_prefix(
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                        arm_uuid_hex,
                    ): float(data.qpos[model.jnt_qposadr[i]])
                    for i in range(model.njnt)
                    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
                },
                source_type=SOURCE_TYPE_EDGE,
                timestamp=time.time(),
            )
            # Arm alert schedule.
            for thresh, name, severity, desc in _ALERT_SCHEDULE:
                if data.time >= thresh and thresh not in _alert_fired:
                    _alert_fired.add(thresh)
                    try:
                        arm_twin.alerts.create(
                            name=name,
                            severity=severity,
                            source_type="edge",
                            description=desc,
                            metadata={"sim_time_s": round(data.time, 2)},
                        )
                        print(f"[ALERT][arm] [{severity}] {name} @ t={data.time:.1f}s")
                    except Exception as exc:
                        print(f"[ALERT][arm] create failed: {exc}")
            # Camera alert schedule.
            for thresh, name, severity, desc in _CAM_ALERT_SCHEDULE:
                if data.time >= thresh and thresh not in _cam_alert_fired:
                    _cam_alert_fired.add(thresh)
                    try:
                        cam_twin.alerts.create(
                            name=name,
                            severity=severity,
                            source_type="edge",
                            description=desc,
                            metadata={"sim_time_s": round(data.time, 2)},
                        )
                        print(f"[ALERT][cam] [{severity}] {name} @ t={data.time:.1f}s")
                    except Exception as exc:
                        print(f"[ALERT][cam] create failed: {exc}")
            if step_i % 2000 == 0:
                print(
                    f"  step={step_i} t={data.time:.1f}s alerts_fired={len(_alert_fired)}"
                )
        print(
            f"Done. t={data.time:.3f}s  arm_alerts={len(_alert_fired)}"
            f"  cam_alerts={len(_cam_alert_fired)}"
        )

        # Stop recorder and trigger backend camera pipeline.
        if recorder is not None:
            n_frames = recorder.stop()
            print(f"[CAMERA] Recorder stopped: {n_frames} frames → {_mkv_host_path}")
            if n_frames > 0 and _mkv_host_path.exists():
                _publish_camera_stored(
                    cw=cw,
                    cam_twin_uuid=cam_twin_uuid,
                    env_uuid=env_uuid,
                    storage_path=_mkv_storage_path,
                    ts=time.time(),
                )
            else:
                print("[CAMERA] No frames recorded — skipping camera_stored publish.")

        # Signal session end for both twins.
        try:
            cw.mqtt.publish_telemetry_end(arm_twin_uuid)
        except Exception:
            pass
        try:
            cw.mqtt.publish_telemetry_end(cam_twin_uuid)
        except Exception:
            pass
        try:
            cw.disconnect()
        except Exception:
            pass
        return

    # Snap MKV timestamp to viewer launch time.
    _run_ts_us = int(time.time() * 1_000_000)
    _run_date = datetime.now(_tz.utc).strftime("%Y-%m-%d")
    if _rec_cam_id >= 0:
        _mkv_storage_path = (
            f"video_telemetry/{cam_twin_uuid}/color_camera"
            f"/{_run_date}/{_run_ts_us}/chunk_0.mkv"
        )
        _mkv_host_path = BACKEND_MEDIA_DIR / _mkv_storage_path
        recorder = SimFrameRecorder(
            output_path=_mkv_host_path,
            cam_id=_rec_cam_id,
            model=model,
            width=320,
            height=240,
            fps=15,
        )
        recorder.start()
        print(f"Frame recorder    : {_mkv_host_path}")

    # Announce camera twin telemetry for the interactive session.
    cw.mqtt.publish_telemetry_start(cam_twin_uuid)

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
                if recorder is not None:
                    recorder.capture(data)

                cw.mqtt.update_joints_state(
                    twin_uuid=arm_twin_uuid,
                    joint_positions={
                        _strip_joint_prefix(
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                            arm_uuid_hex,
                        ): float(data.qpos[model.jnt_qposadr[i]])
                        for i in range(model.njnt)
                        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
                    },
                    source_type=SOURCE_TYPE_EDGE,
                    timestamp=time.time(),
                )

                viewer.sync()

                # ── Arm alert schedule ────────────────────────────────────
                for thresh, name, severity, desc in _ALERT_SCHEDULE:
                    if data.time >= thresh and thresh not in _alert_fired:
                        _alert_fired.add(thresh)
                        try:
                            arm_twin.alerts.create(
                                name=name,
                                severity=severity,
                                source_type="edge",
                                description=desc,
                                metadata={"sim_time_s": round(data.time, 2)},
                            )
                            print(
                                f"[ALERT][arm] [{severity}] {name} @ t={data.time:.1f}s"
                            )
                        except Exception as exc:
                            print(f"[ALERT][arm] create failed: {exc}")
                # ── Camera alert schedule ─────────────────────────────────
                for thresh, name, severity, desc in _CAM_ALERT_SCHEDULE:
                    if data.time >= thresh and thresh not in _cam_alert_fired:
                        _cam_alert_fired.add(thresh)
                        try:
                            cam_twin.alerts.create(
                                name=name,
                                severity=severity,
                                source_type="edge",
                                description=desc,
                                metadata={"sim_time_s": round(data.time, 2)},
                            )
                            print(
                                f"[ALERT][cam] [{severity}] {name} @ t={data.time:.1f}s"
                            )
                        except Exception as exc:
                            print(f"[ALERT][cam] create failed: {exc}")

                rem = model.opt.timestep - (time.perf_counter() - t0)
                if rem > 0:
                    time.sleep(rem)
    finally:
        try:
            streaming.stop()
        except Exception:
            pass
        if recorder is not None:
            try:
                n_frames = recorder.stop()
                print(
                    f"[CAMERA] Recorder stopped: {n_frames} frames → {_mkv_host_path}"
                )
                if n_frames > 0 and _mkv_host_path.exists():
                    _publish_camera_stored(
                        cw=cw,
                        cam_twin_uuid=cam_twin_uuid,
                        env_uuid=env_uuid,
                        storage_path=_mkv_storage_path,
                        ts=time.time(),
                    )
            except Exception as exc:
                print(f"[CAMERA] Recorder stop failed: {exc}")
        try:
            cw.mqtt.publish_telemetry_end(cam_twin_uuid)
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
    run_p = sub.add_parser("run", help="Launch the MuJoCo simulation")
    run_p.add_argument(
        "--headless", action="store_true", help="Run without viewer (CI / smoke-test)"
    )

    args = parser.parse_args()
    {"create": cmd_create, "export": cmd_export, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()

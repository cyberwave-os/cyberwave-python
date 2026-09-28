"""
Hand-eye calibration: SO-101 arm + Standard Camera, docked at the "gripper" link.

The joint-name mapping (CYBERWAVE_TO_URDF_JOINT) and link name (FK_FRAME) are
confirmed against a live run against the real twins. The board size (BOARD) is
still marked VERIFY below -- measure your printed board, this file's default is
a placeholder. Link validation, docking, and intrinsics handling are all
checked against the real backend at import time and fail loudly rather than
guessing.
"""

import math
import os
import threading
import time

from cyberwave import Cyberwave
import cyberwave
from cyberwave.calibration import (
    CharucoBoard,
    HandEyeSession,
    matrix_to_quat_wxyz,
)
from cyberwave.driver.kinematics.arm import ArmKinematicsConfig, BaseKinematicsManipulator

ARM_TWIN_UUID = os.environ.get("ARM_TWIN_UUID", "c3426ffe-2f98-40a8-b7d7-3e828e14ad2d")

CAMERA_TWIN_UUID = os.environ.get("CAMERA_TWIN_UUID", "ee531f91-7fc0-4ee6-89b6-3547177e1c2c")
ENV_UUID = os.environ.get("ENV_UUID", "fd82e73d-6f27-45ad-be1b-84c9d86de880")

# The camera is mounted on this link. Checked against the arm's own schema at
# runtime below, so a typo here fails fast instead of silently docking to
# "base_link" (the backend's default when attach_to_link is omitted).
# Note: validated via arm.get_schema("/links"), not the generated
# get_twin_links() REST call -- that endpoint's response deserializer has a
# bug in this checkout (AttributeError: 'Optional[str]'), a codegen issue,
# not something to route around here since regenerating the client would
# overwrite any local patch.
FK_FRAME = "gripper"

# --- SO-101 kinematics ------------------------------------------------------
# Cached locally by an earlier `cyberwave so101` run. Confirmed by reading the
# file directly: link "gripper" exists, and its five arm joints (base->gripper)
# are named "1".."6" in kinematic order, not by name -- "6" is the wrist->jaw
# grip-open/close joint and does not move the "gripper" frame itself, so it's
# excluded from arm_joints on purpose.
ARM_URDF_PATH = os.environ.get(
    "ARM_URDF_PATH",
    os.path.expanduser("~/.cyberwave/so101_lib/urdf/SO101/so101_new_calib.urdf"),
)
URDF_JOINT_NAMES = ("1", "2", "3", "4", "5")

# Confirmed by an actual run (twin reports joints ['_1', ..., '_6']): the
# twin's own joint names are the URDF's numeric names with a leading
# underscore, one-to-one. "_6" (the jaw) is excluded for the same reason "6"
# is excluded from URDF_JOINT_NAMES above.
CYBERWAVE_TO_URDF_JOINT = {f"_{n}": n for n in URDF_JOINT_NAMES}

# --- board --------------------------------------------------------------
# VERIFY: measure the printed squares with calipers -- this is the only scale
# input to the whole calibration. Generate a printable image with:
#   from cyberwave.calibration import CharucoBoard
#   CharucoBoard(squares=(5, 7), square_size_m=0.030, marker_size_m=0.022).generate_image()
BOARD = CharucoBoard(squares=(11,8), square_size_m=0.015, marker_size_m=0.011, dictionary="DICT_4X4_50")

SAMPLES = int(os.environ.get("SAMPLES", "12"))
MAX_RESIDUAL_M = 0.005


def _resolve_intrinsics() -> dict | None:
    """Operator-supplied intrinsics, or None to let the session solve its own.

    There is deliberately no FOV-derived fallback. Deriving fx/fy from the twin's
    declared field of view is fine for drawing an FOV cone and not for metric work,
    and being an *input* to hand-eye its error surfaced only as an inflated
    residual with nothing pointing at the cause. Passing None instead makes
    HandEyeSession solve real intrinsics from the same board views it is already
    capturing.
    """
    env = {k: os.environ.get(k) for k in ("FX", "FY", "CX", "CY")}
    if all(env.values()):
        return {k.lower(): float(v) for k, v in env.items()}
    print(
        "No FX/FY/CX/CY env vars set — intrinsics will be solved from the first "
        "board views captured below. Vary the board's angle and distance: views "
        "that are all fronto-parallel leave the focal length poorly constrained."
    )
    return None


def _urdf_joint_positions(joint_view) -> dict:
    """Map the live joint view onto URDF joint names.

    joint_view must be the *same* JointStateView instance across calls, not a
    fresh arm_twin.joints.get() each time -- get() opens a new MQTT subscription
    per call, and its first read can time out to an all-zero fallback pose,
    silently freezing FK at zero-joint configuration on every subsequent sample.
    """
    live = dict(joint_view)
    return {
        urdf_name: live[cw_name]
        for cw_name, urdf_name in CYBERWAVE_TO_URDF_JOINT.items()
        if cw_name in live
    }


cw = Cyberwave()

arm = cw.twins.get(ARM_TWIN_UUID)
camera = cw.twins.get(CAMERA_TWIN_UUID)
joint_view = arm.joints.get()
live_joints = dict(joint_view)
print(f"twin reports joints: {sorted(live_joints)}")
unmapped = set(live_joints) - set(CYBERWAVE_TO_URDF_JOINT)
if unmapped:
    print(
        f"WARNING: twin has joint(s) {sorted(unmapped)} not covered by "
        "CYBERWAVE_TO_URDF_JOINT — these are ignored by forward kinematics. "
        "If one of them is actually an arm-positioning joint (not the gripper's "
        "own open/close), the FK below will be wrong."
    )

valid_links = [link.get("name") for link in (arm.get_schema("/links") or []) if isinstance(link, dict)]
if FK_FRAME not in valid_links:
    raise RuntimeError(f"{FK_FRAME!r} is not a link on {arm.name}. Valid links: {valid_links}")

already_docked = (
    str(camera._data_get("attach_to_twin_uuid") or "") == str(arm.uuid)
    and camera._data_get("attach_to_link") == FK_FRAME
)
if already_docked:
    print(f"camera already docked to {arm.name}::{FK_FRAME}")
else:
    scene = cw.get_scene(ENV_UUID)
    scene.dock(child_twin=camera, parent_twin=arm, link_name=FK_FRAME)
    camera.refresh()
    print(f"docked camera to {arm.name}::{FK_FRAME}")

kinematics = BaseKinematicsManipulator(
    ArmKinematicsConfig(
        urdf_path=ARM_URDF_PATH,
        ee_frame=FK_FRAME,
        arm_joints=URDF_JOINT_NAMES,
    )
)

# Pre-flight: fail here rather than partway through a capture session.
if camera.camera.get_frame(format="numpy", source="cloud") is None:
    raise RuntimeError(
        "No frame available from the camera twin. "
        "Check that the camera driver container is streaming."
    )
intrinsics = _resolve_intrinsics()

session = HandEyeSession(
    camera_twin=camera,
    arm_twin=arm,
    fk_frame=FK_FRAME,
    board=BOARD,
    intrinsics=intrinsics,
    kinematics=kinematics,
    # "local" fails: cyberwave-driver-ee531f91 (this camera's own dedicated
    # driver container, not a so101 child-camera) holds /dev/video0
    # exclusively to serve frames -- confirmed via a direct cv2.VideoCapture
    # test from the host while it's running. "remote_edge" (MQTT take_photo)
    # isn't supported by this camera's driver at all. "cloud" (the REST
    # latest-frame cache that same driver populates) is the only source that
    # actually fits this setup; add_sample() below retries it a few times
    # since the cache can have brief gaps.
    frame_source="cloud",
)

def _print_end_effector_state(session, first_rotation) -> None:
    """Print the FK pose just captured, and its rotation drift from the first sample.

    Rotation spread across samples is exactly what the solver needs -- printing it
    per-capture catches a flat "0.0deg spread" degeneracy (wrist not actually
    reorienting, or a joint-mapping/FK bug) immediately instead of after 12 captures.
    """
    pose = session.samples[-1].base_to_gripper
    x, y, z = pose[:3, 3] * 1000.0
    w, qx, qy, qz = matrix_to_quat_wxyz(pose[:3, :3])
    if first_rotation is None:
        drift = 0.0
    else:
        r_rel = first_rotation.T @ pose[:3, :3]
        drift = math.degrees(math.acos(max(-1.0, min(1.0, (r_rel.trace() - 1.0) / 2.0))))
    print(
        f"    end-effector: pos=({x:.1f}, {y:.1f}, {z:.1f}) mm, "
        f"quat(wxyz)=({w:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}), "
        f"rotation from first capture={drift:.1f} deg"
    )


print("Clamp the board where the camera can see it.")
print("Reorient the wrist between captures — tilt it, don't just slide it.")
first_rotation = None
FRAME_RETRIES = 5
FRAME_RETRY_DELAY_S = 1.0
# The edge driver publishes idle joint telemetry on a timer (~2s), so the live
# view can still hold the *previous* pose for a moment after the arm stops.
# Pressing Enter promptly would then pair a fresh image with a stale pose --
# silently wrong input, which shows up as duplicate end-effector poses across
# consecutive captures and an inflated residual. Block until a genuinely new
# telemetry message lands before reading the joints.
JOINT_FRESH_TIMEOUT_S = 8.0

_joint_updated = threading.Event()
joint_view.on_update(lambda *_: _joint_updated.set())


def _wait_for_fresh_joints() -> None:
    _joint_updated.clear()
    if not _joint_updated.wait(JOINT_FRESH_TIMEOUT_S):
        print(
            f"  WARNING: no joint telemetry for {JOINT_FRESH_TIMEOUT_S:.0f}s — the pose "
            "below may be stale. Is the driver's idle joint telemetry running?"
        )


last_joints = None

while session.sample_count < SAMPLES:
    input(f"[{session.sample_count + 1}/{SAMPLES}] Move the arm, let it fully settle, then press Enter...")
    _wait_for_fresh_joints()
    joints_now = _urdf_joint_positions(joint_view)
    # Identical readings mean the arm didn't actually move (positions are tick
    # quantized, so a real move always changes them). Such a sample adds no
    # information to the solve and, if it came from stale telemetry, actively
    # pairs a new image with an old pose.
    if joints_now == last_joints:
        print("  joints unchanged since the last capture — move the arm, then retry")
        continue
    try:
        for attempt in range(1, FRAME_RETRIES + 1):
            try:
                session.add_sample(joint_positions=joints_now)
                break
            except cyberwave.exceptions.CyberwaveValidationError as exc:
                if "No frame available" not in str(exc) or attempt == FRAME_RETRIES:
                    raise
                print(f"  no frame yet (attempt {attempt}/{FRAME_RETRIES}) — retrying...")
                time.sleep(FRAME_RETRY_DELAY_S)
    except cyberwave.calibration.handeye.BoardNotDetectedError:
        print("  board not detected in the camera frame — try again")
        continue
    last_joints = joints_now
    print(f"  captured ({session.sample_count} total)")
    _print_end_effector_state(session, first_rotation)
    if first_rotation is None:
        first_rotation = session.samples[-1].base_to_gripper[:3, :3]

result = session.solve()
print(
    f"residual: {result.residual_translation_m * 1000:.2f} mm, "
    f"{result.residual_rotation_deg:.2f} deg over {result.sample_count} samples "
    f"(method={result.method})"
)

if result.residual_translation_m < MAX_RESIDUAL_M:
    camera.calibration.set(result, board=BOARD, intrinsics=intrinsics, fk_frame=FK_FRAME)
    print("saved to the camera twin's docking offset")
else:
    print(
        f"residual exceeds {MAX_RESIDUAL_M * 1000:.0f}mm — not saving. "
        "Recapture with more rotation spread, or re-check the VERIFY notes above "
        "(joint mapping, board size, intrinsics)."
    )

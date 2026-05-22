#!/usr/bin/env python3
"""
UGV startup and readiness checks, then optional gimbal head motion over MQTT.

Runs in order:
- ``cw.affect`` from ``CYBERWAVE_CONTROL_SOURCE`` (default ``tele`` → real-world)
- optional MQTT **telemetry wake** (small ±pan wiggle) so REST joint-state can
  catch up before the joint summary (skip with ``UGV_SKIP_TELEMETRY_WAKE=1``)
- joint-state summary
- optional ``motion.asset.pose`` if ``STARTUP_POSE_NAME`` is set
- camera frame readiness sampling
- head pan/tilt startup over MQTT (neutral, pan left/right/center, tilt nod, look-up),
  unless ``UGV_SKIP_HEAD_STARTUP=1``

Head motion uses ``update_joints_state`` with a timestamp (aggregated payload) and
``CYBERWAVE_HEAD_*`` environment variables for joint names and envelope tuning.

Cloud motor calibration (when your twin has ``joint_calibration`` rows) lives on
``Twin.get_calibration`` / ``Twin.update_calibration`` in the SDK, not in this script.

Configuration:
    Required: CYBERWAVE_API_KEY
    Optional: repo-root .env file (loaded automatically if present)
    Optional: UGV_SKIP_HEAD_STARTUP=1 to skip the gimbal sequence
    Optional: UGV_SKIP_TELEMETRY_WAKE=1 to skip the pre-summary MQTT pan wiggle
    Optional: UGV_TELEMETRY_WAKE_PAN_DEG (default 4), UGV_TELEMETRY_WAKE_HOLD_S (default 0.35)
    Optional: UGV_JOINT_REST_POLL_ATTEMPTS (default 5), UGV_JOINT_REST_POLL_INTERVAL_S (default 0.35)
    Optional: CYBERWAVE_CONTROL_SOURCE=tele|sim
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from cyberwave import Cyberwave
from cyberwave.constants import SOURCE_TYPE_SIM, SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE
from cyberwave.exceptions import CyberwaveError

try:
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parent.parent
    _env_file = _repo_root / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass


# --- twin / readiness ---------------------------------------------------------
TWIN_REGISTRY_ID = os.getenv("CYBERWAVE_TWIN_REGISTRY", "unitree/go2")
TWIN_UUID = os.getenv("CYBERWAVE_TWIN_UUID") or None
ENVIRONMENT_ID = os.getenv("CYBERWAVE_ENVIRONMENT_ID") or None
STARTUP_POSE_NAME = None
POSE_TRANSITION_MS = 1200
POSE_HOLD_MS = 300

READINESS_FRAMES = 3
READINESS_INTERVAL_MS = 400
MIN_JPEG_BYTES = 2000

# --- head / gimbal (CYBERWAVE_HEAD_* env) -------------------------------------
PAN_JOINT = (os.getenv("CYBERWAVE_HEAD_PAN_JOINT") or "").strip()
TILT_JOINT = (os.getenv("CYBERWAVE_HEAD_TILT_JOINT") or "").strip()
TILT_INVERT = (os.getenv("CYBERWAVE_HEAD_TILT_INVERT") or "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(str(raw).strip())


def _env_int_positive(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return max(1, default)
    try:
        return max(1, int(float(str(raw).strip())))
    except ValueError:
        return max(1, default)


NEUTRAL_PAN_DEG = _env_float("CYBERWAVE_HEAD_NEUTRAL_PAN_DEG", 0.0)
NEUTRAL_TILT_DEG = _env_float("CYBERWAVE_HEAD_NEUTRAL_TILT_DEG", 0.0)
SWEEP_PAN_AMPLITUDE_DEG = _env_float("CYBERWAVE_HEAD_PATROL_PAN_AMPLITUDE_DEG", 22.0)
STARTUP_PAUSE_S = _env_float("CYBERWAVE_HEAD_STARTUP_PAUSE_S", 0.45)
SWEEP_DWELL_S = _env_float("CYBERWAVE_HEAD_STARTUP_SWEEP_DWELL_S", 0.4)
STARTUP_ACK_NOD_DEG = _env_float("CYBERWAVE_HEAD_STARTUP_ACK_NOD_DEG", 5.0)
FINAL_LOOK_UP_TILT_DEG = _env_float("CYBERWAVE_HEAD_STARTUP_FINAL_LOOK_UP_DEG", 20.0)

# Small pan motion before joint summary so cloud GET joint-states may reflect recent MQTT.
TELEMETRY_WAKE_PAN_DEG = _env_float("UGV_TELEMETRY_WAKE_PAN_DEG", 4.0)
TELEMETRY_WAKE_HOLD_S = _env_float("UGV_TELEMETRY_WAKE_HOLD_S", 0.35)

JOINT_REST_POLL_ATTEMPTS = _env_int_positive("UGV_JOINT_REST_POLL_ATTEMPTS", 5)
JOINT_REST_POLL_INTERVAL_S = max(0.0, _env_float("UGV_JOINT_REST_POLL_INTERVAL_S", 0.35))

ACK_NOD_UP_S = 0.14
ACK_NOD_DOWN_S = 0.14
ACK_NOD_CENTER_S = 0.12


def configure_head_startup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("cyberwave.mqtt").setLevel(logging.INFO)


def _joint_source_type(control_source: str) -> str:
    if control_source == SOURCE_TYPE_SIM:
        return SOURCE_TYPE_SIM_TELE
    return control_source


def _ensure_mqtt_for_head_commands(robot) -> None:
    try:
        if hasattr(robot, "_connect_to_mqtt_if_not_connected"):
            robot._connect_to_mqtt_if_not_connected()
        mqtt = getattr(getattr(robot, "client", None), "mqtt", None)
        twin_uuid = getattr(robot, "uuid", None)
        if mqtt is not None and twin_uuid and hasattr(mqtt, "publish_telemetry_start"):
            mqtt.publish_telemetry_start(twin_uuid)
        if mqtt is not None and not bool(getattr(mqtt, "connected", False)):
            cfg = getattr(getattr(robot, "client", None), "config", None)
            host = getattr(cfg, "mqtt_host", None)
            port = getattr(cfg, "mqtt_port", None)
            tls = getattr(cfg, "mqtt_use_tls", None)
            raise CyberwaveError(
                "MQTT client is not connected; joint publishes are dropped by the SDK.\n"
                f"Broker settings: host={host!r} port={port!r} tls={tls!r}\n"
                "If this is intermittent, check outbound connectivity to the broker (8883/TLS)."
            )
    except CyberwaveError:
        raise
    except Exception as exc:
        cfg = getattr(getattr(robot, "client", None), "config", None)
        host = getattr(cfg, "mqtt_host", None)
        port = getattr(cfg, "mqtt_port", None)
        tls = getattr(cfg, "mqtt_use_tls", None)
        raise CyberwaveError(
            "MQTT is required for head startup (joint publishes over MQTT). "
            f"Could not connect/prepare MQTT: {exc}\n"
            f"Broker settings: host={host!r} port={port!r} tls={tls!r}"
        ) from exc


def _candidate_joint_names(robot) -> List[str]:
    names: List[str] = []
    getters: List[Callable[[], object]] = []
    if hasattr(robot, "joints") and hasattr(robot.joints, "list"):
        getters.append(robot.joints.list)
    if hasattr(robot, "get_controllable_joint_names"):
        getters.append(robot.get_controllable_joint_names)

    for getter in getters:
        try:
            raw = getter()
        except Exception:
            continue
        if isinstance(raw, list):
            for item in raw:
                if item is not None:
                    names.append(str(item))

    seen: set[str] = set()
    out: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _resolve_pan_tilt_joints(
    robot,
    pan_joint_cfg: str,
    tilt_joint_cfg: str,
) -> Tuple[Optional[str], Optional[str]]:
    pan = pan_joint_cfg.strip() or None
    tilt = tilt_joint_cfg.strip() or None
    if pan and tilt:
        return pan, tilt

    lowered = [(n, n.lower()) for n in _candidate_joint_names(robot)]
    if not lowered:
        return pan, tilt

    if not pan:
        for name, low in lowered:
            if any(
                key in low
                for key in (
                    "pan",
                    "head_yaw",
                    "camera_yaw",
                    "neck_yaw",
                    "yaw_joint",
                )
            ):
                pan = name
                break

    if not pan:
        for name, low in lowered:
            if "camera" in low and ("horiz" in low or "yaw" in low or "az" in low):
                pan = name
                break

    if not pan:
        for name, low in lowered:
            if "pt_" in low and "base" in low and "link1" in low and "link2" not in low:
                pan = name
                break
    if not tilt:
        for name, low in lowered:
            if "pt_" in low and "link1" in low and "link2" in low and "base" not in low:
                tilt = name
                break

    if not tilt:
        for name, low in lowered:
            if any(
                key in low
                for key in (
                    "tilt",
                    "head_pitch",
                    "camera_pitch",
                    "neck_pitch",
                    "pitch_joint",
                )
            ):
                tilt = name
                break

    if not tilt:
        for name, low in lowered:
            if "camera" in low and ("vert" in low or "pitch" in low or "elev" in low):
                tilt = name
                break

    if pan == tilt and pan is not None:
        tilt = None

    return pan, tilt


def _final_tilt_deg(tilt_deg: float) -> float:
    return -tilt_deg if TILT_INVERT else tilt_deg


def _apply_gimbal_pose(
    robot,
    *,
    pan_joint: str,
    tilt_joint: str,
    pan_deg: float,
    tilt_deg: float,
    control_source: str,
) -> None:
    tilt_cmd = _final_tilt_deg(tilt_deg)
    pan_rad = math.radians(pan_deg)
    tilt_rad = math.radians(tilt_cmd)
    print(
        f"  mqtt joints: pan={pan_deg:.1f} deg ({pan_rad:.4f} rad), "
        f"tilt={tilt_deg:.1f} deg (command tilt={tilt_cmd:.1f} deg -> {tilt_rad:.4f} rad, "
        f"invert={TILT_INVERT})"
    )
    _ensure_mqtt_for_head_commands(robot)
    st = _joint_source_type(control_source)
    positions_rad = {pan_joint: pan_rad, tilt_joint: tilt_rad}
    mqtt = robot.client.mqtt
    mqtt.update_joints_state(
        robot.uuid,
        positions_rad,
        source_type=st,
        timestamp=time.time(),
    )
    try:
        robot.joints.refresh()
    except Exception:
        pass


def _run_gimbal_sequence(
    robot,
    *,
    pan_joint: str,
    tilt_joint: str,
    control_source: str,
) -> None:
    npan = float(NEUTRAL_PAN_DEG)
    ntilt = float(NEUTRAL_TILT_DEG)
    amp = max(0.0, float(SWEEP_PAN_AMPLITUDE_DEG))
    dwell = max(0.05, float(SWEEP_DWELL_S))
    pause = max(0.05, float(STARTUP_PAUSE_S))
    nod = max(0.0, float(STARTUP_ACK_NOD_DEG))

    print(
        f"head startup: neutral pan={npan:.1f} deg tilt={ntilt:.1f} deg "
        f"(pause {pause:.2f}s)"
    )
    _apply_gimbal_pose(
        robot,
        pan_joint=pan_joint,
        tilt_joint=tilt_joint,
        pan_deg=npan,
        tilt_deg=ntilt,
        control_source=control_source,
    )
    time.sleep(pause)

    did_cal = False
    if amp > 0.0:
        did_cal = True
        pan_left = npan - amp
        pan_right = npan + amp
        print(f"head startup: session cal pan left ({pan_left:.1f} deg) …")
        _apply_gimbal_pose(
            robot,
            pan_joint=pan_joint,
            tilt_joint=tilt_joint,
            pan_deg=pan_left,
            tilt_deg=ntilt,
            control_source=control_source,
        )
        time.sleep(dwell)
        print(f"head startup: session cal pan right ({pan_right:.1f} deg) …")
        _apply_gimbal_pose(
            robot,
            pan_joint=pan_joint,
            tilt_joint=tilt_joint,
            pan_deg=pan_right,
            tilt_deg=ntilt,
            control_source=control_source,
        )
        time.sleep(dwell)
        print("head startup: session cal pan center (dead center) …")
        _apply_gimbal_pose(
            robot,
            pan_joint=pan_joint,
            tilt_joint=tilt_joint,
            pan_deg=npan,
            tilt_deg=ntilt,
            control_source=control_source,
        )
        time.sleep(dwell)

    if nod > 0.0:
        did_cal = True
        print(f"head startup: acknowledge nod ±{nod:.1f} deg at neutral tilt …")
        _apply_gimbal_pose(
            robot,
            pan_joint=pan_joint,
            tilt_joint=tilt_joint,
            pan_deg=npan,
            tilt_deg=ntilt + nod,
            control_source=control_source,
        )
        time.sleep(ACK_NOD_UP_S)
        _apply_gimbal_pose(
            robot,
            pan_joint=pan_joint,
            tilt_joint=tilt_joint,
            pan_deg=npan,
            tilt_deg=ntilt - nod,
            control_source=control_source,
        )
        time.sleep(ACK_NOD_DOWN_S)
        _apply_gimbal_pose(
            robot,
            pan_joint=pan_joint,
            tilt_joint=tilt_joint,
            pan_deg=npan,
            tilt_deg=ntilt,
            control_source=control_source,
        )
        time.sleep(ACK_NOD_CENTER_S)

    if did_cal:
        time.sleep(pause)

    final_tilt = ntilt + float(FINAL_LOOK_UP_TILT_DEG)
    print(
        f"head startup: final look-up tilt={final_tilt:.1f} deg "
        f"(neutral + {FINAL_LOOK_UP_TILT_DEG:.1f} deg)"
    )
    _apply_gimbal_pose(
        robot,
        pan_joint=pan_joint,
        tilt_joint=tilt_joint,
        pan_deg=npan,
        tilt_deg=final_tilt,
        control_source=control_source,
    )


def run_head_startup_for_robot(robot, *, control_source: str) -> bool:
    candidates = _candidate_joint_names(robot)
    if candidates:
        print("candidate joints:", ", ".join(candidates[:32]))

    pan_joint, tilt_joint = _resolve_pan_tilt_joints(robot, PAN_JOINT, TILT_JOINT)
    if not pan_joint or not tilt_joint:
        print(
            "head startup: skipped (could not resolve pan/tilt). "
            "Set CYBERWAVE_HEAD_PAN_JOINT / CYBERWAVE_HEAD_TILT_JOINT in .env, "
            f"or expose joint names on the twin. Candidates: {', '.join(candidates[:32])}"
        )
        return False

    print(f"resolved pan joint: {pan_joint!r}")
    print(f"resolved tilt joint: {tilt_joint!r}")
    _run_gimbal_sequence(
        robot,
        pan_joint=pan_joint,
        tilt_joint=tilt_joint,
        control_source=control_source,
    )
    print("head startup: completed")
    return True


def run_telemetry_wake_wiggle(robot, *, control_source: str) -> bool:
    """
    Nudge pan ± a few degrees over MQTT around neutral, then return to neutral.

    Helps REST ``joint_states`` reflect what the edge last saw after publishes,
    when the first GET would otherwise look ``0`` while hardware is not centered.
    """
    if TELEMETRY_WAKE_PAN_DEG <= 0.0:
        return False

    pan_joint, tilt_joint = _resolve_pan_tilt_joints(robot, PAN_JOINT, TILT_JOINT)
    if not pan_joint or not tilt_joint:
        print(
            "telemetry wake: skipped (pan/tilt not resolved). "
            "Set CYBERWAVE_HEAD_PAN_JOINT / CYBERWAVE_HEAD_TILT_JOINT to enable."
        )
        return False

    d = float(TELEMETRY_WAKE_PAN_DEG)
    hold = max(0.05, float(TELEMETRY_WAKE_HOLD_S))
    npan = float(NEUTRAL_PAN_DEG)
    ntilt = float(NEUTRAL_TILT_DEG)
    print(
        f"telemetry wake: MQTT pan ±{d:.1f}° around neutral "
        f"(hold {hold:.2f}s per step), then joint summary"
    )
    _apply_gimbal_pose(
        robot,
        pan_joint=pan_joint,
        tilt_joint=tilt_joint,
        pan_deg=npan + d,
        tilt_deg=ntilt,
        control_source=control_source,
    )
    time.sleep(hold)
    _apply_gimbal_pose(
        robot,
        pan_joint=pan_joint,
        tilt_joint=tilt_joint,
        pan_deg=npan - d,
        tilt_deg=ntilt,
        control_source=control_source,
    )
    time.sleep(hold)
    _apply_gimbal_pose(
        robot,
        pan_joint=pan_joint,
        tilt_joint=tilt_joint,
        pan_deg=npan,
        tilt_deg=ntilt,
        control_source=control_source,
    )
    time.sleep(max(0.1, hold * 0.5))
    return True


def _joint_positions_near_zero(joints: Dict[str, float], *, eps: float = 1e-4) -> bool:
    if not joints:
        return True
    return all(abs(float(v)) <= eps for v in joints.values())


def _refresh_joint_map_from_rest(robot) -> Dict[str, float]:
    """Poll GET joint-states; some twins only update REST after a delay (or never)."""
    last: Dict[str, float] = {}
    for i in range(JOINT_REST_POLL_ATTEMPTS):
        try:
            robot.joints.refresh()
        except CyberwaveError:
            raise
        last = robot.joints.get_all() or {}
        if not last:
            if i + 1 < JOINT_REST_POLL_ATTEMPTS and JOINT_REST_POLL_INTERVAL_S > 0:
                time.sleep(JOINT_REST_POLL_INTERVAL_S)
            continue
        if not _joint_positions_near_zero(last):
            if i > 0:
                print(
                    f"joint check: REST reported non-zero positions after poll "
                    f"{i + 1}/{JOINT_REST_POLL_ATTEMPTS}"
                )
            return last
        if i + 1 < JOINT_REST_POLL_ATTEMPTS and JOINT_REST_POLL_INTERVAL_S > 0:
            time.sleep(JOINT_REST_POLL_INTERVAL_S)
    return last


def _print_joint_summary(robot) -> None:
    try:
        joints = _refresh_joint_map_from_rest(robot)
    except CyberwaveError as exc:
        print(f"joint check: failed to fetch joint states ({exc})")
        return
    if not joints:
        print(
            "joint check: empty joint map (twin may have no published joint states "
            "yet, or this twin is not exposing joints via the joint-states API)"
        )
        return
    print(f"joint check: {len(joints)} joint(s) online")
    if _joint_positions_near_zero(joints):
        print(
            "joint check: note — all positions are still ~0 rad after MQTT wake and "
            f"{JOINT_REST_POLL_ATTEMPTS} REST poll(s). MQTT commands can still move the "
            "robot; this GET endpoint may not mirror tele/op state for this twin."
        )
    first_items = list(sorted(joints.items()))[:6]
    for name, position_rad in first_items:
        print(f"  {name}: {position_rad:.4f} rad")


def _run_startup_pose(robot) -> None:
    if not STARTUP_POSE_NAME:
        print("startup pose: skipped (STARTUP_POSE_NAME is None)")
        return

    print(f"startup pose: applying {STARTUP_POSE_NAME!r}")
    try:
        robot.motion.asset.pose(
            STARTUP_POSE_NAME,
            transition_ms=POSE_TRANSITION_MS,
            hold_ms=POSE_HOLD_MS,
            sync=True,
        )
        print("startup pose: completed")
    except Exception as exc:
        print(f"startup pose: failed: {exc}")
        keyframes = robot.motion.asset.list_keyframes()
        names = [str(k.get("name", "")) for k in keyframes if k.get("name")]
        if names:
            print("available asset poses:", ", ".join(names[:12]))
        raise


def _check_camera_readiness(robot) -> None:
    print(
        f"camera readiness: capturing {READINESS_FRAMES} frame(s) "
        f"every {READINESS_INTERVAL_MS} ms"
    )
    frames = robot.capture_frames(
        READINESS_FRAMES,
        interval_ms=READINESS_INTERVAL_MS,
        format="bytes",
    )
    sizes = [len(frame) for frame in frames]
    if not sizes or any(size <= 0 for size in sizes):
        raise CyberwaveError("camera readiness failed: one or more empty frames")

    smallest = min(sizes)
    median = int(statistics.median(sizes))
    if smallest < MIN_JPEG_BYTES:
        print(
            "camera readiness: warning — JPEG payloads are very small "
            f"(min={smallest} bytes, median={median} bytes). "
            "A live stream can still be active elsewhere while /latest-frame "
            f"returns a placeholder; try increasing MIN_JPEG_BYTES (now {MIN_JPEG_BYTES}) "
            "or verify media pipeline / sensor_id / cw.affect(...) mode."
        )
    else:
        print(
            "camera readiness: ok "
            f"(min={smallest} bytes, median={median} bytes)"
        )


def main() -> int:
    if not (os.getenv("CYBERWAVE_API_KEY") or "").strip():
        print("ERROR: CYBERWAVE_API_KEY is required", file=sys.stderr)
        return 1

    skip_head = (os.getenv("UGV_SKIP_HEAD_STARTUP") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    skip_wake = (os.getenv("UGV_SKIP_TELEMETRY_WAKE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    control_source = (os.getenv("CYBERWAVE_CONTROL_SOURCE") or SOURCE_TYPE_TELE).strip()

    cw = Cyberwave()
    try:
        if control_source == SOURCE_TYPE_SIM:
            cw.affect("simulation")
        else:
            cw.affect("real-world")

        if TWIN_UUID:
            robot = cw.twin(
                TWIN_REGISTRY_ID or None,
                environment_id=ENVIRONMENT_ID,
                twin_id=TWIN_UUID,
            )
            print(f"twin: {robot.uuid} (existing twin_id)")
        else:
            robot = cw.twin(TWIN_REGISTRY_ID, environment_id=ENVIRONMENT_ID)
            print(f"twin: {robot.uuid} ({TWIN_REGISTRY_ID})")

        if (not skip_wake) or (not skip_head):
            configure_head_startup_logging()

        if not skip_wake:
            try:
                run_telemetry_wake_wiggle(robot, control_source=control_source)
            except CyberwaveError as exc:
                print(f"telemetry wake: failed ({exc})")
        else:
            print("telemetry wake: skipped (UGV_SKIP_TELEMETRY_WAKE)")

        _print_joint_summary(robot)
        _run_startup_pose(robot)

        try:
            _check_camera_readiness(robot)
        except Exception as exc:
            print(f"camera readiness: skipped/failed ({exc})")

        if not skip_head:
            try:
                run_head_startup_for_robot(robot, control_source=control_source)
            except CyberwaveError as exc:
                print(f"head startup: failed ({exc})")
        else:
            print("head startup: skipped (UGV_SKIP_HEAD_STARTUP)")

        print("startup sequence: done")
        return 0
    finally:
        cw.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

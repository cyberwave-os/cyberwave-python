"""
UR7 Santa's Little Helper — MQTT joint control sequence with vacuum gripper.

Moves through a series of joint positions, toggling a vacuum gripper.

Env vars:
    CYBERWAVE_API_KEY       API key
    CYBERWAVE_MQTT_HOST     MQTT broker host
    CYBERWAVE_TWIN_UUID     UR7 twin UUID

Requirements:
    pip install cyberwave
"""

import os
import time

from cyberwave import Cyberwave, SOURCE_TYPE_TELE

VACUUM_ON = 0.5
VACUUM_OFF = -0.5

POSITIONS = [
    {
        "elbow_joint": -1.575,
        "shoulder_lift_joint": -1.572,
        "shoulder_pan_joint": 1.563,
        "wrist_1_joint": -1.569,
        "wrist_2_joint": 1.563,
        "wrist_3_joint": -0.007,
        "ee_fixed_joint": VACUUM_OFF,
    },
    {
        "elbow_joint": -2.410,
        "shoulder_lift_joint": -1.317,
        "shoulder_pan_joint": 0.864,
        "wrist_1_joint": -0.984,
        "wrist_2_joint": 1.563,
        "wrist_3_joint": -0.705,
        "ee_fixed_joint": VACUUM_OFF,
    },
    {
        "elbow_joint": -2.495,
        "shoulder_lift_joint": -1.442,
        "shoulder_pan_joint": 0.864,
        "wrist_1_joint": -0.774,
        "wrist_2_joint": 1.562,
        "wrist_3_joint": -0.705,
        "ee_fixed_joint": VACUUM_OFF,
    },
    {
        "elbow_joint": -2.410,
        "shoulder_lift_joint": -1.317,
        "shoulder_pan_joint": 0.864,
        "wrist_1_joint": -0.984,
        "wrist_2_joint": 1.563,
        "wrist_3_joint": -0.705,
        "ee_fixed_joint": VACUUM_ON,
    },
    {
        "elbow_joint": -1.352,
        "shoulder_lift_joint": -2.462,
        "shoulder_pan_joint": -1.294,
        "wrist_1_joint": -0.895,
        "wrist_2_joint": 1.575,
        "wrist_3_joint": -2.958,
        "ee_fixed_joint": VACUUM_ON,
    },
    {
        "elbow_joint": -1.090,
        "shoulder_lift_joint": -3.050,
        "shoulder_pan_joint": -1.320,
        "wrist_1_joint": -0.551,
        "wrist_2_joint": 1.647,
        "wrist_3_joint": -2.985,
        "ee_fixed_joint": VACUUM_ON,
    },
    {
        "elbow_joint": -1.352,
        "shoulder_lift_joint": -2.462,
        "shoulder_pan_joint": -1.294,
        "wrist_1_joint": -0.895,
        "wrist_2_joint": 1.575,
        "wrist_3_joint": -2.958,
        "ee_fixed_joint": VACUUM_OFF,
    },
]

twin_uuid = os.environ["CYBERWAVE_TWIN_UUID"]

cw = Cyberwave(
    api_key=os.environ.get("CYBERWAVE_API_KEY"),
    mqtt_host=os.environ.get("CYBERWAVE_MQTT_HOST", "localhost"),
    mqtt_port=int(os.environ.get("CYBERWAVE_MQTT_PORT", "8883")),
)
cw.mqtt.connect()
time.sleep(0.5)

print(f"Moving through {len(POSITIONS)} positions…")
for i, joints in enumerate(POSITIONS):
    print(f"  Position {i + 1}/{len(POSITIONS)}")
    cw.mqtt.update_joints_state(
        twin_uuid=twin_uuid,
        joint_positions=joints,
        source_type=SOURCE_TYPE_TELE,
    )
    time.sleep(3)

print("Done.")
cw.disconnect()

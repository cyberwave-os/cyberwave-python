from __future__ import annotations

from typing import Optional

from .client import Client
from .types import Pose2D, Pose3D, JointTargets


class Arm:
    def __init__(self, client: Client, twin_uuid: str):
        self.client = client
        self.twin_uuid = twin_uuid

    async def move_joints(self, targets: JointTargets):
        await self.client._request(
            "POST",
            f"/twins/{self.twin_uuid}/commands",
            json={"name": "arm.move_joints", "payload": {"joints": targets.positions}},
        )

    async def move_pose(self, pose: Pose3D | Pose2D):
        payload = {"pose": {"x": pose.x, "y": pose.y}}
        if hasattr(pose, "z") and getattr(pose, "z") is not None:
            payload["pose"]["z"] = getattr(pose, "z")
        await self.client._request(
            "POST",
            f"/twins/{self.twin_uuid}/commands",
            json={"name": "arm.move_pose", "payload": payload},
        )

    async def open_gripper(self):
        await self.client._request(
            "POST",
            f"/twins/{self.twin_uuid}/commands",
            json={"name": "gripper.open"},
        )

    async def close_gripper(self):
        await self.client._request(
            "POST",
            f"/twins/{self.twin_uuid}/commands",
            json={"name": "gripper.close"},
        )



"""Capability handles (locomotion, joints, flight, …) — not sensor namespaces."""

from .flight import FlightHandle
from .gripper import GripperHandle
from .joints import JointsHandle, controllable_joint_names
from .locomotion import LocomotionHandle
from .pose import PoseHandle
from .power import PowerHandle

__all__ = [
    "FlightHandle",
    "GripperHandle",
    "JointsHandle",
    "controllable_joint_names",
    "LocomotionHandle",
    "PoseHandle",
    "PowerHandle",
]

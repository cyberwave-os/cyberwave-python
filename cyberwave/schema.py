# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Common schema definition for robot description format conversion.

This module defines a unified intermediate representation that can capture
the semantics of different robot description formats while preserving
format-specific information through extensions.
"""

from __future__ import annotations
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class JointType(Enum):
    """Supported joint types across formats."""

    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CONTINUOUS = "continuous"
    FIXED = "fixed"
    FLOATING = "floating"
    PLANAR = "planar"
    SPHERICAL = "spherical"  # SDF/USD specific
    UNIVERSAL = "universal"  # SDF/USD specific


class GeometryType(Enum):
    """Supported geometry types."""

    UNKNOWN = "unknown"
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    MESH = "mesh"
    PLANE = "plane"
    CAPSULE = "capsule"  # MJCF/SDF specific
    ELLIPSOID = "ellipsoid"  # SDF specific


class ActuatorType(Enum):
    """Supported actuator models."""

    DC_MOTOR = "dc_motor"  # Direct torque control
    SERVO = "servo"  # PD control
    VELOCITY = "velocity"  # Velocity damping control
    POSITION = "position"  # PD control
    TORQUE = "torque"  # Direct torque control
    MUSCLE = "muscle"  # Biomechanical muscle model


@dataclass
class Vector3:
    """3D vector representation."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> list[float]:
        """Convert to list format."""
        return [self.x, self.y, self.z]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_list(cls, values: list[float]) -> "Vector3":
        """Create from list of values."""
        return cls(values[0], values[1], values[2])


@dataclass
class Quaternion:
    """Quaternion representation for rotations."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_list(self) -> list[float]:
        """Convert to list format [x, y, z, w]."""
        return [self.x, self.y, self.z, self.w]

    @classmethod
    def from_rpy(cls, roll: float, pitch: float, yaw: float) -> "Quaternion":
        """Create quaternion from roll-pitch-yaw angles (Fixed XYZ convention)."""

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls(x, y, z, w)


@dataclass
class Pose:
    """6DOF pose representation."""

    position: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)

    @classmethod
    def from_xyzrpy(cls, xyz: list[float], rpy: list[float]) -> "Pose":
        """Create pose from position and RPY orientation."""
        return cls(
            position=Vector3.from_list(xyz),
            orientation=Quaternion.from_rpy(rpy[0], rpy[1], rpy[2]),
        )


@dataclass
class Inertia:
    """Inertia tensor representation."""

    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 inertia matrix."""
        return np.array(
            [
                [self.ixx, self.ixy, self.ixz],
                [self.ixy, self.iyy, self.iyz],
                [self.ixz, self.iyz, self.izz],
            ]
        )


@dataclass
class Material:
    """Material properties for visual and collision elements."""

    name: str | None = None
    color: list[float] | None = None  # RGBA
    texture: str | None = None
    specular: list[float] | None = None
    emissive: list[float] | None = None
    shininess: float | None = None


@dataclass
class Geometry:
    """Geometric shape definition."""

    type: GeometryType

    # Box parameters
    size: Vector3 | None = None

    # Cylinder parameters
    radius: float | None = None
    length: float | None = None

    # Sphere parameters
    # radius is shared

    # Mesh parameters
    filename: str | None = None
    scale: Vector3 | None = None

    # Format-specific extensions
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class Visual:
    """Visual representation of a link."""

    name: str | None = None
    pose: Pose = field(default_factory=Pose)
    geometry: Geometry | None = None
    material: Material | None = None
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class Collision:
    """Collision representation of a link."""

    name: str | None = None
    pose: Pose = field(default_factory=Pose)
    geometry: Geometry | None = None

    # Collision group reference (defined in CollisionConfig.groups)
    group: str = "default"

    # Contact properties
    mu_static: float | None = None
    mu_dynamic: float | None = None
    restitution: float | None = None
    stiffness: float | None = None
    damping: float | None = None

    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class Link:
    """Robot link definition."""

    name: str

    # Inertial properties
    mass: float = 0.0
    center_of_mass: Vector3 = field(default_factory=Vector3)
    inertia: Inertia = field(default_factory=Inertia)

    # Geometric representations
    visuals: list[Visual] = field(default_factory=list)
    collisions: list[Collision] = field(default_factory=list)

    # Format-specific extensions
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class JointLimits:
    """Joint limit specification."""

    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None

    # Additional limits for specific formats
    acceleration: float | None = None  # MJCF
    jerkmax: float | None = None  # MJCF


@dataclass
class JointDynamics:
    """Joint dynamics properties."""

    damping: float = 0.0
    friction: float = 0.0
    spring_reference: float = 0.0
    spring_stiffness: float = 0.0

    # Format-specific properties
    armature: float | None = None  # MJCF
    backlash: float | None = None


@dataclass
class Joint:
    """Robot joint definition."""

    name: str
    type: JointType
    parent_link: str
    child_link: str

    # Kinematic properties
    pose: Pose = field(default_factory=Pose)
    axis: Vector3 = field(default_factory=lambda: Vector3(0, 0, 1))

    # Constraints
    limits: JointLimits | None = None
    dynamics: JointDynamics | None = None

    # Safety and calibration
    safety_controller: dict[str, float] | None = None
    calibration: dict[str, float] | None = None

    # Default position
    home_position: float | None = None
    """Default position of the joint.
    
    In meters for prismatic joints, in radians for revolute joints.
    """

    # Format-specific extensions
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class Actuator:
    """Actuator/motor definition."""

    name: str
    joint: str
    type: ActuatorType

    # Motor parameters
    torque_constant: float | None = None
    gear_ratio: float | None = None
    max_current: float | None = None
    max_torque: float | None = None
    max_velocity: float | None = None

    # Control parameters
    kp: float | None = None  # Proportional gain
    ki: float | None = None  # Integral gain
    kd: float | None = None  # Derivative gain

    # Physical properties
    resistance: float | None = None
    inductance: float | None = None
    efficiency: float | None = None

    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sensor:
    """Sensor definition."""

    name: str
    type: str  # IMU, camera, lidar, force_torque, etc.
    parent_link: str
    pose: Pose = field(default_factory=Pose)

    # Update rate
    update_rate: float | None = None

    # Noise model
    noise: dict[str, float | list[float]] | None = None

    # Sensor-specific parameters
    parameters: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContactSurface:
    """Contact surface properties."""

    mu_static: float = 0.8
    mu_dynamic: float = 0.7
    restitution: float = 0.1
    stiffness: float = 10000.0
    damping: float = 20.0

    # Advanced contact parameters
    slip_compliance: dict[str, float] | None = None
    soft_cfm: float | None = None  # Constraint force mixing
    soft_erp: float | None = None  # Error reduction parameter


@dataclass
class Contact:
    """Contact definition for collision detection."""

    name: str
    link: str
    surface: ContactSurface = field(default_factory=ContactSurface)
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicsSolver:
    """Physics solver configuration."""

    type: str = "quick"  # quick, ode, bullet, dart, simbody
    iterations: int = 50
    sor: float = 1.3  # Successive Over-Relaxation parameter (ODE)
    min_step_size: float = 0.0001
    max_step_size: float = 0.001
    tolerance: float | None = None
    # Format-specific extensions
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class Physics:
    """World-level physics properties."""

    gravity: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, -9.81))
    timestep: float = 0.001
    solver: PhysicsSolver = field(default_factory=PhysicsSolver)
    contact: ContactSurface = field(default_factory=ContactSurface)
    # Format-specific extensions
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    """Scene/render properties."""

    ambient: list[float] | None = None  # RGBA ambient light
    background: list[float] | None = None  # RGBA background color
    shadows: bool = True
    grid: bool = False
    origin_visual: bool = False
    # Format-specific extensions
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollisionGroup:
    """Named collision group with MuJoCo-style bitmask properties.

    Groups are defined in CollisionConfig and referenced by Collision.group.
    """

    name: str
    contype: int = 1  # Bitmask: what collision type(s) this group belongs to
    conaffinity: int = 1  # Bitmask: what collision type(s) this group collides with
    # To disable collisions, set both contype=0 and conaffinity=0


@dataclass
class CollisionExclude:
    """Exclude all collisions between two bodies (links).

    Maps to MuJoCo <contact><exclude body1="..." body2="..."/></contact>
    """

    body1: str  # Link name
    body2: str  # Link name


@dataclass
class CollisionPair:
    """Explicit geom-pair collision with optional contact overrides.

    Maps to MuJoCo <contact><pair geom1="..." geom2="..." .../></contact>

    Note: Bypasses parent-child filtering and contype/conaffinity checks.
    Geoms must be on different bodies (same-body collisions not allowed).
    """

    geom1: str  # Collision name (Collision.name) or link__collision format
    geom2: str  # Collision name (Collision.name) or link__collision format

    # Optional contact property overrides for this pair
    friction: list[float] | None = None  # [sliding, torsional, rolling]
    solref: list[float] | None = None  # [timeconst, dampratio]
    solimp: list[float] | None = None  # [dmin, dmax, width, mid, power]
    condim: int | None = None  # Contact dimensionality (1, 3, 4, 6)
    margin: float | None = None  # Distance margin for contact detection
    gap: float | None = None  # Initial gap between geoms

    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollisionConfig:
    """World-level collision configuration.

    Defines collision groups, body-level exclusions, and explicit geom pairs.
    """

    # Named collision groups (e.g., "default", "ROBOT", "ENV")
    groups: dict[str, CollisionGroup] = field(default_factory=dict)

    # Body-level exclusions (e.g., adjacent links, self-collision suppression)
    excludes: list[CollisionExclude] = field(default_factory=list)

    # Explicit geom pairs with optional contact overrides
    pairs: list[CollisionPair] = field(default_factory=list)

    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class Metadata:
    """Robot model metadata."""

    name: str
    version: str = "1.0"  # TODO(AI): What is this used for?
    author: str | None = None
    description: str | None = None
    source_format: str | None = None
    creation_date: str | None = None
    units: str = "SI"  # meters, kg, seconds, radians

    # Format-specific metadata
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommonSchema:
    """Unified robot description schema.

    This schema serves as an intermediate representation that can capture
    the semantics of different robot description formats while preserving
    format-specific information through extensions.

    The schema follows a hierarchical structure:
    - Metadata: Robot-level information
    - Physics: World-level physics properties (gravity, timestep, solver, contact)
    - Scene: Scene/render properties (ambient, background, shadows)
    - Links: Physical bodies with inertial, visual, and collision properties
    - Joints: Kinematic connections between links
    - Actuators: Motors and drive systems
    - Sensors: Sensing systems
    - Contacts: Contact surface definitions

    Extensions allow format-specific features to be preserved during
    conversion while maintaining compatibility across formats.
    """

    metadata: Metadata
    physics: Physics | None = None
    scene: Scene | None = None
    collision_config: CollisionConfig | None = None
    links: list[Link] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    actuators: list[Actuator] = field(default_factory=list)
    sensors: list[Sensor] = field(default_factory=list)
    contacts: list[Contact] = field(default_factory=list)

    # Global extensions for format-specific features
    extensions: dict[str, Any] = field(default_factory=dict)

    def get_link(self, name: str) -> Link | None:
        """Get link by name."""
        for link in self.links:
            if link.name == name:
                return link
        return None

    def get_joint(self, name: str) -> Joint | None:
        """Get joint by name."""
        for joint in self.joints:
            if joint.name == name:
                return joint
        return None

    def get_actuator(self, name: str) -> Actuator | None:
        """Get actuator by name."""
        for actuator in self.actuators:
            if actuator.name == name:
                return actuator
        return None

    def get_root_links(self) -> list[Link]:
        """Get links that are not children of any joint (root links).

        Note: Joints with parent_link='world' are excluded from this calculation,
        so their child links are still considered roots.
        """
        child_links = {
            joint.child_link for joint in self.joints if joint.parent_link != "world"
        }
        return [link for link in self.links if link.name not in child_links]

    def get_single_root_link(self) -> Link:
        """Get the single canonical root link for this schema.

        This enforces the invariant that valid asset schemas have exactly one
        root link. Callers that rely on a well-defined base link (e.g. when
        composing robots into scenes) must use this helper instead of
        interpreting ``get_root_links()`` themselves.

        Raises:
            ValueError: If there are 0 or more than 1 root links.
        """
        roots = self.get_root_links()
        if len(roots) == 0:
            raise ValueError(
                f"Schema {self.metadata.name!r} has no root links "
                "(possible kinematic loop or malformed model)."
            )
        if len(roots) > 1:
            root_names = [link.name for link in roots]
            raise ValueError(
                f"Schema {self.metadata.name!r} has multiple root links: {root_names}. "
                "Expected exactly one canonical root link."
            )
        return roots[0]

    def get_kinematic_tree(self) -> dict[str, list[str]]:
        """Get kinematic tree structure as parent->children mapping."""
        tree = {}
        for joint in self.joints:
            parent = joint.parent_link
            child = joint.child_link
            if parent not in tree:
                tree[parent] = []
            tree[parent].append(child)
        return tree

    def validate(self) -> list[str]:
        """Validate schema consistency and return list of issues.

        Returns:
            List of validation error messages
        """
        issues = []

        # Check for duplicate names
        link_names = [link.name for link in self.links]
        if len(link_names) != len(set(link_names)):
            issues.append("Duplicate link names found.")

        joint_names = [joint.name for joint in self.joints]
        if len(joint_names) != len(set(joint_names)):
            issues.append("Duplicate joint names found.")

        # Check joint references
        # Note: 'world' is a special case - it's a valid parent reference but not an actual link
        for joint in self.joints:
            if joint.parent_link != "world" and not self.get_link(joint.parent_link):
                issues.append(
                    f"Joint {joint.name!r} references unknown parent link: {joint.parent_link}."
                )
            if not self.get_link(joint.child_link):
                issues.append(
                    f"Joint {joint.name!r} references unknown child link: {joint.child_link}."
                )

        # Check actuator references
        for actuator in self.actuators:
            if not self.get_joint(actuator.joint):
                issues.append(
                    f"Actuator {actuator.name!r} references unknown joint: {actuator.joint}."
                )

        # Check sensor references
        # Note: 'world' is a special case - it's a valid parent reference but not an actual link
        for sensor in self.sensors:
            if sensor.parent_link != "world" and not self.get_link(sensor.parent_link):
                issues.append(
                    f"Sensor {sensor.name!r} references unknown parent link: {sensor.parent_link}."
                )

        # Check for kinematic loops (basic check)
        try:
            roots = self.get_root_links()
            if len(roots) == 0:
                issues.append("No root links found - possible kinematic loop.")
        except Exception:
            issues.append("Error analyzing kinematic structure.")

        return issues

    def to_dict(self) -> dict:
        """Convert schema to dictionary representation.

        Uses cattrs for automatic recursive conversion. Vector3/Quaternion are
        serialized as canonical dicts ({"x": ..., "y": ..., "z": ...}).

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        # Import here to avoid circular import (serialization imports schema types)
        from cyberwave_robot_format.serialization import converter

        return converter.unstructure(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommonSchema":
        """Reconstruct CommonSchema from dictionary representation.

        Uses cattrs for automatic recursive conversion. Vector3/Quaternion must
        be dicts (list format is not accepted).

        Args:
            data: Dictionary as produced by to_dict()

        Returns:
            CommonSchema instance
        """
        # Import here to avoid circular import (serialization imports schema types)
        from cyberwave_robot_format.serialization import converter

        return converter.structure(data, cls)

    def merge_in(
        self,
        other: CommonSchema,
        instance_name: str,
        spawn_pose: Pose,
        fixed_base: bool = True,
    ) -> None:
        """Merge another schema into this one with deterministic namespacing.

        This method enables composition of multiple robot schemas into one by:
        1. Prefixing all names from `other` with `<instance_name>__` to avoid collisions
        2. Rewriting all internal references consistently
        3. Adding a world joint (FIXED or FREE) to position the merged robot in space

        Args:
            other: The schema to merge into this one
            instance_name: Prefix for all names from `other` (must match [A-Za-z0-9_]+)
            spawn_pose: Pose for spawning the merged robot in world frame (always required)
            fixed_base: If True, creates FIXED joint (robot bolted in place).
                       If False, creates FLOATING joint (6-DOF mobile robot like wheeled platform/drone)

        Raises:
            ValueError: If instance_name is invalid, or if ``other`` does not have
                exactly one root link.

        Example:
            >>> scene = CommonSchema(metadata=Metadata(name="scene"))
            >>> # Fixed-base robot arm
            >>> scene.merge_in(arm_schema, "arm1", spawn_pose=Pose(), fixed_base=True)
            >>> # Mobile wheeled robot
            >>> scene.merge_in(mobile_schema, "robot1",
            ...                spawn_pose=Pose(position=Vector3(2, 0, 0)),
            ...                fixed_base=False)

        Note:
            This method mutates `self`. The `other` schema is deep-copied before modification.
            Cross-robot attachments (e.g., arm-on-platform) are not yet supported but can be
            added in the future via an additional parameter.
        """
        # Validate instance_name
        if not instance_name:
            raise ValueError("instance_name cannot be empty.")
        if not re.match(r"^[A-Za-z0-9_]+$", instance_name):
            raise ValueError(
                f"instance_name must match [A-Za-z0-9_]+, got: {instance_name!r}"
            )

        # Determine root link for spawn joint (must be unique)
        root_link = other.get_single_root_link()

        # Deep copy to avoid mutating the original
        other_copy = deepcopy(other)

        # Build name mapping
        prefix = f"{instance_name}__"

        def prefixed(name: str) -> str:
            """Apply prefix to a name."""
            return f"{prefix}{name}"

        # Rewrite all links
        for link in other_copy.links:
            link.name = prefixed(link.name)

        # Rewrite all joints and their references
        for joint in other_copy.joints:
            joint.name = prefixed(joint.name)
            if joint.parent_link != "world":
                joint.parent_link = prefixed(joint.parent_link)
            joint.child_link = prefixed(joint.child_link)

        # Rewrite all actuators and their references
        for actuator in other_copy.actuators:
            actuator.name = prefixed(actuator.name)
            actuator.joint = prefixed(actuator.joint)

        # Rewrite all sensors and their references
        for sensor in other_copy.sensors:
            sensor.name = prefixed(sensor.name)
            if sensor.parent_link != "world":
                sensor.parent_link = prefixed(sensor.parent_link)

        # Rewrite all contacts and their references
        for contact in other_copy.contacts:
            contact.name = prefixed(contact.name)
            contact.link = prefixed(contact.link)

        # Merge collision_config with prefixing
        if other_copy.collision_config:
            # Ensure self has collision_config
            if self.collision_config is None:
                self.collision_config = CollisionConfig()

            # Merge groups (copy without prefixing group names - they're global identifiers)
            for group_name, group in other_copy.collision_config.groups.items():
                if group_name not in self.collision_config.groups:
                    self.collision_config.groups[group_name] = group

            # Merge excludes with prefixed body names
            for exclude in other_copy.collision_config.excludes:
                prefixed_exclude = CollisionExclude(
                    body1=prefixed(exclude.body1), body2=prefixed(exclude.body2)
                )
                self.collision_config.excludes.append(prefixed_exclude)

            # Merge pairs with prefixed geom names
            for pair in other_copy.collision_config.pairs:
                prefixed_pair = CollisionPair(
                    geom1=prefixed(pair.geom1),
                    geom2=prefixed(pair.geom2),
                    friction=pair.friction,
                    solref=pair.solref,
                    solimp=pair.solimp,
                    condim=pair.condim,
                    margin=pair.margin,
                    gap=pair.gap,
                    extensions=pair.extensions,
                )
                self.collision_config.pairs.append(prefixed_pair)

        # Append all entities to self
        self.links.extend(other_copy.links)
        self.joints.extend(other_copy.joints)
        self.actuators.extend(other_copy.actuators)
        self.sensors.extend(other_copy.sensors)
        self.contacts.extend(other_copy.contacts)

        # Add spawn joint (always required)
        spawn_joint = Joint(
            name=prefixed("spawn"),
            type=JointType.FIXED if fixed_base else JointType.FLOATING,
            parent_link="world",
            child_link=prefixed(root_link.name),
            pose=spawn_pose,
        )
        self.joints.append(spawn_joint)

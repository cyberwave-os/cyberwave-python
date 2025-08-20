from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal, Any
import numpy as np

@dataclass
class Mesh:
    """
    A triangle / textured mesh in world coordinates.
    """
    path: Path                 # .glb / .obj / …
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))   # 4×4 homogeneous transform
    mime_type: str = "model/gltf-binary"
    metadata: Dict[str, str] = field(default_factory=dict)
    id: Optional[Union[int, str]] = None


@dataclass
class Joint:
    """
    A single joint in a kinematic chain.
    """
    name: str
    parent: Optional[str]      # None for root
    pose: np.ndarray           # 4×4 transform relative to parent


@dataclass
class Skeleton:
    """
    A full kinematic chain (e.g. 6-DOF arm) in world coordinates.
    """
    joints: List[Joint]
    reference_mesh: Optional[Mesh] = None      # optional visualisation
    metadata: Dict[str, str] = field(default_factory=dict)
    id: Optional[Union[int, str]] = None

# --- NEW Dataclasses --- 

@dataclass
class Point3D:
    x: float
    y: float
    z: float

@dataclass
class Wall:
    start: Point3D
    end: Point3D
    height: float
    thickness: float
    id: Optional[Union[int, str]] = None

@dataclass
class Door:
    position: Point3D
    width: float
    height: float
    rotation: float # Degrees recommended for simplicity, backend can convert if needed
    id: Optional[Union[int, str]] = None

# Add Window similarly if needed

@dataclass
class FloorPlan:
    """ Represents the static layout of a level. """
    width: float
    length: float
    walls: List[Wall] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    # windows: List[Window] = field(default_factory=list)
    origin_transform: np.ndarray = field(default_factory=lambda: np.eye(4)) # Pose of the plan's origin in project coords
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PinholeCameraIntrinsics:
    """ Pinhole camera model parameters. """
    fx: float
    fy: float
    cx: float
    cy: float
    coeffs: Optional[List[float]] = None # Distortion coeffs (k1, k2, p1, p2, k3)
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class Sensor:
    """ Represents a sensor instance linked to the platform. """
    sensor_type: str # e.g., "camera/rgb", "lidar/3d", "imu", "temperature"
    pose: np.ndarray = field(default_factory=lambda: np.eye(4)) # Pose relative to its parent (e.g., level or robot)
    parent_entity_type: Optional[Literal["level", "robot", "fixed_asset"]] = None
    parent_entity_id: Optional[Union[int, str]] = None
    stream_topic: Optional[str] = None # e.g., MQTT topic or websocket ID for data
    metadata: Dict[str, Any] = field(default_factory=dict)
    camera_intrinsics: Optional[PinholeCameraIntrinsics] = None
    id: Optional[Union[int, str]] = None

@dataclass
class BoxShape:
    center: Tuple[float, float, float]
    size: Tuple[float, float, float] # width, length, height
    rotation_xyz_degrees: Optional[Tuple[float, float, float]] = None # Euler

@dataclass
class PolygonShape:
    vertices: List[Tuple[float, float, float]] = field(default_factory=list) # Expect 3D vertices

@dataclass
class Zone:
    """ A defined spatial area within a level. """
    name: str
    shape_type: Literal["box", "polygon"]
    shape_details: Union[BoxShape, PolygonShape]
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[Union[int, str]] = None

# ---------------------------------------------------------------------
# helper → convert Mesh/Skeleton → Rerun archetypes  (when logging)
# ---------------------------------------------------------------------
import rerun as rr

def log_mesh_rr(entity_path: str, mesh: Mesh):
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=None,       # let rerun parse the file
            mesh_format=rr.MeshFormat.AUTO,
            mesh_file=mesh.path,
            transform=mesh.transform,
            albedo_factor=rr.Color(200, 200, 200, 255),
        ),
    )

def log_skeleton_rr(entity_path: str, skel: Skeleton):
    # 1. log each joint as a coordinate frame (InstancePoses3D)
    for j in skel.joints:
        rr.log(
            f"{entity_path}/{j.name}",
            rr.Arrow3D(origin=[0, 0, 0], vector=[0, 0, 0.2]),  # small axis arrow
        )
        rr.log(
            f"{entity_path}/{j.name}",
            rr.Transform3D(translation=j.pose[:3, 3], mat3x3=j.pose[:3, :3]),
        )
    # 2. option-ally, log parent-child edges as LineStrips3D
    edges = [
        ([j.pose[:3, 3], skel.joints[idx_of(j.parent)].pose[:3, 3]])
        for j in skel.joints if j.parent
    ]
    if edges:
        rr.log(entity_path + "/edges", rr.LineStrips3D(lines=edges))
    # 3. log reference mesh (if any) once per skeleton
    if skel.reference_mesh:
        log_mesh_rr(entity_path + "/ref_mesh", skel.reference_mesh)

# TODO: Add Rerun logging helpers for FloorPlan, Sensor, Zone 
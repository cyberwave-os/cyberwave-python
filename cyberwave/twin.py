"""
High-level Twin abstraction for intuitive digital twin control
"""

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Callable, Type

if TYPE_CHECKING:
    from .client import Cyberwave
    from .camera import CameraStreamer
    from .motion import TwinMotionHandle, TwinNavigationHandle

from .exceptions import CyberwaveError


# Load capabilities cache for runtime class selection
_CAPABILITIES_CACHE: Optional[Dict[str, Any]] = None


def _load_capabilities_cache() -> Dict[str, Any]:
    """Load the capabilities cache from JSON file."""
    global _CAPABILITIES_CACHE
    if _CAPABILITIES_CACHE is None:
        cache_path = Path(__file__).parent / "assets_capabilities.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                _CAPABILITIES_CACHE = json.load(f)
        else:
            _CAPABILITIES_CACHE = {}
    return _CAPABILITIES_CACHE


def _get_asset_capabilities(registry_id: str) -> Dict[str, Any]:
    """Get capabilities for an asset by registry_id."""
    cache = _load_capabilities_cache()
    asset_data = cache.get(registry_id, {})
    return asset_data.get("capabilities", {})


class JointController:
    """Controller for robot joints"""

    def __init__(self, twin: "Twin"):
        self.twin = twin
        self._joint_states: Optional[Dict[str, float]] = None

    def refresh(self):
        """Refresh joint states from the server"""
        try:
            states = self.twin.client.twins.get_joint_states(self.twin.uuid)
            if hasattr(states, "joint_states"):
                self._joint_states = {
                    js.joint_name: js.position for js in states.joint_states
                }
            else:
                self._joint_states = {}
        except Exception as e:
            raise CyberwaveError(f"Failed to refresh joint states: {e}")

    def get(self, joint_name: str) -> float:
        """Get current position of a joint"""
        if self._joint_states is None:
            self.refresh()

        # After refresh, _joint_states should be a dict
        if self._joint_states is None or joint_name not in self._joint_states:
            raise CyberwaveError(f"Joint '{joint_name}' not found")

        return self._joint_states[joint_name]

    def set(
        self,
        joint_name: str,
        position: float,
        degrees: bool = True,
        timestamp: Optional[float] = None,
    ):
        """
        Set position of a joint

        Args:
            joint_name: Name of the joint
            position: Target position
            degrees: If True, position is in degrees; otherwise radians
        """
        if degrees:
            position = math.radians(position)

        try:
            # Connect to MQTT if not already connected
            self.twin._connect_to_mqtt_if_not_connected()

            # Update joint state via MQTT
            self.twin.client.mqtt.update_joint_state(
                self.twin.uuid, joint_name, position=position, timestamp=timestamp
            )

            # Update cached state
            if self._joint_states is None:
                self._joint_states = {}
            self._joint_states[joint_name] = position

        except Exception as e:
            raise CyberwaveError(f"Failed to set joint '{joint_name}': {e}")

    def __getattr__(self, name: str) -> float:
        """Allow accessing joints as attributes (e.g., joints.arm_joint)"""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return self.get(name)

    def __setattr__(self, name: str, value: float):
        """Allow setting joints as attributes (e.g., joints.arm_joint = 45)"""
        if name in ["twin", "_joint_states"]:
            super().__setattr__(name, value)
        else:
            self.set(name, value)

    def list(self) -> List[str]:
        """Get list of all joint names"""
        if self._joint_states is None:
            self.refresh()
        if self._joint_states is None:
            return []
        return list(self._joint_states.keys())

    def get_all(self) -> Dict[str, float]:
        """Get all joint states as a dictionary"""
        if self._joint_states is None:
            self.refresh()
        if self._joint_states is None:
            return {}
        return self._joint_states.copy()


class Twin:
    """
    High-level abstraction for a digital twin.

    Provides intuitive methods for controlling position, rotation, scale,
    and joint states of a digital twin.

    Example:
        >>> twin = client.twin("the-robot-studio/so101")
        >>> twin.edit_position(x=1, y=0, z=0.5)
        >>> twin.rotate(yaw=90)
        >>> twin.joints.arm_joint = 45
    """

    def __init__(self, client: "Cyberwave", twin_data: Any):
        """
        Initialize a Twin instance

        Args:
            client: Cyberwave client instance
            twin_data: Twin schema data from API
        """
        self.client = client
        self._data = twin_data
        self.joints = JointController(self)

        # Cache for current state
        self._position: Optional[Dict[str, float]] = None
        self._rotation: Optional[Dict[str, float]] = None
        self._scale: Optional[Dict[str, float]] = None

        # Lazy-initialized handles
        self._motion: Optional["TwinMotionHandle"] = None
        self._navigation: Optional["TwinNavigationHandle"] = None

    @property
    def motion(self) -> "TwinMotionHandle":
        """
        Get motion handle for pose and animation control.

        Example:
            >>> twin.motion.asset.pose("Picking from below", transition_ms=800)
            >>> twin.motion.twin.animation("wave")
        """
        if self._motion is None:
            from .motion import TwinMotionHandle
            self._motion = TwinMotionHandle(self)
        return self._motion

    @property
    def navigation(self) -> "TwinNavigationHandle":
        """
        Get navigation handle for waypoint-based movement.

        Example:
            >>> twin.navigation.goto([1, 2, 0])
            >>> twin.navigation.follow_path([[0, 0, 0], [1, 0, 0]])
        """
        if self._navigation is None:
            from .motion import TwinNavigationHandle
            self._navigation = TwinNavigationHandle(self)
        return self._navigation

    @property
    def uuid(self) -> str:
        """Get twin UUID"""
        return (
            self._data.uuid
            if hasattr(self._data, "uuid")
            else str(self._data.get("uuid", ""))
        )

    @property
    def name(self) -> str:
        """Get twin name"""
        return (
            self._data.name
            if hasattr(self._data, "name")
            else str(self._data.get("name", ""))
        )

    @property
    def asset_id(self) -> str:
        """Get asset ID"""
        return (
            self._data.asset_uuid
            if hasattr(self._data, "asset_uuid")
            else str(self._data.get("asset_uuid", ""))
        )

    @property
    def environment_id(self) -> str:
        """Get environment ID"""
        return (
            self._data.environment_uuid
            if hasattr(self._data, "environment_uuid")
            else str(self._data.get("environment_uuid", ""))
        )

    def refresh(self):
        """Refresh twin data from the server"""
        try:
            self._data = self.client.twins.get(self.uuid)
            self._position = None
            self._rotation = None
            self._scale = None
        except Exception as e:
            raise CyberwaveError(f"Failed to refresh twin: {e}")

    def edit_position(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ):
        """
        Edit the twin's position in the environment.

        NOTE: Does not move the twin in the real world.

        Args:
            x: X coordinate (optional, keeps current if None)
            y: Y coordinate (optional, keeps current if None)
            z: Z coordinate (optional, keeps current if None)
        """
        # Get current position if needed
        current = self._get_current_position()

        update_data = {
            "position_x": x if x is not None else current.get("x", 0),
            "position_y": y if y is not None else current.get("y", 0),
            "position_z": z if z is not None else current.get("z", 0),
        }

        self._update_state(update_data)

        # Update cache
        self._position = {
            "x": update_data["position_x"],
            "y": update_data["position_y"],
            "z": update_data["position_z"],
        }

    def edit_rotation(
        self,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        quaternion: Optional[List[float]] = None,
    ):
        """
        Edit the twin's rotation in the environment.
        NOTE: Does not rotate the twin in the real world.

        Args:
            yaw: Yaw angle in degrees (rotation around Z axis)
            pitch: Pitch angle in degrees (rotation around Y axis)
            roll: Roll angle in degrees (rotation around X axis)
            quaternion: Quaternion [x, y, z, w] (alternative to euler angles)
        """
        if quaternion is not None:
            if len(quaternion) != 4:
                raise CyberwaveError("Quaternion must be [x, y, z, w]")

            update_data = {
                "rotation_x": quaternion[0],
                "rotation_y": quaternion[1],
                "rotation_z": quaternion[2],
                "rotation_w": quaternion[3],
            }
        else:
            # Convert euler angles to quaternion
            quat = self._euler_to_quaternion(roll or 0, pitch or 0, yaw or 0)
            update_data = {
                "rotation_x": quat[0],
                "rotation_y": quat[1],
                "rotation_z": quat[2],
                "rotation_w": quat[3],
            }

        self._update_state(update_data)

        # Update cache
        self._rotation = {
            "x": update_data["rotation_x"],
            "y": update_data["rotation_y"],
            "z": update_data["rotation_z"],
            "w": update_data["rotation_w"],
        }

    def edit_scale(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ):
        """
        Edit the twin's scale in the environment.
        NOTE: Does not scale the twin in the real world (nothing can be scaled in the real world).

        Args:
            x: X scale factor
            y: Y scale factor
            z: Z scale factor
        """
        current = self._get_current_scale()

        update_data = {
            "scale_x": x if x is not None else current.get("x", 1),
            "scale_y": y if y is not None else current.get("y", 1),
            "scale_z": z if z is not None else current.get("z", 1),
        }

        self._update_state(update_data)

        # Update cache
        self._scale = {
            "x": update_data["scale_x"],
            "y": update_data["scale_y"],
            "z": update_data["scale_z"],
        }

    def delete(self) -> None:
        """Delete this twin"""
        try:
            self.client.twins.delete(self.uuid)  # type: ignore
        except Exception as e:
            raise CyberwaveError(f"Failed to delete twin: {e}")

    def _update_state(self, data: Dict[str, Any]):
        """Update twin state via API"""
        try:
            self.client.twins.update_state(self.uuid, data)  # type: ignore
        except Exception as e:
            raise CyberwaveError(f"Failed to update twin state: {e}")

    def _get_current_position(self) -> Dict[str, float]:
        """Get current position from cache or server"""
        if self._position is None:
            self.refresh()
            if hasattr(self._data, "position_x"):
                self._position = {
                    "x": self._data.position_x,
                    "y": self._data.position_y,
                    "z": self._data.position_z,
                }
            else:
                self._position = {"x": 0, "y": 0, "z": 0}
        return self._position

    def _get_current_scale(self) -> Dict[str, float]:
        """Get current scale from cache or server"""
        if self._scale is None:
            self.refresh()
            if hasattr(self._data, "scale_x"):
                self._scale = {
                    "x": self._data.scale_x,
                    "y": self._data.scale_y,
                    "z": self._data.scale_z,
                }
            else:
                self._scale = {"x": 1, "y": 1, "z": 1}
        return self._scale

    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
        """
        Convert euler angles (degrees) to quaternion

        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees

        Returns:
            [x, y, z, w] quaternion
        """
        # Convert to radians
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        # Calculate quaternion
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

        return [x, y, z, w]

    def __repr__(self) -> str:
        return f"Twin(uuid='{self.uuid}', name='{self.name}')"

    def _connect_to_mqtt_if_not_connected(self):
        """Connect to MQTT if not connected"""
        if not self.client.mqtt.connected:
            self.client.mqtt.connect()

    def subscribe(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to real-time updates"""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.subscribe_twin(self.uuid, on_update)

    def subscribe_position(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to movement updates"""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.subscribe_twin_position(self.uuid, on_update)

    def subscribe_rotation(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to rotation updates"""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.subscribe_twin_rotation(self.uuid, on_update)

    def subscribe_joints(self, on_update: Callable[[Dict[str, Any]], None]):
        """Subscribe to joint updates"""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.subscribe_joint_states(self.uuid, on_update)

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get twin capabilities from the underlying data."""
        if hasattr(self._data, "capabilities"):
            return self._data.capabilities or {}
        elif isinstance(self._data, dict):
            return self._data.get("capabilities", {})
        return {}

    def has_capability(self, capability: str) -> bool:
        """Check if the twin has a specific capability."""
        return bool(self.capabilities.get(capability, False))

    def has_sensor(self, sensor_type: Optional[str] = None) -> bool:
        """Check if the twin has sensors, optionally of a specific type."""
        sensors = self.capabilities.get("sensors", [])
        if not sensors:
            return False
        if sensor_type is None:
            return True
        return any(s.get("type") == sensor_type for s in sensors)


class CameraTwin(Twin):
    """
    Twin with camera/sensor capabilities.

    Provides methods for video streaming and frame capture for twins
    that have RGB or depth sensors.

    Example:
        >>> twin = client.twin("unitree/go2")  # Returns CameraTwin if has sensors
        >>> await twin.start_streaming(fps=15)
        >>> frame = twin.capture_frame()
    """

    _camera_streamer: Optional["CameraStreamer"] = None

    def start_streaming(self, fps: int = 30, camera_id: int = 0) -> "CameraStreamer":
        """
        Start RGB camera streaming.

        Args:
            fps: Frames per second (default: 10)
            camera_id: Camera device ID (default: 0)

        Returns:
            CameraStreamer instance for managing the stream
        """
        self._camera_streamer = self.client.video_stream(
            twin_uuid=self.uuid,
            camera_id=camera_id,
            fps=fps,
        )
        return self._camera_streamer

    def stop_streaming(self) -> None:
        """Stop camera streaming."""
        if self._camera_streamer is not None:
            # The streamer handles cleanup in its stop method
            self._camera_streamer = None

    def capture_frame(self) -> bytes:
        """
        Capture a single frame from the RGB camera.

        Note: Requires an active stream or will start one temporarily.

        Returns:
            Raw frame bytes
        """
        raise NotImplementedError(
            "capture_frame() requires an active stream. "
            "Use start_streaming() first to begin capturing frames."
        )

    def __repr__(self) -> str:
        sensors = self.capabilities.get("sensors", [])
        sensor_types = [s.get("type", "unknown") for s in sensors]
        return f"CameraTwin(uuid='{self.uuid}', name='{self.name}', sensors={sensor_types})"


class DepthCameraTwin(CameraTwin):
    """
    Twin with depth camera capabilities.

    Extends CameraTwin with depth-specific methods for point cloud
    generation and depth frame capture.
    """

    def start_depth_streaming(self, fps: int = 10) -> "CameraStreamer":
        """
        Start depth camera streaming.

        Args:
            fps: Frames per second (default: 10)

        Returns:
            CameraStreamer instance for managing the stream
        """
        return self.start_streaming(fps=fps)

    def stop_depth_streaming(self) -> None:
        """Stop depth streaming."""
        self.stop_streaming()

    def capture_depth_frame(self) -> bytes:
        """
        Capture a single depth frame.

        Returns:
            Raw depth frame bytes
        """
        raise NotImplementedError(
            "capture_depth_frame() requires an active depth stream. "
            "Use start_depth_streaming() first."
        )

    def get_point_cloud(self) -> List[tuple]:
        """
        Get point cloud from depth sensor.

        Returns:
            List of (x, y, z) tuples representing 3D points
        """
        raise NotImplementedError(
            "get_point_cloud() requires depth sensor data processing. "
            "This feature is not yet implemented."
        )

    def __repr__(self) -> str:
        return f"DepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteTwin(Twin):
    """
    Twin that can locomote across space.

    Provides methods for locomotion including movement and rotation.

    Note: Flying twins can locomoate AND fly, so a flying twin is a subset of the LocomoteTwin
    """

    def move(self, position: List[float]):
        """
        Move the twin to a specific position, relative to the zero of the environment.

        NOTE: This does move the real-world robot

        Args:
            position: [x, y, z] coordinates
        """
        if len(position) != 3:
            raise CyberwaveError("Position must be [x, y, z]")

        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.update_twin_position(
            self.uuid, {"x": position[0], "y": position[1], "z": position[2]}
        )

    def move_forward(self, distance: float):
        """
        Move in the direction the twin is facing.

        NOTE: This does move the real-world robot

        Args:
            distance: Distance to move
        """
        # TODO: Implement this. First we should figure the direction the twin is facing, then the relative position to move and move it.
        raise NotImplementedError("move_forward() is not implemented")

    def rotate(
        self,
        *,
        w: Optional[float] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
    ) -> None:
        """
        Rotate the twin using either a quaternion or Euler angles.

        NOTE: This does rotate the real-world robot

        You can provide either:
        - Quaternion values (w, x, y, z)
        - Euler angles in degrees (yaw, pitch, roll)

        Mixing quaternion and Euler angle parameters will raise a ValueError.

        Args:
            w: Quaternion w component
            x: Quaternion x component
            y: Quaternion y component
            z: Quaternion z component
            yaw: Rotation around vertical axis in degrees
            pitch: Rotation around lateral axis in degrees
            roll: Rotation around longitudinal axis in degrees
        """
        # Check if quaternion values are provided
        quaternion_provided = any(v is not None for v in [w, x, y, z])
        euler_provided = any(v is not None for v in [yaw, pitch, roll])

        if quaternion_provided and euler_provided:
            raise ValueError(
                "Cannot mix quaternion (w, x, y, z) and Euler angle (yaw, pitch, roll) "
                "parameters. Use one format or the other."
            )

        if quaternion_provided:
            # Use quaternion directly, defaulting missing values
            rotation = {
                "w": w if w is not None else 1.0,
                "x": x if x is not None else 0.0,
                "y": y if y is not None else 0.0,
                "z": z if z is not None else 0.0,
            }
        elif euler_provided:
            # Convert Euler angles to quaternion
            # Default to 0 for any missing angle
            yaw_rad = math.radians(yaw if yaw is not None else 0.0)
            pitch_rad = math.radians(pitch if pitch is not None else 0.0)
            roll_rad = math.radians(roll if roll is not None else 0.0)

            # Euler to quaternion conversion (ZYX order)
            cy = math.cos(yaw_rad * 0.5)
            sy = math.sin(yaw_rad * 0.5)
            cp = math.cos(pitch_rad * 0.5)
            sp = math.sin(pitch_rad * 0.5)
            cr = math.cos(roll_rad * 0.5)
            sr = math.sin(roll_rad * 0.5)

            rotation = {
                "w": cr * cp * cy + sr * sp * sy,
                "x": sr * cp * cy - cr * sp * sy,
                "y": cr * sp * cy + sr * cp * sy,
                "z": cr * cp * sy - sr * sp * cy,
            }
        else:
            # Default to identity quaternion
            rotation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}

        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.update_twin_rotation(self.uuid, rotation)


class FlyingTwin(Twin):
    """
    Twin with flight capabilities (drones, UAVs).

    Provides methods for aerial control including takeoff, landing,
    and hovering.
    """

    def takeoff(self, altitude: float = 1.0) -> None:
        """
        Take off to the specified altitude.

        Args:
            altitude: Target altitude in meters (default: 1.0)
        """
        self._connect_to_mqtt_if_not_connected()
        # Send takeoff command via MQTT
        self.client.mqtt.publish(
            f"twins/{self.uuid}/commands/takeoff", {"altitude": altitude}
        )

    def land(self) -> None:
        """Land the drone."""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.publish(f"twins/{self.uuid}/commands/land", {})

    def hover(self) -> None:
        """Hover in place."""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.publish(f"twins/{self.uuid}/commands/hover", {})

    def __repr__(self) -> str:
        return f"FlyingTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperTwin(Twin):
    """
    Twin with gripper/manipulation capabilities.

    Provides methods for controlling grippers and end effectors.
    """

    def grip(self, force: float = 1.0) -> None:
        """
        Close the gripper with specified force.

        Args:
            force: Grip force (0.0 to 1.0, default: 1.0)
        """
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.publish(
            f"twins/{self.uuid}/commands/grip", {"force": max(0.0, min(1.0, force))}
        )

    def release(self) -> None:
        """Open the gripper."""
        self._connect_to_mqtt_if_not_connected()
        self.client.mqtt.publish(f"twins/{self.uuid}/commands/release", {})

    def __repr__(self) -> str:
        return f"GripperTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingCameraTwin(FlyingTwin, CameraTwin):
    """Twin with both flight and camera capabilities (camera drones)."""

    def __repr__(self) -> str:
        return f"FlyingCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperCameraTwin(GripperTwin, CameraTwin):
    """Twin with both gripper and camera capabilities (manipulators with vision)."""

    def __repr__(self) -> str:
        return f"GripperCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperDepthCameraTwin(GripperTwin, DepthCameraTwin):
    """Twin with both gripper and depth camera capabilities (manipulators with vision)."""

    def __repr__(self) -> str:
        return f"GripperDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteGripperTwin(LocomoteTwin, GripperTwin):
    """Twin with both locomotive and gripper capabilities (robots with grippers)."""

    def __repr__(self) -> str:
        return f"LocomoteGripperTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingGripperDepthCameraTwin(FlyingTwin, GripperDepthCameraTwin):
    """Twin with both flight and gripper and depth camera capabilities (drones with vision)."""

    def __repr__(self) -> str:
        return f"FlyingGripperDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteGripperDepthCameraTwin(LocomoteTwin, GripperDepthCameraTwin):
    """Twin with both locomotive and gripper and depth camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteGripperDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteDepthCameraTwin(LocomoteTwin, DepthCameraTwin):
    """Twin with both locomotive and depth camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteGripperCameraTwin(LocomoteTwin, GripperCameraTwin):
    """Twin with both locomotive and gripper and camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteGripperCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteCameraTwin(LocomoteTwin, CameraTwin):
    """Twin with both locomotive and camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingGripperCameraTwin(FlyingTwin, GripperCameraTwin):
    """Twin with both flight and gripper and camera capabilities (drones with vision)."""

    def __repr__(self) -> str:
        return f"FlyingGripperCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingDepthCameraTwin(FlyingTwin, DepthCameraTwin):
    """Twin with both flight and depth camera capabilities (drones with vision)."""

    def __repr__(self) -> str:
        return f"FlyingDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


def _select_twin_class(capabilities: Dict[str, Any]) -> Type[Twin]:
    """
    Select the appropriate Twin subclass based on capabilities.

    Args:
        capabilities: Asset capabilities dictionary

    Returns:
        The most appropriate Twin subclass
    """
    has_sensors = bool(capabilities.get("sensors", []))
    has_depth = any(s.get("type") == "depth" for s in capabilities.get("sensors", []))
    can_fly = capabilities.get("can_fly", False)
    can_locomote = capabilities.get("can_locomote", False)
    can_grip = capabilities.get("can_grip", False)

    # Select class based on combination of capabilities
    if can_fly:
        if can_grip and has_depth:
            return FlyingGripperDepthCameraTwin
        elif can_grip and has_sensors:
            return FlyingGripperCameraTwin
        elif has_sensors:
            return FlyingCameraTwin
        elif has_depth:
            return FlyingDepthCameraTwin
        elif can_grip:
            return FlyingGripperCameraTwin
        else:
            return FlyingTwin
    elif can_locomote:
        if can_grip and has_depth:
            return LocomoteGripperDepthCameraTwin
        elif can_grip and has_sensors:
            return LocomoteGripperCameraTwin
        elif can_grip:
            return LocomoteGripperTwin
        elif has_depth:
            return LocomoteDepthCameraTwin
        elif has_sensors:
            return LocomoteCameraTwin
        else:
            return LocomoteTwin
    elif can_grip and has_sensors:
        return GripperCameraTwin
    elif can_grip and has_depth:
        return GripperDepthCameraTwin
    elif can_fly:
        return FlyingTwin
    elif can_locomote:
        return LocomoteTwin
    elif can_grip:
        return GripperTwin
    elif has_depth:
        return DepthCameraTwin
    elif has_sensors:
        return CameraTwin
    else:
        return Twin


def create_twin(
    client: "Cyberwave",
    twin_data: Any,
    registry_id: Optional[str] = None,
) -> Twin:
    """
    Factory function to create the appropriate Twin subclass.

    This function examines the twin's capabilities and returns an instance
    of the most appropriate Twin subclass, providing IDE autocomplete
    for capability-specific methods.

    Args:
        client: Cyberwave client instance
        twin_data: Twin schema data from API
        registry_id: Optional asset registry ID for capability lookup

    Returns:
        Appropriate Twin subclass instance (CameraTwin, FlyingTwin, etc.)

    Example:
        >>> twin = create_twin(client, twin_data, "unitree/go2")
        >>> # twin is CameraTwin with start_streaming() available
    """
    # Get capabilities - prefer cached JSON which has complete capability data
    capabilities = {}

    if registry_id:
        # Use cached capabilities from JSON (most complete source)
        capabilities = _get_asset_capabilities(registry_id)

    # Fall back to twin_data capabilities if no cached data
    if not capabilities:
        if hasattr(twin_data, "capabilities") and twin_data.capabilities:
            caps = twin_data.capabilities
            # Convert to dict if it's an object
            capabilities = (
                caps if isinstance(caps, dict) else getattr(caps, "__dict__", {})
            )
        elif isinstance(twin_data, dict) and twin_data.get("capabilities"):
            capabilities = twin_data["capabilities"]

    # Select and instantiate the appropriate class
    twin_class = _select_twin_class(capabilities)
    return twin_class(client, twin_data)

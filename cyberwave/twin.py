"""
High-level Twin abstraction for intuitive digital twin control
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List, Callable
import math

if TYPE_CHECKING:
    from .client import Cyberwave

from .exceptions import CyberwaveError


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

    def set(self, joint_name: str, position: float, degrees: bool = True):
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
            # Type ignore for auto-generated client
            self.twin.client.twins.update_joint_state(  # type: ignore
                self.twin.uuid, joint_name, position
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
        >>> twin.move(x=1, y=0, z=0.5)
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
            self._data.asset
            if hasattr(self._data, "asset")
            else str(self._data.get("asset", ""))
        )

    @property
    def environment_id(self) -> str:
        """Get environment ID"""
        return (
            self._data.environment
            if hasattr(self._data, "environment")
            else str(self._data.get("environment", ""))
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

    def move(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ):
        """
        Move the twin to a new position

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

    def move_to(self, position: List[float]):
        """
        Move to a specific position

        Args:
            position: [x, y, z] coordinates
        """
        if len(position) != 3:
            raise CyberwaveError("Position must be [x, y, z]")

        self.move(x=position[0], y=position[1], z=position[2])

    def rotate(
        self,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        quaternion: Optional[List[float]] = None,
    ):
        """
        Rotate the twin

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

    def scale(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ):
        """
        Scale the twin

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

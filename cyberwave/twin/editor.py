from __future__ import annotations

from typing import List, Optional


class TwinEditorMixin:
    """REST-only scene layout edits (PR0 extract)."""

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

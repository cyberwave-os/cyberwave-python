"""
Cyberwave Scene Composition Module.

This module provides a high-level API for programmatically composing scenes in Cyberwave.
It allows adding assets (as twins) to environments, with transforms and docking.

Architecture:
- Each Twin owns its universal_schema (snapshot of Asset.universal_schema at creation)
- Twin flat fields store state: position_x/y/z, rotation_w/x/y/z, scale_x/y/z, joint_states
- Scene exports compose all twins' schemas on-demand via compose_environment_schema_from_twins()
- Environment does not store universal_schema
"""

import logging
from typing import Optional, List, Any

from cyberwave.client import Cyberwave
from .schema import CommonSchema, Pose

logger = logging.getLogger(__name__)


class Scene:
    """
    Represents a Cyberwave Environment Scene that can be modified programmatically.

    Usage:
        client = Cyberwave(...)
        scene = client.get_scene("env_uuid")

        # Add a robot
        go2 = scene.add_twin("unitree/go2", name="go2_robot", position=[0, 0, 0])

        # Get the composed schema (for inspection)
        schema = scene.get_composed_schema()
    """

    def __init__(self, client: Cyberwave, environment_id: str):
        self.client = client
        self.environment_id = environment_id
        self._twins: List[Any] = []
        self._load_twins()

    def _load_twins(self):
        """Load twins from the environment."""
        try:
            # Fetch environment and its twins
            base_url = self.client.config.base_url
            if base_url.endswith("/"):
                base_url = base_url[:-1]

            twins_url = f"{base_url}/api/v1/environments/{self.environment_id}/twins"
            logger.info(f"Fetching twins from: {twins_url}")

            response = self.client._api_client.call_api("GET", twins_url)
            import json

            try:
                twins_data = json.loads(response.read().decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode twins data: {response.data}")
                raise

            self._twins = twins_data if isinstance(twins_data, list) else []
            logger.info(f"Loaded {len(self._twins)} twins from environment")

        except Exception as e:
            logger.warning(f"Failed to load twins: {e}")
            self._twins = []

    def get_composed_schema(self) -> CommonSchema:
        """Get the composed scene schema from all twins.

        Fetches the composed schema from the backend API, which uses
        compose_environment_schema_from_twins() to merge all twins' schemas.

        Returns:
            CommonSchema containing world link and all twins' entities merged
        """
        import json

        base_url = self.client.config.base_url
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        # Fetch composed schema from backend API
        schema_url = f"{base_url}/api/v1/environments/{self.environment_id}/universal-schema.json"
        response = self.client._api_client.call_api("GET", schema_url)
        schema_dict = json.loads(response.read().decode("utf-8"))

        return CommonSchema.from_dict(schema_dict)

    # For backward compatibility, provide schema property
    @property
    def schema(self) -> CommonSchema:
        """Get the composed schema (for backward compatibility)."""
        return self.get_composed_schema()

    def add_twin(
        self,
        asset_key: str,
        name: Optional[str] = None,
        pose: Optional[Pose] = None,
        position: Optional[List[float]] = None,
        orientation: Optional[List[float]] = None,
        fixed_base: bool = False,
        **kwargs,
    ) -> Any:
        """
        Add an Asset to the scene as a Twin.

        The backend creates the Twin and initializes its universal_schema.

        Args:
            asset_key: The registry ID or UUID of the asset (e.g. "unitree/go2")
            name: Optional name for the twin. If None, generated from asset name.
            pose: Pose object for spawning.
            position: [x, y, z] list (alternative to pose).
            orientation: [r, p, y] or [x, y, z, w] list (alternative to pose).
            fixed_base: Whether the twin base is fixed to the world (default False).

        Returns:
            The created Twin object from the API.
        """
        # Prepare position/rotation
        if pose is None:
            if position is None:
                position = [0, 0, 0]
            if orientation is None:
                orientation = [0, 0, 0, 1]  # Identity quaternion [x, y, z, w]
        else:
            position = [pose.position.x, pose.position.y, pose.position.z]
            orientation = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]

        # Create Twin via API - backend will initialize Twin.universal_schema
        twin = self.client.twin(
            asset_key=asset_key,
            environment_id=self.environment_id,
            name=name,
            position=position,
            orientation=orientation,
            fixed_base=fixed_base,
        )

        # Reload twins list
        self._load_twins()

        return twin

    def dock(
        self,
        child_twin: Any,
        parent_twin: Any,
        link_name: str,
        offset_position: Optional[List[float]] = None,
        offset_rotation: Optional[List[float]] = None,
    ) -> Any:
        """Dock a child twin to a parent twin's link with optional offset.

        Args:
            child_twin: The twin to dock (can be Twin object or UUID string)
            parent_twin: The parent twin to dock to (can be Twin object or UUID string)
            link_name: Name of the link on parent twin to attach to (e.g., "torso_link")
            offset_position: Optional [x, y, z] offset relative to parent link. If not provided, database defaults are used.
            offset_rotation: Optional [x, y, z, w] quaternion offset relative to parent link. If not provided, database defaults are used.

        Returns:
            Updated Twin object
        """
        # Get UUIDs from twin objects if needed
        child_uuid = child_twin.uuid if hasattr(child_twin, "uuid") else str(child_twin)
        parent_uuid = (
            parent_twin.uuid if hasattr(parent_twin, "uuid") else str(parent_twin)
        )

        # Prepare update kwargs - start with required docking fields
        update_kwargs = {
            "attach_to_twin_uuid": parent_uuid,
            "attach_to_link": link_name,
        }

        # Only add offset fields if user explicitly provided them
        # If not provided, omit from API call - database defaults will be used
        if offset_position is not None:
            update_kwargs["attach_offset_x"] = offset_position[0]
            update_kwargs["attach_offset_y"] = offset_position[1]
            update_kwargs["attach_offset_z"] = offset_position[2]

        if offset_rotation is not None:
            if len(offset_rotation) != 4:
                raise ValueError("offset_rotation must be [x, y, z, w] quaternion")
            update_kwargs["attach_offset_rotation_x"] = offset_rotation[0]
            update_kwargs["attach_offset_rotation_y"] = offset_rotation[1]
            update_kwargs["attach_offset_rotation_z"] = offset_rotation[2]
            update_kwargs["attach_offset_rotation_w"] = offset_rotation[3]

        # Update twin
        updated_twin = self.client.twins.update(child_uuid, **update_kwargs)

        # Reload twins list
        self._load_twins()

        return updated_twin

    def undock(self, twin: Any) -> Any:
        """
        Detach a twin from its parent (undock).

        Args:
            twin: The twin to undock (can be Twin object or UUID string)

        Returns:
            Updated Twin object
        """
        # Get UUID from twin object if needed
        twin_uuid = twin.uuid if hasattr(twin, "uuid") else str(twin)

        # Update twin to remove docking
        updated_twin = self.client.twins.update(
            twin_uuid,
            attach_to_twin_uuid="",  # Empty string detaches
        )

        # Reload twins list
        self._load_twins()

        return updated_twin

    def refresh(self):
        """Reload twins from the environment."""
        self._load_twins()
        logger.info(f"Refreshed scene with {len(self._twins)} twins")

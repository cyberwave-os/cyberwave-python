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
from typing import Optional, List, Any, Sequence

from cyberwave.client import Cyberwave
from .placement import Bounds, CenteredPlacement, compute_centered_placement
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
            except json.JSONDecodeError:
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

        # Map position/orientation lists to the flat fields expected by TwinCreateSchema.
        # (position_x/y/z, rotation_w/x/y/z — not list-based aliases)
        pos = position or [0.0, 0.0, 0.0]
        ori = orientation or [0.0, 0.0, 0.0, 1.0]  # [x, y, z, w]

        # Create Twin via API - backend will initialize Twin.universal_schema
        twin = self.client.twin(
            asset_key=asset_key,
            environment_id=self.environment_id,
            name=name,
            position_x=float(pos[0]),
            position_y=float(pos[1]),
            position_z=float(pos[2]),
            rotation_x=float(ori[0]),
            rotation_y=float(ori[1]),
            rotation_z=float(ori[2]),
            rotation_w=float(ori[3]),
            fixed_base=fixed_base,
        )

        # Reload twins list
        self._load_twins()

        return twin

    def add_twin_centered(
        self,
        asset_key: str,
        *,
        center: Sequence[float],
        asset_bounds: Bounds,
        dimensions: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        rotation: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        fixed_base: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Add an Asset to the scene authored in MuJoCo-style centered notation.

        Unlike :meth:`add_twin`, which expects ``position`` to mean the
        asset's *link origin* in world coordinates (URDF/Cyberwave
        convention), this helper places the asset's *geometric center*
        at ``center`` and (optionally) rescales the asset so its
        world-space AABB matches ``dimensions``. Internally it computes
        the equivalent ``position_*`` / ``scale_*`` / ``rotation_*``
        fields and writes them through the same ``client.twin(...)``
        creation path — backend semantics are unchanged.

        Args:
            asset_key: Registry ID or unified slug of the asset to
                instantiate (e.g. ``"cyberwave/generic_cube"``).
            center: World-space position ``(x, y, z)`` of the asset's
                geometric center.
            asset_bounds: Asset-local AABB
                ``((min_x, min_y, min_z), (max_x, max_y, max_z))``. For
                ``cyberwave/generic_cube`` use
                :data:`cyberwave.placement.GENERIC_CUBE_BOUNDS`. Other
                assets require explicit bounds; see the SDK README's
                "Centered placement" section for guidance.
            dimensions: Optional world dimensions ``(sx, sy, sz)`` of
                the asset's AABB. Mutually exclusive with ``scale``.
            scale: Explicit per-axis scale. Mutually exclusive with
                ``dimensions``. Use this when you want to set the
                center without changing scale.
            rotation: Optional unit quaternion ``(x, y, z, w)``
                orienting the asset's local frame in world. Defaults to
                identity.
            name: Optional twin name. If ``None``, the backend
                generates one from the asset.
            fixed_base: Whether the twin base is fixed to the world.
            **kwargs: Additional twin creation kwargs forwarded to
                :meth:`Cyberwave.twin` (e.g. ``metadata=``).

        Returns:
            The created Twin object from the API (same return type as
            :meth:`add_twin`).

        Example:
            >>> from cyberwave.placement import GENERIC_CUBE_BOUNDS
            >>> support = scene.add_twin_centered(
            ...     "cyberwave/generic_cube",
            ...     name="support_box",
            ...     center=(0.525, 0.0, 0.36),
            ...     dimensions=(0.70, 0.80, 0.72),
            ...     asset_bounds=GENERIC_CUBE_BOUNDS,
            ...     fixed_base=True,
            ... )
        """
        placement: CenteredPlacement = compute_centered_placement(
            center=center,
            asset_bounds=asset_bounds,
            dimensions=dimensions,
            scale=scale,
            rotation=rotation,
        )

        twin = self.client.twin(
            asset_key=asset_key,
            environment_id=self.environment_id,
            name=name,
            position_x=float(placement.position[0]),
            position_y=float(placement.position[1]),
            position_z=float(placement.position[2]),
            rotation_x=float(placement.rotation[0]),
            rotation_y=float(placement.rotation[1]),
            rotation_z=float(placement.rotation[2]),
            rotation_w=float(placement.rotation[3]),
            scale_x=float(placement.scale[0]),
            scale_y=float(placement.scale[1]),
            scale_z=float(placement.scale[2]),
            fixed_base=fixed_base,
            **kwargs,
        )

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

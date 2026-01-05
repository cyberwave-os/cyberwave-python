"""
Motion and navigation handles for digital twin control.

Provides ergonomic wrappers around REST API endpoints for:
- Pose/keyframe control (twin.motion.asset.pose("name"))
- Animation playback (twin.motion.asset.animation("name"))
- Navigation commands (twin.navigation.goto([x, y, z]))
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from .navigation import NavigationPlan

if TYPE_CHECKING:
    from .twin import Twin


class ScopedMotionHandle:
    """
    Motion handle scoped to a specific context (asset, twin, or environment).

    Provides methods for applying poses and animations within a specific scope.

    Example:
        >>> twin.motion.asset.pose("Picking from below", transition_ms=800)
        >>> twin.motion.twin.animation("wave")
    """

    def __init__(
        self,
        parent: "TwinMotionHandle",
        scope: str,
        environment_uuid: Optional[str] = None,
    ):
        self._parent = parent
        self._scope = scope
        self._environment_uuid = environment_uuid

    def list_keyframes(self) -> List[Dict[str, Any]]:
        """List available keyframes/poses for this scope."""
        return self._parent.list_keyframes(
            scope=self._scope, environment_uuid=self._environment_uuid
        )

    def list_animations(self) -> List[Dict[str, Any]]:
        """List available animations for this scope."""
        return self._parent.list_animations(
            scope=self._scope, environment_uuid=self._environment_uuid
        )

    def pose(
        self,
        name: Optional[str] = None,
        *,
        joints: Optional[Dict[str, float]] = None,
        environment_uuid: Optional[str] = None,
        preview: bool = False,
        sync: bool = False,
        source_type: Optional[str] = None,
        transition_ms: Optional[int] = None,
        hold_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Apply a saved pose/keyframe to the twin.

        Args:
            name: Name of the saved pose to apply
            joints: Optional dict of joint positions to apply directly
            environment_uuid: Environment context for the pose
            preview: If True, preview without executing
            sync: If True, wait for completion
            source_type: Source type for tracking
            transition_ms: Duration of the transition in milliseconds
            hold_ms: Duration to hold the pose after reaching it

        Returns:
            Response from the action endpoint

        Example:
            >>> twin.motion.asset.pose("Picking from below", transition_ms=800)
        """
        return self._parent.pose(
            name,
            joints=joints,
            scope=self._scope,
            environment_uuid=environment_uuid or self._environment_uuid,
            preview=preview,
            sync=sync,
            source_type=source_type,
            transition_ms=transition_ms,
            hold_ms=hold_ms,
        )

    def animation(
        self,
        name: str,
        *,
        environment_uuid: Optional[str] = None,
        preview: bool = False,
        sync: bool = False,
        source_type: Optional[str] = None,
        transition_ms: Optional[int] = None,
        hold_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Play a saved animation on the twin.

        Args:
            name: Name of the animation to play
            environment_uuid: Environment context for the animation
            preview: If True, preview without executing
            sync: If True, wait for completion
            source_type: Source type for tracking
            transition_ms: Duration of transitions between poses
            hold_ms: Duration to hold each pose

        Returns:
            Response from the action endpoint

        Example:
            >>> twin.motion.asset.animation("wave", transition_ms=500)
        """
        return self._parent.animation(
            name,
            scope=self._scope,
            environment_uuid=environment_uuid or self._environment_uuid,
            preview=preview,
            sync=sync,
            source_type=source_type,
            transition_ms=transition_ms,
            hold_ms=hold_ms,
        )

    def plan(
        self,
        plan: Dict[str, Any],
        *,
        environment_uuid: Optional[str] = None,
        preview: bool = False,
        sync: bool = False,
        source_type: Optional[str] = None,
        tick_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a motion plan on the twin.

        Args:
            plan: Motion plan dictionary
            environment_uuid: Environment context
            preview: If True, preview without executing
            sync: If True, wait for completion
            source_type: Source type for tracking
            tick_ms: Tick rate for the plan execution

        Returns:
            Response from the action endpoint
        """
        return self._parent.plan(
            plan,
            scope=self._scope,
            environment_uuid=environment_uuid or self._environment_uuid,
            preview=preview,
            sync=sync,
            source_type=source_type,
            tick_ms=tick_ms,
        )


class TwinMotionHandle:
    """
    Handle for twin motion actions (poses, animations, plans).

    Access via `twin.motion`:
        >>> twin.motion.asset.pose("name")  # Asset-scoped pose
        >>> twin.motion.twin.animation("wave")  # Twin-scoped animation
        >>> twin.motion.environment.list_keyframes()  # Environment keyframes
    """

    def __init__(self, twin: "Twin"):
        self._twin = twin
        self.uuid = twin.uuid

    @property
    def environment(self) -> ScopedMotionHandle:
        """Get motion handle scoped to the environment."""
        return ScopedMotionHandle(self, "environment")

    @property
    def twin(self) -> ScopedMotionHandle:
        """Get motion handle scoped to this twin."""
        return ScopedMotionHandle(self, "twin")

    @property
    def asset(self) -> ScopedMotionHandle:
        """Get motion handle scoped to the asset."""
        return ScopedMotionHandle(self, "asset")

    def in_environment(self, environment_uuid: str) -> ScopedMotionHandle:
        """Get motion handle scoped to a specific environment."""
        return ScopedMotionHandle(self, "environment", environment_uuid=environment_uuid)

    def list_keyframes(
        self, scope: str = "twin", environment_uuid: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available keyframes/poses."""
        data = self._get_motions(environment_uuid=environment_uuid)
        keyframes = data.get("keyframes", [])
        if scope == "auto":
            return keyframes
        return [kf for kf in keyframes if kf.get("scope") == scope]

    def list_animations(
        self, scope: str = "twin", environment_uuid: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available animations."""
        data = self._get_motions(environment_uuid=environment_uuid)
        animations = data.get("animations", [])
        if scope == "auto":
            return animations
        return [anim for anim in animations if anim.get("scope") == scope]

    def pose(
        self,
        name: Optional[str] = None,
        *,
        joints: Optional[Dict[str, float]] = None,
        scope: str = "twin",
        environment_uuid: Optional[str] = None,
        preview: bool = False,
        sync: bool = False,
        source_type: Optional[str] = None,
        transition_ms: Optional[int] = None,
        hold_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply a pose/keyframe to the twin."""
        payload: Dict[str, Any] = {
            "action_type": "pose",
            "scope": scope,
            "execution": "sync" if sync else "async",
            "preview": preview,
        }
        if name:
            payload["name"] = name
        if joints:
            payload["joints"] = joints
        if environment_uuid:
            payload["environment_uuid"] = environment_uuid
        if source_type:
            payload["source_type"] = source_type
        if transition_ms is not None:
            payload["transition_ms"] = transition_ms
        if hold_ms is not None:
            payload["hold_ms"] = hold_ms
        return self._post_action(payload)

    def animation(
        self,
        name: str,
        *,
        scope: str = "twin",
        environment_uuid: Optional[str] = None,
        preview: bool = False,
        sync: bool = False,
        source_type: Optional[str] = None,
        transition_ms: Optional[int] = None,
        hold_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Play an animation on the twin."""
        payload: Dict[str, Any] = {
            "action_type": "animation",
            "name": name,
            "scope": scope,
            "execution": "sync" if sync else "async",
            "preview": preview,
        }
        if environment_uuid:
            payload["environment_uuid"] = environment_uuid
        if source_type:
            payload["source_type"] = source_type
        if transition_ms is not None:
            payload["transition_ms"] = transition_ms
        if hold_ms is not None:
            payload["hold_ms"] = hold_ms
        return self._post_action(payload)

    def plan(
        self,
        plan: Dict[str, Any],
        *,
        scope: str = "twin",
        environment_uuid: Optional[str] = None,
        preview: bool = False,
        sync: bool = False,
        source_type: Optional[str] = None,
        tick_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a motion plan on the twin."""
        payload: Dict[str, Any] = {
            "action_type": "plan",
            "plan": plan,
            "scope": scope,
            "execution": "sync" if sync else "async",
            "preview": preview,
        }
        if environment_uuid:
            payload["environment_uuid"] = environment_uuid
        if source_type:
            payload["source_type"] = source_type
        if tick_ms is not None:
            payload["tick_ms"] = tick_ms
        return self._post_action(payload)

    def _get_motions(self, environment_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Get available motions for the twin."""
        api_client = self._twin.client.api.api_client
        
        query_params = []
        if environment_uuid:
            query_params.append(("environment_uuid", environment_uuid))

        _param = api_client.param_serialize(
            method="GET",
            resource_path=f"/api/v1/twins/{self.uuid}/motions",
            query_params=query_params,
            auth_settings=["CustomTokenAuthentication"],
        )

        response_data = api_client.call_api(*_param)
        response_data.read()

        # Return raw JSON response
        import json
        return json.loads(response_data.data.decode("utf-8"))

    def _post_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Post an action to the twin."""
        api_client = self._twin.client.api.api_client

        _param = api_client.param_serialize(
            method="POST",
            resource_path=f"/api/v1/twins/{self.uuid}/actions",
            body=payload,
            header_params={"Content-Type": "application/json"},
            auth_settings=["CustomTokenAuthentication"],
        )

        response_data = api_client.call_api(*_param)
        response_data.read()

        import json
        return json.loads(response_data.data.decode("utf-8"))


class TwinNavigationHandle:
    """
    Handle for twin navigation commands.

    Provides waypoint-based movement for mobile robots and drones.

    Access via `twin.navigation`:
        >>> twin.navigation.goto([1, 2, 0])
        >>> twin.navigation.follow_path([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        >>> twin.navigation.stop()
    """

    def __init__(self, twin: "Twin"):
        self._twin = twin
        self.uuid = twin.uuid
        self._controller_policy_uuid: Optional[str] = None

    def use_controller(self, policy_uuid: str) -> "TwinNavigationHandle":
        """Set a default navigation controller policy UUID for future plans."""
        self._controller_policy_uuid = policy_uuid
        return self

    def clear_controller(self) -> "TwinNavigationHandle":
        """Clear the default navigation controller."""
        self._controller_policy_uuid = None
        return self

    def plan(
        self,
        *,
        plan_id: Optional[str] = None,
        name: Optional[str] = None,
        controller_policy_uuid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NavigationPlan:
        """Create a new navigation plan builder."""
        return NavigationPlan(
            plan_id=plan_id,
            name=name,
            controller_policy_uuid=controller_policy_uuid or self._controller_policy_uuid,
            metadata=metadata,
        )

    def goto(
        self,
        position: Sequence[float],
        *,
        rotation: Optional[Sequence[float]] = None,
        yaw: Optional[float] = None,
        controller_policy_uuid: Optional[str] = None,
        environment_uuid: Optional[str] = None,
        source_type: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Navigate the twin to a specific position.

        Args:
            position: Target [x, y, z] coordinates
            rotation: Target rotation as quaternion [w, x, y, z]
            yaw: Target yaw angle in radians (alternative to rotation)
            controller_policy_uuid: Navigation controller to use
            environment_uuid: Environment context
            source_type: Source type for tracking
            constraints: Navigation constraints
            metadata: Additional metadata

        Returns:
            Response from the navigation endpoint
        """
        if yaw is not None and rotation is not None:
            raise ValueError("Specify either rotation or yaw, not both")

        payload: Dict[str, Any] = {
            "command": "goto",
            "position": [float(value) for value in position],
        }
        if yaw is not None:
            payload["yaw"] = float(yaw)
        if rotation is not None:
            if len(rotation) != 4:
                raise ValueError("rotation must be [w, x, y, z]")
            payload["rotation"] = [float(value) for value in rotation]
        if environment_uuid:
            payload["environment_uuid"] = environment_uuid
        if source_type:
            payload["source_type"] = source_type
        if constraints:
            payload["constraints"] = constraints
        if metadata:
            payload["metadata"] = metadata

        return self._send_command(payload, controller_policy_uuid=controller_policy_uuid)

    def follow_path(
        self,
        waypoints: Iterable[Any] | NavigationPlan,
        *,
        wait_s: float = 0.0,
        max_loops: int = 1,
        yaw: Optional[float] = None,
        controller_policy_uuid: Optional[str] = None,
        environment_uuid: Optional[str] = None,
        source_type: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Follow a path of waypoints.

        Args:
            waypoints: List of waypoints or a NavigationPlan
            wait_s: Time to wait at each waypoint
            max_loops: Number of times to repeat the path
            yaw: Target yaw angle for all waypoints
            controller_policy_uuid: Navigation controller to use
            environment_uuid: Environment context
            source_type: Source type for tracking
            constraints: Navigation constraints
            metadata: Additional metadata

        Returns:
            Response from the navigation endpoint
        """
        normalized = self._normalize_waypoints(waypoints)
        payload: Dict[str, Any] = {
            "command": "path",
            "waypoints": normalized,
        }
        if yaw is not None:
            payload["yaw"] = float(yaw)
        if environment_uuid:
            payload["environment_uuid"] = environment_uuid
        if source_type:
            payload["source_type"] = source_type
        if constraints:
            payload["constraints"] = constraints

        nav_metadata = dict(metadata or {})
        if wait_s > 0:
            nav_metadata["wait_s"] = float(wait_s)
        if max_loops > 1:
            nav_metadata["max_loops"] = int(max_loops)
        if nav_metadata:
            payload["metadata"] = nav_metadata

        return self._send_command(payload, controller_policy_uuid=controller_policy_uuid)

    def stop(
        self,
        *,
        controller_policy_uuid: Optional[str] = None,
        environment_uuid: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stop navigation immediately."""
        payload: Dict[str, Any] = {"command": "stop"}
        if environment_uuid:
            payload["environment_uuid"] = environment_uuid
        if source_type:
            payload["source_type"] = source_type
        return self._send_command(payload, controller_policy_uuid=controller_policy_uuid)

    def pause(
        self,
        *,
        controller_policy_uuid: Optional[str] = None,
        environment_uuid: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pause navigation."""
        payload: Dict[str, Any] = {"command": "pause"}
        if environment_uuid:
            payload["environment_uuid"] = environment_uuid
        if source_type:
            payload["source_type"] = source_type
        return self._send_command(payload, controller_policy_uuid=controller_policy_uuid)

    def resume(
        self,
        *,
        controller_policy_uuid: Optional[str] = None,
        environment_uuid: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resume paused navigation."""
        payload: Dict[str, Any] = {"command": "resume"}
        if environment_uuid:
            payload["environment_uuid"] = environment_uuid
        if source_type:
            payload["source_type"] = source_type
        return self._send_command(payload, controller_policy_uuid=controller_policy_uuid)

    def _send_command(
        self,
        payload: Dict[str, Any],
        *,
        controller_policy_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a navigation command."""
        resolved_policy = controller_policy_uuid or self._controller_policy_uuid
        if resolved_policy:
            payload["controller_policy_uuid"] = resolved_policy

        api_client = self._twin.client.api.api_client

        _param = api_client.param_serialize(
            method="POST",
            resource_path=f"/api/v1/twins/{self.uuid}/navigation",
            body=payload,
            header_params={"Content-Type": "application/json"},
            auth_settings=["CustomTokenAuthentication"],
        )

        response_data = api_client.call_api(*_param)
        response_data.read()

        import json
        return json.loads(response_data.data.decode("utf-8"))

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> List[float]:
        """Convert yaw angle to quaternion [w, x, y, z]."""
        half = float(yaw) * 0.5
        return [math.cos(half), 0.0, 0.0, math.sin(half)]

    @staticmethod
    def _normalize_waypoints(waypoints: Iterable[Any] | NavigationPlan) -> List[Dict[str, Any]]:
        """Normalize waypoints to a list of dictionaries."""
        if isinstance(waypoints, NavigationPlan):
            return waypoints.waypoints
        builder = NavigationPlan()
        builder.extend(waypoints)
        return builder.waypoints


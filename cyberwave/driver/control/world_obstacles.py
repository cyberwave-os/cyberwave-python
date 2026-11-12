"""WorldObstaclesMixin -- runtime world collision geometry for in-process kinematics.

A robot description contains no desk, fixture or wall, so an in-process IK
solver or local planner happily routes the arm through a table it cannot see.
This mixin lets an operator (or an executor) describe that geometry at runtime
over the standard ``twin/command`` channel:

* ``add_obstacles`` -- replace the complete world scene (``geometries: []``
  removes all of it, including a default floor);
* ``remove_obstacle`` -- drop one obstacle by id.

Geometry is validated with :func:`cyberwave.driver.motion.scene.parse_scene_set`
(box/sphere/cylinder primitives, one planning frame, unit quaternions) and
persisted in the twin's top-level metadata under
:attr:`WorldObstaclesMixin.WORLD_OBSTACLES_METADATA_KEY`, so it survives a
restart: call :meth:`WorldObstaclesMixin.restore_world_obstacles_from_metadata`
from ``configure()`` (after the kinematics exist) and
:meth:`WorldObstaclesMixin.persist_default_world_obstacles_if_absent` from
``activate()``. Legacy keys (``pinocchio_obstacles``) are read for backward
compatibility; writes always use the new key.

## Replacement is atomic: build, validate, persist, then swap

A running solver is never mutated. A candidate kinematics model is built with
the new geometry and validated; only if that succeeds is the scene persisted,
and only if persisting succeeds is the candidate swapped in. Any failure leaves
the previous solver and scene untouched and reports through
``_on_arm_command_failed`` (``ArmCapabilityMixin``). Updates are refused while
motion is in flight (:meth:`WorldObstaclesMixin._world_obstacles_busy_detail`):
changing the collision scene under an executing trajectory would invalidate
the checks it was planned against. Drivers should also skip their command tick
while :attr:`WorldObstaclesMixin.world_obstacles_updating` is True.

## Seams

* :meth:`_build_world_obstacle_kinematics` / :meth:`_install_world_obstacle_kinematics`
  -- default: ``self._build_arm_kinematics(arm.config, obstacles=...)`` on the
  ``ArmCapabilityMixin`` controller, swapped in with ``ArmController.replace_kinematics``.
* :meth:`_default_world_obstacles` -- geometry in force before any command
  (e.g. a configured floor); default none.
* :meth:`_world_obstacles_busy_detail` -- why an update must wait; default none.
* :meth:`_on_world_obstacles_replaced` -- e.g. re-audit configured postures.

This does not update an external planner's scene (MoveIt's, for example);
that is ``scene_set``, a separate interface.
"""

from __future__ import annotations

import logging
from typing import Any

from ..interface import CallbackGroup, CommandArg, CommandArgs, TopicSpec
from ..interface.args import DriverOperationMode
from ..kinematics.arm.controller import validate_kinematics_candidate
from ..motion.scene import PLANNING_FRAME, Primitive, parse_scene_set
from ..support.utils import driver_logger

logger = logging.getLogger(__name__)

_ALL_OPERATION_MODES = frozenset(DriverOperationMode)


class WorldObstaclesMixin:
    """Runtime world obstacles: MQTT commands, twin-metadata persistence, atomic swap."""

    WORLD_OBSTACLES_METADATA_KEY: str = "world_obstacles"
    """Top-level twin metadata key the scene is written to."""

    LEGACY_WORLD_OBSTACLES_METADATA_KEYS: tuple[str, ...] = ("pinocchio_obstacles",)
    """Older keys still read (never written) when the current key is absent."""

    WORLD_OBSTACLES_FRAME: str = PLANNING_FRAME
    """The one frame obstacle poses must be expressed in."""

    use_world_obstacle_commands: bool = True
    """Register ``add_obstacles`` / ``remove_obstacle`` in ``define_interface``."""

    # ── state ────────────────────────────────────────────────────────────────

    def _init_world_obstacles(self) -> None:
        """Optional explicit init; every reader also tolerates its absence."""
        self._world_obstacles: tuple[Primitive, ...] | None = None
        self._world_obstacles_updating = False

    @property
    def world_obstacles_updating(self) -> bool:
        """True while a replacement is building; skip command ticks meanwhile."""
        return bool(getattr(self, "_world_obstacles_updating", False))

    def current_world_obstacles(self) -> tuple[Primitive, ...]:
        """The installed world scene, or the default one before any replacement."""
        current = getattr(self, "_world_obstacles", None)
        if current is not None:
            return current
        return tuple(self._default_world_obstacles())

    # ── command registration ─────────────────────────────────────────────────

    def define_interface(self, iface: Any) -> None:
        super().define_interface(iface)  # type: ignore[misc]
        if type(self).use_world_obstacle_commands:
            self._register_world_obstacle_commands(iface)

    def _register_world_obstacle_commands(self, iface: Any) -> None:
        frame = type(self).WORLD_OBSTACLES_FRAME
        cmd_topic = TopicSpec(
            namespace="twin", leaf="command", payload_schema_ref="TwinCommandPayload"
        )
        iface.add_listener(
            cmd_topic,
            CallbackGroup(callback=self._on_add_obstacles_command),
            command=CommandArgs(
                name="add_obstacles",
                args=(
                    CommandArg(
                        "geometries",
                        None,
                        "Complete world scene: [{id, primitive: box|sphere|cylinder, "
                        "dimensions (m), pose:{position:{x,y,z} m, "
                        "orientation:{x,y,z,w}}}]. Replaces any default geometry; "
                        "[] removes all world geometry.",
                    ),
                    CommandArg(
                        "frame_id",
                        frame,
                        f"All geometry poses must be in the {frame} frame",
                    ),
                ),
                description=(
                    "Replace the in-process world collision obstacles at runtime "
                    "and store them in twin metadata; self-collision is unchanged. "
                    "Does not update an external planner's scene."
                ),
            ),
            # Obstacles are configuration, not motion: settable in every mode.
            operation_modes=_ALL_OPERATION_MODES,
        )
        iface.add_listener(
            cmd_topic,
            CallbackGroup(callback=self._on_remove_obstacle_command),
            command=CommandArgs(
                name="remove_obstacle",
                args=(CommandArg("id", None, "ID of one active world obstacle"),),
                description=(
                    "Remove one in-process world obstacle by ID and update twin "
                    "metadata. Does not update an external planner's scene."
                ),
            ),
            operation_modes=_ALL_OPERATION_MODES,
        )

    # ── command handlers ─────────────────────────────────────────────────────

    def _on_add_obstacles_command(self, payload: dict[str, Any]) -> None:
        """Atomically replace the world scene after full validation."""
        data = payload.get("data")
        try:
            if not isinstance(data, dict):
                raise ValueError("add_obstacles data must be an object")  # noqa: TRY004 - reported as a command failure
            scene = parse_scene_set(
                {
                    "scene_revision": "world-obstacles-runtime",
                    "frame": data.get("frame_id", type(self).WORLD_OBSTACLES_FRAME),
                    "objects": data.get("geometries"),
                },
                planning_frame=type(self).WORLD_OBSTACLES_FRAME,
            )
            self.replace_world_obstacles(scene.objects)
        except Exception as exc:  # noqa: BLE001 - any failure is a refused command
            self._report_world_obstacles_failure("add_obstacles", str(exc))

    def _on_remove_obstacle_command(self, payload: dict[str, Any]) -> None:
        """Remove one id from the active world scene."""
        try:
            data = payload.get("data")
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("id"), str)
                or not data["id"]
            ):
                raise ValueError("remove_obstacle requires a non-empty id")
            current = self.current_world_obstacles()
            if not any(obstacle.id == data["id"] for obstacle in current):
                raise ValueError(f"unknown world obstacle id {data['id']!r}")
            self.replace_world_obstacles(
                tuple(obstacle for obstacle in current if obstacle.id != data["id"])
            )
        except Exception as exc:  # noqa: BLE001 - any failure is a refused command
            self._report_world_obstacles_failure("remove_obstacle", str(exc))

    def _report_world_obstacles_failure(self, command: str, reason: str) -> None:
        failed = getattr(self, "_on_arm_command_failed", None)
        if callable(failed):
            failed(command, reason)
        else:
            logger.warning("%s failed: %s", command, reason)

    # ── replacement ──────────────────────────────────────────────────────────

    def replace_world_obstacles(
        self, obstacles: tuple[Primitive, ...], *, persist: bool = True
    ) -> None:
        """Build then swap the in-process collision scene; never mutate a running solver.

        Raises ``ValueError`` when busy or when the candidate is invalid, and
        propagates build/persist errors -- in every failure case the previous
        solver and scene stay installed.
        """
        obstacles = tuple(obstacles)
        if self.world_obstacles_updating:
            raise ValueError("a motion or obstacle update is in progress")
        busy = self._world_obstacles_busy_detail()
        if busy is not None:
            raise ValueError(f"a motion or obstacle update is in progress ({busy})")
        self._world_obstacles_updating = True
        try:
            candidate = self._build_world_obstacle_kinematics(obstacles)
            if persist:
                self._persist_world_obstacles(obstacles)
            self._install_world_obstacle_kinematics(candidate)
            self._world_obstacles = obstacles
        finally:
            self._world_obstacles_updating = False
        self._on_world_obstacles_replaced(obstacles)
        self._world_obstacles_logger().info(
            f"World obstacle scene replaced: {len(obstacles)} obstacle(s)"
        )

    def _build_world_obstacle_kinematics(self, obstacles: tuple[Primitive, ...]) -> Any:
        """Build and validate a candidate kinematics model with *obstacles*.

        Default: the ``ArmCapabilityMixin`` controller's config through
        ``self._build_arm_kinematics(config, obstacles=obstacles)``. The
        candidate must keep a valid self-collision model when the installed one
        has it, and must actually pair any requested geometry with the arm
        (``obstacle_collision_available``) -- otherwise the scene would be
        silently ignored while reported as active.
        """
        arm = getattr(self, "_arm_ctrl", None)
        if arm is None:
            raise ValueError("the kinematics solver is not configured")
        candidate = self._build_arm_kinematics(arm.config, obstacles=obstacles)  # type: ignore[attr-defined]
        validate_kinematics_candidate(arm.loaded_kinematics, candidate, obstacles)
        return candidate

    def _install_world_obstacle_kinematics(self, candidate: Any) -> None:
        """Swap the validated candidate into the ``ArmCapabilityMixin`` controller."""
        self._arm_ctrl.replace_kinematics(candidate)  # type: ignore[attr-defined]

    def _default_world_obstacles(self) -> tuple[Primitive, ...]:
        """Geometry in force before any replacement (e.g. a configured floor)."""
        return ()

    def _world_obstacles_busy_detail(self) -> str | None:
        """Why an update must wait (motion in flight), or None. Default: never busy."""
        return None

    def _on_world_obstacles_replaced(self, obstacles: tuple[Primitive, ...]) -> None:
        """Hook after a successful swap (e.g. re-audit home/rest postures)."""
        del obstacles

    def _world_obstacles_logger(self) -> Any:
        return driver_logger(self, logger)

    # ── persistence ──────────────────────────────────────────────────────────

    def world_obstacles_metadata(
        self, obstacles: tuple[Primitive, ...]
    ) -> dict[str, Any]:
        """World geometry in the obstacle frame (metres, xyzw) for twin metadata.

        The same shape ``add_obstacles`` accepts, so a stored scene can be
        re-sent verbatim.
        """
        return {
            "frame_id": type(self).WORLD_OBSTACLES_FRAME,
            "geometries": [
                {
                    "id": obstacle.id,
                    "primitive": obstacle.kind,
                    "dimensions": list(obstacle.dimensions),
                    "pose": {
                        "position": dict(zip(("x", "y", "z"), obstacle.pose.position)),
                        "orientation": dict(
                            zip(("x", "y", "z", "w"), obstacle.pose.orientation)
                        ),
                    },
                }
                for obstacle in obstacles
            ],
        }

    def _persist_world_obstacles(self, obstacles: tuple[Primitive, ...]) -> None:
        """Write the scene to the twin's top-level metadata (shallow-merged server-side)."""
        self._cw.twins.update(  # type: ignore[attr-defined]
            str(self.twin_uuid),  # type: ignore[attr-defined]
            metadata={
                type(self).WORLD_OBSTACLES_METADATA_KEY: self.world_obstacles_metadata(
                    obstacles
                )
            },
        )

    def _twin_metadata(self) -> dict[str, Any] | None:
        twin = getattr(self, "_twin", None)
        metadata = (
            twin.get("metadata")
            if isinstance(twin, dict)
            else getattr(twin, "metadata", None)
        )
        return metadata if isinstance(metadata, dict) else None

    def _saved_world_obstacles_key(self, metadata: dict[str, Any] | None) -> str | None:
        if metadata is None:
            return None
        for key in (
            type(self).WORLD_OBSTACLES_METADATA_KEY,
            *type(self).LEGACY_WORLD_OBSTACLES_METADATA_KEYS,
        ):
            if key in metadata:
                return key
        return None

    def restore_world_obstacles_from_metadata(self) -> bool:
        """Restore the saved scene (no write-back). False when nothing is saved.

        Call from ``configure()`` once the kinematics exist, before activation.
        A saved scene that does not validate or build RAISES: starting with a
        silently different collision scene than the one the operator stored is
        worse than not starting.
        """
        metadata = self._twin_metadata()
        key = self._saved_world_obstacles_key(metadata)
        if key is None or metadata is None:
            return False
        saved = metadata[key]
        if not isinstance(saved, dict):
            raise ValueError(f"twin.metadata.{key} must be an object")  # noqa: TRY004 - same contract as parse_scene_set
        scene = parse_scene_set(
            {
                "scene_revision": "world-obstacles-persisted",
                "frame": saved.get("frame_id"),
                "objects": saved.get("geometries"),
            },
            planning_frame=type(self).WORLD_OBSTACLES_FRAME,
        )
        self.replace_world_obstacles(scene.objects, persist=False)
        return True

    def persist_default_world_obstacles_if_absent(self) -> None:
        """Record the default scene when no saved scene exists (call from ``activate``).

        Makes the geometry actually in force visible (and removable) from the
        twin. Errors are logged, never raised: activation must not fail on a
        metadata write.
        """
        if getattr(self, "_cw", None) is None or getattr(self, "_twin", None) is None:
            return
        if self._saved_world_obstacles_key(self._twin_metadata()) is not None:
            return
        try:
            self._persist_world_obstacles(self.current_world_obstacles())
        except Exception as exc:  # noqa: BLE001 - activation must not fail on a metadata write
            self._world_obstacles_logger().error(
                f"Could not update twin world obstacles metadata: {exc}"
            )


__all__ = ["WorldObstaclesMixin"]

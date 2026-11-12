"""Collision-aware, multi-seed IK on top of :class:`BaseKinematicsManipulator`.

The base solver answers "which joints put the TCP here" from one seed and never
asks whether those joints put the arm through itself, the table, or a box. This
subclass adds the robot-agnostic policy every arm driver needs before it hands a
solution to hardware:

* **Joint limit overrides**, intersected with the description and never wider.
* **Self-collision** from a reviewed SRDF (``removeCollisionPairs``), with a
  check that the SRDF actually describes the loaded model, and an optional
  driver-supplied development fallback pairing.
* **World collision**: an optional floor slab, or a runtime scene of
  :class:`~cyberwave.driver.motion.scene.Primitive` obstacles.
* **An attached payload**: one primitive carried on the tool frame, checked
  against the world and against every arm link except the ones holding it.
* **Multi-seed deterministic IK** (caller's start, zero, seeded samples), with
  ``restart=False`` for path continuity, joint unwrapping toward the start, and
  diagnostics that say *which* guard refused a target.

pinocchio stays the forward-kinematics authority and stays lazily imported (via
the base class); ``coal``/``hppfcl`` are imported only when primitives are built.
A driver typically subclasses this only to map its own named limit profiles onto
``joint_limit_overrides`` and to supply its fallback pairing.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from cyberwave._geometry import core as _geometry_core
from cyberwave.driver.motion.scene import Pose, Primitive

from .base import BaseKinematicsManipulator
from .config import ArmKinematicsConfig

logger = logging.getLogger(__name__)

#: The floor slab's thickness (m). Only the TOP face is the working surface; the
#: slab is given depth so a link cannot tunnel through an infinitely thin plane
#: between two collision samples.
#: Collision geometry as loaded from disk, keyed by description. Scene and
#: payload rebuilds copy it instead of re-reading and re-meshing every link
#: (hundreds of ms per command on a meshed arm). Meshes edited on disk while the
#: driver runs are not picked up, as the URDF itself is read once.
_LOADED_GEOMETRY: dict[tuple[str, tuple[str, ...], tuple[str, ...]], Any] = {}

FLOOR_THICKNESS_M = 0.02

#: Default half-extent of the slab in x and y (m). Large enough that a typical
#: tabletop arm cannot reach around its edge; raise it for longer-reach arms.
FLOOR_HALF_EXTENT_M = 2.0

#: Where the active self-collision pairs came from. Only ``srdf`` is production.
COLLISION_PAIR_SOURCE_SRDF = "srdf"
#: The SRDF loaded, but it names links this description does not have -- so the
#: matrix does not describe the model being checked. Not valid for production.
COLLISION_PAIR_SOURCE_SRDF_MISMATCH = "srdf_link_mismatch"
#: A driver-supplied development pairing; never reported as a valid model.
COLLISION_PAIR_SOURCE_HEURISTIC = "heuristic"
COLLISION_PAIR_SOURCE_NONE = "none"

#: ``geometry_model -> [(i, j), ...]`` -- geometry index pairs to check when no
#: SRDF is configured.
FallbackPairs = Callable[[Any], Sequence[tuple[int, int]]]


class CollisionModelUnavailable(RuntimeError):
    """Raised when a reviewed collision model was required and is not available.

    The point of a required collision model is that "we could not check" and
    "we checked and it was clear" stop being the same observable outcome.
    """


def mesh_package_dirs(urdf_path: str) -> list[str]:
    """Directories in which to resolve the URDF's ``package://`` mesh URIs.

    pinocchio needs a directory *containing* the ROS package directory, so for
    ``<...>/src/<pkg>/urdf/robot.urdf`` the answer is ``<...>/src``. Every
    ancestor is offered rather than only that one because the URDF may be
    installed at a different depth (a ROS ``share/`` tree, a Docker image
    layout, or a loose copy); pinocchio tries each in turn and only fails if
    none resolves.
    """
    ancestors: list[str] = []
    current = os.path.dirname(os.path.abspath(urdf_path))
    while current and current != os.path.dirname(current):
        ancestors.append(current)
        current = os.path.dirname(current)
    return ancestors


def split_by_parent_joint(
    *, distal_joint_min: int, proximal_joint_max: int
) -> FallbackPairs:
    """A development pairing: distal geometry vs proximal geometry.

    Pairs every geometry whose parent joint index is ``>= distal_joint_min``
    with every geometry whose parent joint index is ``<= proximal_joint_max``.
    ``addAllCollisionPairs()`` alone reports a collision in every configuration
    (adjacent links always touch), and deriving the split from parent joints
    rather than literal geometry indices keeps it independent of geometry
    ordering. Unreviewed by construction, so the solver reports it as
    ``heuristic`` and never as a valid collision model.
    """

    def _pairs(geom: Any) -> list[tuple[int, int]]:
        distal = [
            i
            for i, obj in enumerate(geom.geometryObjects)
            if obj.parentJoint >= distal_joint_min
        ]
        proximal = [
            i
            for i, obj in enumerate(geom.geometryObjects)
            if obj.parentJoint <= proximal_joint_max
        ]
        return [(i, j) for i in distal for j in proximal]

    return _pairs


def floor_slab_primitive(
    floor_z: float,
    *,
    half_extent_m: float = FLOOR_HALF_EXTENT_M,
    thickness_m: float = FLOOR_THICKNESS_M,
    identifier: str = "floor",
) -> Primitive:
    """The floor slab as a world-scene box whose TOP face sits on *floor_z*.

    Lets a driver report (and persist) its startup floor as an ordinary
    obstacle, replaceable through the same scene path as any other, with exactly
    the geometry the floor gate checks -- rather than re-deriving the slab
    placement and drifting from it.
    """
    return Primitive(
        identifier,
        "box",
        (2.0 * float(half_extent_m), 2.0 * float(half_extent_m), float(thickness_m)),
        Pose(
            (0.0, 0.0, float(floor_z) - float(thickness_m) / 2.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


class CollisionAwareKinematics(BaseKinematicsManipulator):
    """BaseKinematicsManipulator with collision gating and multi-seed IK policy."""

    #: Seeds tried per IK call, including the caller's own start configuration.
    #: Early exit means the average cost is far below N x a single solve, because
    #: most targets succeed on the first seed. 1 reproduces the base solver.
    IK_SEED_ATTEMPTS: int = 8
    #: Iteration cap for seeds 3..N (the sampled ones). A sampled seed that has
    #: not converged in a few hundred iterations rarely converges at all, so a
    #: pose nobody can reach -- the common case in a ranked grasp search -- fails
    #: several times faster. The caller's start and the zero configuration keep
    #: the full ``ik_max_iter`` budget.
    IK_RANDOM_SEED_MAX_ITER: int = 200
    #: Fixed RNG seed for the sampled IK seeds: the same request always produces
    #: the same answer, because a solver whose result depends on call history
    #: cannot be debugged from a log.
    IK_SEED_RNG: int = 0xC1BE

    #: Name of the floor/table geometry injected into the collision model.
    #: Prefixed so it can never be confused with a link the description declares.
    FLOOR_GEOM_NAME: str = "cw_floor"
    #: Prefix of runtime obstacle geometry names (followed by the primitive id).
    OBSTACLE_GEOM_PREFIX: str = "world_"
    #: Prefix of the attached payload's geometry name (followed by its id).
    PAYLOAD_GEOM_PREFIX: str = "payload_"

    #: Why :meth:`ik` returned ``None``. Deliberately not the word "unreachable":
    #: nothing here tests reachability, and a DLS walk that fails to converge
    #: from the seeds offered is a statement about the search, not about the arm.
    IK_NO_CONVERGENCE = "no IK solution from the offered seeds"
    #: Prefix of the refusal message; the scenes that refused follow it. Exposed
    #: so callers can match on the category without matching whole sentences.
    IK_REFUSED_PREFIX = "every converged IK solution was refused: "

    def __init__(
        self,
        config: ArmKinematicsConfig,
        *,
        joint_limit_overrides: Mapping[str, tuple[float, float]] | None = None,
        reject_self_collision: bool = True,
        srdf_path: str = "",
        require_collision_model: bool = False,
        fallback_pairs: FallbackPairs | None = None,
        reject_floor_collision: bool = False,
        floor_z: float = 0.0,
        floor_half_extent_m: float = FLOOR_HALF_EXTENT_M,
        floor_thickness_m: float = FLOOR_THICKNESS_M,
        obstacles: tuple[Primitive, ...] | None = None,
        attached_payload: Primitive | None = None,
        payload_touch_links: Sequence[str] = (),
        package_dirs: Sequence[str] | None = None,
    ) -> None:
        """Build the model, narrow its limits, then build the collision gates.

        * ``joint_limit_overrides``: per-joint ``(lower, upper)`` rad, intersected
          with the description (never widened). ``None`` keeps the URDF limits.
        * ``srdf_path``: reviewed SRDF; the only pairing reported as valid.
        * ``require_collision_model``: raise :class:`CollisionModelUnavailable`
          unless a valid SRDF model was built -- for planned hardware execution.
        * ``fallback_pairs``: development pairing used when ``srdf_path`` is
          empty (see :func:`split_by_parent_joint`); ``None`` means no
          self-collision gate without an SRDF.
        * ``reject_floor_collision`` / ``floor_z``: a slab whose TOP face sits at
          ``floor_z`` metres in the base frame. Ignored when ``obstacles`` is
          given: a runtime scene replaces the startup floor entirely.
        * ``obstacles``: runtime world scene in the base frame; ``()`` loads
          geometry with no world objects (an explicitly empty scene).
        * ``attached_payload``: one primitive whose pose is relative to the TCP
          (``ee_frame`` plus ``tool_offset``), carried rigidly with the tool.
          It is paired with the world (floor and obstacles) and with every arm
          geometry except the geometry rigidly attached to the same joint (the
          gripper holding it) and the links named in ``payload_touch_links``.
          Raises :class:`CollisionModelUnavailable` when it cannot be built:
          reporting a payload as carried while checking nothing is the one
          answer this must never give.
        * ``package_dirs``: where to resolve ``package://`` meshes; defaults to
          every ancestor of the URDF (:func:`mesh_package_dirs`).
        """
        super().__init__(config)
        self._apply_joint_limit_overrides(joint_limit_overrides)
        self._reject_self_collision = bool(reject_self_collision)
        self._srdf_path = str(srdf_path or "")
        self._require_collision_model = bool(require_collision_model)
        self._fallback_pairs = fallback_pairs
        self._package_dirs = None if package_dirs is None else list(package_dirs)
        self._geom_model: Any = None
        self._geom_data: Any = None
        self._collision_pair_source = COLLISION_PAIR_SOURCE_NONE
        self._reject_floor_collision = bool(reject_floor_collision)
        #: Metres, in the robot's own base frame, of the working surface's TOP
        #: face. 0.0 is the arm's mounting plane.
        self._floor_z = float(floor_z)
        self._floor_half_extent_m = float(floor_half_extent_m)
        self._floor_thickness_m = float(floor_thickness_m)
        self._floor_geom_id: int | None = None
        #: Indices into ``geom_model.collisionPairs`` that involve the floor.
        #: Kept so self-collision and floor collision stay separately reportable:
        #: "the wrist hit the shoulder" and "the gripper hit the table" have
        #: different remedies, and one method answering both would name neither.
        self._floor_pair_indices: frozenset[int] = frozenset()
        self._obstacle_pairs: dict[int, str] = {}
        #: Pair indices between the attached payload and the arm itself.
        self._payload_pairs: frozenset[int] = frozenset()
        self._payload_id: str | None = None
        self._payload_touch_links = tuple(payload_touch_links)
        self._last_ik_refusal: str | None = None
        if self._require_collision_model and not self._srdf_path:
            raise CollisionModelUnavailable(
                "a reviewed SRDF is required for planned hardware execution, and "
                "srdf_path is empty. A fallback pairing is a development aid: it "
                "was never reviewed against this description and does not know "
                "about payload geometry."
            )
        # Geometry is loaded for EITHER gate. Nesting the floor inside the
        # self-collision branch would make `reject_self_collision=False` with
        # `reject_floor_collision=True` silently build no floor gate at all -- a
        # config an operator has every reason to read as "check the table, skip
        # the arm".
        if (
            self._reject_self_collision
            or self._reject_floor_collision
            or obstacles is not None
            or attached_payload is not None
        ):
            geom = self._load_collision_geometry(config.urdf_path)
            if attached_payload is not None and geom is None:
                raise CollisionModelUnavailable(
                    f"{self._label} collision geometry is unavailable, so the "
                    f"payload {attached_payload.id!r} cannot be carried"
                )
            if geom is not None:
                if self._reject_self_collision:
                    self._pair_self_collision(geom)
                # Before the world, so the floor/obstacle pairing below pairs
                # the payload with the world like any other moving geometry.
                if attached_payload is not None:
                    self._build_payload_collision(attached_payload)
                # AFTER the self-collision pairing, never before: the reviewed
                # SRDF matrix is defined over the description's own links, and
                # adding a geometry before addAllCollisionPairs() would inject
                # world pairs the SRDF has no opinion about into the middle of a
                # reviewed matrix.
                if obstacles is not None:
                    self._build_obstacle_collision(obstacles)
                elif self._reject_floor_collision:
                    self._build_floor_collision()
        if self._require_collision_model and not self.collision_model_valid:
            raise CollisionModelUnavailable(
                "the reviewed SRDF collision model could not be built from "
                f"{self._srdf_path!r} and {config.urdf_path!r}; refusing to "
                "report 'no collision' when nothing was actually checked"
            )

    @property
    def _label(self) -> str:
        """Log prefix: the concrete class, so a subclass's logs name the robot."""
        return type(self).__name__

    # ── availability ─────────────────────────────────────────────────────────

    @property
    def collision_pair_source(self) -> str:
        """``srdf``, ``srdf_link_mismatch``, ``heuristic`` or ``none``."""
        return self._collision_pair_source

    @property
    def collision_model_valid(self) -> bool:
        """True only for a reviewed SRDF matrix that actually loaded.

        The fallback deliberately reports False. It is a development aid, and a
        development aid that reports itself as valid is how an unreviewed model
        ends up gating hardware.
        """
        return (
            self._collision_pair_source == COLLISION_PAIR_SOURCE_SRDF
            and self._geom_model is not None
        )

    @property
    def self_collision_available(self) -> bool:
        """True when the gate is live: rejection is on *and* pairs were formed.

        Distinct from :meth:`in_self_collision` returning ``False``, which means
        "not colliding **or** unknown". Callers that log or report a safety state
        must read this, not infer availability from a negative answer.

        Requires a live pair source, not just loaded geometry: a floor-only
        configuration loads the same geometry and forms no self pairs, and would
        otherwise report a gate that checks nothing as available.
        """
        return (
            self._geom_model is not None
            and self._geom_data is not None
            and self._collision_pair_source != COLLISION_PAIR_SOURCE_NONE
        )

    @property
    def floor_collision_available(self) -> bool:
        """True when the floor gate is live: requested, geometry built, pairs formed.

        Same distinction as :attr:`self_collision_available` -- a negative answer
        from :meth:`in_floor_collision` means "clear **or** never checked".
        Deliberately independent of :attr:`self_collision_available`.
        """
        return (
            bool(self._floor_pair_indices)
            and self._geom_model is not None
            and self._geom_data is not None
        )

    @property
    def obstacle_collision_available(self) -> bool:
        """True when at least one runtime obstacle is paired with the arm."""
        return bool(self._obstacle_pairs) and self._geom_data is not None

    @property
    def attached_payload_id(self) -> str | None:
        """The id of the payload carried in the collision model, or ``None``."""
        return self._payload_id

    @property
    def floor_z(self) -> float:
        """Metres, robot base frame: the working surface's top face."""
        return self._floor_z

    def floor_primitive(self, identifier: str = "floor") -> Primitive | None:
        """The live floor slab as a scene primitive; ``None`` when the gate is off.

        Feeding this back as ``obstacles=(kin.floor_primitive(),)`` reproduces the
        same geometry, so a driver can expose its configured floor as the first
        entry of a replaceable world scene.
        """
        if not self.floor_collision_available:
            return None
        return floor_slab_primitive(
            self._floor_z,
            half_extent_m=self._floor_half_extent_m,
            thickness_m=self._floor_thickness_m,
            identifier=identifier,
        )

    # ── joint limits ─────────────────────────────────────────────────────────

    def _apply_joint_limit_overrides(
        self, overrides: Mapping[str, tuple[float, float]] | None
    ) -> None:
        """Intersect the model's limits with *overrides*, never widening them.

        pinocchio's limit vectors are writable, so narrowing them here makes
        both ``joint_limits()`` and the clamp inside ``ik()`` honour the
        override with no method overrides. Intersecting rather than assigning is
        deliberate: an override must never authorise motion the model says is
        impossible.
        """
        if not overrides:
            return
        for name, (lower, upper) in overrides.items():
            idx = self._arm_idx_q.get(name)
            if idx is None:
                continue
            model_lo = float(self._model.lowerPositionLimit[idx])
            model_hi = float(self._model.upperPositionLimit[idx])
            new_lo, new_hi = max(model_lo, float(lower)), min(model_hi, float(upper))
            if new_lo >= new_hi:
                raise ValueError(
                    f"joint limit override leaves {name} with an empty range: "
                    f"model ({model_lo:.4f}, {model_hi:.4f}) vs override "
                    f"({lower:.4f}, {upper:.4f})"
                )
            self._model.lowerPositionLimit[idx] = new_lo
            self._model.upperPositionLimit[idx] = new_hi

    # ── collision model construction ─────────────────────────────────────────

    def _load_collision_geometry(self, urdf_path: str) -> Any:
        """The URDF's collision geometry, or ``None`` when it cannot be loaded.

        Off-robot the meshes referenced by a URDF are often absent, so this is
        best-effort: on failure every gate degrades to "never collides" and logs
        once, rather than making IK unusable in unit tests. Read the
        ``*_available`` properties to tell the two apart, or set
        ``require_collision_model`` to fail closed.
        """
        if not urdf_path:
            return None
        pin = self._pin
        package_dirs = (
            self._package_dirs
            if self._package_dirs is not None
            else mesh_package_dirs(urdf_path)
        )
        key = (urdf_path, tuple(self._config.locked_joints), tuple(package_dirs))
        try:
            loaded = _LOADED_GEOMETRY.get(key)
            if loaded is None:
                loaded = pin.buildGeomFromUrdf(
                    self._model,
                    urdf_path,
                    pin.GeometryType.COLLISION,
                    package_dirs=package_dirs,
                )
                _LOADED_GEOMETRY[key] = loaded
            # A copy: the pairs and world objects added below must not reach the
            # cached model, while the meshes and their BVHs are shared.
            geom = loaded.copy()
        except Exception as exc:  # noqa: BLE001 - degrade, and say so
            logger.warning(
                "%s collision geometry unavailable (%s); neither the "
                "self-collision nor the world gate will reject anything",
                self._label,
                exc,
            )
            return None
        # Assigned up front so a floor-only configuration has a usable model even
        # though no self pairs are ever formed. _pair_self_collision drops both
        # back to None if the pairing itself fails.
        self._geom_model = geom
        self._geom_data = pin.GeometryData(geom)
        return geom

    def _pair_self_collision(self, geom: Any) -> None:
        """Form the self-collision pairs on *geom*.

        Leaves :attr:`collision_pair_source` at ``none`` when no pairing could be
        formed, which is what :attr:`self_collision_available` reports on.
        """
        try:
            pin = self._pin
            if self._srdf_path:
                # The official path: every pair, minus the reviewed exclusions.
                # pinocchio owns the SRDF parsing -- a second parser in
                # production code is a second thing to keep in step with every
                # other consumer of the same SRDF.
                geom.addAllCollisionPairs()
                before = len(geom.collisionPairs)
                pin.removeCollisionPairs(self._model, geom, self._srdf_path)
                # Rebuilt because collisionResults is sized by the pair list.
                self._geom_data = pin.GeometryData(geom)

                # Does this matrix actually describe THIS description?
                #
                # removeCollisionPairs silently ignores exclusions naming links
                # the model does not have. Point a payload SRDF at the bare
                # vendor URDF and it loads cleanly while the payload -- the
                # entire point of that SRDF -- is absent from the model being
                # checked. Reporting that as a valid collision model is worse
                # than reporting none: it is a confident answer about geometry
                # nobody looked at.
                missing = self._srdf_links_absent_from_model()
                if missing:
                    self._collision_pair_source = COLLISION_PAIR_SOURCE_SRDF_MISMATCH
                    logger.error(
                        "%s SRDF %s names links absent from the loaded "
                        "description (%s). The collision matrix does not describe "
                        "this robot; self-collision results EXCLUDE that geometry. "
                        "Load the description the SRDF was derived from, or use "
                        "the matching SRDF.",
                        self._label,
                        self._srdf_path,
                        ", ".join(sorted(missing)),
                    )
                    return

                self._collision_pair_source = COLLISION_PAIR_SOURCE_SRDF
                logger.info(
                    "%s self-collision from reviewed SRDF %s "
                    "(%d geometries, %d pairs after removing %d exclusions)",
                    self._label,
                    self._srdf_path,
                    geom.ngeoms,
                    len(geom.collisionPairs),
                    before - len(geom.collisionPairs),
                )
                return

            pairs = (
                list(self._fallback_pairs(geom))
                if self._fallback_pairs is not None
                else []
            )
            for first, second in pairs:
                geom.addCollisionPair(pin.CollisionPair(first, second))
            if not pairs:
                logger.warning(
                    "%s has no SRDF and no fallback collision pairs; "
                    "self-collision rejection is inactive",
                    self._label,
                )
                return
            self._geom_data = pin.GeometryData(geom)
            self._collision_pair_source = COLLISION_PAIR_SOURCE_HEURISTIC
            logger.warning(
                "%s self-collision using a HEURISTIC fallback pairing "
                "(%d geometries, %d pairs). Development only: unreviewed, and "
                "blind to payload geometry. Planned hardware execution requires "
                "a reviewed SRDF.",
                self._label,
                geom.ngeoms,
                len(pairs),
            )
        except Exception as exc:  # noqa: BLE001 - degrade both gates together
            # Both gates go down together: the pair list may be half-built, and a
            # world gate riding on a model we failed to finish is not a gate.
            logger.warning(
                "%s self-collision checking unavailable (%s); neither IK nor "
                "joint commands will reject colliding configurations",
                self._label,
                exc,
            )
            self._geom_model = None
            self._geom_data = None

    def _collision_shapes(self) -> Any:
        """The primitive-shape module: ``coal`` on current pinocchio, else hppfcl.

        pinocchio renamed hppfcl to coal. Try the current name first so the
        fallback is the legacy path, not the other way round.
        """
        try:
            import coal

            return coal
        except ImportError:  # pragma: no cover - depends on the pinocchio build
            import hppfcl

            return hppfcl

    def _pair_with_world(self, world_id: int) -> list[int]:
        """Pair every movable geometry with geometry *world_id*; the pair indices.

        Pairing rule: every geometry EXCEPT those rigidly attached to the model
        root (``parentJoint == 0``). Structural rather than a name match, so it
        holds for any floor height and for a description that renames its links.
        A base bolted to the surface intersects it in every configuration and
        would make the gate refuse everything. The gripper is deliberately IN:
        it is the part an operator drives into the table.
        """
        pin = self._pin
        geom = self._geom_model
        indices: list[int] = []
        for index in range(world_id):
            if geom.geometryObjects[index].parentJoint == 0:
                continue
            geom.addCollisionPair(pin.CollisionPair(index, world_id))
            indices.append(len(geom.collisionPairs) - 1)
        return indices

    def _build_floor_collision(self) -> None:
        """Add the working surface as a collision object and pair the arm to it.

        One plane, supplied by configuration, and nothing else: desk edges,
        boxes and fixtures remain unmodelled unless sent as runtime obstacles.
        """
        if self._geom_model is None or self._geom_data is None:
            logger.warning(
                "%s floor collision requested but no collision geometry "
                "loaded; the floor gate is INACTIVE",
                self._label,
            )
            return
        pin = self._pin
        try:
            fcl = self._collision_shapes()
            geom = self._geom_model
            placement = pin.SE3.Identity()
            # Position the slab so its TOP face sits exactly on floor_z; a slab
            # centred on floor_z would put the working surface half a thickness
            # too high and silently steal travel.
            placement.translation = np.array(
                [0.0, 0.0, self._floor_z - self._floor_thickness_m / 2.0]
            )
            slab = fcl.Box(
                2.0 * self._floor_half_extent_m,
                2.0 * self._floor_half_extent_m,
                self._floor_thickness_m,
            )
            floor_id = geom.addGeometryObject(
                pin.GeometryObject(self.FLOOR_GEOM_NAME, 0, placement, slab)
            )
            floor_pairs = self._pair_with_world(floor_id)
            if not floor_pairs:
                logger.warning(
                    "%s floor collision: no geometry could be paired with the "
                    "floor; the gate is INACTIVE",
                    self._label,
                )
                geom.removeGeometryObject(self.FLOOR_GEOM_NAME)
                return
            self._floor_geom_id = floor_id
            self._floor_pair_indices = frozenset(floor_pairs)
            # GeometryData caches placements per geometry, so it must be rebuilt
            # after the model grew -- reusing the old one silently checks the
            # pairs against a stale, shorter placement table.
            self._geom_data = pin.GeometryData(geom)
            logger.info(
                "%s floor collision active: surface top z=%.4f m in the base "
                "frame, %d pair(s) (%s)",
                self._label,
                self._floor_z,
                len(floor_pairs),
                ", ".join(
                    geom.geometryObjects[
                        geom.collisionPairs[i].first
                        if geom.collisionPairs[i].second == floor_id
                        else geom.collisionPairs[i].second
                    ].name
                    for i in floor_pairs
                ),
            )
        except Exception as exc:  # noqa: BLE001 - degrade, and say so
            logger.warning(
                "%s floor collision unavailable (%s); IK will not reject "
                "solutions that reach the surface",
                self._label,
                exc,
            )
            self._floor_geom_id = None
            self._floor_pair_indices = frozenset()

    def _primitive_shape(self, fcl: Any, obstacle: Primitive) -> Any:
        if obstacle.kind == "box":
            return fcl.Box(*obstacle.dimensions)
        if obstacle.kind == "sphere":
            return fcl.Sphere(obstacle.dimensions[0])
        # shape_msgs uses [height, radius]; FCL takes (radius, height).
        return fcl.Cylinder(obstacle.dimensions[1], obstacle.dimensions[0])

    def _placement(self, pose: Pose) -> Any:
        """``pin.SE3`` of a named xyzw :class:`Pose`.

        The SDK holds no rotation maths of its own: the shared geometry core
        turns the (x, y, z, w) orientation into the row-major 3x3 matrix.
        """
        geometry = _geometry_core()
        rotation = geometry.quat.to_matrix(
            geometry.Quaternion.from_xyzw(pose.orientation)
        )
        return self._pin.SE3(
            np.asarray(rotation, dtype=float).reshape(3, 3),
            np.array(pose.position, dtype=float),
        )

    def _build_payload_collision(self, payload: Primitive) -> None:
        """Carry *payload* on the TCP: a geometry fixed to the tool's joint.

        Placed at ``ee_frame placement * tool_offset * payload.pose`` in the
        parent joint's frame, so it moves exactly as the TCP does. Paired with
        every geometry already in the model except the ones rigidly attached to
        the same joint -- in a reduced model that is the wrist link, the tool and
        every locked finger: the hand holding the object touches it by
        definition -- and any link named in ``payload_touch_links``. The world
        is paired afterwards by :meth:`_pair_with_world`, like any moving part.
        """
        geom = self._geom_model
        if geom is None or self._geom_data is None:
            raise CollisionModelUnavailable(
                f"{self._label} collision geometry is unavailable, so the "
                f"payload {payload.id!r} cannot be carried"
            )
        pin, fcl = self._pin, self._collision_shapes()
        frame = self._model.frames[self._frame_id]
        parent_joint = int(frame.parentJoint)
        placement = frame.placement
        if self._tool_offset is not None:
            placement = placement * pin.SE3(np.asarray(self._tool_offset, dtype=float))
        placement = placement * self._placement(payload.pose)
        touch_frames = {
            self._model.getFrameId(name)
            for name in self._payload_touch_links
            if self._model.existFrame(name)
        }
        existing = len(geom.geometryObjects)
        payload_id = geom.addGeometryObject(
            pin.GeometryObject(
                f"{self.PAYLOAD_GEOM_PREFIX}{payload.id}",
                parent_joint,
                int(self._frame_id),
                placement,
                self._primitive_shape(fcl, payload),
            )
        )
        pairs: list[int] = []
        for index in range(existing):
            other = geom.geometryObjects[index]
            if other.parentJoint == parent_joint or other.parentFrame in touch_frames:
                continue
            geom.addCollisionPair(pin.CollisionPair(index, payload_id))
            pairs.append(len(geom.collisionPairs) - 1)
        self._payload_pairs = frozenset(pairs)
        self._payload_id = payload.id
        self._geom_data = pin.GeometryData(geom)
        logger.info(
            "%s carrying payload %r on the tool frame: %d arm pair(s)",
            self._label,
            payload.id,
            len(pairs),
        )

    def _build_obstacle_collision(self, obstacles: tuple[Primitive, ...]) -> None:
        """Add a complete runtime world scene in the base frame, after SRDF pairing."""
        geom = self._geom_model
        if geom is None:
            raise CollisionModelUnavailable(
                f"{self._label} collision geometry is unavailable"
            )
        pin, fcl = self._pin, self._collision_shapes()
        pairs: dict[int, str] = {}
        for obstacle in obstacles:
            shape = self._primitive_shape(fcl, obstacle)
            placement = self._placement(obstacle.pose)
            obstacle_id = geom.addGeometryObject(
                pin.GeometryObject(
                    f"{self.OBSTACLE_GEOM_PREFIX}{obstacle.id}", 0, placement, shape
                )
            )
            for index in self._pair_with_world(obstacle_id):
                pairs[index] = obstacle.id
        if obstacles and not pairs:
            raise CollisionModelUnavailable(
                "no robot geometry could be paired with obstacles"
            )
        self._obstacle_pairs = pairs
        self._geom_data = pin.GeometryData(geom)
        logger.info(
            "%s world scene active: %d obstacle(s), %d pair(s)",
            self._label,
            len(obstacles),
            len(pairs),
        )

    def _srdf_links_absent_from_model(self) -> set[str]:
        """Link names the SRDF excludes that this description does not define.

        A deliberately minimal read of ``disable_collisions`` attributes. This is
        a cross-check, not a second pair extractor -- pinocchio still owns the
        removal itself, so there is no duplicated semantics to keep in step, only
        the question "are these the links we loaded?".
        """
        try:
            root = ET.parse(self._srdf_path).getroot()
        except Exception as exc:  # noqa: BLE001 - unreadable SRDF is reported
            logger.warning("could not read SRDF %s: %s", self._srdf_path, exc)
            return set()

        referenced = {
            name
            for entry in root.findall("disable_collisions")
            for name in (entry.get("link1"), entry.get("link2"))
            if name
        }
        known = {frame.name for frame in self._model.frames}
        return referenced - known

    # ── collision queries ────────────────────────────────────────────────────

    def _colliding_pairs(self, positions: dict[str, float]) -> set[int]:
        """Indices of every colliding pair at *positions*; empty if unavailable.

        Evaluates ALL pairs rather than stopping at the first hit, because the
        callers need to know *which* scene was touched -- stopping early would
        report whichever pair happened to be ordered first.
        """
        if self._geom_model is None or self._geom_data is None:
            return set()
        pin = self._pin
        q = self._q_from_dict(positions)
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateGeometryPlacements(
            self._model, self._data, self._geom_model, self._geom_data
        )
        pin.computeCollisions(self._geom_model, self._geom_data, False)
        return {
            index
            for index in range(len(self._geom_model.collisionPairs))
            if self._geom_data.collisionResults[index].isCollision()
        }

    def in_self_collision(self, positions: dict[str, float]) -> bool:
        """True when *positions* puts the arm in collision **with itself**.

        Deliberately excludes the world pairs: a method named for self-collision
        that also answered for the table would make every "no self-collision" log
        line ambiguous about what was checked. Use :meth:`collision_reason` for
        world contact.

        False when collision geometry is unavailable -- "unknown, not cleared";
        :attr:`self_collision_available` distinguishes the two.
        """
        return bool(
            self._colliding_pairs(positions)
            - self._floor_pair_indices
            - self._obstacle_pairs.keys()
            - self._payload_pairs
        )

    def in_floor_collision(self, positions: dict[str, float]) -> bool:
        """True when *positions* puts any movable link through the floor slab.

        False when the floor gate is inactive -- :attr:`floor_collision_available`
        is what distinguishes "clear" from "not checked".
        """
        return bool(self._colliding_pairs(positions) & self._floor_pair_indices)

    def in_collision(self, positions: dict[str, float]) -> bool:
        """True for self-collision or active world contact. What ``ik()`` gates on."""
        return bool(self._colliding_pairs(positions))

    def collision_reason(self, positions: dict[str, float]) -> str | None:
        """Which scene *positions* touches, for an error a caller can act on.

        ``None`` means nothing checked collided -- which is not the same as safe;
        see the ``*_available`` properties.
        """
        hits = self._colliding_pairs(positions)
        if not hits:
            return None
        scenes = []
        if (
            hits
            - self._floor_pair_indices
            - self._obstacle_pairs.keys()
            - self._payload_pairs
        ):
            scenes.append("self-collision")
        if hits & self._payload_pairs:
            scenes.append(f"attached payload {self._payload_id!r} contact with the arm")
        if hits & self._floor_pair_indices:
            scenes.append(f"contact with the floor plane at z={self._floor_z:.4f} m")
        scenes.extend(
            f"contact with obstacle {name!r}"
            for name in sorted(
                {self._obstacle_pairs[i] for i in hits if i in self._obstacle_pairs}
            )
        )
        return " and ".join(scenes)

    # ── IK policy: multi-seed restart ────────────────────────────────────────

    def _ik_seeds(self, start: dict[str, float]) -> list[dict[str, float]]:
        """Seed sequence: caller's start, zero, then deterministic samples.

        ``start`` first keeps incremental commands (``ee_move``) landing on the
        nearby branch. The sampler is seeded from :attr:`IK_SEED_RNG` so the same
        request always produces the same answer.
        """
        arm = self._config.arm_joints
        seeds = [dict(start), {n: 0.0 for n in arm}]
        extra = max(0, int(self.IK_SEED_ATTEMPTS) - len(seeds))
        if extra:
            rng = np.random.default_rng(self.IK_SEED_RNG)
            lows = np.array(
                [self._model.lowerPositionLimit[self._arm_idx_q[n]] for n in arm]
            )
            highs = np.array(
                [self._model.upperPositionLimit[self._arm_idx_q[n]] for n in arm]
            )
            for _ in range(extra):
                sample = rng.uniform(lows, highs)
                seeds.append({n: float(v) for n, v in zip(arm, sample)})
        return seeds[: max(1, int(self.IK_SEED_ATTEMPTS))]

    def _solve_seeds(
        self, start: dict[str, float], restart: bool
    ) -> list[dict[str, float]]:
        """The seeds one solve may use. Just the caller's when continuity binds."""
        return self._ik_seeds(start) if restart else [dict(start)]

    def _seed_budget(self, index: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Per-seed solver kwargs: the sampled seeds get the shorter iteration cap.

        An explicit ``max_iter`` from the caller wins for every seed.
        """
        if index < 2 or "max_iter" in kwargs:
            return dict(kwargs)
        return {**kwargs, "max_iter": int(self.IK_RANDOM_SEED_MAX_ITER)}

    def _unwrap_toward(
        self, solution: dict[str, float], reference: dict[str, float]
    ) -> dict[str, float]:
        """Shift each joint by +-2*pi to sit nearest *reference*, if still legal.

        Without this, a solution that is physically identical to the reference
        but numerically a full turn away reads as a huge jump, and the continuity
        check in ``ArmController.pose_path_to_joints`` rejects a valid path. The
        legality check is not optional: adding 2*pi can push a joint past its
        limit, and in that case the unwrapped candidate is discarded rather than
        clamped to a different pose.

        Inert on an arm whose every joint span is under 2*pi (no in-limits value
        then has an in-limits 2*pi-equivalent); it starts doing real work the
        moment a description declares a wide or continuous joint.
        """
        out = dict(solution)
        for name, value in solution.items():
            idx = self._arm_idx_q.get(name)
            if idx is None:
                continue
            ref = reference.get(name)
            if ref is None:
                continue
            candidate = value
            while candidate - ref > np.pi:
                candidate -= 2.0 * np.pi
            while ref - candidate > np.pi:
                candidate += 2.0 * np.pi
            lower = float(self._model.lowerPositionLimit[idx])
            upper = float(self._model.upperPositionLimit[idx])
            # Only take the unwrapped value if it is still legal; otherwise the
            # original (already in-limits) value stands.
            if lower - 1e-9 <= candidate <= upper + 1e-9:
                out[name] = candidate
        return out

    def _refusal(self, solution: dict[str, float]) -> str | None:
        """Which scene refuses *solution*; ``None`` when clear or unchecked."""
        if self._geom_data is None:
            return None
        return self.collision_reason(solution)

    def ik(  # type: ignore[override]
        self,
        start: dict[str, float],
        target: np.ndarray,
        *,
        restart: bool = True,
        **kwargs: Any,
    ) -> dict[str, float] | None:
        """Damped-least-squares IK retried across several seeds; first clear hit wins.

        ``restart=False`` uses ONLY the caller's ``start`` seed. Path solving
        needs it: the fallback seeds are the zero configuration (a singularity on
        many arms) and random samples, and a solution found from one of those
        sits on a different IK branch than the waypoint before it. A branch change
        smaller than the path's ``max_joint_step`` would land in the path and the
        wrist would wander while the TCP stays on the line. For a straight-line
        path "there is no continuous solution" is the honest answer.

        Solutions are unwrapped toward the caller's *start* -- not toward the
        seed that succeeded -- because continuity is defined against where the
        arm actually is. A colliding candidate is discarded and the search
        continues; :attr:`last_ik_refusal` records why, when every converged
        candidate was refused.
        """
        self._last_ik_refusal = None
        refused: list[str] = []
        for index, seed in enumerate(self._solve_seeds(start, restart)):
            solution = super().ik(seed, target, **self._seed_budget(index, kwargs))
            if solution is None:
                continue
            unwrapped = self._unwrap_toward(solution, start)
            # collision_reason costs the same as in_collision -- both evaluate
            # every pair once -- so naming the scene here is free, and it is the
            # only place the answer is known.
            reason = self._refusal(unwrapped)
            if reason is not None:
                refused.append(reason)
                continue
            return unwrapped
        if refused:
            self._last_ik_refusal = "; ".join(sorted(set(refused)))
        return None

    def ik_position(  # type: ignore[override]
        self,
        start: dict[str, float],
        target_position_m: np.ndarray,
        *,
        restart: bool = True,
        **kwargs: Any,
    ) -> dict[str, float] | None:
        """Position-only IK with the same multi-seed collision policy as :meth:`ik`.

        ``restart=False`` means the same thing it does on :meth:`ik`.
        """
        self._last_ik_refusal = None
        refused: list[str] = []
        for index, seed in enumerate(self._solve_seeds(start, restart)):
            solution = super().ik_position(
                seed, target_position_m, **self._seed_budget(index, kwargs)
            )
            if solution is None:
                continue
            unwrapped = self._unwrap_toward(solution, start)
            reason = self._refusal(unwrapped)
            if reason is not None:
                refused.append(reason)
                continue
            return unwrapped
        if refused:
            self._last_ik_refusal = "; ".join(sorted(set(refused)))
        return None

    def diagnose_ik(self, start: dict[str, float], target: np.ndarray) -> str | None:
        """Which guard refused *target*, or ``None`` if it actually solves.

        Only ever called on a failure path. ``ik`` cannot report this itself
        without changing its return contract, and a caller told merely "no
        path" cannot tell an unreachable pose from a scene it should move away
        from -- the two have opposite remedies.
        """
        converged = False
        refusals: list[str] = []
        for index, seed in enumerate(self._ik_seeds(start)):
            solution = super().ik(seed, target, **self._seed_budget(index, {}))
            if solution is None:
                continue
            converged = True
            reason = self._refusal(self._unwrap_toward(solution, start))
            if reason is not None:
                refusals.append(reason)
                continue
            return None  # a seed solved cleanly; nothing to diagnose
        if not converged:
            return self.IK_NO_CONVERGENCE
        # Report every distinct scene that refused, not just the last seed's. The
        # seeds land on different IK branches, so one may hit the shoulder and
        # another the table -- and being told only one of those sends the
        # operator to the wrong remedy.
        return self.IK_REFUSED_PREFIX + "; ".join(sorted(set(refusals)))

    def diagnose_ik_position(
        self,
        start: dict[str, float],
        target_position_m: np.ndarray,
        *,
        posture_reference: dict[str, float] | None = None,
    ) -> str | None:
        """Position-only counterpart of :meth:`diagnose_ik`."""
        converged = False
        refusals: list[str] = []
        for seed in self._ik_seeds(start):
            solution = super().ik_position(
                seed, target_position_m, posture_reference=posture_reference
            )
            if solution is None:
                continue
            converged = True
            reason = self._refusal(self._unwrap_toward(solution, start))
            if reason is not None:
                refusals.append(reason)
                continue
            return None
        if not converged:
            return self.IK_NO_CONVERGENCE
        return self.IK_REFUSED_PREFIX + "; ".join(sorted(set(refusals)))

    @property
    def last_ik_refusal(self) -> str | None:
        """Why the most recent :meth:`ik` returned ``None``, if a scene refused it.

        ``None`` means no seed converged at all -- a statement about the search,
        not about a collision. A **diagnostic side-channel**, not part of the
        result: read it immediately after the call that returned ``None``, on
        the same thread. The next call overwrites it and it makes no promise
        across concurrent callers, which is why nothing gates on it.
        """
        return self._last_ik_refusal

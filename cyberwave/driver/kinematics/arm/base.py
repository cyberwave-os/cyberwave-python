"""BaseKinematicsManipulator — config-driven pinocchio FK/IK for manipulators.

Generalizes per-robot kinematics code: joint names, EE frame, locked joints,
and tool/base/rot axis mappings all come from ``ArmKinematicsConfig`` instead
of module constants. pinocchio is imported lazily so the SDK does not require
it unless a driver actually uses manipulator kinematics.
"""

from __future__ import annotations

import numpy as np

from .config import ArmKinematicsConfig


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = (float(value) for value in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


_PIN_INSTALL_HINT = (
    "pinocchio is required for BaseKinematicsManipulator. Install it with: "
    'pip install "cyberwave[drivers]"'
)


def _import_pin():
    try:
        import pinocchio as pin  # noqa: PLC0415  (lazy by design)
    except ImportError as exc:  # missing module OR broken native libs
        raise ImportError(_PIN_INSTALL_HINT) from exc
    return pin


class BaseKinematicsManipulator:
    """FK/IK on an arm using a reduced model with ``locked_joints`` locked."""

    def __init__(self, config: ArmKinematicsConfig) -> None:
        pin = _import_pin()
        self._pin = pin
        self._config = config
        full = pin.buildModelFromUrdf(config.urdf_path)
        lock = [
            full.getJointId(n) for n in config.locked_joints if full.existJointName(n)
        ]
        self._model = pin.buildReducedModel(full, lock, pin.neutral(full))
        self._data = self._model.createData()
        self._frame_id = self._model.getFrameId(config.ee_frame)
        self._arm_idx_q = {
            n: self._model.joints[self._model.getJointId(n)].idx_q
            for n in config.arm_joints
        }
        self._arm_idx_v = {
            n: self._model.joints[self._model.getJointId(n)].idx_v
            for n in config.arm_joints
        }
        self._base_frame = (
            self._root_frame()
            if config.base_frame is None
            else self._validate_base_frame(config.base_frame)
        )
        self._tool_offset = config.tool_offset
        self._tool_offset_inv = (
            None if config.tool_offset is None else np.linalg.inv(config.tool_offset)
        )

    @property
    def base_frame(self) -> str:
        """The frame ``fk`` poses are expressed in: ``config.base_frame``, or
        the URDF root link when that is unset."""
        return self._base_frame

    def _root_frame(self) -> str:
        """The URDF root link: the first body frame, attached to the universe."""
        pin = self._pin
        return next(
            frame.name
            for frame in self._model.frames
            if frame.type == pin.FrameType.BODY
        )

    def _validate_base_frame(self, name: str) -> str:
        """Fail fast unless *name* really is the frame ``fk`` measures from.

        ``oMf`` is relative to the model origin, so declaring a ``base_frame``
        that is offset from the root would publish poses in the root frame under
        a different frame's name — a silent, systematic error in every
        downstream transform chain. Piper is the motivating case: its URDF root
        is ``dummy_link`` and ``base_link`` is a fixed child, correct only
        because that joint happens to be identity.
        """
        pin = self._pin
        if not self._model.existFrame(name):
            raise ValueError(
                f"base_frame {name!r} is not present in {self._config.urdf_path!r}. "
                "It must name the URDF root (or a frame fixed to it at identity)."
            )
        frame_id = self._model.getFrameId(name)
        q = pin.neutral(self._model)
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacement(self._model, self._data, frame_id)
        placement = self._data.oMf[frame_id]
        if not (
            np.allclose(placement.translation, 0.0, atol=1e-9)
            and np.allclose(placement.rotation, np.eye(3), atol=1e-9)
        ):
            raise ValueError(
                f"base_frame {name!r} is not the frame forward kinematics are "
                f"measured from: it sits at translation "
                f"{np.round(placement.translation, 6).tolist()} relative to the "
                "model origin. Use the URDF root frame, or a frame fixed to it "
                "at identity."
            )
        return name

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return {
            n: (
                float(self._model.lowerPositionLimit[i]),
                float(self._model.upperPositionLimit[i]),
            )
            for n, i in self._arm_idx_q.items()
        }

    def velocity_limits(self) -> dict[str, float]:
        """Per-joint velocity limits (rad/s) from the URDF; only finite, positive
        limits are returned (unlimited joints are omitted)."""
        out: dict[str, float] = {}
        for name, i in self._arm_idx_v.items():
            limit = float(self._model.velocityLimit[i])
            if np.isfinite(limit) and limit > 0.0:
                out[name] = limit
        return out

    def _q_from_dict(self, positions: dict[str, float]) -> np.ndarray:
        q = self._pin.neutral(self._model)
        for name, i in self._arm_idx_q.items():
            if name in positions:
                q[i] = float(positions[name])
        return q

    def _q_to_dict(self, q: np.ndarray) -> dict[str, float]:
        return {name: float(q[i]) for name, i in self._arm_idx_q.items()}

    def fk(self, positions: dict[str, float]) -> np.ndarray:
        """Return the 4x4 homogeneous transform of the TCP (``ee_frame`` when
        ``config.tool_offset`` is None)."""
        q = self._q_from_dict(positions)
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacement(self._model, self._data, self._frame_id)
        pose = self._data.oMf[self._frame_id].homogeneous
        if self._tool_offset is None:
            return pose
        return pose @ self._tool_offset

    def frame_pose(self, positions: dict[str, float], frame: str) -> np.ndarray:
        """The 4x4 transform of any frame of the robot description, in
        :attr:`base_frame`, at *positions*.

        Raises ``KeyError`` for a frame the description does not have.
        """
        if not self._model.existFrame(frame):
            raise KeyError(f"no frame {frame!r} in {self._config.urdf_path!r}")
        frame_id = self._model.getFrameId(frame)
        q = self._q_from_dict(positions)
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacement(self._model, self._data, frame_id)
        return np.array(self._data.oMf[frame_id].homogeneous)

    def interpolate_pose(
        self, start: np.ndarray, target: np.ndarray, t: float
    ) -> np.ndarray:
        """Pose at fraction *t* along a straight line from *start* to *target*.

        Translation is linear and rotation is SLERPed, deliberately **not**
        SE3 screw interpolation (``pin.SE3.Interpolate``), which curves the
        translation and would violate the straight-line approach the guarded
        descent depends on.
        """
        pin = self._pin
        a = np.asarray(start, dtype=float)
        b = np.asarray(target, dtype=float)
        rot_a, rot_b = a[:3, :3], b[:3, :3]
        out = np.eye(4)
        out[:3, :3] = rot_a @ pin.exp3(pin.log3(rot_a.T @ rot_b) * float(t))
        out[:3, 3] = (1.0 - float(t)) * a[:3, 3] + float(t) * b[:3, 3]
        return out

    def pose_waypoints(
        self, start: np.ndarray, target: np.ndarray, steps: int
    ) -> list[np.ndarray]:
        """*steps* poses from *start* (exclusive) to *target* (inclusive)."""
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        return [
            self.interpolate_pose(start, target, i / steps) for i in range(1, steps + 1)
        ]

    def ik(
        self,
        start: dict[str, float],
        target: np.ndarray,
        *,
        eps: float | None = None,
        max_iter: int | None = None,
        dt: float | None = None,
        damp: float | None = None,
    ) -> dict[str, float] | None:
        """Damped least-squares CLIK seeded from *start*. None on failure."""
        pin = self._pin
        cfg = self._config
        eps = cfg.ik_eps if eps is None else eps
        max_iter = cfg.ik_max_iter if max_iter is None else max_iter
        dt = cfg.ik_dt if dt is None else dt
        damp = cfg.ik_damp if damp is None else damp

        target_arr = np.asarray(target, dtype=float)
        if self._tool_offset_inv is not None:
            # ``target`` is a TCP pose; solve for the frame that carries it.
            target_arr = target_arr @ self._tool_offset_inv
        target_se3 = pin.SE3(target_arr)
        q = self._q_from_dict(start)
        lower = self._model.lowerPositionLimit
        upper = self._model.upperPositionLimit
        success = False
        for _ in range(max_iter):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacement(self._model, self._data, self._frame_id)
            i_md = self._data.oMf[self._frame_id].actInv(target_se3)
            err = pin.log(i_md).vector
            if np.linalg.norm(err) < eps:
                success = True
                break
            jac = pin.computeFrameJacobian(self._model, self._data, q, self._frame_id)
            jac = -np.dot(pin.Jlog6(i_md.inverse()), jac)
            v = -jac.T.dot(np.linalg.solve(jac.dot(jac.T) + damp * np.eye(6), err))
            q = pin.integrate(self._model, q, v * dt)
            q = np.clip(q, lower, upper)
        if not success:
            return None
        if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6):
            return None
        return self._q_to_dict(q)

    def ik_position(
        self,
        start: dict[str, float],
        target_position_m: np.ndarray,
        *,
        posture_reference: dict[str, float] | None = None,
        posture_gain: float = 0.2,
        eps: float | None = None,
        max_iter: int | None = None,
        dt: float | None = None,
        damp: float | None = None,
    ) -> dict[str, float] | None:
        """Position-only TCP IK in the base frame; orientation is unconstrained.

        ``posture_reference`` is a soft null-space preference, not an orientation
        constraint. Supplying the trajectory's initial configuration prevents a
        long position-only path from needlessly driving a joint into a hard stop
        when equivalent arm postures remain available.
        """
        pin = self._pin
        cfg = self._config
        eps = cfg.ik_eps if eps is None else eps
        max_iter = cfg.ik_max_iter if max_iter is None else max_iter
        dt = cfg.ik_dt if dt is None else dt
        damp = cfg.ik_damp if damp is None else damp
        target = np.asarray(target_position_m, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_position_m must be a finite [x_m, y_m, z_m]")
        if not np.isfinite(posture_gain) or posture_gain < 0.0:
            raise ValueError("posture_gain must be finite and non-negative")

        q = self._q_from_dict(start)
        q_reference = self._q_from_dict(posture_reference or start)
        lower = self._model.lowerPositionLimit
        upper = self._model.upperPositionLimit
        for _ in range(max_iter):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacement(self._model, self._data, self._frame_id)
            frame = self._data.oMf[self._frame_id]
            offset = np.zeros(3)
            if self._tool_offset is not None:
                offset = frame.rotation @ self._tool_offset[:3, 3]
            error = target - (frame.translation + offset)
            if np.linalg.norm(error) < eps:
                if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6):
                    return None
                return self._q_to_dict(q)

            jacobian = pin.computeFrameJacobian(
                self._model,
                self._data,
                q,
                self._frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            position_jacobian = jacobian[:3, :] - _skew(offset) @ jacobian[3:, :]
            task_inverse = position_jacobian.T @ np.linalg.solve(
                position_jacobian @ position_jacobian.T + damp * np.eye(3),
                np.eye(3),
            )
            velocity = task_inverse @ error
            nullspace = np.eye(self._model.nv) - task_inverse @ position_jacobian
            velocity += nullspace @ (posture_gain * (q_reference - q))
            q = pin.integrate(self._model, q, velocity * dt)
            q = np.clip(q, lower, upper)
        return None

    def apply_translation(
        self,
        pose: np.ndarray,
        components: dict[str, float],
        *,
        frame: str = "tool",
    ) -> np.ndarray:
        """Return *pose* translated by forward/up/left (m) in tool or base frame."""
        axes = self._config.tool_axes if frame == "tool" else self._config.base_axes
        local = np.zeros(3)
        for name, value in components.items():
            if name in axes and value:
                local = local + float(value) * axes[name]
        out = np.array(pose, dtype=float)
        if frame == "tool":
            out[:3, 3] = pose[:3, 3] + pose[:3, :3].dot(local)
        else:
            out[:3, 3] = pose[:3, 3] + local
        return out

    def apply_rotation(
        self,
        pose: np.ndarray,
        axis: str,
        angle: float,
        *,
        frame: str = "tool",
    ) -> np.ndarray:
        """Return *pose* rotated by *angle* (rad) about a named rot axis."""
        rot = self._pin.exp3(self._config.rot_axes[axis] * float(angle))
        out = np.array(pose, dtype=float)
        if frame == "tool":
            out[:3, :3] = pose[:3, :3].dot(rot)
        else:
            out[:3, :3] = rot.dot(pose[:3, :3])
        return out

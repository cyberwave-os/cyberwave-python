"""BaseKinematicsManipulator — config-driven pinocchio FK/IK for manipulators.

Generalizes per-robot kinematics code: joint names, EE frame, locked joints,
and tool/base/rot axis mappings all come from ``ArmKinematicsConfig`` instead
of module constants. pinocchio is imported lazily so the SDK does not require
it unless a driver actually uses manipulator kinematics.
"""

from __future__ import annotations

import numpy as np

from .config import ArmKinematicsConfig

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
            full.getJointId(n)
            for n in config.locked_joints
            if full.existJointName(n)
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
        """Return the 4x4 homogeneous transform of the EE frame."""
        q = self._q_from_dict(positions)
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacement(self._model, self._data, self._frame_id)
        return self._data.oMf[self._frame_id].homogeneous

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

        target_se3 = pin.SE3(np.asarray(target))
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

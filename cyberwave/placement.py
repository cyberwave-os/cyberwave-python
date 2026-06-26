"""SDK-side helpers for centered/MuJoCo-style scene placement.

Cyberwave persists twin pose in URDF link-origin notation: a twin's
``position_*`` fields locate the asset's *link origin* in world coordinates
and ``scale_*`` rescales the asset-local mesh bounds. For primitive
shapes whose STL origin sits at a corner or face rather than the
geometric center, authoring scenes directly in link-origin notation
forces users to manually subtract the asset's local-center offset
every time — and that offset depends on the asset's bounds *and* on
the per-axis scale they wish to apply.

This module exposes pure helpers that translate MuJoCo-style
``(center, dimensions, rotation)`` authoring into the equivalent
``(position, scale, rotation)`` triple stored on the existing
Cyberwave twin fields. The persisted schema and the MuJoCo export are
unchanged; the conversion happens entirely client-side.

Math
----
For an asset whose local mesh bounds form an axis-aligned box
``(min_local, max_local)``:

* ``local_size = max_local - min_local``
* ``local_center = (min_local + max_local) / 2``

For a twin placed in world with link-origin position ``p``, unit
quaternion ``Q`` (asset-local to world) and per-axis scale ``S``:

* ``world_center      = p + Q * (S \u2299 local_center)``
* ``world_dimensions  = S \u2299 local_size``

Inverting to convert centered authoring into Cyberwave notation:

* ``S = world_dimensions / local_size``  (``S = scale`` when ``dimensions=None``)
* ``p = world_center - Q * (S \u2299 local_center)``

The conversion is exact for axis-aligned local bounds and a unit
quaternion. The local bounds are typically derived from the asset's
universal schema; for assets whose schema does not expose usable
bounds the caller supplies an explicit override.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (x, y, z, w) — matches SDK convention
Bounds = Tuple[Vec3, Vec3]  # ((min_x, min_y, min_z), (max_x, max_y, max_z))


# ---------------------------------------------------------------------------
# Well-known asset bounds
# ---------------------------------------------------------------------------

# Local bounds of ``cyberwave/generic_cube`` (measured from the bundled
# ``TheCube.stl``). The STL is *not* centred on the origin: it sits on
# the z=0 plane (origin at the bottom face) and is slightly off-centre
# in y, which is why placing it with link-origin notation produces a
# ~4 cm offset from the visual center for unit-scaled cubes.
GENERIC_CUBE_BOUNDS: Bounds = ((-0.5, -0.4522, 0.0), (0.5, 0.5478, 1.0))


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CenteredPlacement:
    """Cyberwave link-origin pose computed from centered notation."""

    position: Vec3
    scale: Vec3
    rotation: Quat

    def to_twin_fields(self) -> dict:
        """Return a flat dict matching ``TwinCreateSchema`` / ``TwinStateUpdateSchema`` fields."""
        return {
            "position_x": self.position[0],
            "position_y": self.position[1],
            "position_z": self.position[2],
            "rotation_x": self.rotation[0],
            "rotation_y": self.rotation[1],
            "rotation_z": self.rotation[2],
            "rotation_w": self.rotation[3],
            "scale_x": self.scale[0],
            "scale_y": self.scale[1],
            "scale_z": self.scale[2],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IDENTITY_QUAT: Quat = (0.0, 0.0, 0.0, 1.0)


def _as_vec3(value: Sequence[float] | None, *, default: Vec3, name: str) -> Vec3:
    if value is None:
        return default
    try:
        items = list(value)
    except TypeError as e:
        raise ValueError(
            f"{name} must be a length-3 iterable of numbers, got {value!r}"
        ) from e
    if len(items) != 3:
        raise ValueError(f"{name} must be length-3, got length {len(items)}: {value!r}")
    try:
        return (float(items[0]), float(items[1]), float(items[2]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must contain numbers, got {value!r}") from e


def _as_quat(value: Sequence[float] | None) -> Quat:
    if value is None:
        return _IDENTITY_QUAT
    try:
        items = list(value)
    except TypeError as e:
        raise ValueError(
            f"rotation must be a length-4 (x, y, z, w) quaternion, got {value!r}"
        ) from e
    if len(items) != 4:
        raise ValueError(
            f"rotation must be a length-4 (x, y, z, w) quaternion, got length {len(items)}"
        )
    try:
        x, y, z, w = (float(v) for v in items)
    except (TypeError, ValueError) as e:
        raise ValueError(f"rotation must contain numbers, got {value!r}") from e
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("rotation quaternion has zero or non-finite norm")
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate vector ``v`` by quaternion ``q`` (xyzw order).

    Uses the standard ``v + 2*q_vec \u00d7 (q_vec \u00d7 v + q_w * v)`` formulation,
    which is numerically stable and avoids constructing a 3x3 matrix.
    """
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _bounds_local_center_and_size(bounds: Bounds) -> tuple[Vec3, Vec3]:
    try:
        (mnx, mny, mnz), (mxx, mxy, mxz) = bounds
    except (TypeError, ValueError) as e:
        raise ValueError(
            "asset_bounds must be ((min_x, min_y, min_z), (max_x, max_y, max_z))"
        ) from e
    mnx, mny, mnz = float(mnx), float(mny), float(mnz)
    mxx, mxy, mxz = float(mxx), float(mxy), float(mxz)
    size = (mxx - mnx, mxy - mny, mxz - mnz)
    for axis, s in zip(("x", "y", "z"), size):
        if not math.isfinite(s) or s <= 0:
            raise ValueError(
                f"asset_bounds local size along {axis} must be a finite positive number; got {s}"
            )
    center = ((mnx + mxx) * 0.5, (mny + mxy) * 0.5, (mnz + mxz) * 0.5)
    return center, size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_centered_placement(
    *,
    center: Sequence[float],
    asset_bounds: Bounds,
    dimensions: Sequence[float] | None = None,
    scale: Sequence[float] | None = None,
    rotation: Sequence[float] | None = None,
) -> CenteredPlacement:
    """Convert MuJoCo-style centered authoring into Cyberwave link-origin fields.

    Args:
        center: World position ``(x, y, z)`` where the asset's geometric
            center should land.
        asset_bounds: Asset-local AABB
            ``((min_x, min_y, min_z), (max_x, max_y, max_z))``. For
            ``cyberwave/generic_cube`` use :data:`GENERIC_CUBE_BOUNDS`.
        dimensions: Desired world dimensions ``(sx, sy, sz)`` of the
            asset's AABB after scaling. Mutually exclusive with
            ``scale``. ``None`` leaves the asset at unit scale unless
            ``scale`` is provided.
        scale: Explicit per-axis scale ``(sx, sy, sz)``. Mutually
            exclusive with ``dimensions``. Useful when keeping the
            current twin scale and only moving the center.
        rotation: Optional unit quaternion ``(x, y, z, w)`` orienting
            the asset's local frame in world. Defaults to identity.

    Returns:
        :class:`CenteredPlacement` carrying the ``position``, ``scale``
        and normalised ``rotation`` to write to the twin's
        ``position_*`` / ``scale_*`` / ``rotation_*`` fields.

    Raises:
        ValueError: If inputs are malformed (non-positive sizes, both
            ``dimensions`` and ``scale`` specified, zero-norm quaternion,
            etc.).
    """
    if dimensions is not None and scale is not None:
        raise ValueError("Pass either 'dimensions' or 'scale', not both.")

    center_vec = _as_vec3(center, default=(0.0, 0.0, 0.0), name="center")
    local_center, local_size = _bounds_local_center_and_size(asset_bounds)

    if dimensions is not None:
        dim_vec = _as_vec3(dimensions, default=(1.0, 1.0, 1.0), name="dimensions")
        for axis, d in zip(("x", "y", "z"), dim_vec):
            if not math.isfinite(d) or d <= 0:
                raise ValueError(
                    f"dimensions[{axis}] must be a positive number; got {d}"
                )
        scale_vec: Vec3 = (
            dim_vec[0] / local_size[0],
            dim_vec[1] / local_size[1],
            dim_vec[2] / local_size[2],
        )
    elif scale is not None:
        scale_vec = _as_vec3(scale, default=(1.0, 1.0, 1.0), name="scale")
        for axis, s in zip(("x", "y", "z"), scale_vec):
            if not math.isfinite(s) or s <= 0:
                raise ValueError(f"scale[{axis}] must be a positive number; got {s}")
    else:
        scale_vec = (1.0, 1.0, 1.0)

    scaled_local_center: Vec3 = (
        local_center[0] * scale_vec[0],
        local_center[1] * scale_vec[1],
        local_center[2] * scale_vec[2],
    )
    quat = _as_quat(rotation)
    rotated = _quat_rotate(quat, scaled_local_center)
    position: Vec3 = (
        center_vec[0] - rotated[0],
        center_vec[1] - rotated[1],
        center_vec[2] - rotated[2],
    )
    return CenteredPlacement(position=position, scale=scale_vec, rotation=quat)


def compute_center_from_origin(
    *,
    position: Sequence[float],
    asset_bounds: Bounds,
    scale: Sequence[float] | None = None,
    rotation: Sequence[float] | None = None,
) -> Vec3:
    """Return the world-space geometric center for a Cyberwave link-origin pose.

    Inverse of :func:`compute_centered_placement` for fixed
    ``scale``/``rotation``. Useful for round-trip checks and for
    recovering the current center of an already-placed twin.

    Args:
        position: Twin ``(position_x, position_y, position_z)`` in world.
        asset_bounds: Asset-local AABB as for
            :func:`compute_centered_placement`.
        scale: Twin ``(scale_x, scale_y, scale_z)``. Defaults to
            ``(1, 1, 1)``.
        rotation: Twin rotation as ``(x, y, z, w)`` quaternion. Defaults
            to identity.

    Returns:
        ``(center_x, center_y, center_z)`` world position of the
        asset's geometric center.
    """
    pos = _as_vec3(position, default=(0.0, 0.0, 0.0), name="position")
    if scale is None:
        scale_vec: Vec3 = (1.0, 1.0, 1.0)
    else:
        scale_vec = _as_vec3(scale, default=(1.0, 1.0, 1.0), name="scale")
    local_center, _ = _bounds_local_center_and_size(asset_bounds)
    scaled_local_center: Vec3 = (
        local_center[0] * scale_vec[0],
        local_center[1] * scale_vec[1],
        local_center[2] * scale_vec[2],
    )
    quat = _as_quat(rotation)
    rotated = _quat_rotate(quat, scaled_local_center)
    return (pos[0] + rotated[0], pos[1] + rotated[1], pos[2] + rotated[2])


__all__ = [
    "Bounds",
    "CenteredPlacement",
    "GENERIC_CUBE_BOUNDS",
    "Quat",
    "Vec3",
    "compute_center_from_origin",
    "compute_centered_placement",
]

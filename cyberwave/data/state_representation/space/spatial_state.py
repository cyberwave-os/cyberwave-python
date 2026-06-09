"""Proto mirror: space/spatial_state.proto."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpatialState:
    reference_frame: str = ""

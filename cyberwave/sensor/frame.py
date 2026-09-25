"""Capture identity shared by asynchronous camera frame providers."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CapturedVideoFrame:
    """One RGB acquisition, not one WebRTC send tick.

    Cache and return the same sample until a new acquisition is available.
    ``acquisition_id`` must increase within a provider's lifetime even for a
    stationary image. Clocks describe acquisition of the source state, not
    rendering/encoding completion. Pixels are copied once and made read-only
    so a reused producer buffer cannot silently change the captured image.

    This module requires neither MuJoCo nor the optional WebRTC stack.
    """

    image: np.ndarray
    capture_wall_time: float
    capture_monotonic: float
    acquisition_id: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.capture_wall_time)
            or self.capture_wall_time <= 0
            or not math.isfinite(self.capture_monotonic)
            or self.capture_monotonic < 0
            or type(self.acquisition_id) is not int
            or self.acquisition_id < 0
        ):
            raise ValueError(
                "Camera acquisition requires finite clocks and a nonnegative integer id"
            )
        if (
            self.image.dtype != np.uint8
            or self.image.ndim != 3
            or self.image.shape[2] != 3
        ):
            raise ValueError("Camera acquisition pixels must be HxWx3 uint8 RGB")
        pixels = np.array(self.image, copy=True, order="C")
        pixels.setflags(write=False)
        object.__setattr__(self, "image", pixels)

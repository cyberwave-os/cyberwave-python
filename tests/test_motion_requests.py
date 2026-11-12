"""The grasp request: bounded, normalised, defaulted, resolved into the planning frame."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cyberwave.driver.motion.guards import MotionRefusal
from cyberwave.driver.motion.requests import (
    DEFAULT_MAX_GRASP_CANDIDATES,
    parse_grasp,
)
from cyberwave.driver.motion.scene import SceneError
from cyberwave.driver.motion.targets import FrameSource

_POSE = {
    "position": {"x": 0.3, "y": 0.0, "z": 0.2},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}


def _candidate(approach=(0.0, 0.0, -1.0)):
    x, y, z = approach
    return {"pose": _POSE, "approach": {"x": x, "y": y, "z": z}}


def _parse(payload, **kwargs):
    kwargs.setdefault("frames", None)
    return parse_grasp(payload, **kwargs)


def test_candidates_are_the_only_required_argument():
    goal = _parse({"candidates": [_candidate((0.0, 0.0, -2.0))]})
    assert goal.candidates[0].approach == (0.0, 0.0, -1.0)
    assert goal.candidates[0].pose.position == (0.3, 0.0, 0.2)
    assert (goal.approach_distance_m, goal.lift_distance_m) == (0.06, 0.05)
    assert goal.plan_only is False


def test_candidate_count_is_bounded():
    payload = {"candidates": [_candidate()] * (DEFAULT_MAX_GRASP_CANDIDATES + 1)}
    with pytest.raises(SceneError, match="bounded"):
        _parse(payload)
    assert len(_parse(payload, max_candidates=100).candidates) == 65


def test_candidates_in_another_frame_are_moved_into_the_planning_frame():
    """Pose and approach both turn with the frame they were given in."""
    camera = np.eye(4)
    camera[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # yawed +90 deg
    camera[:3, 3] = (0.1, 0.0, 0.5)
    frames = FrameSource("base_link", np.eye(4), lambda name: camera)
    goal = _parse(
        {
            "candidates": [
                {"pose": [0.2, 0.0, -0.3, 0, 0, 0], "approach": [1.0, 0.0, 0.0]}
            ],
            "reference_frame": "camera_link",
        },
        frames=frames,
    )
    candidate = goal.candidates[0]
    assert candidate.pose.position == pytest.approx((0.1, 0.2, 0.2))
    assert candidate.approach == pytest.approx((0.0, 1.0, 0.0))
    s = math.sqrt(0.5)
    assert candidate.pose.orientation == pytest.approx((0.0, 0.0, s, s))


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ([], "grasp data must be an object"),
        ({"candidates": {}}, "candidates must be a non-empty list"),
        ({"candidates": ["x"]}, "candidates[0] must be an object with a pose"),
        (
            {"candidates": [{"pose": _POSE, "approach": "down"}]},
            "candidates[0].approach must be {x, y, z} or [x, y, z]",
        ),
        (
            {"candidates": [_candidate()], "approach_distance_m": -1},
            "approach_distance_m must be positive",
        ),
        (
            {"candidates": [_candidate()], "approach_distance_m": "far"},
            "approach_distance_m is not a number: 'far'",
        ),
    ],
)
def test_an_invalid_grasp_is_refused_as_a_whole(payload, detail):
    with pytest.raises(SceneError) as excinfo:
        _parse(payload)
    assert excinfo.value.refusal.code is MotionRefusal.INVALID_REQUEST
    assert excinfo.value.refusal.detail == detail

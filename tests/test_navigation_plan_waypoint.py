"""NavigationPlan waypoint field tests."""

from cyberwave.navigation import NavigationPlan


def test_waypoint_carries_duration_seconds() -> None:
    plan = NavigationPlan()
    plan.waypoint(1.0, 2.0, 0.0, duration_seconds=2.5, waypoint_id="wp-1")

    waypoint = plan.build()["waypoints"][0]
    assert waypoint["duration_seconds"] == 2.5


def test_waypoint_omits_duration_when_unset() -> None:
    plan = NavigationPlan()
    plan.waypoint(1.0, 2.0, 0.0, waypoint_id="wp-1")

    waypoint = plan.build()["waypoints"][0]
    assert "duration_seconds" not in waypoint


def test_extend_carries_duration_seconds() -> None:
    plan = NavigationPlan()
    plan.extend(
        [
            {"position": [0.0, 0.0, 0.0], "duration_seconds": 1.5},
            {"position": [1.0, 0.0, 0.0]},
        ]
    )

    waypoints = plan.build()["waypoints"]
    assert waypoints[0]["duration_seconds"] == 1.5
    assert "duration_seconds" not in waypoints[1]

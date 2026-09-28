"""Contract tests for the edge-side hand-eye flow, at its SDK home.

Ported from the SO-101 driver's characterisation suite when the flow moved into
the SDK. Same purpose: pin the exact output of the functions that produce what
leaves a driver, because that output is a cross-component contract the dashboard
renders and implements no calibration logic against.

Assert whole dicts and exact strings, not shapes. A loose assertion here is
exactly what would let a silent UI regression through.
"""

from __future__ import annotations

import json

import pytest

from cyberwave.calibration import CharucoBoard
from cyberwave.calibration.config import DEFAULT_TARGET_SAMPLES, HandEyeFlowConfig
from cyberwave.calibration.flow import (
    FOCUS_DRIFT_TOLERANCE,
    HandEyeError,
    HandEyeFlow,
    resolve_docked_fk_frame,
)
from cyberwave.calibration.handeye import MIN_SAMPLES
from cyberwave.calibration.presentation import (
    STATE_APPLIED,
    STATE_ERROR,
    STATE_SOLVED,
    buttons_for_state,
    describe_state,
    preflight_findings,
    preflight_warning,
    verdict,
)
from cyberwave.calibration.testing import (
    FakeCameraTwin,
    FakeFrameSource,
    FakeHandEyeResult,
    FakeHandEyeSession,
    FakeJointSource,
    assert_frame_source_conforms,
    assert_joint_source_conforms,
)

GOOD_STABILITY_M = 0.004
BAD_STABILITY_M = 0.008
GOOD_RESIDUAL_M = 0.005


class Client:
    def __init__(self, twin=None):
        self._twin = twin or FakeCameraTwin()

    def twin(self, twin_id=None, **kwargs):
        return self._twin


def make_config(**overrides) -> HandEyeFlowConfig:
    kwargs = {
        "camera_twin_uuid": "camera-uuid",
        "arm_twin_uuid": "arm-uuid",
        "fk_frame": "gripper",
        "joint_name_map": {f"_{i}": str(i) for i in range(1, 6)},
        "kinematics": object(),
        "board": CharucoBoard(
            squares=(23, 12),
            square_size_m=0.011,
            marker_size_m=0.008,
            dictionary="DICT_4X4_250",
        ),
        "good_stability_m": GOOD_STABILITY_M,
        "bad_stability_m": BAD_STABILITY_M,
        "good_residual_m": GOOD_RESIDUAL_M,
        "settle_tolerance_rad": 0.01,
        "requested_capture_size": (1280, 720),
        "working_distance_m": 0.30,
        "button_flow": "test_handeye",
    }
    kwargs.update(overrides)
    return HandEyeFlowConfig(**kwargs)


def make_flow(**overrides) -> HandEyeFlow:
    config = overrides.pop("config", None) or make_config()
    return HandEyeFlow(
        client=overrides.pop("client", Client()),
        config=config,
        joint_source=overrides.pop(
            "joint_source",
            FakeJointSource({f"_{i}": 0.1 * i for i in range(1, 6)}),
        ),
        frame_source=overrides.pop("frame_source", FakeFrameSource()),
        **overrides,
    )


@pytest.fixture
def flow() -> HandEyeFlow:
    return make_flow()


# --- config ---------------------------------------------------------------


class TestConfig:
    def test_urdf_names_derive_from_the_map(self):
        """One source of truth: two lists that must agree is a latent bug."""
        config = make_config(joint_name_map={"a": "1", "b": "2"})
        assert config.urdf_joint_names == ("1", "2")

    def test_the_map_is_frozen_after_construction(self):
        config = make_config()
        with pytest.raises(TypeError):
            config.joint_name_map["_9"] = "9"  # type: ignore[index]

    def test_a_joint_absent_from_the_map_is_not_solved_for(self):
        """The SO-101 jaw is omitted because it does not move the wrist frame."""
        config = make_config(joint_name_map={f"_{i}": str(i) for i in range(1, 6)})
        assert "6" not in config.urdf_joint_names


# --- snapshot -------------------------------------------------------------


class TestSnapshot:
    def test_a_fresh_run(self, flow):
        assert flow.snapshot() == {
            "state": "capturing",
            "camera_twin_uuid": "camera-uuid",
            "fk_frame": "gripper",
            "samples_captured": 0,
            "samples_usable": 0,
            "samples_target": DEFAULT_TARGET_SAMPLES,
            "min_samples": MIN_SAMPLES,
            "last_capture": None,
            "board": flow._board.to_metadata(),
            "intrinsics_source": None,
        }

    def test_supplied_intrinsics_are_stamped_as_supplied(self):
        """Provenance is fixed when the intrinsics are chosen, never re-resolved,
        so a stored calibration records what the solve actually used."""
        flow = make_flow(intrinsics={"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0})
        meta = flow.snapshot()

        assert meta["intrinsics_source"] == "env"
        assert meta["intrinsics"] == {
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
        }

    def test_capture_size_is_width_then_height(self, flow):
        """The dashboard indexes this positionally as [0]x[1]."""
        flow._frame_shape = (480, 640)
        assert flow.snapshot()["capture_size"] == [640, 480]

    def test_the_solved_blob(self, flow):
        flow._session = FakeHandEyeSession()
        flow._session.sample_count = 5
        flow._state = STATE_SOLVED
        flow._result = FakeHandEyeResult()

        meta = flow.snapshot()

        assert meta["method"] == "park"
        assert meta["sample_count"] == 5
        assert meta["residual_translation_m"] == 0.0029
        assert meta["residual_rotation_deg"] == 0.31
        assert meta["stability_m"] == 0.0026
        assert meta["verdict"] == "good"
        assert meta["relative_translation_m"] == 0.0051

    def test_a_degenerate_leave_one_out_omits_stability(self, flow):
        flow._session = FakeHandEyeSession()
        flow._state = STATE_SOLVED
        flow._result = FakeHandEyeResult(leave_one_out=None)

        meta = flow.snapshot()

        assert "stability_m" not in meta
        assert meta["verdict"] == "marginal"

    def test_error_carries_code_and_message(self, flow):
        flow._state = STATE_ERROR
        flow._error = HandEyeError("bus_unavailable", "Could not open the bus.")

        meta = flow.snapshot()

        assert meta["error"] == "bus_unavailable"
        assert meta["error_description"] == "Could not open the bus."

    def test_the_blob_is_json_serialisable(self, flow):
        """It travels as MQTT JSON: a stray numpy scalar or tuple breaks it."""
        flow._session = FakeHandEyeSession()
        flow._state = STATE_SOLVED
        flow._result = FakeHandEyeResult()
        flow._frame_shape = (720, 1280)

        assert json.loads(json.dumps(flow.snapshot()))["verdict"] == "good"


# --- verdict --------------------------------------------------------------


class TestVerdict:
    @pytest.fixture
    def config(self):
        return make_config()

    @pytest.mark.parametrize(
        "stability_m, expected",
        [
            (0.0, "good"),
            (GOOD_STABILITY_M, "good"),
            (GOOD_STABILITY_M + 1e-9, "marginal"),
            (BAD_STABILITY_M, "marginal"),
            (BAD_STABILITY_M + 1e-9, "bad"),
        ],
    )
    def test_stability_thresholds(self, config, stability_m, expected):
        assert verdict(0.001, stability_m, config=config) == expected

    def test_thresholds_come_from_config_not_constants(self):
        """A second arm sets its own bars; the same numbers must land differently."""
        loose = make_config(good_stability_m=0.02, bad_stability_m=0.05)
        assert verdict(0.001, 0.01, config=loose) == "good"
        assert verdict(0.001, 0.01, config=make_config()) == "bad"

    def test_no_stability_falls_back_capped_at_marginal(self, config):
        assert verdict(0.0, None, config=config) == "marginal"

    def test_no_stability_and_a_gross_residual_is_bad(self, config):
        assert verdict(GOOD_RESIDUAL_M * 3 + 1e-9, None, config=config) == "bad"

    def test_a_stable_answer_that_fits_badly_is_downgraded(self, config):
        assert verdict(GOOD_RESIDUAL_M * 6 + 1e-9, 0.001, config=config) == "marginal"

    @pytest.mark.parametrize(
        "residual_m, stability_m",
        [(0.001, 0.002), (0.001, None), (0.05, None), (0.001, 0.05)],
    )
    def test_a_solve_always_yields_a_renderable_verdict(
        self, config, residual_m, stability_m
    ):
        """An unrecognised verdict renders no badge at all -- the one state that
        tells the operator nothing."""
        assert verdict(residual_m, stability_m, config=config) in {
            "good",
            "marginal",
            "bad",
        }


# --- buttons --------------------------------------------------------------


class TestButtons:
    def _labels(self, flow):
        return [b["label"] for b in buttons_for_state(flow)]

    def test_capturing_below_the_minimum_offers_no_solve(self, flow):
        flow._session = FakeHandEyeSession()
        flow._session.sample_count = MIN_SAMPLES - 1
        assert self._labels(flow) == ["Capture sample", "Cancel"]

    def test_capturing_at_the_minimum_offers_solve(self, flow):
        flow._session = FakeHandEyeSession()
        flow._session.sample_count = MIN_SAMPLES
        assert self._labels(flow) == ["Capture sample", "Solve", "Cancel"]

    def test_solved_offers_apply_capture_more_and_discard(self, flow):
        flow._session = FakeHandEyeSession()
        flow._state = STATE_SOLVED
        assert self._labels(flow) == [
            "Apply to docking offset",
            "Capture more",
            "Discard and exit",
        ]

    def test_applied_offers_only_done(self, flow):
        flow._state = STATE_APPLIED
        assert self._labels(flow) == ["Done"]

    def test_error_offers_restart_and_cancel(self, flow):
        flow._state = STATE_ERROR
        assert self._labels(flow) == ["Restart calibration", "Cancel"]

    def test_the_payload_carries_the_configured_flow_discriminator(self, flow):
        """So a driver with several button flows can tell whose press this is."""
        flow._state = STATE_APPLIED
        (button,) = buttons_for_state(flow)

        assert button["payload"] == {
            "flow": "test_handeye",
            "action": "cancel",
            "camera_twin_uuid": "camera-uuid",
            "target_samples": DEFAULT_TARGET_SAMPLES,
        }

    def test_every_button_matches_what_the_dashboard_validates(self, flow):
        flow._session = FakeHandEyeSession()
        for button in buttons_for_state(flow):
            assert set(button) == {"label", "payload"}
            assert isinstance(button["label"], str) and button["label"].strip()


# --- describe_state -------------------------------------------------------


class TestDescribeState:
    def test_capturing_counts_and_coaches_toward_rotation(self, flow):
        flow._session = FakeHandEyeSession()
        flow._session.sample_count = 3

        name, description, severity = describe_state(flow, flow.snapshot())

        assert name == f"Hand-eye calibration - 3/{DEFAULT_TARGET_SAMPLES} captures"
        assert description == (
            "Hand-guide the wrist to a new orientation, let it settle, then click "
            "Capture sample. Tilt it about genuinely different axes rather than "
            "sliding it around: rotation is what makes the solve possible."
        )
        assert severity == "warning"

    def test_solved_leads_with_the_number_the_verdict_is_keyed_on(self, flow):
        flow._session = FakeHandEyeSession()
        flow._session.sample_count = 5
        flow._state = STATE_SOLVED
        flow._result = FakeHandEyeResult()

        name, description, severity = describe_state(flow, flow.snapshot())

        assert name == "Hand-eye calibration - review"
        assert description == (
            "Solved from 5 captures. "
            "Dropping any single capture moves the result by 2.6 mm. "
            "It reconciles the observed motions to 5.1 mm / 1.42deg. "
            "That is a good result."
        )
        assert severity == "info"

    def test_applied_is_terminal(self, flow):
        flow._state = STATE_APPLIED
        name, _, severity = describe_state(flow, flow.snapshot())

        assert name == "Hand-eye calibration applied"
        assert severity == "info"

    def test_error_reports_the_drivers_own_message(self, flow):
        flow._state = STATE_ERROR
        flow._error = HandEyeError("camera_unavailable", "Could not open the camera.")

        name, description, severity = describe_state(flow, flow.snapshot())

        assert name == "Hand-eye calibration failed"
        assert description == "Could not open the camera."
        assert severity == "error"

    def test_zero_markers_blames_the_dictionary_or_the_framing(self, flow):
        flow._session = FakeHandEyeSession()
        flow._last_capture = "not_detected"
        flow._last_capture_detail = {
            "markers_detected": 0,
            "dictionary": "DICT_4X4_250",
        }

        _, description, _ = describe_state(flow, flow.snapshot())

        assert " No DICT_4X4_250 markers were found at all." in description

    def test_a_transposed_grid_names_the_fix(self, flow):
        """Swapping the axes located the board, so the config is wrong, not the
        framing -- do not send the operator to re-aim a correct camera."""
        flow._session = FakeHandEyeSession()
        flow._last_capture = "not_detected"
        flow._last_capture_detail = {
            "markers_detected": 40,
            "transposed_corners": 30,
        }
        meta = flow.snapshot()
        meta["board"] = {"squares": [12, 23]}

        _, description, _ = describe_state(flow, meta)

        assert "the grid appears transposed" in description
        assert "Set the board's square count to [23, 12]" in description


# --- capture --------------------------------------------------------------


class TestCapture:
    def test_a_capture_brackets_the_frame_grab_with_joint_reads(self, flow):
        """An arm in the same place before and after the grab cannot have moved
        during it -- which is how "the arm was still" becomes a measurement."""
        joints = FakeJointSource({f"_{i}": 0.1 * i for i in range(1, 6)})
        flow = make_flow(joint_source=joints)
        flow._session = FakeHandEyeSession()
        flow._frame_shape = (720, 1280)

        flow.capture_sample()

        assert joints.reads == 2

    def test_bus_joint_names_are_renamed_to_urdf_names(self):
        """The map is data, so an arm with real joint names needs no code change."""
        config = make_config(joint_name_map={"shoulder": "j1", "elbow": "j2"})
        flow = make_flow(
            config=config,
            joint_source=FakeJointSource({"shoulder": 0.5, "elbow": -0.25}),
        )
        session = FakeHandEyeSession()
        flow._session = session
        flow._frame_shape = (720, 1280)

        flow.capture_sample()

        assert session.last_joint_positions == {"j1": 0.5, "j2": -0.25}

    def test_a_joint_the_bus_never_reports_is_refused(self, flow):
        config = make_config(joint_name_map={"_1": "1", "_missing": "2"})
        flow = make_flow(config=config, joint_source=FakeJointSource({"_1": 0.0}))
        flow._session = FakeHandEyeSession()

        with pytest.raises(HandEyeError) as exc:
            flow.capture_sample()

        assert exc.value.code == "joint_mapping"

    def test_a_lost_bus_ends_the_run(self, flow):
        flow = make_flow(joint_source=FakeJointSource(None))
        flow._session = FakeHandEyeSession()

        with pytest.raises(HandEyeError) as exc:
            flow.capture_sample()

        assert exc.value.code == "bus_unavailable"

    def test_a_closed_run_refuses_further_captures(self, flow):
        flow._session = FakeHandEyeSession()
        flow.close()

        with pytest.raises(HandEyeError) as exc:
            flow.capture_sample()

        assert exc.value.code == "run_ended"

    def test_close_releases_both_devices_and_is_idempotent(self):
        joints, frames = FakeJointSource({"_1": 0.0}), FakeFrameSource()
        flow = make_flow(joint_source=joints, frame_source=frames)

        flow.close()
        flow.close()

        assert joints.disconnected and frames.released
        assert not flow.holds_devices


# --- docking validation ---------------------------------------------------


class TestResolveDockedFkFrame:
    """A wrong fk_frame yields a wrong transform with a *good* residual, so
    nothing downstream can detect it. This is the only guard."""

    def test_it_returns_the_link_the_twin_reports(self):
        twin = FakeCameraTwin(attach_to_twin_uuid="arm-1", attach_to_link="wrist")
        assert resolve_docked_fk_frame(twin, arm_twin_uuid="arm-1") == "wrist"

    def test_a_camera_docked_elsewhere_is_refused(self):
        twin = FakeCameraTwin(attach_to_twin_uuid="other-arm")
        with pytest.raises(HandEyeError) as exc:
            resolve_docked_fk_frame(twin, arm_twin_uuid="arm-1")
        assert exc.value.code == "not_docked_here"

    def test_a_camera_docked_to_nothing_is_refused(self):
        twin = FakeCameraTwin(attach_to_twin_uuid="")
        with pytest.raises(HandEyeError) as exc:
            resolve_docked_fk_frame(twin, arm_twin_uuid="arm-1")
        assert exc.value.code == "not_docked_here"

    def test_a_camera_with_no_link_is_refused(self):
        twin = FakeCameraTwin(attach_to_twin_uuid="arm-1", attach_to_link="")
        with pytest.raises(HandEyeError) as exc:
            resolve_docked_fk_frame(twin, arm_twin_uuid="arm-1")
        assert exc.value.code == "no_attach_link"

    def test_an_unreadable_twin_is_reported_as_such(self):
        class Broken:
            def _data_get(self, key):
                raise RuntimeError("network down")

        with pytest.raises(HandEyeError) as exc:
            resolve_docked_fk_frame(Broken(), arm_twin_uuid="arm-1")
        assert exc.value.code == "docking_unreadable"

    def test_the_twins_link_beats_a_disagreeing_payload(self):
        """A payload is not a source of truth about where hardware is bolted."""
        twin = FakeCameraTwin(attach_to_twin_uuid="arm-1", attach_to_link="wrist")
        resolved = resolve_docked_fk_frame(
            twin, arm_twin_uuid="arm-1", requested_frame="base"
        )
        assert resolved == "wrist"


# --- protocol conformance -------------------------------------------------


class TestShippedFakesConform:
    """The shipped fakes must satisfy the protocols they stand in for, or every
    driver that reuses them is testing against a fiction."""

    def test_the_fake_joint_source_conforms(self):
        assert_joint_source_conforms(FakeJointSource({"_1": 0.5}))

    def test_the_fake_frame_source_conforms(self):
        assert_frame_source_conforms(FakeFrameSource())

    def test_conformance_rejects_a_frame_source_returning_none(self):
        class Bad:
            def grab_fresh_frame(self):
                return None

            def release(self):
                pass

        with pytest.raises(AssertionError):
            assert_frame_source_conforms(Bad())

    def test_conformance_rejects_a_joint_source_returning_a_list(self):
        class Bad:
            def read_joint_positions(self):
                return [0.1, 0.2]

            def disconnect(self):
                pass

        with pytest.raises(AssertionError):
            assert_joint_source_conforms(Bad())


# --- the runner's run lifecycle -------------------------------------------


class TestRunnerLifecycle:
    """Regression coverage for the alert that hung on "Done".

    Observed on hardware: a full run (12 captures, solve, apply) left the alert
    stuck on "Waiting..." when the operator pressed Done. Applying resumed the
    preempted operation, that resume path called ``release()`` to hand the devices
    back, and releasing *detached the flow* -- so the later Done press found no
    run and silently did nothing, never resolving the alert.
    """

    @pytest.fixture
    def wiring(self):
        from unittest.mock import MagicMock

        arm_twin = MagicMock(name="arm_twin")
        cam_twin = FakeCameraTwin()

        class Client:
            def twin(self, twin_id=None, **kwargs):
                return cam_twin if twin_id == "camera-uuid" else arm_twin

        finished: List[str] = []
        from cyberwave.calibration.flow import HandEyeRunner

        runner = HandEyeRunner(
            client=Client(),
            arm_twin_uuid="arm-uuid",
            button_flow="test_handeye",
            on_finished=lambda: finished.append("resumed"),
        )
        flow = make_flow(client=Client())
        flow._session = FakeHandEyeSession()
        flow._session.sample_count = 5
        flow._result = FakeHandEyeResult()
        flow._state = STATE_SOLVED
        flow.alert_uuid = "alert-1"
        runner.start(flow)
        return runner, flow, arm_twin, finished

    def test_done_resolves_the_alert_after_an_apply(self, wiring):
        """THE regression. Every step in between must not orphan the run."""
        runner, flow, arm_twin, _ = wiring

        runner.step("apply", solved_at="2026-01-01T00:00:00Z")
        # The resume that apply triggers hands the devices back, exactly as the
        # driver's own resume path does.
        runner.release()

        assert runner.handle_button({"flow": "test_handeye", "action": "cancel"})

        arm_twin.alerts.get.assert_any_call("alert-1")
        arm_twin.alerts.get.return_value.resolve.assert_called_once()

    def test_releasing_an_applied_run_keeps_it_attached(self, wiring):
        """Its devices are already closed, so there is nothing to hand back --
        and detaching would lose the only reference that can resolve the alert."""
        runner, flow, _, _ = wiring
        runner.step("apply", solved_at="2026-01-01T00:00:00Z")

        assert runner.release() is None
        assert runner.flow is flow

    def test_releasing_a_live_run_hands_it_to_the_caller_to_close(self, wiring):
        """The whole point of release(): something else needs the camera.

        The runner detaches and returns the run; closing the devices is the
        caller's job, because only it knows whether it is handing them to another
        operation or shutting down.
        """
        runner, flow, _, _ = wiring

        released = runner.release()

        assert released is flow
        assert runner.flow is None
        assert flow.holds_devices  # still open -- the caller closes it
        released.close()
        assert not flow.holds_devices

    def test_release_leaves_the_alert_alone(self, wiring):
        """Whoever *ends* the run owns resolving it."""
        runner, _, arm_twin, _ = wiring
        arm_twin.alerts.get.return_value.resolve.reset_mock()

        runner.release()

        arm_twin.alerts.get.return_value.resolve.assert_not_called()

    def test_applying_resumes_exactly_once(self, wiring):
        """Not twice: the operator sees one hand-back, not a stutter."""
        runner, _, _, finished = wiring

        runner.step("apply", solved_at="2026-01-01T00:00:00Z")

        assert finished == ["resumed"]

    def test_done_on_an_already_released_live_run_is_a_no_op(self, wiring):
        """A cancel after something else legitimately took the devices."""
        runner, flow, arm_twin, _ = wiring
        runner.release()  # live run, so this really does detach
        arm_twin.alerts.get.return_value.resolve.reset_mock()

        runner.handle_button({"flow": "test_handeye", "action": "cancel"})

        arm_twin.alerts.get.return_value.resolve.assert_not_called()

    def test_a_foreign_flow_discriminator_is_not_ours(self, wiring):
        runner, _, _, _ = wiring
        assert not runner.handle_button({"flow": "other_handeye", "action": "cancel"})


# --- focus drift ----------------------------------------------------------


class FocusingFrameSource(FakeFrameSource):
    """A frame source whose lens can be made to move between captures.

    ``focus_value`` is an OPTIONAL part of the FrameSource contract, which is why
    it lives here rather than on the shipped fake: a ROS image topic or a recorded
    run has no lens to ask, and the base fake models that case.
    """

    def __init__(self, focus=0.0, **kw):
        super().__init__(**kw)
        self.focus = focus

    def focus_value(self):
        return self.focus


class TestFocusDrift:
    """The solve fits ONE camera model for the session, so a lens that refocuses
    mid-run makes fx/fy genuinely differ per view -- the same assumption that
    mixing frame sizes breaks, and that one is already rejected outright.

    Measured rather than trusted: this compares what the lens DID, not what the
    camera claims about autofocus. V4L2 reports 0 both for "autofocus off" and
    for "no such control", and an HTTP MJPEG source has no local lens at all, so
    the claim is unreliable exactly where it would matter most.
    """

    def test_a_still_lens_captures_normally(self):
        frames = FocusingFrameSource(focus=120.0)
        flow = make_flow(frame_source=frames)
        flow._session = FakeHandEyeSession()

        flow.capture_sample()
        flow.capture_sample()

        assert flow.captured_count == 2

    def test_a_lens_that_moves_refuses_the_run(self):
        frames = FocusingFrameSource(focus=120.0)
        flow = make_flow(frame_source=frames)
        flow._session = FakeHandEyeSession()
        flow.capture_sample()

        frames.focus = 200.0
        with pytest.raises(HandEyeError) as exc:
            flow.capture_sample()

        assert exc.value.code == "focus_changed"

    def test_the_moved_capture_is_not_stored(self):
        """A rejected capture must not reach the session: it was taken at a focal
        length the others were not."""
        frames = FocusingFrameSource(focus=120.0)
        flow = make_flow(frame_source=frames)
        flow._session = FakeHandEyeSession()
        flow.capture_sample()

        frames.focus = 200.0
        with pytest.raises(HandEyeError):
            flow.capture_sample()

        assert flow.captured_count == 1

    def test_jitter_within_one_device_step_is_tolerated(self):
        """UVC reports focus in steps of 5 and a stationary lens can wobble by
        one. Tripping on that would make an otherwise fine camera unusable."""
        frames = FocusingFrameSource(focus=120.0)
        flow = make_flow(frame_source=frames)
        flow._session = FakeHandEyeSession()
        flow.capture_sample()

        frames.focus = 120.0 + FOCUS_DRIFT_TOLERANCE
        flow.capture_sample()

        assert flow.captured_count == 2

    def test_drift_is_measured_from_the_first_capture_not_the_previous_one(self):
        """A lens creeping a step at a time would otherwise wander arbitrarily
        far while every individual comparison passed."""
        frames = FocusingFrameSource(focus=100.0)
        flow = make_flow(frame_source=frames)
        flow._session = FakeHandEyeSession()
        flow.capture_sample()

        # Each step is inside the tolerance relative to the one before it, so a
        # previous-capture comparison would wave every one of these through.
        frames.focus = 104.0
        flow.capture_sample()

        frames.focus = 108.0
        with pytest.raises(HandEyeError) as exc:
            flow.capture_sample()

        assert exc.value.code == "focus_changed"
        assert flow.captured_count == 2

    def test_a_source_with_no_focus_control_is_unaffected(self):
        """FakeFrameSource offers no focus_value at all -- the optional-member
        case every non-V4L2 source lands in."""
        flow = make_flow(frame_source=FakeFrameSource())
        flow._session = FakeHandEyeSession()

        flow.capture_sample()
        flow.capture_sample()

        assert flow.captured_count == 2

    def test_a_source_whose_focus_read_raises_is_unaffected(self):
        """Never fail a capture over a diagnostic read."""

        class Broken(FocusingFrameSource):
            def focus_value(self):
                raise RuntimeError("no such control")

        flow = make_flow(frame_source=Broken())
        flow._session = FakeHandEyeSession()

        flow.capture_sample()
        flow.capture_sample()

        assert flow.captured_count == 2

# --- preflight camera check -----------------------------------------------


class ReportingFrameSource(FakeFrameSource):
    """A frame source that answers the optional ``describe_settings`` readout."""

    def __init__(self, settings, **kwargs) -> None:
        super().__init__(**kwargs)
        self._settings = settings

    def describe_settings(self):
        return self._settings


class TestPreflightWarning:
    """The pure judgement, which is where the camera-quirk knowledge lives."""

    def test_autofocus_on_is_named(self):
        assert "autofocus is on" in preflight_warning({"autofocus": 1.0})

    def test_auto_exposure_on_is_named_for_both_encodings(self):
        """V4L2 says 3, OpenCV's other backends normalise the same state to 0.75."""
        for value in (3.0, 0.75):
            assert "auto-exposure is on" in preflight_warning({"auto_exposure": value})

    def test_both_wrong_asks_for_both_in_one_command(self):
        text = preflight_warning({"autofocus": 1, "auto_exposure": 3})
        assert "focus_automatic_continuous=0,auto_exposure=1" in text
        assert "autofocus and auto-exposure are on" in text

    def test_one_wrong_asks_only_for_that_one(self):
        text = preflight_warning({"autofocus": 1, "auto_exposure": 1})
        assert "auto_exposure=1" not in text
        assert "autofocus is on" in text

    def test_the_copyable_command_lives_in_the_findings(self):
        """describe_state appends capture guidance after this text, so no position
        in the paragraph ends a command cleanly. The findings block gives it its
        own element; that is the one meant to be copied."""
        (finding,) = preflight_findings({"autofocus": 1, "device": "/dev/video0"})
        assert finding["fix"].endswith("-c focus_automatic_continuous=0")

    def test_the_description_carries_no_markdown(self):
        """TextWithLinks renders a plain <p>: backticks and "--" print literally."""
        text = preflight_warning({"autofocus": 1, "auto_exposure": 3})
        assert "`" not in text and "--" not in text

    @pytest.mark.parametrize(
        "settings",
        [
            {"autofocus": 0.0, "auto_exposure": 1.0},  # both off
            {"autofocus": 0.0, "auto_exposure": 0.25},  # normalised manual exposure
            {},
            None,
        ],
    )
    def test_silent_when_nothing_is_provably_wrong(self, settings):
        assert preflight_warning(settings) is None

    @pytest.mark.parametrize("reading", [-1.0, -1, -3.0])
    def test_a_negative_reading_is_never_evidence(self, reading):
        """OpenCV answers a negative for a property the backend cannot supply.

        That is the absence of a reading, and it is truthy -- warning on it tells
        an operator their fixed-focus webcam has autofocus on, and hands them a
        v4l2-ctl command that fails with "unknown control".
        """
        assert preflight_warning({"autofocus": reading}) is None
        assert preflight_findings({"autofocus": reading}) == []

    @pytest.mark.parametrize("reading", [2.0, 3.0, 0.5])
    def test_only_an_affirmative_one_counts(self, reading):
        """The reading crosses a package boundary, so anything but 1 means
        "no answer" rather than "on"."""
        assert preflight_warning({"autofocus": reading}) is None

    def test_one_does_count(self):
        assert preflight_warning({"autofocus": 1}) is not None
        assert preflight_warning({"autofocus": 1.0}) is not None

    def test_zero_is_never_evidence(self):
        """V4L2 answers 0 for a control that is off AND for one that does not exist.

        Warning on 0 would fire on every HTTP MJPEG source, which has no controls
        at all -- so 0 has to mean silence even though it costs a real detection.
        """
        assert preflight_warning({"autofocus": 0, "auto_exposure": 0}) is None


class TestPreflightFindings:
    """The structured half, which the dashboard renders as WARNING blocks."""

    def test_one_finding_per_misconfigured_control(self):
        findings = preflight_findings({"autofocus": 1, "auto_exposure": 3})
        assert [f["title"] for f in findings] == [
            "Autofocus is on",
            "Auto-exposure is on",
        ]

    def test_every_finding_carries_title_why_and_fix(self):
        for finding in preflight_findings({"autofocus": 1, "auto_exposure": 3}):
            assert finding["title"] and finding["why"] and finding["fix"]

    def test_the_fix_names_the_real_device(self):
        """A command the operator must reconstruct is one they will not apply."""
        (finding,) = preflight_findings({"autofocus": 1, "device": "/dev/video2"})
        assert finding["fix"] == "v4l2-ctl -d /dev/video2 -c focus_automatic_continuous=0"

    def test_a_readout_without_a_device_still_gives_a_usable_shape(self):
        (finding,) = preflight_findings({"autofocus": 1})
        assert finding["fix"] == "v4l2-ctl -d <device> -c focus_automatic_continuous=0"

    def test_each_fix_changes_only_its_own_control(self):
        """Per-block commands, so fixing one thing cannot silently set another."""
        findings = preflight_findings({"autofocus": 1, "auto_exposure": 3})
        assert "auto_exposure" not in findings[0]["fix"]
        assert "focus_automatic_continuous" not in findings[1]["fix"]

    def test_empty_whenever_the_full_warning_is_silent(self):
        for settings in ({"autofocus": 0, "auto_exposure": 1}, {}, None):
            assert preflight_findings(settings) == []
            assert preflight_warning(settings) is None

    def test_the_two_renderings_always_agree(self):
        """One detector feeds both, so they can never disagree about a camera."""
        for settings in (
            {"autofocus": 1},
            {"auto_exposure": 0.75},
            {"autofocus": 1, "auto_exposure": 3},
            {"autofocus": 0, "auto_exposure": 0},
        ):
            assert bool(preflight_findings(settings)) == (
                preflight_warning(settings) is not None
            )


class TestPreflightInTheFlow:
    def test_the_warning_reaches_the_alert(self):
        flow = make_flow(frame_source=ReportingFrameSource({"autofocus": 1}))
        assert "autofocus is on" in flow.snapshot()["preflight_warning"]

    def test_the_findings_reach_the_alert(self):
        """Separate key: the dashboard renders these as blocks, not as prose."""
        flow = make_flow(frame_source=ReportingFrameSource({"autofocus": 1}))
        (finding,) = flow.snapshot()["preflight_findings"]
        assert finding["title"] == "Autofocus is on"

    def test_it_is_judged_before_the_first_capture(self):
        """The whole point: acting on it means restarting, so it cannot wait."""
        source = ReportingFrameSource({"autofocus": 1})
        flow = make_flow(frame_source=source)
        assert source.grabs == 0
        assert flow.snapshot().get("preflight_warning")

    def test_it_stays_out_of_the_description(self):
        """The warning blocks own this entirely. In the description it was eight
        lines of prose that pushed the capture instructions off the card."""
        flow = make_flow(frame_source=ReportingFrameSource({"autofocus": 1}))
        _, description, _ = describe_state(flow, flow.snapshot())
        assert description.startswith("Hand-guide the wrist")
        assert "autofocus" not in description.lower()

    def test_a_clean_camera_adds_nothing(self):
        flow = make_flow(frame_source=ReportingFrameSource({"autofocus": 0}))
        assert "preflight_warning" not in flow.snapshot()
        assert "preflight_findings" not in flow.snapshot()
        _, description, _ = describe_state(flow, flow.snapshot())
        assert description.startswith("Hand-guide the wrist")

    def test_a_source_that_cannot_report_is_fine(self):
        """RealSense and HTTP MJPEG sources have no V4L2 controls to read."""
        assert "preflight_warning" not in make_flow().snapshot()

    def test_a_readout_that_raises_does_not_break_the_run(self):
        class Exploding(FakeFrameSource):
            def describe_settings(self):
                raise RuntimeError("camera went away")

        flow = make_flow(frame_source=Exploding())
        assert "preflight_warning" not in flow.snapshot()

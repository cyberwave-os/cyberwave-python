"""Edge-side guided hand-eye calibration: collect captures, solve, apply.

The maths lives in :mod:`~cyberwave.calibration.handeye`; this is the *flow* that
drives it from a robot driver — pairing each camera frame with the joint angles
the arm was in, tracking run state, and writing the answer to the camera twin's
docking offset.

**Why this must run on the edge.** :class:`~cyberwave.calibration.handeye.HandEyeSession`
warns that no transport guarantees ``get_frame`` returns an image captured *after*
the arm stopped moving, and a stale frame paired with the current pose is a
silently corrupt sample that no later statistic reveals. A driver colocated with
the hardware reads the bus and grabs the frame on demand, so each sample is
genuinely simultaneous and that failure mode does not exist.

**Devices are injected.** The flow never opens a device itself: it takes a
:class:`~cyberwave.calibration.sources.JointSource` and a
:class:`~cyberwave.calibration.sources.FrameSource`, so (for example) a serial
servo bus with a V4L2 webcam, or a CAN arm with a depth camera, are equally valid
without this module changing. Everything robot-specific — thresholds measured on that arm, the joint
name mapping, the board — arrives in a
:class:`~cyberwave.calibration.config.HandEyeFlowConfig`.

Torque is never enabled. The arm is meant to be hand-guided between captures, so
the joint source is required to read without energising it.

Typical driver wiring::

    flow = HandEyeFlow(
        client=client,
        config=HandEyeFlowConfig(...),
        joint_source=MyBus(...),      # already connected
        frame_source=MyCamera(...),
    )
    runner = HandEyeRunner(
        client=client, arm_twin_uuid=arm_uuid,
        button_flow=config.button_flow, idle_timeout_s=900.0,
        on_finished=resume_whatever_was_running,
    )
    runner.start(flow)                # publishes the guided alert
    runner.handle_button(payload)     # each operator press

The alert carries its own state in ``metadata.hand_eye`` plus the buttons the UI
should draw; the dashboard renders driver-declared state and implements no
calibration logic of its own. See :mod:`~cyberwave.calibration.presentation`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..exceptions import CyberwaveValidationError
from .board import CharucoBoard, resolve_dictionary
from .config import HandEyeFlowConfig
from .handeye import (
    BoardNotDetectedError,
    HandEyeDegenerateError,
    HandEyeSession,
    MIN_SAMPLES,
)
from .presentation import (
    ALERT_TYPE,
    STATE_APPLIED,
    STATE_CAPTURING,
    STATE_ERROR,
    STATE_SOLVED,
    buttons_for_state,
    capture_mode_warning,
    describe_state,
    preflight_findings,
    preflight_warning,
    stability_from_result,
    verdict,
)
from .sources import FrameSource, JointSource

logger = logging.getLogger(__name__)

#: How far the lens may move between captures before the run is refused.
#:
#: The solve fits ONE camera model for the whole session, so a lens that
#: refocuses mid-run makes fx/fy genuinely differ per view and the fitted model
#: wrong for some of them -- showing up only as an inflated residual with nothing
#: pointing at the cause. Same assumption ``add_sample`` already protects by
#: rejecting mixed frame sizes outright.
#:
#: Measured, not trusted: this compares what the lens actually DID between
#: captures, rather than asking the camera whether autofocus is off. A source
#: that reports no focus position at all reports a constant, which reads as
#: "never moved" and correctly raises nothing.
#:
#: UVC reports focus in device units, conventionally 0-255 in steps of 5, and a
#: stationary lens can still jitter by one step. The tolerance is one step: large
#: enough that a still lens never trips it, small enough that a real refocus --
#: which moves tens of units -- always does.
FOCUS_DRIFT_TOLERANCE = 5.0


class HandEyeError(Exception):
    """A hand-eye step failed in a way the operator has to act on.

    Carries a stable ``code`` so the frontend can map it to guidance without
    parsing prose.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _joint_drift(
    before: Optional[Dict[str, float]], after: Optional[Dict[str, float]]
) -> Optional[Dict[str, Any]]:
    """Largest per-joint movement between two reads of the same arm.

    Reading the joints again *after* the frame grab is what turns "the arm was
    still" from an assumption into a measurement: an arm in one place before the
    grab and the same place after cannot have moved during it.
    """
    if not before or not after:
        return None
    shared = sorted(set(before) & set(after))
    if not shared:
        return None
    per_joint = {name: abs(float(after[name]) - float(before[name])) for name in shared}
    worst = max(per_joint, key=lambda name: per_joint[name])
    return {"max_rad": round(per_joint[worst], 6), "worst_joint": worst}


def intrinsics_from_twin(
    camera_twin: Any, frame_shape: Tuple[int, ...]
) -> Optional[Dict[str, float]]:
    """Intrinsics already stored on the camera twin for *this* capture profile.

    Reads ``twin.camera.intrinsics_for(width, height)`` -- a real measurement
    someone previously made on this device, keyed by stream profile. Matched
    against the frame that actually arrived, never the twin's declared size:
    intrinsics do not transfer between profiles, and the SDK deliberately returns
    ``None`` rather than rescaling one, so a mismatch has to fall through to
    solving rather than be papered over.

    Returns ``None`` -- meaning "fall through to solving from board views" -- when
    the SDK is too old to expose the accessor, when the twin carries no
    calibration, when nothing is stored for this profile, or when the lookup
    fails. A stored calibration is a convenience here, never a requirement: the
    session can always fit its own intrinsics, so nothing about this is worth
    failing a run over.
    """
    try:
        # Inside the try, not before it: `camera_twin.camera` resolves through
        # Twin.__getattr__ into capability lookup, which can raise more than
        # AttributeError (a failed capability fetch, say). getattr(..., default)
        # only swallows AttributeError, so anything else would escape a function
        # whose contract -- and the `except` below -- is that it never fails a run.
        camera = getattr(camera_twin, "camera", None)
        accessor = getattr(camera, "intrinsics_for", None)
        if accessor is None:
            # The per-profile sensor calibration API does not exist yet: nothing
            # in the platform stores camera intrinsics today (see
            # persistence.py), so this path is the norm, not a legacy fallback.
            return None
        height_px, width_px = int(frame_shape[0]), int(frame_shape[1])
        stored = accessor(width_px, height_px)
        if stored is None:
            logger.info(
                "Hand-eye: camera twin has no stored intrinsics for %dx%d; they "
                "will be solved from the captured board views",
                width_px,
                height_px,
            )
            return None
        intrinsics = {
            "fx": float(stored.fx_px),
            "fy": float(stored.fy_px),
            "cx": float(stored.cx_px),
            "cy": float(stored.cy_px),
        }
        logger.info(
            "Hand-eye: using intrinsics stored on the camera twin for %dx%d "
            "(source %s, RMS %s px)",
            width_px,
            height_px,
            getattr(stored, "source", "unknown"),
            getattr(stored, "rms_reprojection_error_px", None),
        )
        return intrinsics
    except Exception:
        # Never fatal: the session fits its own intrinsics when handed none.
        logger.warning(
            "Hand-eye: could not read stored intrinsics off the camera twin; "
            "they will be solved from the captured board views",
            exc_info=True,
        )
        return None


class HandEyeFlow:
    """One guided eye-in-hand calibration run.

    Owns the motor bus (read-only) and the camera device for its lifetime, so a
    capture is a synchronous (frame, joint-angle) pair. Single-use: build one per
    run and :meth:`close` it when the flow ends.

    Thread-safety: every public method takes ``_lock``. Button presses arrive on
    the driver's command worker thread while nothing else touches the flow, but
    the devices must not be reopened concurrently.
    """

    def __init__(
        self,
        *,
        client: Any,
        config: HandEyeFlowConfig,
        joint_source: JointSource,
        frame_source: FrameSource,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> None:
        self._client = client
        self._config = config
        #: Required, not lazily built: the flow no longer knows how to open any
        #: particular device, which is what lets it serve more than one robot.
        self._bus: Any = joint_source
        self._capture: Any = frame_source
        self._camera_twin_uuid = config.camera_twin_uuid
        self._robot_twin_uuid = config.arm_twin_uuid
        self._fk_frame = config.fk_frame
        self._target_samples = config.target_samples
        self._board = config.board
        self._board_spec = dict(config.board_spec or {})
        self._kinematics = config.kinematics
        #: Caller-supplied intrinsics, when known up front. ``None`` means the
        #: session either finds a calibration stored on the camera twin or solves
        #: real ones from the captured board views.
        self._intrinsics = dict(intrinsics) if intrinsics else None

        # Guards flow state (samples, result, alert uuid). Held across a capture,
        # which blocks on device I/O and board detection.
        self._lock = threading.RLock()
        # Guards *only* the device handles, so :meth:`close` never waits on an
        # in-flight capture. Both source protocols document their teardown as safe
        # to call from another thread at any point, so a wedged read can be
        # force-released; routing teardown through ``_lock`` would throw that away
        # and let a hung camera block the caller's command worker.
        self._device_lock = threading.RLock()
        #: Set once :meth:`close` has run. ``close`` deliberately does not hold
        #: ``_lock``, so it can land *while* a capture is between its own device
        #: calls; without this flag that capture's next ``_ensure_*`` would see a
        #: cleared handle and re-acquire the device on a run the caller has
        #: already dropped, leaking it until the process exits.
        self._closed = False
        self._alert_uuid: Optional[str] = None
        self._state = STATE_CAPTURING
        self._last_capture: Optional[str] = None
        #: Why the last capture missed the board, when it did.
        self._last_capture_detail: Dict[str, Any] = {}
        self._error: Optional[HandEyeError] = None
        self._result: Any = None
        self._session: Any = None
        #: ``(height, width)`` the intrinsics were derived from, so a mid-run mode
        #: change can be caught instead of silently rescaling later board poses.
        self._frame_shape: Optional[Tuple[int, int]] = None
        #: Lens position at the first capture. Every later capture is compared
        #: against it, so a lens that creeps a little at a time is caught just the
        #: same as one that jumps -- comparing against the previous capture would
        #: let it drift arbitrarily far in small steps.
        self._focus_reference: Optional[float] = None
        #: Which source the intrinsics came from, fixed when they were chosen and
        #: never re-resolved, so a stored calibration records what the solve used.
        #: ``"env"``, not ``"supplied"``: this token is a wire contract. The
        #: dashboard's label map, the ``intrinsics_source`` union in
        #: lib/types/alert.ts, presentation.py's contract docstring and the guard
        #: in :meth:`solve` all key on ``env``. The flow itself no longer knows the
        #: values came from an environment variable, but renaming the *token*
        #: breaks the UI silently -- it would render the bare word instead of a
        #: label. Rename the meaning in a comment, never the value on the wire.
        self._intrinsics_source: Optional[str] = "env" if self._intrinsics else None
        #: Whether the mode the camera granted can decode this board, judged once
        #: on the first frame. Carried into the alert so a resolution problem does
        #: not masquerade as bad framing.
        self._capture_warning: Optional[str] = None
        #: What the camera is set to, read back at construction -- the frame source
        #: arrives already open, so this is the earliest point the answer exists,
        #: and it has to be earlier than the first capture to be worth anything.
        #: Optional on the protocol: a source with no controls to report says so by
        #: not implementing it.
        _camera_settings = self._read_camera_settings()
        self._preflight_warning: Optional[str] = preflight_warning(_camera_settings)
        #: The same findings as rendered blocks for the alert body -- the dashboard
        #: shows each as a WARNING with its reason and the command that fixes it.
        self._preflight_findings: List[Dict[str, str]] = preflight_findings(
            _camera_settings
        )
        if self._preflight_warning:
            logger.warning("Hand-eye: %s", self._preflight_warning)
        self._last_activity_at = time.monotonic()

    # --- device + model lifecycle -------------------------------------------

    def _raise_if_closed(self) -> None:
        """Refuse to acquire a device once the run has been torn down.

        Callers hold ``_device_lock``. Raising here turns the race into a clean
        failure for the straggling capture instead of a device nobody owns: the
        flow is already unreachable from ``_handeye_flow`` by this point, so
        anything opened now would never be closed.
        """
        if self._closed:
            raise HandEyeError(
                "run_ended",
                "This hand-eye run has already ended (cancelled, applied, or idle "
                "for too long). Start a new calibration.",
            )

    def _ensure_bus(self) -> Any:
        """The joint source. Present from construction; only the closed check runs."""
        with self._device_lock:
            self._raise_if_closed()
            return self._bus

    def _ensure_capture(self) -> Any:
        """The frame source. Present from construction."""
        with self._device_lock:
            self._raise_if_closed()
            return self._capture

    def _read_camera_settings(self) -> Optional[Mapping[str, Any]]:
        """The frame source's own account of its settings, or ``None``.

        ``getattr`` rather than a required protocol method: the readout is a
        diagnostic, and a source that cannot answer must cost nothing. Failure is
        swallowed for the same reason -- no run should die over a warning it was
        only ever going to print.
        """
        reader = getattr(self._capture, "describe_settings", None)
        if reader is None:
            return None
        try:
            return reader()
        except Exception:
            logger.debug("Could not read the camera settings", exc_info=True)
            return None

    def _ensure_kinematics(self) -> Any:
        """Forward kinematics, supplied by the caller.

        Duck-typed (``fk(dict) -> 4x4``) rather than constructed here, so this
        module needs no ``cyberwave.driver`` import and hand-eye stays installable
        without pinocchio.
        """
        if self._kinematics is None:
            raise HandEyeError(
                "no_kinematics",
                "No kinematics object was configured, so joint angles cannot be "
                "turned into an end-effector pose.",
            )
        return self._kinematics

    def _ensure_session(self, frame_shape: Tuple[int, ...]) -> Any:
        """Build the SDK session once the first frame has told us the real size."""
        if self._session is not None:
            # Intrinsics were derived from the first frame's dimensions, so a later
            # mode change would silently scale every subsequent board pose against
            # the wrong principal point. Refuse rather than mix scales.
            if self._frame_shape is not None and tuple(frame_shape[:2]) != self._frame_shape:
                raise HandEyeError(
                    "frame_size_changed",
                    f"The camera switched from {self._frame_shape[1]}x{self._frame_shape[0]} "
                    f"to {int(frame_shape[1])}x{int(frame_shape[0])} mid-run. The "
                    "intrinsics no longer match, so the captures cannot be mixed -- "
                    "restart the calibration.",
                )
            return self._session

        camera_twin = self._client.twin(twin_id=self._camera_twin_uuid)
        if self._intrinsics is None:
            # Record which source won at the moment it won, and never re-resolve:
            # re-deciding this when the alert renders could report a different
            # provenance than the solve actually used.
            #
            # Order here is: a calibration already stored on the camera twin, else
            # solve it from the captured board views. A caller-supplied override
            # (``intrinsics=``) has already won by this point and skips the block
            # entirely -- its provenance is stamped in the constructor. There is
            # deliberately no field-of-view rung: deriving fx/fy from a declared FOV
            # is good enough to draw a frustum and not good enough for metric work,
            # and being an *input* to hand-eye its error surfaced only as an inflated
            # residual with nothing pointing at the cause.
            stored = intrinsics_from_twin(camera_twin, frame_shape)
            if stored is not None:
                self._intrinsics = stored
                self._intrinsics_source = "stored"
            else:
                self._intrinsics_source = "solved"
            logger.info(
                "Hand-eye intrinsics (%s): %s",
                self._intrinsics_source,
                self._intrinsics or "to be solved from captured board views",
            )

        arm_twin = self._client.twin(twin_id=self._robot_twin_uuid)
        # validate_attachment stays on: it is what rules out FK reported about a
        # link the camera is not docked to, which yields a wrong transform with a
        # good residual.
        self._session = HandEyeSession(
            camera_twin=camera_twin,
            arm_twin=arm_twin,
            fk_frame=self._fk_frame,
            board=self._board,
            intrinsics=self._intrinsics,
            kinematics=self._ensure_kinematics(),
        )
        self._frame_shape = (int(frame_shape[0]), int(frame_shape[1]))
        # Judged once, on the mode the camera actually granted rather than the one
        # requested. A warning only -- the detector downstream has the final say.
        self._capture_warning = capture_mode_warning(
            self._board,
            frame_shape,
            self._intrinsics,
            working_distance_m=self._config.working_distance_m,
            requested_size=self._config.requested_capture_size,
        )
        if self._capture_warning:
            logger.warning("Hand-eye: %s", self._capture_warning)
        return self._session

    def close(self) -> None:
        """Release the bus and the camera. Idempotent, and safe from any thread.

        Deliberately takes only ``_device_lock``: this runs on the command worker
        via ``_stop_current_operation``, and must not be blocked by a capture that
        is stuck reading a device someone unplugged.
        """
        with self._device_lock:
            bus, capture = self._bus, self._capture
            # Clear the handles and latch ``_closed`` in one critical section: a
            # concurrent capture must find the run shut, not a blank slate to
            # re-open (see ``_raise_if_closed``).
            self._closed = True
            self._bus = None
            self._capture = None
        if bus is not None:
            try:
                bus.disconnect()
            except Exception:
                logger.debug("Hand-eye: bus disconnect failed", exc_info=True)
        if capture is not None:
            try:
                capture.release()
            except Exception:
                logger.debug("Hand-eye: frame source release failed", exc_info=True)

    # --- capture / solve / apply -------------------------------------------

    def _grab_frame(self) -> Any:
        """One frame that postdates the arm coming to rest.

        How that freshness is achieved is the source's business -- for the default
        V4L2 camera it is a buffer drain, because V4L2 hands back whatever is
        queued, which after a pause is an image from before the arm was moved.
        """
        return self._ensure_capture().grab_fresh_frame()

    def _detection_diagnostics(self, frame: Any) -> Dict[str, Any]:
        """Count raw ArUco markers in a frame the board detector rejected.

        A bare "board not found" gives the operator nowhere to go, and the three
        causes need opposite actions:

        * **0 markers** -- the printed board does not match ``dictionary``, or it is
          out of frame / too small / motion-blurred.
        * **some markers, no board** -- right dictionary, but too little of the grid
          is visible, or ``squares`` does not match the print.
        * **markers whose ids all belong to the board, but still no board** -- the
          grid is almost certainly transposed. ``detectBoard`` interpolates *zero*
          corners when ``squares`` is ``(rows, cols)`` instead of ``(cols, rows)``,
          which looks identical to "not enough of the board is visible" unless we
          actually retry the swap. So we do, and report it.

        Best-effort and only on the failure path, so it costs nothing normally.
        """
        dictionary_name = getattr(self._board, "dictionary", None)
        if not dictionary_name:
            return {}
        try:
            import cv2

            dictionary = resolve_dictionary(cv2, str(dictionary_name))
            array = frame if getattr(frame, "ndim", 0) == 2 else cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY
            )
            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(
                    dictionary, cv2.aruco.DetectorParameters()
                )
                _corners, ids, _rejected = detector.detectMarkers(array)
            else:  # pragma: no cover - OpenCV < 4.7
                _corners, ids, _rejected = cv2.aruco.detectMarkers(array, dictionary)
            detail: Dict[str, Any] = {
                "markers_detected": 0 if ids is None else int(len(ids)),
                "dictionary": str(dictionary_name),
            }
            if ids is not None and len(ids):
                detail.update(self._transpose_diagnostics(frame, ids))
            return detail
        except Exception:
            logger.debug("Hand-eye: detection diagnostics failed", exc_info=True)
            return {}

    def _transpose_diagnostics(self, frame: Any, ids: Any) -> Dict[str, Any]:
        """Test whether a transposed ``squares`` would have found the board.

        Only meaningful once markers were found, so it is called on that branch
        alone. Returns the count of detected ids that belong to *this* board -- ids
        outside the range mean a different board or dictionary, not a transpose --
        plus the corner count a swapped grid yields, when the board exposes one.
        """
        out: Dict[str, Any] = {}
        squares = getattr(self._board, "squares", None)
        if not squares or len(squares) != 2:
            return out
        try:
            import cv2

            found = {int(i) for i in ids.ravel()}
            cols, rows = int(squares[0]), int(squares[1])
            # A ChArUco board of n squares carries n//2 markers, ids 0..n//2-1.
            legal = (cols * rows) // 2
            out["markers_in_board_range"] = sum(1 for i in found if 0 <= i < legal)

            swapped = CharucoBoard(
                squares=(rows, cols),
                square_size_m=float(self._board.square_size_m),
                marker_size_m=float(self._board.marker_size_m),
                dictionary=str(self._board.dictionary),
            )
            corners = swapped.correspondences(cv2, frame, minimum=1)
            out["transposed_corners"] = 0 if corners is None else int(len(corners[0]))
        except Exception:
            logger.debug("Hand-eye: transpose diagnostics failed", exc_info=True)
        return out

    def _read_bus_positions(self) -> Dict[str, float]:
        """Raw joint angles straight off the bus, under their twin-schema names.

        Split out from :meth:`_joint_positions` so a capture can read the arm a
        second time after the frame grab -- the cheap check that it did not move
        during the capture -- without repeating the URDF renaming.
        """
        bus = self._ensure_bus()
        positions = bus.read_joint_positions()
        return dict(positions) if positions else {}

    def _joint_positions(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Current joint angles as ``(urdf_named, raw_bus_named)``.

        Both are returned so a recorded run shows the bus read as well as the
        renamed one: a joint-mapping bug is then visible in the file rather than
        something to infer from a wrong pose.
        """
        bus = self._ensure_bus()
        positions = bus.read_joint_positions()
        if positions is None:
            raise HandEyeError(
                "bus_unavailable",
                "Lost the connection to the arm mid-calibration. "
                "Check the connection and restart the calibration.",
            )
        if not positions:
            raise HandEyeError(
                "no_joint_read",
                "The arm reported no joint angles. Hold still and capture again.",
            )

        # Rename bus joints to URDF joints through the injected map, rather than
        # by a name rule. On the SO-101 that map is ``_1``->``1`` etc. with the jaw
        # absent, because the jaw does not move the wrist frame; on another arm it
        # is whatever that bus calls its joints.
        mapped = {
            urdf_name: positions[bus_name]
            for bus_name, urdf_name in self._config.joint_name_map.items()
            if bus_name in positions
        }
        missing = [n for n in self._config.joint_name_map.values() if n not in mapped]
        if missing:
            raise HandEyeError(
                "joint_mapping",
                f"The arm did not report joint(s) {missing}, so forward kinematics "
                "cannot run. Check the arm's joint calibration, and that the "
                "configured joint_name_map matches the names the bus reports.",
            )
        return mapped, dict(positions)

    def _read_focus(self) -> Optional[float]:
        """The lens position the frame source reports, or ``None``.

        ``focus_value`` is an optional part of the :class:`FrameSource` contract:
        a ROS image topic or a recorded run has no lens to ask, and must not be
        penalised for it.
        """
        reader = getattr(self._capture, "focus_value", None)
        if reader is None:
            return None
        try:
            value = reader()
        except Exception:
            # Never fail a capture over a diagnostic read.
            logger.debug("Could not read the lens focus position", exc_info=True)
            return None
        return float(value) if value is not None else None

    def _check_focus_is_unchanged(self, index: int) -> None:
        """Refuse the run if the lens has moved since the first capture.

        Raised rather than warned, and fatal to the run rather than to the one
        capture. By the time this trips, the captures already stored were taken at
        a different focal length than this one, so there is no subset of them that
        is jointly valid -- dropping only the newest would leave the operator
        capturing more samples into a session that can no longer be solved
        correctly. The error state offers "Restart calibration", which is exactly
        the recovery: disable autofocus, then start again.
        """
        focus = self._read_focus()
        if focus is None:
            return
        if self._focus_reference is None:
            self._focus_reference = focus
            logger.debug("Hand-eye: lens focus reference is %.1f", focus)
            return
        moved = abs(focus - self._focus_reference)
        if moved <= FOCUS_DRIFT_TOLERANCE:
            return
        raise HandEyeError(
            "focus_changed",
            f"The camera refocused between captures (lens moved {moved:.0f} units "
            f"by capture {index}). Every capture has to share one focal length, so "
            "the samples taken so far cannot be combined with any taken now. "
            "Turn autofocus off in the camera's settings, then restart the "
            "calibration.",
        )

    def capture_sample(self) -> Dict[str, Any]:
        """Pair the pose the arm is in now with a detection of the board."""

        with self._lock:
            # Bracket the frame grab with joint reads. The pairing is already
            # simultaneous by construction -- both reads are local and on demand --
            # but an arm in the same place before and after the grab cannot have
            # moved during it, and drift is invisible in the residual afterwards.
            before = self._read_bus_positions()
            frame = self._grab_frame()
            joint_positions, raw_positions = self._joint_positions()
            drift = _joint_drift(before, raw_positions)
            index = self.captured_count + 1
            if drift is not None and drift["max_rad"] > self._config.settle_tolerance_rad:
                # Loud but not fatal: the operator is hand-guiding the arm, and a
                # capture taken while it was still drifting is exactly the kind of
                # sample that looks fine and quietly widens the solve.
                logger.warning(
                    "Hand-eye capture %d: arm moved %.4f rad (joint %s) across the "
                    "frame grab -- let it settle before capturing",
                    index,
                    drift["max_rad"],
                    drift["worst_joint"],
                )

            self._check_focus_is_unchanged(index)

            try:
                session = self._ensure_session(frame.shape)
            except CyberwaveValidationError as exc:
                # The SDK session validates that the camera really is docked to
                # this arm at fk_frame. That check is the only thing standing
                # between a mislabelled link and a wrong transform with a good
                # residual, so surface it with its own code rather than as an
                # unexpected error.
                raise HandEyeError("attachment_mismatch", str(exc)) from exc

            try:
                session.add_sample(joint_positions=joint_positions, image=frame)
            except BoardNotDetectedError as exc:
                # Not fatal to the run, but never silently skipped: a session that
                # quietly drops captures ends up confident and wrong.
                self._last_capture = "not_detected"
                self._last_capture_detail = self._detection_diagnostics(frame)
                # The session distinguishes "no board at all" from "too little of
                # it", and its wording is more specific than anything rebuilt from
                # the diagnostics here, so carry it through to the operator.
                self._last_capture_detail["reason"] = str(exc)
                logger.info(
                    "Hand-eye: board not detected (%s); board spec %s",
                    self._last_capture_detail or "no diagnostics",
                    self._board.to_metadata(),
                )
                self._state = STATE_CAPTURING
                return self.snapshot()

            self._last_capture = "detected"
            self._last_capture_detail = {}
            self._state = STATE_CAPTURING
            # A fresh capture invalidates the previous solve.
            self._result = None
            return self.snapshot()

    def solve(self) -> Dict[str, Any]:
        """Solve for the camera's pose on the wrist from the captured samples."""

        with self._lock:
            if self._session is None or self._session.sample_count < MIN_SAMPLES:
                raise HandEyeError(
                    "too_few_samples",
                    f"Need at least {MIN_SAMPLES} captures before solving "
                    f"(have {self.sample_count}). Aim for {self._target_samples}.",
                )
            try:
                self._result = self._session.solve()
            except HandEyeDegenerateError as exc:
                raise HandEyeError("degenerate", str(exc)) from exc
            except CyberwaveValidationError as exc:
                raise HandEyeError("solve_failed", str(exc)) from exc

            # solve() refits intrinsics from every captured view, so the values
            # cached at capture time are now stale. Re-read them or the alert and
            # the stored provenance would report intrinsics the solve did not use.
            solved = getattr(self._session, "solved_intrinsics", None)
            # The `!= "env"` conjunct is belt-and-braces, not the real protection:
            # HandEyeSession refuses to refit at all when the caller pinned
            # intrinsics (handeye.py, _solve_intrinsics_and_pose_samples), so
            # `solved` is already None on that path.
            if self._intrinsics_source != "env" and solved is not None:
                self._intrinsics = solved.as_dict()
                logger.info(
                    "Hand-eye intrinsics refitted from %d board views (RMS %.3f px): %s",
                    solved.view_count,
                    solved.rms_reprojection_px,
                    self._intrinsics,
                )

            self._state = STATE_SOLVED
            logger.info(
                "Hand-eye solved: residual_t=%.4f m residual_r=%.3f deg",
                self._result.residual_translation_m,
                self._result.residual_rotation_deg,
            )
            return self.snapshot()

    def apply(self, solved_at: Optional[str] = None) -> Dict[str, Any]:
        """Write the solved transform to the camera twin's docking offset."""
        with self._lock:
            if self._result is None:
                raise HandEyeError(
                    "not_solved", "Nothing to apply -- solve the calibration first."
                )

            camera_twin = self._client.twin(twin_id=self._camera_twin_uuid)
            # ``calibration`` comes from CameraCapableMixin, and the SDK's twin
            # factory only selects CameraTwin when the twin reports imaging
            # sensors. A twin with empty capabilities falls back to plain Twin,
            # where this would be a bare AttributeError.
            if not hasattr(camera_twin, "calibration"):
                raise HandEyeError(
                    "not_a_camera_twin",
                    f"Twin {self._camera_twin_uuid} does not report an imaging sensor, "
                    "so the calibration cannot be written to it. Sync the twin with its "
                    "asset so its camera capability is populated.",
                )
            camera_twin.calibration.set(
                self._result,
                board=self._board,
                intrinsics=self._intrinsics,
                fk_frame=self._fk_frame,
                solved_at=solved_at,
                # Persist the bars this run was judged against. The dashboard
                # renders a badge from the stored record long after the run is
                # gone, and these are measured per arm -- without them it has to
                # hardcode one robot's numbers and shows a green badge for a
                # different arm whose "good" is somewhere else entirely.
                quality_thresholds={
                    "good_stability_m": self._config.good_stability_m,
                    "bad_stability_m": self._config.bad_stability_m,
                    "good_residual_m": self._config.good_residual_m,
                },
            )
            self._state = STATE_APPLIED
            logger.info(
                "Hand-eye applied to camera twin %s docking offset", self._camera_twin_uuid
            )
            return self.snapshot()

    def fail(self, error: HandEyeError) -> Dict[str, Any]:
        """Move the flow into its error state so the alert can show guidance."""
        with self._lock:
            self._error = error
            self._state = STATE_ERROR
            return self.snapshot()

    # --- state ------------------------------------------------------------

    @property
    def alert_uuid(self) -> Optional[str]:
        return self._alert_uuid

    @alert_uuid.setter
    def alert_uuid(self, value: Optional[str]) -> None:
        self._alert_uuid = value

    @property
    def camera_twin_uuid(self) -> str:
        return self._camera_twin_uuid

    @property
    def board_spec(self) -> Dict[str, Any]:
        """The board override this run was started with, if any.

        Carried into the alert's button payloads so "Restart calibration" restarts
        *this* run rather than a default one -- a wrong board spec is a likely
        reason to be restarting, which is exactly when silently reverting to the
        defaults is most confusing.
        """
        return dict(self._board_spec)

    @property
    def target_samples(self) -> int:
        return self._target_samples

    def touch(self) -> None:
        """Mark an operator action, resetting the idle clock.

        Called once by the driver's step dispatcher rather than from each step:
        that dispatcher is the only path an operator action arrives on, so
        stamping there cannot be forgotten when a new step is added.
        """
        self._last_activity_at = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since the last operator action on this run.

        Lets the caller reap a run someone walked away from: while one is open the
        camera is held, so the twin shows no stream at all.
        """
        return max(0.0, time.monotonic() - self._last_activity_at)

    @property
    def config(self) -> HandEyeFlowConfig:
        """This run's configuration. Read by the presentation layer for the
        button-flow discriminator and the quality thresholds."""
        return self._config

    @property
    def holds_devices(self) -> bool:
        """True while this run still has the bus or the camera open.

        The flow object outlives its devices: after ``apply`` they are released but
        the flow is kept so the alert's "Done" button has something to resolve. Callers
        deciding whether the devices are free must ask this, not whether a flow exists.
        """
        with self._device_lock:
            return self._bus is not None or self._capture is not None

    @property
    def state(self) -> str:
        return self._state

    @property
    def sample_count(self) -> int:
        """Captures that are usable hand-eye samples — what ``solve`` works from."""
        return self._session.sample_count if self._session is not None else 0

    @property
    def captured_count(self) -> int:
        """Everything the operator has captured, buffered captures included.

        While intrinsics are still being solved a capture has no board pose yet,
        so it is not a *sample* — but it is not lost either, and counting only
        samples leaves the progress indicator stuck at zero for the first several
        captures of every run. Falls back to the sample count on an older SDK
        that has no ``captured_count``.
        """
        if self._session is None:
            return 0
        return int(getattr(self._session, "captured_count", self._session.sample_count))

    def snapshot(self) -> Dict[str, Any]:
        """The ``metadata.hand_eye`` blob describing the flow right now."""
        with self._lock:
            result = self._result
            meta: Dict[str, Any] = {
                "state": self._state,
                "camera_twin_uuid": self._camera_twin_uuid,
                "fk_frame": self._fk_frame,
                # Progress the operator sees. Every accepted capture counts: a view
                # showing too little of the board is refused outright, so this only
                # advances on captures a solve can use.
                "samples_captured": self.captured_count,
                # What ``solve`` can actually work from. Equal to
                # ``samples_captured`` now that the floors are enforced at capture;
                # kept distinct so a divergence stays visible rather than assumed
                # away.
                "samples_usable": self.sample_count,
                "samples_target": self._target_samples,
                "min_samples": MIN_SAMPLES,
                "last_capture": self._last_capture,
                **(
                    {"last_capture_detail": dict(self._last_capture_detail)}
                    if self._last_capture_detail
                    else {}
                ),
                "board": self._board.to_metadata(),
                "intrinsics_source": self._intrinsics_source,
            }
            if self._intrinsics is not None:
                meta["intrinsics"] = dict(self._intrinsics)
            if self._frame_shape is not None:
                # The mode the camera granted, not the one requested -- V4L2
                # substitutes silently and the difference decides whether the board
                # decodes at all.
                meta["capture_size"] = [self._frame_shape[1], self._frame_shape[0]]
            if self._capture_warning:
                meta["capture_warning"] = self._capture_warning
            # Carried but not rendered -- the dashboard shows preflight_findings.
            # This is the prose form, kept because the alert is the persisted
            # record of the run, the same reason intrinsics and method ride along.
            if self._preflight_warning:
                meta["preflight_warning"] = self._preflight_warning
            if self._preflight_findings:
                meta["preflight_findings"] = list(self._preflight_findings)
            if result is not None:
                stability_m = stability_from_result(result)
                meta.update(
                    {
                        "method": result.method,
                        "sample_count": result.sample_count,
                        "residual_translation_m": round(result.residual_translation_m, 6),
                        "max_residual_translation_m": round(
                            result.max_residual_translation_m, 6
                        ),
                        "residual_rotation_deg": round(result.residual_rotation_deg, 4),
                        "max_residual_rotation_deg": round(
                            result.max_residual_rotation_deg, 4
                        ),
                        "verdict": verdict(
                            result.residual_translation_m,
                            stability_m,
                            config=self._config,
                        ),
                    }
                )
                if stability_m is not None:
                    meta["stability_m"] = round(stability_m, 6)
                # Accuracy of fit, reported alongside stability because they answer
                # different questions: how well the transform reconciles the
                # observed motions, versus whether the answer holds up when the
                # evidence changes. ``getattr`` so an older SDK still solves.
                for key in (
                    "relative_rotation_deg",
                    "relative_translation_m",
                    "relative_translation_ratio",
                ):
                    value = getattr(result, key, None)
                    if value is not None:
                        meta[key] = round(float(value), 6)
            if self._error is not None:
                meta["error"] = self._error.code
                meta["error_description"] = self._error.message
            return meta


def resolve_docked_fk_frame(
    camera_twin: Any,
    *,
    arm_twin_uuid: str,
    requested_frame: Optional[str] = None,
) -> str:
    """The link a camera twin is docked to, validated as calibratable.

    Read the dock off the twin rather than trusting a command payload. The solved
    transform is expressed relative to whichever link the camera is docked to, so
    a wrong ``fk_frame`` produces a wrong answer with a *good* residual — nothing
    downstream can detect it, which makes this a silent-corruption guard rather
    than a convenience.

    *requested_frame*, when given, is accepted only if it agrees with the twin;
    otherwise it is ignored with a warning. That is deliberate: a payload is not
    a source of truth about where hardware is bolted.

    Raises:
        HandEyeError: with code ``docking_unreadable`` when the twin's fields
            cannot be read, ``not_docked_here`` when the camera belongs to another
            twin (or none), or ``no_attach_link`` when it is docked to the arm but
            not to a specific link.
    """
    try:
        attach_parent = str(camera_twin._data_get("attach_to_twin_uuid") or "").strip()
        attach_link = str(camera_twin._data_get("attach_to_link") or "").strip()
    except Exception as exc:
        raise HandEyeError(
            "docking_unreadable",
            f"Could not read the docking fields from the camera twin: {exc}. Check "
            "the twin still exists and the edge can reach the platform, then try "
            "again.",
        ) from exc

    if attach_parent != str(arm_twin_uuid):
        raise HandEyeError(
            "not_docked_here",
            f"Camera twin is docked to {attach_parent or 'nothing'}, not to this "
            "arm. Dock the camera to this arm before calibrating -- the measured "
            "offset is written to the link it is actually mounted on.",
        )
    if not attach_link:
        raise HandEyeError(
            "no_attach_link",
            "Camera twin is docked to this arm but not to a specific link, so "
            "there is no frame to measure the mount against. Pick the link the "
            "camera is physically mounted on, then calibrate.",
        )

    if requested_frame and requested_frame != attach_link:
        logger.warning(
            "Ignoring requested fk_frame %r: the camera twin is docked to %r",
            requested_frame,
            attach_link,
        )
    return attach_link


class HandEyeRunner:
    """Owns the at-most-one active run for a robot, and its guided alert.

    A driver holds one of these instead of module-global flow state. It is the
    layer between "an operator pressed a button" and the flow's own
    capture/solve/apply, and it keeps the alert in step with every transition.

    Not a :class:`~cyberwave.driver.base.BaseDriver` mixin on purpose: the drivers
    that need hand-eye today are plain command dispatchers, and a mixin would only
    serve a base class none of them use. Wrapping this in one later is ~30 lines.

    Args:
        client: Cyberwave client, used for the twin's alert API.
        arm_twin_uuid: Twin the alert hangs off. Note the dashboard dispatches a
            button press to *this* twin's command topic, not the camera's.
        button_flow: ``payload.flow`` value this runner answers to, so a driver
            with several button-driven flows can tell them apart.
        idle_timeout_s: How long a run may sit untouched before
            :meth:`reap_if_idle` ends it. While a run is open the camera is held,
            so the twin shows no stream at all.
        on_finished: Called after a run ends or aborts, to put the robot back the
            way hand-eye found it (resume the preempted operation, restart idle
            streaming). Never called for a failure that claimed no devices.
    """

    def __init__(
        self,
        *,
        client: Any,
        arm_twin_uuid: str,
        button_flow: str,
        idle_timeout_s: float = 900.0,
        on_finished: Optional[Callable[[], None]] = None,
    ) -> None:
        self._client = client
        self._arm_twin_uuid = str(arm_twin_uuid)
        self._button_flow = str(button_flow)
        self._idle_timeout_s = float(idle_timeout_s)
        self._on_finished = on_finished
        self._flow: Optional[HandEyeFlow] = None
        self._lock = threading.RLock()

    # --- state ------------------------------------------------------------

    @property
    def flow(self) -> Optional[HandEyeFlow]:
        with self._lock:
            return self._flow

    def is_running(self) -> bool:
        """Whether a run currently holds the devices.

        Asks the flow, not merely whether one exists: after ``apply`` the flow is
        kept so the alert's "Done" button has something to resolve, but its bus
        and camera are already released and the twin may stream again.
        """
        with self._lock:
            flow = self._flow
        return bool(flow is not None and flow.holds_devices)

    def start(self, flow: HandEyeFlow) -> None:
        """Adopt *flow* as the active run and publish its first alert.

        Ends any previous run first — restarting is a normal operator action, and
        two runs must never hold the same devices.
        """
        with self._lock:
            previous = self._flow
        if previous is not None:
            logger.info("Hand-eye calibration already running; restarting it")
            # resume=False: this method is by definition about to adopt a
            # replacement run whose devices the caller has already claimed, so
            # resuming the preempted controller and the idle camera streams here
            # would restart them onto devices that are being taken again.
            self.end(resume=False)

        with self._lock:
            self._flow = flow
        self.publish_alert()

    def release(self) -> Optional[HandEyeFlow]:
        """Hand back the devices of a run that still holds them, and detach it.

        Callers use this to free the bus and camera so something else can claim
        them — it deliberately leaves the alert alone, because whoever *ends* the
        run owns resolving it.

        An **applied** run is left attached on purpose. Its devices are already
        closed by that point, so there is nothing to hand back; detaching it would
        only lose the one reference that can still resolve its alert, leaving the
        operator's "Done" press with nothing to act on and the alert stuck showing
        "Waiting...". :meth:`end` bypasses this via ``force=True``.
        """
        return self._release(force=False)

    def _release(self, *, force: bool) -> Optional[HandEyeFlow]:
        with self._lock:
            flow = self._flow
            if flow is None:
                return None
            if not force and not flow.holds_devices:
                # Nothing to release, and detaching would orphan the alert.
                return None
            self._flow = None
        return flow

    def end(self, *, resolve_alert: bool = True, resume: bool = True) -> None:
        """End the active run: release its devices, resolve its alert, resume.

        Ends an applied run too — that is exactly what the alert's "Done" button
        does — so this detaches unconditionally rather than through
        :meth:`release`'s applied-run guard.

        ``resume=False`` is for a caller already on its way to starting something
        else: ``on_finished`` would start a competing operation and restart camera
        streams on the device just freed.
        """
        flow = self._release(force=True)
        if flow is None:
            return
        alert_uuid = flow.alert_uuid
        flow.close()
        if resolve_alert and alert_uuid:
            try:
                self._client.twin(twin_id=self._arm_twin_uuid).alerts.get(
                    alert_uuid
                ).resolve()
            except Exception:
                logger.debug("Could not resolve the hand-eye alert", exc_info=True)
        if resume:
            self._finished()

    def reap_if_idle(self) -> bool:
        """End the run if nobody has touched it for *idle_timeout_s*.

        Returns whether a run was reaped. Meant to be called from a periodic tick.
        """
        with self._lock:
            flow = self._flow
        if flow is None or not flow.holds_devices:
            return False
        if flow.idle_seconds() < self._idle_timeout_s:
            return False
        logger.info(
            "Ending hand-eye calibration after %.0fs idle -- it holds the camera",
            flow.idle_seconds(),
        )
        self.end()
        return True

    # --- operator actions -------------------------------------------------

    def handle_button(self, data: Mapping[str, Any]) -> bool:
        """Route one alert button press. Returns whether it was ours.

        ``start``/``restart`` are not handled here: they need the driver's own
        config and device discovery, so the driver keeps that entry point and this
        returns ``False`` for them, leaving its dispatcher to act.
        """
        if not isinstance(data, Mapping):
            return False
        if data.get("flow") != self._button_flow:
            return False

        action = str(data.get("action") or "").strip().lower()
        if action == "cancel":
            self.end()
            return True
        if action in {"capture", "solve", "apply"}:
            self.step(action)
            return True
        # start/restart belong to the driver; say so rather than swallowing them.
        return False

    def step(self, action: str, *, solved_at: Optional[str] = None) -> None:
        """Run one capture/solve/apply against the active flow and republish.

        Every path republishes the alert, including the failure paths. That is
        load-bearing: the dashboard clears a pressed button's "Waiting..." state
        only when the alert's ``updated_at`` changes, so a step that returns
        without publishing leaves every button disabled for good.
        """
        with self._lock:
            flow = self._flow
        if flow is None:
            logger.warning("Hand-eye %r with no active run; ignoring", action)
            return

        # One place an operator action arrives, so the idle clock resets here
        # rather than inside each step.
        flow.touch()

        try:
            if action == "capture":
                flow.capture_sample()
            elif action == "solve":
                flow.solve()
            elif action == "apply":
                flow.apply(solved_at=solved_at)
            else:
                logger.debug("Unknown hand-eye action %r", action)
                return
        except HandEyeError as exc:
            logger.warning("Hand-eye %s failed (%s): %s", action, exc.code, exc.message)
            flow.fail(exc)
        except Exception as exc:
            logger.exception("Hand-eye %s raised", action)
            flow.fail(HandEyeError("unexpected", str(exc)))

        self.publish_alert()

        # Applying is terminal: the offset is written, so release the devices and
        # let the twin stream again. The alert stays up so the operator sees the
        # result; its "Done" button resolves it and clears the run.
        if flow.state == STATE_APPLIED:
            flow.close()
            self._finished()

    # --- alert ------------------------------------------------------------

    def publish_alert(self) -> None:
        """Create or update the run's alert to match its current state.

        Updated in place rather than one alert per step: a dozen captures would
        otherwise leave a dozen alerts behind.
        """
        with self._lock:
            flow = self._flow
        if flow is None:
            return

        meta_hand_eye = flow.snapshot()
        name, description, severity = describe_state(flow, meta_hand_eye)
        metadata = {"hand_eye": meta_hand_eye, "buttons": buttons_for_state(flow)}

        try:
            robot = self._client.twin(twin_id=self._arm_twin_uuid)
            if flow.alert_uuid:
                alert = robot.alerts.get(flow.alert_uuid)
                # Severity travels with the state: a failed run must escalate, and
                # a successful one must stop looking like it wants attention.
                alert.update(
                    name=name,
                    description=description,
                    severity=severity,
                    metadata=metadata,
                )
                return

            alert = robot.alerts.create(
                name=name,
                description=description,
                severity=severity,
                alert_type=ALERT_TYPE,
                metadata=metadata,
            )
            flow.alert_uuid = alert.uuid
            # Re-stamp the buttons now that the payloads can carry the alert uuid.
            alert.update(
                metadata={
                    "hand_eye": meta_hand_eye,
                    "buttons": buttons_for_state(flow),
                }
            )
        except Exception:
            logger.exception("Failed to publish the hand-eye calibration alert")

    def _finished(self) -> None:
        if self._on_finished is None:
            return
        try:
            self._on_finished()
        except Exception:
            logger.exception("Hand-eye on_finished callback raised")

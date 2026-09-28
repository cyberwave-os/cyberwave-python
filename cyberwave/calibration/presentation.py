"""Operator-facing rendering of a hand-eye run: alert copy, buttons, verdict.

Pure functions of a flow snapshot. No client, no devices, no globals — and no
numpy or cv2 at import, so a node can render an alert on a machine that cannot
run the solve.

These produce the ``metadata.hand_eye`` blob's *presentation*, and that blob is a
cross-component contract: the dashboard renders driver-declared state and
implements no calibration logic of its own. It gates on
``alert_type == "handeye_calibration"``, reads the blob by key, and branches on
specific values — ``state``, ``last_capture == "not_detected"``, ``verdict`` in
good/marginal/bad, ``intrinsics_source`` in env/stored/solved. Renaming a key or
changing one of those values breaks the UI *silently*: the blob still arrives, it
just stops rendering. Two fields are load-bearing in non-obvious ways —
``residual_translation_m`` gates the entire metrics block including the stability
readout, and ``capture_size`` is indexed positionally as a two-element array.

Button order is part of the contract too: a press comes back as an *index*, not a
payload. And a label containing "restart" makes the dashboard resolve the alert as
a side effect, so labels are behaviour, not decoration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config import DEFAULT_TARGET_SAMPLES, HandEyeFlowConfig
from .handeye import MIN_SAMPLES

#: Flow states, carried as ``metadata.hand_eye.state``.
STATE_CAPTURING = "capturing"
STATE_SOLVED = "solved"
STATE_APPLIED = "applied"
STATE_ERROR = "error"

#: ``alert_type`` the dashboard matches to render this flow at all.
ALERT_TYPE = "handeye_calibration"

#: Corners a *diagnostic* re-detection must find before its result is treated as
#: evidence about what went wrong — enough to be sure the board was really
#: located and not a few stray corner hits. Deliberately not the session's own
#: corner floor, which is enforced inside the solve and needs no restating here.
TRANSPOSE_EVIDENCE_CORNERS = 6

#: Modules across a DICT_4X4 marker, including its mandatory one-module border.
MARKER_MODULES = 6

#: Pixels per module the ArUco detector needs before decoding gets unreliable.
MIN_PX_PER_MODULE = 3.0


#: ``CAP_PROP_AUTO_EXPOSURE`` values that unambiguously mean "auto". V4L2 itself
#: uses 3 for aperture-priority (auto) and 1 for manual; OpenCV's other backends
#: report the same pair normalised to 0.75 and 0.25. Every other value -- 0 above
#: all, which is equally what an unimplemented property returns -- is not evidence
#: of anything, and this check stays silent on it rather than guessing.
AUTO_EXPOSURE_ON_VALUES = (3.0, 0.75)


#: Each entry is one thing the operator can fix, as ``(label, long, v4l2_control)``.
#: The label is a bare noun so the summary can inflect it; the scannable line the dashboard shows in amber next to the
#: capture counter; the long form carries the reason into the alert description.
#: Both come from here so the two never drift apart and the UI holds no camera
#: knowledge of its own.
def _preflight_problems(
    settings: Optional[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    """The camera settings that will corrupt this session, worst first.

    One entry per thing the operator can actually change, carrying every wording
    the UI needs: ``label`` names the control, ``why_short`` is the one-line
    consequence for the alert body, ``why_long`` the fuller sentence for the
    description, and ``control`` the ``v4l2-ctl`` argument that fixes it. All of
    it lives here so the dashboard holds no camera knowledge of its own and the
    two renderings can never drift apart.
    """
    if not settings:
        return []

    problems: List[Dict[str, str]] = []
    # Affirmatively 1, not merely truthy. ``CAP_PROP_AUTOFOCUS`` is a plain 0/1
    # flag, but the reading reaches here from an adapter this package ships
    # independently of, and every other value it could carry -- OpenCV's negative
    # "property unavailable" sentinel above all -- means "no answer", not "on".
    # Warning on those accuses a camera that has no autofocus control of leaving
    # it switched on. 0 remains ambiguous (off, or unsupported) either way.
    autofocus = settings.get("autofocus")
    if autofocus is not None and float(autofocus) == 1.0:
        problems.append(
            {
                "label": "Autofocus",
                "why_short": (
                    "The lens can refocus between samples, changing the intrinsics "
                    "the solve fits."
                ),
                "why_long": (
                    "autofocus is on, so the lens can refocus between samples and "
                    "change the very intrinsics the solve fits"
                ),
                "control": "focus_automatic_continuous=0",
            }
        )
    auto_exposure = settings.get("auto_exposure")
    if auto_exposure is not None and float(auto_exposure) in AUTO_EXPOSURE_ON_VALUES:
        problems.append(
            {
                "label": "Auto-exposure",
                "why_short": (
                    "Brightness and motion blur change between samples, so the board "
                    "detector gets a moving target."
                ),
                "why_long": (
                    "auto-exposure is on, so brightness and motion blur change "
                    "between samples and the board detector gets a moving target"
                ),
                "control": "auto_exposure=1",
            }
        )
    return problems


def preflight_findings(
    settings: Optional[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    """One rendered finding per misconfigured control, for the alert body.

    Each entry is ``{title, why, fix}``, already worded -- the dashboard shows
    them as warning blocks and does no wording of its own. ``fix`` is a complete
    command naming the real device where the readout supplied one, so it can be
    pasted rather than reconstructed.
    """
    problems = _preflight_problems(settings)
    if not problems:
        return []
    device = str((settings or {}).get("device") or "").strip() or "<device>"
    return [
        {
            "title": f"{p['label']} is on",
            "why": p["why_short"],
            "fix": f"v4l2-ctl -d {device} -c {p['control']}",
        }
        for p in problems
    ]


def preflight_warning(settings: Optional[Mapping[str, Any]]) -> Optional[str]:
    """What is wrong with the camera *before* the operator spends a session on it.

    Hand-eye fits one camera model across every sample, so a lens that refocuses
    or an exposure that rebalances midway invalidates the intrinsics the solve
    rests on. Both are defaults on consumer webcams. Caught during the run this
    costs the operator twelve repositions; caught here it costs a checkbox.

    A warning, never a refusal, for the same reason the read-back is untrustworthy:
    V4L2 answers 0 both for "this control is off" and "there is no such control",
    so a hard gate would block perfectly good HTTP MJPEG and RealSense sources that
    simply have nothing to report. Silence here means "nothing provable is wrong",
    not "all clear" -- which is why the drift check during the run stays.

    This is the description wording. The alert body renders
    :func:`preflight_findings` instead, which says the same thing in blocks.

    Returns the operator-facing text, or ``None``.
    """
    problems = _preflight_problems(settings)
    if not problems:
        return None

    device = str((settings or {}).get("device") or "").strip() or "<device>"
    fix = ",".join(p["control"] for p in problems)
    subject = " and ".join(p["label"].lower() for p in problems)
    verb = "are" if len(problems) == 2 else "is"
    # One sentence, no markdown: the alert renders this as a plain paragraph, so
    # backticks print as backticks and newlines collapse. The reasoning lives in
    # preflight_findings, which the body shows as separate warning blocks; saying
    # it twice pushed the capture instructions off the bottom of the card.
    # The command sits mid-sentence because describe_state appends the capture
    # instructions after this: in one plain paragraph there is no position where a
    # command ends cleanly. A copyable command is what preflight_findings is for,
    # where it gets its own element; here readability wins.
    return (
        f"Before you start: {subject} {verb} on and will corrupt the result. "
        f"Fix with v4l2-ctl -d {device} -c {fix} and restart the calibration."
    )


def capture_mode_warning(
    board: Any,
    frame_shape: Tuple[int, ...],
    intrinsics: Optional[Dict[str, float]],
    *,
    working_distance_m: float,
    requested_size: Tuple[int, int],
) -> Optional[str]:
    """Whether the mode the camera *granted* can decode this board, or ``None``.

    V4L2 silently substitutes the nearest supported mode, so the size actually
    granted can be well below the size requested -- and the consequence is not a
    visibly broken image, it is a marker that stops decoding somewhere out at
    working distance. That surfaces to the operator as "board not found" on a
    frame that looks perfectly fine, which sends them to re-light and re-aim a
    setup whose only problem is resolution.

    A warning, never a refusal: the estimate below is deliberately coarse (one
    assumed working distance, and a focal length guessed from the frame width when
    none is known yet), and it is downstream of a real detector that gets the
    final say. Being told "this mode is marginal for your board" is useful; being
    stopped by an estimate is not.

    Returns the operator-facing text, or ``None`` when the mode is adequate or
    cannot be judged.
    """
    marker_size_m = getattr(board, "marker_size_m", None)
    if not marker_size_m:
        # A checkerboard has no markers to decode, so this check does not apply.
        return None

    height_px, width_px = int(frame_shape[0]), int(frame_shape[1])
    if height_px <= 0 or width_px <= 0:
        return None

    # Prefer a real focal length. With none yet -- the common case, since
    # intrinsics are usually solved from these very captures -- approximate it
    # from the frame width, which for this class of webcam lands near the ~1030 px
    # measured at 1280x720 and scales with the granted mode.
    focal_px = float((intrinsics or {}).get("fy") or 0.0) or float(width_px) * 0.8
    marker_px = focal_px * float(marker_size_m) / working_distance_m
    px_per_module = marker_px / MARKER_MODULES
    if px_per_module >= MIN_PX_PER_MODULE:
        return None

    # The distance at which this mode *does* decode, so the guidance is a number
    # the operator can act on rather than "move closer".
    usable_m = (
        focal_px
        * float(marker_size_m)
        / (MARKER_MODULES * MIN_PX_PER_MODULE)
    )
    requested_w, requested_h = requested_size
    return (
        f"The camera granted {width_px}x{height_px} (requested "
        f"{requested_w}x{requested_h}). At that size this board's "
        f"{float(marker_size_m) * 1000:.0f} mm markers span about "
        f"{px_per_module:.1f} pixels per module at "
        f"{working_distance_m * 100:.0f} cm, below the ~"
        f"{MIN_PX_PER_MODULE:.0f} the detector needs. Work within about "
        f"{usable_m * 100:.0f} cm of the board, print a larger one, or free up USB "
        "bandwidth so the camera can grant the full resolution."
    )


def stability_from_result(result: Any) -> Optional[float]:
    """Largest per-axis leave-one-out standard deviation, in metres.

    ``None`` when the solve carried no leave-one-out: validation was skipped, or
    the set is ill-conditioned enough that dropping any sample leaves no solvable
    subset. :func:`verdict` treats that as "cannot say" rather than as a pass.
    """
    loo = getattr(result, "leave_one_out", None)
    if loo is None:
        return None
    value = getattr(loo, "max_stddev_m", None)
    return None if value is None else float(value)


def verdict(
    residual_m: Optional[float],
    stability_m: Optional[float],
    *,
    config: HandEyeFlowConfig,
) -> str:
    """Coarse quality label for the UI, keyed on leave-one-out stability.

    *stability_m* is the largest per-axis standard deviation of the mount position
    across leave-one-out re-solves. It is the primary gate because it answers the
    question the operator actually has -- "will this number hold?" -- by measuring
    how much the answer moves when the evidence changes. The residual answers a
    weaker question: whether the observations agree with each other. Those come
    apart in practice, and when they disagree stability is the one that matched
    reproducibility across independent sessions on this arm.

    The residual is kept as a secondary check for a grossly bad fit.

    A "good" verdict means "nothing detectably wrong", not "accurate to this
    number". Neither statistic can see a systematically mislabelled input.
    """
    if stability_m is None:
        # No leave-one-out to key on, from either of two causes:
        #
        # * a short run -- ``MIN_LEAVE_ONE_OUT_SAMPLES`` is ``MIN_SAMPLES + 1``
        #   (4) while ``solve`` admits 3, so a minimal three-capture run lands
        #   here as a matter of course, not because anything is wrong;
        # * an ill-conditioned set, where the subsets themselves would not solve.
        #
        # Neither is distinguishable here, so fall back to the residual and cap at
        # "marginal": without a stability figure there is no evidence for "good",
        # and a set that cannot survive losing one observation does not deserve it.
        if residual_m is None:
            return "unknown"
        if residual_m > config.good_residual_m * 3:
            return "bad"
        return "marginal"

    if stability_m > config.bad_stability_m:
        return "bad"
    if stability_m > config.good_stability_m:
        return "marginal"
    # Stable, but check the residual has not blown up: a set can be internally
    # stable and still fit poorly.
    if residual_m is not None and residual_m > config.good_residual_m * 6:
        return "marginal"
    return "good"


def build_button(label: str, action: str, flow: Any) -> Dict[str, Any]:
    """One ``metadata.buttons`` entry routed back to this flow.

    Carries the run's own board spec and sample target, not just the camera twin:
    ``restart`` re-enters ``_handle_handeye_start`` with this payload as its
    ``data``, so anything omitted here is silently replaced by a default on
    restart.
    """
    payload: Dict[str, Any] = {
        "flow": flow.config.button_flow,
        "action": action,
        "camera_twin_uuid": flow.camera_twin_uuid,
        "target_samples": flow.target_samples,
    }
    board_spec = flow.board_spec
    if board_spec:
        payload["board"] = board_spec
    if flow.alert_uuid:
        payload["alert_uuid"] = flow.alert_uuid
    return {"label": label, "payload": payload}


def buttons_for_state(flow: Any) -> List[Dict[str, Any]]:
    """The actions that make sense in the flow's current state."""
    state = flow.state
    if state == STATE_ERROR:
        return [
            build_button("Restart calibration", "restart", flow),
            build_button("Cancel", "cancel", flow),
        ]
    if state == STATE_APPLIED:
        return [build_button("Done", "cancel", flow)]
    if state == STATE_SOLVED:
        return [
            build_button("Apply to docking offset", "apply", flow),
            build_button("Capture more", "capture", flow),
            # "Discard and exit", not "Discard": this ends the whole run and drops
            # every capture with it. The bare label sat next to "Capture more" and
            # read as "discard the solve, keep my captures", which is the one thing
            # it does not do.
            build_button("Discard and exit", "cancel", flow),
        ]

    buttons = [build_button("Capture sample", "capture", flow)]
    if flow.sample_count >= MIN_SAMPLES:
        buttons.append(build_button("Solve", "solve", flow))
    buttons.append(build_button("Cancel", "cancel", flow))
    return buttons


def describe_state(flow: Any, meta: Dict[str, Any]) -> Tuple[str, str, str]:
    """``(name, description, severity)`` for the alert in its current state."""
    state = flow.state
    captured = meta.get("samples_captured", 0)
    target = meta.get("samples_target", DEFAULT_TARGET_SAMPLES)

    if state == STATE_ERROR:
        return (
            "Hand-eye calibration failed",
            meta.get("error_description") or "The calibration could not be completed.",
            "error",
        )

    if state == STATE_APPLIED:
        return (
            "Hand-eye calibration applied",
            "The camera's docking offset now reflects the measured mount. The 3D view "
            "and any exported scene both place it where it really is.",
            "info",
        )

    if state == STATE_SOLVED:
        verdict = meta.get("verdict", "unknown")
        note = {
            "good": "That is a good result.",
            "marginal": (
                "That is borderline. Capturing more views with the wrist tilted about "
                "different axes, and at a range of distances from the board, will "
                "tighten it."
            ),
            "bad": (
                "That is too loose to trust. The usual causes, in order: too little "
                "variety in wrist orientation, all captures at a similar distance "
                "from the board, a board that moved, or wrong intrinsics."
            ),
        }.get(verdict, "")
        stability_m = meta.get("stability_m")
        if stability_m is not None:
            # Lead with the number the verdict is keyed on. Quoting the residual
            # here instead invites the operator to judge by a statistic the
            # verdict does not use, which is how a stable answer gets rejected and
            # an unstable one accepted.
            measure = (
                f"Dropping any single capture moves the result by "
                f"{stability_m * 1000.0:.1f} mm."
            )
        else:
            measure = (
                f"The captures disagree by "
                f"{(meta.get('residual_translation_m') or 0.0) * 1000.0:.1f} mm / "
                f"{meta.get('residual_rotation_deg', 0.0):.2f}deg."
            )
        relative_t = meta.get("relative_translation_m")
        relative_r = meta.get("relative_rotation_deg")
        if relative_t is not None and relative_r is not None:
            measure += (
                f" It reconciles the observed motions to "
                f"{relative_t * 1000.0:.1f} mm / {relative_r:.2f}deg."
            )
        return (
            "Hand-eye calibration - review",
            f"Solved from {meta.get('sample_count', captured)} captures. "
            f"{measure} {note}".strip(),
            "info",
        )

    hint = ""
    if meta.get("last_capture") == "not_detected":
        # Point at the actual cause. Zero raw markers and "markers but no board"
        # need opposite fixes, and "not found" alone sends people to re-light a
        # board whose dictionary simply does not match.
        detail = meta.get("last_capture_detail") or {}
        found = detail.get("markers_detected")
        board = meta.get("board") or {}
        if found == 0:
            hint = (
                f" No {detail.get('dictionary', 'ArUco')} markers were found at all. "
                "Either the printed board uses a different dictionary, or it is out "
                "of frame, too far away, or motion-blurred."
            )
        elif isinstance(found, int) and found > 0:
            grid = board.get("squares") or board.get("inner_corners")
            transposed = detail.get("transposed_corners")
            in_range = detail.get("markers_in_board_range")
            if isinstance(transposed, int) and transposed >= TRANSPOSE_EVIDENCE_CORNERS:
                # Swapping the axes located the board, so the configuration is
                # wrong, not the framing. Say so rather than sending the operator
                # to re-aim a camera that was already pointed correctly.
                swapped = list(reversed(list(grid))) if grid else None
                hint = (
                    f" {found} marker(s) were found and the grid appears transposed: "
                    f"the configured square count {grid} finds no board, but "
                    f"{swapped} finds {transposed} corners. Set the board's square "
                    f"count to {swapped} (columns, rows)."
                )
            elif detail.get("reason"):
                # The session already counted exactly what it saw and against
                # which floor. Ordered after the transpose check, which diagnoses a
                # misconfiguration the session cannot see.
                hint = f" {detail['reason']}"
            elif isinstance(in_range, int) and in_range >= TRANSPOSE_EVIDENCE_CORNERS:
                hint = (
                    f" {found} marker(s) were found and {in_range} of them belong to "
                    f"this board, but too little of the grid is placeable. Bring more "
                    f"of the board into view, square-on and in focus."
                )
            else:
                hint = (
                    f" {found} marker(s) were found but not enough of the grid to "
                    f"place the board. Bring more of it into view, or check the "
                    f"square count matches the print ({grid})."
                )
        else:
            hint = (
                " The board was not found in the last capture -- check the whole "
                "board is in frame, in focus and evenly lit."
            )
    return (
        f"Hand-eye calibration - {captured}/{target} captures",
        (
            # Deliberately NOT the preflight text: the alert body renders
            # preflight_findings as its own warning blocks, which is the only
            # place with room for a reason and a copyable command. Repeating it
            # here buried the capture instructions under eight lines of prose.
            "Hand-guide the wrist to a new orientation, let it settle, then click "
            "Capture sample. Tilt it about genuinely different axes rather than "
            "sliding it around: rotation is what makes the solve possible."
            + hint
            # Appended last, and only when the granted mode is marginal. This is the
            # cause that looks exactly like bad framing, so it belongs next to the
            # framing advice rather than in a log line nobody reads.
            + (f" {meta['capture_warning']}" if meta.get("capture_warning") else "")
        ),
        "warning",
    )

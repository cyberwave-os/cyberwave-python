"""Device seams for the edge-side hand-eye flow.

The flow needs exactly two things from the hardware, and neither is specific to
any one robot:

* a **joint source** — the arm's joint angles *now*, read without energising it,
  so the arm stays hand-guidable between captures;
* a **frame source** — an image captured *after* the arm came to rest.

Both are :class:`~typing.Protocol` s, so an existing driver class satisfies one by
shape alone with no base class to inherit and no registration step. That is
deliberate: on the SO-101 driver the joint source is an existing torque-free bus
session class that satisfied this shape with no edits at all.

Why :meth:`FrameSource.grab_fresh_frame` is one method rather than OpenCV's
``grab()``/``read()`` pair: *freshness is the obligation, not the mechanism.* A
V4L2 webcam needs its driver queue drained, because V4L2 hands back whatever is
queued and after a pause that is an image from before the arm moved. A RealSense,
a GigE camera and a ROS image topic each need something different, and some need
nothing. Exposing the cv2 pair would freeze one camera's workaround into the
contract every other camera then has to fake.

Neither protocol is required to be thread-safe. The flow serialises access under
its own device lock, but it does call the teardown methods from a *different*
thread than an in-flight capture on purpose — so a wedged read can be released —
which is why both are documented as idempotent.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class JointSource(Protocol):
    """Reads the arm's joint angles now, without energising it.

    Marked :func:`~typing.runtime_checkable` only so a *test* can assert
    conformance cheaply. The flow never ``isinstance``-gates on it: a structural
    check verifies method names and nothing about their behaviour, so it would
    buy false confidence. Failures surface at the real call instead, as a coded
    error the operator can act on.
    """

    def read_joint_positions(self) -> Mapping[str, float] | None:
        """Angles in radians, keyed by whatever the bus calls its joints.

        The flow renames them to URDF joints through its configured
        ``joint_name_map``, so a bus with real joint names needs no adaptation.

        Returns ``None`` when the bus is gone — the flow treats that as a lost
        device and ends the run — and an empty mapping on a transient empty read.
        """
        ...

    def disconnect(self) -> None:
        """Release the bus. Idempotent, and safe to call from another thread."""
        ...


@runtime_checkable
class FrameSource(Protocol):
    """Provides an image captured after the arm came to rest."""

    def grab_fresh_frame(self) -> Any:
        """One frame that postdates the arm stopping, as a BGR ``ndarray``.

        Implementations own whatever staleness handling their transport needs;
        see the module docstring. Raise rather than returning ``None`` on
        failure, so the flow can surface a coded error.
        """
        ...

    def release(self) -> None:
        """Release the camera. Idempotent, and safe to call from another thread."""
        ...

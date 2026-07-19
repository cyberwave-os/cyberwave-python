from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .compat import math, time

from ..constants import SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE
from ..exceptions import CyberwaveError

from ._helpers import (
    _default_control_source_type,
    _normalize_locomotion_source_type,
    _run_coroutine_blocking,
)
from .base import Twin
from .mixins import (
    CameraCapableMixin,
    FlightCapableMixin,
    GripperCapableMixin,
    JointsCapableMixin,
    LocomotionCapableMixin,
    MotionCapableMixin,
    NavigationCapableMixin,
    PolicyCapableMixin,
    PowerCapableMixin,
    SpatialPoseCapableMixin,
)
from .simulation_support import SimLevel, simulation_level

if TYPE_CHECKING:
    from ..camera import CameraStreamer
    from ..managers.simulations import Simulation

logger = logging.getLogger(__name__)


class CameraTwin(CameraCapableMixin, Twin):
    """
    Twin with camera/sensor capabilities.

    Provides methods for video streaming and frame capture for twins
    that have RGB or depth sensors.

    Example:
        >>> twin = client.twin("unitree/go2")  # Returns CameraTwin if has sensors
        >>> await twin.stream_video_background(fps=15)
        >>> frame = twin.capture_frame()
    """

    _camera_streamer: Optional["CameraStreamer"] = None
    _active_simulation: Optional["Simulation"] = None

    @property
    def default_camera_name(self) -> str:
        """Default sensor/camera id for WebRTC signaling (``sensor`` in offers).

        Uses the first entry in :attr:`Twin.capabilities` ``sensors`` and its ``id``,
        same rule as :func:`cyberwave.sensor.manager._infer_config_from_twin`.
        Falls back to ``"default"`` when missing or empty.
        """
        sid = self._default_imaging_sensor_id()
        return str(sid) if sid is not None else "default"

    def streamer(self) -> "CameraStreamer":
        """Get the camera streamer."""
        if self._camera_streamer is None:
            raise CyberwaveError("Camera streamer not initialized")
        return self._camera_streamer

    def _is_simulation_mode(self) -> bool:
        """True when the client is affecting the simulator (``cw.affect("simulation")``)."""
        return (
            getattr(getattr(self.client, "config", None), "runtime_mode", "live")
            == "simulation"
        )

    def _block_on_simulation(self, sim: "Simulation") -> None:
        """Block until Ctrl+C, then stop ``sim`` and clear the active handle.

        Shared body of the simulation branch of :meth:`start_streaming` for both
        ``CameraTwin`` and ``DepthCameraTwin``.
        """
        print(f"Started simulation {sim.simulation_id}. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                sim.stop()
            finally:
                self._active_simulation = None

    @simulation_level(SimLevel.MUJOCO)
    async def stream_video_background(
        self,
        fps: int = 30,
        camera_id: int | str = 0,
        fourcc: Optional[str] = None,
        camera_name: Optional[str] = None,
        **kwargs,
    ) -> "CameraStreamer | Simulation":
        """Start streaming in the background. Non-blocking.

        In simulation mode consumes the environment's already-running MuJoCo
        simulation (start one with ``cw.affect("sim")`` or
        ``cw.environments.simulations.start(...)``) and returns its
        :class:`~cyberwave.managers.simulations.Simulation` handle. In live mode
        returns the ``CameraStreamer`` as before.

        Args:
            fps: Frames per second (default: 30)
            camera_id: Camera device ID or stream URL (default: 0)
            fourcc: Optional FOURCC for local V4L2/USB cameras (e.g. ``'MJPG'``). If omitted,
                :class:`~cyberwave.sensor.camera_cv2.CV2VideoTrack` tries ``MJPG`` by default.
            camera_name: WebRTC signaling sensor id; defaults to :attr:`default_camera_name`.
            **kwargs: Additional arguments forwarded to :meth:`~cyberwave.client.Cyberwave.video_stream`
                (e.g. ``resolution``, ``keyframe_interval``, ``frame_callback``, ``time_reference``).

        Returns:
            CameraStreamer instance for managing the stream (live mode), or
            Simulation handle (simulation mode).
        """
        if self._is_simulation_mode():
            # _resolve_simulation_stream() may block on wait_until_active; offload
            # it so an "await" here doesn't stall the caller's event loop.
            return await asyncio.to_thread(self._resolve_simulation_stream)

        self._camera_streamer = self.client.video_stream(
            twin_uuid=self.uuid,
            camera_id=camera_id,
            fps=fps,
            fourcc=fourcc,
            camera_name=camera_name or self.default_camera_name,
            **kwargs,
        )
        await self._camera_streamer.start()
        return self._camera_streamer

    async def stop_streaming(self) -> None:
        """Stop the active simulation (sim mode) or the local camera stream."""
        if self._active_simulation is not None:
            sim = self._active_simulation
            self._active_simulation = None
            sim.stop()
            return
        if self._camera_streamer is not None:
            await self._camera_streamer.stop()
            self._camera_streamer = None

    def _resolve_simulation_stream(self) -> "Simulation":
        """Resolve the environment's running MuJoCo simulation for streaming.

        The ``@simulation_level(SimLevel.MUJOCO)`` gate on the streaming
        entrypoints guarantees a MuJoCo simulation is already running (started
        via ``cw.affect("sim")`` or ``cw.environments.simulations.start(...)``);
        this consumes it as the camera-frame producer and stores the handle on
        ``self._active_simulation`` so :meth:`stop_streaming` can stop it. The SDK
        never starts a simulation from a streaming call.
        """
        from ..managers.simulations import running_simulation

        sim = running_simulation(self)
        if sim is not None and sim.status != "running":
            sim.wait_until_active()
        self._active_simulation = sim
        return sim

    @simulation_level(SimLevel.MUJOCO)
    def start_streaming(
        self,
        fps: int = 30,
        camera_id: int | str = 0,
        camera_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Stream video until Ctrl+C. Blocking.

        In simulation mode (``cw.affect("simulation")``) this consumes the
        environment's already-running MuJoCo simulation (its camera streams from
        the simulation) and blocks until Ctrl+C — stopping the simulation on exit.
        Start the simulation first with ``cw.affect("sim")`` or
        ``cw.environments.simulations.start(...)``. In live mode it pushes the
        local camera as before.

        Args:
            fps: Frames per second (default: 30)
            camera_id: Camera device ID or stream URL (default: 0)
            camera_name: WebRTC signaling sensor id; defaults to :attr:`default_camera_name`.
            **kwargs: Additional arguments forwarded to :meth:`~cyberwave.client.Cyberwave.video_stream`
                (e.g. ``fourcc``, ``resolution``, ``keyframe_interval``, ``frame_callback``).
        """
        if self._is_simulation_mode():
            self._block_on_simulation(self._resolve_simulation_stream())
            return

        self._camera_streamer = self.client.video_stream(
            twin_uuid=self.uuid,
            camera_id=camera_id,
            fps=fps,
            camera_name=camera_name or self.default_camera_name,
            **kwargs,
        )

        async def _run():
            await self._camera_streamer.start()
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await self._camera_streamer.stop()
                self._camera_streamer = None

        try:
            _run_coroutine_blocking(_run())
        except KeyboardInterrupt:
            pass
        finally:
            if self._camera_streamer is not None:
                try:
                    _run_coroutine_blocking(self._camera_streamer.stop())
                except Exception:
                    pass
                self._camera_streamer = None

    def __repr__(self) -> str:
        sensors = self.capabilities.get("sensors", [])
        sensor_types = [s.get("type", "unknown") for s in sensors]
        return f"CameraTwin(uuid='{self.uuid}', name='{self.name}', sensors={sensor_types})"


class DepthCameraTwin(CameraTwin):
    """
    Twin with depth camera capabilities.

    Extends CameraTwin with depth-specific methods for point cloud
    generation and depth frame capture.
    """

    _camera_streamer: Optional["CameraStreamer"] = None

    def streamer(self) -> "CameraStreamer":
        """Get the camera streamer."""
        if self._camera_streamer is None:
            raise CyberwaveError("Camera streamer not initialized")
        return self._camera_streamer

    async def stop_streaming(self) -> None:
        """Stop the active simulation (sim mode) or the local camera stream."""
        if self._active_simulation is not None:
            sim = self._active_simulation
            self._active_simulation = None
            sim.stop()
            return
        if self._camera_streamer is not None:
            # The streamer handles cleanup in its stop method
            await self._camera_streamer.stop()
            self._camera_streamer = None

    @simulation_level(SimLevel.MUJOCO)
    async def stream_video_background(
        self,
        fps: int = 30,
        camera_id: int | str = 0,
        fourcc: Optional[str] = None,
        camera_name: Optional[str] = None,
        *,
        enable_depth: bool = True,
        **kwargs,
    ) -> "CameraStreamer | Simulation":
        """Start video streaming in the background. Non-blocking.

        In simulation mode consumes the environment's already-running MuJoCo
        simulation (start one with ``cw.affect("sim")`` or
        ``cw.environments.simulations.start(...)``) and returns its
        :class:`~cyberwave.managers.simulations.Simulation` handle. In live mode
        returns the ``CameraStreamer`` as before.

        Note:
            The simulation is environment-wide, so in simulation mode
            ``enable_depth`` (and the other local-camera kwargs) are ignored —
            what streams is whatever the simulation publishes for this twin.

        Args:
            fps: Frames per second (default: 30)
            camera_id: Camera device ID (default: 0)
            fourcc: Optional FOURCC code (inherited from CameraTwin, unused for RealSense)
            camera_name: WebRTC signaling sensor id; defaults to :attr:`default_camera_name`.
            enable_depth: Enable depth streaming (default: True for DepthCameraTwin).
                Ignored in simulation mode.
            **kwargs: Additional arguments forwarded to :meth:`~cyberwave.client.Cyberwave.video_stream`
                (e.g. ``resolution``, ``keyframe_interval``, ``time_reference``).

        Returns:
            CameraStreamer instance for managing the stream (live mode), or
            Simulation handle (simulation mode).
        """
        if self._is_simulation_mode():
            # _resolve_simulation_stream() may block on wait_until_active; offload
            # it so an "await" here doesn't stall the caller's event loop.
            return await asyncio.to_thread(self._resolve_simulation_stream)

        self._camera_streamer = self.client.video_stream(
            twin_uuid=self.uuid,
            camera_type="realsense",
            camera_id=camera_id,
            fps=fps,
            enable_depth=enable_depth,
            camera_name=camera_name or self.default_camera_name,
            **kwargs,
        )
        await self._camera_streamer.start()
        return self._camera_streamer

    @simulation_level(SimLevel.MUJOCO)
    def start_streaming(
        self,
        fps: int = 30,
        camera_id: int | str = 0,
        enable_depth: bool = True,
        camera_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Stream video until Ctrl+C. Blocking.

        In simulation mode (``cw.affect("simulation")``) this consumes the
        environment's already-running MuJoCo simulation (its camera streams from
        the simulation) and blocks until Ctrl+C — stopping the simulation on exit.
        Start the simulation first with ``cw.affect("sim")`` or
        ``cw.environments.simulations.start(...)``. In live mode it pushes the
        local camera as before.

        Note:
            The simulation is environment-wide, so in simulation mode
            ``enable_depth`` (and the other local-camera kwargs) are ignored —
            what streams is whatever the simulation publishes for this twin.

        Args:
            fps: Frames per second (default: 30)
            camera_id: Camera device ID (default: 0)
            enable_depth: Enable depth streaming (default: True for DepthCameraTwin).
                Ignored in simulation mode.
            camera_name: WebRTC signaling sensor id; defaults to :attr:`default_camera_name`.
            **kwargs: Additional arguments forwarded to :meth:`~cyberwave.client.Cyberwave.video_stream`
                (e.g. ``resolution``, ``keyframe_interval``, ``time_reference``).
        """
        if self._is_simulation_mode():
            self._block_on_simulation(self._resolve_simulation_stream())
            return

        self._camera_streamer = self.client.video_stream(
            twin_uuid=self.uuid,
            camera_type="realsense",
            camera_id=camera_id,
            fps=fps,
            enable_depth=enable_depth,
            camera_name=camera_name or self.default_camera_name,
            **kwargs,
        )

        async def _run():
            await self._camera_streamer.start()
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await self._camera_streamer.stop()
                self._camera_streamer = None

        try:
            _run_coroutine_blocking(_run())
        except KeyboardInterrupt:
            pass
        finally:
            if self._camera_streamer is not None:
                try:
                    _run_coroutine_blocking(self._camera_streamer.stop())
                except Exception:
                    pass
                self._camera_streamer = None

    def _depth_sensor_handle(self) -> Any:
        """Return the DepthSensorHandle for this twin's depth sensor."""
        for entry in self.capabilities.get("sensors", []):
            if isinstance(entry, dict) and str(entry.get("type")) == "depth":
                key = str(entry.get("id") or entry.get("name") or "")
                return self._imaging_handle(sensor_id=key)
        raise CyberwaveError("No depth sensor on this twin")

    def capture_depth_frame(self) -> Any:
        """Deprecated — use ``twin.camera['<id>'].get_frame()``.

        Returns whatever ``get_frame()`` returns for the MQTT raw depth path: a
        numpy ``uint16 H×W`` array (not raw bytes).
        """
        import warnings

        warnings.warn(
            "DepthCameraTwin.capture_depth_frame() is deprecated; use "
            "twin.camera['<id>'].get_frame() (or twin.camera.get_frame())",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._depth_sensor_handle().get_frame(source="mqtt", raw=True)

    def get_point_cloud(self, *, timeout: float = 3.0) -> Any:
        """Deprecated — use ``twin.camera['<id>'].get_pointcloud()``."""
        import warnings

        warnings.warn(
            "DepthCameraTwin.get_point_cloud() is deprecated; use "
            "twin.camera['<id>'].get_pointcloud() (or twin.camera.get_pointcloud())",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._depth_sensor_handle().get_pointcloud(timeout=timeout)

    def __repr__(self) -> str:
        return f"DepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class JointTwin(JointsCapableMixin, PolicyCapableMixin, MotionCapableMixin, Twin):
    """Manipulator twin with joint teleop (stationary arms — not legged locomotion)."""

    def __repr__(self) -> str:
        return f"JointTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteTwin(
    SpatialPoseCapableMixin,
    LocomotionCapableMixin,
    PolicyCapableMixin,
    MotionCapableMixin,
    NavigationCapableMixin,
    PowerCapableMixin,
    Twin,
):
    """
    Twin that can locomote across space.

    Provides methods for locomotion including movement and rotation.

    Note: Flying twins can locomoate AND fly, so a flying twin is a subset of the LocomoteTwin
    """

    def move(self, position: List[float]):
        """
        DEPRECATED: See warning

        Support for move will be dropped in future versions of the SDK
        """
        logger.warning(
            """move() is deprecated as a way to send commands. You have these two options:
                - Use edit_position if you want to edit the digital twin position in your environemnt, in order to reproduce a real environment in Cyberwave
                - Use move_forward or move_backward if you want your robot to navigate the world
            """
        )
        return

    def move_forward(
        self,
        distance: float = 0.3,
        *,
        duration: float = 1.0,
        rate_hz: float = 20.0,
        source_type: Optional[str] = None,
    ) -> None:
        """Move forward via :attr:`locomotion` (speed in m/s, burst + stop)."""
        return self.locomotion.move_forward(
            distance,
            duration=duration,
            rate_hz=rate_hz,
            source_type=source_type,
        )

    def move_backward(
        self,
        distance: float = 0.3,
        *,
        duration: float = 1.0,
        rate_hz: float = 20.0,
        source_type: Optional[str] = None,
    ) -> None:
        """Move backward via :attr:`locomotion` (speed in m/s, burst + stop)."""
        return self.locomotion.move_backward(
            distance,
            duration=duration,
            rate_hz=rate_hz,
            source_type=source_type,
        )

    def turn_left(
        self,
        angle: float = 0.5,
        *,
        duration: float = 1.0,
        rate_hz: float = 20.0,
        source_type: Optional[str] = None,
    ) -> None:
        """Turn left via :attr:`locomotion` (yaw rate in rad/s, burst + stop)."""
        return self.locomotion.turn_left(
            angle,
            duration=duration,
            rate_hz=rate_hz,
            source_type=source_type,
        )

    def turn_right(
        self,
        angle: float = 0.5,
        *,
        duration: float = 1.0,
        rate_hz: float = 20.0,
        source_type: Optional[str] = None,
    ) -> None:
        """Turn right via :attr:`locomotion` (yaw rate in rad/s, burst + stop)."""
        return self.locomotion.turn_right(
            angle,
            duration=duration,
            rate_hz=rate_hz,
            source_type=source_type,
        )

    def rotate(
        self,
        *,
        w: Optional[float] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
    ) -> None:
        """
        DEPRECATED: Use edit_rotation instead
        """
        logger.warning("rotate() is deprecated. Use edit_rotation() instead.")
        self.edit_rotation(yaw=yaw, pitch=pitch, roll=roll)


class FlyingTwin(FlightCapableMixin, LocomoteTwin):
    """
    Twin with flight capabilities (drones, UAVs).

    Inherits from :class:`LocomoteTwin`, so flying twins also expose
    ``move_forward`` / ``move_backward`` / ``turn_left`` /
    ``turn_right`` (plus ``ascend`` / ``descend`` / ``strafe_*`` and
    the ``stop`` zero-axis command) — these are the canonical
    continuous-stick commands and they drive **the real aircraft**
    on every Cyberwave drone driver:

    Aerial-specific methods include takeoff, landing, return-to-home,
    hovering, gimbal control, and the DJI service / safety surface
    (set home, compass calibration, reboot, emergency stop).

    All commands publish on the canonical
    ``{topic_prefix}cyberwave/twin/{uuid}/command`` topic with the
    standard ``{source_type, command, data, timestamp}`` envelope —
    the contract every Cyberwave edge driver uses.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @simulation_level(SimLevel.PLAYGROUND)
    def _send_drone_command(
        self,
        command: str,
        data: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = None,
    ) -> str:
        """
        Publish a single command on the canonical drone-command topic.

        Returns the resolved ``source_type`` so callers can decide
        whether to also persist sim-mode metadata
        (e.g. ``set_hovering_status``) — that's only meaningful when
        the command was sent in ``sim_tele``, since on a live aircraft
        the edge driver owns the metadata.

        Raises:
            ValueError: If the resolved source type is not one of
                ``"tele"`` / ``"sim_tele"``. Mirrors the validation
                applied to ``LocomoteTwin.move_forward`` etc.
        """
        if source_type is None:
            source_type = _default_control_source_type(self.client)
        source_type = _normalize_locomotion_source_type(source_type)
        if source_type not in [SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE]:
            raise ValueError(
                f"Invalid source type '{source_type}' for drone command "
                f"'{command}'. Use cw.affect('simulation') or "
                "cw.affect('real-world'), or pass source_type='sim' / "
                "'sim_tele' / 'tele' directly."
            )

        resolved = self._resolve_topic_and_payload(
            command=command,
            data=dict(data) if data else {},
            source_type=source_type,
        )
        self._publish_resolved(resolved)
        return resolved.source_type  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Flight-phase commands
    # ------------------------------------------------------------------

    def takeoff(
        self,
        altitude: float = 1.0,
        *,
        source_type: Optional[str] = None,
    ) -> None:
        """
        Take off to the specified altitude.

        Args:
            altitude: Target altitude in meters (default: 1.0).
            source_type: ``"sim_tele"``/``"sim"`` for simulation,
                ``"tele"`` for the real aircraft. Falls back to the
                client-level setting from ``cw.affect()``.
        """
        resolved = self._send_drone_command(
            "takeoff",
            data={"altitude": altitude},
            source_type=source_type,
        )
        # In live (tele) mode the edge driver owns the hovering
        # status flag (it flips it once the FC reports motors-on /
        # in-flight); only mirror it in sim mode where there is no
        # driver to do that for us.
        if resolved == SOURCE_TYPE_SIM_TELE:
            self.set_hovering_status(hovering=True, hovering_altitude=altitude)

    def land(self, *, source_type: Optional[str] = None) -> None:
        """
        Land the drone.

        On the DJI Mini driver this triggers ``KeyStartAutoLanding``
        and arms the landing-confirmation listener — if the firmware
        asks the operator to confirm (over water / glass / glossy
        surfaces), a Cyberwave alert is raised and a second
        ``land()`` call from the operator confirms the touchdown.
        """
        resolved = self._send_drone_command("land", source_type=source_type)
        if resolved == SOURCE_TYPE_SIM_TELE:
            self.set_hovering_status(hovering=False)

    def cancel_takeoff(self, *, source_type: Optional[str] = None) -> None:
        """Abort an in-progress automatic takeoff (DJI MSDK ``KeyStopTakeoff``)."""
        self._send_drone_command("cancel_takeoff", source_type=source_type)

    def cancel_landing(self, *, source_type: Optional[str] = None) -> None:
        """Abort an in-progress automatic landing (DJI MSDK ``KeyStopAutoLanding``)."""
        self._send_drone_command("cancel_landing", source_type=source_type)

    def hover(self, *, source_type: Optional[str] = None) -> None:
        """
        Hover in place.

        On a real DJI aircraft this is effectively a no-op at the
        SDK level — the drone hovers automatically when the RC2
        sticks are centred — but it's still useful in ``sim_tele``
        to flip the metadata flag that prevents the simulator from
        applying gravity to the twin.
        """
        resolved = self._send_drone_command("hover", source_type=source_type)
        if resolved == SOURCE_TYPE_SIM_TELE:
            self.set_hovering_status(hovering=True)

    # ------------------------------------------------------------------
    # Return-to-home
    # ------------------------------------------------------------------

    def return_to_home(self, *, source_type: Optional[str] = None) -> None:
        """
        Return to the home location (DJI MSDK ``KeyStartGoHome``).

        Some firmwares prompt the operator to confirm before
        beginning the return flight. The driver surfaces that prompt
        as a Cyberwave alert and a second ``return_to_home()`` call
        confirms it (mirrors the landing-confirmation flow).
        """
        self._send_drone_command("return_to_home", source_type=source_type)

    def cancel_return_to_home(self, *, source_type: Optional[str] = None) -> None:
        """
        Cancel a return-to-home in progress.

        While the firmware is parked on a confirmation prompt this
        routes through ``KeyGoHomeConfirm(false)`` — once the return
        flight is actually under way it flows through
        ``KeyStopGoHome``. The edge driver picks the right SDK call
        based on the current state.
        """
        self._send_drone_command("cancel_return_to_home", source_type=source_type)

    # ------------------------------------------------------------------
    # Service / safety
    # ------------------------------------------------------------------

    def set_home_here(self, *, source_type: Optional[str] = None) -> None:
        """Reset the home location to the aircraft's current GPS position."""
        self._send_drone_command("set_home_here", source_type=source_type)

    def start_compass_calibration(self, *, source_type: Optional[str] = None) -> None:
        """Begin compass calibration."""
        self._send_drone_command("start_compass_calibration", source_type=source_type)

    def stop_compass_calibration(self, *, source_type: Optional[str] = None) -> None:
        """Stop an in-progress compass calibration."""
        self._send_drone_command("stop_compass_calibration", source_type=source_type)

    def reboot(self, *, source_type: Optional[str] = None) -> None:
        """Reboot the aircraft (DJI MSDK ``KeyRebootDevice``)."""
        self._send_drone_command("reboot", source_type=source_type)

    def emergency_stop(self, *, source_type: Optional[str] = None) -> None:
        """
        Best-effort emergency stop.
        """
        self._send_drone_command("emergency_stop", source_type=source_type)

    # ------------------------------------------------------------------
    # Gimbal control
    # ------------------------------------------------------------------

    def gimbal_rotate(
        self,
        *,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        yaw: Optional[float] = None,
        mode: str = "absolute",
        duration: Optional[float] = None,
        source_type: Optional[str] = None,
    ) -> None:
        """
        Rotate the gimbal to a target pitch/roll/yaw.

        Maps to DJI MSDK v5's ``GimbalKey.KeyRotateByAngle``. On the
        Mini 4 Pro only the pitch axis is mechanically controllable
        (range approximately ``[-90°, +30°]``); roll and yaw are
        accepted but the hardware ignores them.

        Args:
            pitch: Target pitch in degrees. Positive = up,
                negative = down. ``None`` leaves it unset (axis is
                not commanded).
            roll: Target roll in degrees, ``None`` for unset.
            yaw: Target yaw in degrees (relative to aircraft heading
                when ``mode="absolute"``), ``None`` for unset.
            mode: ``"absolute"`` (default — angle is interpreted
                relative to the aircraft heading) or ``"relative"``
                (angle is a delta from the current gimbal attitude).
                Anything unrecognised falls back to ``"absolute"``
                on the driver side.
            duration: Rotation duration in seconds, ``None`` to use
                the SDK default. Useful for cinematic moves.
            source_type: ``"tele"`` / ``"sim_tele"`` (auto-resolved
                from ``cw.affect()`` if omitted).

        Example::

            drone.gimbal_rotate(pitch=-45.0, duration=2.0)   # tilt down 45°
            drone.gimbal_rotate(pitch=10.0, mode="relative")  # +10° from current
        """
        # Build only the fields the user actually set so the driver
        # can distinguish "leave this axis alone" (key absent) from
        # "command axis to 0" (key=0).
        data: Dict[str, Any] = {}
        if pitch is not None:
            data["pitch"] = float(pitch)
        if roll is not None:
            data["roll"] = float(roll)
        if yaw is not None:
            data["yaw"] = float(yaw)
        if duration is not None:
            # `duration` is the documented wire field; the driver
            # also accepts `time` and `duration_sec` as aliases.
            data["duration"] = float(duration)
        # Always include `mode` so the driver doesn't have to fall
        # back to its own default and the wire payload stays
        # self-describing for log diffs.
        data["mode"] = mode

        self._send_drone_command("gimbal_rotate", data=data, source_type=source_type)

    def gimbal_recenter(self, *, source_type: Optional[str] = None) -> None:
        """
        Recenter the gimbal to pitch=0 / mode=absolute.

        Convenience wrapper around :meth:`gimbal_rotate` matching
        the keyboard "Recenter Gimbal" binding (``;`` key on
        ``controller:dji-keyboard:v1`` — held ``J`` / ``N`` drive
        the gimbal up / down via :meth:`gimbal_rotate_speed`).
        """
        self.gimbal_rotate(pitch=0.0, mode="absolute", source_type=source_type)

    def pan_camera(
        self,
        angle_deg: float,
        *,
        yaw_rate_deg_s: float = 30.0,
        refresh_hz: float = 5.0,
        source_type: Optional[str] = None,
    ) -> None:
        """
        Pan the camera view by yawing the aircraft.

        On drones with a multi-axis gimbal (Matrice 30, Mavic 3
        Enterprise, …) gimbal yaw rotates the camera directly via
        :meth:`gimbal_rotate`. The Mini 4 Pro (and the rest of the
        Mini / Mavic Mini line) ships with a **pitch-only** gimbal:
        the mechanism that would rotate the camera in yaw simply
        isn't there. The only way to change the camera's heading is
        to yaw the airframe — that's what this helper does, on top
        of the off-RC teleoperation surface introduced by the
        Virtual Stick release.

        Implementation: re-publishes ``turn_left`` / ``turn_right``
        at ``refresh_hz`` for the duration needed to cover
        ``angle_deg`` at ``yaw_rate_deg_s``, then explicitly zeros
        the yaw axis. The refresh cadence has to stay inside the
        DJI Android driver's 500 ms command-stale watchdog; anything
        below 2 Hz risks the watchdog snapping the target to zero
        mid-pan.

        Args:
            angle_deg: Target rotation in degrees. Positive yaws
                the aircraft (and camera view) **left** /
                counter-clockwise from above; negative yaws right /
                clockwise. ``0`` is a no-op.
            yaw_rate_deg_s: Yaw rate in degrees per second
                (default ``30°/s`` — a gentle pan suitable for
                video). Must be positive.
            refresh_hz: Re-send cadence while the pan is in flight
                (default ``5 Hz``). Must be > 2 Hz to stay inside
                the 500 ms command-stale watchdog on the DJI
                Android driver.
            source_type: ``"tele"`` / ``"sim_tele"`` (auto-resolved
                from ``cw.affect()`` if omitted).

        Example::

            drone.takeoff(altitude=2.0)
            time.sleep(4)
            drone.gimbal_rotate(pitch=-90.0)   # camera straight down
            drone.pan_camera(angle_deg=90.0)   # aircraft yaws 90° left
            drone.land()

        Note: On the DJI Android driver, off-RC teleop is opt-in per
        twin via ``metadata.drivers.default.virtual_stick = true``.
        Without that flag the driver rejects the underlying
        ``turn_left`` / ``turn_right`` commands with a ``failed``
        MQTT status — :meth:`pan_camera` will publish, but the
        aircraft won't move.
        """
        if abs(angle_deg) < 1e-6:
            return
        if yaw_rate_deg_s <= 0:
            raise ValueError(f"yaw_rate_deg_s must be positive (got {yaw_rate_deg_s})")
        if refresh_hz <= 2:
            raise ValueError(
                f"refresh_hz must be > 2 Hz to stay inside the 500 ms "
                f"command-stale watchdog (got {refresh_hz})"
            )

        yaw_rate_rad_s = math.radians(yaw_rate_deg_s)
        total_duration_s = abs(angle_deg) / yaw_rate_deg_s
        period_s = 1.0 / refresh_hz

        turn_fn = self.turn_left if angle_deg > 0 else self.turn_right

        # Publish-then-check ordering: a single quick pan still gets at
        # least one fresh axis update before we zero out, and on a
        # multi-tick pan the loop hits the watchdog window on every
        # iteration rather than going stale just before the last sample.
        t_start = time.monotonic()
        while True:
            # Single MQTT pulse per refresh (``duration=0``); pan_camera owns
            # cadence and the final zero — not the locomotion burst+stop path.
            turn_fn(
                angle=yaw_rate_rad_s,
                source_type=source_type,
                duration=0,
            )
            elapsed = time.monotonic() - t_start
            remaining = total_duration_s - elapsed
            if remaining <= 0:
                break
            time.sleep(min(period_s, remaining))

        turn_fn(angle=0.0, source_type=source_type, duration=0)

    def gimbal_rotate_speed(
        self,
        *,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        yaw: Optional[float] = None,
        source_type: Optional[str] = None,
    ) -> None:
        """
        Rotate the gimbal at a constant speed (DJI MSDK ``KeyRotateBySpeed``).

        Units are 0.1°/s per the MSDK contract — i.e. ``pitch=100``
        means 10°/s. Valid range is ``[-3599, 3599]`` (i.e.
        ``±359.9°/s``). Each call drives the gimbal for a short
        window influenced by call frequency and airlink quality, so
        sustained motion needs the command re-issued.

        Args:
            pitch: Pitch speed in 0.1°/s, ``None`` for unset.
            roll: Roll speed in 0.1°/s, ``None`` for unset.
            yaw: Yaw speed in 0.1°/s, ``None`` for unset.
            source_type: ``"tele"`` / ``"sim_tele"`` (auto-resolved
                from ``cw.affect()`` if omitted).
        """
        data: Dict[str, Any] = {}
        if pitch is not None:
            data["pitch"] = float(pitch)
        if roll is not None:
            data["roll"] = float(roll)
        if yaw is not None:
            data["yaw"] = float(yaw)

        self._send_drone_command(
            "gimbal_rotate_speed",
            data=data,
            source_type=source_type,
        )

    # ------------------------------------------------------------------
    # Hovering status helpers
    # ------------------------------------------------------------------

    def is_hovering(self) -> bool:
        """
        Return True if this twin is currently in hovering mode.

        The hovering state is stored in ``twin.metadata.status.controller_requested_hovering``.
        This method reads the locally-cached twin data; call :meth:`refresh`
        first if you need the latest server-side value.

        Returns:
            bool: True when metadata.status.controller_requested_hovering is True, False otherwise.
        """
        meta: Dict[str, Any] = {}
        if hasattr(self._data, "metadata") and self._data.metadata:
            meta = dict(self._data.metadata)
        elif isinstance(self._data, dict):
            meta = self._data.get("metadata") or {}
        return bool(meta.get("status", {}).get("controller_requested_hovering", False))

    def get_hovering_status(self) -> Dict[str, Any]:
        """
        Return the hovering status dict from this twin's metadata.

        The returned dict follows the schema::

            {
                "controller_requested_hovering": bool,
                "controller_requested_hovering_altitude": float | None,  # altitude in metres
            }

        This method reads the locally-cached twin data; call :meth:`refresh`
        first if you need the latest server-side value.

        Returns:
            dict: Hovering status with keys ``controller_requested_hovering`` and
            optionally ``controller_requested_hovering_altitude``.
        """
        meta: Dict[str, Any] = {}
        if hasattr(self._data, "metadata") and self._data.metadata:
            meta = dict(self._data.metadata)
        elif isinstance(self._data, dict):
            meta = self._data.get("metadata") or {}
        status = meta.get("status") or {}
        return {
            "controller_requested_hovering": bool(
                status.get("controller_requested_hovering", False)
            ),
            "controller_requested_hovering_altitude": status.get(
                "controller_requested_hovering_altitude"
            ),
        }

    def set_hovering_status(
        self,
        *,
        hovering: bool,
        hovering_altitude: Optional[float] = None,
    ) -> None:
        """
        Persist the hovering status to the twin's metadata on the server.

        This performs a deep-merge into ``twin.metadata.status`` so that
        other metadata fields are not overwritten.

        Args:
            hovering: Whether the drone is currently hovering.
            hovering_altitude: Current altitude in meters. Required (or
                strongly recommended) when ``hovering`` is True.  Pass
                ``None`` to leave any existing value unchanged.

        Example::

            twin.set_hovering_status(hovering=True, hovering_altitude=2.5)
            twin.set_hovering_status(hovering=False)

        The values are persisted under
        ``twin.metadata.status.controller_requested_hovering`` and
        ``twin.metadata.status.controller_requested_hovering_altitude``.
        """
        # Read current metadata so we can merge rather than overwrite
        meta: Dict[str, Any] = {}
        if hasattr(self._data, "metadata") and self._data.metadata:
            meta = dict(self._data.metadata)
        elif isinstance(self._data, dict):
            meta = dict(self._data.get("metadata") or {})

        status: Dict[str, Any] = dict(meta.get("status") or {})
        status["controller_requested_hovering"] = hovering
        if hovering_altitude is not None:
            status["controller_requested_hovering_altitude"] = hovering_altitude
        elif not hovering:
            # Clear altitude when landing so stale values don't persist
            status.pop("controller_requested_hovering_altitude", None)

        meta["status"] = status

        try:
            self.client.twins.update(self.uuid, metadata=meta)  # type: ignore[union-attr]
        except Exception as exc:
            raise CyberwaveError(
                f"Failed to update hovering status for twin {self.uuid}: {exc}"
            ) from exc

        # Keep local cache in sync
        if hasattr(self._data, "metadata"):
            self._data.metadata = meta  # type: ignore[assignment]
        elif isinstance(self._data, dict):
            self._data["metadata"] = meta

    def __repr__(self) -> str:
        return f"FlyingTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperTwin(GripperCapableMixin, PolicyCapableMixin, Twin):
    """
    Twin with gripper/manipulation capabilities.

    Provides methods for controlling grippers and end effectors.
    """

    def grip(self, force: float = 1.0) -> None:
        """Close the gripper via :attr:`gripper` handle."""
        return self.gripper.grip(force)

    def release(self) -> None:
        """Open the gripper via :attr:`gripper` handle."""
        return self.gripper.release()

    def __repr__(self) -> str:
        return f"GripperTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperJointTwin(JointTwin, GripperTwin):
    """Manipulator with joints and gripper (e.g. SO-101)."""

    def __repr__(self) -> str:
        return f"GripperJointTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingCameraTwin(FlyingTwin, CameraTwin):
    """Twin with both flight and camera capabilities (camera drones)."""

    def __repr__(self) -> str:
        return f"FlyingCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperCameraTwin(GripperTwin, CameraTwin):
    """Twin with both gripper and camera capabilities (manipulators with vision)."""

    def __repr__(self) -> str:
        return f"GripperCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperJointCameraTwin(GripperJointTwin, CameraTwin):
    """Manipulator with joints, gripper, and camera (e.g. Kinova Gen3 + Robotiq)."""

    def __repr__(self) -> str:
        return f"GripperJointCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperDepthCameraTwin(GripperTwin, DepthCameraTwin):
    """Twin with both gripper and depth camera capabilities (manipulators with vision)."""

    def __repr__(self) -> str:
        return f"GripperDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class GripperJointDepthCameraTwin(GripperJointTwin, DepthCameraTwin):
    """Manipulator with joints, gripper, and depth camera."""

    def __repr__(self) -> str:
        return f"GripperJointDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteGripperTwin(LocomoteTwin, GripperTwin):
    """Twin with both locomotive and gripper capabilities (robots with grippers)."""

    def __repr__(self) -> str:
        return f"LocomoteGripperTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingGripperDepthCameraTwin(FlyingTwin, GripperDepthCameraTwin):
    """Twin with both flight and gripper and depth camera capabilities (drones with vision)."""

    def __repr__(self) -> str:
        return f"FlyingGripperDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteGripperDepthCameraTwin(LocomoteTwin, GripperDepthCameraTwin):
    """Twin with both locomotive and gripper and depth camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteGripperDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteDepthCameraTwin(LocomoteTwin, DepthCameraTwin):
    """Twin with both locomotive and depth camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteGripperCameraTwin(LocomoteTwin, GripperCameraTwin):
    """Twin with both locomotive and gripper and camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteGripperCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteCameraTwin(LocomoteTwin, CameraTwin):
    """Twin with both locomotive and camera capabilities (robots with vision)."""

    def __repr__(self) -> str:
        return f"LocomoteCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteJointTwin(LocomoteTwin, JointTwin):
    """Locomoting platform with controllable joints (e.g. legged robots without sensors)."""

    def __repr__(self) -> str:
        return f"LocomoteJointTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteJointCameraTwin(LocomoteJointTwin, CameraTwin):
    """Legged platform with joints and a camera (e.g. Unitree Go2, ANYmal)."""

    def __repr__(self) -> str:
        return f"LocomoteJointCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class LocomoteJointDepthCameraTwin(LocomoteJointTwin, DepthCameraTwin):
    """Legged platform with joints and a depth camera (e.g. Spot, RealSense-equipped quadrupeds)."""

    def __repr__(self) -> str:
        return f"LocomoteJointDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingGripperCameraTwin(FlyingTwin, GripperCameraTwin):
    """Twin with both flight and gripper and camera capabilities (drones with vision)."""

    def __repr__(self) -> str:
        return f"FlyingGripperCameraTwin(uuid='{self.uuid}', name='{self.name}')"


class FlyingDepthCameraTwin(FlyingTwin, DepthCameraTwin):
    """Twin with both flight and depth camera capabilities (drones with vision)."""

    def __repr__(self) -> str:
        return f"FlyingDepthCameraTwin(uuid='{self.uuid}', name='{self.name}')"

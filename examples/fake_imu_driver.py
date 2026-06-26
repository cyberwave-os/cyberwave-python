#!/usr/bin/env python3
"""Fake 6-DOF IMU driver for an Intel RealSense D455 twin.

The D455 catalog asset (``intel/realsensed455``) has RGB + depth only — no hardware
IMU. This example attaches a **synthetic** 6-DOF stream (random gyro/accel each
tick at ``imu_rate_hz``) so you can exercise the driver registry, MQTT schema, and
``twin.imu.get()`` read path end-to-end.

Demo workflow (``python examples/fake_imu_driver.py``)
------------------------------------------------------

1. **Define** driver interface (``define_interface``), params, and payloads (this module).
2. **Create twin** — resolve ``CYBERWAVE_TWIN_UUID`` or default D455 twin; add ``d455_imu``
   to capabilities when missing.
3. **Register catalog** — ``twin.driver.set_schema(FakeImu6dDriver.get_manifest())`` *before* ``run()`` (optional ``get_manifest(path="cw-driver.yml")`` to export YAML).
4. **Start driver** — ``driver.run_async()`` publishes IMU @ 50 Hz; demo calls ``twin.imu.get()``.
5. **Rotate** — ``twin.commands.rotate(...)``; driver logs ``command received, rotating sensor``.

Use ``--serve`` to skip the demo and run until Ctrl+C.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from cyberwave.driver import (
    CallbackGroup,
    CommandArgs,
    BaseDriver,
    DriverLifecycleState,
    DriverOperationMode,
    PublisherArgs,
    ProtocolArgs,
    TopicSpec,
)
from cyberwave.manifest.driver_config import TWIN_IMU_TOPIC_SLUG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experimental wire schema (documented here until platform catalog adds it)
# ---------------------------------------------------------------------------

IMU_PAYLOAD_SCHEMA_REF = "ImuPayload"
DEFAULT_REGISTRY_ID = "intel/realsensed455"
DEFAULT_SENSOR_ID = "d455_imu"
IMU_SOURCE_TYPE = "live"

# ``--demo``: run driver this long and sample ``twin.imu.get()`` at three checkpoints
DEMO_RUN_SECONDS = 15.0
DEMO_AFTER_START_DELAY_S = 3.0
DEMO_BEFORE_END_DELAY_S = 12.0
DEMO_GET_TIMEOUT_S = 5.0
DEMO_ROTATE_AXIS = "yaw"
DEMO_ROTATE_AMOUNT_DEG = 30.0

# roll/pitch/yaw name the gyro axis for teleop ergonomics
GyroAxisName = Literal["roll", "pitch", "yaw"]
_GYRO_AXIS_INDEX: dict[GyroAxisName, Literal["x", "y", "z"]] = {
    "roll": "x",
    "pitch": "y",
    "yaw": "z",
}


class Vector3(TypedDict):
    x: float
    y: float
    z: float


class ImuPayload(TypedDict, total=False):
    """6-DOF IMU sample on ``cyberwave/twin/{uuid}/imu`` (experimental).

    The topic slug identifies the stream; no ``type`` field on the wire.

    - ``gyro`` — angular rate (rad/s), body frame
    - ``accel`` — linear acceleration (m/s²), including gravity when stationary

    Inbound samples may use legacy ``angular_velocity`` / ``linear_acceleration``;
    readers normalize those to ``gyro`` / ``accel`` without duplicating on the wire.
    """

    source_type: Literal["live"]
    timestamp: float
    gyro: Vector3
    accel: Vector3
    sensor_id: str
    dof: Literal[6]


def build_imu_payload(
    *,
    gyro: Vector3,
    accel: Vector3,
    sensor_id: str = DEFAULT_SENSOR_ID,
) -> ImuPayload:
    return ImuPayload(
        source_type=IMU_SOURCE_TYPE,
        timestamp=time.time(),
        dof=6,
        gyro=gyro,
        accel=accel,
        sensor_id=sensor_id,
    )


def d455_imu_capability_entry() -> dict[str, Any]:
    """Capability row to add on the twin when using this example with a D455."""
    return {"id": DEFAULT_SENSOR_ID, "type": "imu", "update_rate": 50.0}


def _capabilities_dict_for_update(twin: Any) -> dict[str, Any]:
    """Mutable capabilities backing store (not the ephemeral ``{}`` from the property)."""
    data = getattr(twin, "_data", None)
    if isinstance(data, dict):
        caps = data.get("capabilities")
        if not isinstance(caps, dict):
            caps = {}
            data["capabilities"] = caps
        return caps
    if data is not None and hasattr(data, "capabilities"):
        caps = getattr(data, "capabilities", None)
        if not isinstance(caps, dict):
            caps = {}
            data.capabilities = caps
        return caps
    caps = getattr(twin, "capabilities", None)
    if isinstance(caps, dict):
        return caps
    return {}


def ensure_imu_capability(twin: Any) -> bool:
    """Merge synthetic IMU sensor into capabilities (D455 has no hardware IMU).

    Returns True when a new sensor row was added.
    """
    entry = d455_imu_capability_entry()
    caps = _capabilities_dict_for_update(twin)
    sensors = [s for s in (caps.get("sensors") or []) if isinstance(s, dict)]
    if any(str(s.get("id") or s.get("name")) == entry["id"] for s in sensors):
        return False
    sensors.append(dict(entry))
    caps["sensors"] = sensors
    logger.info(
        "Added IMU capability sensor_id=%s to twin (in-memory for this process)",
        entry["id"],
    )
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

GRAVITY_M_S2 = 9.81

# Per-axis uniform noise on top of commanded gyro / gravity (tune via env if needed)
_DEFAULT_GYRO_NOISE_RAD_S = 0.15
_DEFAULT_ACCEL_NOISE_M_S2 = 0.35


def random_imu_vectors(
    *,
    gyro_base: Vector3 | None = None,
    gyro_noise_rad_s: float = _DEFAULT_GYRO_NOISE_RAD_S,
    accel_noise_m_s2: float = _DEFAULT_ACCEL_NOISE_M_S2,
) -> tuple[Vector3, Vector3]:
    """Build noisy 6-DOF vectors for one MQTT sample."""
    base = gyro_base or Vector3(x=0.0, y=0.0, z=0.0)

    def _jitter(center: float, span: float) -> float:
        return center + random.uniform(-span, span)

    gyro = Vector3(
        x=_jitter(base["x"], gyro_noise_rad_s),
        y=_jitter(base["y"], gyro_noise_rad_s),
        z=_jitter(base["z"], gyro_noise_rad_s),
    )
    accel = Vector3(
        x=random.uniform(-accel_noise_m_s2, accel_noise_m_s2),
        y=random.uniform(-accel_noise_m_s2, accel_noise_m_s2),
        z=_jitter(GRAVITY_M_S2, accel_noise_m_s2),
    )
    return gyro, accel


@dataclass
class FakeImu6dParams:
    """Hardware-free knobs (env overrides optional)."""

    sensor_id: str = DEFAULT_SENSOR_ID
    imu_rate_hz: float = 50.0
    gyro_noise_rad_s: float = _DEFAULT_GYRO_NOISE_RAD_S
    accel_noise_m_s2: float = _DEFAULT_ACCEL_NOISE_M_S2

    @classmethod
    def from_env(cls) -> FakeImu6dParams:
        return cls(
            sensor_id=os.getenv("FAKE_IMU_SENSOR_ID", DEFAULT_SENSOR_ID),
            imu_rate_hz=float(os.getenv("FAKE_IMU_RATE_HZ", "50")),
            gyro_noise_rad_s=float(
                os.getenv("FAKE_IMU_GYRO_NOISE_RAD_S", str(_DEFAULT_GYRO_NOISE_RAD_S))
            ),
            accel_noise_m_s2=float(
                os.getenv("FAKE_IMU_ACCEL_NOISE_M_S2", str(_DEFAULT_ACCEL_NOISE_M_S2))
            ),
        )


@dataclass
class _Imu6dState:
    """Internal gyro (rad/s) + accel (m/s²)."""

    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = GRAVITY_M_S2

    def gyro(self) -> Vector3:
        return Vector3(x=self.gyro_x, y=self.gyro_y, z=self.gyro_z)

    def accel(self) -> Vector3:
        return Vector3(x=self.accel_x, y=self.accel_y, z=self.accel_z)

    def set_gyro_rate_deg_s(self, axis: GyroAxisName, rate_deg_s: float) -> None:
        rate_rad_s = math.radians(rate_deg_s)
        idx = _GYRO_AXIS_INDEX[axis]
        if idx == "x":
            self.gyro_x = rate_rad_s
        elif idx == "y":
            self.gyro_y = rate_rad_s
        else:
            self.gyro_z = rate_rad_s

    def zero_gyro(self) -> None:
        self.gyro_x = self.gyro_y = self.gyro_z = 0.0


class FakeImu6dDriver(BaseDriver):
    """Synthetic 6-DOF IMU for D455 twins; ``rotate`` sets body-frame gyro rate."""

    REGISTRY_ID = DEFAULT_REGISTRY_ID
    driver_family = "python"
    TELEMETRY_PUBLISH_RATE_HZ = 2.0
    TICK_RATE_HZ = 50.0
    _ALERT_COMPONENT = "fake_imu_driver"

    def __init__(
        self,
        twin: Any | None = None,
        params: FakeImu6dParams | None = None,
        **kwargs: Any,
    ) -> None:
        self._imu_params = params or FakeImu6dParams.from_env()
        self._imu = _Imu6dState()
        self._last_rotate: dict[str, Any] | None = None
        super().__init__(self._imu_params, twin=twin, **kwargs)

    def define_interface(self, iface) -> None:
        cmd = TopicSpec(
            namespace="twin",
            leaf="command",
            payload_schema_ref="TwinCommandPayload",
            description="Set synthetic gyro rate on one axis (deg/s)",
        )
        imu = TopicSpec(
            topic_slug=TWIN_IMU_TOPIC_SLUG,
            payload_schema_ref=IMU_PAYLOAD_SCHEMA_REF,
            description="6-DOF IMU: gyro (rad/s) + accel (m/s²)",
            enable_zenoh=True,
            zenoh_channel="imu",
        )
        all_modes = frozenset(DriverOperationMode)

        iface.add_listener(
            cmd,
            CallbackGroup(callback=self._on_rotate),
            protocol=ProtocolArgs(source_types=["tele", "live", "edge"]),
            command=CommandArgs(name="rotate"),
            operation_modes=all_modes,
        )
        iface.add_publisher(
            imu,
            CallbackGroup(callback=self._publish_imu_sample),
            protocol=ProtocolArgs(
                source_types=[IMU_SOURCE_TYPE],
                units={
                    "gyro_x": "rad/s",
                    "gyro_y": "rad/s",
                    "gyro_z": "rad/s",
                    "accel_x": "m/s²",
                    "accel_y": "m/s²",
                    "accel_z": "m/s²",
                },
                direction_notes="6-DOF IMU stream (gyro + accel)",
            ),
            publisher=PublisherArgs(rate_hz=self._imu_params.imu_rate_hz),
            operation_modes=all_modes,
        )

    async def on_configure(self) -> None:
        logger.info(
            "FakeImu6dDriver configured (asset=%s, sensor_id=%s, imu_rate_hz=%s)",
            self.registry_id,
            self._imu_params.sensor_id,
            self._imu_params.imu_rate_hz,
        )

    async def on_connect_to_device(self) -> None:
        logger.info("Fake 6-DOF IMU ready (no physical device on D455)")

    async def on_register_callbacks(self) -> None:
        pass

    async def on_activate(self) -> None:
        self._emit_driver_info(sensor_id=self._imu_params.sensor_id, dof=6)
        topic = TWIN_IMU_TOPIC_SLUG.format(twin_uuid=self.twin_uuid)
        logger.info(
            "IMU stream active: %s @ %.0f Hz (source_type=%s)",
            topic,
            self._imu_params.imu_rate_hz,
            IMU_SOURCE_TYPE,
        )
        if self._registry_zenoh_requested():
            logger.info(
                "Dual transport: MQTT %s + Zenoh channel imu (DataBus starts when ACTIVE)",
                topic,
            )

    async def on_shutdown(self) -> None:
        pass

    def _transition_to(self, state: DriverLifecycleState) -> None:
        prev = self.lifecycle_state
        super()._transition_to(state)
        if prev == state:
            return
        self.create_twin_alert(
            name=f"Driver lifecycle: {prev.value} → {state.value}",
            description=f"Component: {self._ALERT_COMPONENT}",
            alert_type="driver_lifecycle",
            severity="info",
            source_type="edge",
            metadata={
                "from_state": prev.value,
                "to_state": state.value,
                "component": self._ALERT_COMPONENT,
                "imu_source_type": IMU_SOURCE_TYPE,
            },
        )
        if state == DriverLifecycleState.ACTIVE:
            self.create_twin_alert(
                name="Driver active — publishing IMU stream",
                description=(
                    f"sensor_id={self._imu_params.sensor_id}, "
                    f"rate_hz={self._imu_params.imu_rate_hz}"
                ),
                alert_type="driver_active",
                severity="info",
                source_type="edge",
                metadata={
                    "sensor_id": self._imu_params.sensor_id,
                    "imu_rate_hz": self._imu_params.imu_rate_hz,
                    "imu_source_type": IMU_SOURCE_TYPE,
                },
            )

    def driver_info_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "sensor_id": self._imu_params.sensor_id,
            "dof": 6,
            "gyro_rad_s": self._imu.gyro(),
            "accel_m_s2": self._imu.accel(),
        }
        if self._last_rotate is not None:
            extra["last_rotate"] = self._last_rotate
        return extra

    def _publish_imu_sample(self) -> ImuPayload:
        """Registry publisher callback — random gyro/accel each tick."""
        gyro, accel = random_imu_vectors(
            gyro_base=self._imu.gyro(),
            gyro_noise_rad_s=self._imu_params.gyro_noise_rad_s,
            accel_noise_m_s2=self._imu_params.accel_noise_m_s2,
        )
        return build_imu_payload(
            gyro=gyro,
            accel=accel,
            sensor_id=self._imu_params.sensor_id,
        )

    def _on_rotate(self, envelope: dict[str, Any]) -> None:
        if envelope.get("command") == "status":
            return
        data = envelope.get("data") or {}
        axis = str(data.get("axis", "")).lower()
        rate_deg_s = float(data.get("amount_deg", data.get("amount", 0.0)))
        logger.info(
            "command received, rotating sensor (command=rotate axis=%s amount_deg=%.2f "
            "source_type=%s)",
            axis or "?",
            rate_deg_s,
            envelope.get("source_type"),
        )
        if axis not in _GYRO_AXIS_INDEX:
            logger.warning(
                "rotate ignored: axis must be roll, pitch, or yaw (got %r)", axis
            )
            return
        if rate_deg_s == 0.0:
            self._imu.zero_gyro()
        else:
            self._imu.set_gyro_rate_deg_s(axis, rate_deg_s)  # type: ignore[arg-type]
        self._last_rotate = {
            "axis": axis,
            "rate_deg_s": rate_deg_s,
            "rate_rad_s": math.radians(rate_deg_s),
            "gyro_rad_s": self._imu.gyro(),
            "timestamp": time.time(),
        }
        logger.info(
            "sensor gyro updated: axis=%s rate=%.2f deg/s -> %s",
            axis,
            rate_deg_s,
            self._imu.gyro(),
        )
        self._emit_driver_info(last_rotate=self._last_rotate)

    @classmethod
    def create(cls) -> FakeImu6dDriver:
        """Open twin from env; call :func:`prepare_demo_twin` before run/demo."""
        return cls(open_demo_twin())


def open_demo_twin() -> Any:
    """Resolve the D455 twin used by this example (``CYBERWAVE_TWIN_UUID`` optional)."""
    from cyberwave import Cyberwave

    cw = Cyberwave()
    twin_id = os.getenv("CYBERWAVE_TWIN_UUID", "").strip()
    if twin_id:
        return cw.twin(FakeImu6dDriver.REGISTRY_ID, twin_id=twin_id)
    return cw.twin(FakeImu6dDriver.REGISTRY_ID)


def prepare_demo_twin(twin: Any) -> None:
    """Step 2 — add ``d455_imu`` to capabilities when the catalog twin has no IMU."""
    twin_uuid = getattr(twin, "uuid", "?")
    logger.info("Demo step 2: twin %s — checking IMU capability", twin_uuid)
    if ensure_imu_capability(twin):
        logger.info("Demo step 2: added synthetic IMU sensor %s", DEFAULT_SENSOR_ID)
    else:
        logger.info("Demo step 2: IMU sensor %s already present", DEFAULT_SENSOR_ID)


def register_demo_schema(twin: Any, driver: FakeImu6dDriver | None = None) -> dict[str, Any]:
    """Step 3 — persist manifest on the twin before the driver loop starts."""
    logger.info("Demo step 3: twin.driver.set_schema(FakeImu6dDriver.get_manifest())")
    schemas = twin.driver.set_schema(FakeImu6dDriver.get_manifest())
    schema = schemas["mqtt"]
    supported = schema.get("commands", {}).get("supported", [])
    logger.info("Demo step 3: catalog commands=%s", supported)
    if "rotate" not in supported:
        logger.warning("Demo step 3: rotate not in catalog after set_schema")
    return schema


def send_demo_rotate(
    twin: Any,
    *,
    axis: str = DEMO_ROTATE_AXIS,
    amount_deg: float = DEMO_ROTATE_AMOUNT_DEG,
) -> None:
    """Step 5 — publish ``rotate`` via ``twin.commands`` (driver must be ACTIVE)."""
    logger.info(
        "Demo step 5: twin.commands.rotate(axis=%r, amount_deg=%s)",
        axis,
        amount_deg,
    )
    rotate_fn = getattr(twin.commands, "rotate", None)
    if not callable(rotate_fn):
        raise RuntimeError(
            "twin.commands.rotate is not bound — run twin.driver.set_schema(driver.manifest) first"
        )
    rotate_fn(axis=axis, amount_deg=amount_deg, source_type=IMU_SOURCE_TYPE)
    logger.info("Demo step 5: rotate MQTT command published")


def _disconnect_sdk_client(twin: Any) -> None:
    """Release MQTT (and Zenoh) on a twin's Cyberwave client without tearing down the twin."""
    client = getattr(twin, "client", None)
    if client is None:
        return
    disconnect = getattr(client, "disconnect", None)
    if not callable(disconnect):
        return
    try:
        disconnect()
    except Exception:
        logger.debug("SDK client disconnect failed", exc_info=True)


def open_consumer_twin(driver: FakeImu6dDriver) -> Any:
    """MQTT consumer on a separate SDK client (avoids blocking the driver asyncio loop)."""
    from cyberwave import Cyberwave

    bound = driver._twin
    twin_uuid = (
        str(bound.uuid)
        if bound is not None and hasattr(bound, "uuid")
        else driver.twin_uuid
    )
    cw = Cyberwave()
    if not getattr(cw.mqtt, "connected", False):
        cw.mqtt.connect()
        deadline = time.monotonic() + 10.0
        while not cw.mqtt.connected and time.monotonic() < deadline:
            time.sleep(0.05)
    return cw.twin(driver.REGISTRY_ID, twin_id=twin_uuid)


def print_imu_sample(label: str, twin: Any, *, timeout: float = DEMO_GET_TIMEOUT_S) -> None:
    """Print one ``twin.imu.get()`` result (or the error)."""
    print(f"\n=== {label} ===", flush=True)
    ensure_imu_capability(twin)
    has_imu = getattr(twin, "has_sensor", None)
    if callable(has_imu) and not has_imu("imu"):
        print(
            "  twin has no imu capability — add sensor "
            f"{DEFAULT_SENSOR_ID!r} to capabilities first",
            flush=True,
        )
        return
    try:
        sample = twin.imu.get(timeout=timeout)
        print(f"  gyro:  {sample.get('gyro')}", flush=True)
        print(f"  accel: {sample.get('accel')}", flush=True)
    except Exception as exc:
        print(f"  {type(exc).__name__}: {exc}", flush=True)


async def run_demo_async(
    driver: FakeImu6dDriver,
    *,
    run_seconds: float = DEMO_RUN_SECONDS,
    after_start_delay_s: float = DEMO_AFTER_START_DELAY_S,
    before_end_delay_s: float = DEMO_BEFORE_END_DELAY_S,
) -> None:
    """Demo: define → twin/IMU → set_schema(manifest) → driver+get → rotate."""
    if driver._twin is None:
        raise RuntimeError("driver has no bound twin")

    twin = driver._twin
    logger.info("Demo step 1: FakeImu6dDriver interface and params defined")
    prepare_demo_twin(twin)
    await asyncio.to_thread(register_demo_schema, twin)

    logger.info(
        "Demo step 4: start driver (%.0fs) + imu.get on consumer client",
        run_seconds,
    )
    loop = asyncio.get_running_loop()
    loop.call_later(run_seconds, driver.request_shutdown)
    driver_task = asyncio.create_task(driver.run_async(), name="fake-imu-driver")

    await asyncio.sleep(after_start_delay_s)
    consumer = await asyncio.to_thread(open_consumer_twin, driver)
    ensure_imu_capability(consumer)
    await asyncio.to_thread(
        print_imu_sample,
        f"4. IMU sample after driver start (~{after_start_delay_s:.0f}s)",
        consumer,
    )

    print("\n=== 5. twin.commands.rotate ===", flush=True)
    try:
        await asyncio.to_thread(send_demo_rotate, consumer)
    except Exception as exc:
        print(f"  {type(exc).__name__}: {exc}", flush=True)

    await asyncio.sleep(1.0)
    await asyncio.to_thread(
        print_imu_sample, "5b. IMU sample after rotate", consumer
    )
    await asyncio.sleep(max(0.0, before_end_delay_s - after_start_delay_s - 1.0))
    await asyncio.to_thread(
        print_imu_sample, f"6. IMU sample before shutdown (~{before_end_delay_s:.0f}s)", consumer
    )

    await driver_task
    _disconnect_sdk_client(consumer)
    print("\nDemo finished.", flush=True)


def run_demo(
    driver: FakeImu6dDriver,
    *,
    run_seconds: float = DEMO_RUN_SECONDS,
    after_start_delay_s: float = DEMO_AFTER_START_DELAY_S,
    before_end_delay_s: float = DEMO_BEFORE_END_DELAY_S,
) -> None:
    """Blocking entry: run :func:`run_demo_async` on the main thread (signal-safe)."""
    asyncio.run(
        run_demo_async(
            driver,
            run_seconds=run_seconds,
            after_start_delay_s=after_start_delay_s,
            before_end_delay_s=before_end_delay_s,
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fake 6-DOF IMU driver for Intel RealSense D455 twins.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run until Ctrl+C (no imu.get demo steps)",
    )
    parser.add_argument(
        "--demo-seconds",
        type=float,
        default=DEMO_RUN_SECONDS,
        help=f"Demo duration when not using --serve (default {DEMO_RUN_SECONDS})",
    )
    parser.add_argument(
        "--write-cw-driver",
        nargs="?",
        const="cw-driver.yml",
        metavar="PATH",
        help="Export cw-driver.yml from define_interface and exit (default: cw-driver.yml)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    driver = FakeImu6dDriver.create()
    if args.write_cw_driver is not None:
        FakeImu6dDriver.get_manifest(path=args.write_cw_driver)
        print(f"Wrote {args.write_cw_driver}", flush=True)
        return 0
    prepare_demo_twin(driver._twin)
    if args.serve:
        logger.info("Serve mode: running until Ctrl+C (no demo steps)")
        driver.run()
        return 0
    logger.info(
        "Demo mode (%.0fs): twin → set_schema(manifest) → driver → imu.get → rotate",
        args.demo_seconds,
    )
    run_demo(driver, run_seconds=args.demo_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

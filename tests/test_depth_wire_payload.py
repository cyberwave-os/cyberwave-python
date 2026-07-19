"""Producer wire payload: ``build_depth_mqtt_payload`` round-trips through ``_decode_depth``.

The depth wire is self-describing via ``dtype`` — ``uint16`` carries millimetres,
``float32`` carries absolute metres. These tests lock the producer helper and the
SDK consumer to the same contract so an fp32 ``/depth`` stream decodes verbatim.
"""

import base64

import numpy as np
import pytest

from cyberwave.twin.sensors.depth import _decode_depth
from cyberwave.utils.depth import (
    DEPTH_OUTPUT_MODE_METRIC_MM,
    build_depth_mqtt_payload,
    depth_to_uint16,
)


def _as_depth_payload(wire: dict) -> dict:
    """Wrap a wire dict as a ``/depth`` MQTT payload for ``_decode_depth``."""
    return {"type": "depth_data", "data": wire}


def test_uint16_payload_is_tagged_and_default() -> None:
    arr = np.array([[0, 500], [1000, 2000]], dtype=np.uint16)  # millimetres
    wire = build_depth_mqtt_payload(arr)
    assert wire["dtype"] == "uint16"
    assert (wire["height"], wire["width"]) == (2, 2)
    out = _decode_depth(_as_depth_payload(wire))
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, arr)


def test_uint16_mode_coerces_non_uint16_input() -> None:
    # Legacy contract: a non-uint16 array under the default mode is cast to uint16.
    wire = build_depth_mqtt_payload(np.array([[1.0, 2.0]], dtype=np.float32))
    assert wire["dtype"] == "uint16"
    out = _decode_depth(_as_depth_payload(wire))
    np.testing.assert_array_equal(out, np.array([[1, 2]], dtype=np.uint16))


def test_float32_payload_carries_metres_verbatim() -> None:
    metres = np.array([[0.0, 0.3, 0.611], [1.04, 2.5, 70.0]], dtype=np.float32)
    wire = build_depth_mqtt_payload(metres, wire_dtype="float32")
    assert wire["dtype"] == "float32"
    assert (wire["height"], wire["width"]) == (2, 3)
    out = _decode_depth(_as_depth_payload(wire))
    assert out.dtype == np.float32
    # No quantisation: values (incl. 70 m, beyond uint16 mm range) survive exactly.
    np.testing.assert_array_equal(out, metres)


def test_float32_tag_always_matches_bytes() -> None:
    # float64 input is normalised to float32 bytes with a matching dtype tag.
    wire = build_depth_mqtt_payload(
        np.array([[1.5, 2.5]], dtype=np.float64), wire_dtype="float32"
    )
    assert wire["dtype"] == "float32"
    raw = np.frombuffer(base64.b64decode(wire["depth_binary"]), dtype=np.float32)
    np.testing.assert_array_equal(raw, np.array([1.5, 2.5], dtype=np.float32))


def test_unsupported_wire_dtype_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported wire_dtype"):
        build_depth_mqtt_payload(np.zeros((1, 1), dtype=np.float32), wire_dtype="int8")


def test_metric_mm_producer_path_roundtrips_via_payload() -> None:
    # metric_mm producer path: metres → uint16 mm → payload → decode → mm.
    metres = np.array([[0.1, 0.25], [1.0, 3.0]], dtype=np.float32)
    u16 = depth_to_uint16(metres, output_mode=DEPTH_OUTPUT_MODE_METRIC_MM)
    wire = build_depth_mqtt_payload(u16)
    out = _decode_depth(_as_depth_payload(wire))
    np.testing.assert_array_equal(
        out, np.array([[100, 250], [1000, 3000]], dtype=np.uint16)
    )

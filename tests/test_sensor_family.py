"""Unit tests for the generic SensorFamily collection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.twin.capability_resolve import resolve_handler_from_capabilities
from cyberwave.twin.sensors.family import SensorFamily


class _FakeHandle:
    def __init__(self, twin, key):
        self._twin = twin
        self.sensor_id = key

    def ping(self):
        return f"ping:{self.sensor_id}"


def _handle_for_key(twin, key):
    ids = [
        str(e.get("id"))
        for e in twin.capabilities.get("sensors", [])
        if isinstance(e, dict)
    ]
    if key not in ids:
        raise KeyError(f"No sensor '{key}'")
    return _FakeHandle(twin, key)


def _twin(sensors):
    data = SimpleNamespace(uuid="t", name="T", capabilities={"sensors": sensors})
    twin = SimpleNamespace(_data=data, capabilities={"sensors": sensors})
    twin.resolve_handler_from_capabilities = (
        lambda handler: resolve_handler_from_capabilities(twin.capabilities, handler)
    )
    return twin


def _family(sensors, *, label="imu", handler="imu"):
    return SensorFamily(
        _twin(sensors),
        handler_key=handler,
        family_label=label,
        public_methods=("metadata", "get", "ping"),
        handle_for_key=_handle_for_key,
    )


def test_keys_len_iter_contains():
    fam = _family([{"id": "a", "type": "imu"}, {"id": "b", "type": "imu"}])
    assert fam.keys() == ["a", "b"]
    assert len(fam) == 2
    assert [h.sensor_id for h in fam] == ["a", "b"]
    assert "a" in fam and "z" not in fam


def test_getitem_by_index_and_key_and_negative():
    fam = _family([{"id": "a", "type": "imu"}, {"id": "b", "type": "imu"}])
    assert fam[0].sensor_id == "a"
    assert fam[-1].sensor_id == "b"
    assert fam["b"].sensor_id == "b"


def test_getitem_index_out_of_range_raises_indexerror():
    fam = _family([{"id": "a", "type": "imu"}])
    with pytest.raises(IndexError):
        fam[5]


def test_getitem_unknown_key_raises_keyerror():
    fam = _family([{"id": "a", "type": "imu"}])
    with pytest.raises(KeyError):
        fam["nope"]


def test_attribute_named_sensor_returns_that_handle():
    fam = _family([{"id": "left", "type": "imu"}, {"id": "right", "type": "imu"}])
    assert fam.right.sensor_id == "right"
    assert fam.left is not fam.right


def test_unknown_method_proxies_to_first_sensor():
    fam = _family([{"id": "a", "type": "imu"}, {"id": "b", "type": "imu"}])
    assert fam.ping() == "ping:a"


def test_handle_is_memoized_per_key():
    # Repeated access to the same sensor (by index, key, or proxy) must return
    # the same handle instance so stateful handles keep their listeners.
    fam = _family([{"id": "a", "type": "imu"}, {"id": "b", "type": "imu"}])
    assert fam[0] is fam[0]
    assert fam[0] is fam["a"]
    assert fam["b"] is fam[-1]


def test_private_attribute_access_does_not_proxy():
    fam = _family([{"id": "a", "type": "imu"}])
    with pytest.raises(AttributeError):
        fam._does_not_exist


def test_values_items_describe():
    fam = _family([{"id": "a", "type": "imu"}, {"id": "b", "type": "imu"}])
    assert [h.sensor_id for h in fam.values()] == ["a", "b"]
    assert [k for k, _ in fam.items()] == ["a", "b"]
    info = fam.describe()
    assert set(info) == {"a", "b"}
    assert info["a"]["sensor_id"] == "a"
    assert info["a"]["type"] == "imu"
    assert info["a"]["family"] == "imu"
    assert info["a"]["handle"] == "_FakeHandle"
    assert info["a"]["methods"] == ["metadata", "get", "ping"]


def test_dir_lists_keys_and_methods():
    fam = _family([{"id": "a", "type": "imu"}])
    names = dir(fam)
    assert "a" in names
    assert "ping" in names
    assert "describe" in names

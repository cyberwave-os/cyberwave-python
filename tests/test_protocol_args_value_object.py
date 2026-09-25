"""B6: ProtocolArgs is a real frozen value object — immutable, non-aliasing."""

import pytest

from cyberwave.driver import ProtocolArgs


def test_source_types_coerced_to_immutable_tuple():
    pa = ProtocolArgs(source_types=["tele", "edit"])
    assert pa.source_types == ("tele", "edit")
    assert not hasattr(pa.source_types, "append")


def test_related_topics_coerced_to_tuple():
    pa = ProtocolArgs(related_topics=["a", "b"])
    assert pa.related_topics == ("a", "b")


def test_no_aliasing_when_reused_across_registrations():
    shared = ["tele"]
    pa = ProtocolArgs(source_types=shared)
    shared.append("edge")  # mutating the caller's list must not affect pa
    assert pa.source_types == ("tele",)


def test_hashable_value_object():
    pa = ProtocolArgs(source_types=["tele", "edit"])
    assert pa in {pa}  # would raise TypeError: unhashable if list-backed


def test_units_round_trips_to_dict_and_is_copied():
    src = {"joint_1": "rad"}
    pa = ProtocolArgs(units=src)
    assert dict(pa.units) == {"joint_1": "rad"}
    src["joint_1"] = "deg"  # caller mutation must not leak in
    assert dict(pa.units) == {"joint_1": "rad"}


def test_none_fields_stay_none():
    pa = ProtocolArgs()
    assert pa.source_types is None
    assert pa.units is None
    assert pa.related_topics is None

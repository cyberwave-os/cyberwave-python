"""The SDK's source_type constants must match the contracts package."""

import pytest

import cyberwave.constants as constants

cyberwave_contracts = pytest.importorskip("cyberwave_contracts")


def test_named_constants_cover_exactly_the_vocabulary():
    named = {
        value
        for name, value in vars(constants).items()
        if name.startswith("SOURCE_TYPE_") and isinstance(value, str)
    }
    assert named == set(cyberwave_contracts.SOURCE_TYPE_VALUES)


def test_source_types_tuple_covers_exactly_the_vocabulary():
    assert len(constants.SOURCE_TYPES) == len(set(constants.SOURCE_TYPES))
    assert set(constants.SOURCE_TYPES) == set(cyberwave_contracts.SOURCE_TYPE_VALUES)

"""Scene.add_twin docking parameters (replay-feed README §13): create, then dock()."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.scene import Scene

ENVIRONMENT = "ba7e2341-b3a9-4fdf-97e9-808a234443a2"
RACK = "d437303a-382a-4d7d-a3e8-e2e5bbca5210"


class FakeTwin:
    def __init__(self, uuid):
        self._data = SimpleNamespace(uuid=uuid)

    @property
    def uuid(self):
        return self._data.uuid


def make_scene(created_uuid="child-uuid"):
    client = MagicMock()
    client.config.base_url = "http://localhost:8000/"
    client._api_client.call_api.return_value.read.return_value = b"[]"
    client.twin.return_value = FakeTwin(created_uuid)
    client.twins.update.return_value = SimpleNamespace(
        uuid=created_uuid, attach_to_twin_uuid=RACK
    )
    scene = Scene(client, ENVIRONMENT)
    client._api_client.call_api.reset_mock()
    return scene, client


def test_add_twin_docks_through_dock_with_offsets():
    scene, client = make_scene()

    twin = scene.add_twin(
        "cyberwave/server-2u",
        name="rack-48-s00",
        fixed_base=True,
        dock_to=RACK,
        dock_link="server_rack_42u_link",
        dock_offset=(0.105, 0, 0.14445),
        dock_rotation=(0, 0, 0, 1),
    )

    assert twin is client.twin.return_value
    client.twin.assert_called_once_with(
        asset_key="cyberwave/server-2u",
        environment_id=ENVIRONMENT,
        name="rack-48-s00",
        position_x=0.0,
        position_y=0.0,
        position_z=0.0,
        rotation_x=0.0,
        rotation_y=0.0,
        rotation_z=0.0,
        rotation_w=1.0,
        fixed_base=True,
    )
    client.twins.update.assert_called_once_with(
        "child-uuid",
        attach_to_twin_uuid=RACK,
        attach_to_link="server_rack_42u_link",
        attach_offset_x=0.105,
        attach_offset_y=0,
        attach_offset_z=0.14445,
        attach_offset_rotation_x=0,
        attach_offset_rotation_y=0,
        attach_offset_rotation_z=0,
        attach_offset_rotation_w=1,
    )
    # The returned object carries the docked twin data without a second fetch.
    assert twin._data is client.twins.update.return_value
    # dock() reloads the twin list once.
    client._api_client.call_api.assert_called_once()


def test_add_twin_accepts_a_parent_twin_object_and_root_link():
    scene, client = make_scene()
    parent = FakeTwin(RACK)

    scene.add_twin(
        "cyberwave/server-2u",
        name="s01",
        dock_to=parent,
        dock_offset=[0.105, 0.0, 0.23335],
    )

    client.twins.update.assert_called_once_with(
        "child-uuid",
        attach_to_twin_uuid=RACK,
        attach_to_link=None,
        attach_offset_x=0.105,
        attach_offset_y=0.0,
        attach_offset_z=0.23335,
    )


def test_add_twin_without_dock_to_keeps_the_old_behaviour():
    scene, client = make_scene()

    twin = scene.add_twin(
        "cyberwave/server-rack-42u", name="rack-48", position=[1, 2, 3], fixed_base=True
    )

    assert twin is client.twin.return_value
    client.twins.update.assert_not_called()
    client._api_client.call_api.assert_called_once()
    assert client.twin.call_args.kwargs["position_z"] == 3.0


def test_dock_offsets_need_dock_to():
    scene, client = make_scene()
    with pytest.raises(ValueError, match="need dock_to"):
        scene.add_twin("cyberwave/server-2u", dock_offset=(0, 0, 1))
    with pytest.raises(ValueError, match="need dock_to"):
        scene.add_twin("cyberwave/server-2u", dock_rotation=(0, 0, 0, 1))
    client.twin.assert_not_called()


def test_bad_dock_arguments_are_rejected_before_the_twin_is_created():
    scene, client = make_scene()
    with pytest.raises(ValueError, match="quaternion"):
        scene.add_twin("cyberwave/server-2u", dock_to=RACK, dock_rotation=(0, 0, 1))
    with pytest.raises(ValueError, match="x, y, z"):
        scene.add_twin("cyberwave/server-2u", dock_to=RACK, dock_offset=(0, 1))
    client.twin.assert_not_called()
    client.twins.update.assert_not_called()

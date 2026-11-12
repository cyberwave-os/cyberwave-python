"""Environment creation selects or creates a project before posting the environment."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.exceptions import CyberwaveAPIError
from cyberwave.resources import EnvironmentManager


@pytest.mark.parametrize("project_id", [None, "explicit-project"])
@pytest.mark.parametrize("has_projects", [False, True])
def test_create_environment_selects_project(project_id, has_projects):
    api = MagicMock()
    api.src_app_api_projects_list_projects.return_value = (
        [SimpleNamespace(uuid="existing-project")] if has_projects else []
    )
    api.src_app_api_projects_create_project.return_value = SimpleNamespace(
        uuid="default-project"
    )
    expected = api.src_app_api_environments_create_environment_for_project.return_value

    result = EnvironmentManager(api).create(
        "HLS test", project_id=project_id, description="Test capture"
    )

    assert result is expected
    creates_default = project_id is None and not has_projects
    if project_id is None:
        api.src_app_api_projects_list_projects.assert_called_once_with()
    else:
        api.src_app_api_projects_list_projects.assert_not_called()
    if creates_default:
        api.src_app_api_projects_create_project.assert_called_once()
        project = api.src_app_api_projects_create_project.call_args.args[0]
        assert project.name == "Default Folder"
        assert project.workspace_uuid is None  # backend chooses the preferred workspace
    else:
        api.src_app_api_projects_create_project.assert_not_called()
    api.src_app_api_environments_create_environment_for_project.assert_called_once()
    selected, payload = (
        api.src_app_api_environments_create_environment_for_project.call_args.args
    )
    assert selected == (
        project_id or ("existing-project" if has_projects else "default-project")
    )
    assert payload.name == "HLS test"
    assert payload.description == "Test capture"


def test_create_environment_reports_default_project_failure():
    api = MagicMock()
    api.src_app_api_projects_list_projects.return_value = []
    api.src_app_api_projects_create_project.side_effect = RuntimeError("No workspace")

    with pytest.raises(CyberwaveAPIError, match="No workspace"):
        EnvironmentManager(api).create("HLS test")

    api.src_app_api_environments_create_environment_for_project.assert_not_called()

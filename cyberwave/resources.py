"""
Resource managers for Cyberwave API entities
"""

from typing import List, Optional, Dict, Any

from cyberwave.rest import (
    DefaultApi,
    WorkspaceSchema,
    WorkspaceResponseSchema,
    WorkspaceUpdateSchema,
    ProjectSchema,
    ProjectCreateSchema,
    EnvironmentSchema,
    EnvironmentCreateSchema,
    AssetSchema,
    AssetCreateSchema,
    AssetUpdateSchema,
    TwinSchema,
    TwinCreateSchema,
    TwinStateUpdateSchema,
    JointStatesSchema,
    JointStateUpdateSchema,
    JointStateSchema,
)

from .exceptions import CyberwaveAPIError


class BaseResourceManager:
    """Base class for resource managers"""

    def __init__(self, api_client: DefaultApi):
        self.api = api_client

    def _handle_error(self, e: Exception, operation: str):
        """Handle API errors consistently"""
        if hasattr(e, "status"):
            body = getattr(e, "body", None)
            response_dict = body if isinstance(body, dict) else None

            # Try to extract request headers if available
            request_headers = None
            if hasattr(e, "request_headers"):
                request_headers = e.request_headers

            raise CyberwaveAPIError(
                f"Failed to {operation}: {str(e)}",
                status_code=int(e.status) if hasattr(e.status, "__int__") else None,
                response_data=response_dict,
                request_headers=request_headers,
            )
        raise CyberwaveAPIError(f"Failed to {operation}: {str(e)}")


class WorkspaceManager(BaseResourceManager):
    """Manager for workspace operations"""

    def list(self) -> List[WorkspaceSchema]:
        """List all workspaces"""
        try:
            response = self.api.src_users_api_workspaces_list_workspaces()
            return response
        except Exception as e:
            self._handle_error(e, "list workspaces")
            raise  # For type checker

    def get(self, workspace_id: str) -> WorkspaceResponseSchema:
        """Get workspace by ID"""
        try:
            return self.api.src_users_api_workspaces_get_workspace(workspace_id)
        except Exception as e:
            self._handle_error(e, f"get workspace {workspace_id}")
            raise  # For type checker

    def update(self, workspace_id: str, data: Dict[str, Any]) -> WorkspaceSchema:
        """Update workspace"""
        try:
            update_schema = WorkspaceUpdateSchema(**data)
            result = self.api.src_users_api_workspaces_update_workspace(
                workspace_id, update_schema
            )
            return result
        except Exception as e:
            self._handle_error(e, f"update workspace {workspace_id}")
            raise  # For type checker


class ProjectManager(BaseResourceManager):
    """Manager for project operations"""

    def list(self, workspace_id: Optional[str] = None) -> List[ProjectSchema]:
        """List all projects, optionally filtered by workspace"""
        try:
            return self.api.src_app_api_projects_list_projects()
        except Exception as e:
            self._handle_error(e, "list projects")
            raise  # For type checker

    def get(self, project_id: str) -> ProjectSchema:
        """Get project by ID"""
        try:
            return self.api.src_app_api_projects_get_project(project_id)
        except Exception as e:
            self._handle_error(e, f"get project {project_id}")
            raise  # For type checker

    def create(
        self, name: str, workspace_id: str, description: str = "", **kwargs
    ) -> ProjectSchema:
        """Create a new project"""
        try:
            create_schema = ProjectCreateSchema(
                name=name,
                description=description,
                workspace_uuid=workspace_id,
                **kwargs,
            )
            return self.api.src_app_api_projects_create_project(create_schema)
        except Exception as e:
            self._handle_error(e, "create project")
            raise  # For type checker

    def delete(self, project_id: str) -> None:
        """Delete a project"""
        try:
            self.api.src_app_api_projects_delete_project(project_id)
        except Exception as e:
            self._handle_error(e, f"delete project {project_id}")


class EnvironmentManager(BaseResourceManager):
    """Manager for environment operations"""

    def list(self, project_id: Optional[str] = None) -> List[EnvironmentSchema]:
        """List all environments, optionally filtered by project"""
        try:
            if project_id:
                return self.api.src_app_api_environments_list_environments_for_project(
                    project_id
                )
            else:
                return self.api.src_app_api_environments_list_all_environments()
        except Exception as e:
            self._handle_error(e, "list environments")
            raise  # For type checker

    def get(self, environment_id: str) -> EnvironmentSchema:
        """Get environment by ID"""
        try:
            return self.api.src_app_api_environments_get_environment(environment_id)
        except Exception as e:
            self._handle_error(e, f"get environment {environment_id}")
            raise  # For type checker

    def create(
        self, name: str, project_id: str, description: str = "", **kwargs
    ) -> EnvironmentSchema:
        """Create a new environment"""
        try:
            create_schema = EnvironmentCreateSchema(
                name=name, description=description, **kwargs
            )
            return self.api.src_app_api_environments_create_environment_for_project(
                project_id, create_schema
            )
        except Exception as e:
            self._handle_error(e, "create environment")
            raise  # For type checker

    def delete(self, environment_id: str, project_id: str) -> None:
        """Delete an environment"""
        try:
            self.api.src_app_api_environments_delete_environment_for_project(
                project_id, environment_id
            )
        except Exception as e:
            self._handle_error(e, f"delete environment {environment_id}")


class AssetManager(BaseResourceManager):
    """Manager for asset operations"""

    def list(self, workspace_id: Optional[str] = None) -> List[AssetSchema]:
        """List all assets, optionally filtered by workspace"""
        try:
            return self.api.src_app_api_assets_list_assets()
        except Exception as e:
            self._handle_error(e, "list assets")
            raise  # For type checker

    def get(self, asset_id: str) -> AssetSchema:
        """Get asset by ID"""
        try:
            return self.api.src_app_api_assets_get_asset(asset_id)
        except Exception as e:
            self._handle_error(e, f"get asset {asset_id}")
            raise  # For type checker

    def create(self, name: str, description: str = "", **kwargs) -> AssetSchema:
        """Create a new asset"""
        try:
            create_schema = AssetCreateSchema(
                name=name, description=description, **kwargs
            )
            return self.api.src_app_api_assets_create_asset(create_schema)
        except Exception as e:
            self._handle_error(e, "create asset")
            raise  # For type checker

    def update(self, asset_id: str, data: Dict[str, Any]) -> AssetUpdateSchema:
        """Update an asset"""
        try:
            update_schema = AssetUpdateSchema(**data)
            return self.api.src_app_api_assets_update_asset(asset_id, update_schema)
        except Exception as e:
            self._handle_error(e, f"update asset {asset_id}")
            raise  # For type checker

    def delete(self, asset_id: str) -> None:
        """Delete an asset"""
        try:
            self.api.src_app_api_assets_delete_asset(asset_id)
        except Exception as e:
            self._handle_error(e, f"delete asset {asset_id}")

    def search(self, query: str) -> List[AssetSchema]:
        """Search for assets by name or tags"""
        try:
            # Assuming there's a search endpoint - adjust if needed
            all_assets = self.list()
            # search by exact match with registry_id first
            candidates = [a for a in all_assets if a.registry_id == query]
            if candidates:
                return candidates
            else:
                # search by name or description
                candidates = [
                    a
                    for a in all_assets
                    if query.lower() in a.name.lower()
                    or query.lower() in a.description.lower()
                ]
                if candidates:
                    return candidates
                else:
                    return []
        except Exception as e:
            self._handle_error(e, "search assets")
            raise  # For type checker


class TwinManager(BaseResourceManager):
    """Manager for digital twin operations"""

    def list(self, environment_id: Optional[str] = None) -> List[TwinSchema]:
        """List all twins, optionally filtered by environment"""
        try:
            if environment_id:
                return self.api.src_app_api_environments_get_environment_twins(
                    environment_id
                )
            else:
                return self.api.src_app_api_twins_list_all_twins()
        except Exception as e:
            self._handle_error(e, "list twins")
            raise  # For type checker

    def get(self, twin_id: str) -> TwinSchema:
        """Get twin by ID"""
        try:
            return self.api.src_app_api_twins_get_twin(twin_id)
        except Exception as e:
            self._handle_error(e, f"get twin {twin_id}")
            raise  # For type checker

    def create(self, asset_id: str, environment_id: str, **kwargs) -> TwinSchema:
        """Create a new twin instance"""
        try:
            create_schema = TwinCreateSchema(
                asset_uuid=asset_id, environment_uuid=environment_id, **kwargs
            )
            return self.api.src_app_api_twins_create_twin(create_schema)
        except Exception as e:
            self._handle_error(e, "create twin")
            raise  # For type checker

    def update_state(self, twin_id: str, data: Dict[str, Any]) -> TwinSchema:
        """Update twin state (position, rotation, scale)"""
        try:
            update_schema = TwinStateUpdateSchema(**data)
            return self.api.src_app_api_twins_update_twin_state(twin_id, update_schema)
        except Exception as e:
            self._handle_error(e, f"update twin state {twin_id}")
            raise  # For type checker

    def delete(self, twin_id: str) -> None:
        """Delete a twin"""
        try:
            self.api.src_app_api_twins_delete_twin(twin_id)
        except Exception as e:
            self._handle_error(e, f"delete twin {twin_id}")

    def get_joint_states(self, twin_id: str) -> JointStatesSchema:
        """Get current joint states for a twin"""
        try:
            return self.api.src_app_api_urdf_get_twin_joint_states(twin_id)
        except Exception as e:
            self._handle_error(e, f"get joint states for twin {twin_id}")
            raise  # For type checker

    def update_joint_state(
        self,
        twin_id: str,
        joint_name: str,
        position: float,
        velocity: Optional[float] = None,
        effort: Optional[float] = None,
    ) -> JointStateSchema:
        """Update a single joint state"""
        try:
            update_schema = JointStateUpdateSchema(
                position=position, velocity=velocity, effort=effort
            )
            return self.api.src_app_api_urdf_update_twin_joint_state(
                twin_id, joint_name, update_schema
            )
        except Exception as e:
            self._handle_error(e, f"update joint state for twin {twin_id}")
            raise  # For type checker

import httpx
import os
import json
from typing import List, Optional, Dict, Any, Union, Literal
import logging
import re
import uuid
import sys
import getpass

# Try importing keyring, but don't fail if it's not installed
try:
    import keyring
    from keyring.errors import KeyringError
    _keyring_available = True
except ImportError:
    keyring = None
    # Define dummy class if keyring not installed to prevent NameError in except blocks
    class KeyringError(Exception):
        pass 
    _keyring_available = False
    logging.getLogger("cyberwave.sdk").info("'keyring' library not found. Falling back to JSON file cache for tokens.")

from .geometry import (
    Mesh, Skeleton, Joint, FloorPlan, Sensor, Zone,
    log_mesh_rr, log_skeleton_rr
)
import aiofiles
import numpy as np
import dataclasses
from .level.schema import LevelDefinition
import warnings
from .constants import (
    DEFAULT_BACKEND_URL,
    BACKEND_URL_ENV_VAR,
    USERNAME_ENV_VAR,
    PASSWORD_ENV_VAR,
)

# Basic logger for the SDK
logger = logging.getLogger("cyberwave.sdk")
logging.basicConfig(level=logging.INFO) # Configure basic logging

# --- Constants ---
API_VERSION_PREFIX = "/api/v1"
# Use standard Authorization header for Bearer tokens
SHARE_TOKEN_HEADER = "Authorization" 
# Simple file-based cache for the token (optional)
TOKEN_CACHE_DIR = os.path.expanduser("~/.cyberwave")
TOKEN_CACHE_FILE = os.path.join(TOKEN_CACHE_DIR, "token_cache.json")
KEYRING_SERVICE_NAME = "cyberwave-sdk" # Service name for keyring

class CyberWaveError(Exception):
    """Base exception for CyberWave client errors."""
    pass

class AuthenticationError(CyberWaveError):
    """Error related to authentication or session tokens."""
    pass

class APIError(CyberWaveError):
    """Error returned from the backend API."""
    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


def _extract_error_detail(response: httpx.Response) -> Any:
    """Helper to safely extract error detail from response."""
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text

# --- Helper for dataclass serialization --- 
# (Needed because default json.dumps doesn't handle numpy arrays)
class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            # Convert numpy arrays to lists
            d = dataclasses.asdict(o)
            for key, value in d.items():
                if isinstance(value, np.ndarray):
                    d[key] = value.tolist()
                elif isinstance(value, list) and value and dataclasses.is_dataclass(value[0]):
                     d[key] = [self.default(item) for item in value] # Recurse for lists of dataclasses
                elif dataclasses.is_dataclass(value):
                     d[key] = self.default(value) # Recurse for nested dataclasses
            return d
        return super().default(o)

class Client:
    """Asynchronous client for interacting with the CyberWave Backend API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        use_token_cache: bool = True,
        timeout: float = 10.0,
        debug: bool = False,
    ):
        """
        Initializes the CyberWave client.

        Args:
            base_url: The base URL of the CyberWave backend. If ``None``,
                the value of ``BACKEND_URL_ENV_VAR`` will be checked and
                used if set, otherwise ``DEFAULT_BACKEND_URL`` is used.
            use_token_cache: If True, attempts to load/save the share token
                from/to ``~/.cyberwave_token_cache.json``.
            timeout: Request timeout in seconds.
            debug: If True, enables verbose debug logging.
        """
        if base_url is None:
            base_url = os.getenv(BACKEND_URL_ENV_VAR, DEFAULT_BACKEND_URL)
        
        # Ensure API prefix is present
        api_base_url = base_url
        if API_VERSION_PREFIX not in base_url:
             if base_url.endswith("/"):
                 api_base_url = base_url.rstrip("/") + API_VERSION_PREFIX
             else:
                 api_base_url = base_url + API_VERSION_PREFIX
        
        # Set debug mode and configure logging
        self._debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
            # Add a console handler if none exists
            if not logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(console_handler)
            logger.debug(f"Debug mode enabled. Verbose logging activated.")
        
        logger.info(f"Initializing CyberWave Client for backend: {api_base_url}")
        self._client = httpx.AsyncClient(base_url=api_base_url, timeout=timeout)
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._use_token_cache = use_token_cache
        self._session_info: Dict[str, Any] = {}

        if self._use_token_cache:
            self._load_token_from_cache()

    async def __aenter__(self) -> "Client":
        """Allow usage of ``async with Client()`` for automatic closing."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def _load_token_from_cache(self):
        loaded_from_keyring = False
        if _keyring_available:
            try:
                logger.debug(f"Attempting to load tokens from keyring service: {KEYRING_SERVICE_NAME}")
                access_token_kr = keyring.get_password(KEYRING_SERVICE_NAME, "access_token")
                refresh_token_kr = keyring.get_password(KEYRING_SERVICE_NAME, "refresh_token")
                session_info_json = keyring.get_password(KEYRING_SERVICE_NAME, "session_info")
                
                if access_token_kr or refresh_token_kr:
                    self._access_token = access_token_kr
                    self._refresh_token = refresh_token_kr
                    try:
                        self._session_info = json.loads(session_info_json) if session_info_json else {}
                    except json.JSONDecodeError:
                         logger.warning("Failed to decode session info from keyring.")
                         self._session_info = {}
                    
                    loaded_from_keyring = True
                    if self._access_token: logger.info(f"Loaded access token from keyring: {self._access_token[:4]}...{self._access_token[-4:]}")
                    if self._refresh_token: logger.info("Loaded refresh token from keyring.")
                else:
                    logger.debug("No tokens found in keyring.")
                    
            except KeyringError as e:
                logger.warning(f"Keyring error while loading tokens: {e}. Keyring might be locked or unavailable.")
            except Exception as e:
                 logger.warning(f"Unexpected error interacting with keyring during load: {e}")

        # Fallback to JSON file cache if not loaded from keyring and caching is enabled
        if not loaded_from_keyring and self._use_token_cache:
            logger.debug(f"Falling back to JSON token cache file: {TOKEN_CACHE_FILE}")
            try: # Inner try for JSON file access
                if os.path.exists(TOKEN_CACHE_FILE):
                    os.makedirs(TOKEN_CACHE_DIR, exist_ok=True)
                    with open(TOKEN_CACHE_FILE, 'r') as f:
                        cache_data = json.load(f)
                        # Only load from file if not already loaded from keyring
                        if not self._access_token: self._access_token = cache_data.get("access_token")
                        if not self._refresh_token: self._refresh_token = cache_data.get("refresh_token")
                        # Load session info, prioritizing any from keyring if loaded
                        if not self._session_info:
                             self._session_info = cache_data.get("session_info", {})
                        else: # If keyring loaded info, merge file info cautiously
                             file_session_info = cache_data.get("session_info", {})
                             # Example: merge only if key doesn't exist from keyring
                             for k, v in file_session_info.items():
                                 self._session_info.setdefault(k, v)
                        
                        if self._access_token: logger.info(f"Loaded access token from JSON cache: {self._access_token[:4]}...{self._access_token[-4:]}")
                        if self._refresh_token: logger.info("Loaded refresh token from JSON cache.")
                        if not self._access_token and not self._refresh_token:
                            logger.info("Token cache file found but contained no tokens.")
                else:
                    logger.info("Token cache file not found.")
            except Exception as e: # Added missing except for inner try block
                logger.warning(f"Failed to load token from JSON cache ({TOKEN_CACHE_FILE}): {e}")

    def _save_token_to_cache(self):
        saved_to_keyring = False
        if _keyring_available:
            try:
                logger.debug(f"Attempting to save tokens to keyring service: {KEYRING_SERVICE_NAME}")
                if self._access_token:
                    keyring.set_password(KEYRING_SERVICE_NAME, "access_token", self._access_token)
                else: # Delete if None
                     try: keyring.delete_password(KEYRING_SERVICE_NAME, "access_token")
                     except keyring.errors.PasswordDeleteError: pass # Ignore if not found
                     except AttributeError: pass # Handle case where keyring.errors doesn't exist
                
                if self._refresh_token:
                    keyring.set_password(KEYRING_SERVICE_NAME, "refresh_token", self._refresh_token)
                else: # Delete if None
                    try: keyring.delete_password(KEYRING_SERVICE_NAME, "refresh_token")
                    except keyring.errors.PasswordDeleteError: pass # Ignore if not found
                    except AttributeError: pass # Handle case where keyring.errors doesn't exist

                # Save session info as JSON string
                session_info_json = json.dumps(self._session_info)
                keyring.set_password(KEYRING_SERVICE_NAME, "session_info", session_info_json)
                
                saved_to_keyring = True
                logger.info("Tokens saved securely to system keyring.")
                
                # Optionally remove the insecure JSON cache file if keyring save succeeds
                if os.path.exists(TOKEN_CACHE_FILE):
                    try:
                        os.remove(TOKEN_CACHE_FILE)
                        logger.info(f"Removed insecure JSON token cache file ({TOKEN_CACHE_FILE}) as tokens were saved to keyring.")
                    except Exception as e:
                         logger.warning(f"Could not remove old JSON cache file: {e}")
                         
            except KeyringError as e:
                logger.warning(f"Keyring error while saving tokens: {e}. Keyring might be locked or unavailable. Falling back to JSON file cache.")
            except Exception as e:
                 logger.warning(f"Unexpected error interacting with keyring during save: {e}. Falling back to JSON file cache.")

        # Fallback to JSON file cache if keyring failed and caching is enabled
        if not saved_to_keyring and self._use_token_cache:
            logger.debug(f"Saving tokens to JSON cache file: {TOKEN_CACHE_FILE}")
            try:
                os.makedirs(TOKEN_CACHE_DIR, exist_ok=True)
                cache_data = {
                    "access_token": self._access_token,
                    "refresh_token": self._refresh_token,
                    "session_info": self._session_info # Session info now includes expires_in
                }
                with open(TOKEN_CACHE_FILE, 'w') as f:
                    json.dump(cache_data, f)
                logger.info(f"Saved session details to JSON cache: {TOKEN_CACHE_FILE}")
            except Exception as e:
                logger.warning(f"Failed to save token to JSON cache: {e}")
            
    def _clear_token_cache(self):
        """Removes tokens from keyring and the JSON cache file."""
        cleared_keyring = False
        if _keyring_available:
            try:
                logger.debug(f"Attempting to delete tokens from keyring service: {KEYRING_SERVICE_NAME}")
                # Use try-except for each delete in case one fails or doesn't exist
                try: keyring.delete_password(KEYRING_SERVICE_NAME, "access_token")
                except keyring.errors.PasswordDeleteError: pass
                except AttributeError: pass
                
                try: keyring.delete_password(KEYRING_SERVICE_NAME, "refresh_token")
                except keyring.errors.PasswordDeleteError: pass
                except AttributeError: pass
                
                try: keyring.delete_password(KEYRING_SERVICE_NAME, "session_info")
                except keyring.errors.PasswordDeleteError: pass
                except AttributeError: pass

                cleared_keyring = True
                logger.info("Attempted to delete tokens from system keyring.") # Changed log msg
            # except keyring.errors.PasswordDeleteError: # This is handled inside now
            #      logger.debug("No tokens found in keyring to delete.") 
            #      cleared_keyring = True 
            except KeyringError as e:
                logger.warning(f"Keyring error while deleting tokens: {e}.")
            except Exception as e:
                 logger.warning(f"Unexpected error interacting with keyring during delete: {e}")
        
        # Always try to remove the JSON file cache if it exists
        if os.path.exists(TOKEN_CACHE_FILE):
            try:
                os.remove(TOKEN_CACHE_FILE)
                logger.info(f"Removed JSON token cache file: {TOKEN_CACHE_FILE}")
            except Exception as e:
                 logger.warning(f"Failed to remove JSON token cache file: {e}")
                 
        # Clear in-memory tokens regardless
        self._access_token = None 
        self._refresh_token = None 
        self._session_info = {} # Clear session info including expires_in

    async def aclose(self):
        """Closes the underlying HTTP client session."""
        await self._client.aclose()
        logger.info("CyberWave Client closed.")

    def has_active_session(self) -> bool:
        """Checks if an access token is currently loaded."""
        return self._access_token is not None

    def get_session_token(self) -> Optional[str]:
        """Returns the currently loaded access token, if any."""
        return self._access_token
        
    def get_session_info(self) -> Dict[str, Any]:
        """Returns cached information about the current session (e.g., share_url)."""
        return self._session_info

    def _get_headers(self, require_auth: bool = True) -> Dict[str, str]:
        """Returns headers for API requests, including Authorization if available."""
        headers = {
            "Accept": "application/json",
            # Add other common headers if needed
        }
        if self._access_token:
            headers[SHARE_TOKEN_HEADER] = f"Bearer {self._access_token}"
        elif require_auth: # Added check here as well for consistency, though _request checks first
            # This part might be redundant now due to the check in _request
            # but serves as a safeguard within header generation itself.
            logger.warning("Auth required but no access token found for header generation.")
            # Consider if raising here is better/worse than in _request
            # For now, rely on _request check raising the error.
            pass
        return headers

    async def add_robot(
        self,
        name: str,
        robot_type: str,
        level_id: Optional[int] = None,
        serial_number: Optional[str] = None,
        status: str = "unknown",
        capabilities: Optional[List[str]] = None,
        initial_pos_x: Optional[float] = None,
        initial_pos_y: Optional[float] = None,
        initial_pos_z: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        current_battery_percentage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Registers a new robot with the backend.
        (Handles token refresh automatically)

        If no active session token exists and level_id is None,
        this might trigger the creation of a new temporary session on the backend
        (depending on backend support). The token will be automatically stored.
        
        Args:
            name: The name of the robot.
            robot_type: The type identifier (e.g., 'agv/model-x').
            level_id: Optional ID of the level to assign the robot to.
                      If None and no active session, a temporary level is created.
                      If provided while a session token exists, it must match the session's level.
            serial_number: Optional serial number.
            status: Initial status (e.g., 'idle', 'charging'). Defaults to 'unknown'.
            capabilities: Optional list of robot capabilities.
            initial_pos_x/y/z: Optional initial coordinates.
            metadata: Optional dictionary for extra data.
            current_battery_percentage: Optional initial battery level.

        Returns:
            A dictionary representing the created robot, potentially including
            'share_token' and 'share_url' if a new session was created.

        Raises:
            APIError: If the backend returns an error.
            CyberWaveError: For other client-side errors.
        """
        payload = {
            "name": name,
            "robot_type": robot_type,
            "level_id": level_id,
            "serial_number": serial_number,
            "status": status,
            "capabilities": capabilities,
            "initial_pos_x": initial_pos_x,
            "initial_pos_y": initial_pos_y,
            "initial_pos_z": initial_pos_z,
            "metadata": metadata,
            "current_battery_percentage": current_battery_percentage,
        }
        clean_payload = payload
        
        logger.debug(f"Sending POST /robots request. Payload: {clean_payload}")

        try:
            response = await self._request(
                method="POST",
                url="/robots",
                json=clean_payload,
            )
            response_data = response.json()
            logger.debug(f"POST /robots successful. Response: {response_data}")

            # Check if a new session was created (token returned in body/header)
            returned_token = response_data.get("share_token") or response.headers.get(SHARE_TOKEN_HEADER)
            
            if returned_token and not self._access_token:
                logger.info(f"New temporary session token received via add_robot: {returned_token[:4]}...{returned_token[-4:]}")
                self._access_token = returned_token
                self._session_info["share_url"] = response_data.get("share_url")
                self._session_info["level_id"] = response_data.get("level_id")
                self._save_token_to_cache()
            elif returned_token and self._access_token and returned_token != self._access_token:
                 logger.warning("add_robot response contained a different token than the active session. Ignoring.")

            return response_data

        except (APIError, AuthenticationError, CyberWaveError) as e:
            logger.error(f"Failed to add robot: {e}")
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from add_robot: {response.text}", exc_info=True)
            raise CyberWaveError(f"Invalid JSON response received: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error in add_robot: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred while adding robot: {e}") from e
             
    # ------------------------------------------------------------------
    # NEW 1/2  –  upload_mesh()
    # ------------------------------------------------------------------
    async def upload_mesh(
        self,
        project_id: int,
        mesh: Mesh,
        target_level_id: int | None = None,
    ) -> dict:
        """Attach a 3-D mesh to a project (optionally at a given level)."""
        headers = self._get_headers()
        
        # 1. Send to CyberWave backend  (multipart so we can stream file)
        try:
            async with aiofiles.open(mesh.path, "rb") as f:
                files = {"file": (mesh.path.name, await f.read(), mesh.mime_type)}
                payload = {
                    "transform": mesh.transform.tolist(),
                    "metadata": mesh.metadata,
                    "level_id": target_level_id,
                }
                logger.debug(f"Sending POST /projects/{project_id}/meshes. Headers: {headers.keys()}, Payload keys: {payload.keys()}, File: {mesh.path.name}")
                response = await self._client.post(
                    f"/projects/{project_id}/meshes",
                    data={"payload": json.dumps(payload)},
                    files=files,
                    headers=headers
                )
            response.raise_for_status()

            response_data = response.json()
            logger.debug(f"POST /projects/{project_id}/meshes successful. Response: {response_data}")

            # 2. OPTIONAL: immediately log to a local Rerun viewer
            try:
                log_mesh_rr(f"project_{project_id}/meshes/{response_data['id']}", mesh)
            except Exception as log_e:
                logger.warning(f"Failed to log mesh to Rerun: {log_e}")

            return response_data

        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during upload_mesh: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except FileNotFoundError:
            logger.error(f"Mesh file not found: {mesh.path}")
            raise CyberWaveError(f"Mesh file not found: {mesh.path}")
        except httpx.RequestError as e:
            logger.error(f"Request failed during upload_mesh: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during upload_mesh: {e}", exc_info=True)
            raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    # ------------------------------------------------------------------
    # NEW 2/2  –  upload_skeleton()
    # ------------------------------------------------------------------
    async def upload_skeleton(
        self,
        project_id: int,
        skeleton: Skeleton,
        semantic_name: str = "robot_arm",
    ) -> dict:
        """Attach a skeleton (joint tree) to a project."""
        headers = self._get_headers()
        
        try:
            joints_serialised = [
                {
                    "name": j.name,
                    "parent": j.parent,
                    "pose": j.pose.tolist(),
                }
                for j in skeleton.joints
            ]
            payload = {
                "name": semantic_name,
                "joints": joints_serialised,
                "metadata": skeleton.metadata,
            }
            logger.debug(f"Sending POST /projects/{project_id}/skeletons. Headers: {headers.keys()}, Payload keys: {payload.keys()}")
            response = await self._client.post(
                f"/projects/{project_id}/skeletons",
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            response_data = response.json()
            logger.debug(f"POST /projects/{project_id}/skeletons successful. Response: {response_data}")
            
            # OPTIONAL: log to Rerun
            try:
                log_skeleton_rr(f"project_{project_id}/skeletons/{response_data['id']}", skeleton)
            except Exception as log_e:
                logger.warning(f"Failed to log skeleton to Rerun: {log_e}")
                
            return response_data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during upload_skeleton: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during upload_skeleton: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during upload_skeleton: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e
             
    # ------------------------------------------------------------------
    # NEW - Workspace Methods
    # ------------------------------------------------------------------
    async def create_workspace(self, name: str, slug: Optional[str] = None) -> Dict[str, Any]: # MODIFIED
        """
        Creates a new workspace.

        Args:
            name: The name of the workspace.
            slug: Optional. The URL-friendly slug for the workspace. 
                  If not provided, the backend may attempt to generate it or it might be required.

        Returns:
            A dictionary representing the created workspace.

        Raises:
            APIError: If the API returns an error.
            CyberWaveError: For other client-side errors.
        """
        payload = {"name": name}
        if slug:
            payload["slug"] = slug
        else:
            # Fallback: attempt to generate a simple slug if not provided.
            # This matches the logic in the importer script for consistency,
            # but ideally, the backend should handle slug generation if it's optional.
            # For now, we ensure slug is always sent based on backend error.
            generated_slug = re.sub(r'\s+', '-', name.lower())
            generated_slug = re.sub(r'[^a-z0-9-]', '', generated_slug)
            payload["slug"] = generated_slug.strip('-')
            if not payload["slug"]: # Handle cases where name results in empty slug
                payload["slug"] = str(uuid.uuid4())[:8] # Fallback to a short UUID part
            logger.info(f"Slug not provided for workspace '{name}', generated slug: '{payload['slug']}'")

        try:
            response = await self._request(  # Use _request helper
                method="POST",
                url="/workspaces",
                json=payload,
                require_auth=False  # Workspace creation should not require auth
            )
            data = response.json()
            token = data.get("access_token")
            if token:
                self._access_token = token
            return data
        except APIError as e:
            logger.error(f"API error during create_workspace ('{name}'): {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during create_workspace ('{name}'): {e}")
            # Wrap other exceptions (like httpx.RequestError) into CyberWaveError
            raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    async def get_workspace_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific workspace by its unique slug.
        Returns None if not found, raises APIError for other errors.
        """
        endpoint = f"/workspaces/slug/{slug}"
        logger.debug(f"Sending GET {endpoint}")
        try:
            # Use the _request helper which handles parsing and errors
            response = await self._request("GET", endpoint)
            return response.json()
        except APIError as e:
            if e.status_code == 404:
                logger.info(f"Workspace with slug '{slug}' not found.")
                return None
            raise e # Re-raise other API errors
        except Exception as e:
            logger.error(f"Unexpected error getting workspace by slug '{slug}': {e}")
            raise CyberWaveError(f"Unexpected error: {e}") from e

    async def get_workspaces(self) -> List[Dict[str, Any]]:
        """Retrieves a list of all accessible workspaces.
        (Handles token refresh automatically)
        """
        # headers = self._get_headers() # Removed: Handled by _request
        
        logger.debug("Sending GET /workspaces.")
        try:
            response = await self._request( # Use _request helper
                method="GET", 
                url="/workspaces"
                # No payload for GET
            )
            # _request already called raise_for_status
            return response.json()
        except (APIError, AuthenticationError, CyberWaveError) as e:
            # Errors raised by _request or _attempt_refresh
            logger.error(f"Failed to get workspaces: {e}")
            raise e # Re-raise the original exception
        except json.JSONDecodeError as e:
            # Error decoding successful response
            logger.error(f"Failed to decode JSON response from get_workspaces: {response.text}", exc_info=True)
            raise CyberWaveError(f"Invalid JSON response received: {e}") from e
        except Exception as e:
            # Catch any other unexpected error specific to this method's logic
            logger.error(f"Unexpected error in get_workspaces: {e}", exc_info=True)
            raise CyberWaveError(f"An unexpected error occurred while getting workspaces: {e}") from e

    # ------------------------------------------------------------------
    # NEW - Project Methods
    # ------------------------------------------------------------------
    async def create_project(self, workspace_id: int, name: str) -> Dict[str, Any]:
        """Creates a new project within a specific workspace.
        (Handles token refresh automatically)
        """
        # headers = self._get_headers() # Removed
        if not self._access_token:
             # This check might be redundant if _request handles 401, but keep for clarity?
             # Or rely solely on _request raising AuthenticationError if token missing/invalid.
             # Let's rely on _request for now.
             pass # _request will handle missing token via _get_headers
        
        payload = {"name": name, "workspace_id": workspace_id} 
        endpoint = f"/workspaces/{workspace_id}/projects/" 
        logger.debug(f"Sending POST {endpoint}. Payload: {payload}")
        try:
            response = await self._request( # Use _request helper
                method="POST",
                url=endpoint,
                json=payload
            )
            # _request already called raise_for_status
            response_data = response.json()
            logger.info(f"Project '{name}' created successfully (ID: {response_data.get('id')}) in Workspace {workspace_id}.")
            return response_data
        except (APIError, AuthenticationError, CyberWaveError) as e:
            logger.error(f"Failed to create project: {e}")
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from create_project: {response.text}", exc_info=True)
            raise CyberWaveError(f"Invalid JSON response received: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error in create_project: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    async def get_projects(self, workspace_id: int) -> List[Dict[str, Any]]:
        """Retrieves projects within a specific workspace."""
        headers = self._get_headers()
        if not self._access_token:
             raise AuthenticationError("Cannot get projects without an active session token.")
        
        endpoint = f"/workspaces/{workspace_id}/projects/" # Use nested endpoint
        logger.debug(f"Sending GET {endpoint}.")
        try:
            response = await self._client.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during get_projects: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during get_projects: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during get_projects: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    # ------------------------------------------------------------------
    # Environment Methods (preferred over deprecated Level APIs)
    # ------------------------------------------------------------------
    async def create_environment(
        self,
        project_uuid: str,
        name: str,
        description: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new environment under a project."""
        headers = self._get_headers()
        if not self._access_token:
            raise AuthenticationError("Cannot create environment without an active session token.")

        endpoint = f"/projects/{project_uuid}/environments"
        payload = {
            "name": name,
            "description": description,
            "settings": settings or {},
        }
        logger.debug(f"Sending POST {endpoint}. Payload keys: {payload.keys()}")
        try:
            response = await self._client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Environment '{name}' created under project {project_uuid}")
            return data
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            logger.error(f"API Error during create_environment: {e.response.status_code} - {detail}")
            raise APIError(e.response.status_code, detail)
        except httpx.RequestError as e:
            logger.error(f"Request failed during create_environment: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during create_environment: {e}", exc_info=True)
            raise CyberWaveError(f"An unexpected error occurred: {e}")

    async def get_environments(self, project_uuid: str) -> List[Dict[str, Any]]:
        """List environments for a project."""
        headers = self._get_headers()
        if not self._access_token:
            raise AuthenticationError("Cannot get environments without an active session token.")

        endpoint = f"/projects/{project_uuid}/environments"
        logger.debug(f"Sending GET {endpoint}.")
        try:
            response = await self._client.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            logger.error(f"API Error during get_environments: {e.response.status_code} - {detail}")
            raise APIError(e.response.status_code, detail)
        except httpx.RequestError as e:
            logger.error(f"Request failed during get_environments: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during get_environments: {e}", exc_info=True)
            raise CyberWaveError(f"An unexpected error occurred: {e}")

    async def get_environment(self, environment_uuid: str) -> Dict[str, Any]:
        """Fetch a single environment by UUID."""
        headers = self._get_headers()
        if not self._access_token:
            raise AuthenticationError("Cannot get environment without an active session token.")

        endpoint = f"/environments/{environment_uuid}"
        logger.debug(f"Sending GET {endpoint}.")
        try:
            response = await self._client.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            logger.error(f"API Error during get_environment: {e.response.status_code} - {detail}")
            raise APIError(e.response.status_code, detail)
        except httpx.RequestError as e:
            logger.error(f"Request failed during get_environment: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during get_environment: {e}", exc_info=True)
            raise CyberWaveError(f"An unexpected error occurred: {e}")

    # ------------------------------------------------------------------
    # NEW - Level Methods
    # ------------------------------------------------------------------
    async def create_level(
        self,
        workspace_id: int,
        project_id: int,
        name: str,
        floor_number: int,
        floor_plan: Optional[Dict[str, Any]] = None # Use Dict for now
    ) -> Dict[str, Any]:
        """[DEPRECATED] Use environments; levels are deprecated."""
        warnings.warn("create_level is deprecated; use environments", DeprecationWarning, stacklevel=2)
        raise NotImplementedError("Levels are deprecated; use environments")

    async def get_levels(self, workspace_id: int, project_id: int) -> List[Dict[str, Any]]: # Added workspace_id
        """[DEPRECATED] Use environments; levels are deprecated."""
        warnings.warn("get_levels is deprecated; use environments", DeprecationWarning, stacklevel=2)
        raise NotImplementedError("Levels are deprecated; use environments")

    # ------------------------------------------------------------------
    # NEW - Floor Plan Methods
    # ------------------------------------------------------------------
    async def upload_floor_plan(self, level_id: int, floor_plan: FloorPlan) -> Dict:
        """[DEPRECATED] No environment equivalent."""
        warnings.warn("upload_floor_plan is deprecated; no environment equivalent", DeprecationWarning, stacklevel=2)
        raise NotImplementedError("upload_floor_plan not supported")
        headers = self._get_headers()
        if not self._access_token:
             raise AuthenticationError("Cannot upload floor plan without an active session token.")

        endpoint = f"/levels/{level_id}/floorplan" # Assuming PUT or POST endpoint
        try:
            # Serialize FloorPlan, converting numpy arrays
            payload = json.loads(json.dumps(floor_plan, cls=DataclassJSONEncoder))
            logger.debug(f"Sending PUT/POST {endpoint}. Payload keys: {payload.keys()}")
            # Use PUT for update, POST for create - adjust as needed based on backend API
            response = await self._client.put(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"Floor plan for Level {level_id} updated successfully.")
            # TODO: Optional Rerun logging for the floor plan?
            return response_data
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during upload_floor_plan: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during upload_floor_plan: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during upload_floor_plan: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    async def get_floor_plan(self, level_id: int) -> Optional[FloorPlan]: # Return type might be FloorPlan
        """[DEPRECATED] No environment equivalent."""
        warnings.warn("get_floor_plan is deprecated; no environment equivalent", DeprecationWarning, stacklevel=2)
        raise NotImplementedError("get_floor_plan not supported")
        headers = self._get_headers()
        if not self._access_token:
             raise AuthenticationError("Cannot get floor plan without an active session token.")

        endpoint = f"/levels/{level_id}/floorplan"
        logger.debug(f"Sending GET {endpoint}.")
        try:
            response = await self._client.get(endpoint, headers=headers)
            if response.status_code == 404:
                logger.info(f"No floor plan found for Level {level_id}.")
                return None
            response.raise_for_status()
            # TODO: Need robust parsing from JSON back to FloorPlan dataclass, 
            # potentially handling numpy arrays within nested structures.
            # For now, returning raw dict.
            raw_data = response.json()
            logger.info(f"Floor plan retrieved for Level {level_id}.")
            # Ideally parse raw_data into FloorPlan object here
            return raw_data # Placeholder: return raw dict
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during get_floor_plan: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during get_floor_plan: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during get_floor_plan: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    # ------------------------------------------------------------------
    # NEW - Sensor Methods
    # ------------------------------------------------------------------
    async def register_sensor(self, sensor: Sensor) -> Dict:
        """Registers a new sensor instance."""
        headers = self._get_headers()
        # Sensor registration might be token-less if it creates a temporary session?

        endpoint = "/sensors"
        try:
            payload = json.loads(json.dumps(sensor, cls=DataclassJSONEncoder))
            logger.debug(f"Sending POST {endpoint}. Payload keys: {payload.keys()}")
            response = await self._client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            sensor_id = response_data.get('id')
            logger.info(f"Sensor '{sensor.sensor_type}' registered successfully (ID: {sensor_id}).")
            # TODO: Optional Rerun logging for sensor pose?
            return response_data
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during register_sensor: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during register_sensor: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during register_sensor: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    # Note: Updating sensor readings might be better handled via direct streaming (MQTT/WebSockets)
    # But providing a simple HTTP push method for completeness.
    async def update_sensor_reading(self, sensor_id: Union[int, str], reading_data: Dict, timestamp: Optional[float] = None) -> None:
        """Pushes a reading for a specific sensor."""
        headers = self._get_headers()
        # This likely needs authentication

        endpoint = f"/sensors/{sensor_id}/readings"
        payload = {"reading": reading_data, "timestamp_unix": timestamp}
        logger.debug(f"Sending POST {endpoint}. Payload keys: {payload.keys()}")
        try:
            response = await self._client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"Reading submitted for Sensor {sensor_id}.")
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during update_sensor_reading: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during update_sensor_reading: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during update_sensor_reading: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    async def upload_sensor_video_frame(self, sensor_uuid: Union[int, str], frame_bytes: bytes) -> Dict[str, Any]:
        """Upload a single raw video frame to a sensor."""
        headers = self._get_headers()
        try:
            response = await self._client.post(
                f"/sensors/{sensor_uuid}/video",
                content=frame_bytes,
                headers={**headers, "Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            raise APIError(e.response.status_code, detail)
        except Exception as e:
            raise CyberWaveError(f"Failed to upload sensor frame: {e}")

    async def initiate_sensor_segment_upload(self, sensor_uuid: Union[int, str], content_type: str = "image/jpeg") -> Dict[str, Any]:
        """Request a signed URL and storage key for uploading a sensor video/image segment."""
        response = await self._request(
            "POST",
            f"/sensors/{sensor_uuid}/segments/initiate_upload",
            json={"content_type": content_type},
        )
        return response.json()

    async def complete_sensor_segment_upload(self, sensor_uuid: Union[int, str], storage_key: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Notify backend that the segment upload is complete and kick off analysis."""
        response = await self._request(
            "POST",
            f"/sensors/{sensor_uuid}/segments/complete",
            json={"storage_key": storage_key, "meta": meta or {}},
        )
        return response.json()

    # ------------------------------------------------------------------
    # NEW - Zone Methods
    # ------------------------------------------------------------------
    async def define_zone(self, level_id: int, zone: Zone) -> Dict:
        """[DEPRECATED] Zones API is not supported; use environments and sensors."""
        warnings.warn("define_zone is deprecated; use environments", DeprecationWarning, stacklevel=2)
        raise NotImplementedError("define_zone not supported")
        headers = self._get_headers()
        if not self._access_token:
             raise AuthenticationError("Cannot define zone without an active session token.")

        endpoint = f"/levels/{level_id}/zones"
        try:
            payload = json.loads(json.dumps(zone, cls=DataclassJSONEncoder))
            logger.debug(f"Sending POST {endpoint}. Payload keys: {payload.keys()}")
            response = await self._client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            zone_id = response_data.get('id')
            logger.info(f"Zone '{zone.name}' defined successfully (ID: {zone_id}) in Level {level_id}.")
            # TODO: Optional Rerun logging for the zone?
            return response_data
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during define_zone: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during define_zone: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during define_zone: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    async def get_zones(self, level_id: int) -> List[Zone]: # Return type might be List[Zone]
        """[DEPRECATED] Zones API is not supported; use environments and sensors."""
        warnings.warn("get_zones is deprecated; use environments", DeprecationWarning, stacklevel=2)
        raise NotImplementedError("get_zones not supported")
        headers = self._get_headers()
        if not self._access_token:
             raise AuthenticationError("Cannot get zones without an active session token.")

        endpoint = f"/levels/{level_id}/zones"
        logger.debug(f"Sending GET {endpoint}.")
        try:
            response = await self._client.get(endpoint, headers=headers)
            response.raise_for_status()
            # TODO: Parse list of dicts back to List[Zone]
            raw_data = response.json()
            logger.info(f"Retrieved {len(raw_data)} zones for Level {level_id}.")
            return raw_data # Placeholder: return list of raw dicts
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during get_zones: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during get_zones: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during get_zones: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    # ------------------------------------------------------------------
    # NEW - Actuation Method
    # ------------------------------------------------------------------
    async def send_command(
        self,
        target_entity_type: Literal["robot", "actuator", "sensor"], # Added sensor
        target_entity_id: Union[int, str],
        command_name: str,
        command_payload: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Sends a command to a target entity via the backend."""
        headers = self._get_headers()
        if not self._access_token:
             raise AuthenticationError("Cannot send command without an active session token.")

        endpoint = f"/commands" # Assuming a central command endpoint
        payload = {
            "target_entity_type": target_entity_type,
            "target_entity_id": str(target_entity_id), # Ensure ID is string for consistency
            "command_name": command_name,
            "payload": command_payload or {}
        }
        logger.debug(f"Sending POST {endpoint}. Command: {command_name} for {target_entity_type} {target_entity_id}")
        try:
            response = await self._client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"Command '{command_name}' sent successfully to {target_entity_type} {target_entity_id}. Response: {response_data}")
            # Response might be simple ack or contain immediate result/status
            return response_data
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during send_command: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during send_command: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during send_command: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e

    # ------------------------------------------------------------------
    async def upload_level_definition(
        self, 
        project_id: int,
        level_definition: LevelDefinition,
    ) -> Dict[str, Any]:
        """
        Uploads a level definition to the backend.
        
        Args:
            project_id: ID of the project to create the level in
            level_definition: The LevelDefinition object (typically loaded from YAML)
            
        Returns:
            The created level object as returned by the backend
        """
        headers = self._get_headers()
        if not self._access_token:
             raise AuthenticationError("Cannot upload level without an active session token.")
        
        # Extract data from the LevelDefinition for the backend
        name = level_definition.metadata.title
        floor_number = level_definition.metadata.floor_number
        
        # Create the level first
        level = await self.create_level(
            project_id=project_id,
            name=name,
            floor_number=floor_number
        )
        level_id = level["id"]
        logger.info(f"Created level '{name}' with ID {level_id}")
        
        # Now handle entities (robots, fixed assets)
        if level_definition.entities:
            for entity in level_definition.entities:
                if entity.archetype == "robot":
                    # Add robot
                    try:
                        position = None
                        if entity.transform and "position" in entity.transform:
                            position = entity.transform["position"]
                            
                        robot = await self.add_robot(
                            name=entity.id,
                            robot_type=entity.reference or "generic_robot",
                            level_id=level_id,
                            capabilities=entity.capabilities,
                            initial_pos_x=position[0] if position else None,
                            initial_pos_y=position[1] if position and len(position) > 1 else None,
                            initial_pos_z=position[2] if position and len(position) > 2 else None,
                            current_battery_percentage=entity.battery_percentage
                        )
                        logger.info(f"Added robot '{entity.id}' to level {level_id}")
                    except Exception as e:
                        logger.error(f"Failed to add robot '{entity.id}': {e}")
                # Other entity types could be handled here
        
        # Handle zones
        if level_definition.zones:
            for zone_def in level_definition.zones:
                try:
                    # Convert to zone format expected by backend
                    coordinates = []
                    if zone_def.geometry and zone_def.geometry.get("coordinates"):
                        coords = zone_def.geometry["coordinates"]
                        # Flatten the first level if it's a GeoJSON-style Polygon
                        if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], list):
                            coordinates = coords[0]
                        else:
                            coordinates = coords
                    
                    zone = Zone(
                        name=zone_def.name,
                        shape_type="polygon",  # Assuming polygon for now
                        coordinates=coordinates
                    )
                    await self.define_zone(level_id, zone)
                    logger.info(f"Added zone '{zone_def.name}' to level {level_id}")
                except Exception as e:
                    logger.error(f"Failed to add zone '{zone_def.name}': {e}")
        
        # Handle assets (more complex, would need mesh upload)
        # This would require more logic to handle different asset types
        
        # Return the created level
        return level

    # ------------------------------------------------------------------
    # NEW - Entity Methods
    # ------------------------------------------------------------------
    async def add_entity(
        self,
        level_id: int,
        entity_id: str,
        entity_type: str,
        name: Optional[str] = None,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        properties: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[str]] = None,
        status: Optional[str] = None,
        parent_id: Optional[str] = None,
        reference: Optional[str] = None,
        archetype: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Adds a new entity to a level.
        
        Args:
            level_id: ID of the level to add the entity to
            entity_id: Unique ID for the entity
            entity_type: Type of entity (e.g., 'robot', 'fixed_asset')
            name: Display name for the entity
            position: [x, y, z] position coordinates
            rotation: [x, y, z] rotation in degrees
            scale: [x, y, z] scale factors
            properties: Dictionary of custom properties
            capabilities: List of capability strings (for robots)
            status: Current status (e.g., 'idle', 'active')
            parent_id: ID of parent entity, if applicable
            reference: Reference to a defined asset
            archetype: Entity archetype
            
        Returns:
            The created entity object as returned by the backend
        """
        headers = self._get_headers()
        if not self._access_token:
             raise AuthenticationError("Cannot add entity without an active session token.")
        
        payload = {
            "id": entity_id,
            "entity_type": entity_type
        }
        
        # Add optional properties if provided
        if name is not None:
            payload["name"] = name
        if position is not None:
            payload["position"] = position
        if rotation is not None:
            payload["rotation"] = rotation
        if scale is not None:
            payload["scale"] = scale
        if properties is not None:
            payload["properties"] = properties
        if capabilities is not None:
            payload["capabilities"] = capabilities
        if status is not None:
            payload["status"] = status
        if parent_id is not None:
            payload["parent_id"] = parent_id
        if reference is not None:
            payload["reference"] = reference
        if archetype is not None:
            payload["archetype"] = archetype
            
        endpoint = f"/levels/{level_id}/entities"
        logger.debug(f"Sending POST {endpoint}. Payload: {payload}")
        try:
            response = await self._client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"Entity '{entity_id}' added successfully to level {level_id}.")
            return response_data
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during add_entity: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during add_entity: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during add_entity: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e
             
    async def get_entities(self, level_id: int) -> List[Dict[str, Any]]:
        """
        Retrieves all entities in a level.
        
        Args:
            level_id: ID of the level to get entities from
            
        Returns:
            List of entity objects
        """
        headers = self._get_headers()
        if not self._access_token:
             raise AuthenticationError("Cannot get entities without an active session token.")
        
        endpoint = f"/levels/{level_id}/entities"
        logger.debug(f"Sending GET {endpoint}.")
        try:
            response = await self._client.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during get_entities: {e.response.status_code} - {e.response.text}")
            detail = _extract_error_detail(e.response)
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during get_entities: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during get_entities: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e
    
    async def delete_entity(self, level_id: int, entity_id: str) -> Dict[str, Any]:
        """
        Delete an entity from a level by its entity_id.
        
        Args:
            level_id: ID of the level containing the entity
            entity_id: Unique entity identifier within the level
            
        Returns:
            Dict containing the deletion result
            
        Raises:
            APIError: If the API returns an error
        """
        headers = self._get_headers()
        
        try:
            response = await self._client.delete(f"/levels/{level_id}/entities/{entity_id}", headers=headers)
            response.raise_for_status()
            
            if response.status_code == 204:  # No content
                return {"success": True, "entity_id": entity_id}
            else:
                return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            raise APIError(e.response.status_code, detail)
        except Exception as e:
            raise CyberWaveError(f"Failed to delete entity: {e}")

    # --- Catalog Asset Definition Methods --- #
            
    async def create_asset_definition(
        self,
        workspace_id: int,
        name: str,
        slug: str,
        definition_type: str, # 'robot', 'fixed_asset', 'sensor', etc.
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Creates a new asset definition within a workspace.

        Args:
            workspace_id: The ID of the workspace.
            name: Name of the asset definition.
            slug: Unique slug for the asset definition.
            definition_type: Type of asset (e.g., 'robot').
            description: Optional description.
            tags: Optional list of tags.
            metadata: Optional dictionary for extra data.

        Returns:
            A dictionary representing the created asset definition.
        
        Raises:
            APIError: If the backend returns an error.
            CyberWaveError: For other client-side errors.
        """
        headers = self._get_headers()
        # Removed the redundant check/warning for self._access_token here
            
        # Add trailing slash to match FastAPI route definition
        endpoint = f"/workspaces/{workspace_id}/asset-definitions/"
        payload = {
            "name": name,
            "slug": slug,
            "definition_type": definition_type,
            "description": description,
            "tags": tags,
            "metadata": metadata,
        }
        # Filter out None values from payload
        clean_payload = {k: v for k, v in payload.items() if v is not None}
        
        logger.debug(f"Sending POST {endpoint} request. Headers: {headers.keys()}, Payload: {clean_payload}")
        
        try:
            response = await self._client.post(endpoint, json=clean_payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"POST {endpoint} successful. Response: {response_data}")
            return response_data
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            logger.error(f"API Error creating asset definition: {e.response.status_code} - {detail}")
            raise APIError(e.response.status_code, detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during create_asset_definition: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e

    async def get_asset_definitions(
        self,
        workspace_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get a list of asset definitions in a workspace.
        
        Args:
            workspace_id: ID of the workspace
            skip: Number of items to skip for pagination
            limit: Maximum number of items to return
            
        Returns:
            List of asset definitions
            
        Raises:
            APIError: If the API returns an error
        """
        headers = self._get_headers()
        
        try:
            response = await self._client.get(
                f"/workspaces/{workspace_id}/asset_definitions/",
                params={"skip": skip, "limit": limit},
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            raise APIError(e.response.status_code, detail)
        except Exception as e:
            raise CyberWaveError(f"Failed to get asset definitions: {e}")
            
    async def get_asset_definition(
        self,
        workspace_id: int,
        definition_id_or_slug: Union[int, str]
    ) -> Dict[str, Any]:
        """
        Get a specific asset definition by ID or slug.
        
        Args:
            workspace_id: ID of the workspace
            definition_id_or_slug: ID or slug of the asset definition
            
        Returns:
            Dictionary representing the asset definition
            
        Raises:
            APIError: If the API returns an error
        """
        headers = self._get_headers()
        
        try:
            response = await self._client.get(
                f"/workspaces/{workspace_id}/asset_definitions/{definition_id_or_slug}",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            raise APIError(e.response.status_code, detail)
        except Exception as e:
            raise CyberWaveError(f"Failed to get asset definition: {e}")
            
    async def update_asset_definition(
        self,
        workspace_id: int,
        definition_id_or_slug: Union[int, str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update an existing asset definition.
        
        Args:
            workspace_id: ID of the workspace
            definition_id_or_slug: ID or slug of the asset definition
            name: Optional new name
            description: Optional new description
            tags: Optional new list of tags
            metadata: Optional new metadata dictionary
            
        Returns:
            Dictionary representing the updated asset definition
            
        Raises:
            APIError: If the API returns an error
        """
        headers = self._get_headers()
        
        payload = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
            
        if not payload:
            return await self.get_asset_definition(workspace_id, definition_id_or_slug)
            
        try:
            response = await self._client.patch(
                f"/workspaces/{workspace_id}/asset_definitions/{definition_id_or_slug}",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            raise APIError(e.response.status_code, detail)
        except Exception as e:
            raise CyberWaveError(f"Failed to update asset definition: {e}")
            
    async def delete_asset_definition(
        self,
        workspace_id: int,
        definition_id_or_slug: Union[int, str]
    ) -> Dict[str, Any]:
        """
        Delete an asset definition.
        
        Args:
            workspace_id: ID of the workspace
            definition_id_or_slug: ID or slug of the asset definition
            
        Returns:
            Dictionary indicating success
            
        Raises:
            APIError: If the API returns an error
        """
        headers = self._get_headers()
        
        try:
            response = await self._client.delete(
                f"/workspaces/{workspace_id}/asset_definitions/{definition_id_or_slug}",
                headers=headers
            )
            response.raise_for_status()
            
            if response.status_code == 204:  # No content
                return {"success": True, "id_or_slug": definition_id_or_slug}
            else:
                return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            raise APIError(e.response.status_code, detail)
        except Exception as e:
            raise CyberWaveError(f"Failed to delete asset definition: {e}")
            
    async def add_geometry_to_asset_definition(
        self,
        workspace_id: int,
        definition_id_or_slug: Union[int, str],
        mesh_id: Optional[int] = None,
        skeleton_id: Optional[int] = None,
        purpose: str = "visual",
        is_primary: bool = False
    ) -> Dict[str, Any]:
        """
        Add geometry (mesh or skeleton) to an asset definition.
        
        Args:
            workspace_id: ID of the workspace
            definition_id_or_slug: ID or slug of the asset definition
            mesh_id: Optional ID of a mesh to link
            skeleton_id: Optional ID of a skeleton to link
            purpose: Purpose of the geometry (e.g., 'visual', 'collision')
            is_primary: Whether this is the primary geometry for the definition
            
        Returns:
            Dictionary representing the created geometry link
            
        Raises:
            APIError: If the API returns an error
            ValueError: If neither mesh_id nor skeleton_id is provided
        """
        if mesh_id is None and skeleton_id is None:
            raise ValueError("Either mesh_id or skeleton_id must be provided")
            
        headers = self._get_headers()
        
        payload = {
            "purpose": purpose,
            "is_primary": is_primary
        }
        
        if mesh_id is not None:
            payload["mesh_id"] = mesh_id
        if skeleton_id is not None:
            payload["skeleton_id"] = skeleton_id
            
        try:
            response = await self._client.post(
                f"/workspaces/{workspace_id}/asset_definitions/{definition_id_or_slug}/geometry",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            raise APIError(e.response.status_code, detail)
        except Exception as e:
            raise CyberWaveError(f"Failed to add geometry to asset definition: {e}")
            
    async def remove_geometry_from_asset_definition(
        self,
        workspace_id: int,
        definition_id_or_slug: Union[int, str],
        geometry_id: int
    ) -> Dict[str, Any]:
        """
        Remove geometry from an asset definition.
        
        Args:
            workspace_id: ID of the workspace
            definition_id_or_slug: ID or slug of the asset definition
            geometry_id: ID of the geometry link to remove
            
        Returns:
            Dictionary indicating success
            
        Raises:
            APIError: If the API returns an error
        """
        headers = self._get_headers()
        
        try:
            response = await self._client.delete(
                f"/workspaces/{workspace_id}/asset_definitions/{definition_id_or_slug}/geometry/{geometry_id}",
                headers=headers
            )
            response.raise_for_status()
            
            if response.status_code == 204:  # No content
                return {"success": True, "geometry_id": geometry_id}
            else:
                return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            raise APIError(e.response.status_code, detail)
        except Exception as e:
            raise CyberWaveError(f"Failed to remove geometry from asset definition: {e}")

    async def _attempt_refresh(self) -> bool:
        """Attempts to refresh the access token using the refresh token."""
        if not self._refresh_token:
            logger.info("No refresh token available to attempt refresh.")
            return False

        logger.info("Access token potentially expired. Attempting refresh...")
        try:
            refresh_response = await self._client.post(
                "/auth/token/refresh",
                data={"refresh_token": self._refresh_token},
                headers={"Content-Type": "application/x-www-form-urlencoded"} 
            )

            if refresh_response.status_code == 401:
                logger.warning("Refresh token failed (401 Unauthorized). Clearing tokens.")
                self._clear_token_cache() 
                return False
                
            refresh_response.raise_for_status() 
            
            response_data = refresh_response.json()
            new_access_token = response_data.get("access_token")
            # Optional: Handle rotated refresh tokens 
            # new_refresh_token = response_data.get("refresh_token") 

            if not new_access_token or response_data.get("token_type", "").lower() != "bearer":
                logger.error("Token refresh failed: Invalid response from refresh endpoint.")
                return False 

            logger.info(f"Token refresh successful. New access token stored: {new_access_token[:4]}...{new_access_token[-4:]}")
            self._access_token = new_access_token
            # if new_refresh_token:
            #     self._refresh_token = new_refresh_token
            #     logger.info("Rotated refresh token stored.")
                
            self._save_token_to_cache() 
            return True

        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during token refresh: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 401: 
                 self._clear_token_cache()
            return False 
        except httpx.RequestError as e:
            logger.error(f"Request failed during token refresh: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during token refresh: {e}", exc_info=True)
            return False 

    async def _request(
        self, 
        method: str, 
        url: str, 
        retry_on_401: bool = True,
        require_auth: bool = True, # Add flag to indicate if auth is required
        **kwargs
    ) -> httpx.Response:
        """Internal helper to make requests, handling token refresh."""
        # Generate a short request ID for tracing
        request_id = str(uuid.uuid4())[:8]
        
        # Check for token BEFORE attempting request if auth is required
        if require_auth and not self._access_token:
            logger.error(f"[{request_id}] Authentication required for {method} {url}, but no token found.")
            raise AuthenticationError("Cannot make authenticated request: No session token available.")

        headers = self._get_headers(require_auth=require_auth) # Pass flag to header helper
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        kwargs['headers'] = headers

        try:
            # Log request details if in debug mode
            if hasattr(self, '_debug') and self._debug:
                logger.debug(f"[{request_id}] Sending {method} request to {url}")
            
            # Make the request
            response = await self._client.request(method, url, **kwargs)
            
            # Log response details if in debug mode
            if hasattr(self, '_debug') and self._debug:
                logger.debug(f"[{request_id}] Response received: {response.status_code}")
            
            response.raise_for_status()
            # Return response object directly
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401 and retry_on_401:
                logger.warning(f"Received 401 Unauthorized for {method} {url}. Attempting token refresh.")
                refresh_successful = await self._attempt_refresh()
                
                if refresh_successful:
                    logger.info(f"Retrying original request: {method} {url}")
                    # Get headers with the new token
                    new_headers = self._get_headers()
                    # Update the original kwargs headers with the new ones,
                    # ensuring the refreshed token takes precedence.
                    original_headers = kwargs.get('headers', {})
                    original_headers.update(new_headers) # Update old headers with new (overwrites Authorization)
                    kwargs['headers'] = original_headers
                    
                    try:
                        # Revert passing copy
                        retry_response = await self._client.request(method, url, **kwargs)
                        retry_response.raise_for_status()
                        # Return response object directly
                        return retry_response
                    except httpx.HTTPStatusError as retry_e:
                        logger.error(f"Request failed even after token refresh: {retry_e.response.status_code}")
                        raise APIError(retry_e.response.status_code, _extract_error_detail(retry_e.response)) from retry_e
                    except Exception as retry_e:
                        logger.error(f"Unexpected error during request retry: {retry_e}", exc_info=True)
                        raise CyberWaveError(f"Unexpected error during request retry: {retry_e}") from retry_e
                else:
                    logger.error("Token refresh failed. Raising AuthenticationError.")
                    raise AuthenticationError("Authentication required. Refresh token invalid or expired.") from e
            else:
                raise APIError(e.response.status_code, _extract_error_detail(e.response)) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}")
            raise CyberWaveError(f"Network request failed: {e}") from e
        except APIError:
            # Propagate already-constructed APIError without modification
            raise
        except Exception as e:
            logger.error(f"Unexpected error during request: {e}", exc_info=True)
            raise CyberWaveError(f"An unexpected error occurred during the request: {e}") from e

    async def login(self, username: Optional[str] = None, password: Optional[str] = None) -> None:
        """
        Authenticates with the backend and stores the session token.  If
        ``username`` or ``password`` are not provided, the method will look for
        them in the environment variables defined by ``USERNAME_ENV_VAR`` and
        ``PASSWORD_ENV_VAR``.  When running in an interactive terminal and still
        missing, the user is prompted to enter the values securely.

        Args:
            username: The username (email). Optional.
            password: The password. Optional.

        Raises:
            AuthenticationError: If login fails.
            APIError: For other API related errors.
        """
        if username is None:
            username = os.getenv(USERNAME_ENV_VAR)
        if password is None:
            password = os.getenv(PASSWORD_ENV_VAR)

        if username is None and sys.stdin.isatty():
            username = input("CyberWave username: ")
        if password is None and sys.stdin.isatty():
            password = getpass.getpass("CyberWave password: ")

        if not username or not password:
            raise AuthenticationError("Username and password are required for login")

        logger.info(f"Attempting to login user: {username}")
        try:
            response = await self._client.post(
                "/auth/token", 
                data={"username": username, "password": password},
                headers={"Content-Type": "application/x-www-form-urlencoded"} # Correct header for form data
            )
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
            
            token_data = response.json()
            self._access_token = token_data.get("access_token")
            self._refresh_token = token_data.get("refresh_token")
            
            # Store session info like user_id and expiry if available
            # Assuming the token endpoint might return more than just tokens
            self._session_info = token_data.copy() # Store all token response data
            if "access_token" in self._session_info: del self._session_info["access_token"]
            if "refresh_token" in self._session_info: del self._session_info["refresh_token"]


            if not self._access_token:
                logger.error("Login successful but no access_token received.")
                raise AuthenticationError("Login successful but no access_token received.")

            logger.info(f"Login successful for {username}. Access token received.")
            if self._refresh_token:
                logger.info("Refresh token also received.")

            if self._use_token_cache:
                self._save_token_to_cache()

        except httpx.HTTPStatusError as e:
            error_detail = _extract_error_detail(e.response)
            logger.error(f"Login failed for {username}: {e.response.status_code} - {error_detail}")
            if e.response.status_code in [401, 403]:
                raise AuthenticationError(f"Login failed: Invalid credentials or insufficient permissions. Server said: {error_detail}")
            else:
                raise APIError(e.response.status_code, error_detail) from e
        except httpx.RequestError as e:
            logger.error(f"Network error during login for {username}: {e}")
            raise CyberWaveError(f"Network error during login: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during login: {e}", exc_info=True)
            raise CyberWaveError(f"An unexpected error occurred during login: {e}")


    async def logout(self) -> None:
        """Logs out the current user and clears the session token."""
        logger.info("Logging out and clearing session tokens.")
        self._access_token = None
        self._refresh_token = None
        self._session_info = {} # Clear session info as well
        if self._use_token_cache:
            self._clear_token_cache() # This will attempt to clear keyring and file cache
        logger.info("Logout complete. Tokens have been cleared.")

    async def get_current_user_info(self) -> Dict[str, Any]:
        """Retrieves information about the currently authenticated user.
        (Handles token refresh automatically)
        
        Returns:
            A dictionary containing user info (e.g., sub, email, name).
            See backend UserInfo schema.
            
        Raises:
            AuthenticationError: If no valid token is available or refresh fails.
            APIError: If the backend returns an error.
            CyberWaveError: For other client-side errors.
        """
        if not self._access_token:
            raise AuthenticationError("Cannot get user info without an active session token.")

        logger.debug("Sending GET /users/me request.")
        try:
            response = await self._request(
                method="GET", 
                url="/users/me"
            )
            return response.json()
        except (APIError, AuthenticationError, CyberWaveError) as e:
            logger.error(f"Failed to get current user info: {e}")
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from get_current_user_info: {response.text}", exc_info=True)
            raise CyberWaveError(f"Invalid JSON response received: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in get_current_user_info: {e}", exc_info=True)
            raise CyberWaveError(f"An unexpected error occurred while getting user info: {e}") from e

    async def register(
        self, 
        email: str, 
        password: str, 
        full_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Registers a new user account.

        Note: This initiates the registration process. The user will need to 
        verify their email address via the link sent to them before they can log in.
        
        Args:
            email: The user's email address.
            password: The desired password.
            full_name: Optional full name for the user.
            
        Returns:
            A dictionary representing the created (but inactive) user.
            See backend UserPublic schema.
            
        Raises:
            APIError: If the email is already registered or other backend error occurs.
            CyberWaveError: For connection or other client-side errors.
        """
        payload = {
            "email": email,
            "password": password,
            "full_name": full_name
        }
        # Filter out None values (although backend schema might handle it)
        clean_payload = {k: v for k, v in payload.items() if v is not None}

        logger.info(f"Attempting registration for user: {email}")
        try:
            # Registration doesn't require prior auth, so no token refresh needed here
            # Use _client directly or _request with retry_on_401=False?
            # Using _client directly is simpler as 401 isn't expected here.
            response = await self._client.post(
                "/users/register", 
                json=clean_payload
                # No Authorization header needed/sent
            )
            response.raise_for_status() # Raise for 4xx (like 400 email exists) / 5xx
            response_data = response.json()
            logger.info(f"Registration request successful for {email}. User created (inactive).")
            return response_data
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            logger.error(f"Registration failed for {email}: {e.response.status_code} - {detail}")
            # Re-raise as APIError, specific detail comes from backend
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during registration: {e}")
            raise CyberWaveError(f"Failed to connect to backend for registration: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from registration: {response.text}", exc_info=True)
            raise CyberWaveError(f"Invalid JSON response received from registration: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during registration: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred during registration: {e}") from e

            
    async def get_asset_definition_by_slug(self, workspace_id: int, slug: str) -> Optional[Dict[str, Any]]:
        """Get a specific asset definition by its workspace ID and slug."""
        try:
            # Construct the correct path using the workspace ID and slug
            path = f"/workspaces/{workspace_id}/asset-definitions/slug/{slug}"
            logger.debug(f"Requesting asset definition by slug: GET {path}")
            response = await self._request("GET", path)
            return response.json()
        except APIError as e:
            if e.status_code == 404:
                logger.warning(f"Asset definition with slug '{slug}' not found in workspace {workspace_id}.")
                return None
            logger.error(f"API error fetching asset definition by slug '{slug}': {e}")
            raise e # Re-raise other API errors
        except Exception as e:
            logger.error(f"Unexpected error fetching asset definition by slug '{slug}': {e}", exc_info=True)
            raise e # Re-raise unexpected errors


    # ------------------------------------------------------------------
    # Asset Catalog Methods (missing from earlier SDK versions)
    # ------------------------------------------------------------------
    async def list_asset_catalogs(self) -> List[Dict[str, Any]]:
        """Retrieve all asset catalogs."""
        response = await self._request("GET", "/asset-catalogs")
        return response.json()

    async def get_asset_catalog(self, catalog_uuid: str) -> Optional[Dict[str, Any]]:
        """Get a single asset catalog by UUID. Returns None if not found."""
        try:
            response = await self._request("GET", f"/asset-catalogs/{catalog_uuid}")
            return response.json()
        except APIError as e:
            if e.status_code == 404:
                return None
            raise

    async def create_asset_catalog(
        self,
        name: str,
        description: str,
        public: bool = False,
    ) -> Dict[str, Any]:
        """Create a new asset catalog."""
        payload = {"name": name, "description": description, "public": public}
        response = await self._request("POST", "/asset-catalogs", json=payload)
        return response.json()

    # ------------------------------------------------------------------
    # Asset Methods
    # ------------------------------------------------------------------
    async def list_assets(self) -> List[Dict[str, Any]]:
        """Retrieve all assets."""
        response = await self._request("GET", "/assets")
        return response.json()

    async def create_asset(
        self,
        name: str,
        description: str,
        asset_catalog_id: int,
        level_id: int,
        registry_id: str | None = None,
    ) -> Dict[str, Any]:
        """Create a new asset and assign it to a catalog and level.

        registry_id is an optional catalog match key (e.g., "so/100").
        """
        payload = {
            "name": name,
            "description": description,
            "asset_catalog_id": asset_catalog_id,
            "level_id": level_id,
        }
        if registry_id is not None:
            payload["registry_id"] = registry_id
        response = await self._request("POST", "/assets", json=payload)
        return response.json()

    async def update_asset_catalog(
        self,
        catalog_uuid: str,
        name: str,
        description: str,
        public: bool = False,
    ) -> Dict[str, Any]:
        """Update an existing asset catalog."""
        payload = {"name": name, "description": description, "public": public}
        response = await self._request(
            "PUT", f"/asset-catalogs/{catalog_uuid}", json=payload
        )
        return response.json()

    async def delete_asset_catalog(self, catalog_uuid: str) -> Dict[str, Any]:
        """Delete an asset catalog by UUID."""
        response = await self._request("DELETE", f"/asset-catalogs/{catalog_uuid}")
        return response.json()

    async def get_asset(self, asset_uuid: str) -> Dict[str, Any]:
        """Retrieve a single asset by UUID."""
        response = await self._request("GET", f"/assets/{asset_uuid}")
        return response.json()

    async def update_asset(
        self,
        asset_uuid: str,
        name: str,
        description: str,
        asset_catalog_id: int,
        level_id: int,
        registry_id: str | None = None,
    ) -> Dict[str, Any]:
        """Update an existing asset."""
        payload = {
            "name": name,
            "description": description,
            "asset_catalog_id": asset_catalog_id,
            "level_id": level_id,
        }
        if registry_id is not None:
            payload["registry_id"] = registry_id
        response = await self._request(
            "PUT", f"/assets/{asset_uuid}", json=payload
        )
        return response.json()

    async def delete_asset(self, asset_uuid: str) -> Dict[str, Any]:
        """Delete an asset by UUID."""
        response = await self._request("DELETE", f"/assets/{asset_uuid}")
        return response.json()

    async def register_device(
        self,
        project_id: int,
        name: str,
        device_type: str,
        asset_catalog_uuid: str | None = None,
    ) -> Dict[str, Any]:
        """Register a new device in a project."""
        payload = {
            "name": name,
            "device_type": device_type,
        }
        if asset_catalog_uuid:
            payload["asset_catalog_uuid"] = asset_catalog_uuid
        response = await self._request("POST", f"/projects/{project_id}/devices", json=payload)
        return response.json()

    async def issue_device_token(self, device_id: int) -> str:
        """Request an offline token for a device."""
        response = await self._request("POST", f"/devices/{device_id}/token")
        data = response.json()
        return data.get("token", "")

    async def send_telemetry(self, device_id: int, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Send telemetry data for a device."""
        payload = {"telemetry": telemetry}
        response = await self._request(
            "POST", f"/devices/{device_id}/telemetry", json=payload
        )
        return response.json()

    async def upload_video_frame(self, device_id: int, frame_bytes: bytes) -> Dict[str, Any]:
        """Upload a single video frame for a device."""
        response = await self._request(
            "POST",
            f"/devices/{device_id}/video",
            content=frame_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        return response.json()

    # --- (End of Client class) ---

    # --- Asset helpers ---
    async def upload_asset_glb(self, asset_uuid: str, file_path: str | os.PathLike[str]) -> Dict[str, Any]:
        """Upload a GLB file to an existing asset."""
        headers = self._get_headers()
        try:
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()
                files = {"file": (os.fspath(file_path), content, "application/octet-stream")}
                response = await self._client.post(
                    f"/assets/{asset_uuid}/glb-file",
                    files=files,
                    headers=headers,
                )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            raise APIError(e.response.status_code, detail)
        except Exception as e:
            raise CyberWaveError(f"Failed to upload GLB: {e}")

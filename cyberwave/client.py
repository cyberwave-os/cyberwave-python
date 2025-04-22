import httpx
import os
import json
from typing import List, Optional, Dict, Any
import logging

# Basic logger for the SDK
logger = logging.getLogger("cyberwave.sdk")
logging.basicConfig(level=logging.INFO) # Configure basic logging

# --- Constants --- 
DEFAULT_BACKEND_URL = "http://localhost:8000"
API_VERSION_PREFIX = "/api/v1"
SHARE_TOKEN_HEADER = "X-Share-Token"
# Simple file-based cache for the token (optional)
TOKEN_CACHE_FILE = os.path.expanduser("~/.cyberwave_token_cache.json")

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


class Client:
    """Asynchronous client for interacting with the CyberWave Backend API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        use_token_cache: bool = True,
        timeout: float = 10.0,
    ):
        """
        Initializes the CyberWave client.

        Args:
            base_url: The base URL of the CyberWave backend.
                      Defaults to http://localhost:8000.
            use_token_cache: If True, attempts to load/save the share token
                             from/to ~/.cyberwave_token_cache.json.
            timeout: Request timeout in seconds.
        """
        if base_url is None:
            base_url = os.getenv("CYBERWAVE_BACKEND_URL", DEFAULT_BACKEND_URL)
        
        # Ensure API prefix is present
        api_base_url = base_url
        if API_VERSION_PREFIX not in base_url:
             if base_url.endswith("/"):
                 api_base_url = base_url.rstrip("/") + API_VERSION_PREFIX
             else:
                 api_base_url = base_url + API_VERSION_PREFIX
        
        logger.info(f"Initializing CyberWave Client for backend: {api_base_url}")
        self._client = httpx.AsyncClient(base_url=api_base_url, timeout=timeout)
        self._share_token: Optional[str] = None
        self._use_token_cache = use_token_cache
        self._session_info: Dict[str, Any] = {}

        if self._use_token_cache:
            self._load_token_from_cache()

    def _load_token_from_cache(self):
        try:
            if os.path.exists(TOKEN_CACHE_FILE):
                with open(TOKEN_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    self._share_token = cache_data.get("share_token")
                    self._session_info = cache_data.get("session_info", {})
                    if self._share_token:
                        logger.info(f"Loaded share token from cache: {self._share_token[:4]}...{self._share_token[-4:]}")
                    else:
                        logger.info("Token cache file found but contained no token.")
            else:
                logger.info("Token cache file not found.")
        except Exception as e:
            logger.warning(f"Failed to load token from cache ({TOKEN_CACHE_FILE}): {e}")

    def _save_token_to_cache(self):
        if not self._use_token_cache:
            return
        try:
            cache_data = {
                "share_token": self._share_token,
                "session_info": self._session_info
            }
            with open(TOKEN_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"Saved session details to cache: {TOKEN_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save token to cache: {e}")
            
    def _clear_token_cache(self):
        """Removes the token cache file."""
        if os.path.exists(TOKEN_CACHE_FILE):
            try:
                os.remove(TOKEN_CACHE_FILE)
                logger.info(f"Removed token cache file: {TOKEN_CACHE_FILE}")
            except Exception as e:
                 logger.warning(f"Failed to remove token cache file: {e}")
        self._share_token = None
        self._session_info = {}

    async def aclose(self):
        """Closes the underlying HTTP client session."""
        await self._client.aclose()
        logger.info("CyberWave Client closed.")

    def has_active_session(self) -> bool:
        """Checks if a share token is currently loaded."""
        return self._share_token is not None

    def get_session_token(self) -> Optional[str]:
        """Returns the currently loaded share token, if any."""
        return self._share_token
        
    def get_session_info(self) -> Dict[str, Any]:
        """Returns cached information about the current session (e.g., share_url)."""
        return self._session_info

    async def add_robot(
        self,
        name: str,
        robot_type: str,
        level_id: Optional[int] = None,
        serial_number: Optional[str] = None,
        status: str = "unknown", # Default status, maps to RobotStatusEnum
        capabilities: Optional[List[str]] = None,
        initial_pos_x: Optional[float] = None,
        initial_pos_y: Optional[float] = None,
        initial_pos_z: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        current_battery_percentage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Registers a new robot with the backend.

        If no active session (share token) exists and level_id is None,
        this will trigger the creation of a new temporary session on the backend.
        The share token for the new session will be automatically stored and cached.

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
        headers = {}
        if self._share_token:
            headers[SHARE_TOKEN_HEADER] = self._share_token
            logger.info(f"Using existing share token: {self._share_token[:4]}...{self._share_token[-4:]}")
        else:
             logger.info("No active share token. Requesting new session if level_id is None.")

        # Construct payload, filtering out None values potentially
        payload = {
            "name": name,
            "robot_type": robot_type,
            "level_id": level_id, # Pass None if not provided
            "serial_number": serial_number,
            "status": status,
            "capabilities": capabilities,
            "initial_pos_x": initial_pos_x,
            "initial_pos_y": initial_pos_y,
            "initial_pos_z": initial_pos_z,
            "metadata": metadata,
            "current_battery_percentage": current_battery_percentage,
        }
        # Filter payload strictly? For now, rely on backend schema validation
        # clean_payload = {k: v for k, v in payload.items() if v is not None}
        clean_payload = payload # Send all for now
        
        logger.debug(f"Sending POST /robots request. Headers: {headers.keys()}, Payload: {clean_payload}")

        try:
            response = await self._client.post("/robots", json=clean_payload, headers=headers)
            response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx

            response_data = response.json()
            logger.debug(f"POST /robots successful. Response: {response_data}")

            # Check if a new session was created (token returned in body/header)
            returned_token = response_data.get("share_token") or response.headers.get(SHARE_TOKEN_HEADER)
            
            if returned_token and not self._share_token:
                logger.info(f"New temporary session created. Received token: {returned_token[:4]}...{returned_token[-4:]}")
                self._share_token = returned_token
                # Store basic info from response
                self._session_info["share_url"] = response_data.get("share_url")
                self._session_info["level_id"] = response_data.get("level_id") # Assuming robot response includes level_id
                self._save_token_to_cache() 
                # TODO: Provide more user feedback about the new session?
            elif returned_token and self._share_token and returned_token != self._share_token:
                 logger.warning("Backend returned a different share token than expected. Ignoring.")
                 # Don't update the token if we already had one

            return response_data # Return the full response dict

        except httpx.HTTPStatusError as e:
            logger.error(f"API Error during add_robot: {e.response.status_code} - {e.response.text}")
            detail = e.response.text
            try:
                detail = e.response.json().get("detail", detail)
            except Exception:
                pass # Keep raw text if JSON parsing fails
            raise APIError(status_code=e.response.status_code, detail=detail) from e
        except httpx.RequestError as e:
            logger.error(f"Request failed during add_robot: {e}")
            raise CyberWaveError(f"Failed to connect to backend: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during add_robot: {e}", exc_info=True)
             raise CyberWaveError(f"An unexpected error occurred: {e}") from e
             
    # --- Add other methods (get_robots, get_workspaces, claim_session, etc.) later --- 
    # async def get_robots(self, ...) -> List[Dict[str, Any]]:
    #     headers = {}
    #     if self._share_token:
    #         headers[SHARE_TOKEN_HEADER] = self._share_token
    #     response = await self._client.get("/robots", headers=headers)
    #     # ... error handling ...
    #     return response.json() 
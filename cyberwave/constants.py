"""Constants and environment variable names used by the CyberWave SDK."""

# Default URL of the backend API if none is provided and no environment
# variable overrides it.
DEFAULT_BACKEND_URL = "http://localhost:8000"

# Environment variables that can override default settings
BACKEND_URL_ENV_VAR = "CYBERWAVE_BACKEND_URL"
USERNAME_ENV_VAR = "CYBERWAVE_USERNAME"
PASSWORD_ENV_VAR = "CYBERWAVE_PASSWORD"

__all__ = [
    "DEFAULT_BACKEND_URL",
    "BACKEND_URL_ENV_VAR",
    "USERNAME_ENV_VAR",
    "PASSWORD_ENV_VAR",
]

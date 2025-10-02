"""Constants and environment variable names used by the CyberWave SDK."""

from enum import Enum

class CyberWaveEnvironment(Enum):
    """Predefined CyberWave backend environments"""
    LOCAL = "local"
    DEV = "dev"
    QA = "qa"
    STAGING = "staging"
    PROD = "prod"

# Environment URL mappings
ENVIRONMENT_URLS = {
    CyberWaveEnvironment.LOCAL: "http://localhost:8000/api/v1",
    CyberWaveEnvironment.DEV: "https://api-dev.cyberwave.com/api/v1",
    CyberWaveEnvironment.QA: "https://api-qa.cyberwave.com/api/v1",
    CyberWaveEnvironment.STAGING: "https://api-staging.cyberwave.com/api/v1",
    CyberWaveEnvironment.PROD: "https://api.cyberwave.com/api/v1",
}

# Default environment is PROD for production-ready experience
DEFAULT_ENVIRONMENT = CyberWaveEnvironment.PROD
DEFAULT_BACKEND_URL = ENVIRONMENT_URLS[DEFAULT_ENVIRONMENT]

# Legacy constant for backward compatibility
LEGACY_DEFAULT_URL = "http://localhost:8000"

# Environment variables that can override default settings
BACKEND_URL_ENV_VAR = "CYBERWAVE_BACKEND_URL"
ENVIRONMENT_ENV_VAR = "CYBERWAVE_ENVIRONMENT"
USERNAME_ENV_VAR = "CYBERWAVE_USERNAME"
PASSWORD_ENV_VAR = "CYBERWAVE_PASSWORD"

def get_backend_url(environment: CyberWaveEnvironment = None, custom_url: str = None) -> str:
    """Get backend URL based on environment or custom URL"""
    if custom_url:
        return custom_url
    
    if environment:
        return ENVIRONMENT_URLS.get(environment, ENVIRONMENT_URLS[DEFAULT_ENVIRONMENT])
    
    # Check environment variable
    import os
    env_name = os.getenv(ENVIRONMENT_ENV_VAR)
    if env_name:
        try:
            env_enum = CyberWaveEnvironment(env_name.lower())
            return ENVIRONMENT_URLS[env_enum]
        except ValueError:
            pass
    
    # Check direct URL override
    url_override = os.getenv(BACKEND_URL_ENV_VAR)
    if url_override:
        return url_override
    
    return DEFAULT_BACKEND_URL

__all__ = [
    "CyberWaveEnvironment",
    "ENVIRONMENT_URLS",
    "DEFAULT_ENVIRONMENT",
    "DEFAULT_BACKEND_URL",
    "LEGACY_DEFAULT_URL",
    "BACKEND_URL_ENV_VAR",
    "ENVIRONMENT_ENV_VAR",
    "USERNAME_ENV_VAR",
    "PASSWORD_ENV_VAR",
    "get_backend_url",
]

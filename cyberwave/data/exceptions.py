"""Data-layer exceptions for the Cyberwave SDK."""

from cyberwave.exceptions import CyberwaveError


class DataBackendError(CyberwaveError):
    """Base for all data-backend errors."""


class BackendUnavailableError(DataBackendError):
    """Raised when the requested backend cannot be initialized.

    Common cause: the optional ``eclipse-zenoh`` package is not installed.
    """


class BackendConfigError(DataBackendError):
    """Raised for invalid or missing backend configuration."""


class ChannelError(DataBackendError):
    """Raised for invalid channel names or key expressions."""


class PublishError(DataBackendError):
    """Raised when a publish operation fails."""


class SubscriptionError(DataBackendError):
    """Raised when a subscribe operation fails."""


class WireFormatError(DataBackendError):
    """Raised when wire-format encoding or decoding fails."""


class RecordingError(DataBackendError):
    """Raised when a recording or replay operation fails."""

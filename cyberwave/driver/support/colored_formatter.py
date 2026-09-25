"""Colored logging formatter for :mod:`cyberwave.driver` processes.

ANSI colors for log levels and bracket prefixes used by
:class:`~cyberwave.driver.BaseDriver` lifecycle logs. Disable via
``CYBERWAVE_COLOR_LOGS``.
"""

import logging
import os
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    """Logging formatter with ANSI colors for levels and bracket prefixes.

    Automatically colorizes:
    - Log level names (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Bracket prefixes in messages ([ERROR], [FAILED], [WARNING], etc.)

    Colors can be disabled by setting CYBERWAVE_COLOR_LOGS=false in environment.
    """

    # ANSI color codes
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m\033[1m",  # Bold Red
    }

    # Bracket prefix colors
    BRACKET_COLORS: ClassVar[dict[str, str]] = {
        "[ERROR]": "\033[91m",  # Red
        "[FAILED]": "\033[91m",  # Red
        "[WARNING]": "\033[93m",  # Yellow
        "[STATE]": "\033[95m",  # Magenta
        "[STARTING]": "\033[96m",  # Cyan
        "[INITIALIZING]": "\033[96m",  # Cyan
        "[SUCCESS]": "\033[92m",  # Green
        "[CONNECTED]": "\033[92m",  # Green
        "[ENABLED]": "\033[92m",  # Green
        "[STARTED]": "\033[92m",  # Green
        "[RESOLVED]": "\033[92m",  # Green
    }

    RESET: ClassVar[str] = "\033[0m"

    def __init__(self, *args, enable_colors: bool = True, **kwargs):
        """Initialize colored formatter.

        Args:
            enable_colors: Whether to enable ANSI color codes (default: True)
        """
        super().__init__(*args, **kwargs)
        self.enable_colors = enable_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: Log record to format

        Returns:
            Formatted log string with ANSI color codes (if enabled)
        """
        if not self.enable_colors:
            return super().format(record)

        # Colorize log level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )

        # Colorize bracket prefixes in message
        message = record.getMessage()
        for bracket, color_code in self.BRACKET_COLORS.items():
            if bracket in message:
                colored_bracket = f"{color_code}{bracket}{self.RESET}"
                message = message.replace(bracket, colored_bracket)

        # Update record message with colored version
        record.msg = message
        record.args = ()  # Clear args since we've already formatted the message

        return super().format(record)


def setup_colored_logging(
    logger: logging.Logger,
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    enable_colors: bool | None = None,
) -> None:
    """Configure a logger with colored output.

    Args:
        logger: Logger instance to configure
        level: Logging level (default: INFO)
        format_string: Log format string
        date_format: Timestamp format string
        enable_colors: Whether to enable colors (default: read from CYBERWAVE_COLOR_LOGS env var)
    """
    # Check environment variable if not explicitly set
    if enable_colors is None:
        enable_colors = os.getenv("CYBERWAVE_COLOR_LOGS", "true").lower() == "true"

    # Create handler with colored formatter
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColoredFormatter(
            format_string, datefmt=date_format, enable_colors=enable_colors
        )
    )

    # Configure logger
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    logger.addHandler(handler)


def get_colored_formatter(enable_colors: bool | None = None) -> ColoredFormatter:
    """Get a ColoredFormatter instance.

    Args:
        enable_colors: Whether to enable colors (default: read from CYBERWAVE_COLOR_LOGS env var)

    Returns:
        ColoredFormatter instance
    """
    if enable_colors is None:
        enable_colors = os.getenv("CYBERWAVE_COLOR_LOGS", "true").lower() == "true"

    return ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        enable_colors=enable_colors,
    )

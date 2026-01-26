"""
Device fingerprinting for Cyberwave Edge.

Generates a stable fingerprint based on hardware characteristics that can be used
to identify this edge device across sessions. Fingerprints are stored in twin
metadata to track which edge devices have connected.
"""

import hashlib
import os
import platform
import socket
from uuid import getnode


def get_device_info() -> dict:
    """
    Collect device information for fingerprinting.
    
    Returns:
        Dictionary with device details:
        - hostname: Machine hostname
        - platform: OS and architecture (e.g., "Darwin-arm64")
        - python_version: Python version string
        - mac_address: Primary MAC address
    """
    # Get MAC address from uuid.getnode()
    mac_int = getnode()
    mac = ':'.join(f'{mac_int:012x}'[i:i+2] for i in range(0, 12, 2))
    
    return {
        "hostname": socket.gethostname(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "python_version": platform.python_version(),
        "mac_address": mac,
    }


def generate_fingerprint(override: str | None = None) -> str:
    """
    Generate a stable fingerprint for this device.
    
    The fingerprint is a readable string combining a hostname prefix with a
    hash suffix derived from hardware characteristics. This ensures:
    - Readability: You can identify the device from the fingerprint
    - Stability: Same device always produces same fingerprint
    - Uniqueness: Different devices produce different fingerprints
    
    Args:
        override: If provided, use this value instead of auto-generating.
                  Also checks CYBERWAVE_EDGE_UUID environment variable.
    
    Returns:
        Fingerprint string like "macbook-pro-a1b2c3d4e5f6"
    
    Examples:
        >>> generate_fingerprint()
        'macbook-pro-a1b2c3d4e5f6'
        
        >>> generate_fingerprint(override="my-custom-id")
        'my-custom-id'
    """
    # Check for explicit override first
    if override:
        return override
    
    # Check environment variable
    if env_override := os.getenv("CYBERWAVE_EDGE_UUID"):
        return env_override
    
    # Auto-generate from hardware characteristics
    info = get_device_info()
    raw = f"{info['hostname']}-{info['mac_address']}-{info['platform']}"
    hash_suffix = hashlib.sha256(raw.encode()).hexdigest()[:12]
    
    # Clean hostname for use in fingerprint
    # - Lowercase
    # - Replace spaces with dashes
    # - Truncate to 15 chars
    hostname_prefix = info['hostname'][:15].lower().replace(' ', '-').replace('.', '-')
    
    # Remove trailing dashes
    hostname_prefix = hostname_prefix.rstrip('-')
    
    return f"{hostname_prefix}-{hash_suffix}"


def format_device_info_table(info: dict | None = None) -> str:
    """
    Format device info as a table for display.
    
    Args:
        info: Device info dict. If None, collects current device info.
    
    Returns:
        Formatted string for terminal display.
    """
    if info is None:
        info = get_device_info()
    
    fingerprint = generate_fingerprint()
    
    # Mask part of MAC address for privacy in display
    mac = info.get("mac_address", "unknown")
    if len(mac) > 8:
        mac_masked = mac[:8] + ":xx:xx:xx"
    else:
        mac_masked = mac
    
    lines = [
        f"Fingerprint: {fingerprint}",
        f"Hostname:    {info.get('hostname', 'unknown')}",
        f"Platform:    {info.get('platform', 'unknown')}",
        f"Python:      {info.get('python_version', 'unknown')}",
        f"MAC:         {mac_masked}",
    ]
    
    return "\n".join(lines)

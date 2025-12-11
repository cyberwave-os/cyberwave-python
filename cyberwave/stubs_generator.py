"""
Stub generator for Cyberwave SDK.

Downloads asset definitions from the API and generates .pyi stub files
that provide IDE autocomplete based on asset capabilities.

Usage:
    python -m cyberwave.stubs_generator

Or with a custom API URL:
    python -m cyberwave.stubs_generator --api-url http://localhost:8000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# API URLs
DEFAULT_API_URL = "https://api.cyberwave.com"
LOCAL_API_URL = "http://localhost:8000"

# Capability to method mapping (for additional methods not in base classes)
CAPABILITY_METHODS = {
    "can_locomote": [
        (
            "move",
            "(self, x: float | None = None, y: float | None = None, z: float | None = None) -> None",
            "Move the twin to a new position",
        ),
        (
            "move_to",
            "(self, position: list[float]) -> None",
            "Move to a specific position [x, y, z]",
        ),
    ],
    "can_actuate": [
        ("joints", "JointController", "Controller for robot joints"),
    ],
    "has_joints": [
        ("joints", "JointController", "Controller for robot joints"),
    ],
}

# Sensor type to method mapping (these are inherited from base classes now)
SENSOR_METHODS = {
    "rgb": [],  # Methods are in CameraTwin base class
    "depth": [],  # Methods are in DepthCameraTwin base class
}

# Common methods available on all twins
COMMON_METHODS = [
    ("uuid", "str", "Get twin UUID"),
    ("name", "str", "Get twin name"),
    ("asset_id", "str", "Get asset ID"),
    ("environment_id", "str", "Get environment ID"),
    ("refresh", "(self) -> None", "Refresh twin data from the server"),
    ("delete", "(self) -> None", "Delete this twin"),
    (
        "rotate",
        "(self, yaw: float | None = None, pitch: float | None = None, roll: float | None = None, quaternion: list[float] | None = None) -> None",
        "Rotate the twin",
    ),
    (
        "scale",
        "(self, x: float | None = None, y: float | None = None, z: float | None = None) -> None",
        "Scale the twin",
    ),
    (
        "subscribe",
        "(self, on_update: Callable[[dict[str, Any]], None]) -> None",
        "Subscribe to real-time updates",
    ),
]


def get_base_class(capabilities: dict[str, Any] | None) -> str:
    """Determine the appropriate base class based on capabilities."""
    if not capabilities:
        return "Twin"

    has_sensors = bool(capabilities.get("sensors", []))
    has_depth = any(s.get("type") == "depth" for s in capabilities.get("sensors", []))
    can_fly = capabilities.get("can_fly", False)
    can_locomote = capabilities.get("can_locomote", False)
    can_grip = capabilities.get("can_grip", False)

    # Select class based on combination of capabilities
    # Select class based on combination of capabilities
    if can_fly:
        if can_grip and has_depth:
            return "FlyingGripperDepthCameraTwin"
        elif can_grip and has_sensors:
            return "FlyingGripperCameraTwin"
        elif has_sensors:
            return "FlyingCameraTwin"
        elif has_depth:
            return "FlyingDepthCameraTwin"
        elif can_grip:
            return "FlyingGripperCameraTwin"
        else:
            return "FlyingTwin"
    elif can_locomote:
        if can_grip and has_depth:
            return "LocomoteGripperDepthCameraTwin"
        elif can_grip and has_sensors:
            return "LocomoteGripperCameraTwin"
        elif can_grip:
            return "LocomoteGripperTwin"
        elif has_depth:
            return "LocomoteDepthCameraTwin"
        elif has_sensors:
            return "LocomoteCameraTwin"
        else:
            return "LocomoteTwin"
    elif can_grip and has_sensors:
        return "GripperCameraTwin"
    elif can_grip and has_depth:
        return "GripperDepthCameraTwin"
    elif can_fly:
        return "FlyingTwin"
    elif can_locomote:
        return "LocomoteTwin"
    elif can_grip:
        return "GripperTwin"
    elif has_depth:
        return "DepthCameraTwin"
    elif has_sensors:
        return "CameraTwin"
    else:
        return "Twin"


def fetch_assets(api_url: str) -> list[dict[str, Any]]:
    """Fetch assets from the Cyberwave API."""
    url = f"{api_url}/api/v1/assets"

    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data if isinstance(data, list) else []
    except HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        return []
    except URLError as e:
        print(f"URL Error: {e.reason}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error fetching assets: {e}", file=sys.stderr)
        return []


def get_methods_for_capabilities(
    capabilities: dict[str, Any] | None,
) -> list[tuple[str, str, str]]:
    """Get list of methods based on capabilities."""
    if not capabilities:
        return []

    methods: list[tuple[str, str, str]] = []
    seen_methods: set[str] = set()

    # Check boolean capabilities
    for cap_name, cap_methods in CAPABILITY_METHODS.items():
        if capabilities.get(cap_name, False):
            for method in cap_methods:
                if method[0] not in seen_methods:
                    methods.append(method)
                    seen_methods.add(method[0])

    # Check sensors
    sensors = capabilities.get("sensors", [])
    if sensors:
        for sensor in sensors:
            sensor_type = sensor.get("type", "")
            if sensor_type in SENSOR_METHODS:
                for method in SENSOR_METHODS[sensor_type]:
                    if method[0] not in seen_methods:
                        methods.append(method)
                        seen_methods.add(method[0])

    return methods


def sanitize_class_name(name: str) -> str:
    """Convert asset name/registry_id to a valid Python class name."""
    # Handle registry_id format: vendor/name -> VendorName
    if "/" in name:
        parts = name.split("/")
        name = "".join(parts)

    # Remove invalid characters and convert to PascalCase
    words = []
    current_word = []
    for char in name:
        if char.isalnum():
            current_word.append(char)
        elif current_word:
            words.append("".join(current_word))
            current_word = []
    if current_word:
        words.append("".join(current_word))

    # Capitalize each word
    class_name = "".join(word.capitalize() for word in words)

    # Ensure it starts with a letter
    if class_name and not class_name[0].isalpha():
        class_name = "Asset" + class_name

    return class_name or "UnknownAsset"


def generate_asset_class(asset: dict[str, Any]) -> str:
    """Generate a stub class for a specific asset."""
    registry_id = asset.get("registry_id") or asset.get("uuid", "unknown")
    name = asset.get("name", "Unknown Asset")
    capabilities = asset.get("capabilities", {})

    class_name = sanitize_class_name(registry_id)
    methods = get_methods_for_capabilities(capabilities)

    lines = [
        f"class {class_name}Twin(Twin):",
        f'    """',
        f"    Digital twin for {name}",
        f"    Registry ID: {registry_id}",
        f'    """',
    ]

    # Add capability-specific methods
    for method_name, signature, docstring in methods:
        if "(" in signature:
            # It's a method
            lines.append(f"    def {method_name}{signature}:")
            lines.append(f'        """{docstring}"""')
            lines.append(f"        ...")
        else:
            # It's a property
            lines.append(f"    @property")
            lines.append(f"    def {method_name}(self) -> {signature}:")
            lines.append(f'        """{docstring}"""')
            lines.append(f"        ...")

    # If no capability-specific methods, just pass
    if not methods:
        lines.append("    pass")

    return "\n".join(lines)


def generate_asset_registry(assets: list[dict[str, Any]]) -> str:
    """Generate the ASSET_REGISTRY mapping."""
    lines = ["# Asset registry mapping registry_id to Twin class"]
    lines.append("ASSET_REGISTRY: dict[str, type[Twin]] = {")

    for asset in assets:
        registry_id = asset.get("registry_id")
        if registry_id:
            class_name = sanitize_class_name(registry_id)
            lines.append(f'    "{registry_id}": {class_name}Twin,')

    lines.append("}")
    return "\n".join(lines)


def generate_stubs(assets: list[dict[str, Any]], output_path: Path) -> None:
    """Generate the .pyi stub file."""
    # Header
    header = '''"""
Type stubs for Cyberwave SDK with asset-specific capabilities.

This file is auto-generated by stubs_generator.py
Do not edit manually.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from .twin import Twin, JointController

'''

    # Generate asset classes
    asset_classes = []
    for asset in assets:
        if asset.get("registry_id"):  # Only generate for assets with registry_id
            asset_classes.append(generate_asset_class(asset))

    # Generate registry
    registry = generate_asset_registry(assets)

    # Combine all
    content = header + "\n\n".join(asset_classes) + "\n\n" + registry + "\n"

    # Write to file
    output_path.write_text(content)
    print(f"Generated stubs at: {output_path}")


def generate_capabilities_cache(
    assets: list[dict[str, Any]], output_path: Path
) -> None:
    """Generate a JSON cache of asset capabilities for runtime use."""
    cache = {}
    for asset in assets:
        registry_id = asset.get("registry_id")
        if registry_id:
            cache[registry_id] = {
                "uuid": asset.get("uuid"),
                "name": asset.get("name"),
                "capabilities": asset.get("capabilities", {}),
            }

    output_path.write_text(json.dumps(cache, indent=2))
    print(f"Generated capabilities cache at: {output_path}")


def generate_client_stubs(assets: list[dict[str, Any]], output_path: Path) -> None:
    """Generate client.pyi stub file with overloaded twin() method for IDE autocomplete."""

    # Group assets by their return type
    type_groups: dict[str, list[str]] = {}
    for asset in assets:
        registry_id = asset.get("registry_id")
        if not registry_id:
            continue
        capabilities = asset.get("capabilities", {})
        base_class = get_base_class(capabilities)
        if base_class not in type_groups:
            type_groups[base_class] = []
        type_groups[base_class].append(registry_id)

    # Generate the stub content
    lines = [
        '"""',
        "Type stubs for Cyberwave client with asset-specific return types.",
        "",
        "This file is auto-generated by stubs_generator.py",
        "Do not edit manually.",
        '"""',
        "",
        "from typing import Literal, Optional, overload",
        "",
        "from .twin import (",
        "    Twin,",
        "    CameraTwin,",
        "    DepthCameraTwin,",
        "    FlyingTwin,",
        "    GripperTwin,",
        "    FlyingCameraTwin,",
        "    GripperCameraTwin,",
        "    LocomoteTwin,",
        "    LocomoteCameraTwin,",
        "    LocomoteGripperTwin,",
        "    LocomoteGripperCameraTwin,",
        "    LocomoteGripperDepthCameraTwin,",
        "    LocomoteDepthCameraTwin,",
        "    LocomoteCameraTwin,",
        ")",
        "from .camera import CameraStreamer",
        "from .controller import EdgeController",
        "from .mqtt_client import CyberwaveMQTTClient",
        "from .resources import (",
        "    WorkspaceManager,",
        "    ProjectManager,",
        "    EnvironmentManager,",
        "    AssetManager,",
        "    TwinManager,",
        ")",
        "from .config import CyberwaveConfig",
        "from .utils import TimeReference",
        "",
        "",
        "class Cyberwave:",
        '    """Main client for the Cyberwave Digital Twin Platform."""',
        "    ",
        "    config: CyberwaveConfig",
        "    workspaces: WorkspaceManager",
        "    projects: ProjectManager",
        "    environments: EnvironmentManager",
        "    assets: AssetManager",
        "    twins: TwinManager",
        "    ",
        "    def __init__(",
        "        self,",
        "        base_url: str | None = None,",
        "        token: str | None = None,",
        "        api_key: str | None = None,",
        "        mqtt_host: str | None = None,",
        "        mqtt_port: int | None = None,",
        "        mqtt_username: str | None = None,",
        "        mqtt_password: str | None = None,",
        "        **config_kwargs,",
        "    ) -> None: ...",
        "    ",
        "    @property",
        "    def mqtt(self) -> CyberwaveMQTTClient: ...",
        "    ",
    ]

    # Generate overloads for each return type
    for return_type, registry_ids in type_groups.items():
        if return_type == "Twin":
            continue  # Skip base Twin, it will be the fallback

        for registry_id in registry_ids:
            lines.extend(
                [
                    "    @overload",
                    f"    def twin(",
                    f"        self,",
                    f'        asset_key: Literal["{registry_id}"],',
                    f"        environment_id: str | None = None,",
                    f"        twin_id: str | None = None,",
                    f"        **kwargs,",
                    f"    ) -> {return_type}: ...",
                    "    ",
                ]
            )

    # Add fallback overload for unknown assets
    lines.extend(
        [
            "    @overload",
            "    def twin(",
            "        self,",
            "        asset_key: str,",
            "        environment_id: str | None = None,",
            "        twin_id: str | None = None,",
            "        **kwargs,",
            "    ) -> Twin: ...",
            "    ",
            "    def twin(",
            "        self,",
            "        asset_key: str,",
            "        environment_id: str | None = None,",
            "        twin_id: str | None = None,",
            "        **kwargs,",
            "    ) -> Twin: ...",
            "    ",
            "    def video_stream(",
            "        self,",
            "        twin_uuid: str,",
            "        camera_id: int = 0,",
            "        fps: int = 10,",
            "        turn_servers: list | None = None,",
            "        time_reference: TimeReference | None = None,",
            "    ) -> CameraStreamer: ...",
            "    ",
            "    def controller(self, twin_uuid: str) -> EdgeController: ...",
            "    ",
            "    def configure(",
            "        self,",
            "        base_url: str | None = None,",
            "        token: str | None = None,",
            "        api_key: str | None = None,",
            "        environment_id: str | None = None,",
            "        workspace_id: str | None = None,",
            "        **kwargs,",
            "    ) -> None: ...",
            "    ",
            "    def disconnect(self) -> None: ...",
            "    ",
            '    def __enter__(self) -> "Cyberwave": ...',
            "    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Generated client stubs at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate type stubs for Cyberwave SDK based on asset capabilities"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API URL to fetch assets from (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated files (default: same as this script)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Use local API URL ({LOCAL_API_URL})",
    )

    args = parser.parse_args()

    api_url = LOCAL_API_URL if args.local else args.api_url
    output_dir = args.output_dir or Path(__file__).parent

    print(f"Fetching assets from {api_url}...")
    assets = fetch_assets(api_url)

    if not assets:
        print("No assets found or failed to fetch assets.", file=sys.stderr)
        print("Make sure the API is running and accessible.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(assets)} assets")

    # Generate stubs
    stubs_path = output_dir / "assets.pyi"
    generate_stubs(assets, stubs_path)

    # Generate capabilities cache
    cache_path = output_dir / "assets_capabilities.json"
    generate_capabilities_cache(assets, cache_path)

    # Generate client stubs for IDE autocomplete
    client_stubs_path = output_dir / "client.pyi"
    generate_client_stubs(assets, client_stubs_path)

    print("Done!")


if __name__ == "__main__":
    main()

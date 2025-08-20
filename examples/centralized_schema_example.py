#!/usr/bin/env python3
"""
Centralized Schema Example (updated)

This example shows how to generate a centralized Level YAML from a minimal
SDK-style environment dict (without using deprecated LevelDefinition types).
"""

from pathlib import Path

from cyberwave import (
    convert_sdk_to_centralized,
    generate_centralized_level_yaml,
    validate_centralized_level,
    CYBERWAVE_LEVEL_API_VERSION,
    CentralizedSchemaError,
)


def main():
    print("🚀 Centralized Schema Example (minimal)")
    print("=" * 50)

    # Minimal SDK-style environment dict
    sdk_level = {
        "metadata": {
            "title": "Centralized Demo",
            "id": "centralized-demo-001",
            "description": "Minimal environment to demonstrate centralized schema generation",
            "units": "meters",
        },
        "environment": {
            "lighting": {"ambient": 0.3, "directional": [{"direction": [-1, -1, -1], "intensity": 0.8}]}
        },
        "entities": [
            {
                "id": "demo_robot_1",
                "archetype": "robot",
                "transform": {"position": [0, 0, 1], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                "capabilities": ["navigate_2d", "camera"],
            },
            {
                "id": "bin_1",
                "archetype": "fixed_asset",
                "transform": {"position": [0.6, 0, 0.8], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                "properties": {"color": [0.2, 0.2, 0.2]},
            },
        ],
    }

    # Convert to centralized format
    try:
        centralized_level = convert_sdk_to_centralized(sdk_level)
        print("✅ Converted to centralized format")
        print(f"   🆔 API Version: {centralized_level['apiVersion']}")
        print(f"   📋 Kind: {centralized_level['kind']}")
        print(f"   📝 Name: {centralized_level['metadata']['name']}")
        print(f"   🎯 Entities: {len(centralized_level['scene']['entities'])}")
    except CentralizedSchemaError as e:
        print(f"❌ Conversion failed: {e}")
        return

    # Generate YAML
    try:
        centralized_yaml = generate_centralized_level_yaml(sdk_level)
        print(f"✅ Generated YAML ({len(centralized_yaml)} chars)")
        preview = centralized_yaml.split("\n")[:10]
        print("\n📋 YAML Preview:")
        for i, line in enumerate(preview, start=1):
            print(f"   {i:2}: {line}")
    except CentralizedSchemaError as e:
        print(f"❌ YAML generation failed: {e}")
        return

    # Validate
    is_valid, errors = validate_centralized_level(centralized_level)
    if is_valid:
        print("✅ Centralized format is valid!")
    else:
        print(f"❌ Validation failed: {errors}")
        return

    # Save
    out_path = Path("examples/output_centralized_format.yml")
    out_path.write_text(centralized_yaml)
    print(f"💾 Saved: {out_path}")


if __name__ == "__main__":
    main()
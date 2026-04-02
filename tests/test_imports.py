"""
Test that all SDK modules can be imported correctly.

This test verifies that the SDK can be imported as users would import it
after installing via 'pip install cyberwave'.

To run these tests:
    cd /path/to/cyberwave-python
    poetry install
    poetry run python tests/test_imports.py
    # or
    poetry run pytest tests/test_imports.py
"""

import sys


def test_basic_imports():
    """Test that basic SDK modules can be imported"""
    from cyberwave import Cyberwave

    assert hasattr(Cyberwave, "twin")
    assert callable(getattr(Cyberwave, "twin"))
    print("✓ Basic imports successful")


def test_exception_imports():
    """Test that exceptions can be imported"""
    from cyberwave import (
        CyberwaveError,
        CyberwaveAPIError,
        CyberwaveConnectionError,
        CyberwaveTimeoutError,
        CyberwaveValidationError,
    )

    assert issubclass(CyberwaveAPIError, CyberwaveError)
    assert issubclass(CyberwaveConnectionError, CyberwaveError)
    assert issubclass(CyberwaveTimeoutError, CyberwaveError)
    assert issubclass(CyberwaveValidationError, CyberwaveError)
    print("✓ Exception imports successful")


def test_client_creation():
    """Test that Cyberwave client can be created"""
    from cyberwave import Cyberwave

    client = Cyberwave(base_url="http://localhost:8000", api_key="test_key")
    assert client.config.base_url == "http://localhost:8000"
    assert client.config.api_key == "test_key"

    assert hasattr(client, "workspaces")
    assert hasattr(client, "projects")
    assert hasattr(client, "environments")
    assert hasattr(client, "assets")
    assert hasattr(client, "twins")
    print("✓ Client creation successful")


def test_config():
    """Test configuration module"""
    from cyberwave import CyberwaveConfig, get_config, set_config
    from cyberwave.config import CyberwaveConfig as DirectConfig

    assert CyberwaveConfig is DirectConfig

    config = CyberwaveConfig(base_url="http://test:8000", api_key="test")
    assert config.base_url == "http://test:8000"
    assert config.api_key == "test"

    assert callable(get_config)
    assert callable(set_config)

    current_config = get_config()
    assert current_config is not None
    print("✓ Config module successful")


def test_compact_api():
    """Test compact API functions"""
    from cyberwave import configure, twin, get_client

    assert callable(configure)
    assert callable(twin)
    assert callable(get_client)
    print("✓ Compact API successful")


def test_mqtt_client():
    """Test MQTT client can be imported and instantiated"""
    from cyberwave import CyberwaveMQTTClient
    from cyberwave.mqtt import CyberwaveMQTTClient as BaseMQTTClient

    client = BaseMQTTClient(
        mqtt_broker="localhost",
        mqtt_port=1883,
        api_key="test_api_key",
        auto_connect=False,
    )

    assert client.mqtt_broker == "localhost"
    assert client.mqtt_port == 1883
    assert client.api_key == "test_api_key"
    assert client.topic_prefix == ""

    wrapper_client = CyberwaveMQTTClient(
        mqtt_broker="localhost",
        mqtt_port=1883,
        api_key="test_api_key",
        auto_connect=False,
    )
    assert wrapper_client.api_key == "test_api_key"
    assert wrapper_client.topic_prefix == ""
    print("✓ MQTT client import successful")


def test_twin_classes():
    """Test that twin classes can be imported"""
    from cyberwave import (
        Twin,
        JointController,
        TwinControllerHandle,
        CameraTwin,
        DepthCameraTwin,
        FlyingTwin,
        GripperTwin,
        FlyingCameraTwin,
        GripperCameraTwin,
        create_twin,
    )

    assert callable(Twin)
    assert callable(CameraTwin)
    assert callable(DepthCameraTwin)
    assert callable(FlyingTwin)
    assert callable(GripperTwin)
    assert callable(FlyingCameraTwin)
    assert callable(GripperCameraTwin)
    assert callable(create_twin)
    assert JointController is not None
    assert TwinControllerHandle is not None
    print("✓ Twin classes import successful")


def test_motion_navigation():
    """Test motion and navigation classes"""
    from cyberwave import (
        TwinMotionHandle,
        ScopedMotionHandle,
        TwinNavigationHandle,
        NavigationPlan,
    )

    assert TwinMotionHandle is not None
    assert ScopedMotionHandle is not None
    assert TwinNavigationHandle is not None
    assert callable(NavigationPlan)
    print("✓ Motion and navigation imports successful")


def test_resource_managers():
    """Test resource manager classes"""
    from cyberwave import (
        WorkspaceManager,
        ProjectManager,
        EnvironmentManager,
        AssetManager,
        TwinManager,
    )

    assert WorkspaceManager is not None
    assert ProjectManager is not None
    assert EnvironmentManager is not None
    assert AssetManager is not None
    assert TwinManager is not None
    assert callable(WorkspaceManager)
    assert callable(ProjectManager)
    assert callable(EnvironmentManager)
    assert callable(AssetManager)
    assert callable(TwinManager)
    print("✓ Resource managers import successful")


def test_camera_streaming():
    """Test camera streaming classes (optional, may not be available)"""
    try:
        from cyberwave import (
            CameraStreamer,
            CV2VideoTrack,
            CV2CameraStreamer,
            VirtualVideoTrack,
            VirtualCameraStreamer,
            CallbackVideoTrack,  # Backwards compatibility
            CallbackCameraStreamer,  # Backwards compatibility
            RealSenseVideoTrack,
            RealSenseStreamer,
            BaseVideoTrack,
            BaseVideoStreamer,
        )
    except ImportError:
        print("✓ Camera streaming imports handled gracefully (dependencies not installed)")
        return

    if CameraStreamer is not None and CV2CameraStreamer is not None:
        assert CameraStreamer is CV2CameraStreamer  # Legacy alias
    print("✓ Camera streaming imports successful")


def test_utils_and_constants():
    """Test utility classes and constants"""
    from cyberwave.utils import TimeReference
    from cyberwave import (
        SOURCE_TYPE_EDGE,
        SOURCE_TYPE_TELE,
        SOURCE_TYPE_EDIT,
        SOURCE_TYPE_SIM,
        SOURCE_TYPES,
    )

    assert TimeReference is not None
    tr = TimeReference()
    assert hasattr(tr, "update")
    assert hasattr(tr, "read")

    assert SOURCE_TYPE_EDGE in SOURCE_TYPES
    assert isinstance(SOURCE_TYPES, (list, tuple))
    assert SOURCE_TYPE_EDGE == "edge"
    assert SOURCE_TYPE_TELE == "tele"
    assert SOURCE_TYPE_EDIT == "edit"
    assert SOURCE_TYPE_SIM == "sim"
    print("✓ Utils and constants import successful")


def test_device_fingerprinting():
    """Test device fingerprinting functions"""
    from cyberwave import (
        generate_fingerprint,
        get_device_info,
        format_device_info_table,
    )

    assert callable(generate_fingerprint)
    assert callable(get_device_info)
    assert callable(format_device_info_table)
    print("✓ Device fingerprinting imports successful")


def test_version():
    """Test version information"""
    from cyberwave import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
    print(f"✓ Version check successful: {__version__}")


def test_keyboard_teleop():
    """Test keyboard teleop classes"""
    from cyberwave import KeyboardBindings, KeyboardTeleop

    assert KeyboardBindings is not None
    assert KeyboardTeleop is not None
    assert callable(KeyboardTeleop)
    print("✓ Keyboard teleop imports successful")


def test_edge_controller():
    """Test edge controller class"""
    from cyberwave import EdgeController

    assert EdgeController is not None
    assert callable(EdgeController)
    print("✓ Edge controller import successful")


def test_all_exports():
    """Test that all items in __all__ can be imported"""
    from cyberwave import __all__

    import cyberwave

    failed_imports = [name for name in __all__ if not hasattr(cyberwave, name)]
    assert not failed_imports, f"Some exports not available: {failed_imports}"
    print(f"✓ All {len(__all__)} exports are available")


if __name__ == "__main__":
    print("Cyberwave SDK Import Tests")
    print("=" * 50)

    tests = [
        test_basic_imports,
        test_exception_imports,
        test_client_creation,
        test_config,
        test_compact_api,
        test_mqtt_client,
        test_twin_classes,
        test_motion_navigation,
        test_resource_managers,
        test_camera_streaming,
        test_utils_and_constants,
        test_device_fingerprinting,
        test_keyboard_teleop,
        test_edge_controller,
        test_version,
        test_all_exports,
    ]

    results = []
    test_names = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            test()
            results.append(True)
        except (AssertionError, Exception) as e:
            print(f"  ✗ {e}")
            results.append(False)
        test_names.append(test.__name__)

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)
    failed = total - passed

    print("\nTest Results:")
    for i, (name, result) in enumerate(zip(test_names, results), 1):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {i:2d}. {status} - {name}")

    print("\n" + "=" * 50)
    print(f"Total: {total} tests")
    print(f"Passed: {passed} tests")
    if failed > 0:
        print(f"Failed: {failed} tests")
    print("=" * 50)

    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} test(s) failed")
        sys.exit(1)

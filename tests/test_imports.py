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
    try:
        from cyberwave import Cyberwave

        # Check that the class exists and has the twin method
        assert hasattr(Cyberwave, 'twin')
        assert callable(getattr(Cyberwave, 'twin'))
        print("✓ Basic imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    return True


def test_exception_imports():
    """Test that exceptions can be imported"""
    try:
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
    except ImportError as e:
        print(f"✗ Exception import error: {e}")
        return False
    return True


def test_client_creation():
    """Test that Cyberwave client can be created"""
    try:
        from cyberwave import Cyberwave

        client = Cyberwave(base_url="http://localhost:8000", api_key="test_key")
        assert client.config.base_url == "http://localhost:8000"
        assert client.config.api_key == "test_key"
        
        # Check that instance attributes are set
        assert hasattr(client, 'workspaces')
        assert hasattr(client, 'projects')
        assert hasattr(client, 'environments')
        assert hasattr(client, 'assets')
        assert hasattr(client, 'twins')
        print("✓ Client creation successful")
    except Exception as e:
        print(f"✗ Client creation error: {e}")
        return False
    return True


def test_config():
    """Test configuration module"""
    try:
        from cyberwave import CyberwaveConfig, get_config, set_config
        from cyberwave.config import CyberwaveConfig as DirectConfig

        # Test direct import
        assert CyberwaveConfig is DirectConfig
        
        config = CyberwaveConfig(base_url="http://test:8000", api_key="test")
        assert config.base_url == "http://test:8000"
        assert config.api_key == "test"
        
        # Test get_config and set_config functions
        assert callable(get_config)
        assert callable(set_config)
        
        # Test that get_config returns a config object
        current_config = get_config()
        assert current_config is not None
        
        print("✓ Config module successful")
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False
    return True


def test_compact_api():
    """Test compact API functions"""
    try:
        from cyberwave import configure, twin, get_client

        # Just test that they exist and are callable
        assert callable(configure)
        assert callable(twin)
        assert callable(get_client)
        print("✓ Compact API successful")
    except Exception as e:
        print(f"✗ Compact API error: {e}")
        return False
    return True


def test_mqtt_client():
    """Test MQTT client can be imported and instantiated"""
    try:
        from cyberwave import CyberwaveMQTTClient
        from cyberwave.mqtt import CyberwaveMQTTClient as BaseMQTTClient
        from cyberwave.config import CyberwaveConfig

        # The exported CyberwaveMQTTClient is a wrapper that takes a config
        # Test the base client directly
        client = BaseMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_password="test_token",
            auto_connect=False,  # Don't actually connect
        )

        assert client.mqtt_broker == "localhost"
        assert client.mqtt_port == 1883
        assert client.mqtt_password == "test_token"
        assert client.topic_prefix == ""

        # Test the wrapper with a config object
        config = CyberwaveConfig(
            base_url="http://localhost:8000",
            api_key="test_key",
            mqtt_host="localhost",
            mqtt_port=1883,
            mqtt_password="test_token",
        )
        wrapper_client = CyberwaveMQTTClient(config)
        assert wrapper_client.topic_prefix == ""

        print("✓ MQTT client import successful")
    except Exception as e:
        print(f"✗ MQTT client error: {e}")
        import traceback

        traceback.print_exc()
        return False
    return True


def test_twin_classes():
    """Test that twin classes can be imported"""
    try:
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
    except Exception as e:
        print(f"✗ Twin classes import error: {e}")
        return False
    return True


def test_motion_navigation():
    """Test motion and navigation classes"""
    try:
        from cyberwave import (
            TwinMotionHandle,
            ScopedMotionHandle,
            TwinNavigationHandle,
            NavigationPlan,
        )

        # All should be importable
        assert TwinMotionHandle is not None
        assert ScopedMotionHandle is not None
        assert TwinNavigationHandle is not None
        assert callable(NavigationPlan)
        print("✓ Motion and navigation imports successful")
    except Exception as e:
        print(f"✗ Motion/navigation import error: {e}")
        return False
    return True


def test_resource_managers():
    """Test resource manager classes"""
    try:
        from cyberwave import (
            WorkspaceManager,
            ProjectManager,
            EnvironmentManager,
            AssetManager,
            TwinManager,
        )

        # All should be importable
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
    except Exception as e:
        print(f"✗ Resource managers import error: {e}")
        return False
    return True


def test_camera_streaming():
    """Test camera streaming classes (optional, may not be available)"""
    try:
        from cyberwave import (
            CameraStreamer,
            CV2VideoTrack,
            CV2CameraStreamer,
            CallbackVideoTrack,
            CallbackCameraStreamer,
            RealSenseVideoTrack,
            RealSenseStreamer,
            BaseVideoTrack,
            BaseVideoStreamer,
        )

        # These may be None if camera dependencies aren't installed
        if CameraStreamer is not None and CV2CameraStreamer is not None:
            assert CameraStreamer is CV2CameraStreamer  # Legacy alias
            # Check all classes are available
            classes_available = all([
                CV2VideoTrack is not None,
                CV2CameraStreamer is not None,
                CallbackVideoTrack is not None,
                CallbackCameraStreamer is not None,
                RealSenseVideoTrack is not None,
                RealSenseStreamer is not None,
                BaseVideoTrack is not None,
                BaseVideoStreamer is not None,
            ])
            if classes_available:
                print("✓ Camera streaming imports successful (all dependencies installed)")
            else:
                print("✓ Camera streaming imports successful (partial dependencies installed)")
        elif CameraStreamer is None and CV2CameraStreamer is None:
            print("✓ Camera streaming imports handled gracefully (dependencies not installed)")
        else:
            # Partial import - still OK
            print("✓ Camera streaming imports handled gracefully (partial dependencies)")
    except ImportError:
        # ImportError is expected if dependencies aren't installed
        print("✓ Camera streaming imports handled gracefully (dependencies not installed)")
    except Exception as e:
        print(f"✗ Camera streaming import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_utils_and_constants():
    """Test utility classes and constants"""
    try:
        # Import TimeReference directly from utils module (more reliable)
        from cyberwave.utils import TimeReference
        
        from cyberwave import (
            SOURCE_TYPE_EDGE,
            SOURCE_TYPE_TELE,
            SOURCE_TYPE_EDIT,
            SOURCE_TYPE_SIM,
            SOURCE_TYPES,
        )

        # TimeReference is a class
        assert TimeReference is not None
        # Verify we can instantiate it
        tr = TimeReference()
        assert hasattr(tr, 'update')
        assert hasattr(tr, 'read')
        
        # Test constants
        assert SOURCE_TYPE_EDGE in SOURCE_TYPES
        assert isinstance(SOURCE_TYPES, (list, tuple))
        assert SOURCE_TYPE_EDGE == "edge"
        assert SOURCE_TYPE_TELE == "tele"
        assert SOURCE_TYPE_EDIT == "edit"
        assert SOURCE_TYPE_SIM == "sim"
        print("✓ Utils and constants import successful")
    except Exception as e:
        print(f"✗ Utils/constants import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_device_fingerprinting():
    """Test device fingerprinting functions"""
    try:
        from cyberwave import (
            generate_fingerprint,
            get_device_info,
            format_device_info_table,
        )

        assert callable(generate_fingerprint)
        assert callable(get_device_info)
        assert callable(format_device_info_table)
        print("✓ Device fingerprinting imports successful")
    except Exception as e:
        print(f"✗ Device fingerprinting import error: {e}")
        return False
    return True


def test_version():
    """Test version information"""
    try:
        from cyberwave import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0
        print(f"✓ Version check successful: {__version__}")
    except Exception as e:
        print(f"✗ Version check error: {e}")
        return False
    return True


def test_keyboard_teleop():
    """Test keyboard teleop classes"""
    try:
        from cyberwave import KeyboardBindings, KeyboardTeleop

        assert KeyboardBindings is not None
        assert KeyboardTeleop is not None
        # KeyboardTeleop should be callable (it's a class)
        assert callable(KeyboardTeleop)
        print("✓ Keyboard teleop imports successful")
    except Exception as e:
        print(f"✗ Keyboard teleop import error: {e}")
        return False
    return True


def test_edge_controller():
    """Test edge controller class"""
    try:
        from cyberwave import EdgeController

        assert EdgeController is not None
        assert callable(EdgeController)
        print("✓ Edge controller import successful")
    except Exception as e:
        print(f"✗ Edge controller import error: {e}")
        return False
    return True


def test_all_exports():
    """Test that all items in __all__ can be imported"""
    try:
        from cyberwave import __all__
        
        # Import everything from __all__
        import cyberwave
        
        failed_imports = []
        for item_name in __all__:
            if not hasattr(cyberwave, item_name):
                failed_imports.append(item_name)
        
        if failed_imports:
            print(f"✗ Some exports not available: {failed_imports}")
            return False
        
        print(f"✓ All {len(__all__)} exports are available")
        return True
    except Exception as e:
        print(f"✗ All exports test error: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        result = test()
        results.append(result)
        test_names.append(test.__name__)

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    failed = total - passed
    
    # Show individual test results
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

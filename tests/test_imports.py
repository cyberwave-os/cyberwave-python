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
        import cyberwave
        # assert hasattr(cyberwave, 'Cyberwave')
        # assert hasattr(cyberwave, 'Twin')
        # assert hasattr(cyberwave, 'configure')
        # assert hasattr(cyberwave, 'twin')
        # assert hasattr(cyberwave, 'simulation')
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
            CyberwaveConnectionError
        )
        assert issubclass(CyberwaveAPIError, CyberwaveError)
        assert issubclass(CyberwaveConnectionError, CyberwaveError)
        print("✓ Exception imports successful")
    except ImportError as e:
        print(f"✗ Exception import error: {e}")
        return False
    return True


def test_client_creation():
    """Test that Cyberwave client can be created"""
    try:
        from cyberwave import Cyberwave
        client = Cyberwave(
            base_url="http://localhost:8000",
            api_key="test_key"
        )
        assert client.config.base_url == "http://localhost:8000"
        assert client.config.api_key == "test_key"
        print("✓ Client creation successful")
    except Exception as e:
        print(f"✗ Client creation error: {e}")
        return False
    return True


def test_config():
    """Test configuration module"""
    try:
        from cyberwave.config import CyberwaveConfig, get_config, set_config
        config = CyberwaveConfig(
            base_url="http://test:8000",
            api_key="test"
        )
        assert config.base_url == "http://test:8000"
        assert config.api_key == "test"
        print("✓ Config module successful")
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False
    return True


def test_compact_api():
    """Test compact API functions"""
    try:
        from cyberwave import configure, simulation
        # Just test that they exist and are callable/accessible
        assert callable(configure)
        assert hasattr(simulation, 'play')
        assert hasattr(simulation, 'pause')
        print("✓ Compact API successful")
    except Exception as e:
        print(f"✗ Compact API error: {e}")
        return False
    return True


if __name__ == "__main__":
    print("Cyberwave SDK Import Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_exception_imports,
        test_client_creation,
        test_config,
        test_compact_api,
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)


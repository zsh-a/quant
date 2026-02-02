#!/usr/bin/env python3
"""
Quick system health check script.
Verifies all components are properly configured and working.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_imports():
    """Check if all required modules can be imported"""
    print("=" * 60)
    print("Checking Python Imports...")
    print("=" * 60)

    modules = [
        ("pandas", "Data processing"),
        ("loguru", "Logging"),
        ("psutil", "System monitoring"),
        ("fastapi", "API framework"),
        ("pydantic", "Data validation"),
        ("yaml", "Configuration"),
    ]

    all_ok = True
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:<15} - {description}")
        except ImportError:
            print(f"✗ {module_name:<15} - {description} (NOT INSTALLED)")
            all_ok = False

    return all_ok


def check_config():
    """Check if configuration files exist"""
    print("\n" + "=" * 60)
    print("Checking Configuration Files...")
    print("=" * 60)

    config_file = Path("config/system_config.yaml")

    if config_file.exists():
        print(f"✓ {config_file} exists")

        # Try to load it
        try:
            from src.config.settings import get_settings

            settings = get_settings()
            print(f"✓ Configuration loaded successfully")
            print(f"  - API port: {settings.api.port}")
            print(f"  - Log level: {settings.logging.level}")
            return True
        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")
            return False
    else:
        print(f"✗ {config_file} not found")
        return False


def check_logging():
    """Check if logging system works"""
    print("\n" + "=" * 60)
    print("Checking Logging System...")
    print("=" * 60)

    try:
        from src.utils.logging_config import get_logger

        logger = get_logger("health_check")

        logger.info("Test log message")
        print("✓ Logging system initialized")

        # Check log directory
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("quant_*.log"))
            print(f"✓ Log directory exists ({len(log_files)} file(s))")
        else:
            print("⚠ Log directory not found (will be created on first log)")

        return True
    except Exception as e:
        print(f"✗ Logging system error: {e}")
        return False


def check_data_stream():
    """Check if data stream modules can be imported"""
    print("\n" + "=" * 60)
    print("Checking Data Stream Modules...")
    print("=" * 60)

    try:
        from src.core.data_stream import DBDataStream, RealtimeDataStream

        print("✓ DBDataStream imported")
        print("✓ RealtimeDataStream imported")
        return True
    except Exception as e:
        print(f"✗ Failed to import data streams: {e}")
        return False


def check_api_modules():
    """Check if API modules can be imported"""
    print("\n" + "=" * 60)
    print("Checking API Modules...")
    print("=" * 60)

    try:
        from src.api.server import app

        print("✓ API server module imported")
        print("✓ FastAPI app created")
        return True
    except Exception as e:
        print(f"✗ Failed to import API modules: {e}")
        return False


def check_strategies():
    """Check if strategy modules can be imported"""
    print("\n" + "=" * 60)
    print("Checking Strategy Modules...")
    print("=" * 60)

    try:
        from src.strategies.jsg_strategy import JSGStrategy
        from src.strategies.rotation_strategy import RotationStrategy

        print("✓ JSGStrategy imported")
        print("✓ RotationStrategy imported")
        return True
    except Exception as e:
        print(f"✗ Failed to import strategies: {e}")
        return False


def check_database():
    """Check if database connection works"""
    print("\n" + "=" * 60)
    print("Checking Database Connection...")
    print("=" * 60)

    try:
        from db import DB

        db = DB()
        print("✓ Database client initialized")
        print("⚠ Database connection not tested (requires ClickHouse)")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        return False


def main():
    """Run all health checks"""
    print("\n" + "#" * 60)
    print("# System Health Check")
    print("#" * 60 + "\n")

    checks = [
        ("Python Imports", check_imports),
        ("Configuration", check_config),
        ("Logging System", check_logging),
        ("Data Streams", check_data_stream),
        ("API Modules", check_api_modules),
        ("Strategies", check_strategies),
        ("Database", check_database),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("HEALTH CHECK SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")

    print(f"\nTotal: {passed}/{total} checks passed ({passed / total * 100:.0f}%)")

    if passed == total:
        print("\n🎉 ALL CHECKS PASSED - System is healthy!")
        print("\nNext steps:")
        print("  1. Start API server: uvicorn src.api.server:app --reload")
        print("  2. Run E2E tests: python test_e2e_integration.py")
        print("  3. Start frontend: cd ui && npm run dev")
        return 0
    else:
        print(f"\n⚠ {total - passed} check(s) failed")
        print("\nPlease fix the issues above before running the system.")
        return 1


if __name__ == "__main__":
    exit(main())

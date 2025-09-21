#!/usr/bin/env python3
"""
Build script for VoiRS FFI Python package.

This script provides a convenient interface for building and testing
the Python package using maturin. It supports development builds,
release builds, and wheel generation.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command and print its description."""
    print(f"\n🔧 {description}")
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stdout:
            print(f"   Stdout: {e.stdout}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return False

def check_dependencies():
    """Check if required build dependencies are available."""
    print("🔍 Checking build dependencies...")
    
    dependencies = {
        'maturin': ['maturin', '--version'],
        'cargo': ['cargo', '--version'],
        'python': ['python', '--version'],
    }
    
    missing = []
    for name, cmd in dependencies.items():
        if not run_command(cmd, f"Checking {name}", check=False):
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nTo install missing dependencies:")
        if 'maturin' in missing:
            print("  pip install maturin")
        if 'cargo' in missing:
            print("  Install Rust: https://rustup.rs/")
        return False
    
    print("✅ All dependencies available")
    return True

def build_package(mode='develop', features=None, target=None):
    """Build the Python package using maturin."""
    cmd = ['maturin']
    
    if mode == 'develop':
        cmd.append('develop')
        description = "Building development package"
    elif mode == 'build':
        cmd.append('build')
        description = "Building package"
    elif mode == 'release':
        cmd.extend(['build', '--release'])
        description = "Building release package"
    else:
        raise ValueError(f"Unknown build mode: {mode}")
    
    if features:
        cmd.extend(['--features', features])
        description += f" with features: {features}"
    
    if target:
        cmd.extend(['--target', target])
        description += f" for target: {target}"
    
    return run_command(cmd, description)

def run_tests():
    """Run Python tests for the package."""
    print("\n🧪 Running Python tests...")
    
    # Check if pytest is available
    if not run_command(['python', '-m', 'pytest', '--version'], 
                      "Checking pytest", check=False):
        print("   Installing pytest...")
        if not run_command(['pip', 'install', 'pytest', 'pytest-asyncio'], 
                          "Installing pytest"):
            return False
    
    # Run tests
    test_cmd = [
        'python', '-m', 'pytest',
        'tests/python/',
        '-v',
        '--tb=short'
    ]
    
    return run_command(test_cmd, "Running Python test suite")

def check_package():
    """Check the built package for common issues."""
    print("\n🔍 Checking package...")
    
    # Try importing the package
    import_cmd = [
        'python', '-c',
        'import voirs_ffi; print(f"✅ Package imported successfully, version: {voirs_ffi.__version__}")'
    ]
    
    return run_command(import_cmd, "Testing package import")

def build_wheel():
    """Build wheel distribution."""
    return run_command(['maturin', 'build', '--release'], "Building wheel")

def main():
    """Main build script entry point."""
    parser = argparse.ArgumentParser(description="Build VoiRS FFI Python package")
    parser.add_argument(
        '--mode', 
        choices=['develop', 'build', 'release'], 
        default='develop',
        help='Build mode (default: develop)'
    )
    parser.add_argument(
        '--features',
        help='Comma-separated list of features to enable'
    )
    parser.add_argument(
        '--target',
        help='Target platform for cross-compilation'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests after building'
    )
    parser.add_argument(
        '--wheel',
        action='store_true',
        help='Build wheel distribution'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check package after building'
    )
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency checking'
    )
    
    args = parser.parse_args()
    
    print("🏗️  VoiRS FFI Python Package Builder")
    print("=" * 40)
    
    # Check dependencies
    if not args.skip_deps and not check_dependencies():
        sys.exit(1)
    
    # Build package
    success = True
    
    if args.wheel:
        success = build_wheel()
    else:
        success = build_package(
            mode=args.mode,
            features=args.features,
            target=args.target
        )
    
    if not success:
        print("\n❌ Build failed")
        sys.exit(1)
    
    # Run optional checks
    if args.check:
        if not check_package():
            print("\n❌ Package check failed")
            sys.exit(1)
    
    if args.test:
        if not run_tests():
            print("\n❌ Tests failed")
            sys.exit(1)
    
    print("\n✅ Build completed successfully!")
    
    if args.mode == 'develop':
        print("\n📝 Next steps:")
        print("   • Test the package: python -c 'import voirs_ffi; print(voirs_ffi.__version__)'")
        print("   • Run tests: python -m pytest tests/python/")
        print("   • Check examples: cd examples && python example.py")

if __name__ == '__main__':
    main()
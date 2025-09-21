#!/usr/bin/env python3
"""
Setup script for VoiRS FFI Python bindings.

This script provides compatibility with legacy build systems and tools
that expect a setup.py file. The actual build configuration is defined
in pyproject.toml using maturin as the build backend.
"""

from setuptools import setup

if __name__ == "__main__":
    setup(
        name="voirs-ffi",
        zip_safe=False,
        # All other configuration is in pyproject.toml
    )
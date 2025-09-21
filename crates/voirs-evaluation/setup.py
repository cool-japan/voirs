#!/usr/bin/env python3
"""
Setup script for VoiRS Evaluation Python bindings.

This script builds the Rust extension module and creates a Python package
for the VoiRS evaluation system, enabling seamless integration with Python
scientific computing tools.
"""

import os
import sys
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext, parallel_compile
from setuptools import setup, Extension
from setuptools_rust import Binding, RustExtension

# Read version from Cargo.toml
def get_version():
    """Extract version from Cargo.toml"""
    cargo_toml = Path(__file__).parent / "Cargo.toml"
    if cargo_toml.exists():
        with open(cargo_toml, 'r') as f:
            for line in f:
                if line.startswith('version'):
                    return line.split('=')[1].strip().strip('"')
    return "0.1.0"

# Read long description from README
def get_long_description():
    """Read long description from README file"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "VoiRS Evaluation System Python Bindings"

setup(
    name="voirs-evaluation",
    version=get_version(),
    author="VoiRS Team",
    author_email="team@voirs.ai",
    description="Python bindings for VoiRS speech evaluation system",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/VoiRS/voirs-evaluation",
    project_urls={
        "Bug Tracker": "https://github.com/VoiRS/voirs-evaluation/issues",
        "Documentation": "https://docs.voirs.ai/evaluation",
        "Source Code": "https://github.com/VoiRS/voirs-evaluation",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    rust_extensions=[
        RustExtension(
            "voirs_evaluation",
            path="Cargo.toml",
            binding=Binding.PyO3,
            features=["python"],
            debug=False,
        )
    ],
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "librosa>=0.8.0",
        "soundfile>=0.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark>=3.4.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "examples": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "streamlit>=1.0.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-benchmark>=3.4.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "streamlit>=1.0.0",
        ],
    },
    zip_safe=False,
    keywords=[
        "speech",
        "audio",
        "evaluation",
        "quality",
        "assessment",
        "tts",
        "synthesis",
        "pronunciation",
        "pesq",
        "stoi",
        "mcd",
        "rust",
        "python",
        "numpy",
        "scipy",
        "scientific computing",
    ],
)
"""
VoiRS FFI - Python bindings for VoiRS speech synthesis library.

This package provides high-performance Python bindings for the VoiRS
speech synthesis engine, built with Rust and exposed through FFI.

Features:
- High-quality text-to-speech synthesis
- Multiple voice options and languages
- Real-time and batch processing
- Audio effects and post-processing
- Memory-efficient operations
- Thread-safe operations

Basic usage:
    >>> from voirs_ffi import VoirsPipeline
    >>> pipeline = VoirsPipeline()
    >>> audio = pipeline.synthesize("Hello, world!")
    >>> audio.play()
"""

try:
    # Import the Rust extension module
    from .voirs_ffi import *  # noqa: F403,F401
    
    # Re-export main classes for convenient access
    from .voirs_ffi import (  # noqa: F401
        VoirsPipeline,
        PyAudioBuffer,
        SynthesisConfig,
        VoiceInfo,
        __version__,
        HAS_NUMPY,
        HAS_GPU,
    )
    
except ImportError as e:
    # Provide helpful error message if the extension module is not available
    import sys
    
    msg = (
        f"Failed to import voirs_ffi extension module: {e}\n\n"
        "This usually means:\n"
        "1. The package was not built correctly\n"
        "2. Missing required system dependencies\n"
        "3. Incompatible Python version or platform\n\n"
        "Please check the installation instructions at:\n"
        "https://docs.voirs.dev/python/installation\n"
    )
    
    print(msg, file=sys.stderr)
    raise ImportError(msg) from e

# Package metadata
__author__ = "VoiRS Team"
__email__ = "team@voirs.dev"
__license__ = "MIT"
__url__ = "https://github.com/cool-japan/voirs"

# Convenience functions for common operations
def create_pipeline(**kwargs):
    """
    Create a VoiRS pipeline with optional configuration.
    
    This is a convenience function that provides a simple way to create
    and configure a synthesis pipeline.
    
    Args:
        **kwargs: Configuration options passed to VoirsPipeline.with_config()
        
    Returns:
        VoirsPipeline: Configured pipeline instance
        
    Example:
        >>> pipeline = create_pipeline(use_gpu=True, num_threads=4)
        >>> audio = pipeline.synthesize("Hello!")
    """
    if kwargs:
        return VoirsPipeline.with_config(**kwargs)
    else:
        return VoirsPipeline()

def synthesize_text(text, **config):
    """
    Quick text-to-speech synthesis with automatic pipeline management.
    
    This is a convenience function for simple synthesis tasks that
    automatically creates and manages a pipeline instance.
    
    Args:
        text (str): Text to synthesize
        **config: Optional configuration for the pipeline
        
    Returns:
        PyAudioBuffer: Synthesized audio
        
    Example:
        >>> audio = synthesize_text("Hello, world!", use_gpu=True)
        >>> audio.save("output.wav")
    """
    pipeline = create_pipeline(**config)
    return pipeline.synthesize(text)

# Version check helper
def check_compatibility():
    """
    Check system compatibility and available features.
    
    Returns:
        dict: Dictionary with compatibility information
        
    Example:
        >>> info = check_compatibility()
        >>> print(f"GPU available: {info['gpu']}")
        >>> print(f"NumPy support: {info['numpy']}")
    """
    return {
        "version": __version__,
        "gpu": HAS_GPU,
        "numpy": HAS_NUMPY,
        "python_version": sys.version,
    }

# Export all public symbols
__all__ = [
    # Core classes
    "VoirsPipeline",
    "PyAudioBuffer", 
    "SynthesisConfig",
    "VoiceInfo",
    
    # Convenience functions
    "create_pipeline",
    "synthesize_text",
    "check_compatibility",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__", 
    "__license__",
    "__url__",
    
    # Feature flags
    "HAS_NUMPY",
    "HAS_GPU",
]
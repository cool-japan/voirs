"""
Utility functions for VoiRS FFI Python bindings.

This module provides helper functions and utilities that enhance
the core FFI functionality with Python-specific conveniences.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

def validate_text_input(text: Union[str, bytes]) -> str:
    """
    Validate and normalize text input for synthesis.
    
    Args:
        text: Input text to validate
        
    Returns:
        str: Normalized text string
        
    Raises:
        TypeError: If input is not string or bytes
        ValueError: If text is empty or only whitespace
    """
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 encoding in text input: {e}")
    
    if not isinstance(text, str):
        raise TypeError(f"Text input must be str or bytes, got {type(text)}")
    
    if not text.strip():
        raise ValueError("Text input cannot be empty or only whitespace")
    
    return text.strip()

def validate_audio_parameters(
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    volume: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate audio parameters and return normalized values.
    
    Args:
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        volume: Volume level (0.0 to 2.0)
        
    Returns:
        dict: Validated parameters
        
    Raises:
        ValueError: If parameters are out of valid range
    """
    result = {}
    
    if sample_rate is not None:
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive integer, got {sample_rate}")
        if sample_rate < 8000 or sample_rate > 48000:
            warnings.warn(f"Sample rate {sample_rate} Hz is outside typical range (8kHz-48kHz)")
        result['sample_rate'] = sample_rate
    
    if channels is not None:
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"Channels must be positive integer, got {channels}")
        if channels > 8:
            warnings.warn(f"High channel count ({channels}) may not be supported by all systems")
        result['channels'] = channels
    
    if volume is not None:
        if not isinstance(volume, (int, float)) or volume < 0.0:
            raise ValueError(f"Volume must be non-negative number, got {volume}")
        if volume > 2.0:
            warnings.warn(f"Volume {volume} exceeds recommended maximum (2.0)")
        result['volume'] = float(volume)
    
    return result

def detect_audio_format(filename: Union[str, Path]) -> str:
    """
    Detect audio format from filename extension.
    
    Args:
        filename: Path to audio file
        
    Returns:
        str: Detected format ('wav', 'flac', 'mp3', etc.)
        
    Raises:
        ValueError: If format cannot be determined
    """
    path = Path(filename)
    extension = path.suffix.lower().lstrip('.')
    
    # Map common extensions to formats
    format_map = {
        'wav': 'wav',
        'wave': 'wav',
        'flac': 'flac',
        'mp3': 'mp3',
        'ogg': 'ogg',
        'oga': 'ogg',
        'm4a': 'aac',
        'aac': 'aac',
        'wma': 'wma',
    }
    
    if extension in format_map:
        return format_map[extension]
    
    raise ValueError(f"Unknown audio format for file: {filename}")

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration (e.g., "1:23.45")
    """
    if seconds < 0:
        return "0:00.00"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes > 0:
        return f"{minutes}:{remaining_seconds:05.2f}"
    else:
        return f"0:{remaining_seconds:05.2f}"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size (e.g., "1.5 MB")
    """
    if size_bytes < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"

def get_system_info() -> Dict[str, Any]:
    """
    Get system information relevant to audio processing.
    
    Returns:
        dict: System information including Python version, platform, etc.
    """
    import platform
    
    info = {
        'python_version': sys.version,
        'python_implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'architecture': platform.architecture(),
    }
    
    try:
        import numpy as np
        info['numpy_version'] = np.__version__
    except ImportError:
        info['numpy_version'] = None
    
    return info

def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional dependencies.
    
    Returns:
        dict: Dictionary mapping dependency names to availability
    """
    dependencies = {}
    
    # Check NumPy
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    # Check SoundFile for audio I/O
    try:
        import soundfile
        dependencies['soundfile'] = True
    except ImportError:
        dependencies['soundfile'] = False
    
    # Check librosa for audio analysis
    try:
        import librosa
        dependencies['librosa'] = True
    except ImportError:
        dependencies['librosa'] = False
    
    # Check matplotlib for visualization
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        dependencies['matplotlib'] = False
    
    return dependencies

class VoiRSError(Exception):
    """Base exception class for VoiRS FFI errors."""
    pass

class VoiRSConfigError(VoiRSError):
    """Exception raised for configuration-related errors."""
    pass

class VoiRSAudioError(VoiRSError):
    """Exception raised for audio processing errors."""
    pass

class VoiRSSynthesisError(VoiRSError):
    """Exception raised for synthesis-related errors."""
    pass
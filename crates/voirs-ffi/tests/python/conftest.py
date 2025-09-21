"""
pytest configuration for VoiRS FFI Python tests.

This module provides fixtures and configuration for comprehensive testing
of the Python bindings for VoiRS speech synthesis.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Generator, AsyncGenerator
import time
import numpy as np

# Import the VoiRS Python module (when available)
try:
    import voirs_ffi as voirs
    VOIRS_AVAILABLE = True
except ImportError:
    VOIRS_AVAILABLE = False


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def voirs_pipeline():
    """Create a VoiRS pipeline for testing."""
    if not VOIRS_AVAILABLE:
        pytest.skip("VoiRS Python bindings not available")
    
    try:
        pipeline = await voirs.VoirsPipeline.create()
        yield pipeline
        # Cleanup would go here if needed
    except Exception as e:
        pytest.skip(f"Could not create VoiRS pipeline: {e}")


@pytest.fixture
def temp_audio_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for audio files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_texts() -> dict:
    """Provide sample texts for testing."""
    return {
        "short": "Hello world",
        "medium": "This is a medium length text for testing speech synthesis functionality.",
        "long": "This is a much longer text that contains multiple sentences. "
                "It should test the speech synthesis system's ability to handle "
                "extended content with proper pacing and natural flow. "
                "The system should maintain consistent quality throughout "
                "the entire synthesis process.",
        "special_chars": "Testing special characters: 123, $50, 75%, @user, #hashtag!",
        "multilingual": "Hello world. Bonjour le monde. Hola mundo.",
        "numbers": "Testing numbers: 1, 2, 3, one hundred twenty-three, 456.78",
        "punctuation": "Testing punctuation: Hello! How are you? Fine, thanks. End.",
    }


@pytest.fixture
def synthesis_configs() -> dict:
    """Provide various synthesis configurations for testing."""
    return {
        "default": {},
        "fast": {"speaking_rate": 1.5},
        "slow": {"speaking_rate": 0.8},
        "high_pitch": {"pitch_shift": 5.0},
        "low_pitch": {"pitch_shift": -5.0},
        "loud": {"volume_gain": 1.5},
        "quiet": {"volume_gain": 0.5},
        "high_quality": {"quality": "high", "sample_rate": 48000},
        "low_quality": {"quality": "low", "sample_rate": 22050},
    }


@pytest.fixture
def audio_formats() -> list:
    """Provide list of audio formats to test."""
    return ["wav", "flac"]  # MP3 might not be available


@pytest.fixture
def performance_metrics():
    """Provide performance tracking utilities."""
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
        
        def start_timer(self, name: str):
            self.metrics[name] = {"start": time.time()}
        
        def end_timer(self, name: str):
            if name in self.metrics:
                self.metrics[name]["end"] = time.time()
                self.metrics[name]["duration"] = (
                    self.metrics[name]["end"] - self.metrics[name]["start"]
                )
        
        def get_duration(self, name: str) -> float:
            return self.metrics.get(name, {}).get("duration", 0.0)
        
        def clear(self):
            self.metrics.clear()
    
    return PerformanceTracker()


@pytest.fixture
def memory_tracker():
    """Provide memory usage tracking utilities."""
    import psutil
    import gc
    
    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.initial_memory = None
        
        def start_tracking(self):
            gc.collect()  # Force garbage collection
            self.initial_memory = self.process.memory_info().rss
        
        def get_memory_usage(self) -> dict:
            current_memory = self.process.memory_info().rss
            return {
                "current_mb": current_memory / 1024 / 1024,
                "initial_mb": (self.initial_memory or 0) / 1024 / 1024,
                "delta_mb": (current_memory - (self.initial_memory or 0)) / 1024 / 1024,
            }
        
        def check_memory_leak(self, threshold_mb: float = 10.0) -> bool:
            """Check if memory usage increased beyond threshold."""
            usage = self.get_memory_usage()
            return usage["delta_mb"] > threshold_mb
    
    return MemoryTracker()


@pytest.fixture
def audio_validator():
    """Provide audio validation utilities."""
    class AudioValidator:
        @staticmethod
        def validate_audio_buffer(audio_buffer):
            """Validate audio buffer properties."""
            assert hasattr(audio_buffer, 'samples'), "Audio buffer missing samples"
            assert hasattr(audio_buffer, 'sample_rate'), "Audio buffer missing sample_rate"
            assert hasattr(audio_buffer, 'channels'), "Audio buffer missing channels"
            assert hasattr(audio_buffer, 'duration'), "Audio buffer missing duration"
            
            # Check for reasonable values
            assert audio_buffer.sample_rate > 0, "Invalid sample rate"
            assert audio_buffer.channels > 0, "Invalid channel count"
            assert audio_buffer.duration >= 0, "Invalid duration"
            
            # Check samples
            samples = audio_buffer.samples
            assert len(samples) > 0, "No audio samples"
            assert all(isinstance(s, (int, float)) for s in samples[:10]), "Invalid sample types"
            
            return True
        
        @staticmethod
        def validate_audio_file(file_path: Path, expected_format: str = None):
            """Validate audio file."""
            assert file_path.exists(), f"Audio file does not exist: {file_path}"
            assert file_path.stat().st_size > 0, "Audio file is empty"
            
            if expected_format:
                assert file_path.suffix.lower() == f".{expected_format.lower()}", \
                    f"Expected {expected_format} format, got {file_path.suffix}"
            
            return True
        
        @staticmethod
        def check_audio_quality(audio_buffer, min_duration: float = 0.1):
            """Check basic audio quality metrics."""
            assert audio_buffer.duration >= min_duration, \
                f"Audio too short: {audio_buffer.duration}s < {min_duration}s"
            
            # Check for silence (all zeros)
            samples = audio_buffer.samples
            non_zero_samples = sum(1 for s in samples if abs(s) > 1e-6)
            silence_ratio = 1.0 - (non_zero_samples / len(samples))
            assert silence_ratio < 0.9, f"Audio mostly silent: {silence_ratio:.2%}"
            
            # Check for clipping
            clipped_samples = sum(1 for s in samples if abs(s) >= 0.99)
            clipping_ratio = clipped_samples / len(samples)
            assert clipping_ratio < 0.01, f"Audio clipping detected: {clipping_ratio:.2%}"
            
            return True
    
    return AudioValidator()


@pytest.fixture
def numpy_available():
    """Check if NumPy is available for testing."""
    try:
        import numpy as np
        return True
    except ImportError:
        return False


# Pytest marks for organizing tests
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow
pytest.mark.memory = pytest.mark.memory
pytest.mark.audio = pytest.mark.audio


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "memory: mark test as memory-related")
    config.addinivalue_line("markers", "audio: mark test as audio-related")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Skip tests if VoiRS is not available
        if not VOIRS_AVAILABLE and "voirs" in item.name.lower():
            item.add_marker(pytest.mark.skip(reason="VoiRS Python bindings not available"))
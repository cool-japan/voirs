"""
VoiRS Recognizer Python Package

A comprehensive voice recognition and analysis library for Python.
This package provides:

- Automatic Speech Recognition (ASR) with Whisper and other models
- Audio analysis including quality metrics, prosody, and speaker features
- Real-time processing capabilities
- Multi-language support
- Performance validation and optimization tools

Basic Usage:
    >>> import voirs_recognizer
    >>> 
    >>> # Load audio file
    >>> audio = voirs_recognizer.load_audio_file("speech.wav")
    >>> 
    >>> # Create ASR system
    >>> recognizer = voirs_recognizer.VoiRSRecognizer()
    >>> 
    >>> # Perform speech recognition
    >>> result = recognizer.recognize(audio)
    >>> print(result.text)
    >>> print(f"Confidence: {result.confidence:.2f}")
    >>> 
    >>> # Analyze audio quality
    >>> analyzer = voirs_recognizer.AudioAnalyzer()
    >>> analysis = analyzer.analyze(audio)
    >>> print(f"SNR: {analysis.get_quality_metric('snr'):.2f} dB")

For more detailed examples, see the examples directory.
"""

# Import the native Rust implementation
from .voirs_recognizer import *

__version__ = "0.1.0"
__author__ = "VoiRS Contributors"
__license__ = "MIT OR Apache-2.0"

# Re-export key classes and functions for convenience
__all__ = [
    # Core classes
    "VoiRSRecognizer",
    "AudioAnalyzer",
    "PerformanceValidator",
    
    # Configuration classes
    "ASRConfig",
    "AudioAnalysisConfig",
    
    # Data classes
    "AudioBuffer",
    "RecognitionResult",
    "WordTimestamp",
    "AudioAnalysisResult",
    
    # Utility functions
    "load_audio_file",
    "load_audio",  # Alias for load_audio_file
    "confidence_to_label",
    
    # Package metadata
    "__version__",
    "__author__",
    "__license__",
]

# Module-level convenience functions
def create_recognizer(language="en-US", word_timestamps=True, confidence_threshold=0.5):
    """
    Create a VoiRS recognizer with common configuration.
    
    Args:
        language: Language code (default: "en-US")
        word_timestamps: Enable word-level timestamps (default: True)
        confidence_threshold: Minimum confidence threshold (default: 0.5)
    
    Returns:
        VoiRSRecognizer: Configured recognizer instance
    """
    config = ASRConfig(
        language=language,
        word_timestamps=word_timestamps,
        confidence_threshold=confidence_threshold
    )
    return VoiRSRecognizer(config)

def create_analyzer(quality_metrics=True, prosody_analysis=True, speaker_analysis=True):
    """
    Create an audio analyzer with common configuration.
    
    Args:
        quality_metrics: Enable quality metric analysis (default: True)
        prosody_analysis: Enable prosody analysis (default: True)
        speaker_analysis: Enable speaker analysis (default: True)
    
    Returns:
        AudioAnalyzer: Configured analyzer instance
    """
    config = AudioAnalysisConfig(
        quality_metrics=quality_metrics,
        prosody_analysis=prosody_analysis,
        speaker_analysis=speaker_analysis
    )
    return AudioAnalyzer(config)

def analyze_audio_file(file_path, sample_rate=None, include_recognition=True, include_analysis=True):
    """
    Convenience function to analyze an audio file with both recognition and analysis.
    
    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate (optional)
        include_recognition: Perform speech recognition (default: True)
        include_analysis: Perform audio analysis (default: True)
    
    Returns:
        dict: Results dictionary with 'recognition' and 'analysis' keys
    """
    # Load audio
    audio = load_audio_file(file_path, sample_rate)
    
    results = {}
    
    # Speech recognition
    if include_recognition:
        recognizer = create_recognizer()
        results['recognition'] = recognizer.recognize(audio)
    
    # Audio analysis
    if include_analysis:
        analyzer = create_analyzer()
        results['analysis'] = analyzer.analyze(audio)
    
    return results

# Package information
def get_package_info():
    """Get package information and version details."""
    return {
        'name': 'voirs-recognizer',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'description': 'Voice recognition and analysis capabilities for VoiRS',
        'features': [
            'Automatic Speech Recognition (ASR)',
            'Audio quality analysis',
            'Prosodic feature extraction',
            'Speaker characteristic analysis',
            'Real-time processing',
            'Multi-language support',
            'Performance validation',
        ]
    }

# Development and debugging helpers
def _debug_info():
    """Internal function for debugging package state."""
    import sys
    import platform
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'package_version': __version__,
    }

# Backward compatibility alias
def load_audio(path, sample_rate=None):
    """Load audio from file path (alias for load_audio_file)."""
    return load_audio_file(path, sample_rate)
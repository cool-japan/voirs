"""
VoiRS FFI Python Type Stubs

Type definitions for the VoiRS speech synthesis library Python bindings.
This file provides comprehensive type hints for IDEs and static type checkers.
"""

from typing import Any, Callable, List, Optional, Union
import numpy as np

# Version and capability constants
__version__: str
HAS_NUMPY: bool
HAS_GPU: bool

class VoirsPipeline:
    """Main synthesis pipeline for text-to-speech generation."""
    
    def __init__(self) -> None:
        """Create a new VoiRS pipeline with default configuration."""
        ...
    
    @staticmethod
    def with_config(
        use_gpu: Optional[bool] = None,
        num_threads: Optional[int] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ) -> "VoirsPipeline":
        """
        Create a VoiRS pipeline with custom configuration.
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
            num_threads: Number of threads for synthesis operations
            cache_dir: Directory for caching models and data
            device: Specific device to use for synthesis
            
        Returns:
            Configured VoirsPipeline instance
            
        Raises:
            RuntimeError: If pipeline creation fails
        """
        ...
    
    def synthesize(self, text: str) -> "PyAudioBuffer":
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio buffer containing synthesized speech
            
        Raises:
            RuntimeError: If synthesis fails
        """
        ...
    
    def synthesize_ssml(self, ssml: str) -> "PyAudioBuffer":
        """
        Synthesize SSML markup to audio.
        
        Args:
            ssml: SSML markup to synthesize
            
        Returns:
            Audio buffer containing synthesized speech
            
        Raises:
            RuntimeError: If SSML synthesis fails
        """
        ...
    
    def batch_synthesize_with_progress(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
    ) -> List["SynthesisResult"]:
        """
        Synthesize multiple texts with progress tracking.
        
        Args:
            texts: List of texts to synthesize
            progress_callback: Optional callback for progress updates
                Signature: (current: int, total: int, progress: float, message: str) -> None
                
        Returns:
            List of SynthesisResult objects with audio and metrics
            
        Raises:
            RuntimeError: If synthesis fails
        """
        ...
    
    def synthesize_streaming(
        self,
        text: str,
        chunk_callback: Callable[[int, int, "PyAudioBuffer"], None],
        chunk_size: Optional[int] = None,
    ) -> "PyAudioBuffer":
        """
        Synthesize text with streaming audio chunks.
        
        Args:
            text: Text to synthesize
            chunk_callback: Callback for audio chunks
                Signature: (chunk_index: int, total_chunks: int, audio_chunk: PyAudioBuffer) -> None
            chunk_size: Size of audio chunks in samples (default: 1024)
            
        Returns:
            Complete PyAudioBuffer
            
        Raises:
            RuntimeError: If synthesis fails
        """
        ...
    
    def synthesize_with_error_callback(
        self,
        text: str,
        error_callback: Optional[Callable[["VoirsErrorInfo"], None]] = None,
    ) -> "PyAudioBuffer":
        """
        Synthesize text with enhanced error handling.
        
        Args:
            text: Text to synthesize
            error_callback: Optional callback for error handling
                Signature: (error_info: VoirsErrorInfo) -> None
                
        Returns:
            PyAudioBuffer containing synthesized audio
            
        Raises:
            RuntimeError: If synthesis fails
        """
        ...
    
    def synthesize_with_callbacks(
        self,
        text: str,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
        chunk_callback: Optional[Callable[[int, int, "PyAudioBuffer"], None]] = None,
        error_callback: Optional[Callable[["VoirsErrorInfo"], None]] = None,
        chunk_size: Optional[int] = None,
    ) -> "PyAudioBuffer":
        """
        Synthesize text with comprehensive callback support.
        
        Args:
            text: Text to synthesize
            progress_callback: Optional progress tracking callback
            chunk_callback: Optional streaming chunk callback
            error_callback: Optional error handling callback
            chunk_size: Size of audio chunks in samples
            
        Returns:
            PyAudioBuffer containing synthesized audio
            
        Raises:
            RuntimeError: If synthesis fails
            ValueError: If callbacks are not callable
        """
        ...
    
    def set_progress_callback(
        self,
        callback: Optional[Callable[[int, int, float, str], None]],
    ) -> None:
        """
        Set global progress callback for long-running operations.
        
        Args:
            callback: Progress callback function or None to disable
                Signature: (current: int, total: int, progress: float, message: str) -> None
                
        Raises:
            ValueError: If callback is not callable
        """
        ...
    
    def set_error_callback(
        self,
        callback: Optional[Callable[["VoirsErrorInfo"], None]],
    ) -> None:
        """
        Set global error callback for error handling.
        
        Args:
            callback: Error callback function or None to disable
                Signature: (error_info: VoirsErrorInfo) -> None
                
        Raises:
            ValueError: If callback is not callable
        """
        ...
    
    def set_voice(self, voice_id: str) -> None:
        """
        Set the voice for synthesis.
        
        Args:
            voice_id: Identifier of the voice to use
            
        Raises:
            RuntimeError: If voice setting fails
        """
        ...
    
    def get_voice(self) -> Optional[str]:
        """
        Get the current voice identifier.
        
        Returns:
            Current voice ID or None if no voice is set
        """
        ...
    
    def list_voices(self) -> List["PyVoiceInfo"]:
        """
        List all available voices.
        
        Returns:
            List of available voice information
            
        Raises:
            RuntimeError: If voice listing fails
        """
        ...
    
    @staticmethod
    def version() -> str:
        """
        Get the VoiRS library version.
        
        Returns:
            Version string
        """
        ...

class PyAudioBuffer:
    """Audio data container with NumPy integration and analysis capabilities."""
    
    def samples(self) -> bytes:
        """
        Get audio samples as raw bytes.
        
        Returns:
            Raw audio data as bytes (little-endian float32)
        """
        ...
    
    def samples_as_list(self) -> List[float]:
        """
        Get audio samples as a list of floats.
        
        Returns:
            List of float32 audio samples
        """
        ...
    
    def as_numpy(self) -> Union[np.ndarray, Any]:
        """
        Get audio samples as NumPy array.
        
        Returns:
            1D array for mono audio, 2D array [frames, channels] for multi-channel
            
        Note:
            Only available when numpy feature is enabled
        """
        ...
    
    def as_planar_numpy(self) -> Union[List[np.ndarray], Any]:
        """
        Get audio samples as planar NumPy arrays.
        
        Returns:
            List of 1D arrays, one per channel
            
        Note:
            Only available when numpy feature is enabled
        """
        ...
    
    @staticmethod
    def from_numpy(
        array: Union[np.ndarray, Any],
        sample_rate: int,
        channels: Optional[int] = None
    ) -> "PyAudioBuffer":
        """
        Create audio buffer from NumPy array.
        
        Args:
            array: NumPy array of audio samples
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (auto-detected if None)
            
        Returns:
            Audio buffer created from the array data
            
        Note:
            Only available when numpy feature is enabled
        """
        ...
    
    def apply_numpy_operation(
        self,
        operation: str,
        args: Optional[Any] = None
    ) -> None:
        """
        Apply NumPy-based audio operation in-place.
        
        Args:
            operation: Operation name ("normalize", "clip", "fade_in", "fade_out")
            args: Operation-specific arguments
            
        Note:
            Only available when numpy feature is enabled
        """
        ...
    
    def get_spectrum(self, window_size: Optional[int] = None) -> Union[np.ndarray, Any]:
        """
        Get frequency spectrum of the audio.
        
        Args:
            window_size: FFT window size (defaults to optimal size)
            
        Returns:
            Frequency spectrum as NumPy array
            
        Note:
            Only available when numpy feature is enabled
        """
        ...
    
    def resample(self, new_sample_rate: int) -> None:
        """
        Resample audio to new sample rate in-place.
        
        Args:
            new_sample_rate: Target sample rate in Hz
        """
        ...
    
    def sample_rate(self) -> int:
        """Get audio sample rate in Hz."""
        ...
    
    def channels(self) -> int:
        """Get number of audio channels."""
        ...
    
    def duration(self) -> float:
        """Get audio duration in seconds."""
        ...
    
    def length(self) -> int:
        """Get total number of audio samples."""
        ...
    
    def save(self, path: str, format: Optional[str] = None) -> None:
        """
        Save audio to file.
        
        Args:
            path: Output file path
            format: Audio format ("wav", "flac", "mp3", "opus", "ogg")
                   Auto-detected from file extension if None
                   
        Raises:
            RuntimeError: If file saving fails
        """
        ...
    
    def play(self, volume: Optional[float] = None, blocking: Optional[bool] = None) -> None:
        """
        Play audio directly to the system's audio output.
        
        Args:
            volume: Playback volume (0.0 to 2.0, default 1.0)
            blocking: Whether to block until playback completes (default True)
            
        Raises:
            ValueError: If volume is out of range
            RuntimeError: If audio playback fails
        """
        ...
    
    def play_async(self, volume: Optional[float] = None) -> None:
        """
        Play audio asynchronously (non-blocking).
        
        Args:
            volume: Playback volume (0.0 to 2.0, default 1.0)
            
        Raises:
            ValueError: If volume is out of range
            RuntimeError: If audio playback fails
        """
        ...
    
    def play_on_device(self, device_name: Optional[str] = None, volume: Optional[float] = None) -> None:
        """
        Play audio on a specific audio device.
        
        Args:
            device_name: Name of the audio device (None for default device)
            volume: Playback volume (0.0 to 2.0, default 1.0)
            
        Raises:
            ValueError: If volume is out of range or device name is invalid
            RuntimeError: If audio device is not available or playback fails
        """
        ...

class PyVoiceInfo:
    """Voice information container."""
    
    id: str
    """Unique voice identifier"""
    
    name: str
    """Human-readable voice name"""
    
    language: str
    """Voice language code (e.g., "en-US")"""
    
    quality: str
    """Voice quality description"""
    
    is_available: bool
    """Whether the voice is currently available"""

class PyStreamingProcessor:
    """
    Real-time streaming audio processor.
    
    Note:
        Only available when numpy feature is enabled
    """
    
    def __init__(self, chunk_size: int, sample_rate: int, channels: int) -> None:
        """
        Create streaming processor.
        
        Args:
            chunk_size: Size of audio chunks to process
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
        """
        ...
    
    def set_callback(self, callback: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Set audio processing callback.
        
        Args:
            callback: Function that takes audio chunk and returns processed chunk
        """
        ...
    
    def process_chunk(self, chunk: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """
        Process a single audio chunk.
        
        Args:
            chunk: Input audio chunk as NumPy array
            
        Returns:
            Processed audio chunk
        """
        ...
    
    def add_samples(self, samples: Union[np.ndarray, Any]) -> Optional[Union[np.ndarray, Any]]:
        """
        Add samples to streaming buffer.
        
        Args:
            samples: Audio samples to add
            
        Returns:
            Processed chunk if buffer is full, None otherwise
        """
        ...
    
    def flush(self) -> Optional[Union[np.ndarray, Any]]:
        """
        Flush remaining samples in buffer.
        
        Returns:
            Final processed chunk if any samples remain
        """
        ...

class PyAudioAnalyzer:
    """
    Audio analysis and feature extraction utilities.
    
    Note:
        Only available when numpy feature is enabled
    """
    
    def __init__(self) -> None:
        """Create audio analyzer."""
        ...
    
    @staticmethod
    def rms_energy(audio: Union[np.ndarray, Any]) -> float:
        """
        Calculate RMS energy of audio signal.
        
        Args:
            audio: Audio samples as NumPy array
            
        Returns:
            RMS energy value
        """
        ...
    
    @staticmethod
    def find_silence(
        audio: Union[np.ndarray, Any],
        threshold: float,
        min_duration: int
    ) -> Union[np.ndarray, Any]:
        """
        Find silent regions in audio.
        
        Args:
            audio: Audio samples as NumPy array
            threshold: Silence threshold (amplitude)
            min_duration: Minimum silence duration in samples
            
        Returns:
            2D array of silence regions [start, end] indices
        """
        ...
    
    @staticmethod
    def zero_crossing_rate(audio: Union[np.ndarray, Any]) -> float:
        """
        Calculate zero crossing rate of audio signal.
        
        Args:
            audio: Audio samples as NumPy array
            
        Returns:
            Zero crossing rate
        """
        ...
    
    @staticmethod
    def spectral_centroid(audio: Union[np.ndarray, Any], sample_rate: int) -> float:
        """
        Calculate spectral centroid of audio signal.
        
        Args:
            audio: Audio samples as NumPy array
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Spectral centroid in Hz
        """
        ...

class PySynthesisConfig:
    """Synthesis configuration parameters."""
    
    def __init__(self) -> None:
        """Create synthesis configuration with default values."""
        ...
    
    speaking_rate: float
    """Speaking rate multiplier (1.0 = normal speed)"""
    
    pitch_shift: float
    """Pitch shift in semitones (0.0 = no change)"""
    
    volume_gain: float
    """Volume gain multiplier (1.0 = no change)"""
    
    enable_enhancement: bool
    """Whether to enable audio enhancement"""
    
    output_format: str
    """Output audio format ("wav", "flac", "mp3")"""
    
    sample_rate: int
    """Output sample rate in Hz"""
    
    quality: str
    """Synthesis quality level ("low", "medium", "high")"""

# Exception types that may be raised
class VoirsError(Exception):
    """Base exception for VoiRS-related errors."""
    pass

class SynthesisError(VoirsError):
    """Raised when synthesis operations fail."""
    pass

class ConfigurationError(VoirsError):
    """Raised when configuration is invalid."""
    pass
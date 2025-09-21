# Error Handling Guide

Comprehensive guide to handling errors and exceptions in VoiRS Python bindings.

## Overview

VoiRS provides a structured error handling system with specific exception types for different error conditions. This guide covers exception types, error recovery strategies, debugging techniques, and best practices.

## Exception Hierarchy

```
Exception
└── VoirsError (base exception)
    ├── SynthesisError
    ├── ConfigurationError
    ├── VoiceNotFoundError
    ├── AudioProcessingError
    └── SystemCompatibilityError
```

## Exception Types

### VoirsError

Base exception for all VoiRS-related errors.

```python
class VoirsError(Exception):
    def __init__(self, message: str, error_code: Optional[int] = None)
    
    @property
    def error_code(self) -> Optional[int]
```

**Properties:**
- `error_code`: Numeric error code for programmatic handling

**Example:**
```python
from voirs_ffi import VoirsError

try:
    # VoiRS operation
    pass
except VoirsError as e:
    print(f"VoiRS error: {e}")
    if e.error_code:
        print(f"Error code: {e.error_code}")
```

### SynthesisError

Raised when text-to-speech synthesis fails.

**Common causes:**
- Invalid text input
- Model loading failures
- Processing pipeline errors
- Resource exhaustion during synthesis

**Example:**
```python
from voirs_ffi import VoirsPipeline, SynthesisError

pipeline = VoirsPipeline()

try:
    # This might fail with very long text
    very_long_text = "word " * 100000
    audio = pipeline.synthesize(very_long_text)
except SynthesisError as e:
    print(f"Synthesis failed: {e}")
    # Handle synthesis failure
```

### ConfigurationError

Raised when configuration parameters are invalid.

**Common causes:**
- Invalid parameter values
- Incompatible parameter combinations
- Missing required settings
- Unsupported feature requests

**Example:**
```python
from voirs_ffi import SynthesisConfig, ConfigurationError

try:
    # Invalid sample rate
    config = SynthesisConfig(sample_rate=99999)
    config.validate()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Use default configuration
    config = SynthesisConfig()
```

### VoiceNotFoundError

Raised when a requested voice is not available.

**Properties:**
- `voice_id`: The voice ID that was not found

**Example:**
```python
from voirs_ffi import VoirsPipeline, VoiceNotFoundError, list_voices

try:
    pipeline = VoirsPipeline()
    pipeline.set_voice("nonexistent-voice")
except VoiceNotFoundError as e:
    print(f"Voice '{e.voice_id}' not found")
    
    # Show available voices
    voices = list_voices()
    print("Available voices:")
    for voice in voices:
        print(f"  {voice.id}: {voice.name}")
```

### AudioProcessingError

Raised when audio processing operations fail.

**Common causes:**
- File format errors
- Invalid audio data
- Codec issues
- Disk space problems

**Example:**
```python
from voirs_ffi import VoirsPipeline, AudioProcessingError

pipeline = VoirsPipeline()
audio = pipeline.synthesize("Hello, world!")

try:
    # This might fail if path is invalid
    audio.save("/invalid/path/output.wav")
except AudioProcessingError as e:
    print(f"Audio processing failed: {e}")
    # Try alternative path
    audio.save("output.wav")
```

### SystemCompatibilityError

Raised when system requirements are not met.

**Common causes:**
- Missing GPU drivers
- Insufficient memory
- Unsupported operating system
- Missing dependencies

**Example:**
```python
from voirs_ffi import SynthesisConfig, SystemCompatibilityError, check_compatibility

try:
    # Check compatibility first
    compatibility = check_compatibility()
    if not compatibility['gpu']:
        raise SystemCompatibilityError("GPU required but not available")
    
    config = SynthesisConfig(use_gpu=True)
except SystemCompatibilityError as e:
    print(f"System compatibility issue: {e}")
    # Fall back to CPU processing
    config = SynthesisConfig(use_gpu=False)
```

## Error Handling Patterns

### Basic Error Handling

```python
from voirs_ffi import VoirsPipeline, VoirsError

def safe_synthesis(text: str) -> Optional[PyAudioBuffer]:
    """Safely synthesize text with error handling."""
    try:
        pipeline = VoirsPipeline()
        return pipeline.synthesize(text)
    except VoirsError as e:
        print(f"Synthesis failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
audio = safe_synthesis("Hello, world!")
if audio:
    audio.save("output.wav")
else:
    print("Synthesis failed")
```

### Specific Exception Handling

```python
from voirs_ffi import (
    VoirsPipeline, VoirsError, SynthesisError, 
    VoiceNotFoundError, ConfigurationError
)

def robust_synthesis(text: str, voice_id: str = None) -> Optional[PyAudioBuffer]:
    """Robust synthesis with specific error handling."""
    try:
        pipeline = VoirsPipeline()
        
        if voice_id:
            pipeline.set_voice(voice_id)
        
        return pipeline.synthesize(text)
        
    except VoiceNotFoundError as e:
        print(f"Voice '{e.voice_id}' not found, using default")
        # Retry without specific voice
        pipeline = VoirsPipeline()
        return pipeline.synthesize(text)
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        # Use safe default configuration
        pipeline = VoirsPipeline()
        return pipeline.synthesize(text)
        
    except SynthesisError as e:
        print(f"Synthesis error: {e}")
        # Could try with different settings
        return None
        
    except VoirsError as e:
        print(f"General VoiRS error: {e}")
        return None

# Usage
audio = robust_synthesis("Hello, world!", "female-1")
```

### Retry Mechanisms

```python
import time
from typing import Optional
from voirs_ffi import VoirsPipeline, VoirsError, SynthesisError

def synthesis_with_retry(
    text: str, 
    max_retries: int = 3, 
    delay: float = 1.0
) -> Optional[PyAudioBuffer]:
    """Synthesize text with automatic retry on failure."""
    
    for attempt in range(max_retries):
        try:
            pipeline = VoirsPipeline()
            return pipeline.synthesize(text)
            
        except SynthesisError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print("All retry attempts failed")
                return None
                
        except VoirsError as e:
            # Don't retry for non-synthesis errors
            print(f"Non-retryable error: {e}")
            return None
    
    return None

# Usage
audio = synthesis_with_retry("Hello, world!", max_retries=3)
```

### Fallback Strategies

```python
from voirs_ffi import (
    VoirsPipeline, SynthesisConfig, VoirsError, 
    SystemCompatibilityError, check_compatibility
)

def create_pipeline_with_fallback() -> VoirsPipeline:
    """Create pipeline with progressive fallback strategies."""
    
    # Try high-performance configuration first
    try:
        compatibility = check_compatibility()
        if compatibility['gpu']:
            config = SynthesisConfig(
                use_gpu=True,
                quality="high",
                num_threads=8
            )
            return VoirsPipeline(config)
    except SystemCompatibilityError:
        print("GPU not available, falling back to CPU")
    
    # Try medium performance configuration
    try:
        config = SynthesisConfig(
            use_gpu=False,
            quality="medium",
            num_threads=4
        )
        return VoirsPipeline(config)
    except VoirsError as e:
        print(f"Medium config failed: {e}")
    
    # Use minimal configuration as last resort
    try:
        config = SynthesisConfig(
            use_gpu=False,
            quality="low",
            num_threads=1,
            memory_limit_mb=128
        )
        return VoirsPipeline(config)
    except VoirsError as e:
        print(f"All configurations failed: {e}")
        raise

# Usage
try:
    pipeline = create_pipeline_with_fallback()
    print("Pipeline created successfully")
except VoirsError as e:
    print(f"Failed to create pipeline: {e}")
```

## Error Recovery Strategies

### Graceful Degradation

```python
from voirs_ffi import VoirsPipeline, SynthesisConfig, VoirsError

class RobustTTS:
    def __init__(self):
        self.pipeline = None
        self.current_config = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize pipeline with best available configuration."""
        configs = [
            # Try configurations in order of preference
            SynthesisConfig(use_gpu=True, quality="high"),
            SynthesisConfig(use_gpu=True, quality="medium"),
            SynthesisConfig(use_gpu=False, quality="medium"),
            SynthesisConfig(use_gpu=False, quality="low"),
        ]
        
        for config in configs:
            try:
                self.pipeline = VoirsPipeline(config)
                self.current_config = config
                print(f"Initialized with quality: {config.quality}, GPU: {config.use_gpu}")
                return
            except VoirsError as e:
                print(f"Config failed: {e}")
                continue
        
        raise VoirsError("Failed to initialize with any configuration")
    
    def synthesize(self, text: str) -> Optional[PyAudioBuffer]:
        """Synthesize with automatic quality reduction on failure."""
        try:
            return self.pipeline.synthesize(text)
        except VoirsError as e:
            print(f"Synthesis failed: {e}")
            return self._synthesize_with_reduced_quality(text)
    
    def _synthesize_with_reduced_quality(self, text: str) -> Optional[PyAudioBuffer]:
        """Try synthesis with progressively lower quality."""
        if self.current_config.quality == "low":
            print("Already at lowest quality, cannot reduce further")
            return None
        
        # Reduce quality
        quality_levels = ["ultra", "high", "medium", "low"]
        current_index = quality_levels.index(self.current_config.quality)
        
        for i in range(current_index + 1, len(quality_levels)):
            try:
                reduced_config = SynthesisConfig(
                    quality=quality_levels[i],
                    use_gpu=self.current_config.use_gpu,
                    num_threads=self.current_config.num_threads
                )
                
                temp_pipeline = VoirsPipeline(reduced_config)
                result = temp_pipeline.synthesize(text)
                
                # Update to working configuration
                self.pipeline = temp_pipeline
                self.current_config = reduced_config
                print(f"Reduced quality to: {quality_levels[i]}")
                return result
                
            except VoirsError as e:
                print(f"Quality {quality_levels[i]} also failed: {e}")
                continue
        
        return None

# Usage
tts = RobustTTS()
audio = tts.synthesize("Hello, world!")
```

### Resource Management

```python
import contextlib
from voirs_ffi import VoirsPipeline, VoirsError

@contextlib.contextmanager
def synthesis_pipeline(**config_kwargs):
    """Context manager for automatic pipeline cleanup."""
    pipeline = None
    try:
        pipeline = VoirsPipeline.with_config(**config_kwargs)
        yield pipeline
    except VoirsError as e:
        print(f"Pipeline error: {e}")
        raise
    finally:
        if pipeline:
            # Cleanup resources
            try:
                pipeline.reset()
            except:
                pass  # Ignore cleanup errors

# Usage
try:
    with synthesis_pipeline(use_gpu=True, quality="high") as pipeline:
        audio = pipeline.synthesize("Hello, world!")
        audio.save("output.wav")
except VoirsError as e:
    print(f"Synthesis failed: {e}")
```

## Debugging Techniques

### Logging and Diagnostics

```python
import logging
from voirs_ffi import VoirsPipeline, VoirsError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_synthesis(text: str):
    """Synthesis with detailed debugging information."""
    try:
        logger.info(f"Starting synthesis for text: '{text[:50]}...'")
        
        pipeline = VoirsPipeline()
        config = pipeline.get_config()
        
        logger.debug(f"Configuration: {config.to_dict()}")
        
        # Get system info
        from voirs_ffi import check_compatibility
        compatibility = check_compatibility()
        logger.debug(f"System compatibility: {compatibility}")
        
        # Synthesize
        audio = pipeline.synthesize(text)
        logger.info(f"Synthesis successful, duration: {audio.duration:.2f}s")
        
        return audio
        
    except VoirsError as e:
        logger.error(f"VoiRS error: {e}")
        if hasattr(e, 'error_code') and e.error_code:
            logger.error(f"Error code: {e.error_code}")
        raise

# Usage
try:
    audio = debug_synthesis("Hello, world!")
except VoirsError as e:
    logger.error(f"Debug synthesis failed: {e}")
```

### Performance Monitoring

```python
import time
import psutil
from voirs_ffi import VoirsPipeline, ProfiledPipeline, VoirsError

class DiagnosticTTS:
    def __init__(self):
        base_pipeline = VoirsPipeline()
        self.pipeline = ProfiledPipeline(base_pipeline)
        self.error_count = 0
        self.success_count = 0
    
    def synthesize_with_diagnostics(self, text: str) -> Optional[PyAudioBuffer]:
        """Synthesize with comprehensive diagnostics."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            audio, stats = self.pipeline.synthesize(text)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            self.success_count += 1
            
            print(f"Synthesis successful:")
            print(f"  Text length: {len(text)} characters")
            print(f"  Processing time: {end_time - start_time:.3f}s")
            print(f"  Memory usage: {end_memory - start_memory:.1f} MB")
            print(f"  Audio duration: {audio.duration:.2f}s")
            print(f"  Synthesis stats: {stats}")
            
            return audio
            
        except VoirsError as e:
            self.error_count += 1
            
            end_time = time.time()
            
            print(f"Synthesis failed:")
            print(f"  Error: {e}")
            print(f"  Text length: {len(text)} characters")
            print(f"  Time until failure: {end_time - start_time:.3f}s")
            print(f"  Error rate: {self.error_count / (self.error_count + self.success_count):.2%}")
            
            return None
    
    def get_summary(self):
        """Get diagnostic summary."""
        total = self.error_count + self.success_count
        if total == 0:
            return "No synthesis attempts"
        
        success_rate = self.success_count / total * 100
        return f"Success rate: {success_rate:.1f}% ({self.success_count}/{total})"

# Usage
diagnostic_tts = DiagnosticTTS()
texts = ["Hello", "This is a longer test", "Very long text " * 100]

for text in texts:
    audio = diagnostic_tts.synthesize_with_diagnostics(text)

print(diagnostic_tts.get_summary())
```

### Error Analysis

```python
from collections import defaultdict
from voirs_ffi import VoirsPipeline, VoirsError

class ErrorAnalyzer:
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_details = []
    
    def safe_synthesize(self, text: str) -> Optional[PyAudioBuffer]:
        """Synthesize with error tracking."""
        try:
            pipeline = VoirsPipeline()
            return pipeline.synthesize(text)
        except VoirsError as e:
            # Track error types
            error_type = type(e).__name__
            self.error_counts[error_type] += 1
            
            # Store error details
            self.error_details.append({
                'error_type': error_type,
                'message': str(e),
                'error_code': getattr(e, 'error_code', None),
                'text_length': len(text),
                'text_preview': text[:50] + '...' if len(text) > 50 else text
            })
            
            return None
    
    def print_error_summary(self):
        """Print comprehensive error analysis."""
        print("Error Analysis Summary:")
        print(f"Total errors: {sum(self.error_counts.values())}")
        
        for error_type, count in self.error_counts.items():
            print(f"  {error_type}: {count}")
        
        print("\nError Details:")
        for i, error in enumerate(self.error_details[-5:], 1):  # Last 5 errors
            print(f"  {i}. {error['error_type']}: {error['message']}")
            print(f"     Text: {error['text_preview']}")
            if error['error_code']:
                print(f"     Code: {error['error_code']}")

# Usage
analyzer = ErrorAnalyzer()

test_texts = [
    "Normal text",
    "",  # Empty text might cause error
    "x" * 100000,  # Very long text
]

for text in test_texts:
    audio = analyzer.safe_synthesize(text)

analyzer.print_error_summary()
```

## Best Practices

### Error Handling Guidelines

1. **Catch specific exceptions**: Handle specific error types when possible
2. **Provide meaningful fallbacks**: Don't just fail silently
3. **Log errors appropriately**: Include context and error details
4. **Validate inputs early**: Check parameters before processing
5. **Use timeouts**: Prevent indefinite blocking on errors

### Code Organization

```python
from typing import Optional, List
from voirs_ffi import VoirsPipeline, VoirsError, SynthesisConfig

class ProductionTTS:
    """Production-ready TTS with comprehensive error handling."""
    
    def __init__(self, config: Optional[SynthesisConfig] = None):
        self.config = config or SynthesisConfig()
        self.pipeline = None
        self.error_history: List[str] = []
        
        self._initialize()
    
    def _initialize(self):
        """Initialize pipeline with error handling."""
        try:
            self.pipeline = VoirsPipeline(self.config)
        except VoirsError as e:
            self._log_error(f"Initialization failed: {e}")
            # Try with minimal config
            self.config = SynthesisConfig(quality="low", use_gpu=False)
            self.pipeline = VoirsPipeline(self.config)
    
    def synthesize(self, text: str, **kwargs) -> Optional[PyAudioBuffer]:
        """Synthesize with full error handling."""
        if not self._validate_input(text):
            return None
        
        try:
            return self.pipeline.synthesize(text, **kwargs)
        except VoirsError as e:
            self._log_error(f"Synthesis failed: {e}")
            return self._attempt_recovery(text, **kwargs)
    
    def _validate_input(self, text: str) -> bool:
        """Validate input text."""
        if not text or not text.strip():
            self._log_error("Empty or whitespace-only text")
            return False
        
        if len(text) > 50000:  # Arbitrary limit
            self._log_error(f"Text too long: {len(text)} characters")
            return False
        
        return True
    
    def _attempt_recovery(self, text: str, **kwargs) -> Optional[PyAudioBuffer]:
        """Attempt error recovery."""
        # Try with reduced quality
        if self.config.quality != "low":
            try:
                temp_config = SynthesisConfig(
                    quality="low",
                    use_gpu=False,
                    num_threads=1
                )
                temp_pipeline = VoirsPipeline(temp_config)
                return temp_pipeline.synthesize(text, **kwargs)
            except VoirsError as e:
                self._log_error(f"Recovery attempt failed: {e}")
        
        return None
    
    def _log_error(self, message: str):
        """Log error with timestamp."""
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        error_msg = f"[{timestamp}] {message}"
        self.error_history.append(error_msg)
        print(error_msg)
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def get_error_history(self) -> List[str]:
        """Get recent error history."""
        return self.error_history.copy()
    
    def health_check(self) -> dict:
        """Perform health check."""
        try:
            test_audio = self.synthesize("test")
            return {
                'status': 'healthy',
                'pipeline_initialized': self.pipeline is not None,
                'test_synthesis': test_audio is not None,
                'recent_errors': len([e for e in self.error_history if 'recent' in e])
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'pipeline_initialized': self.pipeline is not None
            }

# Usage
tts = ProductionTTS()

# Health check
health = tts.health_check()
print(f"TTS Health: {health['status']}")

# Synthesis with error handling
audio = tts.synthesize("Hello, world!")
if audio:
    audio.save("output.wav")
else:
    print("Synthesis failed")
    print("Recent errors:", tts.get_error_history()[-3:])
```

### Testing Error Conditions

```python
import pytest
from voirs_ffi import (
    VoirsPipeline, SynthesisConfig, VoirsError, 
    SynthesisError, ConfigurationError, VoiceNotFoundError
)

def test_error_handling():
    """Test various error conditions."""
    
    # Test invalid configuration
    with pytest.raises(ConfigurationError):
        config = SynthesisConfig(sample_rate=-1)
        config.validate()
    
    # Test voice not found
    with pytest.raises(VoiceNotFoundError):
        pipeline = VoirsPipeline()
        pipeline.set_voice("nonexistent-voice")
    
    # Test synthesis with invalid input
    pipeline = VoirsPipeline()
    
    # Empty text should handle gracefully
    audio = pipeline.synthesize("")
    assert audio is None or audio.duration == 0
    
    # Very long text might fail
    very_long_text = "word " * 100000
    try:
        audio = pipeline.synthesize(very_long_text)
        # If it succeeds, that's also valid
    except SynthesisError:
        # Expected for very long text
        pass

if __name__ == "__main__":
    test_error_handling()
    print("Error handling tests passed")
```

## Common Error Scenarios

### Initialization Errors

```python
# GPU not available
from voirs_ffi import SynthesisConfig, SystemCompatibilityError

try:
    config = SynthesisConfig(use_gpu=True)
    # ... use config
except SystemCompatibilityError:
    config = SynthesisConfig(use_gpu=False)
```

### Resource Exhaustion

```python
# Out of memory
from voirs_ffi import VoirsPipeline, VoirsError
import gc

try:
    pipeline = VoirsPipeline()
    # ... process many requests
except VoirsError as e:
    if "memory" in str(e).lower():
        gc.collect()  # Force garbage collection
        pipeline.reset()  # Clear internal caches
```

### Network/File System Errors

```python
# File save errors
from voirs_ffi import AudioProcessingError
import os

try:
    audio.save("output.wav")
except AudioProcessingError as e:
    # Try alternative location
    temp_path = os.path.join(os.path.expanduser("~"), "output.wav")
    audio.save(temp_path)
```

This comprehensive error handling guide provides the foundation for building robust applications with VoiRS Python bindings.
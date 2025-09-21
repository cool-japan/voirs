# Memory Management Guide

Comprehensive guide to memory management best practices in VoiRS Python bindings.

## Overview

VoiRS Python bindings handle most memory management automatically, but understanding memory usage patterns and optimization techniques is crucial for building efficient applications, especially for long-running services or batch processing.

## Memory Architecture

### Components

1. **Python Objects**: Python wrapper objects for VoiRS components
2. **Rust FFI Layer**: Native Rust code handling actual processing
3. **Audio Buffers**: Large memory allocations for audio data
4. **Model Cache**: Cached neural network models and voice data
5. **Configuration Data**: Settings and metadata

### Memory Flow

```
Python Application
       ↓
Python Bindings (PyO3)
       ↓
Rust FFI Layer
       ↓
VoiRS Core Engine
       ↓
System Memory (Audio/Models)
```

## Automatic Memory Management

### Reference Counting

VoiRS uses automatic reference counting for most objects:

```python
from voirs_ffi import VoirsPipeline

# Pipeline created with ref count = 1
pipeline = VoirsPipeline()

# Additional reference
pipeline2 = pipeline
# Ref count = 2

# When variables go out of scope, ref count decreases
del pipeline
# Ref count = 1

# When last reference is removed, memory is freed
del pipeline2
# Ref count = 0, memory freed
```

### Garbage Collection Integration

```python
import gc
from voirs_ffi import VoirsPipeline

def create_pipelines():
    """Function that creates temporary pipelines."""
    pipelines = []
    for i in range(10):
        pipeline = VoirsPipeline()
        pipelines.append(pipeline)
    return pipelines
    # Local variables go out of scope here

# Create and discard pipelines
results = create_pipelines()

# Force garbage collection
gc.collect()  # Ensures cleanup of any cycles
```

## Memory Monitoring

### Basic Memory Tracking

```python
import psutil
import os
from voirs_ffi import VoirsPipeline

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Baseline memory
baseline = get_memory_usage()
print(f"Baseline memory: {baseline:.1f} MB")

# Create pipeline
pipeline = VoirsPipeline()
after_creation = get_memory_usage()
print(f"After pipeline creation: {after_creation:.1f} MB")
print(f"Pipeline overhead: {after_creation - baseline:.1f} MB")

# Synthesize audio
audio = pipeline.synthesize("Hello, world!")
after_synthesis = get_memory_usage()
print(f"After synthesis: {after_synthesis:.1f} MB")
print(f"Audio memory: {after_synthesis - after_creation:.1f} MB")

# Clean up
del audio
del pipeline
final = get_memory_usage()
print(f"After cleanup: {final:.1f} MB")
```

### Advanced Memory Profiling

```python
import tracemalloc
from voirs_ffi import VoirsPipeline

def profile_memory_usage():
    """Profile memory usage with detailed tracing."""
    
    # Start tracing
    tracemalloc.start()
    
    # Take snapshot before
    snapshot1 = tracemalloc.take_snapshot()
    
    # Perform VoiRS operations
    pipeline = VoirsPipeline()
    audio = pipeline.synthesize("This is a test sentence for memory profiling.")
    audio.save("test_output.wav")
    
    # Take snapshot after
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Top 10 memory allocations:")
    for index, stat in enumerate(top_stats[:10], 1):
        print(f"{index}. {stat}")
    
    # Clean up
    del audio
    del pipeline
    tracemalloc.stop()

# Run profiling
profile_memory_usage()
```

### Memory Monitoring Context Manager

```python
import contextlib
import psutil
import os

@contextlib.contextmanager
def memory_monitor(operation_name="Operation"):
    """Context manager for monitoring memory usage."""
    
    def get_memory():
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    start_memory = get_memory()
    peak_memory = start_memory
    
    try:
        yield lambda: get_memory() - start_memory  # Return current usage function
    finally:
        end_memory = get_memory()
        peak_memory = max(peak_memory, end_memory)
        
        print(f"{operation_name} Memory Usage:")
        print(f"  Start: {start_memory:.1f} MB")
        print(f"  End: {end_memory:.1f} MB")
        print(f"  Peak: {peak_memory:.1f} MB")
        print(f"  Net change: {end_memory - start_memory:.1f} MB")

# Usage
from voirs_ffi import VoirsPipeline

with memory_monitor("TTS Synthesis") as get_current_usage:
    pipeline = VoirsPipeline()
    print(f"After pipeline creation: +{get_current_usage():.1f} MB")
    
    audio = pipeline.synthesize("Hello, world!")
    print(f"After synthesis: +{get_current_usage():.1f} MB")
    
    audio.save("output.wav")
    print(f"After save: +{get_current_usage():.1f} MB")
```

## Memory Optimization Strategies

### Pipeline Reuse

```python
from voirs_ffi import VoirsPipeline

class OptimizedTTS:
    """Memory-optimized TTS class with pipeline reuse."""
    
    def __init__(self):
        self.pipeline = VoirsPipeline()
        self._synthesis_count = 0
        self._cleanup_threshold = 100
    
    def synthesize(self, text: str) -> PyAudioBuffer:
        """Synthesize with automatic cleanup."""
        audio = self.pipeline.synthesize(text)
        self._synthesis_count += 1
        
        # Periodic cleanup
        if self._synthesis_count >= self._cleanup_threshold:
            self.cleanup()
            self._synthesis_count = 0
        
        return audio
    
    def cleanup(self):
        """Manual cleanup of internal caches."""
        print("Performing periodic cleanup...")
        self.pipeline.reset()  # Clear internal caches
        
        # Force Python garbage collection
        import gc
        gc.collect()

# Usage
tts = OptimizedTTS()

# Process many texts efficiently
texts = ["Text " + str(i) for i in range(200)]
for text in texts:
    audio = tts.synthesize(text)
    # Process audio...
    del audio  # Explicitly delete when done
```

### Batch Processing

```python
from typing import List
from voirs_ffi import VoirsPipeline, PyAudioBuffer

def batch_synthesize(texts: List[str], batch_size: int = 10) -> List[PyAudioBuffer]:
    """Memory-efficient batch synthesis."""
    
    results = []
    pipeline = VoirsPipeline()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = []
        
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Process batch
        for text in batch:
            audio = pipeline.synthesize(text)
            batch_results.append(audio)
        
        results.extend(batch_results)
        
        # Clear cache every batch to prevent memory buildup
        if i % (batch_size * 5) == 0:  # Every 5 batches
            pipeline.reset()
            import gc
            gc.collect()
    
    return results

# Usage
large_text_list = ["Sample text " + str(i) for i in range(1000)]
audio_results = batch_synthesize(large_text_list, batch_size=20)
```

### Streaming Processing

```python
from typing import Iterator
from voirs_ffi import VoirsPipeline

def streaming_synthesize(texts: Iterator[str]) -> Iterator[PyAudioBuffer]:
    """Memory-efficient streaming synthesis."""
    
    pipeline = VoirsPipeline()
    count = 0
    
    for text in texts:
        audio = pipeline.synthesize(text)
        yield audio
        
        count += 1
        
        # Periodic cleanup
        if count % 50 == 0:
            pipeline.reset()
            import gc
            gc.collect()

# Usage with generator
def text_generator():
    """Generator that produces text on demand."""
    for i in range(10000):
        yield f"Generated text number {i}"

# Process without loading all into memory
for audio in streaming_synthesize(text_generator()):
    # Process each audio as it's generated
    audio.save(f"output_{hash(audio)}.wav")
    # Audio goes out of scope and is freed immediately
```

## Audio Buffer Management

### Efficient Audio Handling

```python
from voirs_ffi import VoirsPipeline
import numpy as np

class MemoryEfficientAudio:
    """Wrapper for memory-efficient audio processing."""
    
    def __init__(self, audio: PyAudioBuffer):
        self._audio = audio
        self._numpy_cache = None
    
    def get_samples(self) -> np.ndarray:
        """Get samples as NumPy array with caching."""
        if self._numpy_cache is None:
            self._numpy_cache = self._audio.to_numpy()
        return self._numpy_cache
    
    def save(self, path: str, format: str = "wav"):
        """Save audio and clear cache to free memory."""
        self._audio.save(path, format)
        # Clear NumPy cache after saving
        self._numpy_cache = None
    
    def process_in_chunks(self, chunk_size: int = 1024) -> Iterator[np.ndarray]:
        """Process audio in chunks to reduce memory usage."""
        samples = self.get_samples()
        for i in range(0, len(samples), chunk_size):
            yield samples[i:i + chunk_size]
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self._numpy_cache = None

# Usage
pipeline = VoirsPipeline()
audio = pipeline.synthesize("Long text for audio processing...")

# Wrap for efficient processing
efficient_audio = MemoryEfficientAudio(audio)

# Process in chunks
for chunk in efficient_audio.process_in_chunks(chunk_size=2048):
    # Process each chunk separately
    processed_chunk = chunk * 0.8  # Example processing
    # Chunk goes out of scope automatically

efficient_audio.save("output.wav")
```

### Audio Format Optimization

```python
from voirs_ffi import VoirsPipeline

def optimize_audio_format(audio: PyAudioBuffer, target_size_mb: float = 1.0) -> PyAudioBuffer:
    """Optimize audio format for target memory size."""
    
    current_size_mb = len(audio.samples) * 4 / 1024 / 1024  # 4 bytes per float
    print(f"Current audio size: {current_size_mb:.2f} MB")
    
    if current_size_mb <= target_size_mb:
        return audio
    
    # Calculate compression needed
    compression_ratio = target_size_mb / current_size_mb
    
    # Option 1: Reduce sample rate
    if compression_ratio > 0.5:
        new_sample_rate = int(audio.sample_rate * compression_ratio)
        new_sample_rate = min(new_sample_rate, 22050)  # Don't go too low
        optimized = audio.resample(new_sample_rate)
        print(f"Reduced sample rate to {new_sample_rate} Hz")
        return optimized
    
    # Option 2: Convert to mono
    if audio.channels > 1:
        optimized = audio.to_mono()
        print("Converted to mono")
        return optimized
    
    # Option 3: Accept current size
    print("Cannot optimize further without quality loss")
    return audio

# Usage
pipeline = VoirsPipeline()
audio = pipeline.synthesize("Very long text that produces large audio file...")

# Optimize for memory usage
optimized_audio = optimize_audio_format(audio, target_size_mb=0.5)
optimized_audio.save("optimized_output.wav")
```

## Configuration for Memory Efficiency

### Memory-Constrained Configuration

```python
from voirs_ffi import SynthesisConfig, VoirsPipeline
import psutil

def create_memory_efficient_config() -> SynthesisConfig:
    """Create configuration optimized for memory usage."""
    
    # Get available memory
    available_mb = psutil.virtual_memory().available / 1024 / 1024
    print(f"Available memory: {available_mb:.0f} MB")
    
    if available_mb < 1024:  # Less than 1GB
        return SynthesisConfig(
            sample_rate=16000,      # Lower sample rate
            quality="low",          # Faster processing, less memory
            use_gpu=False,          # CPU often uses less memory
            num_threads=1,          # Reduce parallelism
            memory_limit_mb=64,     # Strict memory limit
            cache_size=5,           # Minimal caching
            enable_preprocessing=False,    # Skip to save memory
            enable_postprocessing=False    # Skip to save memory
        )
    
    elif available_mb < 4096:  # Less than 4GB
        return SynthesisConfig(
            sample_rate=22050,
            quality="medium",
            use_gpu=False,
            num_threads=2,
            memory_limit_mb=256,
            cache_size=25,
            enable_preprocessing=True,
            enable_postprocessing=False
        )
    
    else:  # 4GB or more
        return SynthesisConfig(
            sample_rate=22050,
            quality="medium",
            use_gpu=True,          # GPU can be more efficient
            num_threads=4,
            memory_limit_mb=512,
            cache_size=50,
            enable_preprocessing=True,
            enable_postprocessing=True
        )

# Usage
config = create_memory_efficient_config()
pipeline = VoirsPipeline(config)
```

### Dynamic Memory Management

```python
import psutil
import time
from voirs_ffi import VoirsPipeline, SynthesisConfig

class AdaptiveMemoryTTS:
    """TTS that adapts to memory pressure."""
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.last_check = 0
        self.check_interval = 10  # seconds
        self._initialize()
    
    def _initialize(self):
        """Initialize with current optimal configuration."""
        self.config = self._get_optimal_config()
        self.pipeline = VoirsPipeline(self.config)
    
    def _get_optimal_config(self) -> SynthesisConfig:
        """Get configuration based on current memory availability."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > 90:  # Critical memory pressure
            return SynthesisConfig(
                quality="low",
                use_gpu=False,
                num_threads=1,
                memory_limit_mb=32,
                cache_size=1
            )
        elif memory_percent > 75:  # High memory pressure
            return SynthesisConfig(
                quality="low",
                use_gpu=False,
                num_threads=2,
                memory_limit_mb=64,
                cache_size=5
            )
        elif memory_percent > 50:  # Moderate pressure
            return SynthesisConfig(
                quality="medium",
                use_gpu=True,
                num_threads=2,
                memory_limit_mb=128,
                cache_size=15
            )
        else:  # Low pressure
            return SynthesisConfig(
                quality="medium",
                use_gpu=True,
                num_threads=4,
                memory_limit_mb=256,
                cache_size=30
            )
    
    def _check_memory_pressure(self):
        """Check if we need to adapt to memory pressure."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return
        
        self.last_check = current_time
        new_config = self._get_optimal_config()
        
        # Check if configuration needs updating
        if (new_config.quality != self.config.quality or 
            new_config.memory_limit_mb != self.config.memory_limit_mb):
            
            print(f"Adapting to memory pressure: {psutil.virtual_memory().percent:.1f}%")
            print(f"Switching to quality: {new_config.quality}")
            
            # Update configuration
            self.config = new_config
            self.pipeline = VoirsPipeline(self.config)
    
    def synthesize(self, text: str) -> PyAudioBuffer:
        """Synthesize with memory pressure adaptation."""
        self._check_memory_pressure()
        return self.pipeline.synthesize(text)

# Usage
adaptive_tts = AdaptiveMemoryTTS()

# This will adapt automatically to memory pressure
for i in range(100):
    audio = adaptive_tts.synthesize(f"Text number {i}")
    # Simulate some other memory-intensive work
    if i % 10 == 0:
        large_data = [0] * 1000000  # Simulate memory pressure
        time.sleep(1)
        del large_data
```

## Memory Leak Prevention

### Common Leak Patterns

```python
from voirs_ffi import VoirsPipeline
import weakref

# BAD: Creating pipelines in loops without cleanup
def bad_pattern():
    """Example of memory leak pattern."""
    pipelines = []
    for i in range(100):
        pipeline = VoirsPipeline()
        pipelines.append(pipeline)
        # Pipelines accumulate in memory!
    return pipelines

# GOOD: Proper cleanup
def good_pattern():
    """Memory-safe pattern."""
    results = []
    pipeline = VoirsPipeline()  # Reuse single pipeline
    
    for i in range(100):
        audio = pipeline.synthesize(f"Text {i}")
        results.append(audio.save(f"output_{i}.wav"))
        del audio  # Explicit cleanup
    
    return results

# BETTER: Context management
@contextlib.contextmanager
def managed_pipeline():
    """Context manager for automatic cleanup."""
    pipeline = VoirsPipeline()
    try:
        yield pipeline
    finally:
        pipeline.reset()  # Clear caches
        del pipeline

def best_pattern():
    """Best practice with context management."""
    results = []
    
    with managed_pipeline() as pipeline:
        for i in range(100):
            audio = pipeline.synthesize(f"Text {i}")
            results.append(audio.save(f"output_{i}.wav"))
            del audio
    
    return results
```

### Leak Detection

```python
import weakref
import gc
from voirs_ffi import VoirsPipeline

class LeakDetector:
    """Detect memory leaks in VoiRS objects."""
    
    def __init__(self):
        self.tracked_objects = []
    
    def track_pipeline(self, pipeline: VoirsPipeline) -> VoirsPipeline:
        """Track a pipeline for leak detection."""
        weak_ref = weakref.ref(pipeline, self._object_deleted)
        self.tracked_objects.append(weak_ref)
        return pipeline
    
    def _object_deleted(self, weak_ref):
        """Callback when tracked object is deleted."""
        print("Pipeline properly cleaned up")
        self.tracked_objects.remove(weak_ref)
    
    def check_leaks(self):
        """Check for potential leaks."""
        # Force garbage collection
        gc.collect()
        
        alive_objects = [ref for ref in self.tracked_objects if ref() is not None]
        if alive_objects:
            print(f"Warning: {len(alive_objects)} pipelines still alive")
            for ref in alive_objects:
                obj = ref()
                if obj:
                    print(f"  Alive object: {type(obj)} at {id(obj)}")
        else:
            print("No leaks detected")

# Usage
detector = LeakDetector()

def test_function():
    pipeline = VoirsPipeline()
    detector.track_pipeline(pipeline)
    
    audio = pipeline.synthesize("Test")
    # ... use audio
    
    # pipeline goes out of scope here

test_function()
detector.check_leaks()  # Should show proper cleanup
```

### Memory Pool Pattern

```python
from typing import List, Optional
from queue import Queue
import threading
from voirs_ffi import VoirsPipeline, PyAudioBuffer

class PipelinePool:
    """Pool of reusable pipelines to prevent repeated allocation."""
    
    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size
        self.available = Queue(maxsize=pool_size)
        self.in_use = set()
        self.lock = threading.Lock()
        
        # Pre-create pipelines
        for _ in range(pool_size):
            pipeline = VoirsPipeline()
            self.available.put(pipeline)
    
    def acquire(self) -> VoirsPipeline:
        """Get a pipeline from the pool."""
        try:
            pipeline = self.available.get_nowait()
            with self.lock:
                self.in_use.add(pipeline)
            return pipeline
        except:
            # Pool exhausted, create temporary pipeline
            print("Pipeline pool exhausted, creating temporary pipeline")
            return VoirsPipeline()
    
    def release(self, pipeline: VoirsPipeline):
        """Return a pipeline to the pool."""
        with self.lock:
            if pipeline in self.in_use:
                self.in_use.remove(pipeline)
                # Clear any caches before returning to pool
                pipeline.reset()
                self.available.put(pipeline)
    
    @contextlib.contextmanager
    def get_pipeline(self):
        """Context manager for automatic acquire/release."""
        pipeline = self.acquire()
        try:
            yield pipeline
        finally:
            self.release(pipeline)
    
    def cleanup(self):
        """Clean up all pipelines."""
        with self.lock:
            # Clear in-use pipelines
            for pipeline in self.in_use:
                del pipeline
            self.in_use.clear()
            
            # Clear available pipelines
            while not self.available.empty():
                pipeline = self.available.get()
                del pipeline

# Usage
pipeline_pool = PipelinePool(pool_size=4)

def process_texts(texts: List[str]) -> List[str]:
    """Process texts using pipeline pool."""
    results = []
    
    for text in texts:
        with pipeline_pool.get_pipeline() as pipeline:
            audio = pipeline.synthesize(text)
            filename = f"output_{hash(text)}.wav"
            audio.save(filename)
            results.append(filename)
            del audio  # Explicit cleanup
    
    return results

# Process many texts efficiently
texts = ["Text " + str(i) for i in range(100)]
output_files = process_texts(texts)

# Cleanup pool when done
pipeline_pool.cleanup()
```

## Best Practices Summary

### Memory Management Guidelines

1. **Reuse Objects**: Create pipelines once and reuse them
2. **Explicit Cleanup**: Use `del` for large objects when done
3. **Monitor Usage**: Track memory usage in production
4. **Configure Limits**: Set appropriate memory limits in configuration
5. **Batch Processing**: Process data in manageable chunks
6. **Cache Management**: Periodically clear caches with `reset()`
7. **Garbage Collection**: Force GC when needed with `gc.collect()`

### Performance vs Memory Trade-offs

```python
from voirs_ffi import SynthesisConfig

# Memory-optimized (slower)
memory_config = SynthesisConfig(
    quality="low",
    cache_size=5,
    memory_limit_mb=64,
    enable_postprocessing=False
)

# Performance-optimized (more memory)
performance_config = SynthesisConfig(
    quality="high",
    cache_size=100,
    memory_limit_mb=1024,
    enable_postprocessing=True,
    use_gpu=True
)

# Balanced approach
balanced_config = SynthesisConfig(
    quality="medium",
    cache_size=25,
    memory_limit_mb=256,
    enable_postprocessing=True
)
```

### Production Deployment

```python
import os
import psutil
from voirs_ffi import SynthesisConfig, VoirsPipeline

class ProductionTTS:
    """Production-ready TTS with memory management."""
    
    def __init__(self):
        self.config = self._create_production_config()
        self.pipeline = VoirsPipeline(self.config)
        self.synthesis_count = 0
        self.cleanup_interval = int(os.getenv('TTS_CLEANUP_INTERVAL', '100'))
    
    def _create_production_config(self) -> SynthesisConfig:
        """Create configuration suitable for production."""
        
        # Get environment-specific settings
        memory_limit = int(os.getenv('TTS_MEMORY_LIMIT_MB', '512'))
        quality = os.getenv('TTS_QUALITY', 'medium')
        use_gpu = os.getenv('TTS_USE_GPU', 'true').lower() == 'true'
        
        # Check actual system resources
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        memory_limit = min(memory_limit, available_memory * 0.1)  # Use max 10% of available
        
        return SynthesisConfig(
            quality=quality,
            use_gpu=use_gpu and psutil.virtual_memory().total > 4 * 1024**3,  # Only if >4GB RAM
            memory_limit_mb=int(memory_limit),
            cache_size=max(10, min(50, int(memory_limit / 10))),  # Scale cache with memory
            num_threads=min(4, os.cpu_count() or 1)
        )
    
    def synthesize(self, text: str) -> PyAudioBuffer:
        """Production synthesis with automatic memory management."""
        
        # Periodic cleanup
        if self.synthesis_count >= self.cleanup_interval:
            self._cleanup()
            self.synthesis_count = 0
        
        # Check memory pressure
        if psutil.virtual_memory().percent > 85:
            self._emergency_cleanup()
        
        audio = self.pipeline.synthesize(text)
        self.synthesis_count += 1
        
        return audio
    
    def _cleanup(self):
        """Regular maintenance cleanup."""
        print("Performing scheduled cleanup...")
        self.pipeline.reset()
        import gc
        gc.collect()
    
    def _emergency_cleanup(self):
        """Emergency cleanup during high memory pressure."""
        print("Emergency cleanup due to memory pressure")
        self._cleanup()
        
        # Could also restart pipeline with lower memory config
        if psutil.virtual_memory().percent > 90:
            print("Switching to emergency low-memory configuration")
            emergency_config = SynthesisConfig(
                quality="low",
                use_gpu=False,
                memory_limit_mb=32,
                cache_size=1
            )
            self.pipeline = VoirsPipeline(emergency_config)

# Deployment
if __name__ == "__main__":
    tts = ProductionTTS()
    
    # Example production usage
    import time
    for i in range(1000):
        audio = tts.synthesize(f"Production text {i}")
        # Process audio...
        del audio
        
        if i % 100 == 0:
            memory_percent = psutil.virtual_memory().percent
            print(f"Processed {i} texts, memory usage: {memory_percent:.1f}%")
```

This comprehensive memory management guide provides the foundation for building efficient, scalable VoiRS applications.
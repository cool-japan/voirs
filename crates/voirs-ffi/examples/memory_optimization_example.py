#!/usr/bin/env python3
"""
VoiRS FFI Memory Optimization Example

This example demonstrates advanced memory management techniques
for optimal performance when using VoiRS FFI Python bindings:

1. Memory pool allocation for batch processing
2. Audio buffer reuse to minimize allocations
3. Memory monitoring and profiling
4. Resource cleanup best practices
5. Memory-efficient streaming synthesis

Best for:
- High-throughput applications
- Long-running processes
- Memory-constrained environments
- Performance-critical applications
"""

import gc
import sys
import time
import psutil
import os
from typing import List, Optional

# Import VoiRS FFI if available
try:
    import voirs_ffi
    VOIRS_AVAILABLE = True
    print("VoiRS FFI Memory Optimization Example (Real API)")
    print("===============================================")
except ImportError:
    print("VoiRS FFI Memory Optimization Example (Simulation)")
    print("================================================")
    print("Note: Install VoiRS FFI for real memory optimizations")
    VOIRS_AVAILABLE = False

class MemoryMonitor:
    """Simple memory monitoring utility."""
    
    def __init__(self):
        self.initial_memory = self.get_memory_usage()
        self.measurements = []
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def record(self, label: str = ""):
        """Record current memory usage."""
        current = self.get_memory_usage()
        delta = current - self.initial_memory
        self.measurements.append((label, current, delta))
        print(f"📊 Memory: {current:.1f}MB (+{delta:.1f}MB) - {label}")
    
    def summary(self):
        """Print memory usage summary."""
        print("\n📈 Memory Usage Summary:")
        print("-" * 40)
        for label, total, delta in self.measurements:
            print(f"  {label:30} {total:7.1f}MB (+{delta:5.1f}MB)")
        
        if self.measurements:
            peak = max(measurement[1] for measurement in self.measurements)
            total_delta = peak - self.initial_memory
            print(f"\n  Peak Usage: {peak:.1f}MB (+{total_delta:.1f}MB)")

class AudioBufferPool:
    """Reusable audio buffer pool to minimize allocations."""
    
    def __init__(self, pool_size: int = 10, buffer_size: int = 44100):
        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.available_buffers = []
        self.used_buffers = []
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Pre-allocate buffers for the pool."""
        print(f"🔧 Initializing buffer pool: {self.pool_size} buffers of {self.buffer_size} samples")
        for _ in range(self.pool_size):
            buffer = [0.0] * self.buffer_size
            self.available_buffers.append(buffer)
    
    def get_buffer(self) -> List[float]:
        """Get a buffer from the pool."""
        if self.available_buffers:
            buffer = self.available_buffers.pop()
            self.used_buffers.append(buffer)
            return buffer
        else:
            # Pool exhausted, allocate new buffer
            print("⚠️  Buffer pool exhausted, allocating new buffer")
            buffer = [0.0] * self.buffer_size
            self.used_buffers.append(buffer)
            return buffer
    
    def return_buffer(self, buffer: List[float]):
        """Return a buffer to the pool."""
        if buffer in self.used_buffers:
            self.used_buffers.remove(buffer)
            # Clear buffer data and return to pool
            for i in range(len(buffer)):
                buffer[i] = 0.0
            self.available_buffers.append(buffer)
    
    def cleanup(self):
        """Clean up all buffers."""
        self.available_buffers.clear()
        self.used_buffers.clear()
        gc.collect()
        print("🧹 Buffer pool cleaned up")

def demonstrate_basic_memory_tracking():
    """Demonstrate basic memory usage tracking."""
    print("\n🔍 Basic Memory Tracking Demo")
    print("-" * 35)
    
    monitor = MemoryMonitor()
    monitor.record("Initial state")
    
    if VOIRS_AVAILABLE:
        # Use real VoiRS API
        try:
            pipeline = voirs_ffi.VoirsPipeline()
            monitor.record("Pipeline created")
            
            # Synthesize some text
            texts = [
                "Memory optimization is crucial for performance.",
                "VoiRS provides efficient audio synthesis.",
                "Proper resource management prevents memory leaks."
            ]
            
            for i, text in enumerate(texts):
                audio = pipeline.synthesize(text)
                monitor.record(f"Synthesis {i+1} completed")
                
                # Force garbage collection to see impact
                gc.collect()
                monitor.record(f"After GC {i+1}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Simulate memory usage
        large_data = []
        for i in range(3):
            # Simulate audio data allocation
            data = [0.0] * 44100  # 1 second at 44.1kHz
            large_data.append(data)
            monitor.record(f"Simulated synthesis {i+1}")
            
            gc.collect()
            monitor.record(f"After GC {i+1}")
    
    monitor.summary()

def demonstrate_buffer_pool_optimization():
    """Demonstrate buffer pool optimization for batch processing."""
    print("\n🔄 Buffer Pool Optimization Demo")
    print("-" * 37)
    
    monitor = MemoryMonitor()
    monitor.record("Starting buffer pool demo")
    
    # Create buffer pool
    buffer_pool = AudioBufferPool(pool_size=5, buffer_size=22050)  # 0.5 second buffers
    monitor.record("Buffer pool created")
    
    if VOIRS_AVAILABLE:
        try:
            pipeline = voirs_ffi.VoirsPipeline()
            
            texts = [
                "Buffer pools reduce allocation overhead.",
                "Reusing memory improves performance.",
                "Smart resource management scales better.",
                "Memory optimization matters for production.",
                "VoiRS FFI supports efficient processing."
            ]
            
            print(f"Processing {len(texts)} texts with buffer pool...")
            
            for i, text in enumerate(texts):
                # Get buffer from pool
                buffer = buffer_pool.get_buffer()
                
                # Synthesize (in real implementation, write to buffer)
                audio = pipeline.synthesize(text)
                
                # Simulate copying to our buffer
                if hasattr(audio, 'samples_as_list'):
                    samples = audio.samples_as_list()
                    copy_length = min(len(buffer), len(samples))
                    for j in range(copy_length):
                        buffer[j] = samples[j]
                
                print(f"  ✅ Processed text {i+1} using pooled buffer")
                
                # Return buffer to pool for reuse
                buffer_pool.return_buffer(buffer)
                
                if i == 2:  # Check memory usage mid-way
                    monitor.record("Mid-processing")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Simulate buffer pool usage
        print("Processing 5 texts with buffer pool...")
        for i in range(5):
            buffer = buffer_pool.get_buffer()
            
            # Simulate audio processing
            for j in range(min(1000, len(buffer))):
                buffer[j] = float(i * j) / 1000.0
            
            print(f"  ✅ Processed text {i+1} using pooled buffer")
            buffer_pool.return_buffer(buffer)
            
            if i == 2:
                monitor.record("Mid-processing")
    
    monitor.record("Processing completed")
    buffer_pool.cleanup()
    monitor.record("After cleanup")
    
    monitor.summary()

def demonstrate_streaming_memory_efficiency():
    """Demonstrate memory-efficient streaming synthesis."""
    print("\n🌊 Streaming Memory Efficiency Demo")
    print("-" * 39)
    
    monitor = MemoryMonitor()
    monitor.record("Starting streaming demo")
    
    long_text = """
    This is a very long piece of text that will be processed in a streaming manner
    to demonstrate how memory usage can be kept low even when processing large amounts
    of audio content. The key is to process the audio in chunks rather than loading
    everything into memory at once. This approach is essential for real-time applications
    and when processing very long audio sequences that might not fit in available memory.
    """
    
    # Split text into chunks for streaming
    sentences = [s.strip() for s in long_text.split('.') if s.strip()]
    print(f"Processing {len(sentences)} text chunks in streaming mode...")
    
    if VOIRS_AVAILABLE:
        try:
            pipeline = voirs_ffi.VoirsPipeline()
            monitor.record("Pipeline ready")
            
            total_samples = 0
            chunk_size = 1024  # Process in 1KB audio chunks
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                print(f"  🎵 Processing chunk {i+1}: '{sentence[:40]}...'")
                
                # Synthesize chunk
                audio = pipeline.synthesize(sentence)
                
                # Process in small chunks to keep memory low
                if hasattr(audio, 'samples_as_list'):
                    samples = audio.samples_as_list()
                    total_samples += len(samples)
                    
                    # Simulate streaming processing (chunk by chunk)
                    for start in range(0, len(samples), chunk_size):
                        end = min(start + chunk_size, len(samples))
                        chunk = samples[start:end]
                        
                        # Process chunk (e.g., play, save, transmit)
                        # In real use: audio_output.write(chunk)
                        time.sleep(0.01)  # Simulate processing
                
                # Clean up immediately after processing
                del audio
                gc.collect()
                
                if i % 2 == 0:  # Monitor every other chunk
                    monitor.record(f"After chunk {i+1}")
            
            print(f"  📊 Total samples processed: {total_samples:,}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Simulate streaming processing
        total_samples = 0
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            print(f"  🎵 Processing chunk {i+1}: '{sentence[:40]}...'")
            
            # Simulate audio synthesis
            estimated_duration = len(sentence) * 0.05  # rough estimate
            chunk_samples = int(44100 * estimated_duration)
            total_samples += chunk_samples
            
            # Simulate small memory footprint processing
            for _ in range(0, chunk_samples, 1024):
                time.sleep(0.001)  # Simulate processing
            
            if i % 2 == 0:
                monitor.record(f"After chunk {i+1}")
        
        print(f"  📊 Estimated samples processed: {total_samples:,}")
    
    monitor.record("Streaming completed")
    monitor.summary()

def demonstrate_memory_best_practices():
    """Demonstrate memory management best practices."""
    print("\n📋 Memory Management Best Practices")
    print("-" * 40)
    
    practices = [
        "✅ Use buffer pools for repeated operations",
        "✅ Process audio in chunks for large files", 
        "✅ Call gc.collect() after large operations",
        "✅ Release references to audio data when done",
        "✅ Monitor memory usage in production",
        "✅ Set resource limits for long-running processes",
        "✅ Use streaming when possible to limit peak usage",
        "✅ Clean up pipelines and resources properly"
    ]
    
    for practice in practices:
        print(f"  {practice}")
        time.sleep(0.1)
    
    print("\n💡 Memory Optimization Tips:")
    print("   • Pre-allocate buffers for known workloads")
    print("   • Use weak references for caches when appropriate")
    print("   • Implement backpressure in streaming scenarios")
    print("   • Profile memory usage during development")
    print("   • Consider using memory-mapped files for large datasets")

def main():
    """Main demonstration function."""
    try:
        print("Memory optimization is crucial for production VoiRS applications!")
        print("This example shows techniques to minimize memory usage.\n")
        
        demonstrate_basic_memory_tracking()
        demonstrate_buffer_pool_optimization()
        demonstrate_streaming_memory_efficiency()
        demonstrate_memory_best_practices()
        
        print("\n🎉 Memory optimization demonstrations completed!")
        
        if VOIRS_AVAILABLE:
            print("\n🚀 You can now apply these techniques in your VoiRS applications:")
            print("   • Use buffer pools for batch processing")
            print("   • Implement streaming for large audio content")
            print("   • Monitor memory usage in production")
            print("   • Clean up resources properly")
        else:
            print("\n📝 Install VoiRS FFI to use these optimizations:")
            print("   pip install voirs-ffi")
            print("   Then run this example again for real API demonstrations")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
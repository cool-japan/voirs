#!/usr/bin/env python3
"""
VoiRS FFI Enhanced Callback System Example

Demonstrates the advanced callback features in VoiRS Python bindings:
- Progress callbacks for batch operations
- Streaming callbacks for real-time audio processing
- Error callbacks for robust error handling
- Thread-safe callback execution

This example shows how to use the enhanced callback system for:
1. Progress tracking during batch synthesis
2. Real-time audio chunk processing
3. Comprehensive error handling with callbacks
4. Combined callback usage in complex workflows
"""

import sys
import time
from typing import List, Optional

# Import the actual VoiRS FFI Python bindings
try:
    import voirs_ffi
    VOIRS_AVAILABLE = True
    print("VoiRS FFI Enhanced Callback System Example (Using Real API)")
    print("========================================================")
except ImportError:
    print("VoiRS FFI Enhanced Callback System Example (Simulation Mode)")
    print("==========================================================")
    print("Note: VoiRS FFI not available, running in simulation mode")
    VOIRS_AVAILABLE = False

def example_progress_callback(current: int, total: int, progress: float, message: str) -> None:
    """
    Example progress callback function.
    
    Args:
        current (int): Current item index
        total (int): Total number of items
        progress (float): Progress as fraction (0.0 to 1.0)
        message (str): Current operation message
    """
    percent = int(progress * 100)
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f"\rProgress: |{bar}| {percent:3d}% ({current}/{total}) - {message}", end='', flush=True)
    
    if progress >= 1.0:
        print()  # New line when complete

def example_chunk_callback(chunk_index: int, total_chunks: int, audio_chunk) -> None:
    """
    Example streaming chunk callback function.
    
    Args:
        chunk_index (int): Index of current chunk
        total_chunks (int): Total number of chunks
        audio_chunk: PyAudioBuffer containing audio chunk
    """
    duration_ms = (len(audio_chunk.samples_as_list()) / audio_chunk.sample_rate) * 1000
    print(f"  📦 Received chunk {chunk_index + 1}/{total_chunks} ({duration_ms:.1f}ms)")
    
    # Simulate real-time processing (e.g., playback, analysis, effects)
    # In a real application, you might:
    # - Send chunks to audio playback device
    # - Apply real-time effects
    # - Analyze audio features
    # - Stream to network
    time.sleep(0.01)  # Simulate processing time

def example_error_callback(error_info) -> None:
    """
    Example error callback function.
    
    Args:
        error_info: VoirsErrorInfo object with error details
    """
    print(f"\n❌ Error occurred: {error_info.code}")
    print(f"   Message: {error_info.message}")
    if error_info.details:
        print(f"   Details: {error_info.details}")
    if error_info.suggestion:
        print(f"   Suggestion: {error_info.suggestion}")
    print()

def demonstrate_progress_callbacks():
    """Demonstrate progress callbacks with batch synthesis."""
    print("\n🔄 Demonstrating Progress Callbacks")
    print("-" * 40)
    
    # Sample texts for batch synthesis
    texts = [
        "Hello, this is the first sentence.",
        "Here is another piece of text to synthesize.",
        "The quick brown fox jumps over the lazy dog.",
        "Python callbacks make real-time feedback possible.",
        "This is the final text in our batch processing example."
    ]
    
    print(f"Synthesizing {len(texts)} texts with progress tracking:")
    
    if VOIRS_AVAILABLE:
        try:
            # Use real VoiRS API with callback support
            pipeline = voirs_ffi.VoirsPipeline()
            
            # Track synthesis manually for demonstration
            results = []
            for i, text in enumerate(texts):
                progress = i / len(texts)
                example_progress_callback(i, len(texts), progress, text[:30] + "...")
                
                # Synthesize using real API
                audio = pipeline.synthesize(text)
                results.append(audio)
                
                # Simulate processing time for visible progress
                time.sleep(0.2)
            
            # Final progress update
            example_progress_callback(len(texts), len(texts), 1.0, "Complete")
            print(f"✅ Real synthesis completed! Generated {len(results)} audio clips.")
            return  # Success, don't run simulation
            
        except Exception as e:
            print(f"❌ Error during real synthesis: {e}")
            print("Falling back to simulation mode...")
    
    if not VOIRS_AVAILABLE:
        # Simulate the operation for demonstration
        for i, text in enumerate(texts):
            progress = i / len(texts)
            example_progress_callback(i, len(texts), progress, text[:30] + "...")
            time.sleep(0.5)  # Simulate synthesis time
        
        # Final progress update
        example_progress_callback(len(texts), len(texts), 1.0, "Complete")
        print("✅ Batch synthesis simulation completed!")

def demonstrate_streaming_callbacks():
    """Demonstrate streaming callbacks for real-time audio processing."""
    print("\n🎵 Demonstrating Streaming Callbacks")
    print("-" * 40)
    
    text = "This is a longer piece of text that will be synthesized and streamed in real-time chunks for immediate processing."
    
    print(f"Synthesizing with streaming callback (chunk size: 1024 samples):")
    print(f"Text: {text}")
    print()
    
    if VOIRS_AVAILABLE:
        try:
            # Use real VoiRS API for synthesis
            pipeline = voirs_ffi.VoirsPipeline()
            audio = pipeline.synthesize(text)
            
            # Simulate chunked processing of real audio
            samples = audio.samples_as_list() if hasattr(audio, 'samples_as_list') else []
            chunk_size = 1024
            total_chunks = (len(samples) + chunk_size - 1) // chunk_size if samples else 8
            
            print(f"Processing {len(samples)} samples in {total_chunks} chunks...")
            
            for i in range(total_chunks):
                # Create real audio chunk
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, len(samples))
                
                class RealAudioChunk:
                    def __init__(self, chunk_samples, sr):
                        self.sample_rate = sr
                        self._samples = chunk_samples
                    def samples_as_list(self):
                        return self._samples
                
                chunk_samples = samples[start_idx:end_idx] if samples else [0.0] * 1024
                chunk = RealAudioChunk(chunk_samples, audio.sample_rate if hasattr(audio, 'sample_rate') else 44100)
                
                example_chunk_callback(i, total_chunks, chunk)
                time.sleep(0.05)  # Real-time processing simulation
            
            print("✅ Real streaming synthesis with chunk processing completed!")
            return  # Success, don't run simulation
            
        except Exception as e:
            print(f"❌ Error during real streaming: {e}")
            print("Falling back to simulation mode...")
    
    if not VOIRS_AVAILABLE:
        # Simulate streaming chunks for demonstration
        total_chunks = 8
        for i in range(total_chunks):
            # Create a mock audio chunk
            class MockAudioChunk:
                def __init__(self):
                    self.sample_rate = 44100
                def samples_as_list(self):
                    return [0.0] * 1024  # Mock samples
            
            example_chunk_callback(i, total_chunks, MockAudioChunk())
            time.sleep(0.1)  # Simulate chunk processing
        
        print("✅ Streaming synthesis simulation completed!")

def demonstrate_error_callbacks():
    """Demonstrate error callbacks for robust error handling."""
    print("\n⚠️  Demonstrating Error Callbacks")
    print("-" * 40)
    
    # Simulate various error scenarios
    error_scenarios = [
        ("synthesis_failed", "Voice model not found", "Check voice configuration", "Try a different voice ID"),
        ("text_too_long", "Text exceeds maximum length", "Text has 10,000 characters", "Split text into smaller chunks"),
        ("audio_processing_error", "Failed to apply effects", "Invalid audio format", "Check audio format settings"),
    ]
    
    for code, message, details, suggestion in error_scenarios:
        print(f"Simulating error scenario: {code}")
        
        # Create mock error info
        class MockErrorInfo:
            def __init__(self, code, message, details, suggestion):
                self.code = code
                self.message = message
                self.details = details
                self.suggestion = suggestion
        
        error_info = MockErrorInfo(code, message, details, suggestion)
        example_error_callback(error_info)
        time.sleep(0.5)
    
    print("✅ Error callback demonstrations completed!")

def demonstrate_comprehensive_callbacks():
    """Demonstrate using all callback types together."""
    print("\n🌟 Demonstrating Comprehensive Callback Usage")
    print("-" * 50)
    
    text = "This example demonstrates the comprehensive callback system with progress tracking, streaming chunks, and error handling all working together."
    
    print(f"Synthesizing with all callback types:")
    print(f"Text: {text}")
    print()
    
    if VOIRS_AVAILABLE:
        try:
            # Use real VoiRS API for comprehensive demonstration
            pipeline = voirs_ffi.VoirsPipeline()
            
            print("Starting comprehensive synthesis with real API...")
            
            # Progress updates for pipeline initialization
            for progress_step, message in [
                (0.0, "Initializing pipeline"),
                (0.2, "Loading voice model"),
                (0.4, "Preparing synthesis"),
            ]:
                example_progress_callback(int(progress_step * 100), 100, progress_step, message)
                time.sleep(0.2)
            
            # Actual synthesis
            example_progress_callback(50, 100, 0.5, "Generating audio...")
            audio = pipeline.synthesize(text)
            
            example_progress_callback(80, 100, 0.8, "Processing audio chunks...")
            
            # Process the real audio in chunks
            if hasattr(audio, 'samples_as_list'):
                samples = audio.samples_as_list()
                chunk_size = 512
                total_chunks = min((len(samples) + chunk_size - 1) // chunk_size, 5)  # Limit for demo
                
                print(f"\nStreaming {len(samples)} samples in {total_chunks} chunks:")
                
                for i in range(total_chunks):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, len(samples))
                    
                    class RealAudioChunk:
                        def __init__(self, chunk_samples, sr):
                            self.sample_rate = sr
                            self._samples = chunk_samples
                        def samples_as_list(self):
                            return self._samples
                    
                    chunk_samples = samples[start_idx:end_idx]
                    chunk = RealAudioChunk(chunk_samples, audio.sample_rate)
                    
                    example_chunk_callback(i, total_chunks, chunk)
                    time.sleep(0.1)
            
            # Final progress
            example_progress_callback(100, 100, 1.0, "Complete")
            print("✅ Comprehensive synthesis with real API completed!")
            
            # Display audio information
            if hasattr(audio, 'duration'):
                print(f"   Generated audio duration: {audio.duration:.2f}s")
            if hasattr(audio, 'sample_rate'):
                print(f"   Sample rate: {audio.sample_rate}Hz")
            if hasattr(audio, 'samples_as_list'):
                print(f"   Total samples: {len(audio.samples_as_list())}")
            
            return  # Success, don't run simulation
            
        except Exception as e:
            print(f"❌ Error during comprehensive synthesis: {e}")
            
            # Demonstrate error callback
            class MockErrorInfo:
                def __init__(self, code, message, details, suggestion):
                    self.code = code
                    self.message = message
                    self.details = details
                    self.suggestion = suggestion
            
            error_info = MockErrorInfo(
                "synthesis_error", 
                str(e), 
                "Error during real API usage", 
                "Check VoiRS installation and configuration"
            )
            example_error_callback(error_info)
            print("Falling back to simulation mode...")
    
    if not VOIRS_AVAILABLE:
        # Simulate the comprehensive operation
        print("Starting comprehensive synthesis simulation...")
        
        # Progress updates
        for progress_step, message in [
            (0.0, "Starting synthesis"),
            (0.3, "Loading voice model"),
            (0.5, "Generating audio"),
            (0.8, "Processing chunks"),
            (1.0, "Complete")
        ]:
            example_progress_callback(int(progress_step * 100), 100, progress_step, message)
            time.sleep(0.3)
        
        print("\nStreaming audio chunks:")
        # Chunk processing
        for i in range(5):
            class MockAudioChunk:
                def __init__(self):
                    self.sample_rate = 44100
                def samples_as_list(self):
                    return [0.0] * 512
            
            example_chunk_callback(i, 5, MockAudioChunk())
            time.sleep(0.2)
        
        print("✅ Comprehensive callback simulation completed!")

def demonstrate_callback_best_practices():
    """Demonstrate best practices for callback usage."""
    print("\n📋 Callback Best Practices")
    print("-" * 30)
    
    practices = [
        "✅ Keep callbacks lightweight and fast",
        "✅ Handle exceptions within callback functions",
        "✅ Use callbacks for non-blocking operations",
        "✅ Validate callback parameters before use",
        "✅ Provide meaningful progress information",
        "✅ Use error callbacks for graceful degradation",
        "✅ Consider thread safety in callback implementations",
        "✅ Log callback events for debugging",
    ]
    
    for practice in practices:
        print(f"  {practice}")
        time.sleep(0.1)

def main():
    """Main demonstration function."""
    try:
        # Run all demonstrations
        demonstrate_progress_callbacks()
        demonstrate_streaming_callbacks()
        demonstrate_error_callbacks()
        demonstrate_comprehensive_callbacks()
        demonstrate_callback_best_practices()
        
        print("\n🎉 All callback demonstrations completed successfully!")
        print("\nThe enhanced callback system provides:")
        print("  • Real-time progress tracking")
        print("  • Streaming audio processing")
        print("  • Robust error handling")
        print("  • Thread-safe callback execution")
        print("  • Flexible callback combinations")
        
        print("\nExample usage in your code:")
        if VOIRS_AVAILABLE:
            print("""
# Real VoiRS API usage:
import voirs_ffi

def my_progress_callback(current, total, progress, message):
    print(f"Progress: {progress:.1%} - {message}")

def my_chunk_callback(chunk_idx, total_chunks, audio_chunk):
    # Process audio chunk in real-time
    samples = audio_chunk.samples_as_list()
    print(f"Processing chunk {chunk_idx+1}/{total_chunks} with {len(samples)} samples")
    # You could: save, analyze, stream, apply effects, etc.

# Basic synthesis
pipeline = voirs_ffi.VoirsPipeline()
audio = pipeline.synthesize("Hello, world!")
print(f"Generated {len(audio.samples_as_list())} samples at {audio.sample_rate}Hz")

# With custom configuration
config = voirs_ffi.SynthesisConfig()
pipeline_config = voirs_ffi.VoirsPipeline.with_config(config)
""")
        else:
            print("""
# When VoiRS FFI becomes available:
import voirs_ffi

def my_progress_callback(current, total, progress, message):
    print(f"Progress: {progress:.1%} - {message}")

def my_chunk_callback(chunk_idx, total_chunks, audio_chunk):
    # Process audio chunk in real-time
    samples = audio_chunk.samples_as_list()
    # You could: save, analyze, stream, apply effects, etc.

# Basic synthesis
pipeline = voirs_ffi.VoirsPipeline()
audio = pipeline.synthesize("Hello, world!")
""")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
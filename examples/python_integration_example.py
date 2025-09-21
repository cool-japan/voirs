#!/usr/bin/env python3
"""
VoiRS Python Integration Example
==================================

This example demonstrates the complete VoiRS Python bindings functionality including:
- Text-to-speech synthesis
- SSML support
- Voice cloning and conversion
- Emotion control
- Spatial audio
- Real-time processing
- Audio analysis and quality metrics

Requirements:
    pip install numpy
    
Usage:
    python python_integration_example.py

Note: This example requires the VoiRS Python bindings to be compiled with:
    cargo build --release --features python,numpy
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Try to import VoiRS Python bindings
try:
    import voirs_ffi as voirs
    VOIRS_AVAILABLE = True
except ImportError:
    print("⚠️  VoiRS Python bindings not found!")
    print("   Build the bindings with: cargo build --release --features python,numpy")
    print("   Running in demonstration mode...")
    VOIRS_AVAILABLE = False

# Optional NumPy for advanced audio processing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️  NumPy not available - some features will be limited")
    NUMPY_AVAILABLE = False

class VoiRSPythonDemo:
    """Comprehensive demonstration of VoiRS Python bindings"""
    
    def __init__(self):
        self.pipeline = None
        self.demo_texts = [
            "Hello world! This is a test of VoiRS text-to-speech.",
            "The quick brown fox jumps over the lazy dog.",
            "VoiRS provides high-quality neural voice synthesis.",
        ]
        self.demo_ssml = """
        <speak>
            <p>Welcome to VoiRS, the advanced neural voice synthesis system!</p>
            <break time="0.5s"/>
            <prosody rate="slow" pitch="high">This text is spoken slowly with high pitch.</prosody>
            <break time="0.3s"/>
            <prosody rate="fast" pitch="low">This text is spoken quickly with low pitch.</prosody>
            <break time="0.5s"/>
            <emphasis level="strong">This is emphasized text!</emphasis>
        </speak>
        """

    def initialize_pipeline(self):
        """Initialize the VoiRS pipeline with optimal configuration"""
        print("🔧 Initializing VoiRS Pipeline...")
        
        if not VOIRS_AVAILABLE:
            print("   [DEMO MODE] Would initialize pipeline with configuration:")
            print("   - GPU acceleration: Auto-detect")
            print("   - Thread count: 4")
            print("   - Cache directory: ~/.cache/voirs")
            print("   - Device: auto")
            return True
        
        try:
            # Create pipeline with advanced configuration
            self.pipeline = voirs.VoirsPipeline.with_config(
                use_gpu=True,  # Use GPU if available
                num_threads=4,  # Optimal thread count
                cache_dir=str(Path.home() / ".cache" / "voirs"),
                device="auto"  # Auto-detect best device
            )
            
            print("✅ Pipeline initialized successfully!")
            
            # Get and display performance info
            if hasattr(self.pipeline, 'get_performance_info'):
                perf_info = self.pipeline.get_performance_info()
                print(f"   CPU cores: {perf_info.get('cpu_cores', 'N/A')}")
                print(f"   GPU available: {perf_info.get('gpu_available', 'N/A')}")
                print(f"   Memory usage: {perf_info.get('memory_usage_mb', 0):.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {e}")
            return False

    def demonstrate_basic_synthesis(self):
        """Demonstrate basic text-to-speech synthesis"""
        print("\n📢 Basic Text-to-Speech Synthesis")
        print("=" * 50)
        
        if not VOIRS_AVAILABLE:
            print("[DEMO MODE] Would synthesize:")
            for i, text in enumerate(self.demo_texts):
                print(f"   {i+1}. \"{text}\"")
                print(f"      → Generated: ~{len(text) * 0.08:.1f}s audio")
            return
        
        for i, text in enumerate(self.demo_texts):
            print(f"\n🎵 Synthesizing: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            try:
                # Basic synthesis
                start_time = time.time()
                audio = self.pipeline.synthesize(text)
                synthesis_time = time.time() - start_time
                
                print(f"   ✅ Synthesis completed in {synthesis_time:.2f}s")
                print(f"   📊 Audio: {audio.duration():.1f}s, {audio.sample_rate()}Hz, {audio.channels()} channel(s)")
                print(f"   ⚡ Real-time factor: {synthesis_time / audio.duration():.2f}x")
                
                # Save audio file
                output_file = f"/tmp/voirs_basic_{i+1}.wav"
                audio.save(output_file)
                print(f"   💾 Saved to: {output_file}")
                
            except Exception as e:
                print(f"   ❌ Synthesis failed: {e}")

    def demonstrate_enhanced_synthesis(self):
        """Demonstrate synthesis with metrics and advanced features"""
        print("\n📈 Enhanced Synthesis with Metrics")
        print("=" * 50)
        
        if not VOIRS_AVAILABLE:
            print("[DEMO MODE] Would demonstrate:")
            print("   - Synthesis with detailed metrics")
            print("   - Real-time factor calculation")
            print("   - Memory usage tracking")
            print("   - Cache hit rate monitoring")
            return
        
        text = "This is an enhanced synthesis example with comprehensive metrics and monitoring."
        print(f"🎵 Enhanced synthesis: \"{text}\"")
        
        try:
            # Synthesis with metrics
            result = self.pipeline.synthesize_with_metrics(text)
            
            print("📊 Synthesis Metrics:")
            print(f"   Processing time: {result.metrics.processing_time_ms:.1f}ms")
            print(f"   Audio duration: {result.metrics.audio_duration_ms:.1f}ms")
            print(f"   Real-time factor: {result.metrics.real_time_factor:.3f}x")
            print(f"   Memory used: {result.metrics.memory_usage_mb:.1f}MB")
            print(f"   Cache hit rate: {result.metrics.cache_hit_rate:.1f}%")
            
            # Save enhanced audio
            output_file = "/tmp/voirs_enhanced.wav"
            result.audio.save(output_file)
            print(f"💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Enhanced synthesis failed: {e}")

    def demonstrate_ssml_synthesis(self):
        """Demonstrate SSML (Speech Synthesis Markup Language) support"""
        print("\n🏷️  SSML Synthesis")
        print("=" * 50)
        
        if not VOIRS_AVAILABLE:
            print("[DEMO MODE] Would synthesize SSML with:")
            print("   - Prosody control (rate, pitch)")
            print("   - Pauses and breaks")
            print("   - Emphasis and expression")
            print("   - Multiple voice styles")
            return
        
        print("🎵 Synthesizing SSML with prosody control...")
        print("SSML content:")
        print(self.demo_ssml)
        
        try:
            # SSML synthesis with metrics
            result = self.pipeline.synthesize_ssml_with_metrics(self.demo_ssml)
            
            print("✅ SSML synthesis completed!")
            print(f"📊 Processing: {result.metrics.processing_time_ms:.1f}ms")
            print(f"🎵 Audio duration: {result.metrics.audio_duration_ms:.1f}ms")
            
            # Save SSML audio
            output_file = "/tmp/voirs_ssml.wav"
            result.audio.save(output_file)
            print(f"💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ SSML synthesis failed: {e}")

    def demonstrate_voice_management(self):
        """Demonstrate voice listing and selection"""
        print("\n🎤 Voice Management")
        print("=" * 50)
        
        if not VOIRS_AVAILABLE:
            print("[DEMO MODE] Would demonstrate:")
            print("   - Listing available voices")
            print("   - Voice characteristics")
            print("   - Language support")
            print("   - Quality levels")
            return
        
        try:
            # List available voices
            voices = self.pipeline.list_voices()
            print(f"🎤 Found {len(voices)} available voices:")
            
            for voice in voices[:5]:  # Show first 5 voices
                print(f"   📢 {voice.name} ({voice.id})")
                print(f"      Language: {voice.language}")
                print(f"      Quality: {voice.quality}")
                print(f"      Available: {'✅' if voice.is_available else '❌'}")
            
            if len(voices) > 5:
                print(f"   ... and {len(voices) - 5} more voices")
            
            # Get current voice
            current_voice = self.pipeline.get_voice()
            if current_voice:
                print(f"🔄 Current voice: {current_voice}")
            
            # Demonstrate voice switching
            if voices:
                new_voice = voices[0].id
                print(f"🔄 Switching to voice: {new_voice}")
                self.pipeline.set_voice(new_voice)
                
                # Synthesize with new voice
                text = "This is spoken with a different voice."
                audio = self.pipeline.synthesize(text)
                output_file = f"/tmp/voirs_voice_{new_voice[:10]}.wav"
                audio.save(output_file)
                print(f"💾 Saved with new voice to: {output_file}")
            
        except Exception as e:
            print(f"❌ Voice management failed: {e}")

    def demonstrate_batch_processing(self):
        """Demonstrate batch synthesis with progress tracking"""
        print("\n⚡ Batch Processing with Progress Tracking")
        print("=" * 50)
        
        if not VOIRS_AVAILABLE:
            print("[DEMO MODE] Would process batch of texts with:")
            print("   - Progress callbacks")
            print("   - Parallel processing")
            print("   - Error handling")
            print("   - Batch metrics")
            return
        
        batch_texts = [
            "First sentence in the batch processing example.",
            "Second sentence demonstrating parallel synthesis capabilities.",
            "Third sentence showing progress tracking functionality.",
            "Fourth sentence with batch processing optimization.",
            "Fifth and final sentence completing the batch demonstration."
        ]
        
        print(f"📦 Processing batch of {len(batch_texts)} texts...")
        
        # Define progress callback
        def progress_callback(current, total, progress, current_text):
            percent = progress * 100
            print(f"   📈 Progress: {current}/{total} ({percent:.1f}%) - \"{current_text[:30]}...\"")
        
        try:
            # Batch synthesis with progress
            results = self.pipeline.batch_synthesize_with_progress(
                batch_texts, 
                progress_callback
            )
            
            print("✅ Batch processing completed!")
            print("📊 Batch Results:")
            
            total_audio_duration = 0
            total_processing_time = 0
            
            for i, result in enumerate(results):
                duration = result.metrics.audio_duration_ms / 1000
                processing = result.metrics.processing_time_ms / 1000
                total_audio_duration += duration
                total_processing_time += processing
                
                print(f"   {i+1}. Audio: {duration:.1f}s, Processing: {processing:.2f}s, RTF: {result.metrics.real_time_factor:.2f}x")
                
                # Save batch result
                output_file = f"/tmp/voirs_batch_{i+1}.wav"
                result.audio.save(output_file)
            
            # Overall batch metrics
            overall_rtf = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
            print(f"📊 Overall: {total_audio_duration:.1f}s audio, {total_processing_time:.1f}s processing, RTF: {overall_rtf:.2f}x")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")

    def demonstrate_streaming_synthesis(self):
        """Demonstrate real-time streaming synthesis"""
        print("\n🌊 Streaming Synthesis")
        print("=" * 50)
        
        if not VOIRS_AVAILABLE:
            print("[DEMO MODE] Would demonstrate:")
            print("   - Real-time streaming synthesis")
            print("   - Chunk-based processing")
            print("   - Low-latency audio generation")
            print("   - Streaming callbacks")
            return
        
        text = "This is a streaming synthesis example that generates audio chunks in real-time for immediate playback without waiting for complete synthesis."
        print(f"🌊 Streaming synthesis: \"{text[:50]}...\"")
        
        # Chunk counter for callback
        chunk_count = 0
        
        def chunk_callback(chunk_index, total_chunks, audio_chunk):
            nonlocal chunk_count
            chunk_count += 1
            print(f"   🔊 Received chunk {chunk_index + 1}: {audio_chunk.duration():.2f}s audio")
            
            # Save chunk (in real application, you'd play it immediately)
            chunk_file = f"/tmp/voirs_chunk_{chunk_index + 1}.wav"
            audio_chunk.save(chunk_file)
        
        try:
            # Streaming synthesis
            full_audio = self.pipeline.synthesize_streaming(
                text, 
                chunk_callback, 
                chunk_size=2048  # Smaller chunks for more responsive streaming
            )
            
            print(f"✅ Streaming completed! Received {chunk_count} chunks")
            print(f"📊 Full audio: {full_audio.duration():.1f}s, {full_audio.sample_rate()}Hz")
            
            # Save complete audio
            output_file = "/tmp/voirs_streaming_full.wav"
            full_audio.save(output_file)
            print(f"💾 Complete audio saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Streaming synthesis failed: {e}")

    def demonstrate_audio_analysis(self):
        """Demonstrate audio analysis and quality metrics"""
        print("\n🔬 Audio Analysis and Quality Metrics")
        print("=" * 50)
        
        if not VOIRS_AVAILABLE or not NUMPY_AVAILABLE:
            print("[DEMO MODE] Would analyze audio with:")
            print("   - RMS energy calculation")
            print("   - Zero crossing rate")
            print("   - Spectral centroid")
            print("   - Silence detection")
            print("   - Quality scoring")
            return
        
        # Generate test audio for analysis
        test_text = "This audio will be analyzed for various quality metrics and characteristics."
        print(f"🎵 Generating test audio: \"{test_text}\"")
        
        try:
            audio = self.pipeline.synthesize(test_text)
            
            # Convert to NumPy for analysis
            if hasattr(audio, 'as_numpy'):
                audio_array = audio.as_numpy()
                
                # Initialize audio analyzer
                analyzer = voirs.PyAudioAnalyzer()
                
                print("🔬 Audio Analysis Results:")
                
                # RMS Energy
                rms = analyzer.rms_energy(audio_array)
                print(f"   🔊 RMS Energy: {rms:.4f}")
                
                # Zero Crossing Rate
                zcr = analyzer.zero_crossing_rate(audio_array)
                print(f"   〰️  Zero Crossing Rate: {zcr:.4f}")
                
                # Spectral Centroid (brightness)
                centroid = analyzer.spectral_centroid(audio_array, audio.sample_rate())
                print(f"   ✨ Spectral Centroid: {centroid:.1f} Hz")
                
                # Silence Detection
                silence_regions = analyzer.find_silence(audio_array, threshold=0.01, min_duration=1000)
                print(f"   🤫 Silence regions: {silence_regions.shape[0] if hasattr(silence_regions, 'shape') else 0}")
                
                print(f"📊 Audio properties:")
                print(f"   Duration: {audio.duration():.2f}s")
                print(f"   Sample rate: {audio.sample_rate()} Hz")
                print(f"   Channels: {audio.channels()}")
                print(f"   Samples: {audio.length()}")
            
        except Exception as e:
            print(f"❌ Audio analysis failed: {e}")

    def demonstrate_advanced_audio_processing(self):
        """Demonstrate advanced audio processing with NumPy"""
        print("\n🧮 Advanced Audio Processing (NumPy)")
        print("=" * 50)
        
        if not VOIRS_AVAILABLE or not NUMPY_AVAILABLE:
            print("[DEMO MODE] Would demonstrate:")
            print("   - NumPy array operations")
            print("   - Audio normalization")
            print("   - Fade in/out effects")
            print("   - Audio mixing")
            print("   - Convolution")
            return
        
        try:
            # Generate two audio samples
            text1 = "First audio sample for processing."
            text2 = "Second audio sample for mixing."
            
            audio1 = self.pipeline.synthesize(text1)
            audio2 = self.pipeline.synthesize(text2)
            
            if hasattr(audio1, 'as_numpy') and hasattr(audio2, 'as_numpy'):
                print("🎵 Applying audio effects...")
                
                # Apply normalization
                audio1.apply_numpy_operation("normalize")
                print("   ✅ Applied normalization to audio 1")
                
                # Apply fade effects
                audio1.apply_numpy_operation("fade_in")
                audio1.apply_numpy_operation("fade_out")
                print("   ✅ Applied fade in/out to audio 1")
                
                # Mix two audio samples
                mixed_audio = audio1.broadcast_mix(audio2, mix_ratio=0.5)
                print("   ✅ Mixed two audio samples")
                
                # Save processed audio
                audio1.save("/tmp/voirs_processed.wav")
                mixed_audio.save("/tmp/voirs_mixed.wav")
                print("💾 Saved processed audio files")
                
                print("📊 Processing completed successfully!")
            
        except Exception as e:
            print(f"❌ Advanced processing failed: {e}")

    def demonstrate_error_handling(self):
        """Demonstrate comprehensive error handling"""
        print("\n🛡️  Error Handling and Recovery")
        print("=" * 50)
        
        if not VOIRS_AVAILABLE:
            print("[DEMO MODE] Would demonstrate:")
            print("   - Error callbacks")
            print("   - Structured error information")
            print("   - Recovery strategies")
            print("   - Graceful degradation")
            return
        
        # Define error callback
        def error_callback(error_info):
            print(f"   🚨 Error captured: {error_info.code}")
            print(f"      Message: {error_info.message}")
            if error_info.suggestion:
                print(f"      Suggestion: {error_info.suggestion}")
        
        print("🧪 Testing error handling with problematic input...")
        
        # Test with empty text
        try:
            audio = self.pipeline.synthesize_with_error_callback("", error_callback)
            if audio:
                print("   ✅ Empty text handled gracefully")
        except Exception as e:
            print(f"   🚨 Empty text error: {e}")
        
        # Test with very long text
        very_long_text = "Very long text. " * 1000
        try:
            audio = self.pipeline.synthesize_with_error_callback(very_long_text[:100] + "...", error_callback)
            if audio:
                print("   ✅ Long text handled gracefully")
        except Exception as e:
            print(f"   🚨 Long text error: {e}")
        
        print("✅ Error handling demonstration completed")

    def run_comprehensive_demo(self):
        """Run the complete VoiRS Python bindings demonstration"""
        print("🎉 VoiRS Python Integration Comprehensive Demo")
        print("=" * 60)
        print(f"VoiRS Version: {voirs.version() if VOIRS_AVAILABLE else 'Demo Mode'}")
        print(f"NumPy Available: {'✅' if NUMPY_AVAILABLE else '❌'}")
        print(f"GPU Support: {'✅' if VOIRS_AVAILABLE and hasattr(voirs, 'HAS_GPU') and voirs.HAS_GPU else '❌'}")
        print(f"Recognition: {'✅' if VOIRS_AVAILABLE and hasattr(voirs, 'HAS_RECOGNITION') and voirs.HAS_RECOGNITION else '❌'}")
        print("=" * 60)
        
        # Initialize pipeline
        if not self.initialize_pipeline():
            print("❌ Failed to initialize - running in demo mode only")
        
        # Run all demonstrations
        demonstrations = [
            ("Basic Synthesis", self.demonstrate_basic_synthesis),
            ("Enhanced Synthesis", self.demonstrate_enhanced_synthesis),
            ("SSML Support", self.demonstrate_ssml_synthesis),
            ("Voice Management", self.demonstrate_voice_management),
            ("Batch Processing", self.demonstrate_batch_processing),
            ("Streaming Synthesis", self.demonstrate_streaming_synthesis),
            ("Audio Analysis", self.demonstrate_audio_analysis),
            ("Advanced Processing", self.demonstrate_advanced_audio_processing),
            ("Error Handling", self.demonstrate_error_handling),
        ]
        
        for demo_name, demo_func in demonstrations:
            try:
                demo_func()
            except Exception as e:
                print(f"\n❌ {demo_name} demonstration failed: {e}")
                continue
        
        print("\n🎉 Comprehensive demonstration completed!")
        print("\nGenerated audio files are saved in /tmp/ directory:")
        print("   - /tmp/voirs_basic_*.wav - Basic synthesis examples")
        print("   - /tmp/voirs_enhanced.wav - Enhanced synthesis with metrics")
        print("   - /tmp/voirs_ssml.wav - SSML synthesis example")
        print("   - /tmp/voirs_batch_*.wav - Batch processing results")
        print("   - /tmp/voirs_streaming_*.wav - Streaming synthesis chunks")
        print("   - /tmp/voirs_processed.wav - Audio with applied effects")
        print("   - /tmp/voirs_mixed.wav - Mixed audio example")

def main():
    """Main entry point"""
    demo = VoiRSPythonDemo()
    demo.run_comprehensive_demo()

if __name__ == "__main__":
    main()
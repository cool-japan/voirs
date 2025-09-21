#!/usr/bin/env python3
"""
VoiRS Speech Recognition Python Example

This example demonstrates how to use VoiRS for speech recognition tasks including:
- Loading and recognizing audio files
- Performing audio analysis
- Real-time audio processing
- Phoneme recognition and forced alignment

Requirements:
- voirs_ffi Python package with recognition support
- numpy (optional, for advanced audio processing)
- An audio file for testing (or use the generated example)
"""

import sys
import time
import numpy as np
from pathlib import Path

try:
    import voirs_ffi
except ImportError:
    print("Error: voirs_ffi package not found. Please install the VoiRS Python bindings.")
    sys.exit(1)

def check_features():
    """Check which features are available in the VoiRS installation."""
    print("=== VoiRS Feature Check ===")
    print(f"Version: {voirs_ffi.__version__}")
    print(f"NumPy support: {voirs_ffi.HAS_NUMPY}")
    print(f"GPU support: {voirs_ffi.HAS_GPU}")
    
    if hasattr(voirs_ffi, 'HAS_RECOGNITION'):
        print(f"Recognition support: {voirs_ffi.HAS_RECOGNITION}")
        if not voirs_ffi.HAS_RECOGNITION:
            print("Warning: Recognition features not available. Compile with 'recognition' feature.")
            return False
    else:
        print("Warning: Recognition support not detected in this VoiRS build.")
        return False
    
    print()
    return True

def create_test_audio():
    """Create a simple test audio buffer for demonstration."""
    print("=== Creating Test Audio ===")
    
    # Create a simple sine wave (440 Hz A note for 2 seconds)
    sample_rate = 16000
    duration = 2.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more realistic
    noise = 0.05 * np.random.randn(len(audio_data)).astype(np.float32)
    audio_data += noise
    
    if voirs_ffi.HAS_NUMPY:
        # Use NumPy integration if available
        audio_buffer = voirs_ffi.PyAudioBuffer.from_numpy(
            None,  # Python context (will be filled by PyO3)
            audio_data,
            sample_rate,
            1  # mono
        )
    else:
        # Fallback: create from samples list
        audio_buffer = voirs_ffi.PyAudioBuffer()
        # Note: This would need to be implemented in the actual bindings
        
    print(f"Created test audio: {duration}s, {sample_rate}Hz, {len(audio_data)} samples")
    return audio_buffer

def demonstrate_audio_analysis():
    """Demonstrate audio analysis capabilities."""
    print("=== Audio Analysis Demo ===")
    
    try:
        # Create audio analyzer
        analyzer = voirs_ffi.PyAudioAnalyzer()
        print("Audio analyzer created successfully")
        
        # Create test audio
        audio = create_test_audio()
        
        # Perform analysis
        print("Analyzing audio...")
        start_time = time.time()
        analysis = analyzer.analyze(audio)
        analysis_time = time.time() - start_time
        
        print(f"Analysis completed in {analysis_time:.3f}s")
        print(f"Results: {analysis}")
        print(f"  - Duration: {analysis.duration_seconds:.2f}s")
        print(f"  - Sample Rate: {analysis.sample_rate}Hz")
        print(f"  - Channels: {analysis.channels}")
        print(f"  - RMS Energy: {analysis.rms_energy:.4f}")
        print(f"  - Zero Crossing Rate: {analysis.zero_crossing_rate:.4f}")
        print(f"  - Spectral Centroid: {analysis.spectral_centroid:.2f}Hz")
        
    except Exception as e:
        print(f"Audio analysis failed: {e}")
    
    print()

def demonstrate_speech_recognition():
    """Demonstrate speech recognition with Whisper."""
    print("=== Speech Recognition Demo ===")
    
    try:
        # Create Whisper ASR model
        print("Loading Whisper model (this may take a moment)...")
        start_time = time.time()
        asr_model = voirs_ffi.PyASRModel.whisper("tiny")  # Use tiny model for speed
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
        
        # Get supported languages
        languages = asr_model.supported_languages()
        print(f"Supported languages: {', '.join(languages[:5])}..." + 
              f" (and {len(languages) - 5} more)" if len(languages) > 5 else "")
        
        # Create test audio (note: this is just a sine wave, so recognition will be poor)
        audio = create_test_audio()
        
        # Perform recognition
        print("Recognizing speech...")
        start_time = time.time()
        result = asr_model.recognize(audio)
        recognition_time = time.time() - start_time
        
        print(f"Recognition completed in {recognition_time:.3f}s")
        print(f"Result: {result}")
        print(f"  - Text: '{result.transcript.text}'")
        print(f"  - Language: {result.transcript.language}")
        print(f"  - Confidence: {result.confidence:.3f}")
        print(f"  - Word Count: {result.transcript.word_count}")
        print(f"  - Processing Time: {result.processing_time_ms:.1f}ms")
        
        if not result.transcript.text.strip():
            print("Note: No speech detected (expected for sine wave test audio)")
        
    except Exception as e:
        print(f"Speech recognition failed: {e}")
        if "not compiled in" in str(e):
            print("Hint: This VoiRS build doesn't include Whisper support.")
        elif "Failed to load" in str(e):
            print("Hint: Whisper model files may not be available or downloadable.")
    
    print()

def demonstrate_file_recognition(audio_file_path):
    """Demonstrate recognition from an audio file."""
    print(f"=== File Recognition Demo: {audio_file_path} ===")
    
    if not Path(audio_file_path).exists():
        print(f"Audio file not found: {audio_file_path}")
        print("Skipping file recognition demo.")
        return
    
    try:
        # Recognize directly from file
        print("Recognizing audio file...")
        start_time = time.time()
        result = voirs_ffi.PyASRModel.recognize_file(audio_file_path, "base")
        recognition_time = time.time() - start_time
        
        print(f"Recognition completed in {recognition_time:.3f}s")
        print(f"Result: {result}")
        print(f"  - Text: '{result.transcript.text}'")
        print(f"  - Confidence: {result.confidence:.3f}")
        print(f"  - Processing Time: {result.processing_time_ms:.1f}ms")
        
        # Also analyze the audio file
        print("\nAnalyzing audio file...")
        analysis = voirs_ffi.PyAudioAnalyzer.analyze_file(audio_file_path)
        print(f"Analysis: {analysis}")
        
    except Exception as e:
        print(f"File recognition failed: {e}")
    
    print()

def demonstrate_phoneme_recognition():
    """Demonstrate phoneme recognition and forced alignment."""
    print("=== Phoneme Recognition Demo ===")
    
    try:
        # Create phoneme recognizer for English
        phoneme_recognizer = voirs_ffi.PyPhonemeRecognizer("en")
        print("Phoneme recognizer created for English")
        
        # Create test audio
        audio = create_test_audio()
        
        # Perform phoneme recognition with known text
        test_text = "hello world"
        print(f"Aligning phonemes for text: '{test_text}'")
        
        start_time = time.time()
        alignments = phoneme_recognizer.recognize(audio, test_text)
        alignment_time = time.time() - start_time
        
        print(f"Phoneme alignment completed in {alignment_time:.3f}s")
        print(f"Found {len(alignments)} phoneme alignments:")
        
        for i, alignment in enumerate(alignments[:10]):  # Show first 10
            print(f"  {i+1}. {alignment}")
        
        if len(alignments) > 10:
            print(f"  ... and {len(alignments) - 10} more")
        
        if not alignments:
            print("Note: No phonemes aligned (expected for sine wave test audio)")
        
    except Exception as e:
        print(f"Phoneme recognition failed: {e}")
    
    print()

def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("=== Performance Monitoring Demo ===")
    
    try:
        # Create multiple audio buffers for batch processing
        print("Creating multiple audio samples for performance testing...")
        audio_samples = [create_test_audio() for _ in range(3)]
        
        # Load ASR model
        asr_model = voirs_ffi.PyASRModel.whisper("tiny")
        
        # Batch recognition with timing
        print("Performing batch recognition...")
        start_time = time.time()
        results = []
        
        for i, audio in enumerate(audio_samples):
            print(f"  Processing sample {i+1}/{len(audio_samples)}...")
            result = asr_model.recognize(audio)
            results.append(result)
        
        total_time = time.time() - start_time
        
        print(f"Batch processing completed in {total_time:.3f}s")
        print(f"Average time per sample: {total_time / len(audio_samples):.3f}s")
        print(f"Throughput: {len(audio_samples) / total_time:.1f} samples/second")
        
        # Calculate statistics
        confidences = [r.confidence for r in results]
        processing_times = [r.processing_time_ms for r in results]
        
        print(f"Confidence: min={min(confidences):.3f}, max={max(confidences):.3f}, "
              f"avg={sum(confidences)/len(confidences):.3f}")
        print(f"Processing time: min={min(processing_times):.1f}ms, "
              f"max={max(processing_times):.1f}ms, "
              f"avg={sum(processing_times)/len(processing_times):.1f}ms")
        
    except Exception as e:
        print(f"Performance monitoring failed: {e}")
    
    print()

def main():
    """Main demonstration function."""
    print("🎤 VoiRS Speech Recognition Python Example")
    print("=" * 50)
    
    # Check if recognition features are available
    if not check_features():
        print("This example requires VoiRS with recognition support.")
        print("Please compile VoiRS with: cargo build --features python-recognition")
        return
    
    try:
        # Run demonstrations
        demonstrate_audio_analysis()
        demonstrate_speech_recognition()
        demonstrate_phoneme_recognition()
        demonstrate_performance_monitoring()
        
        # Try file recognition if an audio file is provided
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
            demonstrate_file_recognition(audio_file)
        else:
            print("💡 Tip: Provide an audio file path as an argument to test file recognition:")
            print("   python speech_recognition_example.py /path/to/audio.wav")
        
        print("🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("- Try with your own audio files")
        print("- Experiment with different Whisper model sizes (tiny, base, small, medium, large)")
        print("- Use real audio with speech for better recognition results")
        print("- Integrate with your own applications")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
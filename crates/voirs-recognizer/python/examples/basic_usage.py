#!/usr/bin/env python3
"""
Basic usage example for VoiRS Recognizer Python package.

This example demonstrates:
1. Loading audio from file or creating synthetic audio
2. Speech recognition with different configurations
3. Audio analysis with quality metrics, prosody, and speaker features
4. Performance validation and optimization
"""

import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path

# Add the package to the path if running from development
sys.path.insert(0, str(Path(__file__).parent.parent))

import voirs_recognizer

def create_synthetic_audio(duration=2.0, sample_rate=16000, frequency=440.0):
    """Create synthetic audio for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a complex signal with multiple harmonics (speech-like)
    signal = (
        0.6 * np.sin(2 * np.pi * frequency * t) +
        0.3 * np.sin(2 * np.pi * frequency * 2 * t) +
        0.1 * np.sin(2 * np.pi * frequency * 3 * t)
    )
    
    # Add some amplitude modulation to simulate speech patterns
    envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 3 * t)
    signal = signal * envelope
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.02, len(signal))
    signal = signal + noise
    
    return signal.astype(np.float32)

def basic_recognition_example():
    """Demonstrate basic speech recognition."""
    print("=== Basic Speech Recognition ===")
    
    # Create synthetic audio
    print("Creating synthetic audio...")
    audio_samples = create_synthetic_audio(duration=3.0, frequency=150.0)
    audio = voirs_recognizer.AudioBuffer(audio_samples.tolist(), 16000)
    
    print(f"Audio: {audio}")
    print(f"Duration: {audio.duration():.2f}s")
    print(f"Sample rate: {audio.sample_rate()}Hz")
    print(f"Channels: {audio.channels()}")
    
    # Create recognizer with default configuration
    print("\nInitializing recognizer...")
    recognizer = voirs_recognizer.create_recognizer(
        language="en-US",
        word_timestamps=True,
        confidence_threshold=0.5
    )
    
    # Perform recognition
    print("Performing speech recognition...")
    start_time = time.time()
    result = recognizer.recognize(audio)
    processing_time = time.time() - start_time
    
    print(f"Recognition completed in {processing_time:.3f}s")
    print(f"Result: {result}")
    print(f"Transcript: '{result.text()}'")
    print(f"Confidence: {result.confidence():.3f}")
    print(f"Language: {result.language()}")
    
    # Show word timestamps if available
    word_timestamps = result.word_timestamps()
    if word_timestamps:
        print(f"\nWord timestamps ({len(word_timestamps)} words):")
        for i, word_info in enumerate(word_timestamps):
            print(f"  {i+1}. {word_info}")
    
    return result

def audio_analysis_example():
    """Demonstrate audio analysis capabilities."""
    print("\n=== Audio Analysis ===")
    
    # Create synthetic audio with different characteristics
    print("Creating synthetic audio for analysis...")
    audio_samples = create_synthetic_audio(duration=2.5, frequency=200.0)
    audio = voirs_recognizer.AudioBuffer(audio_samples.tolist(), 16000)
    
    # Create analyzer
    print("Initializing audio analyzer...")
    analyzer = voirs_recognizer.create_analyzer(
        quality_metrics=True,
        prosody_analysis=True,
        speaker_analysis=True
    )
    
    # Perform analysis
    print("Performing audio analysis...")
    start_time = time.time()
    analysis = analyzer.analyze(audio)
    processing_time = time.time() - start_time
    
    print(f"Analysis completed in {processing_time:.3f}s")
    print(f"Analysis result: {analysis}")
    
    # Show quality metrics
    quality_metrics = analysis.quality_metrics()
    if quality_metrics:
        print(f"\nQuality Metrics ({len(quality_metrics)} metrics):")
        for metric, value in quality_metrics.items():
            print(f"  • {metric}: {value:.3f}")
    
    # Show prosody features
    prosody_features = analysis.prosody_features()
    if prosody_features:
        print(f"\nProsody Features ({len(prosody_features)} features):")
        for feature, value in prosody_features.items():
            print(f"  • {feature}: {value:.3f}")
    
    # Show speaker features
    speaker_features = analysis.speaker_features()
    if speaker_features:
        print(f"\nSpeaker Features ({len(speaker_features)} features):")
        for feature, value in speaker_features.items():
            print(f"  • {feature}: {value:.3f}")
    
    return analysis

def performance_validation_example():
    """Demonstrate performance validation."""
    print("\n=== Performance Validation ===")
    
    # Create test audio
    audio_samples = create_synthetic_audio(duration=1.0)
    audio = voirs_recognizer.AudioBuffer(audio_samples.tolist(), 16000)
    
    # Create performance validator
    validator = voirs_recognizer.PerformanceValidator()
    
    # Simulate processing time
    processing_time = 0.15  # 150ms
    
    # Validate RTF (Real-Time Factor)
    rtf, rtf_passed = validator.validate_rtf(audio, processing_time)
    print(f"RTF Validation:")
    print(f"  • RTF: {rtf:.3f}")
    print(f"  • Status: {'✅ PASS' if rtf_passed else '❌ FAIL'}")
    print(f"  • Processing time: {processing_time:.3f}s")
    print(f"  • Audio duration: {audio.duration():.3f}s")
    
    # Validate memory usage
    try:
        memory_usage, memory_passed = validator.estimate_memory_usage()
        print(f"Memory Validation:")
        print(f"  • Memory usage: {memory_usage / (1024*1024):.1f} MB")
        print(f"  • Status: {'✅ PASS' if memory_passed else '❌ FAIL'}")
    except Exception as e:
        print(f"Memory validation error: {e}")
    
    return rtf, rtf_passed

def configuration_examples():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Examples ===")
    
    # ASR Configuration variations
    print("ASR Configuration Options:")
    
    configs = [
        ("Fast", voirs_recognizer.ASRConfig(confidence_threshold=0.3, beam_size=1)),
        ("Balanced", voirs_recognizer.ASRConfig(confidence_threshold=0.5, beam_size=3)),
        ("Accurate", voirs_recognizer.ASRConfig(confidence_threshold=0.7, beam_size=5)),
    ]
    
    for name, config in configs:
        print(f"  • {name}: {config}")
    
    # Audio Analysis Configuration variations
    print("\nAudio Analysis Configuration Options:")
    
    analysis_configs = [
        ("Quality Only", voirs_recognizer.AudioAnalysisConfig(
            quality_metrics=True, prosody_analysis=False, speaker_analysis=False
        )),
        ("Prosody Only", voirs_recognizer.AudioAnalysisConfig(
            quality_metrics=False, prosody_analysis=True, speaker_analysis=False
        )),
        ("Speaker Only", voirs_recognizer.AudioAnalysisConfig(
            quality_metrics=False, prosody_analysis=False, speaker_analysis=True
        )),
        ("Full Analysis", voirs_recognizer.AudioAnalysisConfig(
            quality_metrics=True, prosody_analysis=True, speaker_analysis=True
        )),
    ]
    
    for name, config in analysis_configs:
        print(f"  • {name}: {config}")

def utility_functions_example():
    """Demonstrate utility functions."""
    print("\n=== Utility Functions ===")
    
    # Confidence to label conversion
    print("Confidence to Label Conversion:")
    confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for conf in confidences:
        label = voirs_recognizer.confidence_to_label(conf)
        print(f"  • {conf:.1f} → '{label}'")
    
    # Package information
    print("\nPackage Information:")
    info = voirs_recognizer.get_package_info()
    for key, value in info.items():
        if key == 'features':
            print(f"  • {key}:")
            for feature in value:
                print(f"    - {feature}")
        else:
            print(f"  • {key}: {value}")

def main():
    """Main function to run all examples."""
    print("🎤 VoiRS Recognizer Python Package Examples")
    print("=" * 45)
    
    try:
        # Basic recognition
        recognition_result = basic_recognition_example()
        
        # Audio analysis
        analysis_result = audio_analysis_example()
        
        # Performance validation
        rtf, rtf_passed = performance_validation_example()
        
        # Configuration examples
        configuration_examples()
        
        # Utility functions
        utility_functions_example()
        
        # Summary
        print("\n" + "=" * 45)
        print("📊 Summary:")
        print(f"  • Recognition completed: {'✅' if recognition_result else '❌'}")
        print(f"  • Analysis completed: {'✅' if analysis_result else '❌'}")
        print(f"  • Performance validation: {'✅' if rtf_passed else '❌'}")
        print(f"  • All examples completed successfully!")
        
        print("\n🎯 Next Steps:")
        print("  • Try with real audio files using load_audio()")
        print("  • Experiment with different configurations")
        print("  • Integrate with your applications")
        print("  • Check out advanced examples in the examples directory")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
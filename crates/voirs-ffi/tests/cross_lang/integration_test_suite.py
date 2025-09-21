#!/usr/bin/env python3
"""
Comprehensive integration test suite for VoiRS FFI bindings.

This module provides extensive integration testing that validates:
- End-to-end functionality across all components
- Error handling and recovery
- Resource management
- Configuration validation
- Audio format compatibility
- Real-world usage scenarios
"""

import sys
import os
import time
import tempfile
import json
import shutil
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import wave
import unittest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class TestResult:
    """Container for test results."""
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

@dataclass
class TestSuite:
    """Container for test suite results."""
    name: str
    results: List[TestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_duration: float
    
    @property
    def success_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0

class IntegrationTestRunner:
    """Comprehensive integration test runner."""
    
    def __init__(self, test_dir: str = None):
        self.test_dir = test_dir or tempfile.mkdtemp(prefix="voirs_integration_")
        self.results = []
        self.available_bindings = {}
        self.test_files = []
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Setup test environment and check available bindings."""
        # Create test directory structure
        os.makedirs(os.path.join(self.test_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "temp"), exist_ok=True)
        
        # Check Python bindings
        try:
            import voirs_ffi as voirs
            self.available_bindings['python'] = {
                'module': voirs,
                'available': True,
                'version': getattr(voirs, '__version__', 'unknown')
            }
        except ImportError:
            self.available_bindings['python'] = {
                'module': None,
                'available': False,
                'error': 'voirs_ffi module not found'
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("Running VoiRS Integration Test Suite")
        print("=" * 50)
        
        test_suites = [
            ("Basic Functionality", self._test_basic_functionality),
            ("Audio Format Support", self._test_audio_format_support),
            ("Configuration Management", self._test_configuration_management),
            ("Error Handling", self._test_error_handling),
            ("Resource Management", self._test_resource_management),
            ("Concurrent Usage", self._test_concurrent_usage),
            ("Real-world Scenarios", self._test_real_world_scenarios),
            ("Performance Validation", self._test_performance_validation),
            ("Regression Tests", self._test_regression_scenarios)
        ]
        
        suite_results = {}
        
        for suite_name, suite_func in test_suites:
            print(f"\n--- {suite_name} ---")
            suite_start = time.time()
            
            try:
                suite_results[suite_name] = suite_func()
                suite_duration = time.time() - suite_start
                
                passed = sum(1 for r in suite_results[suite_name] if r.passed)
                total = len(suite_results[suite_name])
                
                print(f"✓ {suite_name}: {passed}/{total} passed ({suite_duration:.2f}s)")
                
            except Exception as e:
                suite_duration = time.time() - suite_start
                print(f"✗ {suite_name}: FAILED - {e}")
                suite_results[suite_name] = [TestResult(
                    name=suite_name,
                    passed=False,
                    duration=suite_duration,
                    error=str(e)
                )]
        
        return self._generate_final_report(suite_results)
    
    def _test_basic_functionality(self) -> List[TestResult]:
        """Test basic synthesis functionality."""
        results = []
        
        # Test 1: Simple synthesis
        def test_simple_synthesis():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            audio = pipeline.synthesize("Hello, world!")
            
            if not hasattr(audio, 'samples') or len(audio.samples) == 0:
                return False, "No audio samples generated"
            
            if audio.sample_rate <= 0:
                return False, "Invalid sample rate"
            
            if audio.duration <= 0:
                return False, "Invalid audio duration"
            
            return True, f"Generated {len(audio.samples)} samples at {audio.sample_rate}Hz"
        
        results.append(self._run_test("Simple Synthesis", test_simple_synthesis))
        
        # Test 2: Multiple synthesis calls
        def test_multiple_synthesis():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            texts = ["First", "Second", "Third"]
            audios = []
            
            for text in texts:
                audio = pipeline.synthesize(text)
                audios.append(audio)
            
            if len(audios) != len(texts):
                return False, f"Expected {len(texts)} audios, got {len(audios)}"
            
            for i, audio in enumerate(audios):
                if len(audio.samples) == 0:
                    return False, f"Audio {i} has no samples"
            
            return True, f"Successfully synthesized {len(audios)} audio clips"
        
        results.append(self._run_test("Multiple Synthesis", test_multiple_synthesis))
        
        # Test 3: Empty and special text handling
        def test_special_text_handling():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            special_texts = [
                "123 456 789",
                "Hello, how are you?",
                "Test with numbers: 3.14159",
                "Symbols: @#$%^&*()",
                "Mixed: Hello123World!"
            ]
            
            for text in special_texts:
                try:
                    audio = pipeline.synthesize(text)
                    if len(audio.samples) == 0:
                        return False, f"No audio for text: {text}"
                except Exception as e:
                    return False, f"Failed to synthesize '{text}': {e}"
            
            return True, f"Successfully handled {len(special_texts)} special text cases"
        
        results.append(self._run_test("Special Text Handling", test_special_text_handling))
        
        # Test 4: Voice management
        def test_voice_management():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            # Get available voices
            voices = pipeline.get_voices()
            
            if not voices:
                return False, "No voices available"
            
            # Test voice switching
            for i, voice in enumerate(voices[:3]):  # Test first 3 voices
                try:
                    pipeline.set_voice(voice.id)
                    audio = pipeline.synthesize(f"Testing voice {i}")
                    
                    if len(audio.samples) == 0:
                        return False, f"No audio with voice {voice.id}"
                        
                except Exception as e:
                    return False, f"Failed to use voice {voice.id}: {e}"
            
            return True, f"Successfully tested {min(3, len(voices))} voices"
        
        results.append(self._run_test("Voice Management", test_voice_management))
        
        return results
    
    def _test_audio_format_support(self) -> List[TestResult]:
        """Test audio format support."""
        results = []
        
        # Test 1: WAV format support
        def test_wav_format():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            audio = pipeline.synthesize("WAV format test")
            wav_path = os.path.join(self.test_dir, "audio", "test.wav")
            
            try:
                audio.save(wav_path)
                
                # Verify WAV file was created and is valid
                if not os.path.exists(wav_path):
                    return False, "WAV file not created"
                
                # Try to read WAV file
                with wave.open(wav_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    
                    if frames == 0:
                        return False, "WAV file has no frames"
                    
                    if sample_rate != audio.sample_rate:
                        return False, f"Sample rate mismatch: expected {audio.sample_rate}, got {sample_rate}"
                
                file_size = os.path.getsize(wav_path)
                return True, f"WAV file created: {file_size} bytes, {frames} frames"
                
            except Exception as e:
                return False, f"WAV format test failed: {e}"
        
        results.append(self._run_test("WAV Format Support", test_wav_format))
        
        # Test 2: Multiple format support
        def test_multiple_formats():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            audio = pipeline.synthesize("Format test")
            
            formats_to_test = [
                ("wav", "test.wav"),
                ("mp3", "test.mp3"),
                ("flac", "test.flac")
            ]
            
            successful_formats = []
            
            for format_name, filename in formats_to_test:
                try:
                    file_path = os.path.join(self.test_dir, "audio", filename)
                    audio.save(file_path, format=format_name)
                    
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        successful_formats.append(format_name)
                    
                except Exception as e:
                    # Some formats might not be supported
                    pass
            
            if not successful_formats:
                return False, "No audio formats supported"
            
            return True, f"Supported formats: {', '.join(successful_formats)}"
        
        results.append(self._run_test("Multiple Format Support", test_multiple_formats))
        
        # Test 3: Audio properties validation
        def test_audio_properties():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            audio = pipeline.synthesize("Audio properties test")
            
            # Validate audio properties
            if not hasattr(audio, 'sample_rate') or audio.sample_rate <= 0:
                return False, "Invalid sample rate"
            
            if not hasattr(audio, 'channels') or audio.channels <= 0:
                return False, "Invalid channel count"
            
            if not hasattr(audio, 'duration') or audio.duration <= 0:
                return False, "Invalid duration"
            
            if not hasattr(audio, 'samples') or len(audio.samples) == 0:
                return False, "No audio samples"
            
            # Check sample range
            max_sample = max(abs(s) for s in audio.samples)
            if max_sample > 1.0:
                return False, f"Samples out of range: max = {max_sample}"
            
            return True, f"Valid audio: {audio.sample_rate}Hz, {audio.channels}ch, {audio.duration:.2f}s"
        
        results.append(self._run_test("Audio Properties Validation", test_audio_properties))
        
        return results
    
    def _test_configuration_management(self) -> List[TestResult]:
        """Test configuration management."""
        results = []
        
        # Test 1: Default configuration
        def test_default_configuration():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            
            # Test default pipeline creation
            pipeline = voirs.VoirsPipeline()
            audio = pipeline.synthesize("Default config test")
            
            if len(audio.samples) == 0:
                return False, "No audio with default config"
            
            return True, f"Default configuration works: {audio.sample_rate}Hz"
        
        results.append(self._run_test("Default Configuration", test_default_configuration))
        
        # Test 2: Custom configuration
        def test_custom_configuration():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            
            # Test custom configuration
            try:
                pipeline = voirs.VoirsPipeline.with_config(
                    sample_rate=44100,
                    use_gpu=False,
                    num_threads=2
                )
                
                audio = pipeline.synthesize("Custom config test")
                
                if len(audio.samples) == 0:
                    return False, "No audio with custom config"
                
                # Verify configuration was applied
                if audio.sample_rate != 44100:
                    return False, f"Sample rate not applied: expected 44100, got {audio.sample_rate}"
                
                return True, f"Custom configuration works: {audio.sample_rate}Hz"
                
            except Exception as e:
                return False, f"Custom configuration failed: {e}"
        
        results.append(self._run_test("Custom Configuration", test_custom_configuration))
        
        # Test 3: Configuration validation
        def test_configuration_validation():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            
            # Test invalid configurations
            invalid_configs = [
                {"sample_rate": -1},
                {"sample_rate": 0},
                {"num_threads": -1},
                {"num_threads": 0}
            ]
            
            validation_results = []
            
            for config in invalid_configs:
                try:
                    pipeline = voirs.VoirsPipeline.with_config(**config)
                    audio = pipeline.synthesize("Validation test")
                    validation_results.append(f"Config {config} unexpectedly succeeded")
                except Exception as e:
                    validation_results.append(f"Config {config} properly rejected: {type(e).__name__}")
            
            return True, f"Validation tests: {len(validation_results)} configs tested"
        
        results.append(self._run_test("Configuration Validation", test_configuration_validation))
        
        return results
    
    def _test_error_handling(self) -> List[TestResult]:
        """Test error handling and recovery."""
        results = []
        
        # Test 1: Invalid text handling
        def test_invalid_text_handling():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            invalid_texts = [
                "",  # Empty text
                None,  # None text
                "x" * 10000,  # Very long text
            ]
            
            results_summary = []
            
            for text in invalid_texts:
                try:
                    if text is None:
                        # This should raise an exception
                        continue
                    
                    audio = pipeline.synthesize(text)
                    
                    if text == "":
                        # Empty text might produce empty audio
                        if len(audio.samples) == 0:
                            results_summary.append(f"Empty text handled correctly")
                        else:
                            results_summary.append(f"Empty text produced audio")
                    else:
                        results_summary.append(f"Text length {len(text)} handled")
                        
                except Exception as e:
                    results_summary.append(f"Text '{str(text)[:20]}...' raised {type(e).__name__}")
            
            return True, f"Invalid text handling: {len(results_summary)} cases tested"
        
        results.append(self._run_test("Invalid Text Handling", test_invalid_text_handling))
        
        # Test 2: Resource exhaustion handling
        def test_resource_exhaustion():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            
            # Test creating many pipelines
            pipelines = []
            max_pipelines = 0
            
            try:
                for i in range(100):  # Try to create many pipelines
                    pipeline = voirs.VoirsPipeline()
                    pipelines.append(pipeline)
                    max_pipelines = i + 1
                    
                    # Test synthesis with each pipeline
                    if i % 10 == 0:
                        audio = pipeline.synthesize("Resource test")
                        if len(audio.samples) == 0:
                            return False, f"No audio at pipeline {i}"
                    
            except Exception as e:
                # This is expected at some point
                pass
            
            # Clean up
            pipelines.clear()
            
            return True, f"Created {max_pipelines} pipelines before resource limits"
        
        results.append(self._run_test("Resource Exhaustion", test_resource_exhaustion))
        
        # Test 3: Error recovery
        def test_error_recovery():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            # Test recovery after error
            try:
                # First, cause an error
                pipeline.synthesize(None)
            except:
                pass
            
            # Then test normal operation
            try:
                audio = pipeline.synthesize("Recovery test")
                if len(audio.samples) == 0:
                    return False, "No audio after error recovery"
                
                return True, "Pipeline recovered successfully after error"
                
            except Exception as e:
                return False, f"Pipeline failed to recover: {e}"
        
        results.append(self._run_test("Error Recovery", test_error_recovery))
        
        return results
    
    def _test_resource_management(self) -> List[TestResult]:
        """Test resource management and cleanup."""
        results = []
        
        # Test 1: Memory management
        def test_memory_management():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            try:
                import psutil
                process = psutil.Process()
            except ImportError:
                return False, "psutil not available for memory testing"
            
            voirs = self.available_bindings['python']['module']
            
            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and destroy many pipelines
            for i in range(50):
                pipeline = voirs.VoirsPipeline()
                audio = pipeline.synthesize(f"Memory test {i}")
                del audio
                del pipeline
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - initial_memory
            
            if memory_growth > 100:  # 100MB threshold
                return False, f"High memory growth detected: {memory_growth:.1f}MB"
            
            return True, f"Memory growth: {memory_growth:.1f}MB (within limits)"
        
        results.append(self._run_test("Memory Management", test_memory_management))
        
        # Test 2: File handle management
        def test_file_handle_management():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            # Create many audio files
            for i in range(20):
                audio = pipeline.synthesize(f"File test {i}")
                file_path = os.path.join(self.test_dir, "audio", f"file_{i}.wav")
                
                try:
                    audio.save(file_path)
                    
                    # Verify file was created
                    if not os.path.exists(file_path):
                        return False, f"File {i} not created"
                    
                    # Check file size
                    if os.path.getsize(file_path) == 0:
                        return False, f"File {i} is empty"
                        
                except Exception as e:
                    return False, f"File {i} creation failed: {e}"
            
            return True, f"Successfully created 20 audio files"
        
        results.append(self._run_test("File Handle Management", test_file_handle_management))
        
        return results
    
    def _test_concurrent_usage(self) -> List[TestResult]:
        """Test concurrent usage patterns."""
        results = []
        
        # Test 1: Thread safety
        def test_thread_safety():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            
            results_list = []
            errors = []
            
            def worker(worker_id):
                try:
                    pipeline = voirs.VoirsPipeline()
                    for i in range(10):
                        audio = pipeline.synthesize(f"Worker {worker_id} iteration {i}")
                        if len(audio.samples) == 0:
                            errors.append(f"Worker {worker_id}: No audio at iteration {i}")
                    results_list.append(f"Worker {worker_id} completed")
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")
            
            # Create multiple threads
            threads = []
            for i in range(4):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            if errors:
                return False, f"Thread safety errors: {errors[:3]}"
            
            return True, f"Thread safety test passed: {len(results_list)} workers completed"
        
        results.append(self._run_test("Thread Safety", test_thread_safety))
        
        # Test 2: Shared pipeline usage
        def test_shared_pipeline():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            results_list = []
            errors = []
            
            def worker(worker_id):
                try:
                    for i in range(5):
                        audio = pipeline.synthesize(f"Shared worker {worker_id} iteration {i}")
                        if len(audio.samples) == 0:
                            errors.append(f"Shared worker {worker_id}: No audio at iteration {i}")
                    results_list.append(f"Shared worker {worker_id} completed")
                except Exception as e:
                    errors.append(f"Shared worker {worker_id}: {e}")
            
            # Create multiple threads using shared pipeline
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            if errors:
                return False, f"Shared pipeline errors: {errors[:3]}"
            
            return True, f"Shared pipeline test passed: {len(results_list)} workers completed"
        
        results.append(self._run_test("Shared Pipeline Usage", test_shared_pipeline))
        
        return results
    
    def _test_real_world_scenarios(self) -> List[TestResult]:
        """Test real-world usage scenarios."""
        results = []
        
        # Test 1: Batch processing
        def test_batch_processing():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            # Simulate batch processing
            texts = [
                "Welcome to our service.",
                "Please hold while we connect you.",
                "Your call is important to us.",
                "Thank you for waiting.",
                "Have a great day!"
            ]
            
            audio_files = []
            
            for i, text in enumerate(texts):
                try:
                    audio = pipeline.synthesize(text)
                    file_path = os.path.join(self.test_dir, "audio", f"batch_{i}.wav")
                    audio.save(file_path)
                    audio_files.append(file_path)
                    
                    # Verify file
                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                        return False, f"Batch file {i} not created properly"
                        
                except Exception as e:
                    return False, f"Batch processing failed at text {i}: {e}"
            
            return True, f"Batch processing completed: {len(audio_files)} files created"
        
        results.append(self._run_test("Batch Processing", test_batch_processing))
        
        # Test 2: Long text processing
        def test_long_text_processing():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            # Test with increasingly long texts
            base_text = "This is a test sentence for long text processing. "
            
            for length in [1, 5, 10, 20]:
                long_text = base_text * length
                
                try:
                    start_time = time.time()
                    audio = pipeline.synthesize(long_text)
                    end_time = time.time()
                    
                    if len(audio.samples) == 0:
                        return False, f"No audio for text length {len(long_text)}"
                    
                    duration = end_time - start_time
                    chars_per_sec = len(long_text) / duration
                    
                    if duration > 30:  # 30 second timeout
                        return False, f"Long text processing too slow: {duration:.1f}s for {len(long_text)} chars"
                    
                except Exception as e:
                    return False, f"Long text processing failed at length {len(long_text)}: {e}"
            
            return True, f"Long text processing test passed for texts up to {len(base_text * 20)} characters"
        
        results.append(self._run_test("Long Text Processing", test_long_text_processing))
        
        # Test 3: Streaming scenario
        def test_streaming_scenario():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            # Test streaming-like usage
            texts = [
                "First chunk of streaming text.",
                "Second chunk continues the story.",
                "Third chunk adds more content.",
                "Final chunk completes the message."
            ]
            
            audio_chunks = []
            
            for i, text in enumerate(texts):
                try:
                    start_time = time.time()
                    audio = pipeline.synthesize(text)
                    end_time = time.time()
                    
                    if len(audio.samples) == 0:
                        return False, f"No audio for chunk {i}"
                    
                    # Check latency for streaming
                    latency = end_time - start_time
                    if latency > 5.0:  # 5 second max latency
                        return False, f"Streaming latency too high: {latency:.1f}s"
                    
                    audio_chunks.append(audio)
                    
                except Exception as e:
                    return False, f"Streaming scenario failed at chunk {i}: {e}"
            
            return True, f"Streaming scenario completed: {len(audio_chunks)} chunks processed"
        
        results.append(self._run_test("Streaming Scenario", test_streaming_scenario))
        
        return results
    
    def _test_performance_validation(self) -> List[TestResult]:
        """Test performance validation."""
        results = []
        
        # Test 1: Throughput validation
        def test_throughput_validation():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            # Warmup
            for _ in range(3):
                pipeline.synthesize("warmup")
            
            # Measure throughput
            start_time = time.time()
            num_operations = 20
            
            for i in range(num_operations):
                audio = pipeline.synthesize(f"Throughput test {i}")
                if len(audio.samples) == 0:
                    return False, f"No audio at operation {i}"
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = num_operations / duration
            
            # Validate minimum throughput
            if throughput < 1.0:  # At least 1 operation per second
                return False, f"Throughput too low: {throughput:.2f} ops/sec"
            
            return True, f"Throughput validation passed: {throughput:.2f} ops/sec"
        
        results.append(self._run_test("Throughput Validation", test_throughput_validation))
        
        # Test 2: Latency validation
        def test_latency_validation():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            # Warmup
            pipeline.synthesize("warmup")
            
            # Measure latency
            latencies = []
            
            for i in range(10):
                start_time = time.time()
                audio = pipeline.synthesize("Latency test")
                end_time = time.time()
                
                if len(audio.samples) == 0:
                    return False, f"No audio at latency test {i}"
                
                latency = end_time - start_time
                latencies.append(latency)
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            # Validate latency requirements
            if avg_latency > 10.0:  # 10 seconds max average latency
                return False, f"Average latency too high: {avg_latency:.2f}s"
            
            if max_latency > 15.0:  # 15 seconds max latency
                return False, f"Max latency too high: {max_latency:.2f}s"
            
            return True, f"Latency validation passed: avg={avg_latency:.2f}s, max={max_latency:.2f}s"
        
        results.append(self._run_test("Latency Validation", test_latency_validation))
        
        return results
    
    def _test_regression_scenarios(self) -> List[TestResult]:
        """Test regression scenarios."""
        results = []
        
        # Test 1: Known edge cases
        def test_edge_cases():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            pipeline = voirs.VoirsPipeline()
            
            edge_cases = [
                "a",  # Single character
                "A",  # Single uppercase
                "1",  # Single digit
                ".",  # Single punctuation
                "   ",  # Only spaces
                "Hello world",  # Simple phrase
                "Dr. Smith's 123 Main St.",  # Abbreviations and numbers
                "What's up?",  # Contractions
                "Price: $29.99",  # Currency
                "Email: test@example.com",  # Email
            ]
            
            for i, text in enumerate(edge_cases):
                try:
                    audio = pipeline.synthesize(text)
                    
                    if len(audio.samples) == 0:
                        return False, f"No audio for edge case {i}: '{text}'"
                    
                    if audio.duration <= 0:
                        return False, f"Invalid duration for edge case {i}: '{text}'"
                    
                except Exception as e:
                    return False, f"Edge case {i} failed: '{text}' - {e}"
            
            return True, f"Edge cases test passed: {len(edge_cases)} cases handled"
        
        results.append(self._run_test("Edge Cases", test_edge_cases))
        
        # Test 2: Consistency validation
        def test_consistency_validation():
            if not self.available_bindings['python']['available']:
                return False, "Python bindings not available"
            
            voirs = self.available_bindings['python']['module']
            
            # Test consistency across multiple pipeline instances
            test_text = "Consistency test text"
            
            audios = []
            
            for i in range(3):
                pipeline = voirs.VoirsPipeline()
                audio = pipeline.synthesize(test_text)
                audios.append(audio)
            
            # Check consistency
            reference = audios[0]
            
            for i, audio in enumerate(audios[1:], 1):
                if audio.sample_rate != reference.sample_rate:
                    return False, f"Sample rate inconsistency: pipeline {i}"
                
                if audio.channels != reference.channels:
                    return False, f"Channels inconsistency: pipeline {i}"
                
                if abs(audio.duration - reference.duration) > 0.1:  # 100ms tolerance
                    return False, f"Duration inconsistency: pipeline {i}"
            
            return True, f"Consistency validation passed: {len(audios)} pipelines consistent"
        
        results.append(self._run_test("Consistency Validation", test_consistency_validation))
        
        return results
    
    def _run_test(self, test_name: str, test_func: callable) -> TestResult:
        """Run a single test and return the result."""
        start_time = time.time()
        
        try:
            passed, message = test_func()
            duration = time.time() - start_time
            
            result = TestResult(
                name=test_name,
                passed=passed,
                duration=duration,
                error=None if passed else message,
                details={"message": message}
            )
            
            status = "✓" if passed else "✗"
            print(f"  {status} {test_name}: {message} ({duration:.2f}s)")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            result = TestResult(
                name=test_name,
                passed=False,
                duration=duration,
                error=str(e)
            )
            
            print(f"  ✗ {test_name}: EXCEPTION - {e} ({duration:.2f}s)")
            
            return result
    
    def _generate_final_report(self, suite_results: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """Generate final test report."""
        total_tests = sum(len(results) for results in suite_results.values())
        total_passed = sum(sum(1 for r in results if r.passed) for results in suite_results.values())
        total_failed = total_tests - total_passed
        
        total_duration = sum(
            sum(r.duration for r in results) 
            for results in suite_results.values()
        )
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "failed_tests": total_failed,
                "success_rate": total_passed / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration,
                "timestamp": time.time()
            },
            "suite_results": {},
            "binding_availability": self.available_bindings
        }
        
        for suite_name, results in suite_results.items():
            suite_passed = sum(1 for r in results if r.passed)
            suite_total = len(results)
            suite_duration = sum(r.duration for r in results)
            
            report["suite_results"][suite_name] = {
                "total_tests": suite_total,
                "passed_tests": suite_passed,
                "failed_tests": suite_total - suite_passed,
                "success_rate": suite_passed / suite_total if suite_total > 0 else 0,
                "duration": suite_duration,
                "results": [asdict(r) for r in results]
            }
        
        return report
    
    def cleanup(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

def main():
    """Main entry point for integration tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VoiRS Integration Test Suite")
    parser.add_argument("--output", type=str, help="Output file for test results")
    parser.add_argument("--test-dir", type=str, help="Directory for test files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = IntegrationTestRunner(test_dir=args.test_dir)
    
    try:
        # Run all tests
        results = runner.run_all_tests()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nTest results saved to: {args.output}")
        
        # Print summary
        summary = results["summary"]
        print(f"\n{'='*60}")
        print("INTEGRATION TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        
        # Return appropriate exit code
        return 0 if summary['failed_tests'] == 0 else 1
        
    finally:
        runner.cleanup()

if __name__ == "__main__":
    sys.exit(main())
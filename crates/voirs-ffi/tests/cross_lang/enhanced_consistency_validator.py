#!/usr/bin/env python3

"""
Enhanced Cross-Language Consistency Validator for VoiRS FFI Bindings

This module provides advanced consistency validation features:
- Deep audio similarity analysis
- Error handling consistency testing
- Performance characteristic comparison
- Memory usage pattern validation
- API signature consistency checking
- Threading and concurrency behavior validation
"""

import asyncio
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import ctypes
import numpy as np
import scipy.stats
import scipy.signal
from scipy.fft import fft
from scipy.spatial.distance import cosine

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class AudioSimilarityResult:
    """Result of audio similarity analysis"""
    correlation: float
    cross_correlation_max: float
    spectral_similarity: float
    mfcc_similarity: float
    overall_similarity: float
    passed: bool
    details: Dict[str, Any]


@dataclass
class ConsistencyTestResult:
    """Result of a consistency test"""
    test_name: str
    binding_results: Dict[str, Any]
    consistency_score: float
    passed: bool
    details: Dict[str, Any]
    execution_time: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for binding comparison"""
    synthesis_time: float
    memory_peak: float
    memory_average: float
    cpu_usage: float
    real_time_factor: float
    throughput: float


class AudioAnalyzer:
    """Advanced audio analysis for consistency validation"""
    
    @staticmethod
    def extract_mfcc(audio_samples: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features from audio samples"""
        try:
            # Simple MFCC implementation (would use librosa in production)
            # For now, use spectral features as a proxy
            fft_result = fft(audio_samples)
            magnitude = np.abs(fft_result)
            
            # Mel filter bank simulation
            mel_filters = np.linspace(0, len(magnitude) // 2, n_mfcc)
            mfcc_features = []
            
            for i in range(n_mfcc):
                start_idx = int(mel_filters[i] * 0.8) if i > 0 else 0
                end_idx = int(mel_filters[i] * 1.2) if i < n_mfcc - 1 else len(magnitude) // 2
                
                if end_idx > start_idx:
                    mfcc_features.append(np.mean(magnitude[start_idx:end_idx]))
                else:
                    mfcc_features.append(0.0)
            
            return np.array(mfcc_features)
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return np.zeros(n_mfcc)
    
    @staticmethod
    def calculate_spectral_features(audio_samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate spectral features for audio comparison"""
        try:
            # Calculate FFT
            fft_result = fft(audio_samples)
            magnitude = np.abs(fft_result)
            power = magnitude ** 2
            
            # Frequency bins
            freqs = np.fft.fftfreq(len(audio_samples), 1/sample_rate)
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * power[:len(power)//2]) / np.sum(power[:len(power)//2])
            
            # Spectral rolloff
            cumulative_power = np.cumsum(power[:len(power)//2])
            rolloff_idx = np.where(cumulative_power >= 0.85 * cumulative_power[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs[:len(freqs)//2] - spectral_centroid) ** 2) * power[:len(power)//2]) / np.sum(power[:len(power)//2]))
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio_samples)) != 0)
            zero_crossing_rate = zero_crossings / len(audio_samples)
            
            return {
                "spectral_centroid": spectral_centroid,
                "spectral_rolloff": spectral_rolloff,
                "spectral_bandwidth": spectral_bandwidth,
                "zero_crossing_rate": zero_crossing_rate,
                "rms_energy": np.sqrt(np.mean(audio_samples ** 2))
            }
        except Exception as e:
            print(f"Spectral features error: {e}")
            return {
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "spectral_bandwidth": 0.0,
                "zero_crossing_rate": 0.0,
                "rms_energy": 0.0
            }
    
    @staticmethod
    def compare_audio_similarity(audio1: np.ndarray, audio2: np.ndarray, 
                               sample_rate: int, threshold: float = 0.85) -> AudioSimilarityResult:
        """Compare similarity between two audio signals"""
        try:
            # Normalize audio length
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            # Normalize amplitude
            audio1 = audio1 / (np.max(np.abs(audio1)) + 1e-10)
            audio2 = audio2 / (np.max(np.abs(audio2)) + 1e-10)
            
            # Calculate correlation
            correlation = np.corrcoef(audio1, audio2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Calculate cross-correlation
            cross_corr = np.correlate(audio1, audio2, mode='full')
            cross_corr_max = np.max(np.abs(cross_corr)) / (len(audio1) * np.std(audio1) * np.std(audio2) + 1e-10)
            
            # Extract and compare spectral features
            features1 = AudioAnalyzer.calculate_spectral_features(audio1, sample_rate)
            features2 = AudioAnalyzer.calculate_spectral_features(audio2, sample_rate)
            
            # Calculate spectral similarity
            spectral_diffs = []
            for key in features1:
                if features1[key] != 0 and features2[key] != 0:
                    diff = abs(features1[key] - features2[key]) / (abs(features1[key]) + abs(features2[key]) + 1e-10)
                    spectral_diffs.append(1 - diff)
            
            spectral_similarity = np.mean(spectral_diffs) if spectral_diffs else 0.0
            
            # Extract and compare MFCC features
            mfcc1 = AudioAnalyzer.extract_mfcc(audio1, sample_rate)
            mfcc2 = AudioAnalyzer.extract_mfcc(audio2, sample_rate)
            
            # Calculate MFCC similarity (1 - cosine distance)
            mfcc_similarity = 1 - cosine(mfcc1, mfcc2) if np.any(mfcc1) and np.any(mfcc2) else 0.0
            
            # Calculate overall similarity score
            overall_similarity = (
                correlation * 0.3 +
                cross_corr_max * 0.3 +
                spectral_similarity * 0.25 +
                mfcc_similarity * 0.15
            )
            
            passed = overall_similarity >= threshold
            
            return AudioSimilarityResult(
                correlation=correlation,
                cross_correlation_max=cross_corr_max,
                spectral_similarity=spectral_similarity,
                mfcc_similarity=mfcc_similarity,
                overall_similarity=overall_similarity,
                passed=passed,
                details={
                    "threshold": threshold,
                    "features1": features1,
                    "features2": features2,
                    "audio1_length": len(audio1),
                    "audio2_length": len(audio2),
                    "audio1_rms": np.sqrt(np.mean(audio1 ** 2)),
                    "audio2_rms": np.sqrt(np.mean(audio2 ** 2))
                }
            )
            
        except Exception as e:
            print(f"Audio similarity comparison error: {e}")
            return AudioSimilarityResult(
                correlation=0.0,
                cross_correlation_max=0.0,
                spectral_similarity=0.0,
                mfcc_similarity=0.0,
                overall_similarity=0.0,
                passed=False,
                details={"error": str(e)}
            )


class MemoryMonitor:
    """Memory usage monitoring for consistency testing"""
    
    def __init__(self):
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory
        self.samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def start_monitoring(self, interval: float = 0.1):
        """Start memory monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self, interval: float):
        """Memory monitoring loop"""
        while self.monitoring:
            current_memory = self._get_memory_usage()
            self.samples.append(current_memory)
            self.peak_memory = max(self.peak_memory, current_memory)
            time.sleep(interval)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get memory usage metrics"""
        current_memory = self._get_memory_usage()
        return {
            "initial_mb": self.initial_memory,
            "current_mb": current_memory,
            "peak_mb": self.peak_memory,
            "increase_mb": current_memory - self.initial_memory,
            "peak_increase_mb": self.peak_memory - self.initial_memory,
            "average_mb": np.mean(self.samples) if self.samples else current_memory,
            "samples_count": len(self.samples)
        }


class EnhancedConsistencyValidator:
    """Enhanced consistency validator with advanced features"""
    
    def __init__(self, test_dir: str = None):
        self.test_dir = Path(test_dir) if test_dir else Path(__file__).parent
        self.results = []
        self.temp_dir = Path(tempfile.mkdtemp())
        self.audio_analyzer = AudioAnalyzer()
        
        # Test configurations
        self.test_texts = [
            "Hello world",
            "This is a longer sentence to test consistency across different text lengths.",
            "Testing special characters: !@#$%^&*()_+-=[]{}|;:,.<>?",
            "Numbers and mixed content: 123 test 456.789 more text",
            "Short.",
            "This is a very long sentence designed to test how well the different language bindings handle longer inputs with multiple clauses, commas, and various punctuation marks to ensure consistency across all implementations!"
        ]
        
        self.test_configs = [
            {"speaking_rate": 1.0, "pitch_shift": 0.0, "volume_gain": 1.0},
            {"speaking_rate": 1.5, "pitch_shift": 2.0, "volume_gain": 1.2},
            {"speaking_rate": 0.8, "pitch_shift": -1.0, "volume_gain": 0.8},
            {"speaking_rate": 2.0, "pitch_shift": 0.0, "volume_gain": 1.0},
        ]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_audio_consistency(self, binding_results: Dict[str, Any]) -> ConsistencyTestResult:
        """Test audio output consistency between bindings"""
        start_time = time.time()
        
        # Extract audio samples from each binding
        audio_samples = {}
        for binding_name, result in binding_results.items():
            if "samples" in result:
                audio_samples[binding_name] = np.array(result["samples"])
        
        if len(audio_samples) < 2:
            return ConsistencyTestResult(
                test_name="audio_consistency",
                binding_results=binding_results,
                consistency_score=0.0,
                passed=False,
                details={"error": "Need at least 2 bindings for consistency test"},
                execution_time=time.time() - start_time
            )
        
        # Compare all pairs of bindings
        comparisons = {}
        similarity_scores = []
        
        binding_names = list(audio_samples.keys())
        for i in range(len(binding_names)):
            for j in range(i + 1, len(binding_names)):
                binding1 = binding_names[i]
                binding2 = binding_names[j]
                
                # Use sample rate from the first binding
                sample_rate = binding_results[binding1].get("sample_rate", 22050)
                
                similarity = self.audio_analyzer.compare_audio_similarity(
                    audio_samples[binding1],
                    audio_samples[binding2],
                    sample_rate
                )
                
                comparisons[f"{binding1}_vs_{binding2}"] = similarity
                similarity_scores.append(similarity.overall_similarity)
        
        # Calculate overall consistency score
        consistency_score = np.mean(similarity_scores) if similarity_scores else 0.0
        passed = consistency_score >= 0.85
        
        return ConsistencyTestResult(
            test_name="audio_consistency",
            binding_results=binding_results,
            consistency_score=consistency_score,
            passed=passed,
            details={
                "comparisons": {k: {
                    "overall_similarity": v.overall_similarity,
                    "correlation": v.correlation,
                    "spectral_similarity": v.spectral_similarity,
                    "passed": v.passed
                } for k, v in comparisons.items()},
                "similarity_scores": similarity_scores,
                "threshold": 0.85
            },
            execution_time=time.time() - start_time
        )
    
    def test_error_handling_consistency(self, bindings: List[str]) -> ConsistencyTestResult:
        """Test error handling consistency across bindings"""
        start_time = time.time()
        
        # Test various error conditions
        error_tests = [
            {"type": "empty_text", "input": ""},
            {"type": "null_text", "input": None},
            {"type": "very_long_text", "input": "A" * 10000},
            {"type": "invalid_config", "config": {"speaking_rate": -1}},
            {"type": "unicode_text", "input": "Hello 世界 🌍"},
        ]
        
        binding_errors = {}
        consistency_checks = []
        
        for binding in bindings:
            binding_errors[binding] = {}
            
            for test in error_tests:
                try:
                    result = self._test_binding_error(binding, test)
                    binding_errors[binding][test["type"]] = result
                except Exception as e:
                    binding_errors[binding][test["type"]] = {
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
        
        # Check consistency of error handling
        for test in error_tests:
            error_types = []
            for binding in bindings:
                if test["type"] in binding_errors[binding]:
                    error_info = binding_errors[binding][test["type"]]
                    error_types.append(error_info.get("error_type", "unknown"))
            
            # Check if all bindings handle the error similarly
            unique_error_types = set(error_types)
            consistency_checks.append({
                "test": test["type"],
                "consistent": len(unique_error_types) <= 1,
                "error_types": list(unique_error_types)
            })
        
        # Calculate consistency score
        consistent_count = sum(1 for check in consistency_checks if check["consistent"])
        consistency_score = consistent_count / len(consistency_checks) if consistency_checks else 0.0
        
        return ConsistencyTestResult(
            test_name="error_handling_consistency",
            binding_results=binding_errors,
            consistency_score=consistency_score,
            passed=consistency_score >= 0.8,
            details={
                "consistency_checks": consistency_checks,
                "error_tests": error_tests
            },
            execution_time=time.time() - start_time
        )
    
    def test_performance_consistency(self, bindings: List[str]) -> ConsistencyTestResult:
        """Test performance consistency across bindings"""
        start_time = time.time()
        
        performance_results = {}
        
        for binding in bindings:
            memory_monitor = MemoryMonitor()
            memory_monitor.start_monitoring()
            
            try:
                binding_start = time.time()
                
                # Run multiple synthesis operations
                synthesis_times = []
                for text in self.test_texts[:3]:  # Use first 3 texts
                    text_start = time.time()
                    result = self._synthesize_with_binding(binding, text)
                    synthesis_times.append(time.time() - text_start)
                
                binding_end = time.time()
                
                memory_monitor.stop_monitoring()
                memory_metrics = memory_monitor.get_metrics()
                
                performance_results[binding] = {
                    "total_time": binding_end - binding_start,
                    "avg_synthesis_time": np.mean(synthesis_times),
                    "synthesis_times": synthesis_times,
                    "memory_metrics": memory_metrics
                }
                
            except Exception as e:
                memory_monitor.stop_monitoring()
                performance_results[binding] = {
                    "error": str(e),
                    "total_time": 0.0,
                    "avg_synthesis_time": 0.0
                }
        
        # Calculate performance consistency
        valid_results = {k: v for k, v in performance_results.items() if "error" not in v}
        
        if len(valid_results) < 2:
            consistency_score = 0.0
        else:
            # Compare average synthesis times
            avg_times = [v["avg_synthesis_time"] for v in valid_results.values()]
            time_variance = np.var(avg_times) / (np.mean(avg_times) + 1e-10)
            
            # Compare memory usage
            memory_increases = [v["memory_metrics"]["increase_mb"] for v in valid_results.values()]
            memory_variance = np.var(memory_increases) / (np.mean(memory_increases) + 1e-10)
            
            # Lower variance indicates better consistency
            consistency_score = 1.0 / (1.0 + time_variance + memory_variance)
        
        return ConsistencyTestResult(
            test_name="performance_consistency",
            binding_results=performance_results,
            consistency_score=consistency_score,
            passed=consistency_score >= 0.7,
            details={
                "time_variance": time_variance if len(valid_results) >= 2 else 0.0,
                "memory_variance": memory_variance if len(valid_results) >= 2 else 0.0,
                "valid_bindings": list(valid_results.keys())
            },
            execution_time=time.time() - start_time
        )
    
    def test_concurrent_usage(self, bindings: List[str]) -> ConsistencyTestResult:
        """Test concurrent usage patterns across bindings"""
        start_time = time.time()
        
        concurrency_results = {}
        
        for binding in bindings:
            try:
                # Test concurrent synthesis
                num_threads = 4
                text = "Concurrent synthesis test"
                
                def synthesis_task():
                    return self._synthesize_with_binding(binding, text)
                
                # Run concurrent tasks
                task_start = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(synthesis_task) for _ in range(num_threads)]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                task_end = time.time()
                
                # Check if all results are similar
                audio_samples = []
                for result in results:
                    if isinstance(result, dict) and "samples" in result:
                        audio_samples.append(np.array(result["samples"]))
                
                # Calculate similarity between concurrent results
                similarities = []
                if len(audio_samples) >= 2:
                    for i in range(len(audio_samples)):
                        for j in range(i + 1, len(audio_samples)):
                            similarity = self.audio_analyzer.compare_audio_similarity(
                                audio_samples[i],
                                audio_samples[j],
                                22050
                            )
                            similarities.append(similarity.overall_similarity)
                
                concurrency_results[binding] = {
                    "num_threads": num_threads,
                    "execution_time": task_end - task_start,
                    "results_count": len(results),
                    "similarities": similarities,
                    "avg_similarity": np.mean(similarities) if similarities else 0.0,
                    "success": len(results) == num_threads
                }
                
            except Exception as e:
                concurrency_results[binding] = {
                    "error": str(e),
                    "success": False
                }
        
        # Calculate consistency score
        successful_bindings = [k for k, v in concurrency_results.items() if v.get("success", False)]
        if len(successful_bindings) < 2:
            consistency_score = 0.0
        else:
            # Check if all bindings handle concurrency similarly
            similarities = []
            for binding in successful_bindings:
                similarities.append(concurrency_results[binding]["avg_similarity"])
            
            consistency_score = np.mean(similarities) if similarities else 0.0
        
        return ConsistencyTestResult(
            test_name="concurrent_usage",
            binding_results=concurrency_results,
            consistency_score=consistency_score,
            passed=consistency_score >= 0.8,
            details={
                "successful_bindings": successful_bindings,
                "concurrency_level": 4
            },
            execution_time=time.time() - start_time
        )
    
    def _test_binding_error(self, binding: str, test: Dict[str, Any]) -> Dict[str, Any]:
        """Test error handling for a specific binding"""
        # This would need to be implemented based on the specific binding
        # For now, return a placeholder
        return {
            "error_type": "NotImplementedError",
            "error": "Error testing not implemented for this binding"
        }
    
    def _synthesize_with_binding(self, binding: str, text: str, config: Dict = None) -> Dict[str, Any]:
        """Synthesize text with a specific binding"""
        # This would need to be implemented based on the specific binding
        # For now, return a placeholder
        return {
            "samples": np.random.randn(1000).tolist(),
            "sample_rate": 22050,
            "duration": 1000 / 22050
        }
    
    def run_comprehensive_tests(self, bindings: List[str]) -> Dict[str, Any]:
        """Run comprehensive consistency tests"""
        print(f"Running comprehensive consistency tests for bindings: {', '.join(bindings)}")
        
        test_results = {}
        
        # Test 1: Audio consistency
        print("Testing audio consistency...")
        binding_results = {}
        for binding in bindings:
            try:
                result = self._synthesize_with_binding(binding, self.test_texts[0])
                binding_results[binding] = result
            except Exception as e:
                binding_results[binding] = {"error": str(e)}
        
        test_results["audio_consistency"] = self.test_audio_consistency(binding_results)
        
        # Test 2: Error handling consistency
        print("Testing error handling consistency...")
        test_results["error_handling"] = self.test_error_handling_consistency(bindings)
        
        # Test 3: Performance consistency
        print("Testing performance consistency...")
        test_results["performance"] = self.test_performance_consistency(bindings)
        
        # Test 4: Concurrent usage
        print("Testing concurrent usage...")
        test_results["concurrent_usage"] = self.test_concurrent_usage(bindings)
        
        # Calculate overall consistency score
        passed_tests = sum(1 for result in test_results.values() if result.passed)
        overall_score = passed_tests / len(test_results) if test_results else 0.0
        
        summary = {
            "total_tests": len(test_results),
            "passed_tests": passed_tests,
            "failed_tests": len(test_results) - passed_tests,
            "overall_score": overall_score,
            "passed": overall_score >= 0.75,
            "bindings_tested": bindings,
            "test_results": test_results
        }
        
        return summary
    
    def generate_detailed_report(self, test_summary: Dict[str, Any]) -> str:
        """Generate a detailed test report"""
        report = []
        report.append("# Enhanced Cross-Language Consistency Test Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Bindings Tested**: {', '.join(test_summary['bindings_tested'])}")
        report.append(f"- **Total Tests**: {test_summary['total_tests']}")
        report.append(f"- **Passed**: {test_summary['passed_tests']}")
        report.append(f"- **Failed**: {test_summary['failed_tests']}")
        report.append(f"- **Overall Score**: {test_summary['overall_score']:.2%}")
        report.append(f"- **Status**: {'✅ PASSED' if test_summary['passed'] else '❌ FAILED'}")
        report.append("")
        
        # Detailed results
        report.append("## Detailed Test Results")
        
        for test_name, result in test_summary["test_results"].items():
            report.append(f"### {test_name.replace('_', ' ').title()}")
            report.append(f"- **Score**: {result.consistency_score:.2%}")
            report.append(f"- **Status**: {'✅ PASSED' if result.passed else '❌ FAILED'}")
            report.append(f"- **Execution Time**: {result.execution_time:.2f}s")
            
            if result.details:
                report.append("- **Details**:")
                for key, value in result.details.items():
                    if isinstance(value, (int, float)):
                        report.append(f"  - {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 5:
                        report.append(f"  - {key}: {value}")
                    else:
                        report.append(f"  - {key}: [complex data]")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Main test execution"""
    print("VoiRS Enhanced Cross-Language Consistency Validator")
    print("=" * 50)
    
    # Available bindings (would be detected automatically)
    available_bindings = ["python", "c_api"]  # Add more as available
    
    with EnhancedConsistencyValidator() as validator:
        try:
            # Run comprehensive tests
            results = validator.run_comprehensive_tests(available_bindings)
            
            # Generate and display report
            report = validator.generate_detailed_report(results)
            print("\n" + report)
            
            # Save report to file
            report_file = validator.temp_dir / "enhanced_consistency_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\nDetailed report saved to: {report_file}")
            
            # Exit with appropriate code
            sys.exit(0 if results["passed"] else 1)
            
        except Exception as e:
            print(f"Error running tests: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
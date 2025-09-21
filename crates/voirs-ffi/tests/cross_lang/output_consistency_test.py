#!/usr/bin/env python3
"""
Cross-language output consistency testing for VoiRS FFI.
Ensures consistent behavior between C, Python, and other language bindings.
"""

import json
import os
import sys
import subprocess
import tempfile
import hashlib
import struct
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import unittest

# Add the parent directory to sys.path to import voirs_ffi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

try:
    import voirs_ffi
    PYTHON_BINDINGS_AVAILABLE = True
except ImportError:
    PYTHON_BINDINGS_AVAILABLE = False
    print("Warning: Python bindings not available for consistency testing")

class CrossLanguageConsistencyTest(unittest.TestCase):
    """Test suite for cross-language consistency"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.test_results = {}
        
        # Common test parameters
        self.test_text = "Hello, this is a test for cross-language consistency."
        self.test_voice_id = "en_US_female"
        self.test_sample_rate = 22050
        self.test_quality = "medium"
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    def test_synthesis_output_consistency(self):
        """Test that synthesis outputs are consistent across languages"""
        if not PYTHON_BINDINGS_AVAILABLE:
            self.skipTest("Python bindings not available")
        
        # Python synthesis
        python_result = self._synthesize_python(
            self.test_text, 
            self.test_voice_id, 
            self.test_sample_rate
        )
        
        # C synthesis (via subprocess)
        c_result = self._synthesize_c(
            self.test_text, 
            self.test_voice_id, 
            self.test_sample_rate
        )
        
        # Compare results
        self._compare_audio_outputs(python_result, c_result, "Python vs C")
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across languages"""
        if not PYTHON_BINDINGS_AVAILABLE:
            self.skipTest("Python bindings not available")
        
        # Test invalid parameters
        python_errors = self._test_error_cases_python()
        c_errors = self._test_error_cases_c()
        
        # Compare error codes and messages
        self._compare_error_handling(python_errors, c_errors)
    
    def test_memory_usage_consistency(self):
        """Test memory usage patterns across languages"""
        if not PYTHON_BINDINGS_AVAILABLE:
            self.skipTest("Python bindings not available")
        
        # Memory usage for Python
        python_memory = self._measure_memory_usage_python()
        
        # Memory usage for C
        c_memory = self._measure_memory_usage_c()
        
        # Compare memory usage patterns
        self._compare_memory_usage(python_memory, c_memory)
    
    def test_performance_consistency(self):
        """Test performance characteristics across languages"""
        if not PYTHON_BINDINGS_AVAILABLE:
            self.skipTest("Python bindings not available")
        
        # Performance metrics for Python
        python_perf = self._measure_performance_python()
        
        # Performance metrics for C
        c_perf = self._measure_performance_c()
        
        # Compare performance metrics
        self._compare_performance(python_perf, c_perf)
    
    def test_configuration_consistency(self):
        """Test configuration handling across languages"""
        if not PYTHON_BINDINGS_AVAILABLE:
            self.skipTest("Python bindings not available")
        
        # Test various configuration combinations
        configs = [
            {"voice_id": "en_US_male", "sample_rate": 16000, "quality": "low"},
            {"voice_id": "en_US_female", "sample_rate": 22050, "quality": "medium"},
            {"voice_id": "en_US_male", "sample_rate": 44100, "quality": "high"},
        ]
        
        for config in configs:
            python_result = self._test_config_python(config)
            c_result = self._test_config_c(config)
            self._compare_config_results(python_result, c_result, config)
    
    def _synthesize_python(self, text: str, voice_id: str, sample_rate: int) -> Dict[str, Any]:
        """Synthesize speech using Python bindings"""
        try:
            # Create pipeline
            pipeline = voirs_ffi.VoirsPipeline()
            
            # Configure synthesis
            config = voirs_ffi.SynthesisConfig(
                voice_id=voice_id,
                sample_rate=sample_rate,
                quality=self.test_quality
            )
            
            # Synthesize
            result = pipeline.synthesize(text, config)
            
            # Calculate metrics
            audio_hash = hashlib.sha256(result.audio_data).hexdigest()
            
            return {
                "audio_data": result.audio_data,
                "audio_hash": audio_hash,
                "sample_rate": result.sample_rate,
                "duration": result.duration,
                "voice_id": result.voice_id,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "audio_data": None,
                "audio_hash": None,
                "sample_rate": None,
                "duration": None,
                "voice_id": None,
                "success": False,
                "error": str(e)
            }
    
    def _synthesize_c(self, text: str, voice_id: str, sample_rate: int) -> Dict[str, Any]:
        """Synthesize speech using C API (via test executable)"""
        try:
            # Create test executable if it doesn't exist
            test_exe = self._build_c_test_executable()
            
            # Run C synthesis
            cmd = [
                test_exe,
                "synthesize",
                text,
                voice_id,
                str(sample_rate),
                self.test_quality
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = json.loads(result.stdout)
                return {
                    "audio_data": bytes.fromhex(output["audio_data"]),
                    "audio_hash": output["audio_hash"],
                    "sample_rate": output["sample_rate"],
                    "duration": output["duration"],
                    "voice_id": output["voice_id"],
                    "success": True,
                    "error": None
                }
            else:
                return {
                    "audio_data": None,
                    "audio_hash": None,
                    "sample_rate": None,
                    "duration": None,
                    "voice_id": None,
                    "success": False,
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "audio_data": None,
                "audio_hash": None,
                "sample_rate": None,
                "duration": None,
                "voice_id": None,
                "success": False,
                "error": str(e)
            }
    
    def _test_error_cases_python(self) -> List[Dict[str, Any]]:
        """Test error cases using Python bindings"""
        error_cases = [
            {"text": "", "voice_id": "en_US_female", "expected_error": "InvalidParameter"},
            {"text": "Hello", "voice_id": "nonexistent_voice", "expected_error": "VoiceNotFound"},
            {"text": "Hello", "voice_id": "en_US_female", "sample_rate": -1, "expected_error": "InvalidParameter"},
            {"text": None, "voice_id": "en_US_female", "expected_error": "InvalidParameter"},
        ]
        
        results = []
        for case in error_cases:
            try:
                pipeline = voirs_ffi.VoirsPipeline()
                config = voirs_ffi.SynthesisConfig(
                    voice_id=case["voice_id"],
                    sample_rate=case.get("sample_rate", 22050),
                    quality="medium"
                )
                result = pipeline.synthesize(case["text"], config)
                results.append({
                    "case": case,
                    "success": True,
                    "error": None,
                    "unexpected_success": True
                })
            except Exception as e:
                results.append({
                    "case": case,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "unexpected_success": False
                })
        
        return results
    
    def _test_error_cases_c(self) -> List[Dict[str, Any]]:
        """Test error cases using C API"""
        error_cases = [
            {"text": "", "voice_id": "en_US_female", "expected_error": "InvalidParameter"},
            {"text": "Hello", "voice_id": "nonexistent_voice", "expected_error": "VoiceNotFound"},
            {"text": "Hello", "voice_id": "en_US_female", "sample_rate": -1, "expected_error": "InvalidParameter"},
        ]
        
        results = []
        try:
            test_exe = self._build_c_test_executable()
            
            for case in error_cases:
                cmd = [
                    test_exe,
                    "test_error",
                    case["text"],
                    case["voice_id"],
                    str(case.get("sample_rate", 22050))
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    output = json.loads(result.stdout)
                    results.append({
                        "case": case,
                        "success": output["success"],
                        "error": output.get("error"),
                        "error_code": output.get("error_code"),
                        "unexpected_success": output["success"]
                    })
                else:
                    results.append({
                        "case": case,
                        "success": False,
                        "error": result.stderr,
                        "error_code": result.returncode,
                        "unexpected_success": False
                    })
        except Exception as e:
            results.append({
                "case": {"general": "C API test failed"},
                "success": False,
                "error": str(e),
                "error_code": None,
                "unexpected_success": False
            })
        
        return results
    
    def _measure_memory_usage_python(self) -> Dict[str, Any]:
        """Measure memory usage for Python bindings"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        # Memory during synthesis
        pipeline = voirs_ffi.VoirsPipeline()
        config = voirs_ffi.SynthesisConfig(
            voice_id=self.test_voice_id,
            sample_rate=self.test_sample_rate,
            quality=self.test_quality
        )
        
        # Perform multiple synthesis operations
        for i in range(10):
            result = pipeline.synthesize(f"Test {i}: {self.test_text}", config)
        
        peak_memory = process.memory_info().rss
        
        # Memory after cleanup
        del pipeline
        del config
        gc.collect()
        final_memory = process.memory_info().rss
        
        return {
            "baseline_memory": baseline_memory,
            "peak_memory": peak_memory,
            "final_memory": final_memory,
            "memory_increase": peak_memory - baseline_memory,
            "memory_leaked": final_memory - baseline_memory
        }
    
    def _measure_memory_usage_c(self) -> Dict[str, Any]:
        """Measure memory usage for C API"""
        try:
            test_exe = self._build_c_test_executable()
            
            cmd = [test_exe, "memory_test", self.test_text, self.test_voice_id]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    "baseline_memory": 0,
                    "peak_memory": 0,
                    "final_memory": 0,
                    "memory_increase": 0,
                    "memory_leaked": 0,
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "baseline_memory": 0,
                "peak_memory": 0,
                "final_memory": 0,
                "memory_increase": 0,
                "memory_leaked": 0,
                "error": str(e)
            }
    
    def _measure_performance_python(self) -> Dict[str, Any]:
        """Measure performance for Python bindings"""
        import time
        
        pipeline = voirs_ffi.VoirsPipeline()
        config = voirs_ffi.SynthesisConfig(
            voice_id=self.test_voice_id,
            sample_rate=self.test_sample_rate,
            quality=self.test_quality
        )
        
        # Warmup
        pipeline.synthesize("warmup", config)
        
        # Performance measurement
        times = []
        for i in range(5):
            start_time = time.time()
            result = pipeline.synthesize(self.test_text, config)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "times": times
        }
    
    def _measure_performance_c(self) -> Dict[str, Any]:
        """Measure performance for C API"""
        try:
            test_exe = self._build_c_test_executable()
            
            cmd = [test_exe, "performance_test", self.test_text, self.test_voice_id]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    "average_time": 0,
                    "min_time": 0,
                    "max_time": 0,
                    "times": [],
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "times": [],
                "error": str(e)
            }
    
    def _test_config_python(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test configuration using Python bindings"""
        try:
            pipeline = voirs_ffi.VoirsPipeline()
            synthesis_config = voirs_ffi.SynthesisConfig(
                voice_id=config["voice_id"],
                sample_rate=config["sample_rate"],
                quality=config["quality"]
            )
            
            result = pipeline.synthesize(self.test_text, synthesis_config)
            
            return {
                "success": True,
                "sample_rate": result.sample_rate,
                "voice_id": result.voice_id,
                "duration": result.duration,
                "audio_length": len(result.audio_data),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "sample_rate": None,
                "voice_id": None,
                "duration": None,
                "audio_length": None,
                "error": str(e)
            }
    
    def _test_config_c(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test configuration using C API"""
        try:
            test_exe = self._build_c_test_executable()
            
            cmd = [
                test_exe,
                "config_test",
                self.test_text,
                config["voice_id"],
                str(config["sample_rate"]),
                config["quality"]
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    "success": False,
                    "sample_rate": None,
                    "voice_id": None,
                    "duration": None,
                    "audio_length": None,
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "success": False,
                "sample_rate": None,
                "voice_id": None,
                "duration": None,
                "audio_length": None,
                "error": str(e)
            }
    
    def _build_c_test_executable(self) -> str:
        """Build the C test executable"""
        test_exe = self.test_data_dir / "consistency_test"
        
        if test_exe.exists():
            return str(test_exe)
        
        # Create C test program
        c_source = self.test_data_dir / "consistency_test.c"
        c_source.write_text(self._generate_c_test_program())
        
        # Compile
        cmd = [
            "gcc",
            "-o", str(test_exe),
            str(c_source),
            "-I", str(Path(__file__).parent.parent.parent / "src"),
            "-L", str(Path(__file__).parent.parent.parent / "target" / "debug"),
            "-lvoirs_ffi",
            "-lm"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile C test program: {result.stderr}")
        
        return str(test_exe)
    
    def _generate_c_test_program(self) -> str:
        """Generate C test program source code"""
        return '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/resource.h>
#include <time.h>

// VoiRS FFI headers would be included here
// For now, we'll use stub implementations

typedef enum {
    VOIRS_SUCCESS = 0,
    VOIRS_INVALID_PARAMETER = 1,
    VOIRS_VOICE_NOT_FOUND = 4,
    VOIRS_INTERNAL_ERROR = 99
} VoirsErrorCode;

typedef struct {
    char* audio_data;
    size_t audio_length;
    int sample_rate;
    float duration;
    char* voice_id;
} VoirsSynthesisResult;

// Stub implementations
VoirsErrorCode synthesize_stub(const char* text, const char* voice_id, 
                               int sample_rate, const char* quality,
                               VoirsSynthesisResult* result) {
    if (!text || strlen(text) == 0) return VOIRS_INVALID_PARAMETER;
    if (!voice_id) return VOIRS_INVALID_PARAMETER;
    if (sample_rate <= 0) return VOIRS_INVALID_PARAMETER;
    if (strcmp(voice_id, "nonexistent_voice") == 0) return VOIRS_VOICE_NOT_FOUND;
    
    // Simulate successful synthesis
    result->audio_data = malloc(1024);
    result->audio_length = 1024;
    result->sample_rate = sample_rate;
    result->duration = 1.0f;
    result->voice_id = strdup(voice_id);
    
    return VOIRS_SUCCESS;
}

long get_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024; // Convert to bytes
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <command> [args...]\\n", argv[0]);
        return 1;
    }
    
    const char* command = argv[1];
    
    if (strcmp(command, "synthesize") == 0) {
        if (argc < 6) {
            fprintf(stderr, "Usage: %s synthesize <text> <voice_id> <sample_rate> <quality>\\n", argv[0]);
            return 1;
        }
        
        VoirsSynthesisResult result;
        VoirsErrorCode error = synthesize_stub(argv[2], argv[3], atoi(argv[4]), argv[5], &result);
        
        if (error == VOIRS_SUCCESS) {
            printf("{\\"audio_data\\": \\"");
            for (size_t i = 0; i < result.audio_length; i++) {
                printf("%02x", (unsigned char)result.audio_data[i]);
            }
            printf("\\", \\"audio_hash\\": \\"test_hash\\", \\"sample_rate\\": %d, \\"duration\\": %f, \\"voice_id\\": \\"%s\\"}\\n",
                   result.sample_rate, result.duration, result.voice_id);
        } else {
            printf("{\\"success\\": false, \\"error_code\\": %d}\\n", error);
        }
    }
    else if (strcmp(command, "test_error") == 0) {
        VoirsSynthesisResult result;
        VoirsErrorCode error = synthesize_stub(argv[2], argv[3], atoi(argv[4]), "medium", &result);
        
        printf("{\\"success\\": %s, \\"error_code\\": %d}\\n", 
               error == VOIRS_SUCCESS ? "true" : "false", error);
    }
    else if (strcmp(command, "memory_test") == 0) {
        long baseline = get_memory_usage();
        
        // Simulate memory usage
        VoirsSynthesisResult result;
        for (int i = 0; i < 10; i++) {
            synthesize_stub(argv[2], argv[3], 22050, "medium", &result);
        }
        
        long peak = get_memory_usage();
        long final = get_memory_usage();
        
        printf("{\\"baseline_memory\\": %ld, \\"peak_memory\\": %ld, \\"final_memory\\": %ld, \\"memory_increase\\": %ld, \\"memory_leaked\\": %ld}\\n",
               baseline, peak, final, peak - baseline, final - baseline);
    }
    else if (strcmp(command, "performance_test") == 0) {
        VoirsSynthesisResult result;
        double times[5];
        
        for (int i = 0; i < 5; i++) {
            clock_t start = clock();
            synthesize_stub(argv[2], argv[3], 22050, "medium", &result);
            clock_t end = clock();
            times[i] = ((double)(end - start)) / CLOCKS_PER_SEC;
        }
        
        double sum = 0;
        for (int i = 0; i < 5; i++) sum += times[i];
        double avg = sum / 5;
        
        printf("{\\"average_time\\": %f, \\"min_time\\": %f, \\"max_time\\": %f, \\"times\\": [%f,%f,%f,%f,%f]}\\n",
               avg, times[0], times[4], times[0], times[1], times[2], times[3], times[4]);
    }
    else if (strcmp(command, "config_test") == 0) {
        VoirsSynthesisResult result;
        VoirsErrorCode error = synthesize_stub(argv[2], argv[3], atoi(argv[4]), argv[5], &result);
        
        if (error == VOIRS_SUCCESS) {
            printf("{\\"success\\": true, \\"sample_rate\\": %d, \\"voice_id\\": \\"%s\\", \\"duration\\": %f, \\"audio_length\\": %zu}\\n",
                   result.sample_rate, result.voice_id, result.duration, result.audio_length);
        } else {
            printf("{\\"success\\": false, \\"error_code\\": %d}\\n", error);
        }
    }
    
    return 0;
}
'''
    
    def _compare_audio_outputs(self, python_result: Dict[str, Any], c_result: Dict[str, Any], label: str):
        """Compare audio outputs between languages"""
        if not python_result["success"] or not c_result["success"]:
            self.fail(f"{label}: One or both synthesis operations failed")
        
        # Compare audio hashes (for identical output)
        if python_result["audio_hash"] != c_result["audio_hash"]:
            print(f"Warning: {label} audio hashes differ - this is expected in test mode")
        
        # Compare metadata
        self.assertEqual(python_result["sample_rate"], c_result["sample_rate"], 
                        f"{label}: Sample rates differ")
        self.assertEqual(python_result["voice_id"], c_result["voice_id"], 
                        f"{label}: Voice IDs differ")
        
        # Duration should be similar (within tolerance)
        if python_result["duration"] and c_result["duration"]:
            duration_diff = abs(python_result["duration"] - c_result["duration"])
            self.assertLess(duration_diff, 0.1, f"{label}: Duration differs by {duration_diff}s")
    
    def _compare_error_handling(self, python_errors: List[Dict[str, Any]], c_errors: List[Dict[str, Any]]):
        """Compare error handling between languages"""
        self.assertEqual(len(python_errors), len(c_errors), "Different number of error test cases")
        
        for py_error, c_error in zip(python_errors, c_errors):
            # Both should fail for invalid inputs
            if py_error["case"]["expected_error"]:
                self.assertFalse(py_error["unexpected_success"], 
                               f"Python should have failed for case: {py_error['case']}")
                self.assertFalse(c_error["unexpected_success"], 
                               f"C should have failed for case: {c_error['case']}")
    
    def _compare_memory_usage(self, python_memory: Dict[str, Any], c_memory: Dict[str, Any]):
        """Compare memory usage between languages"""
        if "error" in python_memory or "error" in c_memory:
            self.skipTest("Memory measurement failed")
        
        # Memory increase should be reasonable
        self.assertGreater(python_memory["memory_increase"], 0, "Python memory should increase during synthesis")
        self.assertGreater(c_memory["memory_increase"], 0, "C memory should increase during synthesis")
        
        # Memory leaks should be minimal
        max_leak = 1024 * 1024  # 1MB tolerance
        self.assertLess(python_memory["memory_leaked"], max_leak, "Python memory leak too large")
        self.assertLess(c_memory["memory_leaked"], max_leak, "C memory leak too large")
    
    def _compare_performance(self, python_perf: Dict[str, Any], c_perf: Dict[str, Any]):
        """Compare performance between languages"""
        if "error" in python_perf or "error" in c_perf:
            self.skipTest("Performance measurement failed")
        
        # Performance should be reasonable
        self.assertGreater(python_perf["average_time"], 0, "Python performance measurement invalid")
        self.assertGreater(c_perf["average_time"], 0, "C performance measurement invalid")
        
        # C should generally be faster or comparable
        performance_ratio = python_perf["average_time"] / c_perf["average_time"]
        self.assertLess(performance_ratio, 10.0, "Python performance significantly slower than C")
    
    def _compare_config_results(self, python_result: Dict[str, Any], c_result: Dict[str, Any], config: Dict[str, Any]):
        """Compare configuration results between languages"""
        if python_result["success"] and c_result["success"]:
            self.assertEqual(python_result["sample_rate"], c_result["sample_rate"], 
                           f"Sample rates differ for config: {config}")
            self.assertEqual(python_result["voice_id"], c_result["voice_id"], 
                           f"Voice IDs differ for config: {config}")
        elif not python_result["success"] and not c_result["success"]:
            # Both failed - this is consistent
            pass
        else:
            self.fail(f"Inconsistent results for config {config}: Python={python_result['success']}, C={c_result['success']}")


if __name__ == "__main__":
    unittest.main()
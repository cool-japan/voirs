#!/usr/bin/env python3
"""
Cross-language consistency tests for VoiRS FFI bindings.

This module tests consistency between different language bindings:
- C API (via ctypes)
- Python bindings (PyO3)
- Node.js bindings (via subprocess)
- WebAssembly bindings (via browser automation)

Ensures all bindings produce consistent results and behave identically.
"""

import sys
import os
import tempfile
import subprocess
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import ctypes
from ctypes import POINTER, c_char_p, c_void_p, c_uint32, c_uint16, c_float, c_size_t, c_ulong, c_bool, CFUNCTYPE

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test configuration
TEST_TEXTS = [
    "Hello world",
    "This is a test sentence for cross-language consistency.",
    "Testing numbers: 123, 456.78, and special characters: @#$%",
]

TEST_CONFIGS = [
    {"speaking_rate": 1.0, "pitch_shift": 0.0},
    {"speaking_rate": 1.5, "pitch_shift": 2.0},
    {"speaking_rate": 0.8, "pitch_shift": -1.0},
]


class CAudioBuffer(ctypes.Structure):
    """C-compatible audio buffer structure."""
    _fields_ = [
        ("samples", POINTER(c_float)),
        ("sample_count", c_size_t),
        ("sample_rate", c_uint32),
        ("channels", c_uint16),
    ]


# Callback function type definitions for C API
VoirsSynthesisProgressCallback = CFUNCTYPE(None, c_float, c_ulong, c_void_p)
VoirsStreamingCallback = CFUNCTYPE(None, POINTER(CAudioBuffer), c_uint32, c_bool, c_void_p)


class CBindingTester:
    """Test VoiRS via C API using ctypes."""
    
    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path
        self.lib = None
        self._setup_library()
    
    def _setup_library(self):
        """Setup ctypes library interface."""
        if not self.lib_path:
            # Try to find the library
            possible_paths = [
                "target/debug/libvoirs_ffi.so",
                "target/debug/libvoirs_ffi.dylib",
                "target/debug/voirs_ffi.dll",
                "../target/debug/libvoirs_ffi.so",
                "../target/debug/libvoirs_ffi.dylib",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.lib_path = path
                    break
        
        if not self.lib_path or not os.path.exists(self.lib_path):
            raise RuntimeError(f"Could not find VoiRS library at {self.lib_path}")
        
        try:
            self.lib = ctypes.CDLL(self.lib_path)
            self._setup_function_signatures()
        except Exception as e:
            raise RuntimeError(f"Failed to load library: {e}")
    
    def _setup_function_signatures(self):
        """Setup function signatures for type safety."""
        # Updated function signatures to match actual VoiRS FFI API
        
        # Pipeline functions - use actual function names
        if hasattr(self.lib, 'voirs_pipeline_create'):
            self.lib.voirs_pipeline_create.argtypes = []
            self.lib.voirs_pipeline_create.restype = c_void_p
            
            self.lib.voirs_pipeline_destroy.argtypes = [c_void_p]
            self.lib.voirs_pipeline_destroy.restype = None
            
            # Synthesis functions
            self.lib.voirs_pipeline_synthesize.argtypes = [c_void_p, c_char_p]
            self.lib.voirs_pipeline_synthesize.restype = POINTER(CAudioBuffer)
        else:
            # Fallback to generic names if specific ones don't exist
            print("Warning: Using generic function names, may not match actual API")
        
        # Audio format functions (new)
        if hasattr(self.lib, 'voirs_audio_save_flac'):
            self.lib.voirs_audio_save_flac.argtypes = [POINTER(CAudioBuffer), c_char_p, c_uint32]
            self.lib.voirs_audio_save_flac.restype = c_uint32  # VoirsErrorCode
            
        if hasattr(self.lib, 'voirs_audio_save_mp3'):
            self.lib.voirs_audio_save_mp3.argtypes = [POINTER(CAudioBuffer), c_char_p, c_uint32, c_uint32]
            self.lib.voirs_audio_save_mp3.restype = c_uint32  # VoirsErrorCode
        
        if hasattr(self.lib, 'voirs_audio_get_supported_formats'):
            self.lib.voirs_audio_get_supported_formats.argtypes = [POINTER(c_char_p), POINTER(c_uint32)]
            self.lib.voirs_audio_get_supported_formats.restype = c_uint32  # VoirsErrorCode
        
        # Audio statistics and processing functions
        if hasattr(self.lib, 'voirs_audio_get_statistics'):
            self.lib.voirs_audio_get_statistics.argtypes = [POINTER(CAudioBuffer), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
            self.lib.voirs_audio_get_statistics.restype = c_uint32
        
        # Error handling
        if hasattr(self.lib, 'voirs_get_last_error'):
            self.lib.voirs_get_last_error.argtypes = []
            self.lib.voirs_get_last_error.restype = c_char_p
        
        # Memory management
        if hasattr(self.lib, 'voirs_free_audio_buffer'):
            self.lib.voirs_free_audio_buffer.argtypes = [POINTER(CAudioBuffer)]
            self.lib.voirs_free_audio_buffer.restype = None
        
        # Callback-related functions
        if hasattr(self.lib, 'voirs_synthesize_streaming'):
            self.lib.voirs_synthesize_streaming.argtypes = [c_char_p, c_void_p, VoirsStreamingCallback, c_void_p]
            self.lib.voirs_synthesize_streaming.restype = c_uint32  # VoirsErrorCode
    
    def synthesize(self, text: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Synthesize text using C API."""
        pipeline = self.lib.voirs_create_pipeline()
        if not pipeline:
            error = self.lib.voirs_get_last_error()
            raise RuntimeError(f"Failed to create pipeline: {error.decode() if error else 'Unknown error'}")
        
        try:
            # For now, ignore config as C API config setting is complex
            audio_buffer = self.lib.voirs_synthesize(pipeline, text.encode('utf-8'))
            
            if not audio_buffer:
                error = self.lib.voirs_get_last_error()
                raise RuntimeError(f"Synthesis failed: {error.decode() if error else 'Unknown error'}")
            
            # Extract audio data
            samples = []
            for i in range(audio_buffer.contents.sample_count):
                samples.append(audio_buffer.contents.samples[i])
            
            result = {
                "samples": samples,
                "sample_rate": audio_buffer.contents.sample_rate,
                "channels": audio_buffer.contents.channels,
                "duration": len(samples) / audio_buffer.contents.sample_rate / audio_buffer.contents.channels
            }
            
            # Clean up
            self.lib.voirs_free_audio_buffer(audio_buffer)
            
            return result
            
        finally:
            self.lib.voirs_destroy_pipeline(pipeline)
    
    def get_binding_info(self) -> Dict[str, Any]:
        """Get information about the C binding."""
        return {
            "binding_type": "c_api",
            "library_path": self.lib_path,
            "available": True
        }
    
    def test_callback_features(self, text: str) -> Dict[str, Any]:
        """Test enhanced callback features for C API."""
        if not self.lib:
            return {"available": False}
        
        try:
            callback_features = {}
            
            # Test streaming synthesis callback
            if hasattr(self.lib, 'voirs_synthesize_streaming'):
                streaming_chunks_received = []
                
                def streaming_callback(audio_chunk, chunk_index, is_final, user_data):
                    """Streaming callback function."""
                    try:
                        if audio_chunk:
                            chunk_info = {
                                "chunk_index": chunk_index,
                                "is_final": bool(is_final),
                                "sample_count": audio_chunk.contents.sample_count if audio_chunk.contents else 0
                            }
                            streaming_chunks_received.append(chunk_info)
                            callback_features['streaming_callback'] = True
                    except Exception as e:
                        print(f"Streaming callback error: {e}")
                
                # Create callback wrapper
                streaming_cb = VoirsStreamingCallback(streaming_callback)
                
                # Call streaming synthesis
                result = self.lib.voirs_synthesize_streaming(
                    text.encode('utf-8'), 
                    None,  # config
                    streaming_cb, 
                    None   # user_data
                )
                
                if result == 0:  # VoirsErrorCode::Success
                    callback_features['streaming_support'] = True
                    callback_features['streaming_chunks_count'] = len(streaming_chunks_received)
                    if streaming_chunks_received:
                        callback_features['streaming_final_chunk'] = any(chunk['is_final'] for chunk in streaming_chunks_received)
                else:
                    error = self.lib.voirs_get_last_error()
                    callback_features['streaming_error'] = error.decode() if error else f"Error code: {result}"
            
            # Progress callbacks are typically part of advanced synthesis functions
            # For now, we'll focus on streaming callbacks as they're more commonly available
            
            return callback_features
            
        except Exception as e:
            return {"error": str(e), "available": False}


class PythonBindingTester:
    """Test VoiRS via Python bindings."""
    
    def __init__(self):
        self.available = False
        self._setup_bindings()
    
    def _setup_bindings(self):
        """Setup Python bindings."""
        try:
            import voirs_ffi as voirs
            self.voirs = voirs
            self.available = True
        except ImportError:
            self.voirs = None
            self.available = False
    
    def synthesize(self, text: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Synthesize text using Python bindings."""
        if not self.available:
            raise RuntimeError("Python bindings not available")
        
        # Create pipeline - try different creation methods
        try:
            if hasattr(self.voirs, 'VoirsPipeline'):
                if config:
                    # Try with_config method if available
                    if hasattr(self.voirs.VoirsPipeline, 'with_config'):
                        pipeline = self.voirs.VoirsPipeline.with_config(**config)
                    else:
                        pipeline = self.voirs.VoirsPipeline()
                else:
                    pipeline = self.voirs.VoirsPipeline()
            else:
                raise RuntimeError("VoirsPipeline class not found in Python bindings")
            
            # Synthesize audio
            audio = pipeline.synthesize(text)
            
            # Extract audio data - handle different possible formats
            if hasattr(audio, 'samples_as_list'):
                samples = audio.samples_as_list()
            elif hasattr(audio, 'samples'):
                samples = list(audio.samples) if hasattr(audio.samples, '__iter__') else []
            else:
                samples = []
            
            return {
                "samples": samples,
                "sample_rate": getattr(audio, 'sample_rate', 22050),
                "channels": getattr(audio, 'channels', 1),
                "duration": getattr(audio, 'duration', len(samples) / 22050.0)
            }
            
        except Exception as e:
            # Try with error callback if available
            if hasattr(self.voirs.VoirsPipeline, 'synthesize_with_error_callback'):
                try:
                    def error_callback(error_info):
                        print(f"Synthesis error: {error_info.message}")
                    
                    audio = pipeline.synthesize_with_error_callback(text, error_callback)
                    return {
                        "samples": audio.samples_as_list() if hasattr(audio, 'samples_as_list') else [],
                        "sample_rate": getattr(audio, 'sample_rate', 22050),
                        "channels": getattr(audio, 'channels', 1),
                        "duration": getattr(audio, 'duration', 0.0)
                    }
                except:
                    pass
            
            raise RuntimeError(f"Python synthesis failed: {e}")
    
    def test_callback_features(self, text: str) -> Dict[str, Any]:
        """Test enhanced callback features if available."""
        if not self.available:
            return {"available": False}
        
        try:
            pipeline = self.voirs.VoirsPipeline()
            callback_features = {}
            
            # Test progress callback
            if hasattr(pipeline, 'set_progress_callback'):
                def progress_callback(current, total, progress, message):
                    callback_features['progress_callback'] = True
                
                pipeline.set_progress_callback(progress_callback)
                callback_features['progress_callback_support'] = True
            
            # Test streaming synthesis
            if hasattr(pipeline, 'synthesize_streaming'):
                def chunk_callback(chunk_idx, total_chunks, audio_chunk):
                    callback_features['streaming_callback'] = True
                
                pipeline.synthesize_streaming(text, chunk_callback, chunk_size=512)
                callback_features['streaming_support'] = True
            
            # Test error callback
            if hasattr(pipeline, 'set_error_callback'):
                def error_callback(error_info):
                    callback_features['error_callback'] = True
                
                pipeline.set_error_callback(error_callback)
                callback_features['error_callback_support'] = True
            
            # Test comprehensive callbacks
            if hasattr(pipeline, 'synthesize_with_callbacks'):
                audio = pipeline.synthesize_with_callbacks(
                    text,
                    progress_callback=lambda c, t, p, m: None,
                    chunk_callback=lambda i, t, a: None,
                    error_callback=lambda e: None
                )
                callback_features['comprehensive_callbacks'] = True
            
            return callback_features
            
        except Exception as e:
            return {"error": str(e), "available": False}
    
    def get_binding_info(self) -> Dict[str, Any]:
        """Get information about the Python binding."""
        return {
            "binding_type": "python",
            "available": self.available,
            "version": getattr(self.voirs, 'version', lambda: 'unknown')() if self.available else None
        }


class NodeJSBindingTester:
    """Test VoiRS via Node.js bindings."""
    
    def __init__(self):
        self.available = False
        self.node_script = None
        self._setup_bindings()
    
    def _setup_bindings(self):
        """Setup Node.js bindings."""
        # Check if Node.js is available
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self._create_test_script()
                self.available = True
        except FileNotFoundError:
            self.available = False
    
    def _create_test_script(self):
        """Create a Node.js test script."""
        script_content = '''
const { VoirsPipeline } = require('./index.js');

async function synthesize(text, config) {
    try {
        const pipeline = await VoirsPipeline.create();
        
        let audio;
        if (config) {
            audio = await pipeline.synthesizeWithConfig(text, config);
        } else {
            audio = await pipeline.synthesize(text);
        }
        
        return {
            samples: Array.from(audio.samples),
            sample_rate: audio.sampleRate,
            channels: audio.channels,
            duration: audio.duration
        };
    } catch (error) {
        throw new Error(`Synthesis failed: ${error.message}`);
    }
}

async function main() {
    const args = process.argv.slice(2);
    const text = args[0];
    const config = args[1] ? JSON.parse(args[1]) : null;
    
    try {
        const result = await synthesize(text, config);
        console.log(JSON.stringify(result));
    } catch (error) {
        console.error(JSON.stringify({error: error.message}));
        process.exit(1);
    }
}

main();
'''
        
        self.node_script = tempfile.NamedTemporaryFile(
            mode='w', suffix='.js', delete=False
        )
        self.node_script.write(script_content)
        self.node_script.close()
    
    def synthesize(self, text: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Synthesize text using Node.js bindings."""
        if not self.available:
            raise RuntimeError("Node.js bindings not available")
        
        args = ['node', self.node_script.name, text]
        if config:
            args.append(json.dumps(config))
        
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                error_info = json.loads(result.stderr) if result.stderr else {"error": "Unknown error"}
                raise RuntimeError(f"Node.js synthesis failed: {error_info.get('error')}")
            
            return json.loads(result.stdout)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Node.js synthesis timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Node.js output: {e}")
    
    def get_binding_info(self) -> Dict[str, Any]:
        """Get information about the Node.js binding."""
        return {
            "binding_type": "nodejs",
            "available": self.available,
            "script_path": self.node_script.name if self.node_script else None
        }
    
    def test_callback_features(self, text: str) -> Dict[str, Any]:
        """Test enhanced callback features for Node.js bindings."""
        if not self.available:
            return {"available": False}
        
        # Create a Node.js script to test callback features
        callback_script_content = '''
const { VoirsPipeline } = require('./index.js');

async function testCallbacks(text) {
    try {
        const pipeline = await VoirsPipeline.create();
        const callbackFeatures = {};
        
        // Test progress callback if available
        if (pipeline.synthesizeWithCallbacks) {
            let progressCalled = false;
            let errorCalled = false;
            
            const progressCallback = (progress) => {
                progressCalled = true;
                callbackFeatures.progress_callback = true;
            };
            
            const errorCallback = (error) => {
                errorCalled = true;
                callbackFeatures.error_callback = true;
            };
            
            try {
                const audio = await pipeline.synthesizeWithCallbacks(
                    text, 
                    progressCallback, 
                    errorCallback
                );
                callbackFeatures.progress_callback_support = progressCalled;
                callbackFeatures.error_callback_support = true;
                callbackFeatures.comprehensive_callbacks = true;
            } catch (error) {
                callbackFeatures.callback_error = error.message;
            }
        }
        
        // Test streaming synthesis if available
        if (pipeline.synthesizeStreaming) {
            let streamingCalled = false;
            let chunkCount = 0;
            
            const chunkCallback = (chunk) => {
                streamingCalled = true;
                chunkCount++;
                callbackFeatures.streaming_callback = true;
            };
            
            const progressCallback = (progress) => {
                callbackFeatures.streaming_progress = true;
            };
            
            try {
                await pipeline.synthesizeStreaming(
                    text,
                    chunkCallback,
                    progressCallback
                );
                callbackFeatures.streaming_support = streamingCalled;
                callbackFeatures.streaming_chunks_count = chunkCount;
            } catch (error) {
                callbackFeatures.streaming_error = error.message;
            }
        }
        
        return callbackFeatures;
        
    } catch (error) {
        return {error: error.message, available: false};
    }
}

async function main() {
    const text = process.argv[2];
    
    try {
        const result = await testCallbacks(text);
        console.log(JSON.stringify(result));
    } catch (error) {
        console.error(JSON.stringify({error: error.message}));
        process.exit(1);
    }
}

main();
'''
        
        # Create temporary script for callback testing
        callback_script = tempfile.NamedTemporaryFile(
            mode='w', suffix='.js', delete=False
        )
        callback_script.write(callback_script_content)
        callback_script.close()
        
        try:
            args = ['node', callback_script.name, text]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                error_info = json.loads(result.stderr) if result.stderr else {"error": "Unknown error"}
                return {"error": f"Node.js callback test failed: {error_info.get('error')}", "available": False}
            
            return json.loads(result.stdout)
            
        except subprocess.TimeoutExpired:
            return {"error": "Node.js callback test timed out", "available": False}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse Node.js callback test output: {e}", "available": False}
        except Exception as e:
            return {"error": str(e), "available": False}
        finally:
            # Clean up temporary callback script
            try:
                os.unlink(callback_script.name)
            except:
                pass
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.node_script:
            try:
                os.unlink(self.node_script.name)
            except:
                pass


class WASMBindingTester:
    """Test VoiRS via WebAssembly bindings."""
    
    def __init__(self):
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        """Check if WASM bindings are available."""
        # For now, we'll skip WASM testing as it requires a browser environment
        # In a real implementation, this could use selenium or puppeteer
        self.available = False
    
    def synthesize(self, text: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Synthesize text using WASM bindings."""
        if not self.available:
            raise RuntimeError("WASM bindings not available (requires browser environment)")
        
        # This would require browser automation
        raise NotImplementedError("WASM testing requires browser automation")
    
    def get_binding_info(self) -> Dict[str, Any]:
        """Get information about the WASM binding."""
        return {
            "binding_type": "wasm",
            "available": self.available,
            "note": "Requires browser environment for testing"
        }


class CrossLanguageConsistencyTester:
    """Main cross-language consistency tester."""
    
    def __init__(self):
        self.testers = {}
        self.results = {}
        self._setup_testers()
    
    def _setup_testers(self):
        """Setup all available testers."""
        # C API tester
        try:
            self.testers['c_api'] = CBindingTester()
        except Exception as e:
            print(f"C API tester not available: {e}")
        
        # Python tester
        try:
            self.testers['python'] = PythonBindingTester()
        except Exception as e:
            print(f"Python tester not available: {e}")
        
        # Node.js tester
        try:
            self.testers['nodejs'] = NodeJSBindingTester()
        except Exception as e:
            print(f"Node.js tester not available: {e}")
        
        # WASM tester
        try:
            self.testers['wasm'] = WASMBindingTester()
        except Exception as e:
            print(f"WASM tester not available: {e}")
    
    def get_available_bindings(self) -> List[str]:
        """Get list of available bindings."""
        available = []
        for name, tester in self.testers.items():
            info = tester.get_binding_info()
            if info.get('available', False):
                available.append(name)
        return available
    
    def run_consistency_tests(self) -> Dict[str, Any]:
        """Run consistency tests across all available bindings."""
        available_bindings = self.get_available_bindings()
        
        if len(available_bindings) < 2:
            return {
                "status": "skipped",
                "reason": f"Need at least 2 bindings, only {len(available_bindings)} available",
                "available_bindings": available_bindings
            }
        
        results = {
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "binding_count": len(available_bindings)
            },
            "binding_info": {},
            "test_results": {},
            "consistency_analysis": {}
        }
        
        # Get binding info
        for name, tester in self.testers.items():
            results["binding_info"][name] = tester.get_binding_info()
        
        # Run tests for each text and config combination
        for i, text in enumerate(TEST_TEXTS):
            for j, config in enumerate(TEST_CONFIGS):
                test_name = f"text_{i}_config_{j}"
                
                try:
                    test_result = self._run_single_consistency_test(
                        text, config, available_bindings, test_name
                    )
                    results["test_results"][test_name] = test_result
                    results["summary"]["total_tests"] += 1
                    
                    if test_result["consistent"]:
                        results["summary"]["passed_tests"] += 1
                    else:
                        results["summary"]["failed_tests"] += 1
                        
                except Exception as e:
                    results["test_results"][test_name] = {
                        "error": str(e),
                        "consistent": False
                    }
                    results["summary"]["total_tests"] += 1
                    results["summary"]["failed_tests"] += 1
        
        # Analyze overall consistency
        results["consistency_analysis"] = self._analyze_consistency(results["test_results"])
        
        return results
    
    def _run_single_consistency_test(
        self, text: str, config: Dict, bindings: List[str], test_name: str
    ) -> Dict[str, Any]:
        """Run a single consistency test across bindings."""
        binding_results = {}
        
        # Collect results from each binding
        for binding in bindings:
            try:
                # All bindings now use synchronous interface
                result = self.testers[binding].synthesize(text, config)
                
                binding_results[binding] = {
                    "success": True,
                    "result": result,
                    "error": None
                }
            except Exception as e:
                binding_results[binding] = {
                    "success": False,
                    "result": None,
                    "error": str(e)
                }
        
        # Check consistency
        successful_results = {
            name: data["result"] 
            for name, data in binding_results.items() 
            if data["success"]
        }
        
        if len(successful_results) < 2:
            return {
                "test_name": test_name,
                "text": text,
                "config": config,
                "binding_results": binding_results,
                "consistent": False,
                "reason": "Insufficient successful results for comparison"
            }
        
        # Compare results
        consistency_check = self._check_result_consistency(successful_results)
        
        return {
            "test_name": test_name,
            "text": text,
            "config": config,
            "binding_results": binding_results,
            "consistent": consistency_check["consistent"],
            "consistency_details": consistency_check,
            "successful_bindings": list(successful_results.keys())
        }
    
    def _check_result_consistency(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Check if results are consistent across bindings."""
        if len(results) < 2:
            return {"consistent": False, "reason": "Need at least 2 results"}
        
        # Get reference result (first one)
        reference_name, reference_result = next(iter(results.items()))
        
        consistency = {
            "consistent": True,
            "reference_binding": reference_name,
            "differences": [],
            "similarity_scores": {}
        }
        
        # Test format support and audio features
        format_test_results = self._test_format_support(results)
        consistency["format_support"] = format_test_results
        
        # Test callback features for Python binding
        if 'python' in results:
            callback_test_results = self._test_callback_features()
            consistency["callback_features"] = callback_test_results
        
        # Compare each result with reference
        for name, result in results.items():
            if name == reference_name:
                continue
            
            # Check basic properties
            properties_match = (
                result["sample_rate"] == reference_result["sample_rate"] and
                result["channels"] == reference_result["channels"]
            )
            
            if not properties_match:
                consistency["consistent"] = False
                consistency["differences"].append({
                    "binding": name,
                    "type": "properties_mismatch",
                    "details": {
                        "sample_rate": {"reference": reference_result["sample_rate"], "actual": result["sample_rate"]},
                        "channels": {"reference": reference_result["channels"], "actual": result["channels"]}
                    }
                })
            
            # Check audio similarity (approximate)
            similarity = self._calculate_audio_similarity(
                reference_result["samples"], result["samples"]
            )
            consistency["similarity_scores"][name] = similarity
            
            if similarity < 0.95:  # 95% similarity threshold
                consistency["consistent"] = False
                consistency["differences"].append({
                    "binding": name,
                    "type": "audio_difference",
                    "similarity": similarity
                })
        
        return consistency
    
    def _calculate_audio_similarity(self, samples1: List[float], samples2: List[float]) -> float:
        """Calculate similarity between two audio sample arrays."""
        if len(samples1) != len(samples2):
            return 0.0
        
        if not samples1:
            return 1.0
        
        # Calculate normalized cross-correlation
        try:
            import numpy as np
            
            a1 = np.array(samples1)
            a2 = np.array(samples2)
            
            # Normalize
            a1 = (a1 - np.mean(a1)) / (np.std(a1) + 1e-10)
            a2 = (a2 - np.mean(a2)) / (np.std(a2) + 1e-10)
            
            # Calculate correlation
            correlation = np.corrcoef(a1, a2)[0, 1]
            return max(0.0, correlation)
            
        except ImportError:
            # Fallback: simple mean squared error based similarity
            mse = sum((s1 - s2) ** 2 for s1, s2 in zip(samples1, samples2)) / len(samples1)
            return max(0.0, 1.0 - min(1.0, mse))
    
    def _test_format_support(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Test audio format support across bindings."""
        format_support = {
            "tested_formats": ["flac", "mp3", "wav"],
            "binding_support": {},
            "consistent_support": True
        }
        
        # Test C API format support
        if 'c_api' in self.testers:
            c_tester = self.testers['c_api']
            if hasattr(c_tester.lib, 'voirs_audio_get_supported_formats'):
                try:
                    # Test if FLAC/MP3 save functions exist
                    flac_support = hasattr(c_tester.lib, 'voirs_audio_save_flac')
                    mp3_support = hasattr(c_tester.lib, 'voirs_audio_save_mp3')
                    
                    format_support["binding_support"]["c_api"] = {
                        "flac": flac_support,
                        "mp3": mp3_support,
                        "wav": True  # Always supported
                    }
                except Exception as e:
                    format_support["binding_support"]["c_api"] = {"error": str(e)}
        
        # Test Python format support
        if 'python' in self.testers:
            py_tester = self.testers['python']
            if py_tester.available:
                try:
                    pipeline = py_tester.voirs.VoirsPipeline()
                    audio = pipeline.synthesize("test")
                    
                    # Test save methods if available
                    flac_support = hasattr(audio, 'save') or hasattr(audio, 'save_flac')
                    mp3_support = hasattr(audio, 'save') or hasattr(audio, 'save_mp3')
                    
                    format_support["binding_support"]["python"] = {
                        "flac": flac_support,
                        "mp3": mp3_support,
                        "wav": True
                    }
                except Exception as e:
                    format_support["binding_support"]["python"] = {"error": str(e)}
        
        return format_support
    
    def _test_callback_features(self) -> Dict[str, Any]:
        """Test callback features across bindings."""
        callback_features = {
            "progress_callbacks": False,
            "streaming_callbacks": False,
            "error_callbacks": False,
            "comprehensive_callbacks": False,
            "bindings_tested": []
        }
        
        # Test Python callback features
        if 'python' in self.testers and self.testers['python'].available:
            try:
                result = self.testers['python'].test_callback_features("test callback")
                callback_features.update(result)
                callback_features["bindings_tested"].append("python")
            except Exception as e:
                callback_features["python_error"] = str(e)
        
        # Test C API callback features
        if 'c_api' in self.testers and self.testers['c_api'].lib:
            try:
                result = self.testers['c_api'].test_callback_features("test callback")
                callback_features.update(result)
                callback_features["bindings_tested"].append("c_api")
            except Exception as e:
                callback_features["c_api_error"] = str(e)
        
        # Test Node.js callback features
        if 'nodejs' in self.testers and self.testers['nodejs'].available:
            try:
                result = self.testers['nodejs'].test_callback_features("test callback")
                callback_features.update(result)
                callback_features["bindings_tested"].append("nodejs")
            except Exception as e:
                callback_features["nodejs_error"] = str(e)
        
        return callback_features
    
    def _analyze_consistency(self, test_results: Dict) -> Dict[str, Any]:
        """Analyze overall consistency across all tests."""
        analysis = {
            "overall_consistent": True,
            "binding_reliability": {},
            "common_issues": [],
            "recommendations": []
        }
        
        # Count successes per binding
        binding_stats = {}
        
        for test_name, test_result in test_results.items():
            if "binding_results" in test_result:
                for binding, result in test_result["binding_results"].items():
                    if binding not in binding_stats:
                        binding_stats[binding] = {"total": 0, "success": 0}
                    
                    binding_stats[binding]["total"] += 1
                    if result["success"]:
                        binding_stats[binding]["success"] += 1
        
        # Calculate reliability scores
        for binding, stats in binding_stats.items():
            if stats["total"] > 0:
                reliability = stats["success"] / stats["total"]
                analysis["binding_reliability"][binding] = {
                    "success_rate": reliability,
                    "total_tests": stats["total"],
                    "successful_tests": stats["success"]
                }
        
        # Check if any tests failed consistency
        failed_tests = [
            name for name, result in test_results.items()
            if not result.get("consistent", False)
        ]
        
        if failed_tests:
            analysis["overall_consistent"] = False
            analysis["failed_tests"] = failed_tests
        
        # Generate recommendations
        if not analysis["overall_consistent"]:
            analysis["recommendations"].append(
                "Some consistency tests failed. Check individual test results for details."
            )
        
        low_reliability_bindings = [
            binding for binding, stats in analysis["binding_reliability"].items()
            if stats["success_rate"] < 0.8
        ]
        
        if low_reliability_bindings:
            analysis["recommendations"].append(
                f"Low reliability bindings detected: {low_reliability_bindings}. "
                "Check error logs and implementation consistency."
            )
        
        return analysis
    
    def cleanup(self):
        """Clean up resources."""
        for tester in self.testers.values():
            if hasattr(tester, 'cleanup'):
                tester.cleanup()


def main():
    """Main entry point for cross-language consistency testing."""
    print("VoiRS Cross-Language Consistency Test")
    print("=" * 50)
    
    tester = CrossLanguageConsistencyTester()
    
    try:
        available_bindings = tester.get_available_bindings()
        print(f"Available bindings: {available_bindings}")
        
        if len(available_bindings) < 2:
            print("Need at least 2 bindings for consistency testing.")
            print("Available bindings:", available_bindings)
            return 1
        
        print(f"Running consistency tests across {len(available_bindings)} bindings...")
        results = tester.run_consistency_tests()
        
        # Print summary
        summary = results["summary"]
        print(f"\nTest Summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success rate: {summary['passed_tests']/summary['total_tests']*100:.1f}%")
        
        # Print binding reliability
        print(f"\nBinding Reliability:")
        for binding, stats in results["consistency_analysis"]["binding_reliability"].items():
            print(f"  {binding}: {stats['success_rate']*100:.1f}% ({stats['successful_tests']}/{stats['total_tests']})")
        
        # Print enhanced features summary
        print(f"\nEnhanced Features Test Summary:")
        
        # Format support summary
        format_results = None
        for test_result in results["test_results"].values():
            if "consistency_details" in test_result and "format_support" in test_result["consistency_details"]:
                format_results = test_result["consistency_details"]["format_support"]
                break
        
        if format_results:
            print("  Format Support:")
            for binding, support in format_results.get("binding_support", {}).items():
                if isinstance(support, dict) and "error" not in support:
                    flac = "✓" if support.get("flac", False) else "✗"
                    mp3 = "✓" if support.get("mp3", False) else "✗"
                    print(f"    {binding}: FLAC {flac}, MP3 {mp3}")
        
        # Callback features summary
        callback_results = None
        for test_result in results["test_results"].values():
            if "consistency_details" in test_result and "callback_features" in test_result["consistency_details"]:
                callback_results = test_result["consistency_details"]["callback_features"]
                break
        
        if callback_results:
            print("  Callback Features:")
            for binding in callback_results.get("bindings_tested", []):
                progress = "✓" if callback_results.get("progress_callback_support", False) else "✗"
                streaming = "✓" if callback_results.get("streaming_support", False) else "✗"
                error = "✓" if callback_results.get("error_callback_support", False) else "✗"
                comprehensive = "✓" if callback_results.get("comprehensive_callbacks", False) else "✗"
                print(f"    {binding}: Progress {progress}, Streaming {streaming}, Error {error}, Comprehensive {comprehensive}")
        
        # Print overall result
        if results["consistency_analysis"]["overall_consistent"]:
            print("\n🎉 All bindings are consistent!")
            print("✅ Cross-language testing completed successfully!")
            return 0
        else:
            print("\n❌ Consistency issues detected!")
            for rec in results["consistency_analysis"]["recommendations"]:
                print(f"  - {rec}")
            print("\n📋 Check individual test results for detailed analysis.")
            return 1
    
    finally:
        tester.cleanup()


if __name__ == "__main__":
    import sys
    sys.exit(main())
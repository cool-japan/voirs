#!/usr/bin/env python3
"""
Enhanced performance benchmarking suite for VoiRS FFI bindings.

This module provides comprehensive performance testing and benchmarking
across different language bindings, including:
- Throughput testing
- Latency analysis
- Memory usage profiling
- Callback performance
- Streaming synthesis benchmarks
- Concurrent usage testing
"""

import sys
import os
import time
import threading
import multiprocessing
import tempfile
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    name: str
    duration: float
    memory_start: float
    memory_end: float
    memory_peak: float
    cpu_usage: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    success_rate: float
    errors: List[str]

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    iterations: int = 100
    warmup_iterations: int = 10
    concurrent_threads: int = 4
    text_lengths: List[int] = None
    timeout_seconds: int = 300
    memory_limit_mb: int = 1024
    
    def __post_init__(self):
        if self.text_lengths is None:
            self.text_lengths = [10, 50, 100, 200, 500]

class PerformanceProfiler:
    """Profiles performance metrics during execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start(self):
        """Start profiling."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.cpu_samples = []
        self.monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self) -> Dict[str, float]:
        """Stop profiling and return metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'duration': end_time - self.start_time,
            'memory_start': self.start_memory,
            'memory_end': end_memory,
            'memory_peak': self.peak_memory,
            'cpu_usage': statistics.mean(self.cpu_samples) if self.cpu_samples else 0
        }
    
    def _monitor_resources(self):
        """Monitor resources in background thread."""
        while self.monitoring:
            try:
                # Update peak memory
                current_memory = self.process.memory_info().rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # Sample CPU usage
                cpu_percent = self.process.cpu_percent()
                if cpu_percent > 0:  # Only add valid samples
                    self.cpu_samples.append(cpu_percent)
                
                time.sleep(0.1)  # Sample every 100ms
            except psutil.NoSuchProcess:
                break
            except Exception:
                continue

class EnhancedPerformanceTester:
    """Enhanced performance testing for VoiRS bindings."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results = {}
        self.available_bindings = {}
        self._setup_bindings()
    
    def _setup_bindings(self):
        """Setup available bindings for testing."""
        # Python bindings
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
        
        # C API bindings (via ctypes)
        try:
            import ctypes
            from ctypes import POINTER, c_char_p, c_void_p, c_uint32, c_float, c_size_t
            
            # Try to find the library
            lib_paths = [
                "target/debug/libvoirs_ffi.so",
                "target/debug/libvoirs_ffi.dylib",
                "target/debug/voirs_ffi.dll",
                "../target/debug/libvoirs_ffi.so",
                "../target/debug/libvoirs_ffi.dylib",
                "../../target/debug/libvoirs_ffi.so",
                "../../target/debug/libvoirs_ffi.dylib",
            ]
            
            lib_path = None
            for path in lib_paths:
                if os.path.exists(path):
                    lib_path = path
                    break
            
            if lib_path:
                lib = ctypes.CDLL(lib_path)
                self.available_bindings['c_api'] = {
                    'lib': lib,
                    'lib_path': lib_path,
                    'available': True
                }
            else:
                self.available_bindings['c_api'] = {
                    'lib': None,
                    'available': False,
                    'error': 'Library not found'
                }
        except Exception as e:
            self.available_bindings['c_api'] = {
                'lib': None,
                'available': False,
                'error': str(e)
            }
    
    def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        print("Running Enhanced Performance Benchmarks")
        print("=" * 50)
        
        results = {
            'config': asdict(self.config),
            'system_info': self._get_system_info(),
            'binding_availability': self._get_binding_availability(),
            'benchmarks': {}
        }
        
        # Run different benchmark categories
        benchmark_categories = [
            ('throughput', self._benchmark_throughput),
            ('latency', self._benchmark_latency),
            ('memory', self._benchmark_memory_usage),
            ('concurrent', self._benchmark_concurrent_usage),
            ('streaming', self._benchmark_streaming),
            ('callbacks', self._benchmark_callbacks),
            ('text_length', self._benchmark_text_length_scaling),
            ('stress', self._benchmark_stress_test)
        ]
        
        for category, benchmark_func in benchmark_categories:
            print(f"\n--- {category.upper()} BENCHMARKS ---")
            try:
                results['benchmarks'][category] = benchmark_func()
                print(f"✓ {category} benchmarks completed")
            except Exception as e:
                print(f"✗ {category} benchmarks failed: {e}")
                results['benchmarks'][category] = {'error': str(e)}
        
        return results
    
    def _benchmark_throughput(self) -> Dict[str, Any]:
        """Benchmark throughput (operations per second)."""
        results = {}
        test_text = "This is a throughput test sentence for performance benchmarking."
        
        for binding_name, binding_info in self.available_bindings.items():
            if not binding_info['available']:
                continue
            
            print(f"Testing {binding_name} throughput...")
            
            # Warmup
            self._run_synthesis_warmup(binding_name, binding_info)
            
            # Benchmark
            profiler = PerformanceProfiler()
            profiler.start()
            
            start_time = time.time()
            successful_ops = 0
            errors = []
            
            for i in range(self.config.iterations):
                try:
                    self._synthesize_with_binding(binding_name, binding_info, test_text)
                    successful_ops += 1
                except Exception as e:
                    errors.append(str(e))
                    if len(errors) > 10:  # Limit error collection
                        break
            
            end_time = time.time()
            profile_data = profiler.stop()
            
            duration = end_time - start_time
            throughput = successful_ops / duration if duration > 0 else 0
            
            results[binding_name] = {
                'throughput_ops_per_sec': throughput,
                'total_operations': successful_ops,
                'total_errors': len(errors),
                'success_rate': successful_ops / self.config.iterations,
                'duration_seconds': duration,
                'profile': profile_data,
                'errors': errors[:5]  # Keep first 5 errors
            }
            
            print(f"  {binding_name}: {throughput:.2f} ops/sec ({successful_ops}/{self.config.iterations} successful)")
        
        return results
    
    def _benchmark_latency(self) -> Dict[str, Any]:
        """Benchmark latency distribution."""
        results = {}
        test_text = "Latency test sentence."
        
        for binding_name, binding_info in self.available_bindings.items():
            if not binding_info['available']:
                continue
            
            print(f"Testing {binding_name} latency...")
            
            # Warmup
            self._run_synthesis_warmup(binding_name, binding_info)
            
            # Collect latency samples
            latencies = []
            errors = []
            
            for i in range(self.config.iterations):
                try:
                    start_time = time.time()
                    self._synthesize_with_binding(binding_name, binding_info, test_text)
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    latencies.append(latency)
                    
                except Exception as e:
                    errors.append(str(e))
            
            if latencies:
                latencies.sort()
                results[binding_name] = {
                    'latency_p50_ms': statistics.quantiles(latencies, n=2)[0],
                    'latency_p95_ms': statistics.quantiles(latencies, n=20)[18],
                    'latency_p99_ms': statistics.quantiles(latencies, n=100)[98],
                    'latency_mean_ms': statistics.mean(latencies),
                    'latency_stddev_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    'latency_min_ms': min(latencies),
                    'latency_max_ms': max(latencies),
                    'successful_samples': len(latencies),
                    'errors': len(errors)
                }
                
                print(f"  {binding_name}: P50={results[binding_name]['latency_p50_ms']:.1f}ms, "
                      f"P95={results[binding_name]['latency_p95_ms']:.1f}ms, "
                      f"P99={results[binding_name]['latency_p99_ms']:.1f}ms")
            else:
                results[binding_name] = {'error': 'No successful latency samples'}
        
        return results
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        results = {}
        test_texts = [f"Memory test sentence {i}" for i in range(50)]
        
        for binding_name, binding_info in self.available_bindings.items():
            if not binding_info['available']:
                continue
            
            print(f"Testing {binding_name} memory usage...")
            
            # Force garbage collection
            gc.collect()
            
            profiler = PerformanceProfiler()
            profiler.start()
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create pipeline
            pipeline = self._create_pipeline(binding_name, binding_info)
            after_create_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Perform synthesis operations
            for text in test_texts:
                try:
                    self._synthesize_with_pipeline(binding_name, pipeline, text)
                except Exception:
                    pass
            
            after_synthesis_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Cleanup
            self._cleanup_pipeline(binding_name, pipeline)
            gc.collect()
            
            profile_data = profiler.stop()
            
            results[binding_name] = {
                'initial_memory_mb': initial_memory,
                'after_create_memory_mb': after_create_memory,
                'after_synthesis_memory_mb': after_synthesis_memory,
                'peak_memory_mb': profile_data['memory_peak'],
                'memory_growth_mb': after_synthesis_memory - initial_memory,
                'pipeline_overhead_mb': after_create_memory - initial_memory,
                'synthesis_memory_mb': after_synthesis_memory - after_create_memory,
                'profile': profile_data
            }
            
            print(f"  {binding_name}: Peak={profile_data['memory_peak']:.1f}MB, "
                  f"Growth={after_synthesis_memory - initial_memory:.1f}MB")
        
        return results
    
    def _benchmark_concurrent_usage(self) -> Dict[str, Any]:
        """Benchmark concurrent usage patterns."""
        results = {}
        test_text = "Concurrent synthesis test."
        
        for binding_name, binding_info in self.available_bindings.items():
            if not binding_info['available']:
                continue
            
            print(f"Testing {binding_name} concurrent usage...")
            
            def worker_function():
                try:
                    pipeline = self._create_pipeline(binding_name, binding_info)
                    for _ in range(10):
                        self._synthesize_with_pipeline(binding_name, pipeline, test_text)
                    self._cleanup_pipeline(binding_name, pipeline)
                    return True
                except Exception as e:
                    return str(e)
            
            # Test different concurrency levels
            for num_threads in [1, 2, 4, 8]:
                profiler = PerformanceProfiler()
                profiler.start()
                
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(worker_function) for _ in range(num_threads)]
                    results_list = [f.result() for f in futures]
                
                end_time = time.time()
                profile_data = profiler.stop()
                
                successful = sum(1 for r in results_list if r is True)
                total_ops = num_threads * 10
                
                if binding_name not in results:
                    results[binding_name] = {}
                
                results[binding_name][f'{num_threads}_threads'] = {
                    'successful_operations': successful * 10,
                    'total_operations': total_ops,
                    'duration_seconds': end_time - start_time,
                    'throughput_ops_per_sec': (successful * 10) / (end_time - start_time),
                    'profile': profile_data,
                    'errors': [r for r in results_list if r is not True]
                }
                
                print(f"  {binding_name} ({num_threads} threads): "
                      f"{(successful * 10) / (end_time - start_time):.1f} ops/sec")
        
        return results
    
    def _benchmark_streaming(self) -> Dict[str, Any]:
        """Benchmark streaming synthesis if available."""
        results = {}
        test_text = "This is a longer text for streaming synthesis benchmarking. " * 5
        
        for binding_name, binding_info in self.available_bindings.items():
            if not binding_info['available']:
                continue
            
            print(f"Testing {binding_name} streaming...")
            
            try:
                pipeline = self._create_pipeline(binding_name, binding_info)
                
                # Test streaming if available
                if binding_name == 'python' and hasattr(binding_info['module'], 'VoirsPipeline'):
                    voirs_pipeline = binding_info['module'].VoirsPipeline()
                    
                    if hasattr(voirs_pipeline, 'synthesize_streaming'):
                        chunks_received = []
                        
                        def chunk_callback(chunk_idx, total_chunks, audio_chunk):
                            chunks_received.append({
                                'chunk_idx': chunk_idx,
                                'total_chunks': total_chunks,
                                'timestamp': time.time()
                            })
                        
                        start_time = time.time()
                        audio = voirs_pipeline.synthesize_streaming(test_text, chunk_callback)
                        end_time = time.time()
                        
                        results[binding_name] = {
                            'streaming_available': True,
                            'total_chunks': len(chunks_received),
                            'streaming_duration_seconds': end_time - start_time,
                            'first_chunk_latency_ms': (chunks_received[0]['timestamp'] - start_time) * 1000 if chunks_received else 0,
                            'chunk_intervals_ms': [
                                (chunks_received[i]['timestamp'] - chunks_received[i-1]['timestamp']) * 1000
                                for i in range(1, len(chunks_received))
                            ]
                        }
                        
                        print(f"  {binding_name}: {len(chunks_received)} chunks in {end_time - start_time:.2f}s")
                    else:
                        results[binding_name] = {'streaming_available': False}
                else:
                    results[binding_name] = {'streaming_available': False}
                
                self._cleanup_pipeline(binding_name, pipeline)
                
            except Exception as e:
                results[binding_name] = {'error': str(e)}
        
        return results
    
    def _benchmark_callbacks(self) -> Dict[str, Any]:
        """Benchmark callback performance."""
        results = {}
        test_text = "Callback performance test."
        
        for binding_name, binding_info in self.available_bindings.items():
            if not binding_info['available']:
                continue
            
            print(f"Testing {binding_name} callbacks...")
            
            try:
                if binding_name == 'python' and hasattr(binding_info['module'], 'VoirsPipeline'):
                    voirs_pipeline = binding_info['module'].VoirsPipeline()
                    
                    # Test progress callback if available
                    if hasattr(voirs_pipeline, 'synthesize_with_callbacks'):
                        callback_calls = []
                        
                        def progress_callback(current, total, progress, message):
                            callback_calls.append({
                                'type': 'progress',
                                'current': current,
                                'total': total,
                                'progress': progress,
                                'timestamp': time.time()
                            })
                        
                        def error_callback(error_info):
                            callback_calls.append({
                                'type': 'error',
                                'error': str(error_info),
                                'timestamp': time.time()
                            })
                        
                        start_time = time.time()
                        audio = voirs_pipeline.synthesize_with_callbacks(
                            test_text,
                            progress_callback=progress_callback,
                            error_callback=error_callback
                        )
                        end_time = time.time()
                        
                        results[binding_name] = {
                            'callbacks_available': True,
                            'total_callbacks': len(callback_calls),
                            'progress_callbacks': len([c for c in callback_calls if c['type'] == 'progress']),
                            'error_callbacks': len([c for c in callback_calls if c['type'] == 'error']),
                            'callback_duration_seconds': end_time - start_time,
                            'callback_overhead_ms': (end_time - start_time) * 1000 / len(callback_calls) if callback_calls else 0
                        }
                        
                        print(f"  {binding_name}: {len(callback_calls)} callbacks in {end_time - start_time:.2f}s")
                    else:
                        results[binding_name] = {'callbacks_available': False}
                else:
                    results[binding_name] = {'callbacks_available': False}
                
            except Exception as e:
                results[binding_name] = {'error': str(e)}
        
        return results
    
    def _benchmark_text_length_scaling(self) -> Dict[str, Any]:
        """Benchmark performance scaling with text length."""
        results = {}
        base_text = "This is a test sentence for length scaling. "
        
        for binding_name, binding_info in self.available_bindings.items():
            if not binding_info['available']:
                continue
            
            print(f"Testing {binding_name} text length scaling...")
            
            binding_results = {}
            
            for length in self.config.text_lengths:
                test_text = base_text * (length // len(base_text) + 1)
                test_text = test_text[:length]
                
                # Run multiple samples
                latencies = []
                for _ in range(10):
                    try:
                        start_time = time.time()
                        self._synthesize_with_binding(binding_name, binding_info, test_text)
                        end_time = time.time()
                        latencies.append(end_time - start_time)
                    except Exception:
                        pass
                
                if latencies:
                    binding_results[f'length_{length}'] = {
                        'text_length': length,
                        'mean_latency_ms': statistics.mean(latencies) * 1000,
                        'min_latency_ms': min(latencies) * 1000,
                        'max_latency_ms': max(latencies) * 1000,
                        'samples': len(latencies),
                        'throughput_chars_per_sec': length / statistics.mean(latencies) if latencies else 0
                    }
                    
                    print(f"  {binding_name} ({length} chars): "
                          f"{statistics.mean(latencies) * 1000:.1f}ms avg, "
                          f"{length / statistics.mean(latencies):.0f} chars/sec")
            
            results[binding_name] = binding_results
        
        return results
    
    def _benchmark_stress_test(self) -> Dict[str, Any]:
        """Run stress tests to find limits."""
        results = {}
        test_text = "Stress test sentence. " * 10
        
        for binding_name, binding_info in self.available_bindings.items():
            if not binding_info['available']:
                continue
            
            print(f"Running {binding_name} stress test...")
            
            # Gradually increase load until failure
            max_concurrent = 1
            max_sustained_ops = 0
            
            for concurrent in [1, 2, 4, 8, 16]:
                try:
                    def stress_worker():
                        count = 0
                        start_time = time.time()
                        while time.time() - start_time < 30:  # 30 second test
                            try:
                                self._synthesize_with_binding(binding_name, binding_info, test_text)
                                count += 1
                            except Exception:
                                break
                        return count
                    
                    with ThreadPoolExecutor(max_workers=concurrent) as executor:
                        futures = [executor.submit(stress_worker) for _ in range(concurrent)]
                        counts = [f.result() for f in futures]
                    
                    total_ops = sum(counts)
                    if total_ops > max_sustained_ops:
                        max_sustained_ops = total_ops
                        max_concurrent = concurrent
                    
                    print(f"  {binding_name} ({concurrent} concurrent): {total_ops} ops in 30s")
                    
                except Exception as e:
                    print(f"  {binding_name} failed at {concurrent} concurrent: {e}")
                    break
            
            results[binding_name] = {
                'max_concurrent_threads': max_concurrent,
                'max_sustained_ops_per_30s': max_sustained_ops,
                'estimated_max_ops_per_sec': max_sustained_ops / 30
            }
        
        return results
    
    def _synthesize_with_binding(self, binding_name: str, binding_info: Dict, text: str):
        """Synthesize text with a specific binding."""
        if binding_name == 'python':
            pipeline = binding_info['module'].VoirsPipeline()
            return pipeline.synthesize(text)
        elif binding_name == 'c_api':
            # Simplified C API usage
            lib = binding_info['lib']
            # This would need proper C API implementation
            raise NotImplementedError("C API synthesis not implemented in this benchmark")
        else:
            raise ValueError(f"Unknown binding: {binding_name}")
    
    def _create_pipeline(self, binding_name: str, binding_info: Dict):
        """Create a pipeline for a specific binding."""
        if binding_name == 'python':
            return binding_info['module'].VoirsPipeline()
        elif binding_name == 'c_api':
            # Return C API pipeline handle
            return binding_info['lib']
        else:
            return None
    
    def _synthesize_with_pipeline(self, binding_name: str, pipeline, text: str):
        """Synthesize with an existing pipeline."""
        if binding_name == 'python':
            return pipeline.synthesize(text)
        elif binding_name == 'c_api':
            # C API synthesis with existing pipeline
            raise NotImplementedError("C API pipeline synthesis not implemented")
        else:
            raise ValueError(f"Unknown binding: {binding_name}")
    
    def _cleanup_pipeline(self, binding_name: str, pipeline):
        """Clean up a pipeline."""
        if binding_name == 'python':
            # Python pipelines are garbage collected
            pass
        elif binding_name == 'c_api':
            # C API cleanup would be needed
            pass
    
    def _run_synthesis_warmup(self, binding_name: str, binding_info: Dict):
        """Run warmup synthesis to prepare the binding."""
        for _ in range(self.config.warmup_iterations):
            try:
                self._synthesize_with_binding(binding_name, binding_info, "warmup")
            except Exception:
                pass
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def _get_binding_availability(self) -> Dict[str, Any]:
        """Get binding availability information."""
        return {
            name: {
                'available': info['available'],
                'error': info.get('error', None),
                'version': info.get('version', None)
            }
            for name, info in self.available_bindings.items()
        }
    
    def generate_report(self, results: Dict[str, Any], output_file: str = None):
        """Generate a comprehensive performance report."""
        if output_file is None:
            output_file = f"performance_report_{int(time.time())}.json"
        
        # Add summary statistics
        results['summary'] = self._generate_summary(results)
        
        # Save JSON report
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nPerformance report saved to: {output_file}")
        
        # Print summary to console
        self._print_summary(results['summary'])
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'total_benchmarks': len(results.get('benchmarks', {})),
            'available_bindings': len([b for b in results.get('binding_availability', {}).values() if b['available']]),
            'system_info': results.get('system_info', {}),
            'binding_comparison': {}
        }
        
        # Compare bindings across benchmarks
        benchmarks = results.get('benchmarks', {})
        
        for binding_name in results.get('binding_availability', {}):
            if not results['binding_availability'][binding_name]['available']:
                continue
            
            binding_summary = {}
            
            # Throughput
            if 'throughput' in benchmarks and binding_name in benchmarks['throughput']:
                binding_summary['throughput_ops_per_sec'] = benchmarks['throughput'][binding_name].get('throughput_ops_per_sec', 0)
            
            # Latency
            if 'latency' in benchmarks and binding_name in benchmarks['latency']:
                binding_summary['latency_p50_ms'] = benchmarks['latency'][binding_name].get('latency_p50_ms', 0)
                binding_summary['latency_p95_ms'] = benchmarks['latency'][binding_name].get('latency_p95_ms', 0)
            
            # Memory
            if 'memory' in benchmarks and binding_name in benchmarks['memory']:
                binding_summary['peak_memory_mb'] = benchmarks['memory'][binding_name].get('peak_memory_mb', 0)
                binding_summary['memory_growth_mb'] = benchmarks['memory'][binding_name].get('memory_growth_mb', 0)
            
            # Concurrent performance
            if 'concurrent' in benchmarks and binding_name in benchmarks['concurrent']:
                concurrent_data = benchmarks['concurrent'][binding_name]
                if '4_threads' in concurrent_data:
                    binding_summary['concurrent_4_threads_ops_per_sec'] = concurrent_data['4_threads'].get('throughput_ops_per_sec', 0)
            
            summary['binding_comparison'][binding_name] = binding_summary
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary to console."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Total benchmarks: {summary['total_benchmarks']}")
        print(f"Available bindings: {summary['available_bindings']}")
        
        print("\nBinding Comparison:")
        print("-" * 40)
        
        for binding_name, binding_data in summary['binding_comparison'].items():
            print(f"\n{binding_name.upper()}:")
            if 'throughput_ops_per_sec' in binding_data:
                print(f"  Throughput: {binding_data['throughput_ops_per_sec']:.2f} ops/sec")
            if 'latency_p50_ms' in binding_data:
                print(f"  Latency P50: {binding_data['latency_p50_ms']:.1f}ms")
            if 'latency_p95_ms' in binding_data:
                print(f"  Latency P95: {binding_data['latency_p95_ms']:.1f}ms")
            if 'peak_memory_mb' in binding_data:
                print(f"  Peak Memory: {binding_data['peak_memory_mb']:.1f}MB")
            if 'memory_growth_mb' in binding_data:
                print(f"  Memory Growth: {binding_data['memory_growth_mb']:.1f}MB")
            if 'concurrent_4_threads_ops_per_sec' in binding_data:
                print(f"  Concurrent (4 threads): {binding_data['concurrent_4_threads_ops_per_sec']:.2f} ops/sec")

def main():
    """Main entry point for enhanced performance testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced VoiRS Performance Benchmarks")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations per test")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--threads", type=int, default=4, help="Number of concurrent threads")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    # Configure benchmarks
    config = BenchmarkConfig(
        iterations=args.iterations if not args.quick else 20,
        warmup_iterations=args.warmup if not args.quick else 3,
        concurrent_threads=args.threads,
        timeout_seconds=300 if not args.quick else 60
    )
    
    # Run benchmarks
    tester = EnhancedPerformanceTester(config)
    results = tester.run_comprehensive_benchmarks()
    
    # Generate report
    tester.generate_report(results, args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
Performance and benchmark tests for VoiRS Python bindings.

Tests performance characteristics, throughput, latency, and resource usage
under various conditions and workloads.
"""

import pytest
import asyncio
import time
import statistics
from pathlib import Path
import gc
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import voirs_ffi as voirs
    VOIRS_AVAILABLE = True
except ImportError:
    VOIRS_AVAILABLE = False

pytestmark = [pytest.mark.performance, pytest.mark.slow]


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestSynthesisPerformance:
    """Test synthesis performance characteristics."""
    
    async def test_synthesis_latency(self, voirs_pipeline, sample_texts, performance_metrics):
        """Test synthesis latency for different text lengths."""
        test_cases = [
            ("short", sample_texts["short"]),
            ("medium", sample_texts["medium"]),
            ("long", sample_texts["long"]),
        ]
        
        results = {}
        
        for case_name, text in test_cases:
            latencies = []
            
            # Warm up
            await voirs_pipeline.synthesize(text[:10])
            
            # Measure multiple runs
            for _ in range(5):
                performance_metrics.start_timer(f"latency_{case_name}")
                await voirs_pipeline.synthesize(text)
                performance_metrics.end_timer(f"latency_{case_name}")
                latencies.append(performance_metrics.get_duration(f"latency_{case_name}"))
            
            results[case_name] = {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            }
        
        # Verify reasonable performance
        assert results["short"]["mean"] < 5.0, f"Short text too slow: {results['short']['mean']:.2f}s"
        assert results["medium"]["mean"] < 10.0, f"Medium text too slow: {results['medium']['mean']:.2f}s"
        assert results["long"]["mean"] < 20.0, f"Long text too slow: {results['long']['mean']:.2f}s"
        
        # Longer texts should take proportionally longer
        assert results["short"]["mean"] < results["medium"]["mean"] < results["long"]["mean"]
        
        print(f"Performance Results: {results}")
    
    async def test_synthesis_throughput(self, voirs_pipeline, sample_texts, performance_metrics):
        """Test synthesis throughput (operations per second)."""
        text = sample_texts["short"]
        duration = 30.0  # Test for 30 seconds
        
        performance_metrics.start_timer("throughput_test")
        start_time = time.time()
        operations = 0
        
        while time.time() - start_time < duration:
            await voirs_pipeline.synthesize(f"{text} {operations}")
            operations += 1
        
        performance_metrics.end_timer("throughput_test")
        actual_duration = performance_metrics.get_duration("throughput_test")
        
        throughput = operations / actual_duration
        
        # Should handle at least 1 operation per second
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} ops/sec"
        
        print(f"Throughput: {throughput:.2f} operations/second ({operations} ops in {actual_duration:.2f}s)")
    
    async def test_batch_performance(self, voirs_pipeline, sample_texts, performance_metrics):
        """Test performance of batch synthesis operations."""
        texts = [
            f"{sample_texts['short']} {i}" for i in range(10)
        ]
        
        # Sequential synthesis
        performance_metrics.start_timer("sequential_batch")
        sequential_results = []
        for text in texts:
            audio = await voirs_pipeline.synthesize(text)
            sequential_results.append(audio)
        performance_metrics.end_timer("sequential_batch")
        
        sequential_time = performance_metrics.get_duration("sequential_batch")
        
        # Verify all results are valid
        assert len(sequential_results) == len(texts)
        for audio in sequential_results:
            assert audio.duration > 0
        
        print(f"Sequential batch: {sequential_time:.2f}s for {len(texts)} texts")
        
        # Performance should be reasonable for batch processing
        avg_time_per_text = sequential_time / len(texts)
        assert avg_time_per_text < 5.0, f"Batch processing too slow: {avg_time_per_text:.2f}s per text"
    
    async def test_memory_usage_stability(self, voirs_pipeline, sample_texts, memory_tracker):
        """Test memory usage stability during extended operations."""
        text = sample_texts["medium"]
        memory_tracker.start_tracking()
        
        initial_memory = memory_tracker.get_memory_usage()
        print(f"Initial memory: {initial_memory['current_mb']:.2f} MB")
        
        # Perform many synthesis operations
        for i in range(20):
            audio = await voirs_pipeline.synthesize(f"{text} {i}")
            # Don't keep references to allow garbage collection
            del audio
            
            # Force garbage collection every few iterations
            if i % 5 == 0:
                gc.collect()
        
        final_memory = memory_tracker.get_memory_usage()
        print(f"Final memory: {final_memory['current_mb']:.2f} MB")
        print(f"Memory delta: {final_memory['delta_mb']:.2f} MB")
        
        # Memory growth should be reasonable
        assert not memory_tracker.check_memory_leak(threshold_mb=100.0), \
            f"Excessive memory growth: {final_memory['delta_mb']:.2f} MB"
    
    async def test_configuration_performance_impact(self, voirs_pipeline, sample_texts, performance_metrics):
        """Test performance impact of different configurations."""
        text = sample_texts["medium"]
        
        configs = {
            "default": {},
            "fast": {"speaking_rate": 2.0},
            "slow": {"speaking_rate": 0.5},
            "high_quality": {"quality": "high", "sample_rate": 48000},
            "low_quality": {"quality": "low", "sample_rate": 22050},
        }
        
        results = {}
        
        for config_name, config in configs.items():
            times = []
            
            # Warm up
            try:
                await voirs_pipeline.synthesize_with_config(text[:10], config)
            except Exception:
                continue  # Skip unsupported configs
            
            # Measure multiple runs
            for _ in range(3):
                performance_metrics.start_timer(f"config_{config_name}")
                try:
                    await voirs_pipeline.synthesize_with_config(text, config)
                    performance_metrics.end_timer(f"config_{config_name}")
                    times.append(performance_metrics.get_duration(f"config_{config_name}"))
                except Exception:
                    break
            
            if times:
                results[config_name] = statistics.mean(times)
        
        print(f"Configuration performance: {results}")
        
        # Fast config should be faster than slow config
        if "fast" in results and "slow" in results:
            assert results["fast"] < results["slow"], \
                f"Fast config not faster: {results['fast']:.2f}s vs {results['slow']:.2f}s"


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestResourceUsage:
    """Test resource usage patterns."""
    
    async def test_cpu_usage_patterns(self, voirs_pipeline, sample_texts):
        """Test CPU usage during synthesis."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for CPU monitoring")
        
        text = sample_texts["long"]
        process = psutil.Process()
        
        # Measure CPU usage during synthesis
        cpu_before = process.cpu_percent()
        await voirs_pipeline.synthesize(text)
        cpu_after = process.cpu_percent()
        
        # Give some time for CPU measurement
        await asyncio.sleep(0.1)
        cpu_during = process.cpu_percent()
        
        print(f"CPU usage - Before: {cpu_before}%, During: {cpu_during}%, After: {cpu_after}%")
        
        # CPU usage should be reasonable (not pegging at 100%)
        assert cpu_during < 95.0, f"CPU usage too high: {cpu_during}%"
    
    async def test_memory_patterns(self, voirs_pipeline, sample_texts, memory_tracker):
        """Test memory usage patterns during different operations."""
        memory_tracker.start_tracking()
        
        operations = [
            ("short_text", lambda: voirs_pipeline.synthesize(sample_texts["short"])),
            ("medium_text", lambda: voirs_pipeline.synthesize(sample_texts["medium"])),
            ("long_text", lambda: voirs_pipeline.synthesize(sample_texts["long"])),
        ]
        
        memory_history = []
        
        for op_name, operation in operations:
            memory_before = memory_tracker.get_memory_usage()
            await operation()
            memory_after = memory_tracker.get_memory_usage()
            
            memory_history.append({
                "operation": op_name,
                "before": memory_before["current_mb"],
                "after": memory_after["current_mb"],
                "delta": memory_after["current_mb"] - memory_before["current_mb"],
            })
        
        print(f"Memory usage patterns: {memory_history}")
        
        # Memory usage should be predictable
        for record in memory_history:
            assert record["delta"] < 50.0, f"Excessive memory usage in {record['operation']}: {record['delta']:.2f} MB"
    
    async def test_concurrent_resource_usage(self, voirs_pipeline, sample_texts, memory_tracker):
        """Test resource usage during concurrent operations."""
        memory_tracker.start_tracking()
        text = sample_texts["medium"]
        
        async def synthesis_task(task_id):
            return await voirs_pipeline.synthesize(f"{text} Task {task_id}")
        
        # Run multiple synthesis tasks concurrently
        tasks = [synthesis_task(i) for i in range(5)]
        
        memory_before = memory_tracker.get_memory_usage()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        memory_after = memory_tracker.get_memory_usage()
        
        # Check that all tasks completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 3, f"Too many concurrent tasks failed: {len(successful_results)}/5"
        
        # Memory usage should be reasonable even with concurrent operations
        memory_delta = memory_after["current_mb"] - memory_before["current_mb"]
        assert memory_delta < 100.0, f"Excessive memory usage during concurrent operations: {memory_delta:.2f} MB"
        
        print(f"Concurrent operations - Memory delta: {memory_delta:.2f} MB")


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestStressTests:
    """Stress tests for extreme conditions."""
    
    async def test_high_frequency_requests(self, voirs_pipeline, sample_texts):
        """Test handling of high-frequency synthesis requests."""
        text = sample_texts["short"]
        
        # Send many requests in quick succession
        tasks = []
        for i in range(50):
            task = voirs_pipeline.synthesize(f"{text} {i}")
            tasks.append(task)
        
        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            pytest.fail("High-frequency requests timed out")
        
        # Check results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful) / len(results)
        print(f"High-frequency test: {len(successful)}/{len(results)} successful ({success_rate:.1%})")
        
        # At least 80% should succeed
        assert success_rate >= 0.8, f"Too many high-frequency requests failed: {success_rate:.1%}"
    
    async def test_large_text_handling(self, voirs_pipeline, audio_validator):
        """Test handling of very large text inputs."""
        # Create a very large text
        base_text = "This is a test sentence for large text handling. "
        large_text = base_text * 100  # ~5000 characters
        
        try:
            audio = await voirs_pipeline.synthesize(large_text)
            audio_validator.validate_audio_buffer(audio)
            
            # Large text should produce long audio
            assert audio.duration > 30.0, f"Large text produced short audio: {audio.duration}s"
            
            print(f"Large text ({len(large_text)} chars) -> {audio.duration:.2f}s audio")
            
        except Exception as e:
            # Some implementations might have text length limits
            if "too long" in str(e).lower() or "limit" in str(e).lower():
                pytest.skip(f"Text length limit reached: {e}")
            else:
                raise
    
    async def test_rapid_voice_switching(self, voirs_pipeline, sample_texts):
        """Test rapid switching between voices."""
        voices = await voirs_pipeline.list_voices()
        
        if len(voices) < 2:
            pytest.skip("Need at least 2 voices for rapid switching test")
        
        text = sample_texts["short"]
        
        # Rapidly switch between voices
        for i in range(10):
            voice = voices[i % len(voices[:2])]  # Alternate between first 2 voices
            await voirs_pipeline.set_voice(voice.id)
            
            # Verify voice was set
            current_voice = await voirs_pipeline.get_voice()
            assert current_voice == voice.id
            
            # Synthesize with current voice
            audio = await voirs_pipeline.synthesize(f"{text} {i}")
            assert audio.duration > 0
        
        print(f"Rapid voice switching completed: {10} switches")
    
    async def test_long_running_stability(self, voirs_pipeline, sample_texts, memory_tracker):
        """Test stability during long-running operations."""
        memory_tracker.start_tracking()
        text = sample_texts["medium"]
        
        # Run for extended period
        start_time = time.time()
        operations = 0
        max_duration = 60.0  # 1 minute test
        
        while time.time() - start_time < max_duration:
            try:
                audio = await voirs_pipeline.synthesize(f"{text} {operations}")
                assert audio.duration > 0
                operations += 1
                
                # Clean up to prevent memory accumulation
                del audio
                
                # Occasional garbage collection
                if operations % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"Operation {operations} failed: {e}")
                break
        
        actual_duration = time.time() - start_time
        
        # Check memory stability
        memory_usage = memory_tracker.get_memory_usage()
        memory_leak = memory_tracker.check_memory_leak(threshold_mb=50.0)
        
        print(f"Long-running test: {operations} operations in {actual_duration:.2f}s")
        print(f"Memory usage: {memory_usage['current_mb']:.2f} MB (delta: {memory_usage['delta_mb']:.2f} MB)")
        
        # Should complete reasonable number of operations
        assert operations > 10, f"Too few operations completed: {operations}"
        
        # Memory should be stable
        assert not memory_leak, f"Memory leak detected: {memory_usage['delta_mb']:.2f} MB"


@pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS Python bindings not available")
class TestScalability:
    """Test scalability characteristics."""
    
    async def test_pipeline_scaling(self, sample_texts, audio_validator):
        """Test performance with multiple pipeline instances."""
        text = sample_texts["short"]
        
        # Create multiple pipelines
        pipelines = []
        for i in range(3):
            try:
                pipeline = await voirs.VoirsPipeline.create()
                pipelines.append(pipeline)
            except Exception as e:
                if i == 0:  # At least one should succeed
                    raise
                print(f"Pipeline {i} creation failed: {e}")
                break
        
        print(f"Created {len(pipelines)} pipelines")
        
        # Use all pipelines concurrently
        tasks = []
        for i, pipeline in enumerate(pipelines):
            task = pipeline.synthesize(f"{text} Pipeline {i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == len(pipelines), f"Pipeline scaling failed: {len(successful)}/{len(pipelines)}"
        
        # Validate all audio
        for audio in successful:
            audio_validator.validate_audio_buffer(audio)
    
    async def test_threading_compatibility(self, voirs_pipeline, sample_texts):
        """Test thread safety and compatibility."""
        text = sample_texts["short"]
        
        def sync_synthesis_wrapper(pipeline, text_with_id):
            # Run async synthesis in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(pipeline.synthesize(text_with_id))
            finally:
                loop.close()
        
        # Run synthesis in multiple threads
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(3):
                future = executor.submit(sync_synthesis_wrapper, voirs_pipeline, f"{text} Thread {i}")
                futures.append(future)
            
            # Wait for all threads to complete
            results = [future.result(timeout=30) for future in futures]
        
        # All should succeed
        assert len(results) == 3
        for audio in results:
            assert audio.duration > 0
        
        print(f"Threading compatibility test: {len(results)} threads completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
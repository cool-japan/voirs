"""
Stress tests for VoiRS Python bindings.

These tests push the system to its limits to identify potential
issues under extreme conditions.
"""

import pytest
import asyncio
import time
import gc
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import tempfile
import os
import signal

try:
    import voirs_ffi as voirs
    VOIRS_AVAILABLE = True
except ImportError:
    VOIRS_AVAILABLE = False

pytestmark = [pytest.mark.stress, pytest.mark.slow, pytest.mark.skipif(not VOIRS_AVAILABLE, reason="VoiRS not available")]


class TestMemoryStress:
    """Test memory-intensive scenarios."""
    
    async def test_memory_pressure_synthesis(self, voirs_pipeline, sample_texts, memory_tracker):
        """Test synthesis under memory pressure."""
        memory_tracker.start_tracking()
        text = sample_texts["long"]
        
        # Create memory pressure by holding references
        audio_buffers = []
        
        try:
            # Synthesize many times and keep references
            for i in range(50):
                audio = await voirs_pipeline.synthesize(f"{text} {i}")
                audio_buffers.append(audio)
                
                # Check memory usage periodically
                if i % 10 == 0:
                    memory_usage = memory_tracker.get_memory_usage()
                    print(f"Iteration {i}: {memory_usage['current_mb']:.2f} MB")
                    
                    # Stop if memory usage becomes excessive
                    if memory_usage['current_mb'] > 500:  # 500MB limit
                        print(f"Memory limit reached at iteration {i}")
                        break
        
        finally:
            # Clean up
            del audio_buffers
            gc.collect()
        
        final_memory = memory_tracker.get_memory_usage()
        print(f"Final memory after cleanup: {final_memory['current_mb']:.2f} MB")
        
        # Memory should be manageable
        assert final_memory['current_mb'] < 1000, f"Excessive memory usage: {final_memory['current_mb']:.2f} MB"
    
    async def test_fragmented_memory_synthesis(self, voirs_pipeline, sample_texts):
        """Test synthesis with fragmented memory."""
        text = sample_texts["medium"]
        
        # Create memory fragmentation
        fragments = []
        for i in range(100):
            fragment = [0] * (1000 * (i % 10 + 1))  # Variable size allocations
            fragments.append(fragment)
        
        try:
            # Synthesize under fragmented memory conditions
            for i in range(10):
                audio = await voirs_pipeline.synthesize(f"{text} {i}")
                assert audio.duration > 0
                
                # Free some fragments randomly
                if i % 3 == 0 and fragments:
                    fragments.pop(0)
                    gc.collect()
        
        finally:
            # Clean up
            del fragments
            gc.collect()
    
    async def test_low_memory_conditions(self, voirs_pipeline, sample_texts, memory_tracker):
        """Test behavior under low memory conditions."""
        memory_tracker.start_tracking()
        text = sample_texts["short"]
        
        # Create large memory allocation to simulate low memory
        try:
            # Allocate significant memory
            large_allocation = bytearray(100 * 1024 * 1024)  # 100MB
            
            # Try synthesis under memory pressure
            for i in range(5):
                audio = await voirs_pipeline.synthesize(f"{text} {i}")
                assert audio.duration > 0
                
        except MemoryError:
            pytest.skip("System doesn't have enough memory for this test")
        finally:
            # Clean up
            if 'large_allocation' in locals():
                del large_allocation
            gc.collect()


class TestConcurrencyStress:
    """Test high-concurrency scenarios."""
    
    async def test_extreme_concurrent_synthesis(self, voirs_pipeline, sample_texts):
        """Test synthesis with extreme concurrency."""
        text = sample_texts["short"]
        
        # Create many concurrent tasks
        num_tasks = 100
        tasks = []
        
        for i in range(num_tasks):
            task = voirs_pipeline.synthesize(f"{text} {i}")
            tasks.append(task)
        
        # Execute all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120.0  # 2 minutes timeout
            )
        except asyncio.TimeoutError:
            pytest.fail("Extreme concurrency test timed out")
        
        # Analyze results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful) / len(results)
        print(f"Extreme concurrency: {len(successful)}/{len(results)} successful ({success_rate:.1%})")
        
        # Should handle reasonable percentage of requests
        assert success_rate >= 0.5, f"Too many concurrent requests failed: {success_rate:.1%}"
        
        # Log failure reasons
        if failed:
            failure_types = {}
            for error in failed[:5]:  # Show first 5 failures
                error_type = type(error).__name__
                failure_types[error_type] = failure_types.get(error_type, 0) + 1
            print(f"Failure types: {failure_types}")
    
    async def test_rapid_pipeline_creation(self):
        """Test rapid creation of multiple pipelines."""
        pipelines = []
        creation_tasks = []
        
        try:
            # Create many pipelines concurrently
            for i in range(10):
                task = voirs.VoirsPipeline.create()
                creation_tasks.append(task)
            
            # Wait for all creations
            results = await asyncio.gather(*creation_tasks, return_exceptions=True)
            
            # Check results
            successful_pipelines = [r for r in results if not isinstance(r, Exception)]
            failed_creations = [r for r in results if isinstance(r, Exception)]
            
            pipelines.extend(successful_pipelines)
            
            print(f"Pipeline creation: {len(successful_pipelines)}/{len(results)} successful")
            
            # At least half should succeed
            assert len(successful_pipelines) >= 5, f"Too few pipelines created: {len(successful_pipelines)}"
            
            # Test that created pipelines work
            if successful_pipelines:
                test_audio = await successful_pipelines[0].synthesize("Test")
                assert test_audio.duration > 0
        
        finally:
            # Clean up pipelines
            del pipelines
            gc.collect()
    
    async def test_mixed_workload_stress(self, voirs_pipeline, sample_texts):
        """Test mixed workload with different operation types."""
        tasks = []
        
        # Mix different types of operations
        for i in range(30):
            if i % 3 == 0:
                # Synthesis task
                task = voirs_pipeline.synthesize(f"{sample_texts['short']} {i}")
            elif i % 3 == 1:
                # Voice listing task
                task = voirs_pipeline.list_voices()
            else:
                # Configuration synthesis task
                config = {"speaking_rate": 1.0 + (i % 5) * 0.1}
                task = voirs_pipeline.synthesize_with_config(sample_texts['medium'], config)
            
            tasks.append(task)
        
        # Execute mixed workload
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful) / len(results)
        
        print(f"Mixed workload: {len(successful)}/{len(results)} successful ({success_rate:.1%})")
        
        # Should handle majority of mixed operations
        assert success_rate >= 0.7, f"Mixed workload success rate too low: {success_rate:.1%}"


class TestResourceExhaustion:
    """Test behavior when resources are exhausted."""
    
    async def test_file_descriptor_exhaustion(self, voirs_pipeline, sample_texts):
        """Test behavior when file descriptors are exhausted."""
        text = sample_texts["short"]
        
        # Open many file descriptors
        temp_files = []
        try:
            # Open many temporary files
            for i in range(100):
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_files.append(temp_file)
                except OSError:
                    break  # Hit file descriptor limit
            
            print(f"Opened {len(temp_files)} file descriptors")
            
            # Try synthesis with exhausted file descriptors
            for i in range(5):
                audio = await voirs_pipeline.synthesize(f"{text} {i}")
                assert audio.duration > 0
        
        finally:
            # Clean up file descriptors
            for temp_file in temp_files:
                try:
                    temp_file.close()
                    os.unlink(temp_file.name)
                except:
                    pass
    
    async def test_thread_exhaustion(self, voirs_pipeline, sample_texts):
        """Test behavior when thread pool is exhausted."""
        text = sample_texts["short"]
        
        # Create many threads
        threads = []
        stop_event = threading.Event()
        
        def worker():
            while not stop_event.is_set():
                time.sleep(0.1)
        
        try:
            # Start many threads
            for i in range(50):
                try:
                    thread = threading.Thread(target=worker)
                    thread.start()
                    threads.append(thread)
                except RuntimeError:
                    break  # Hit thread limit
            
            print(f"Created {len(threads)} threads")
            
            # Try synthesis with thread exhaustion
            for i in range(3):
                audio = await voirs_pipeline.synthesize(f"{text} {i}")
                assert audio.duration > 0
        
        finally:
            # Clean up threads
            stop_event.set()
            for thread in threads:
                try:
                    thread.join(timeout=1.0)
                except:
                    pass
    
    async def test_disk_space_exhaustion(self, voirs_pipeline, sample_texts, temp_audio_dir):
        """Test behavior when disk space is low."""
        text = sample_texts["short"]
        
        # Try to fill up disk space (safely)
        large_files = []
        try:
            # Create large files to simulate disk space issues
            for i in range(5):
                large_file = temp_audio_dir / f"large_file_{i}.dat"
                try:
                    with open(large_file, 'wb') as f:
                        f.write(b'0' * (10 * 1024 * 1024))  # 10MB each
                    large_files.append(large_file)
                except OSError:
                    break  # Disk full or error
            
            print(f"Created {len(large_files)} large files")
            
            # Try synthesis under disk pressure
            for i in range(3):
                audio = await voirs_pipeline.synthesize(f"{text} {i}")
                assert audio.duration > 0
        
        finally:
            # Clean up large files
            for large_file in large_files:
                try:
                    large_file.unlink()
                except:
                    pass


class TestLongRunningStress:
    """Test long-running stress scenarios."""
    
    async def test_continuous_synthesis_stress(self, voirs_pipeline, sample_texts, memory_tracker):
        """Test continuous synthesis over extended period."""
        memory_tracker.start_tracking()
        text = sample_texts["medium"]
        
        start_time = time.time()
        max_duration = 300.0  # 5 minutes
        operations = 0
        errors = 0
        
        try:
            while time.time() - start_time < max_duration:
                try:
                    audio = await voirs_pipeline.synthesize(f"{text} {operations}")
                    assert audio.duration > 0
                    operations += 1
                    
                    # Clean up periodically
                    if operations % 20 == 0:
                        gc.collect()
                        memory_usage = memory_tracker.get_memory_usage()
                        print(f"Operations: {operations}, Memory: {memory_usage['current_mb']:.2f} MB")
                        
                        # Check for memory leaks
                        if memory_tracker.check_memory_leak(threshold_mb=200.0):
                            print("Memory leak detected, stopping test")
                            break
                
                except Exception as e:
                    errors += 1
                    print(f"Error at operation {operations}: {e}")
                    
                    # Too many errors, stop
                    if errors > 10:
                        break
                    
                    # Wait before retry
                    await asyncio.sleep(0.1)
        
        finally:
            actual_duration = time.time() - start_time
            final_memory = memory_tracker.get_memory_usage()
            
            print(f"Continuous stress test:")
            print(f"  Duration: {actual_duration:.2f}s")
            print(f"  Operations: {operations}")
            print(f"  Errors: {errors}")
            print(f"  Rate: {operations/actual_duration:.2f} ops/sec")
            print(f"  Final memory: {final_memory['current_mb']:.2f} MB")
            
            # Should complete reasonable number of operations
            assert operations > 50, f"Too few operations completed: {operations}"
            
            # Error rate should be reasonable
            error_rate = errors / max(operations, 1)
            assert error_rate < 0.1, f"Too many errors: {error_rate:.1%}"
    
    async def test_endurance_with_variations(self, voirs_pipeline, sample_texts):
        """Test endurance with varying workloads."""
        texts = [sample_texts["short"], sample_texts["medium"], sample_texts["long"]]
        
        start_time = time.time()
        max_duration = 180.0  # 3 minutes
        operations = 0
        
        while time.time() - start_time < max_duration:
            # Vary the workload
            text = texts[operations % len(texts)]
            
            try:
                if operations % 5 == 0:
                    # Occasionally use configuration
                    config = {"speaking_rate": 1.0 + (operations % 10) * 0.1}
                    audio = await voirs_pipeline.synthesize_with_config(text, config)
                else:
                    # Regular synthesis
                    audio = await voirs_pipeline.synthesize(text)
                
                assert audio.duration > 0
                operations += 1
                
                # Occasional voice operations
                if operations % 20 == 0:
                    voices = await voirs_pipeline.list_voices()
                    if voices:
                        await voirs_pipeline.set_voice(voices[0].id)
            
            except Exception as e:
                print(f"Error at operation {operations}: {e}")
                # Continue with next operation
                await asyncio.sleep(0.1)
        
        actual_duration = time.time() - start_time
        rate = operations / actual_duration
        
        print(f"Endurance test: {operations} operations in {actual_duration:.2f}s ({rate:.2f} ops/sec)")
        
        # Should maintain reasonable performance
        assert operations > 30, f"Too few operations: {operations}"
        assert rate > 0.1, f"Performance too low: {rate:.3f} ops/sec"


class TestRecoveryStress:
    """Test recovery from stress conditions."""
    
    async def test_recovery_after_overload(self, voirs_pipeline, sample_texts):
        """Test recovery after system overload."""
        text = sample_texts["short"]
        
        # Phase 1: Overload the system
        print("Phase 1: Overloading system...")
        overload_tasks = []
        for i in range(200):
            task = voirs_pipeline.synthesize(f"{text} overload {i}")
            overload_tasks.append(task)
        
        # Wait for overload to complete (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(*overload_tasks, return_exceptions=True),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            print("Overload phase timed out (expected)")
        
        # Phase 2: Cool down period
        print("Phase 2: Cool down period...")
        await asyncio.sleep(5.0)
        gc.collect()
        
        # Phase 3: Test recovery
        print("Phase 3: Testing recovery...")
        recovery_results = []
        for i in range(10):
            try:
                audio = await voirs_pipeline.synthesize(f"{text} recovery {i}")
                assert audio.duration > 0
                recovery_results.append(True)
            except Exception as e:
                print(f"Recovery test {i} failed: {e}")
                recovery_results.append(False)
        
        # Should recover reasonably well
        success_rate = sum(recovery_results) / len(recovery_results)
        print(f"Recovery success rate: {success_rate:.1%}")
        
        assert success_rate >= 0.7, f"Poor recovery after overload: {success_rate:.1%}"
    
    async def test_graceful_degradation(self, voirs_pipeline, sample_texts):
        """Test graceful degradation under stress."""
        text = sample_texts["medium"]
        
        # Gradually increase load and measure performance
        load_levels = [1, 5, 10, 20, 50]
        performance_results = {}
        
        for load_level in load_levels:
            print(f"Testing load level: {load_level}")
            
            start_time = time.time()
            tasks = []
            
            for i in range(load_level):
                task = voirs_pipeline.synthesize(f"{text} load {i}")
                tasks.append(task)
            
            # Execute and measure
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Calculate metrics
            successful = [r for r in results if not isinstance(r, Exception)]
            success_rate = len(successful) / len(results)
            avg_time = (end_time - start_time) / load_level
            
            performance_results[load_level] = {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'total_time': end_time - start_time
            }
            
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Average time: {avg_time:.2f}s")
            
            # Brief pause between load levels
            await asyncio.sleep(1.0)
        
        # Analyze graceful degradation
        print("\nPerformance degradation analysis:")
        for load_level, metrics in performance_results.items():
            print(f"Load {load_level}: {metrics['success_rate']:.1%} success, {metrics['avg_time']:.2f}s avg")
        
        # Even under high load, should maintain some level of service
        high_load_success = performance_results[max(load_levels)]['success_rate']
        assert high_load_success > 0.3, f"Poor degradation under high load: {high_load_success:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure for stress tests
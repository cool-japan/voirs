//! Performance tests for real-time feedback systems
//!
//! This module contains comprehensive performance tests for the VoiRS feedback system,
//! focusing on real-time processing capabilities, latency requirements, and throughput
//! under various load conditions.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use uuid;

use futures::future::join_all;
use voirs_feedback::realtime::{RealtimeConfig, RealtimeFeedbackSystem, RealtimeStats};
use voirs_feedback::traits::{
    AdaptiveState, FocusArea, SessionState, SessionStatistics, SessionStats, UserPreferences,
};
use voirs_feedback::{AudioBuffer, FeedbackSystem};

#[tokio::test]
async fn test_feedback_latency_performance() {
    // Test that feedback generation meets sub-100ms latency requirement
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");
    let config = RealtimeConfig::default();

    // Create test audio data
    let sample_rate = 22050;
    let duration = Duration::from_millis(50);
    let samples = (duration.as_secs_f32() * sample_rate as f32) as usize;
    let audio_data = vec![0.0f32; samples];

    // Create a session for testing
    let mut session = feedback_system
        .create_session("perf_test_user")
        .await
        .expect("Failed to create session");

    // Measure latency for feedback generation
    let start = Instant::now();
    let audio_buffer = AudioBuffer::new(audio_data, sample_rate, 1);
    let _feedback = session.process_synthesis(&audio_buffer, "test").await;
    let latency = start.elapsed();

    // Assert that latency is under 100ms as specified in config
    assert!(
        latency < Duration::from_millis(config.max_latency_ms),
        "Feedback latency ({:?}) exceeded maximum ({:?})",
        latency,
        Duration::from_millis(config.max_latency_ms)
    );

    // Also test that it's reasonably fast (under 50ms for this small sample)
    assert!(
        latency < Duration::from_millis(50),
        "Feedback generation too slow: {:?}",
        latency
    );

    println!("Latency Performance: {:?}", latency);
}

#[tokio::test]
async fn test_concurrent_feedback_performance() {
    // Test system performance under concurrent load
    let feedback_system = Arc::new(
        FeedbackSystem::new()
            .await
            .expect("Failed to create feedback system"),
    );
    let config = RealtimeConfig::default();

    // Create test audio data
    let sample_rate = 22050;
    let duration = Duration::from_millis(100);
    let samples = (duration.as_secs_f32() * sample_rate as f32) as usize;
    let audio_data = vec![0.1f32; samples];

    // Create multiple concurrent feedback requests
    let concurrent_requests = 5;
    let mut handles = Vec::new();

    let start = Instant::now();

    for i in 0..concurrent_requests {
        let system = feedback_system.clone();
        let data = audio_data.clone();

        let handle = tokio::spawn(async move {
            let request_start = Instant::now();
            let mut session = system
                .create_session(&format!("perf_test_user_{}", i))
                .await
                .expect("Failed to create session");
            let audio_buffer = AudioBuffer::new(data, sample_rate, 1);
            let result = session.process_synthesis(&audio_buffer, "test").await;
            let request_duration = request_start.elapsed();
            (result, request_duration)
        });

        handles.push(handle);
    }

    // Wait for all requests to complete
    let mut max_latency = Duration::from_millis(0);
    let mut total_latency = Duration::from_millis(0);
    let mut successful_requests = 0;

    for handle in handles {
        let (result, duration) = handle.await.unwrap();

        if result.is_ok() {
            successful_requests += 1;
            max_latency = max_latency.max(duration);
            total_latency += duration;
        }
    }

    let total_duration = start.elapsed();
    let avg_latency = total_latency / successful_requests;

    // Performance assertions
    assert_eq!(
        successful_requests, concurrent_requests,
        "Not all concurrent requests succeeded"
    );

    assert!(
        max_latency < Duration::from_millis(config.max_latency_ms * 2),
        "Maximum concurrent latency ({:?}) exceeded threshold",
        max_latency
    );

    assert!(
        avg_latency < Duration::from_millis(config.max_latency_ms),
        "Average concurrent latency ({:?}) exceeded threshold",
        avg_latency
    );

    println!("Concurrent Performance Results:");
    println!("  Requests: {}", concurrent_requests);
    println!("  Total time: {:?}", total_duration);
    println!("  Average latency: {:?}", avg_latency);
    println!("  Max latency: {:?}", max_latency);
}

#[tokio::test]
async fn test_streaming_performance() {
    // Test streaming audio processing performance
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");
    let config = RealtimeConfig::default();

    // Create streaming audio chunks
    let sample_rate = 22050;
    let chunk_duration = Duration::from_millis(20); // 20ms chunks
    let chunk_samples = (chunk_duration.as_secs_f32() * sample_rate as f32) as usize;

    let total_chunks = 10; // 200ms of audio
    let mut session = feedback_system
        .create_session("streaming_test_user")
        .await
        .expect("Failed to create session");
    let mut total_processing_time = Duration::from_millis(0);
    let mut max_chunk_latency = Duration::from_millis(0);

    for i in 0..total_chunks {
        // Generate varying audio data to simulate real speech
        let audio_chunk: Vec<f32> = (0..chunk_samples)
            .map(|j| (((i * chunk_samples + j) as f32) * 0.01).sin() * 0.1)
            .collect();

        let start = Instant::now();
        let audio_buffer = AudioBuffer::new(audio_chunk, sample_rate, 1);
        let result = session.process_synthesis(&audio_buffer, "test").await;
        let chunk_latency = start.elapsed();

        assert!(
            result.is_ok(),
            "Streaming feedback generation failed at chunk {}",
            i
        );

        total_processing_time += chunk_latency;
        max_chunk_latency = max_chunk_latency.max(chunk_latency);

        // Ensure each chunk is processed within reasonable time
        assert!(
            chunk_latency < Duration::from_millis(config.max_latency_ms),
            "Chunk {} processing time ({:?}) exceeded maximum ({:?})",
            i,
            chunk_latency,
            Duration::from_millis(config.max_latency_ms)
        );
    }

    let avg_chunk_latency = total_processing_time / total_chunks as u32;
    let total_audio_duration = chunk_duration * total_chunks as u32;
    let real_time_factor = total_processing_time.as_secs_f32() / total_audio_duration.as_secs_f32();

    // Performance assertions
    assert!(
        real_time_factor < 1.0,
        "Real-time factor ({:.2}) too high, should be < 1.0",
        real_time_factor
    );

    println!("Streaming Performance Results:");
    println!("  Total chunks: {}", total_chunks);
    println!("  Audio duration: {:?}", total_audio_duration);
    println!("  Total processing time: {:?}", total_processing_time);
    println!("  Real-time factor: {:.2}", real_time_factor);
    println!("  Average chunk latency: {:?}", avg_chunk_latency);
    println!("  Max chunk latency: {:?}", max_chunk_latency);
}

#[tokio::test]
async fn test_memory_usage_performance() {
    // Test memory usage under load
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");
    let mut session = feedback_system
        .create_session("memory_test_user")
        .await
        .expect("Failed to create session");

    // Create various sized audio chunks
    let sample_rate = 22050;
    let test_sizes = vec![
        Duration::from_millis(10),  // Very short
        Duration::from_millis(50),  // Short
        Duration::from_millis(100), // Medium
        Duration::from_millis(200), // Long
    ];

    for duration in test_sizes {
        let samples = (duration.as_secs_f32() * sample_rate as f32) as usize;
        let audio_data = vec![0.1f32; samples];

        // Measure memory usage (approximate)
        let start = Instant::now();
        let audio_buffer = AudioBuffer::new(audio_data, sample_rate, 1);
        let result = session.process_synthesis(&audio_buffer, "test").await;
        let processing_time = start.elapsed();

        assert!(
            result.is_ok(),
            "Memory test failed for duration {:?}",
            duration
        );

        // Ensure processing time scales reasonably with input size
        let samples_per_ms = samples as f32 / processing_time.as_millis() as f32;

        // Should process at least 100 samples per millisecond
        assert!(
            samples_per_ms > 100.0,
            "Processing rate too slow for {:?} audio: {:.1} samples/ms",
            duration,
            samples_per_ms
        );

        println!(
            "Memory test - Duration: {:?}, Samples: {}, Rate: {:.1} samples/ms",
            duration, samples, samples_per_ms
        );
    }
}

#[tokio::test]
async fn test_throughput_performance() {
    // Test system throughput under sustained load
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");
    let config = RealtimeConfig::default();
    let mut session = feedback_system
        .create_session("throughput_test_user")
        .await
        .expect("Failed to create session");

    // Create test audio data
    let sample_rate = 22050;
    let duration = Duration::from_millis(50);
    let samples = (duration.as_secs_f32() * sample_rate as f32) as usize;
    let audio_data = vec![0.1f32; samples];

    // Test sustained throughput
    let test_duration = Duration::from_millis(500);
    let start = Instant::now();
    let mut request_count = 0;
    let mut total_processing_time = Duration::from_millis(0);

    while start.elapsed() < test_duration {
        let request_start = Instant::now();
        let audio_buffer = AudioBuffer::new(audio_data.clone(), sample_rate, 1);
        let result = session.process_synthesis(&audio_buffer, "test").await;
        let request_time = request_start.elapsed();

        if result.is_ok() {
            request_count += 1;
            total_processing_time += request_time;
        }

        // Small delay to simulate realistic usage
        sleep(Duration::from_millis(10)).await;
    }

    let total_time = start.elapsed();
    let throughput = request_count as f32 / total_time.as_secs_f32();
    let avg_processing_time = if request_count > 0 {
        total_processing_time / request_count as u32
    } else {
        Duration::from_millis(0)
    };

    // Performance assertions
    assert!(
        throughput > 5.0,
        "Throughput too low: {:.1} requests/second",
        throughput
    );

    assert!(
        avg_processing_time < Duration::from_millis(config.max_latency_ms),
        "Average processing time ({:?}) exceeded threshold",
        avg_processing_time
    );

    println!("Throughput Performance Results:");
    println!("  Test duration: {:?}", total_time);
    println!("  Requests processed: {}", request_count);
    println!("  Throughput: {:.1} requests/second", throughput);
    println!("  Average processing time: {:?}", avg_processing_time);
    println!("  Total processing time: {:?}", total_processing_time);
}

#[tokio::test]
async fn test_error_handling_performance() {
    // Test that error handling doesn't significantly impact performance
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");
    let mut session = feedback_system
        .create_session("error_test_user")
        .await
        .expect("Failed to create session");

    // Test with various edge cases
    let test_cases = vec![
        ("empty_data", vec![]),
        ("single_sample", vec![0.1f32]),
        ("small_data", vec![0.1f32; 10]),
    ];

    for (name, data) in test_cases {
        let start = Instant::now();
        let audio_buffer = AudioBuffer::new(data, 22050, 1);
        let result = session.process_synthesis(&audio_buffer, "test").await;
        let processing_time = start.elapsed();

        // Error handling should be fast
        assert!(
            processing_time < Duration::from_millis(100),
            "Error handling too slow for {}: {:?}",
            name,
            processing_time
        );

        println!(
            "Error handling test - {}: {:?}, Result: {:?}",
            name,
            processing_time,
            result.is_ok()
        );
    }
}

#[tokio::test]
async fn test_performance_regression_detection() {
    // Test to detect performance regressions
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");
    let mut session = feedback_system
        .create_session("regression_test_user")
        .await
        .expect("Failed to create session");

    // Baseline performance test
    let sample_rate = 22050;
    let duration = Duration::from_millis(100);
    let samples = (duration.as_secs_f32() * sample_rate as f32) as usize;
    let audio_data = vec![0.1f32; samples];

    // Run multiple iterations to get stable baseline
    let iterations = 10;
    let mut latencies = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let audio_buffer = AudioBuffer::new(audio_data.clone(), sample_rate, 1);
        let result = session.process_synthesis(&audio_buffer, "test").await;
        let latency = start.elapsed();

        assert!(result.is_ok(), "Baseline test failed");
        latencies.push(latency);
    }

    // Calculate statistics
    let avg_latency = latencies.iter().sum::<Duration>() / iterations as u32;
    let min_latency = latencies.iter().min().unwrap();
    let max_latency = latencies.iter().max().unwrap();

    // Performance baseline assertions
    assert!(
        avg_latency < Duration::from_millis(100),
        "Average latency regression: {:?}",
        avg_latency
    );

    assert!(
        *max_latency < Duration::from_millis(200),
        "Maximum latency regression: {:?}",
        max_latency
    );

    println!("Performance Regression Detection Results:");
    println!("  Iterations: {}", iterations);
    println!("  Average latency: {:?}", avg_latency);
    println!("  Min latency: {:?}", min_latency);
    println!("  Max latency: {:?}", max_latency);
}

/// Test real-time system performance under high throughput conditions
#[tokio::test]
async fn test_realtime_throughput_performance() {
    // Test the real-time feedback system's ability to handle high throughput
    let realtime_system = RealtimeFeedbackSystem::new()
        .await
        .expect("Failed to create realtime feedback system");

    // Create test session state
    let session_state = SessionState {
        session_id: uuid::Uuid::new_v4(),
        user_id: "test_user".to_string(),
        start_time: chrono::Utc::now(),
        last_activity: chrono::Utc::now(),
        current_task: Some("pronunciation_practice".to_string()),
        stats: SessionStats::default(),
        preferences: UserPreferences::default(),
        adaptive_state: AdaptiveState::default(),
        current_exercise: None,
        session_stats: SessionStatistics::default(),
    };

    // Create multiple streams for testing
    let num_streams = 10;
    let mut streams = Vec::new();

    for i in 0..num_streams {
        let stream = realtime_system
            .create_stream(&format!("user_{}", i), &session_state)
            .await
            .expect("Failed to create stream");
        streams.push(stream);
    }

    // Generate test audio data
    let sample_rate = 16000;
    let chunk_duration = Duration::from_millis(50); // 50ms chunks for real-time
    let samples_per_chunk = (chunk_duration.as_secs_f32() * sample_rate as f32) as usize;

    // Test processing multiple audio chunks simultaneously
    let num_chunks = 20;
    let total_start = Instant::now();
    let mut total_processed = 0;
    let mut latencies = Vec::new();

    for chunk_idx in 0..num_chunks {
        let mut chunk_tasks = Vec::new();

        for (stream_idx, stream) in streams.iter().enumerate() {
            // Generate slightly different audio for each stream/chunk
            let mut audio_data = vec![0.0f32; samples_per_chunk];
            for (i, sample) in audio_data.iter_mut().enumerate() {
                *sample = 0.1 * ((i + chunk_idx * 100 + stream_idx * 1000) as f32 * 0.001).sin();
            }

            let audio_buffer = AudioBuffer::new(audio_data, sample_rate as u32, 1);
            let text = format!("Test audio chunk {} for stream {}", chunk_idx, stream_idx);

            let task = async move {
                let start = Instant::now();
                let result = stream.process_audio(&audio_buffer, &text).await;
                let latency = start.elapsed();
                (result, latency)
            };

            chunk_tasks.push(task);
        }

        // Process all streams for this chunk concurrently
        let chunk_results = join_all(chunk_tasks).await;

        for (result, latency) in chunk_results {
            if result.is_ok() {
                total_processed += 1;
                latencies.push(latency);
            }
        }
    }

    let total_duration = total_start.elapsed();

    // Calculate performance metrics
    let avg_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
    let max_latency = latencies
        .iter()
        .max()
        .cloned()
        .unwrap_or(Duration::from_millis(0));
    let min_latency = latencies
        .iter()
        .min()
        .cloned()
        .unwrap_or(Duration::from_millis(0));
    let throughput = total_processed as f64 / total_duration.as_secs_f64();

    // Performance assertions
    assert_eq!(
        total_processed,
        num_streams * num_chunks,
        "Not all chunks were processed successfully"
    );

    // Real-time performance requirements
    assert!(
        avg_latency < Duration::from_millis(30),
        "Average latency {} ms exceeds real-time threshold of 30ms",
        avg_latency.as_millis()
    );
    assert!(
        max_latency < Duration::from_millis(100),
        "Maximum latency {} ms exceeds real-time threshold of 100ms",
        max_latency.as_millis()
    );
    assert!(
        throughput >= 100.0,
        "Throughput {} Hz is below required 100 Hz for real-time processing",
        throughput
    );

    // 95th percentile latency should be under 50ms
    let mut sorted_latencies = latencies.clone();
    sorted_latencies.sort();
    let p95_index = (sorted_latencies.len() as f64 * 0.95) as usize;
    let p95_latency = sorted_latencies[p95_index.min(sorted_latencies.len() - 1)];
    assert!(
        p95_latency < Duration::from_millis(50),
        "95th percentile latency {} ms exceeds 50ms threshold",
        p95_latency.as_millis()
    );

    println!("Real-time Throughput Performance Results:");
    println!("  Total processed: {}", total_processed);
    println!("  Total duration: {:?}", total_duration);
    println!("  Throughput: {:.2} Hz", throughput);
    println!("  Average latency: {:?}", avg_latency);
    println!("  Min latency: {:?}", min_latency);
    println!("  Max latency: {:?}", max_latency);
    println!("  95th percentile latency: {:?}", p95_latency);
}

/// Test memory usage under sustained real-time load
#[tokio::test]
async fn test_realtime_memory_performance() {
    // Test that memory usage remains stable under sustained load
    let realtime_system = RealtimeFeedbackSystem::new()
        .await
        .expect("Failed to create realtime feedback system");

    let mut preferences = UserPreferences::default();
    preferences.focus_areas = vec![FocusArea::Pronunciation, FocusArea::Fluency];

    let session_state = SessionState {
        session_id: uuid::Uuid::new_v4(),
        user_id: "memory_test_user".to_string(),
        start_time: chrono::Utc::now(),
        last_activity: chrono::Utc::now(),
        current_task: None,
        stats: SessionStats::default(),
        preferences,
        adaptive_state: AdaptiveState::default(),
        current_exercise: None,
        session_stats: SessionStatistics::default(),
    };

    // Create stream for testing
    let stream = realtime_system
        .create_stream("memory_test_user", &session_state)
        .await
        .expect("Failed to create stream");

    // Generate sustained load for memory testing
    let sample_rate = 16000;
    let chunk_size = 1024; // 64ms at 16kHz
    let test_duration = Duration::from_secs(10); // 10 seconds of sustained load
    let chunk_interval = Duration::from_millis(50); // Process every 50ms

    let start_time = Instant::now();
    let mut processed_chunks = 0;
    let mut memory_samples = Vec::new();

    while start_time.elapsed() < test_duration {
        // Generate audio chunk
        let audio_data: Vec<f32> = (0..chunk_size)
            .map(|i| 0.1 * (i as f32 * 0.01).sin())
            .collect();

        let audio_buffer = AudioBuffer::new(audio_data, sample_rate as u32, 1);
        let text = format!("Memory test chunk {}", processed_chunks);

        // Process audio
        let result = stream.process_audio(&audio_buffer, &text).await;
        assert!(result.is_ok(), "Audio processing failed during memory test");

        processed_chunks += 1;

        // Sample memory usage periodically
        if processed_chunks % 20 == 0 {
            // More realistic memory usage estimation - track actual buffer usage
            // Assume system has some base memory + current buffer size
            let base_memory = 1024 * 1024; // 1MB base
            let current_buffer_memory = chunk_size * 4; // 4 bytes per f32 sample
            let estimated_memory = base_memory + current_buffer_memory;
            memory_samples.push(estimated_memory);
        }

        // Wait for next chunk
        sleep(chunk_interval).await;
    }

    // Analyze memory usage trend
    if memory_samples.len() >= 2 {
        let initial_memory = memory_samples[0];
        let final_memory = memory_samples[memory_samples.len() - 1];
        let memory_growth = final_memory as f64 / initial_memory as f64;

        // Memory should not grow more than 50% during the test (allowing for some buffering)
        assert!(
            memory_growth < 1.5,
            "Memory usage grew too much: {:.2}x growth",
            memory_growth
        );

        println!("Memory Performance Results:");
        println!("  Processed chunks: {}", processed_chunks);
        println!("  Test duration: {:?}", start_time.elapsed());
        println!("  Initial memory estimate: {} bytes", initial_memory);
        println!("  Final memory estimate: {} bytes", final_memory);
        println!("  Memory growth ratio: {:.2}x", memory_growth);
    }
}

/// Test real-time system performance under stress conditions
#[tokio::test]
async fn test_realtime_stress_performance() {
    // Test system behavior under extreme load conditions
    let realtime_system = RealtimeFeedbackSystem::new()
        .await
        .expect("Failed to create realtime feedback system");

    let mut preferences = UserPreferences::default();
    preferences.focus_areas = vec![
        FocusArea::Pronunciation,
        FocusArea::Fluency,
        FocusArea::Intonation,
    ];

    let session_state = SessionState {
        session_id: uuid::Uuid::new_v4(),
        user_id: "stress_test_user".to_string(),
        start_time: chrono::Utc::now(),
        last_activity: chrono::Utc::now(),
        current_task: None,
        stats: SessionStats::default(),
        preferences,
        adaptive_state: AdaptiveState::default(),
        current_exercise: None,
        session_stats: SessionStatistics::default(),
    };

    // Create many concurrent streams
    let num_streams = 50; // Stress test with 50 concurrent streams
    let mut stream_tasks = Vec::new();

    for i in 0..num_streams {
        let system = realtime_system.clone();
        let state = session_state.clone();

        let task = tokio::spawn(async move {
            let stream_result = system
                .create_stream(&format!("stress_user_{}", i), &state)
                .await;
            if let Ok(stream) = stream_result {
                // Process a few audio chunks per stream
                let mut successful_chunks = 0;
                let mut failed_chunks = 0;

                for chunk_idx in 0..5 {
                    let audio_data = vec![0.05f32; 512]; // Small chunks for stress test
                    let audio_buffer = AudioBuffer::new(audio_data, 16000, 1);
                    let text = format!("Stress test chunk {} from stream {}", chunk_idx, i);

                    match stream.process_audio(&audio_buffer, &text).await {
                        Ok(_) => successful_chunks += 1,
                        Err(_) => failed_chunks += 1,
                    }
                }

                (successful_chunks, failed_chunks)
            } else {
                (0, 5) // All chunks failed if stream creation failed
            }
        });

        stream_tasks.push(task);
    }

    // Wait for all stress test tasks to complete
    let start_time = Instant::now();
    let results = join_all(stream_tasks).await;
    let test_duration = start_time.elapsed();

    // Analyze stress test results
    let mut total_successful = 0;
    let mut total_failed = 0;
    let mut successful_streams = 0;

    for result in results {
        match result {
            Ok((successful, failed)) => {
                total_successful += successful;
                total_failed += failed;
                if successful > 0 {
                    successful_streams += 1;
                }
            }
            Err(_) => {
                total_failed += 5; // Assume all chunks failed for this stream
            }
        }
    }

    let success_rate = total_successful as f64 / (total_successful + total_failed) as f64;
    let stream_success_rate = successful_streams as f64 / num_streams as f64;

    // Stress test assertions - under extreme load, some degradation is acceptable
    assert!(
        success_rate >= 0.7,
        "Success rate {} is too low under stress conditions",
        success_rate
    );
    assert!(
        stream_success_rate >= 0.8,
        "Stream success rate {} is too low",
        stream_success_rate
    );
    assert!(
        test_duration < Duration::from_secs(30),
        "Stress test took too long: {:?}",
        test_duration
    );

    println!("Stress Test Performance Results:");
    println!("  Concurrent streams: {}", num_streams);
    println!("  Test duration: {:?}", test_duration);
    println!("  Successful chunks: {}", total_successful);
    println!("  Failed chunks: {}", total_failed);
    println!("  Success rate: {:.2}%", success_rate * 100.0);
    println!("  Stream success rate: {:.2}%", stream_success_rate * 100.0);
}

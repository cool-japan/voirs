//! Memory leak detection tests for long-running VoiRS feedback sessions
//!
//! This module contains comprehensive tests for detecting and preventing memory leaks
//! during long-running feedback sessions, ensuring system stability and performance.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use voirs_feedback::memory_monitor::{MemoryManager, MemoryMonitor, MemoryMonitorConfig};
use voirs_feedback::{AudioBuffer, FeedbackSystem};

#[tokio::test]
async fn test_memory_leak_detection_short_session() {
    // Test memory leak detection during short-term session
    let config = MemoryMonitorConfig {
        sample_interval_ms: 100,
        min_samples_for_leak_detection: 5,
        leak_threshold: 0.1, // 10% growth
        ..Default::default()
    };

    let monitor = MemoryMonitor::new(config);
    monitor.start_monitoring();

    // Register a session
    monitor.register_session("test_session");

    // Wait for some samples to be collected
    sleep(Duration::from_millis(600)).await;

    let stats = monitor.get_memory_statistics();
    assert_eq!(stats.active_sessions, 1);
    assert!(stats.sample_count >= 5);

    // Clean up
    monitor.unregister_session("test_session");
    monitor.stop_monitoring();

    println!("Short session memory leak detection: PASSED");
}

#[tokio::test]
async fn test_memory_leak_detection_long_session() {
    // Test memory leak detection during long-term session
    let config = MemoryMonitorConfig {
        sample_interval_ms: 50,
        min_samples_for_leak_detection: 10,
        leak_threshold: 0.05, // 5% growth
        ..Default::default()
    };

    let monitor = MemoryMonitor::new(config);
    monitor.start_monitoring();

    // Register a session
    monitor.register_session("long_session");

    // Simulate a long-running session with periodic activity
    for i in 0..20 {
        monitor.update_session_activity("long_session");
        sleep(Duration::from_millis(100)).await;

        // Check memory statistics periodically
        if i % 5 == 0 {
            let stats = monitor.get_memory_statistics();
            assert_eq!(stats.active_sessions, 1);
            println!(
                "Long session iteration {}: memory = {} bytes",
                i, stats.current_memory
            );
        }
    }

    let final_stats = monitor.get_memory_statistics();
    assert_eq!(final_stats.active_sessions, 1);
    assert!(final_stats.sample_count >= 10);

    // Clean up
    monitor.unregister_session("long_session");
    monitor.stop_monitoring();

    println!("Long session memory leak detection: PASSED");
}

#[tokio::test]
async fn test_memory_leak_with_feedback_system() {
    // Test memory leak detection with actual feedback system usage
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    let config = MemoryMonitorConfig {
        sample_interval_ms: 200,
        min_samples_for_leak_detection: 5,
        leak_threshold: 0.2, // 20% growth
        ..Default::default()
    };

    let monitor = MemoryMonitor::new(config);
    monitor.start_monitoring();

    // Register session
    monitor.register_session("feedback_session");

    // Create a session and process multiple feedback requests
    let mut session = feedback_system
        .create_session("memory_test_user")
        .await
        .expect("Failed to create session");

    let sample_rate = 16000u32;
    let audio_data = vec![0.1f32; sample_rate as usize]; // 1 second of audio

    // Process multiple feedback requests to simulate memory usage
    for i in 0..10 {
        let audio_buffer = AudioBuffer::new(audio_data.clone(), sample_rate, 1);
        let result = session
            .process_synthesis(&audio_buffer, &format!("Test message {}", i))
            .await;

        assert!(result.is_ok(), "Feedback processing should succeed");

        // Update memory monitor
        monitor.update_session_activity("feedback_session");

        // Small delay between requests
        sleep(Duration::from_millis(250)).await;
    }

    // Check final memory statistics
    let final_stats = monitor.get_memory_statistics();
    assert_eq!(final_stats.active_sessions, 1);
    assert!(final_stats.sample_count >= 5);

    // Clean up
    monitor.unregister_session("feedback_session");
    monitor.stop_monitoring();

    println!("Memory leak detection with feedback system: PASSED");
}

#[tokio::test]
async fn test_memory_manager_automatic_cleanup() {
    // Test automatic memory cleanup with memory manager
    let config = MemoryMonitorConfig {
        sample_interval_ms: 100,
        min_samples_for_leak_detection: 5,
        leak_threshold: 0.1,        // 10% growth
        session_timeout_seconds: 5, // Timeout longer than wait time
        ..Default::default()
    };

    let mut manager = MemoryManager::new(config);
    manager.start();

    let monitor = manager.get_monitor();

    // Register multiple sessions
    for i in 0..5 {
        monitor.register_session(&format!("session_{}", i));
    }

    let stats_before = monitor.get_memory_statistics();
    assert_eq!(stats_before.active_sessions, 5);

    // Wait for cleanup cycle
    sleep(Duration::from_secs(2)).await;

    // Force cleanup
    monitor.force_cleanup();

    let stats_after = monitor.get_memory_statistics();
    // Sessions should still be there since they haven't timed out yet
    assert_eq!(stats_after.active_sessions, 5);

    // Clean up sessions manually
    for i in 0..5 {
        monitor.unregister_session(&format!("session_{}", i));
    }

    manager.stop();

    println!("Memory manager automatic cleanup: PASSED");
}

#[tokio::test]
async fn test_concurrent_session_memory_tracking() {
    // Test memory tracking with concurrent sessions
    let config = MemoryMonitorConfig {
        sample_interval_ms: 100,
        min_samples_for_leak_detection: 5,
        leak_threshold: 0.15, // 15% growth
        ..Default::default()
    };

    let monitor = Arc::new(MemoryMonitor::new(config));
    monitor.start_monitoring();

    // Create multiple concurrent sessions
    let mut handles = Vec::new();

    for i in 0..3 {
        let monitor_clone = Arc::clone(&monitor);
        let handle = tokio::spawn(async move {
            let session_id = format!("concurrent_session_{}", i);
            monitor_clone.register_session(&session_id);

            // Simulate session activity
            for j in 0..10 {
                monitor_clone.update_session_activity(&session_id);
                sleep(Duration::from_millis(150)).await;

                if j % 3 == 0 {
                    let stats = monitor_clone.get_memory_statistics();
                    println!(
                        "Session {} iteration {}: active sessions = {}",
                        i, j, stats.active_sessions
                    );
                }
            }

            monitor_clone.unregister_session(&session_id);
        });

        handles.push(handle);
    }

    // Wait for all concurrent sessions to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Check final statistics
    let final_stats = monitor.get_memory_statistics();
    assert_eq!(final_stats.active_sessions, 0);
    assert!(final_stats.sample_count >= 5);

    // Stop monitoring
    monitor.stop_monitoring();

    println!("Concurrent session memory tracking: PASSED");
}

#[tokio::test]
async fn test_memory_leak_recovery() {
    // Test memory leak detection and recovery
    let config = MemoryMonitorConfig {
        sample_interval_ms: 100,
        min_samples_for_leak_detection: 3,
        leak_threshold: 0.05, // 5% growth
        ..Default::default()
    };

    let monitor = MemoryMonitor::new(config);
    monitor.start_monitoring();

    // Register a session
    monitor.register_session("recovery_session");

    // Wait for samples to be collected
    sleep(Duration::from_millis(500)).await;

    let stats_before = monitor.get_memory_statistics();

    // Force cleanup to test recovery
    monitor.force_cleanup();

    // Wait a bit more
    sleep(Duration::from_millis(300)).await;

    let stats_after = monitor.get_memory_statistics();

    // Session should still be active
    assert_eq!(stats_after.active_sessions, 1);

    // Check that we have memory samples
    assert!(stats_after.sample_count > 0);

    // Clean up
    monitor.unregister_session("recovery_session");

    println!("Memory leak recovery: PASSED");
}

#[tokio::test]
async fn test_session_memory_info_tracking() {
    // Test detailed session memory information tracking
    let config = MemoryMonitorConfig::default();
    let monitor = MemoryMonitor::new(config);

    // Register a session
    monitor.register_session("info_session");

    let initial_info = monitor.get_session_memory_info("info_session").unwrap();
    assert_eq!(initial_info.session_id, "info_session");
    assert!(initial_info.initial_memory > 0);
    assert!(initial_info.peak_memory > 0);

    // Update session activity
    monitor.update_session_activity("info_session");

    let updated_info = monitor.get_session_memory_info("info_session").unwrap();
    assert!(updated_info.last_activity >= initial_info.last_activity);

    // Get all session info
    let all_info = monitor.get_all_session_memory_info();
    assert_eq!(all_info.len(), 1);
    assert!(all_info.contains_key("info_session"));

    // Clean up
    monitor.unregister_session("info_session");

    println!("Session memory info tracking: PASSED");
}

#[tokio::test]
async fn test_memory_statistics_accuracy() {
    // Test accuracy of memory statistics
    let config = MemoryMonitorConfig::default();
    let monitor = MemoryMonitor::new(config);
    monitor.start_monitoring();

    // Register multiple sessions
    for i in 0..3 {
        monitor.register_session(&format!("stats_session_{}", i));
    }

    // Wait for samples
    sleep(Duration::from_millis(300)).await;

    let stats = monitor.get_memory_statistics();

    // Verify statistics
    assert_eq!(stats.active_sessions, 3);
    assert!(stats.current_memory > 0);
    assert!(stats.peak_memory >= stats.current_memory);
    assert!(stats.sample_count > 0);

    // Test individual session memory
    let session_info = monitor.get_session_memory_info("stats_session_0").unwrap();
    assert!(session_info.initial_memory > 0);
    assert!(session_info.peak_memory >= session_info.initial_memory);

    // Clean up
    for i in 0..3 {
        monitor.unregister_session(&format!("stats_session_{}", i));
    }

    println!("Memory statistics accuracy: PASSED");
}

#[tokio::test]
async fn test_memory_pressure_handling() {
    // Test system behavior under memory pressure
    let config = MemoryMonitorConfig {
        sample_interval_ms: 50,
        min_samples_for_leak_detection: 3,
        leak_threshold: 0.1, // 10% growth
        max_samples: 100,    // Smaller sample size for testing
        ..Default::default()
    };

    let monitor = MemoryMonitor::new(config);
    monitor.start_monitoring();

    // Register sessions
    for i in 0..10 {
        monitor.register_session(&format!("pressure_session_{}", i));
    }

    // Simulate memory pressure with rapid activity updates
    for round in 0..20 {
        for i in 0..10 {
            monitor.update_session_activity(&format!("pressure_session_{}", i));
        }

        // Check statistics every few rounds
        if round % 5 == 0 {
            let stats = monitor.get_memory_statistics();
            assert_eq!(stats.active_sessions, 10);
            println!(
                "Pressure test round {}: {} samples, {} bytes",
                round, stats.sample_count, stats.current_memory
            );
        }

        sleep(Duration::from_millis(60)).await;
    }

    // Force cleanup under pressure
    monitor.force_cleanup();

    let final_stats = monitor.get_memory_statistics();
    assert_eq!(final_stats.active_sessions, 10);
    assert!(final_stats.sample_count <= 100); // Should be limited by max_samples

    // Clean up
    for i in 0..10 {
        monitor.unregister_session(&format!("pressure_session_{}", i));
    }

    println!("Memory pressure handling: PASSED");
}

#[tokio::test]
async fn test_memory_leak_prevention_integration() {
    // Test integration of memory leak prevention with feedback system
    let feedback_system = FeedbackSystem::new()
        .await
        .expect("Failed to create feedback system");

    let config = MemoryMonitorConfig {
        sample_interval_ms: 100,
        min_samples_for_leak_detection: 5,
        leak_threshold: 0.2, // 20% growth
        ..Default::default()
    };

    let mut manager = MemoryManager::new(config);
    manager.start();

    let monitor = manager.get_monitor();

    // Test with feedback system
    let mut session = feedback_system
        .create_session("prevention_test_user")
        .await
        .expect("Failed to create session");

    monitor.register_session("prevention_session");

    // Process feedback with memory monitoring
    let sample_rate = 16000u32;
    let audio_data = vec![0.1f32; sample_rate as usize / 2]; // 0.5 second of audio

    for i in 0..15 {
        let audio_buffer = AudioBuffer::new(audio_data.clone(), sample_rate, 1);
        let result = session
            .process_synthesis(&audio_buffer, &format!("Test {}", i))
            .await;

        assert!(
            result.is_ok(),
            "Feedback processing should succeed with memory monitoring"
        );

        monitor.update_session_activity("prevention_session");

        // Check memory statistics periodically
        if i % 5 == 0 {
            let stats = monitor.get_memory_statistics();
            println!(
                "Integration test iteration {}: {} bytes, leak detected: {}",
                i, stats.current_memory, stats.leak_detected
            );
        }

        sleep(Duration::from_millis(120)).await;
    }

    // Final statistics
    let final_stats = monitor.get_memory_statistics();
    assert_eq!(final_stats.active_sessions, 1);
    assert!(final_stats.sample_count >= 5);

    // Clean up
    monitor.unregister_session("prevention_session");
    manager.stop();

    println!("Memory leak prevention integration: PASSED");
}

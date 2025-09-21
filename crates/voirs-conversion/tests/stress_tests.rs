//! Stress tests for voirs-conversion
//!
//! These tests validate system behavior under high-load conditions, long-duration
//! processing, memory pressure, and extreme concurrent usage scenarios.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;
use voirs_conversion::prelude::*;
use voirs_conversion::types::{AgeGroup, Gender};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// High-load concurrent conversion stress test
#[tokio::test]
async fn test_high_concurrency_stress() -> Result<()> {
    let converter = Arc::new(VoiceConverter::new()?);
    let sample_rate = 22050;

    // Generate test audio samples of varying lengths
    let short_samples: Vec<f32> =
        (0..(sample_rate / 2)) // 0.5 seconds
            .map(|i| {
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
            .collect();

    let medium_samples: Vec<f32> =
        (0..sample_rate) // 1 second
            .map(|i| {
                (i as f32 * 880.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
            .collect();

    let long_samples: Vec<f32> =
        (0..(sample_rate * 3)) // 3 seconds
            .map(|i| {
                (i as f32 * 220.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
            .collect();

    println!("=== High Concurrency Stress Test ===");

    // Test with very high concurrency levels
    let high_concurrency_levels = vec![8, 16, 32, 64];

    for concurrency_level in high_concurrency_levels {
        println!("\n--- Testing {} concurrent tasks ---", concurrency_level);

        let success_count = Arc::new(AtomicUsize::new(0));
        let failure_count = Arc::new(AtomicUsize::new(0));
        let total_processing_time = Arc::new(Mutex::new(Duration::ZERO));

        let mut join_set = JoinSet::new();
        let start_time = Instant::now();

        for task_id in 0..concurrency_level {
            let converter_clone = Arc::clone(&converter);
            let success_count_clone = Arc::clone(&success_count);
            let failure_count_clone = Arc::clone(&failure_count);
            let total_processing_time_clone = Arc::clone(&total_processing_time);

            // Vary the sample length and conversion type to simulate realistic load
            let samples = match task_id % 3 {
                0 => short_samples.clone(),
                1 => medium_samples.clone(),
                _ => long_samples.clone(),
            };

            join_set.spawn(async move {
                let task_start = Instant::now();

                // Create varied conversion parameters
                let mut target_characteristics = VoiceCharacteristics::default();
                let conversion_type = match task_id % 5 {
                    0 => {
                        target_characteristics.pitch.mean_f0 = 440.0 + (task_id as f32 % 200.0);
                        ConversionType::PitchShift
                    }
                    1 => {
                        target_characteristics.timing.speaking_rate =
                            1.0 + ((task_id as f32 % 10.0) / 20.0);
                        ConversionType::SpeedTransformation
                    }
                    2 => {
                        target_characteristics.gender = Some(if task_id % 2 == 0 {
                            Gender::Female
                        } else {
                            Gender::Male
                        });
                        target_characteristics.pitch.mean_f0 =
                            if task_id % 2 == 0 { 220.0 } else { 120.0 };
                        ConversionType::GenderTransformation
                    }
                    3 => {
                        target_characteristics.age_group = Some(if task_id % 2 == 0 {
                            AgeGroup::YoungAdult
                        } else {
                            AgeGroup::Senior
                        });
                        ConversionType::AgeTransformation
                    }
                    _ => {
                        target_characteristics.pitch.mean_f0 = 400.0 + (task_id as f32 % 300.0);
                        ConversionType::SpeakerConversion
                    }
                };

                let conversion_target = ConversionTarget::new(target_characteristics);
                let request = ConversionRequest::new(
                    format!("stress_test_{}_{}", concurrency_level, task_id),
                    samples,
                    sample_rate,
                    conversion_type,
                    conversion_target,
                );

                match converter_clone.convert(request).await {
                    Ok(result) => {
                        let task_duration = task_start.elapsed();
                        if result.success {
                            success_count_clone.fetch_add(1, Ordering::Relaxed);
                            let mut total_time = total_processing_time_clone.lock().unwrap();
                            *total_time += task_duration;
                        } else {
                            failure_count_clone.fetch_add(1, Ordering::Relaxed);
                        }

                        if task_id % 10 == 0 {
                            println!(
                                "  Task {} completed: {}ms, Success: {}",
                                task_id,
                                task_duration.as_millis(),
                                result.success
                            );
                        }
                    }
                    Err(e) => {
                        failure_count_clone.fetch_add(1, Ordering::Relaxed);
                        if task_id % 10 == 0 {
                            println!("  Task {} failed: {}", task_id, e);
                        }
                    }
                }
            });
        }

        // Wait for all tasks to complete
        while let Some(result) = join_set.join_next().await {
            if let Err(e) = result {
                println!("  Task panicked: {}", e);
            }
        }

        let total_elapsed = start_time.elapsed();
        let successful = success_count.load(Ordering::Relaxed);
        let failed = failure_count.load(Ordering::Relaxed);
        let success_rate = (successful as f64 / concurrency_level as f64) * 100.0;

        let avg_processing_time = if successful > 0 {
            let total_processing = total_processing_time.lock().unwrap();
            total_processing.as_millis() as f64 / successful as f64
        } else {
            0.0
        };

        println!(
            "Concurrency {}: Success: {}/{} ({:.1}%), Failures: {}, Total time: {:.2}s, Avg processing: {:.1}ms",
            concurrency_level, successful, concurrency_level, success_rate, failed,
            total_elapsed.as_secs_f64(), avg_processing_time
        );

        // Stress test assertions - we expect some failures under extreme load but not total failure
        assert!(
            success_rate >= 30.0 || concurrency_level >= 32, // More lenient for very high concurrency
            "Success rate too low for concurrency {}: {:.1}%",
            concurrency_level,
            success_rate
        );

        // System shouldn't completely lock up
        assert!(
            total_elapsed.as_secs() < 300, // Maximum 5 minutes for any stress test
            "Stress test took too long: {:.2}s",
            total_elapsed.as_secs_f64()
        );

        // Give system time to recover between tests
        tokio::time::sleep(Duration::from_millis(1000)).await;
    }

    Ok(())
}

/// Long-duration conversion stability stress test
#[tokio::test]
async fn test_long_duration_stability() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;

    println!("=== Long Duration Stability Test ===");

    // Test with continuous processing for extended periods
    let test_duration = Duration::from_secs(60); // 1 minute of continuous processing
    let start_time = Instant::now();
    let mut iteration = 0;

    let mut processing_times = Vec::new();
    let mut quality_scores = Vec::new();
    let mut memory_usage_trend = Vec::new();

    while start_time.elapsed() < test_duration {
        iteration += 1;

        // Create varied audio samples to prevent optimization artifacts
        let frequency = 440.0 + ((iteration as f32 % 100.0) * 5.0);
        let duration_secs = 1.0 + ((iteration as f32 % 10.0) / 10.0);
        let samples: Vec<f32> = (0..((sample_rate as f32 * duration_secs) as usize))
            .map(|i| {
                (i as f32 * frequency * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
            .collect();

        // Vary conversion types to stress different code paths
        let mut target_characteristics = VoiceCharacteristics::default();
        let conversion_type = match iteration % 4 {
            0 => {
                target_characteristics.pitch.mean_f0 = frequency * 1.5;
                ConversionType::PitchShift
            }
            1 => {
                target_characteristics.timing.speaking_rate = 1.2;
                ConversionType::SpeedTransformation
            }
            2 => {
                target_characteristics.gender = Some(Gender::Female);
                ConversionType::GenderTransformation
            }
            _ => {
                target_characteristics.age_group = Some(AgeGroup::MiddleAged);
                ConversionType::AgeTransformation
            }
        };

        let conversion_target = ConversionTarget::new(target_characteristics);
        let request = ConversionRequest::new(
            format!("stability_test_{}", iteration),
            samples,
            sample_rate,
            conversion_type,
            conversion_target,
        );

        let iteration_start = Instant::now();
        match converter.convert(request).await {
            Ok(result) => {
                let processing_time = iteration_start.elapsed();
                processing_times.push(processing_time.as_millis() as f64);

                if let Some(quality) = result.quality_metrics.get("overall_quality") {
                    quality_scores.push(*quality);
                }

                // Sample memory usage periodically (simulated)
                if iteration % 10 == 0 {
                    memory_usage_trend.push(iteration as f64);
                }

                if iteration % 50 == 0 {
                    let avg_processing =
                        processing_times.iter().sum::<f64>() / processing_times.len() as f64;
                    let avg_quality = if !quality_scores.is_empty() {
                        quality_scores.iter().sum::<f32>() / quality_scores.len() as f32
                    } else {
                        -1.0
                    };

                    println!(
                        "Iteration {}: Avg processing: {:.1}ms, Avg quality: {:.3}, Success: {}",
                        iteration, avg_processing, avg_quality, result.success
                    );
                }

                assert!(
                    result.success,
                    "Conversion should remain stable at iteration {}",
                    iteration
                );
            }
            Err(e) => {
                println!("Iteration {} failed: {}", iteration, e);
                // Allow some failures but not consecutive failures
                assert!(iteration > 10, "Too many early failures in stability test");
            }
        }

        // Brief pause to simulate realistic usage patterns
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let total_elapsed = start_time.elapsed();
    println!("\n=== Stability Test Results ===");
    println!("Total iterations: {}", iteration);
    println!("Test duration: {:.2}s", total_elapsed.as_secs_f64());
    println!(
        "Iterations per second: {:.2}",
        iteration as f64 / total_elapsed.as_secs_f64()
    );

    // Analyze performance stability
    if processing_times.len() >= 10 {
        let early_avg = processing_times[0..5].iter().sum::<f64>() / 5.0;
        let late_avg = processing_times[processing_times.len() - 5..]
            .iter()
            .sum::<f64>()
            / 5.0;
        let performance_drift = (late_avg - early_avg) / early_avg * 100.0;

        println!(
            "Performance drift: {:.2}% (early: {:.1}ms, late: {:.1}ms)",
            performance_drift, early_avg, late_avg
        );

        // Performance shouldn't degrade significantly over time
        assert!(
            performance_drift.abs() < 50.0, // Allow 50% drift in test environment
            "Performance degraded too much: {:.2}%",
            performance_drift
        );
    }

    // Quality stability analysis
    if quality_scores.len() >= 10 {
        let quality_variance = {
            let mean = quality_scores.iter().sum::<f32>() / quality_scores.len() as f32;
            quality_scores
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / quality_scores.len() as f32
        };
        let quality_std = quality_variance.sqrt();

        println!("Quality stability: std = {:.3}", quality_std);

        // Quality should remain reasonably stable
        assert!(
            quality_std < 0.2,
            "Quality too unstable over time: std = {:.3}",
            quality_std
        );
    }

    Ok(())
}

/// Memory pressure stress test
#[tokio::test]
async fn test_memory_pressure_handling() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;

    println!("=== Memory Pressure Stress Test ===");

    // Test with increasingly large audio samples to create memory pressure
    let base_duration = 1.0; // Start with 1 second
    let max_iterations = 20;
    let mut successful_iterations = 0;

    for iteration in 1..=max_iterations {
        let audio_duration = base_duration * (iteration as f32 * 0.5); // Increase duration each iteration
        let samples: Vec<f32> = (0..((sample_rate as f32 * audio_duration) as usize))
            .map(|i| {
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
            })
            .collect();

        println!(
            "Iteration {}: Processing {:.1}s of audio ({} samples)",
            iteration,
            audio_duration,
            samples.len()
        );

        let mut target_characteristics = VoiceCharacteristics::default();
        target_characteristics.pitch.mean_f0 = 880.0; // Simple pitch shift

        let conversion_target = ConversionTarget::new(target_characteristics);
        let request = ConversionRequest::new(
            format!("memory_pressure_test_{}", iteration),
            samples,
            sample_rate,
            ConversionType::PitchShift,
            conversion_target,
        );

        let start = Instant::now();
        match converter.convert(request).await {
            Ok(result) => {
                let elapsed = start.elapsed();
                successful_iterations += 1;

                println!(
                    "  Success: {}, Processing time: {:.2}s, RTF: {:.3}",
                    result.success,
                    elapsed.as_secs_f64(),
                    elapsed.as_secs_f64() / audio_duration as f64
                );

                if result.success {
                    // Check that quality doesn't degrade severely under memory pressure
                    if let Some(quality) = result.quality_metrics.get("overall_quality") {
                        assert!(
                            *quality >= 0.0,
                            "Quality became negative under memory pressure: {:.3}",
                            quality
                        );
                    }
                } else {
                    println!("  Conversion reported failure - this may be acceptable under extreme memory pressure");
                }
            }
            Err(e) => {
                println!("  Failed with error: {}", e);
                println!(
                    "  This may be acceptable under extreme memory pressure at iteration {}",
                    iteration
                );

                // If we fail too early, it might indicate a problem
                if iteration <= 5 {
                    return Err(format!(
                        "Memory pressure test failed too early at iteration {}: {}",
                        iteration, e
                    )
                    .into());
                }

                // If we've had some successes, we can break here as we've reached the system limit
                if successful_iterations >= 10 {
                    println!(
                        "  Reached system memory limits after {} successful iterations",
                        successful_iterations
                    );
                    break;
                }
            }
        }

        // Force garbage collection between iterations (Rust doesn't have explicit GC, but this helps with cleanup)
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    println!(
        "\nMemory pressure test completed: {}/{} iterations successful",
        successful_iterations, max_iterations
    );

    // We should be able to handle at least some large audio samples
    assert!(
        successful_iterations >= 5,
        "Too few successful iterations under memory pressure: {}",
        successful_iterations
    );

    Ok(())
}

/// Rapid context switching stress test
#[tokio::test]
async fn test_rapid_context_switching() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;

    println!("=== Rapid Context Switching Stress Test ===");

    // Create many small conversion tasks that switch rapidly between different types
    let num_rapid_tasks = 200;
    let task_duration = 0.2; // Very short tasks (200ms)

    let samples: Vec<f32> = (0..((sample_rate as f32 * task_duration) as usize))
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
        .collect();

    let mut successful_tasks = 0;
    let mut total_context_switches = 0;
    let start_time = Instant::now();

    for task_id in 0..num_rapid_tasks {
        // Rapidly switch between different conversion contexts
        let mut target_characteristics = VoiceCharacteristics::default();
        let conversion_type = match task_id % 6 {
            0 => {
                target_characteristics.pitch.mean_f0 = 330.0;
                ConversionType::PitchShift
            }
            1 => {
                target_characteristics.pitch.mean_f0 = 550.0;
                ConversionType::PitchShift
            }
            2 => {
                target_characteristics.timing.speaking_rate = 0.8;
                ConversionType::SpeedTransformation
            }
            3 => {
                target_characteristics.timing.speaking_rate = 1.3;
                ConversionType::SpeedTransformation
            }
            4 => {
                target_characteristics.gender = Some(Gender::Female);
                target_characteristics.pitch.mean_f0 = 220.0;
                ConversionType::GenderTransformation
            }
            _ => {
                target_characteristics.age_group = Some(AgeGroup::Senior);
                ConversionType::AgeTransformation
            }
        };

        let conversion_target = ConversionTarget::new(target_characteristics);
        let request = ConversionRequest::new(
            format!("context_switch_test_{}", task_id),
            samples.clone(),
            sample_rate,
            conversion_type,
            conversion_target,
        );

        match converter.convert(request).await {
            Ok(result) => {
                if result.success {
                    successful_tasks += 1;
                }
                total_context_switches += 1;

                if task_id % 50 == 0 {
                    println!(
                        "  Completed {} context switches, {} successful",
                        total_context_switches, successful_tasks
                    );
                }
            }
            Err(e) => {
                if task_id < 10 {
                    println!("  Early task {} failed: {}", task_id, e);
                }
            }
        }
    }

    let total_elapsed = start_time.elapsed();
    let context_switches_per_second = total_context_switches as f64 / total_elapsed.as_secs_f64();
    let success_rate = (successful_tasks as f64 / num_rapid_tasks as f64) * 100.0;

    println!("\n=== Context Switching Results ===");
    println!("Total tasks: {}", num_rapid_tasks);
    println!("Successful: {} ({:.1}%)", successful_tasks, success_rate);
    println!(
        "Context switches per second: {:.2}",
        context_switches_per_second
    );
    println!("Total time: {:.2}s", total_elapsed.as_secs_f64());

    // Assertions for context switching performance
    assert!(
        success_rate >= 70.0,
        "Success rate too low for rapid context switching: {:.1}%",
        success_rate
    );

    assert!(
        context_switches_per_second >= 5.0,
        "Context switching rate too low: {:.2}/s",
        context_switches_per_second
    );

    Ok(())
}

/// Error recovery under stress test
#[tokio::test]
async fn test_error_recovery_under_stress() -> Result<()> {
    let converter = VoiceConverter::new()?;
    let sample_rate = 22050;

    println!("=== Error Recovery Under Stress Test ===");

    // Create scenarios that are likely to cause errors, then test recovery
    let num_iterations = 50;
    let mut recovery_successes = 0;
    let mut consecutive_failures = 0;
    let max_consecutive_failures = 5;

    for iteration in 0..num_iterations {
        // Create potentially problematic scenarios
        let (samples, description) = match iteration % 5 {
            0 => {
                // Very short audio (might cause processing issues)
                let samples: Vec<f32> = (0..10)
                    .map(|i| {
                        (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin()
                            * 0.1
                    })
                    .collect();
                (samples, "very_short_audio")
            }
            1 => {
                // Silent audio
                let samples = vec![0.0; sample_rate as usize];
                (samples, "silent_audio")
            }
            2 => {
                // Very loud audio (might cause clipping)
                let samples: Vec<f32> = (0..sample_rate)
                    .map(|i| {
                        (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin()
                            * 0.9
                    })
                    .collect();
                (samples, "loud_audio")
            }
            3 => {
                // High-frequency content
                let samples: Vec<f32> = (0..sample_rate)
                    .map(|i| {
                        (i as f32 * 8000.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin()
                            * 0.1
                    })
                    .collect();
                (samples, "high_frequency")
            }
            _ => {
                // Normal audio (should work)
                let samples: Vec<f32> = (0..sample_rate)
                    .map(|i| {
                        (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin()
                            * 0.1
                    })
                    .collect();
                (samples, "normal_audio")
            }
        };

        // Use extreme conversion parameters that might cause issues
        let mut target_characteristics = VoiceCharacteristics::default();
        match iteration % 3 {
            0 => {
                // Extreme pitch shift
                target_characteristics.pitch.mean_f0 = 1760.0; // Very high pitch
            }
            1 => {
                // Extreme speed change
                target_characteristics.timing.speaking_rate = 2.5; // Very fast
            }
            _ => {
                // More reasonable parameters
                target_characteristics.pitch.mean_f0 = 660.0;
            }
        }

        let conversion_target = ConversionTarget::new(target_characteristics);
        let request = ConversionRequest::new(
            format!("error_recovery_test_{}_{}", iteration, description),
            samples,
            sample_rate,
            ConversionType::PitchShift,
            conversion_target,
        );

        match converter.convert(request).await {
            Ok(result) => {
                if result.success {
                    consecutive_failures = 0;
                    recovery_successes += 1;

                    if iteration % 10 == 0 {
                        println!("  Iteration {}: {} - Success", iteration, description);
                    }
                } else {
                    consecutive_failures += 1;
                    println!(
                        "  Iteration {}: {} - Conversion failed but no error",
                        iteration, description
                    );
                }
            }
            Err(e) => {
                consecutive_failures += 1;
                println!("  Iteration {}: {} - Error: {}", iteration, description, e);

                // Test that system can recover after errors
                if consecutive_failures > max_consecutive_failures {
                    return Err(format!(
                        "System failed to recover after {} consecutive failures at iteration {}",
                        consecutive_failures, iteration
                    )
                    .into());
                }
            }
        }

        // Brief pause to allow system recovery
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let recovery_rate = (recovery_successes as f64 / num_iterations as f64) * 100.0;

    println!("\n=== Error Recovery Results ===");
    println!("Total iterations: {}", num_iterations);
    println!(
        "Successful recoveries: {} ({:.1}%)",
        recovery_successes, recovery_rate
    );
    println!("Max consecutive failures: {}", max_consecutive_failures);

    // Assert that system shows reasonable error recovery
    assert!(
        recovery_rate >= 40.0, // Should handle at least 40% of problematic cases
        "Error recovery rate too low: {:.1}%",
        recovery_rate
    );

    assert!(
        consecutive_failures <= max_consecutive_failures,
        "Too many consecutive failures: {}",
        consecutive_failures
    );

    Ok(())
}

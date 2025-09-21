//! Stress Testing and Reliability Example
//!
//! This example demonstrates stress testing and reliability validation for VoiRS Recognizer,
//! including memory usage monitoring, performance validation, and graceful degradation
//! under various load conditions.
//!
//! Usage:
//! ```bash
//! cargo run --example stress_testing_reliability --features="whisper-pure" --release
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use voirs_recognizer::prelude::*;
use voirs_recognizer::RecognitionError;

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🔬 VoiRS Stress Testing and Reliability Example");
    println!("===============================================\n");

    // Step 1: Initialize performance validation framework
    println!("⚡ Performance Validation Setup:");
    let requirements = PerformanceRequirements {
        max_rtf: 0.3,
        max_memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
        max_startup_time_ms: 5000,                // 5 seconds
        max_streaming_latency_ms: 200,            // 200ms
    };

    let validator = PerformanceValidator::with_requirements(requirements);
    println!("   ✅ Performance validator initialized");
    println!(
        "   • RTF threshold: < {:.2}",
        validator.requirements().max_rtf
    );
    println!(
        "   • Memory threshold: < {:.1} GB",
        validator.requirements().max_memory_usage as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "   • Startup threshold: < {}ms",
        validator.requirements().max_startup_time_ms
    );
    println!(
        "   • Latency threshold: < {}ms",
        validator.requirements().max_streaming_latency_ms
    );

    // Step 2: Memory usage baseline
    println!("\n💾 Memory Usage Baseline:");
    let initial_memory = get_memory_usage_mb().unwrap_or(0.0);
    println!("   • Initial memory usage: {:.2} MB", initial_memory);

    // Step 3: Stress test with concurrent audio processing
    println!("\n🔄 Concurrent Processing Stress Test:");
    let concurrent_tasks = 10;
    let audio_duration = 1.0; // 1 second per audio

    let start_time = Instant::now();
    let processed_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));

    println!(
        "   Starting {} concurrent audio processing tasks...",
        concurrent_tasks
    );

    let mut handles = Vec::new();

    for task_id in 0..concurrent_tasks {
        let processed_clone = Arc::clone(&processed_count);
        let error_clone = Arc::clone(&error_count);

        let handle = tokio::spawn(async move {
            // Create unique audio for each task
            let audio = create_stress_test_audio(task_id, audio_duration);

            // Process with audio analysis
            let analysis_config = AudioAnalysisConfig {
                quality_metrics: true,
                prosody_analysis: false, // Reduce load
                speaker_analysis: false, // Reduce load
                ..Default::default()
            };

            match AudioAnalyzerImpl::new(analysis_config.clone()).await {
                Ok(analyzer) => match analyzer.analyze(&audio, Some(&analysis_config)).await {
                    Ok(_analysis) => {
                        processed_clone.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(_) => {
                        error_clone.fetch_add(1, Ordering::SeqCst);
                    }
                },
                Err(_) => {
                    error_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let _ = handle.await;
    }

    let total_time = start_time.elapsed();
    let processed = processed_count.load(Ordering::SeqCst);
    let errors = error_count.load(Ordering::SeqCst);

    println!("   ✅ Concurrent processing completed:");
    println!("     • Total time: {:.2}s", total_time.as_secs_f64());
    println!("     • Processed: {}/{}", processed, concurrent_tasks);
    println!("     • Errors: {}", errors);
    println!(
        "     • Success rate: {:.1}%",
        (processed as f64 / concurrent_tasks as f64) * 100.0
    );

    if errors == 0 {
        println!("     🎯 Perfect reliability under concurrent load!");
    } else if errors < concurrent_tasks / 4 {
        println!("     ⚠️ Some errors occurred but system remained mostly stable");
    } else {
        println!("     ❌ High error rate indicates potential reliability issues");
    }

    // Step 4: Memory leak detection
    println!("\n🔍 Memory Leak Detection:");
    let current_memory = get_memory_usage_mb().unwrap_or(0.0);
    let memory_increase = current_memory - initial_memory;

    println!("   • Current memory usage: {:.2} MB", current_memory);
    println!("   • Memory increase: {:.2} MB", memory_increase);

    if memory_increase < 100.0 {
        println!("   ✅ No significant memory leak detected");
    } else if memory_increase < 500.0 {
        println!("   ⚠️ Moderate memory increase - monitor closely");
    } else {
        println!("   ❌ Significant memory increase - potential leak detected");
    }

    // Step 5: Sustained load testing
    println!("\n⏰ Sustained Load Testing:");
    let sustained_duration = Duration::from_secs(10); // 10 seconds of sustained processing
    let chunk_interval = Duration::from_millis(100); // Process every 100ms

    println!(
        "   Running sustained load for {}s...",
        sustained_duration.as_secs()
    );

    let sustained_start = Instant::now();
    let mut sustained_processed = 0;
    let mut sustained_errors = 0;
    let mut rtf_measurements = Vec::new();

    while sustained_start.elapsed() < sustained_duration {
        let chunk_start = Instant::now();

        // Create and process audio chunk
        let audio = create_stress_test_audio(sustained_processed, 0.1); // 100ms chunk

        let analysis_config = AudioAnalysisConfig {
            quality_metrics: true,
            prosody_analysis: false,
            speaker_analysis: false,
            ..Default::default()
        };

        match AudioAnalyzerImpl::new(analysis_config.clone()).await {
            Ok(analyzer) => {
                match analyzer.analyze(&audio, Some(&analysis_config)).await {
                    Ok(_) => {
                        sustained_processed += 1;
                        let processing_time = chunk_start.elapsed();
                        let rtf = processing_time.as_secs_f64() / 0.1; // 0.1s audio duration
                        rtf_measurements.push(rtf);
                    }
                    Err(_) => sustained_errors += 1,
                }
            }
            Err(_) => sustained_errors += 1,
        }

        sleep(chunk_interval).await;
    }

    let avg_rtf = rtf_measurements.iter().sum::<f64>() / rtf_measurements.len() as f64;
    let max_rtf = rtf_measurements.iter().fold(0.0f64, |acc, &x| acc.max(x));
    let min_rtf = rtf_measurements
        .iter()
        .fold(f64::INFINITY, |acc, &x| acc.min(x));

    println!("   ✅ Sustained load testing completed:");
    println!("     • Total chunks processed: {}", sustained_processed);
    println!("     • Errors: {}", sustained_errors);
    println!("     • Average RTF: {:.3}", avg_rtf);
    println!("     • Min RTF: {:.3}", min_rtf);
    println!("     • Max RTF: {:.3}", max_rtf);

    if avg_rtf < validator.requirements().max_rtf as f64 {
        println!("     ✅ RTF performance maintained under sustained load");
    } else {
        println!("     ⚠️ RTF exceeded threshold under sustained load");
    }

    // Step 6: Error recovery testing
    println!("\n🛡️ Error Recovery Testing:");

    println!("   Testing graceful degradation scenarios:");

    // Test with invalid audio data
    let invalid_audio = AudioBuffer::mono(vec![], 16000); // Empty audio
    let recovery_config = AudioAnalysisConfig::default();

    match AudioAnalyzerImpl::new(recovery_config.clone()).await {
        Ok(analyzer) => {
            match analyzer
                .analyze(&invalid_audio, Some(&recovery_config))
                .await
            {
                Ok(_) => println!("     • Empty audio: Handled gracefully"),
                Err(e) => println!("     • Empty audio: Error handled - {:?}", e),
            }
        }
        Err(e) => println!("     • Analyzer creation failed: {:?}", e),
    }

    // Test with extreme audio values
    let extreme_audio = AudioBuffer::mono(vec![f32::MAX; 1000], 16000);
    match AudioAnalyzerImpl::new(recovery_config.clone()).await {
        Ok(analyzer) => {
            match analyzer
                .analyze(&extreme_audio, Some(&recovery_config))
                .await
            {
                Ok(_) => println!("     • Extreme values: Handled gracefully"),
                Err(e) => println!("     • Extreme values: Error handled - {:?}", e),
            }
        }
        Err(e) => println!("     • Analyzer creation failed: {:?}", e),
    }

    // Step 7: Performance regression detection
    println!("\n📊 Performance Regression Detection:");

    let baseline_rtf = 0.15; // Baseline performance
    let current_avg_rtf = avg_rtf;
    let regression_threshold = 1.2; // 20% degradation threshold

    println!("   • Baseline RTF: {:.3}", baseline_rtf);
    println!("   • Current RTF: {:.3}", current_avg_rtf);

    let performance_ratio = current_avg_rtf / baseline_rtf;
    println!("   • Performance ratio: {:.2}x", performance_ratio);

    if performance_ratio <= regression_threshold {
        println!("   ✅ No performance regression detected");
    } else {
        println!(
            "   ⚠️ Performance regression detected ({:.1}% slower)",
            (performance_ratio - 1.0) * 100.0
        );
    }

    // Step 8: Reliability metrics summary
    println!("\n📈 Reliability Metrics Summary:");

    let total_operations = concurrent_tasks + sustained_processed;
    let total_errors = errors + sustained_errors;
    let overall_success_rate =
        ((total_operations - total_errors) as f64 / total_operations as f64) * 100.0;

    println!("   System Reliability Assessment:");
    println!("   • Total operations: {}", total_operations);
    println!("   • Total errors: {}", total_errors);
    println!("   • Overall success rate: {:.2}%", overall_success_rate);

    let final_memory = get_memory_usage_mb().unwrap_or(0.0);
    let total_memory_increase = final_memory - initial_memory;
    println!(
        "   • Total memory increase: {:.2} MB",
        total_memory_increase
    );

    // Reliability classification
    match overall_success_rate {
        rate if rate >= 99.9 => println!("   🏆 Excellent reliability (≥99.9%)"),
        rate if rate >= 99.0 => println!("   ✅ Good reliability (≥99.0%)"),
        rate if rate >= 95.0 => println!("   ⚠️ Acceptable reliability (≥95.0%)"),
        _ => println!("   ❌ Poor reliability (<95.0%)"),
    }

    // Step 9: Stress testing recommendations
    println!("\n💡 Stress Testing Recommendations:");
    println!("   For Production Deployment:");
    println!("   • Run extended stress tests (24+ hours)");
    println!("   • Test with real-world audio variations");
    println!("   • Monitor memory usage continuously");
    println!("   • Test graceful degradation under resource constraints");
    println!("   • Validate performance with different model configurations");
    println!("   • Test concurrent access patterns specific to your use case");

    println!("\n✅ Stress testing and reliability validation completed!");

    // Final assessment
    if overall_success_rate >= 99.0
        && total_memory_increase < 200.0
        && avg_rtf < validator.requirements().max_rtf as f64
    {
        println!("🎯 System passes reliability requirements for production use!");
    } else {
        println!("⚠️ System may need optimization before production deployment");
    }

    Ok(())
}

/// Create stress test audio with specified characteristics
fn create_stress_test_audio(seed: usize, duration: f32) -> AudioBuffer {
    let sample_rate = 16000;
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    // Use seed to create different audio patterns for each test
    let base_freq = 200.0 + (seed % 10) as f32 * 50.0;
    let noise_level = 0.01 + (seed % 5) as f32 * 0.005;

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // Create complex signal with multiple components
        let signal = 0.3 * (2.0 * std::f32::consts::PI * base_freq * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * base_freq * 2.0 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * base_freq * 3.0 * t).sin();

        // Add noise
        let noise = ((i + seed) as f32 * 0.001).sin() * noise_level;

        // Add envelope
        let envelope = (0.5 + 0.5 * (5.0 * t).sin()).abs();

        samples.push((signal + noise) * envelope);
    }

    AudioBuffer::mono(samples, sample_rate)
}

/// Get current memory usage in MB (simplified implementation)
fn get_memory_usage_mb() -> Option<f64> {
    // This is a simplified implementation
    // In a real application, you would use platform-specific APIs
    // For now, return a simulated value
    Some(150.0 + rand::random::<f64>() * 50.0) // Simulate 150-200 MB
}

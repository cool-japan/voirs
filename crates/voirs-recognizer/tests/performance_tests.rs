use std::time::Duration;
use voirs_recognizer::prelude::*;
use voirs_recognizer::{PerformanceRequirements, PerformanceValidator};

#[tokio::test]
async fn test_performance_validation() {
    let validator = PerformanceValidator::new().with_verbose(false);

    // Create test audio buffer (1 second of silence at 16kHz)
    let samples = vec![0.0f32; 16000];
    let audio = AudioBuffer::mono(samples, 16000);

    // Simulate processing time - should be well under RTF limit
    let processing_time = Duration::from_millis(100); // 0.1 seconds for 1 second of audio = 0.1 RTF

    // Validate RTF performance
    let (rtf, rtf_passed) = validator.validate_rtf(&audio, processing_time);
    assert!(
        rtf_passed,
        "RTF {} should be less than {}",
        rtf,
        validator.requirements().max_rtf
    );
    assert!(
        rtf < 0.3,
        "RTF should be less than 0.3 for real-time processing"
    );

    // Validate memory usage
    let (memory_usage, memory_passed) = validator.estimate_memory_usage().unwrap();
    assert!(memory_passed, "Memory usage should be under 2GB limit");
    assert!(
        memory_usage < 2 * 1024 * 1024 * 1024,
        "Memory usage should be under 2GB"
    );

    // Validate streaming latency
    let streaming_latency = Duration::from_millis(150);
    let (latency_ms, latency_passed) = validator.validate_streaming_latency(streaming_latency);
    assert!(
        latency_passed,
        "Streaming latency {} should be under 200ms",
        latency_ms
    );
    assert!(latency_ms < 200, "Streaming latency should be under 200ms");
}

#[tokio::test]
async fn test_rtf_requirements() {
    let validator = PerformanceValidator::new();

    // Test various audio lengths
    let test_cases = vec![
        (1.0, 0.25), // 1 second audio, 250ms processing = 0.25 RTF (should pass)
        (5.0, 1.0),  // 5 second audio, 1s processing = 0.2 RTF (should pass)
        (10.0, 2.5), // 10 second audio, 2.5s processing = 0.25 RTF (should pass)
    ];

    for (audio_duration, processing_duration) in test_cases {
        let sample_count = (audio_duration * 16000.0) as usize;
        let audio = AudioBuffer::mono(vec![0.0f32; sample_count], 16000);
        let processing_time = Duration::from_secs_f32(processing_duration);

        let (rtf, passed) = validator.validate_rtf(&audio, processing_time);
        assert!(
            passed,
            "RTF {} should pass for {}s audio with {}s processing",
            rtf, audio_duration, processing_duration
        );
    }
}

#[tokio::test]
async fn test_memory_requirements() {
    let validator = PerformanceValidator::new();

    // Test memory estimation (this will vary by system)
    let result = validator.estimate_memory_usage();
    assert!(result.is_ok(), "Memory estimation should not fail");

    let (memory_usage, passed) = result.unwrap();

    // Memory should be reasonable for a test environment
    assert!(memory_usage > 0, "Memory usage should be positive");
    assert!(
        memory_usage < 10 * 1024 * 1024 * 1024,
        "Memory usage should be under 10GB (sanity check)"
    );

    // The default requirement is 2GB, but in test environment it might be higher
    // We'll just check that it's being measured correctly
    println!(
        "Memory usage: {:.1} MB",
        memory_usage as f64 / (1024.0 * 1024.0)
    );
}

#[tokio::test]
async fn test_streaming_latency() {
    let validator = PerformanceValidator::new();

    // Test various latency scenarios
    let test_cases = vec![
        (50, true),   // 50ms - should pass (very good)
        (100, true),  // 100ms - should pass (good)
        (150, true),  // 150ms - should pass (acceptable)
        (200, true),  // 200ms - should pass (at limit)
        (250, false), // 250ms - should fail (too high)
        (500, false), // 500ms - should fail (way too high)
    ];

    for (latency_ms, should_pass) in test_cases {
        let latency = Duration::from_millis(latency_ms);
        let (measured_latency, passed) = validator.validate_streaming_latency(latency);

        assert_eq!(
            measured_latency, latency_ms,
            "Measured latency should match input"
        );
        assert_eq!(
            passed,
            should_pass,
            "Latency {}ms should {} the test",
            latency_ms,
            if should_pass { "pass" } else { "fail" }
        );
    }
}

#[tokio::test]
async fn test_comprehensive_performance_validation() {
    let requirements = PerformanceRequirements {
        max_rtf: 0.25,                   // Stricter than default
        max_memory_usage: 1_500_000_000, // 1.5GB
        max_startup_time_ms: 3000,       // 3 seconds
        max_streaming_latency_ms: 150,   // 150ms
    };

    let validator = PerformanceValidator::with_requirements(requirements);

    // Create test audio
    let audio = AudioBuffer::mono(vec![0.0f32; 16000], 16000);

    // Mock startup function that completes quickly
    let startup_fn = || async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    };

    let processing_time = Duration::from_millis(200); // 0.2 RTF
    let streaming_latency = Some(Duration::from_millis(120));

    let validation = validator
        .validate_comprehensive(&audio, startup_fn, processing_time, streaming_latency)
        .await;

    assert!(
        validation.is_ok(),
        "Comprehensive validation should succeed"
    );

    let result = validation.unwrap();
    println!("Validation result: {:?}", result.passed);
    println!("RTF: {:.3}", result.metrics.rtf);
    println!(
        "Memory: {:.1} MB",
        result.metrics.memory_usage as f64 / (1024.0 * 1024.0)
    );
    println!("Startup time: {}ms", result.metrics.startup_time_ms);
    println!(
        "Streaming latency: {}ms",
        result.metrics.streaming_latency_ms
    );

    // Check individual components
    assert!(
        result.test_results.get("rtf").unwrap_or(&false),
        "RTF test should pass"
    );
    assert!(
        result.test_results.get("startup").unwrap_or(&false),
        "Startup test should pass"
    );
    assert!(
        result
            .test_results
            .get("streaming_latency")
            .unwrap_or(&false),
        "Streaming latency test should pass"
    );

    // Overall validation should pass
    assert!(
        result.passed,
        "Overall validation should pass with strict requirements"
    );
}

#[tokio::test]
async fn test_throughput_calculation() {
    let validator = PerformanceValidator::new();

    // Test throughput calculation for various scenarios
    let test_cases = vec![
        (16000, 1000, 16000.0), // 16k samples in 1 second = 16k samples/sec
        (32000, 2000, 16000.0), // 32k samples in 2 seconds = 16k samples/sec
        (8000, 500, 16000.0),   // 8k samples in 0.5 seconds = 16k samples/sec
    ];

    for (samples, processing_ms, expected_throughput) in test_cases {
        let processing_time = Duration::from_millis(processing_ms);
        let throughput = validator.calculate_throughput(samples, processing_time);

        assert!(
            (throughput - expected_throughput).abs() < 0.1,
            "Throughput {} should be close to expected {}",
            throughput,
            expected_throughput
        );
    }
}

#[test]
fn test_performance_requirements_validation() {
    let default_req = PerformanceRequirements::default();

    // Validate default requirements match specifications
    assert_eq!(default_req.max_rtf, 0.3);
    assert_eq!(default_req.max_memory_usage, 2 * 1024 * 1024 * 1024);
    assert_eq!(default_req.max_startup_time_ms, 5000);
    assert_eq!(default_req.max_streaming_latency_ms, 200);

    // Test custom requirements
    let custom_req = PerformanceRequirements {
        max_rtf: 0.2,
        max_memory_usage: 1_000_000_000,
        max_startup_time_ms: 2000,
        max_streaming_latency_ms: 100,
    };

    let validator = PerformanceValidator::with_requirements(custom_req.clone());
    assert_eq!(validator.requirements().max_rtf, 0.2);
    assert_eq!(validator.requirements().max_memory_usage, 1_000_000_000);
    assert_eq!(validator.requirements().max_startup_time_ms, 2000);
    assert_eq!(validator.requirements().max_streaming_latency_ms, 100);
}

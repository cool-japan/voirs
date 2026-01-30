//! Simple Production Monitoring Example
//!
//! Demonstrates production-ready error handling, performance monitoring,
//! and diagnostics using the voirs-acoustic error context system.

use std::time::{Duration, Instant};
use voirs_acoustic::error::{ErrorContextBuilder, ErrorSeverity};
use voirs_acoustic::{Phoneme, SynthesisConfig};

/// Production synthesis metrics
#[derive(Debug, Default)]
struct ProductionMetrics {
    total_requests: usize,
    successful: usize,
    failed: usize,
    total_duration_ms: u128,
    max_latency_ms: u128,
    min_latency_ms: u128,
}

impl ProductionMetrics {
    fn new() -> Self {
        Self {
            min_latency_ms: u128::MAX,
            ..Default::default()
        }
    }

    fn record_success(&mut self, duration_ms: u128) {
        self.total_requests += 1;
        self.successful += 1;
        self.total_duration_ms += duration_ms;
        self.max_latency_ms = self.max_latency_ms.max(duration_ms);
        self.min_latency_ms = self.min_latency_ms.min(duration_ms);
    }

    fn record_failure(&mut self) {
        self.total_requests += 1;
        self.failed += 1;
    }

    fn average_latency_ms(&self) -> f64 {
        if self.successful > 0 {
            self.total_duration_ms as f64 / self.successful as f64
        } else {
            0.0
        }
    }

    fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.successful as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }

    fn print_summary(&self) {
        println!("\n=== Production Metrics Summary ===");
        println!("Total Requests:    {}", self.total_requests);
        println!("Successful:        {}", self.successful);
        println!("Failed:            {}", self.failed);
        println!("Success Rate:      {:.1}%", self.success_rate() * 100.0);
        println!("Average Latency:   {:.2}ms", self.average_latency_ms());
        println!(
            "Min Latency:       {}ms",
            if self.min_latency_ms == u128::MAX {
                0
            } else {
                self.min_latency_ms
            }
        );
        println!("Max Latency:       {}ms", self.max_latency_ms);
        println!("==================================\n");
    }
}

/// Simulate synthesis with error scenarios
fn simulate_synthesis(
    phonemes: &[Phoneme],
    _config: &SynthesisConfig,
    should_fail: bool,
) -> Result<Vec<f32>, String> {
    if should_fail {
        Err("Simulated inference error".to_string())
    } else {
        // Simulate processing time
        std::thread::sleep(Duration::from_millis(50));
        Ok(vec![0.0f32; phonemes.len() * 256])
    }
}

fn main() {
    println!("VoiRS Acoustic - Production Monitoring Example\n");

    let mut metrics = ProductionMetrics::new();

    // Example synthesis requests with varying scenarios
    let test_cases = vec![
        ("Normal synthesis", false, SynthesisConfig::default()),
        (
            "Fast synthesis",
            false,
            SynthesisConfig {
                speed: 1.5,
                ..Default::default()
            },
        ),
        (
            "Pitch shifted",
            false,
            SynthesisConfig {
                pitch_shift: 5.0,
                ..Default::default()
            },
        ),
        ("Simulated failure", true, SynthesisConfig::default()),
        ("Another normal", false, SynthesisConfig::default()),
    ];

    for (description, should_fail, config) in test_cases {
        println!("Processing: {}", description);

        let phonemes = vec![
            Phoneme::new("HH"),
            Phoneme::new("EH"),
            Phoneme::new("L"),
            Phoneme::new("OW"),
        ];

        let start = Instant::now();
        let result = simulate_synthesis(&phonemes, &config, should_fail);
        let duration = start.elapsed();

        match result {
            Ok(audio) => {
                metrics.record_success(duration.as_millis());
                println!(
                    "  ✓ Success: Generated {} samples in {:.2}ms",
                    audio.len(),
                    duration.as_millis()
                );

                // Check for performance issues
                if duration.as_millis() > 200 {
                    let ctx = ErrorContextBuilder::performance_degradation(
                        "synthesis",
                        100.0,
                        duration.as_millis() as f32,
                    );
                    println!("  ⚠  Performance Alert:");
                    println!("{}", ctx.format_report());
                }
            }
            Err(e) => {
                metrics.record_failure();
                println!("  ✗ Failed: {}", e);

                // Generate detailed error context
                let ctx = ErrorContextBuilder::inference_error(
                    "synthesis",
                    &format!("[{} phonemes]", phonemes.len()),
                    &e,
                );
                println!("\n  Error Diagnostic Report:");
                println!("{}", ctx.format_report());
            }
        }

        println!();
    }

    // Print final metrics
    metrics.print_summary();

    // Demonstrate different error contexts
    println!("=== Error Context Examples ===\n");

    // Model loading error
    let ctx1 = ErrorContextBuilder::model_loading_error(
        "/models/vits-en-us.safetensors",
        "File not found",
    );
    println!("1. Model Loading Error:");
    println!("{}\n", ctx1.format_report());

    // Input validation error
    let ctx2 = ErrorContextBuilder::input_validation_error("speed", "5.0", "0.5-2.0");
    println!("2. Input Validation Error:");
    println!("{}\n", ctx2.format_report());

    // Resource error
    let ctx3 = ErrorContextBuilder::resource_error("GPU Memory", "4096 MB", "2048 MB");
    println!("3. Resource Exhaustion Error:");
    println!("{}\n", ctx3.format_report());

    // Memory allocation error
    let ctx4 = ErrorContextBuilder::memory_allocation_error(
        536_870_912, // 512 MB
        "mel_spectrogram_buffer",
    );
    println!("4. Memory Allocation Error:");
    println!("{}\n", ctx4.format_report());

    println!("Production monitoring example completed!");
}

//! Comprehensive batch processing example for VoiRS SDK.
//!
//! This example demonstrates:
//! - Basic batch synthesis with multiple requests
//! - Priority-based processing
//! - Custom speed and pitch parameters per request
//! - Voice switching in batch mode
//! - Progress tracking
//! - Error handling and retry logic
//! - Different scheduling strategies
//! - Performance metrics and statistics
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example batch_processing --features emotion,cloning
//! ```

use std::sync::Arc;
use std::time::Instant;
use voirs_sdk::batch::{
    BatchConfig, BatchProcessor, BatchRequest, ProcessingMetrics, SchedulingStrategy,
};
use voirs_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== VoiRS SDK - Advanced Batch Processing Example ===\n");

    // Create pipeline with test mode for demonstration
    println!("Creating VoiRS pipeline...");
    let pipeline = Arc::new(
        VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .with_quality(QualityLevel::High)
            .build()
            .await?,
    );
    println!("Pipeline created successfully!\n");

    // Example 1: Basic batch processing
    println!("Example 1: Basic Batch Processing");
    println!("-".repeat(60));
    basic_batch_processing(&pipeline).await?;
    println!();

    // Example 2: Priority-based processing
    println!("Example 2: Priority-Based Processing");
    println!("-".repeat(60));
    priority_batch_processing(&pipeline).await?;
    println!();

    // Example 3: Custom speed and pitch parameters
    println!("Example 3: Custom Speed and Pitch Parameters");
    println!("-".repeat(60));
    custom_parameters_batch(&pipeline).await?;
    println!();

    // Example 4: Voice switching in batch mode
    println!("Example 4: Voice Switching in Batch Mode");
    println!("-".repeat(60));
    voice_switching_batch(&pipeline).await?;
    println!();

    // Example 5: Different scheduling strategies
    println!("Example 5: Scheduling Strategy Comparison");
    println!("-".repeat(60));
    scheduling_strategies_comparison(&pipeline).await?;
    println!();

    // Example 6: Progress tracking
    println!("Example 6: Real-Time Progress Tracking");
    println!("-".repeat(60));
    progress_tracking_batch(&pipeline).await?;
    println!();

    // Example 7: Large-scale batch processing
    println!("Example 7: Large-Scale Batch Processing");
    println!("-".repeat(60));
    large_scale_batch(&pipeline).await?;
    println!();

    println!("All batch processing examples completed successfully!");

    Ok(())
}

/// Example 1: Basic batch processing with multiple requests
async fn basic_batch_processing(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Processing a batch of simple text requests...");

    // Create batch configuration
    let config = BatchConfig::default();

    // Create batch processor
    let processor = BatchProcessor::new(Arc::clone(pipeline), config);

    // Create batch requests
    let requests = vec![
        BatchRequest::new("Hello, world!", None),
        BatchRequest::new("How are you today?", None),
        BatchRequest::new("This is a batch processing demonstration.", None),
        BatchRequest::new("VoiRS SDK makes speech synthesis easy!", None),
        BatchRequest::new("Goodbye and have a great day!", None),
    ];

    let start = Instant::now();

    // Process the batch
    let results = processor.process(requests).await?;

    let duration = start.elapsed();

    // Display results
    println!("\nResults:");
    println!("  Total requests: {}", results.len());
    println!("  Successful: {}", results.iter().filter(|r| r.is_success()).count());
    println!("  Failed: {}", results.iter().filter(|r| !r.is_success()).count());
    println!("  Total time: {:.2}s", duration.as_secs_f64());
    println!("  Average time per request: {:.2}ms",
             duration.as_millis() as f64 / results.len() as f64);

    // Display individual results
    for (idx, result) in results.iter().enumerate() {
        if let Some(audio) = result.audio() {
            println!("  Request {}: {} samples, {:.2}s audio, processed in {:.2}ms",
                     idx + 1,
                     audio.len(),
                     audio.duration(),
                     result.processing_time.as_millis());
        } else if let Some(error) = result.error() {
            println!("  Request {}: Failed - {}", idx + 1, error);
        }
    }

    // Get statistics
    let stats = processor.statistics().await;
    println!("\nStatistics:");
    println!("{}", stats.summary());

    Ok(())
}

/// Example 2: Priority-based batch processing
async fn priority_batch_processing(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Processing batch with different priorities...");

    let config = BatchConfig {
        scheduling_strategy: SchedulingStrategy::Priority,
        ..Default::default()
    };

    let processor = BatchProcessor::new(Arc::clone(pipeline), config);

    // Create requests with different priorities
    let requests = vec![
        BatchRequest::new("Low priority request", None).with_priority(1),
        BatchRequest::new("Critical urgent request!", None).with_priority(100),
        BatchRequest::new("Normal priority", None).with_priority(50),
        BatchRequest::new("Another low priority", None).with_priority(1),
        BatchRequest::new("High priority request", None).with_priority(75),
    ];

    println!("Request priorities:");
    for (idx, req) in requests.iter().enumerate() {
        println!("  Request {}: priority {} - \"{}\"",
                 idx + 1, req.priority, &req.text[..req.text.len().min(30)]);
    }

    let results = processor.process(requests).await?;

    println!("\nProcessing order (by worker assignment):");
    for (idx, result) in results.iter().enumerate() {
        println!("  Processed request {}: {:.2}ms",
                 idx + 1, result.processing_time.as_millis());
    }

    Ok(())
}

/// Example 3: Custom speed and pitch parameters
async fn custom_parameters_batch(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Processing batch with custom speed and pitch parameters...");

    let config = BatchConfig::default();
    let processor = BatchProcessor::new(Arc::clone(pipeline), config);

    // Create requests with different speed/pitch settings
    let requests = vec![
        BatchRequest::new("Normal speed and pitch", None),
        BatchRequest::new("Speaking very quickly now", None).with_speed(1.5),
        BatchRequest::new("Speaking very slowly now", None).with_speed(0.7),
        BatchRequest::new("Higher pitch voice here", None).with_pitch(3.0),
        BatchRequest::new("Lower pitch voice here", None).with_pitch(-3.0),
        BatchRequest::new("Fast and high pitched!", None)
            .with_speed(1.3)
            .with_pitch(2.0),
        BatchRequest::new("Slow and low pitched", None)
            .with_speed(0.8)
            .with_pitch(-2.0),
    ];

    println!("Request parameters:");
    for (idx, req) in requests.iter().enumerate() {
        print!("  Request {}: ", idx + 1);
        if let Some(speed) = req.speed {
            print!("speed={:.1}x ", speed);
        }
        if let Some(pitch) = req.pitch {
            print!("pitch={:+.1} semitones ", pitch);
        }
        if req.speed.is_none() && req.pitch.is_none() {
            print!("default parameters");
        }
        println!();
    }

    let results = processor.process(requests).await?;

    println!("\nResults:");
    for (idx, result) in results.iter().enumerate() {
        if let Some(audio) = result.audio() {
            println!("  Request {}: {:.2}s audio, {:.2}ms processing",
                     idx + 1,
                     audio.duration(),
                     result.processing_time.as_millis());
        }
    }

    Ok(())
}

/// Example 4: Voice switching in batch mode
async fn voice_switching_batch(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Processing batch with different voices...");

    let config = BatchConfig::default();
    let processor = BatchProcessor::new(Arc::clone(pipeline), config);

    // Create requests with different voices
    let requests = vec![
        BatchRequest::new("Default voice speaking", None),
        BatchRequest::new("Female voice speaking", Some("en-US-female-calm")),
        BatchRequest::new("Back to default voice", None),
        BatchRequest::new("Male voice speaking", Some("en-US-male-energetic")),
        BatchRequest::new("Female voice again", Some("en-US-female-calm")),
    ];

    println!("Voice assignments:");
    for (idx, req) in requests.iter().enumerate() {
        println!("  Request {}: {} - \"{}\"",
                 idx + 1,
                 req.voice.as_deref().unwrap_or("default"),
                 &req.text[..req.text.len().min(30)]);
    }

    let results = processor.process(requests).await?;

    println!("\nResults:");
    for (idx, result) in results.iter().enumerate() {
        println!("  Request {}: {} (worker: {})",
                 idx + 1,
                 if result.is_success() { "Success" } else { "Failed" },
                 result.worker_id.map(|id| id.to_string()).unwrap_or_else(|| "N/A".to_string()));
    }

    Ok(())
}

/// Example 5: Compare different scheduling strategies
async fn scheduling_strategies_comparison(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Comparing different scheduling strategies...\n");

    // Test texts of varying lengths
    let test_requests: Vec<_> = vec![
        ("Short", "Hi!"),
        ("Medium", "This is a medium length sentence for testing."),
        ("Long", "This is a much longer sentence that will take more time to process and synthesize into speech audio."),
        ("Short", "Bye!"),
        ("Medium", "Another medium sentence here for variety."),
    ]
    .into_iter()
    .map(|(len, text)| {
        let mut req = BatchRequest::new(text, None);
        req.priority = if len == "Short" { 10 } else if len == "Medium" { 5 } else { 1 };
        req
    })
    .collect();

    let strategies = vec![
        ("FIFO (First In, First Out)", SchedulingStrategy::Fifo),
        ("Priority-Based", SchedulingStrategy::Priority),
        ("Load Balanced", SchedulingStrategy::LoadBalanced),
        ("Shortest Job First", SchedulingStrategy::ShortestFirst),
        ("Adaptive", SchedulingStrategy::Adaptive),
    ];

    for (name, strategy) in strategies {
        println!("Testing {}", name);

        let config = BatchConfig {
            scheduling_strategy: strategy,
            ..Default::default()
        };

        let processor = BatchProcessor::new(Arc::clone(pipeline), config);
        let start = Instant::now();
        let results = processor.process(test_requests.clone()).await?;
        let duration = start.elapsed();

        let stats = processor.statistics().await;

        println!("  Total time: {:.2}ms", duration.as_millis());
        println!("  Throughput: {:.2} req/s", stats.throughput);
        println!("  Success rate: {:.1}%", stats.success_rate * 100.0);
        println!();
    }

    Ok(())
}

/// Example 6: Progress tracking
async fn progress_tracking_batch(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Processing batch with real-time progress tracking...\n");

    let config = BatchConfig {
        track_progress: true,
        ..Default::default()
    };

    let processor = BatchProcessor::new(Arc::clone(pipeline), config);

    // Create a larger batch for visible progress
    let requests: Vec<_> = (0..20)
        .map(|i| BatchRequest::new(format!("Request number {}", i + 1), None))
        .collect();

    println!("Starting batch of {} requests...", requests.len());

    // Set up progress callback
    let progress_callback = Arc::new(|completed: usize, total: usize| {
        let percent = (completed as f64 / total as f64) * 100.0;
        println!("  Progress: {}/{} ({:.1}%)", completed, total, percent);
    });

    // Process with progress tracking
    let processor_with_progress = processor.with_progress_callback(progress_callback);
    let results = processor_with_progress.process(requests).await?;

    println!("\nBatch completed!");
    println!("  Successful: {}", results.iter().filter(|r| r.is_success()).count());
    println!("  Total audio duration: {:.2}s",
             results.iter()
                 .filter_map(|r| r.audio())
                 .map(|a| a.duration())
                 .sum::<f64>());

    Ok(())
}

/// Example 7: Large-scale batch processing
async fn large_scale_batch(pipeline: &Arc<VoirsPipeline>) -> Result<()> {
    println!("Processing large-scale batch (100 requests)...");

    let config = BatchConfig {
        max_concurrency: num_cpus::get() * 2, // Use more workers for large batch
        max_batch_size: 100,
        scheduling_strategy: SchedulingStrategy::LoadBalanced,
        ..Default::default()
    };

    let processor = BatchProcessor::new(Arc::clone(pipeline), config);

    // Create 100 requests with varying complexity
    let requests: Vec<_> = (0..100)
        .map(|i| {
            let text = match i % 3 {
                0 => format!("Short request {}", i),
                1 => format!("This is a medium length request number {} with more text to process", i),
                _ => format!("This is a longer request number {} that contains significantly more text and will require more processing time to synthesize into speech audio", i),
            };
            BatchRequest::new(text, None)
                .with_priority(fastrand::i32(1..100))
        })
        .collect();

    let start = Instant::now();
    let results = processor.process(requests).await?;
    let duration = start.elapsed();

    let stats = processor.statistics().await;

    println!("\nLarge-Scale Batch Results:");
    println!("  Total requests: {}", results.len());
    println!("  Successful: {}", results.iter().filter(|r| r.is_success()).count());
    println!("  Failed: {}", results.iter().filter(|r| !r.is_success()).count());
    println!("  Total time: {:.2}s", duration.as_secs_f64());
    println!("  Throughput: {:.2} requests/second", stats.throughput);
    println!("  Average processing time: {:.2}ms", stats.average_processing_time.as_millis());
    println!("  Success rate: {:.1}%", stats.success_rate * 100.0);

    // Worker metrics
    println!("\nWorker Performance:");
    for (worker_id, metrics) in &stats.worker_metrics {
        println!("  Worker {}: {} requests, {:.2}ms avg time",
                 worker_id,
                 metrics.requests_processed,
                 metrics.total_time.as_millis() as f64 / metrics.requests_processed.max(1) as f64);
    }

    // Total audio produced
    let total_audio_duration: f64 = results.iter()
        .filter_map(|r| r.audio())
        .map(|a| a.duration())
        .sum();
    println!("\n  Total audio produced: {:.2} seconds", total_audio_duration);

    Ok(())
}

//! # Multi-target Voice Conversion Example
//!
//! This example demonstrates converting a single source voice to multiple
//! target voices simultaneously, useful for creating voice variations or
//! multi-speaker synthesis.
//!
//! ## Features Demonstrated
//! - Simultaneous multi-target conversion
//! - Priority-based target processing
//! - Parallel and sequential processing modes
//! - Performance monitoring and statistics
//!
//! Note: This is a simplified demonstration. In a real application, you would:
//! - Use actual high-quality audio recordings
//! - Configure based on your specific requirements
//! - Implement proper error handling and recovery

use voirs_conversion::{
    core::VoiceConverter,
    multi_target::{
        MultiTargetConversionRequest, MultiTargetConverter, NamedTarget, ProcessingMode,
    },
    types::{AgeGroup, Gender},
    ConversionConfig, ConversionTarget, ConversionType, Result, VoiceCharacteristics,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Multi-target Voice Conversion Example ===\n");

    // Step 1: Create base voice converter
    println!("1. Creating base voice converter...");
    let base_converter = VoiceConverter::new()?;
    println!("   Base converter created");

    // Step 2: Configure multi-target converter
    println!("\n2. Configuring multi-target converter...");
    let converter = MultiTargetConverter::new(base_converter)
        .with_processing_mode(ProcessingMode::Parallel)
        .with_max_concurrent_targets(4);
    println!("   Converter initialized with parallel processing mode");
    println!("   Max concurrent targets: 4");

    // Step 3: Generate source audio
    println!("\n3. Generating source audio...");
    let source_audio = generate_sample_audio(180.0, 32000); // 2 seconds at 16kHz
    println!(
        "   Source audio: {} samples ({:.2}s)",
        source_audio.len(),
        source_audio.len() as f32 / 16000.0
    );

    // Step 4: Define multiple target voices with priorities
    println!("\n4. Defining multiple target voices...");

    let targets = vec![
        // Target 1: Young male voice (high priority = 100)
        NamedTarget::new(
            "male_young".to_string(),
            ConversionTarget::new(VoiceCharacteristics::for_gender(Gender::Male)),
        )
        .with_priority(100),
        // Target 2: Older female voice (medium priority = 50)
        NamedTarget::new(
            "female_senior".to_string(),
            ConversionTarget::new(VoiceCharacteristics::for_age(AgeGroup::Senior)),
        )
        .with_priority(50),
        // Target 3: Child voice (medium priority = 50)
        NamedTarget::new(
            "child".to_string(),
            ConversionTarget::new(VoiceCharacteristics::for_age(AgeGroup::Child)),
        )
        .with_priority(50),
        // Target 4: Deep male voice (low priority = 10)
        NamedTarget::new(
            "male_deep".to_string(),
            ConversionTarget::new({
                let mut chars = VoiceCharacteristics::for_gender(Gender::Male);
                chars.pitch.mean_f0 = 95.0; // Deep voice
                chars.age_group = Some(AgeGroup::MiddleAged);
                chars
            }),
        )
        .with_priority(10),
        // Target 5: High-pitched female (high priority = 100)
        NamedTarget::new(
            "female_high_pitch".to_string(),
            ConversionTarget::new({
                let mut chars = VoiceCharacteristics::for_gender(Gender::Female);
                chars.pitch.mean_f0 = 250.0; // High pitch
                chars.age_group = Some(AgeGroup::YoungAdult);
                chars
            }),
        )
        .with_priority(100),
    ];

    println!("   Defined {} target voices:", targets.len());
    for target in &targets {
        let chars = &target.target.characteristics;
        println!(
            "   - {}: {:?} age_group {:?}, pitch {:.0}Hz (priority: {})",
            target.name, chars.gender, chars.age_group, chars.pitch.mean_f0, target.priority
        );
    }

    // Step 5: Create multi-target conversion request
    println!("\n5. Creating conversion request...");
    let request = MultiTargetConversionRequest::new(
        "demo_request_1".to_string(),
        source_audio.clone(),
        16000,
        ConversionType::SpeakerConversion,
        targets.clone(),
    )
    .with_quality_level(0.85);

    println!("   Request ID: {}", request.id);
    println!("   Conversion type: {:?}", request.conversion_type);
    println!("   Quality level: {}", request.quality_level);

    // Step 6: Perform multi-target conversion
    println!("\n6. Performing multi-target conversion (parallel mode)...");
    let start = std::time::Instant::now();

    let results = converter.convert_multi_target(request).await?;

    let duration = start.elapsed();
    println!("   Parallel conversion completed in {:?}", duration);
    println!("   Success: {}", results.success);
    println!("   Targets processed: {}", results.stats.targets_processed);
    println!(
        "   Successful conversions: {}",
        results.stats.successful_conversions
    );
    println!(
        "   Failed conversions: {}",
        results.stats.failed_conversions
    );

    // Step 7: Analyze results
    println!("\n7. Analyzing conversion results...");
    for (name, result) in &results.target_results {
        if result.success {
            let quality = result
                .objective_quality
                .as_ref()
                .map(|q| q.overall_score)
                .unwrap_or(0.0);
            println!(
                "   {}: {} samples, quality: {:.2}",
                name,
                result.converted_audio.len(),
                quality
            );
        } else {
            println!(
                "   {}: FAILED - {}",
                name,
                result.error_message.as_deref().unwrap_or("Unknown error")
            );
        }
    }

    // Step 8: Processing statistics
    println!("\n8. Processing Statistics:");
    let stats = &results.stats;
    println!("   - Targets processed: {}", stats.targets_processed);
    println!("   - Successful: {}", stats.successful_conversions);
    println!("   - Failed: {}", stats.failed_conversions);
    println!(
        "   - Average processing time: {:?}",
        stats.average_processing_time
    );
    println!("   - Max processing time: {:?}", stats.max_processing_time);
    println!("   - Min processing time: {:?}", stats.min_processing_time);
    println!("   - Parallel processing: {}", stats.parallel_processing);

    // Step 9: Compare with sequential processing
    println!("\n9. Comparing with sequential processing...");
    let sequential_converter = MultiTargetConverter::new(VoiceConverter::new()?)
        .with_processing_mode(ProcessingMode::Sequential)
        .with_max_concurrent_targets(1);

    // Use first 3 targets for comparison
    let sequential_targets = targets.iter().take(3).cloned().collect();
    let sequential_request = MultiTargetConversionRequest::new(
        "demo_request_2".to_string(),
        source_audio.clone(),
        16000,
        ConversionType::SpeakerConversion,
        sequential_targets,
    );

    let sequential_start = std::time::Instant::now();
    let sequential_results = sequential_converter
        .convert_multi_target(sequential_request)
        .await?;
    let sequential_duration = sequential_start.elapsed();

    println!(
        "   Sequential conversion completed in {:?}",
        sequential_duration
    );
    println!(
        "   Speedup potential: {:.2}x",
        sequential_duration.as_secs_f64() / duration.as_secs_f64()
    );

    // Step 10: Processing mode demonstration
    println!("\n10. Available processing modes:");
    println!("   - Sequential: Process targets one by one (lower memory)");
    println!("   - Parallel: Process targets concurrently (faster, higher memory)");
    println!("   - Adaptive: Automatically choose based on system resources");

    // Step 11: Priority demonstration
    println!("\n11. Priority-based processing:");
    println!("   High priority targets (100):");
    for target in &targets {
        if target.priority >= 100 {
            println!("   - {}", target.name);
        }
    }
    println!("   Medium priority targets (50):");
    for target in &targets {
        if target.priority >= 50 && target.priority < 100 {
            println!("   - {}", target.name);
        }
    }
    println!("   Low priority targets (<50):");
    for target in &targets {
        if target.priority < 50 {
            println!("   - {}", target.name);
        }
    }

    // Step 12: Resource usage analysis
    println!("\n12. Resource Usage Analysis:");
    println!("   - Parallel mode:");
    println!("     * Wall-clock time: {:?}", duration);
    println!(
        "     * Targets per second: {:.2}",
        targets.len() as f64 / duration.as_secs_f64()
    );
    println!("   - Sequential mode (3 targets):");
    println!("     * Wall-clock time: {:?}", sequential_duration);
    println!(
        "     * Targets per second: {:.2}",
        3.0 / sequential_duration.as_secs_f64()
    );

    println!("\n=== Example completed successfully ===");
    println!("\nNote: This is a demonstration of the API structure.");
    println!("For full functionality, you would need to:");
    println!("  1. Use actual high-quality audio recordings");
    println!("  2. Configure voice converter with appropriate models");
    println!("  3. Implement custom target parameters if needed");
    println!("  4. Add proper error handling for production use");
    println!("\nSee the test files for complete working examples.");

    Ok(())
}

/// Generate sample audio with specific fundamental frequency
fn generate_sample_audio(f0_hz: f32, sample_count: usize) -> Vec<f32> {
    use std::f32::consts::PI;

    (0..sample_count)
        .map(|i| {
            let t = i as f32 / 16000.0;
            // Generate harmonics for more realistic voice
            let fundamental = (2.0 * PI * f0_hz * t).sin() * 0.5;
            let harmonic2 = (2.0 * PI * f0_hz * 2.0 * t).sin() * 0.25;
            let harmonic3 = (2.0 * PI * f0_hz * 3.0 * t).sin() * 0.15;

            // Add some amplitude modulation for naturalness
            let envelope = (PI * 2.0 * t).sin().abs();
            (fundamental + harmonic2 + harmonic3) * envelope
        })
        .collect()
}

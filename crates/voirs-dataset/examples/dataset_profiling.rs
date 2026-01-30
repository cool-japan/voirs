//! Example demonstrating comprehensive dataset profiling

use voirs_dataset::datasets::dummy::{DummyConfig, DummyDataset};
use voirs_dataset::profiling::DatasetProfiler;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dataset Profiling Example ===\n");

    // Create a dummy dataset for demonstration
    println!("Creating dummy dataset with 50 samples...");
    let config = DummyConfig {
        num_samples: 50,
        sample_rate: 22050,
        min_duration: 1.0,
        max_duration: 5.0,
        seed: Some(42), // Use fixed seed for reproducibility
        ..Default::default()
    };

    let dataset = DummyDataset::with_config(config);

    println!("Dataset created. Starting profiling...\n");

    // Profile the dataset
    let profile = DatasetProfiler::profile(&dataset).await?;

    // Generate and print the report
    println!("{}", profile.generate_report());

    // Save the profile to a JSON file
    let output_path = std::env::temp_dir().join("dataset_profile.json");
    profile.save_json(&output_path)?;

    println!("\n=== Profile Details ===\n");

    // Show detailed sample rate distribution
    println!("Sample Rate Distribution:");
    for (rate, count) in &profile.audio_stats.sample_rate_distribution {
        println!("  {} Hz: {} samples", rate, count);
    }

    // Show top words
    println!("\nTop 10 Most Common Words:");
    for (i, (word, count)) in profile.text_stats.common_words.iter().enumerate() {
        println!("  {}. '{}': {}", i + 1, word, count);
    }

    // Show speaker distribution
    println!("\nSamples per Speaker:");
    let mut speakers: Vec<_> = profile.speaker_stats.samples_per_speaker.iter().collect();
    speakers.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
    for (speaker, count) in speakers {
        let duration = profile
            .speaker_stats
            .duration_per_speaker
            .get(speaker)
            .unwrap_or(&0.0);
        println!("  {}: {} samples ({:.2}s)", speaker, count, duration);
    }

    // Show quality indicators
    println!("\n=== Quality Indicators ===");
    println!(
        "  Balance Score: {:.3} (1.0 = perfectly balanced)",
        profile.speaker_stats.balance_score
    );
    println!(
        "  Dynamic Range: {:.2} dB (higher = better)",
        profile.audio_stats.avg_dynamic_range_db
    );
    println!(
        "  Silence Ratio: {:.2}% (lower = more speech content)",
        profile.audio_stats.silence_ratio
    );

    println!("\n=== Performance Analysis ===");
    println!(
        "  Profiling completed in {:.2}s",
        profile.performance_stats.profiling_time_seconds
    );
    println!(
        "  Average load time: {:.2}ms per sample",
        profile.performance_stats.avg_sample_load_time_ms
    );
    println!(
        "  Throughput: {:.1} samples/second",
        profile.performance_stats.samples_per_second
    );

    println!("\nProfile saved to: {:?}", output_path);

    // Demonstrate loading the profile back
    println!("\nLoading profile from file...");
    let loaded_profile = voirs_dataset::profiling::DatasetProfile::load_json(&output_path)?;
    println!(
        "Successfully loaded. Total samples: {}",
        loaded_profile.basic_stats.total_samples
    );

    Ok(())
}

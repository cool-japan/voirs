//! Comprehensive example demonstrating advanced dataset statistics
//!
//! This example shows how to:
//! - Compute comprehensive statistical analyses of datasets
//! - Analyze distributions (duration, text length, quality)
//! - Generate histograms and percentiles
//! - Compute language and speaker statistics
//! - Perform correlation analysis
//! - Export statistics to JSON
//!
//! Run with: cargo run --example dataset_statistics

use voirs_dataset::{
    datasets::statistics::{DatasetStatistics, StatisticsConfig},
    AudioData, DatasetSample, LanguageCode, QualityMetrics, SpeakerInfo,
};

fn create_diverse_dataset(count: usize) -> Vec<DatasetSample> {
    let mut samples = Vec::new();

    for i in 0..count {
        // Varying durations (0.5s to 10s)
        let duration = 0.5 + (i as f32 % 20.0) * 0.5;

        // Varying text lengths
        let text = match i % 5 {
            0 => "Short text.".to_string(),
            1 => "Medium length text with some more words.".to_string(),
            2 => "This is a longer text sample with multiple sentences. It contains more information.".to_string(),
            3 => "Very long text sample that goes on and on with lots of details and information about various topics and subjects that might be interesting.".to_string(),
            _ => format!("Sample text number {} with varying content", i),
        };

        let audio = AudioData::silence(duration, 22050, 1);

        // Varying quality metrics
        let base_quality = 0.7 + (i as f64 % 20.0) * 0.015;
        let quality = QualityMetrics {
            overall_quality: Some(base_quality as f32),
            snr: Some((15.0 + (i as f64 % 15.0) * 1.0) as f32),
            clipping: Some((0.001 + (i as f64 % 10.0) * 0.01) as f32),
            dynamic_range: Some((35.0 + (i as f64 % 20.0) * 1.0) as f32),
            spectral_quality: Some((0.75 + (i as f64 % 15.0) * 0.015) as f32),
        };

        // Multiple speakers with different characteristics
        let speaker_id = i % 8; // 8 different speakers
        let speaker = SpeakerInfo {
            id: format!("speaker_{:03}", speaker_id),
            name: Some(format!("Speaker {}", speaker_id)),
            gender: Some(if speaker_id % 2 == 0 {
                "male".to_string()
            } else {
                "female".to_string()
            }),
            age: Some((20 + (speaker_id * 5)) as u32),
            accent: if speaker_id % 3 == 0 {
                Some("British".to_string())
            } else {
                Some("American".to_string())
            },
            metadata: Default::default(),
        };

        // Multiple languages
        let language = match i % 4 {
            0 => LanguageCode::EnUs,
            1 => LanguageCode::Ja,
            2 => LanguageCode::De,
            _ => LanguageCode::Fr,
        };

        let sample = DatasetSample::new(format!("sample_{:05}", i), text, audio, language)
            .with_quality(quality)
            .with_speaker(speaker);

        samples.push(sample);
    }

    samples
}

fn main() {
    println!("=== VoiRS Advanced Dataset Statistics Example ===\n");

    // Create a diverse test dataset
    let samples = create_diverse_dataset(200);
    println!("Created test dataset with {} samples\n", samples.len());

    // Example 1: Basic statistics with default configuration
    println!("Example 1: Basic statistics (default config)");
    println!("─────────────────────────────────────────────\n");
    {
        let stats = DatasetStatistics::from_samples(&samples);
        stats.print_summary();
    }

    // Example 2: Fast statistics (minimal computation)
    println!("\n\nExample 2: Fast statistics (no percentiles, no deep analysis)");
    println!("──────────────────────────────────────────────────────────────\n");
    {
        let config = StatisticsConfig::fast();
        let stats = DatasetStatistics::from_samples_with_config(&samples, config);

        println!("Total samples: {}", stats.total_samples);
        println!("Mean duration: {:.2}s", stats.duration_stats.mean);
        println!(
            "Mean text length: {:.0} chars",
            stats.text_length_stats.mean
        );
        println!("Mean quality: {:.3}", stats.quality_stats.mean);
    }

    // Example 3: Comprehensive statistics with all features
    println!("\n\nExample 3: Comprehensive statistics (full analysis)");
    println!("───────────────────────────────────────────────────────\n");
    {
        let config = StatisticsConfig::comprehensive();
        let stats = DatasetStatistics::from_samples_with_config(&samples, config);

        println!("=== Comprehensive Analysis ===");
        println!("Total samples: {}", stats.total_samples);
        println!("Total duration: {:.2} hours", stats.total_duration / 3600.0);

        println!("\n--- Detailed Percentiles (Duration) ---");
        for (key, value) in &stats.duration_stats.percentiles {
            println!("  {}: {:.2}s", key, value);
        }

        println!("\n--- Distribution Shape Analysis ---");
        println!("Duration skewness: {:.3}", stats.duration_stats.skewness);
        println!("Duration kurtosis: {:.3}", stats.duration_stats.kurtosis);
        println!(
            "Approximately normal: {}",
            stats.duration_stats.is_approximately_normal()
        );

        let (lower, upper) = stats.duration_stats.outlier_bounds();
        println!("Outlier bounds (IQR method): [{:.2}, {:.2}]", lower, upper);
    }

    // Example 4: Language-specific analysis
    println!("\n\nExample 4: Language-specific statistics");
    println!("────────────────────────────────────────\n");
    {
        let config = StatisticsConfig {
            include_language_stats: true,
            include_speaker_stats: false,
            include_correlations: false,
            ..Default::default()
        };
        let stats = DatasetStatistics::from_samples_with_config(&samples, config);

        if let Some(ref lang_stats) = stats.language_stats {
            println!("Languages in dataset: {}", lang_stats.sample_counts.len());

            for (lang, count) in &lang_stats.sample_counts {
                println!("\n{} ({} samples):", lang.as_str(), count);

                if let Some(dur_stats) = lang_stats.duration_stats.get(lang) {
                    println!(
                        "  Duration: mean={:.2}s, median={:.2}s",
                        dur_stats.mean, dur_stats.median
                    );
                }

                if let Some(text_stats) = lang_stats.text_length_stats.get(lang) {
                    println!(
                        "  Text length: mean={:.0} chars, median={:.0} chars",
                        text_stats.mean, text_stats.median
                    );
                }

                if let Some(qual_stats) = lang_stats.quality_stats.get(lang) {
                    println!(
                        "  Quality: mean={:.3}, median={:.3}",
                        qual_stats.mean, qual_stats.median
                    );
                }
            }
        }
    }

    // Example 5: Speaker analysis
    println!("\n\nExample 5: Speaker statistics");
    println!("──────────────────────────────\n");
    {
        let config = StatisticsConfig {
            include_speaker_stats: true,
            include_language_stats: false,
            include_correlations: false,
            ..Default::default()
        };
        let stats = DatasetStatistics::from_samples_with_config(&samples, config);

        if let Some(ref speaker_stats) = stats.speaker_stats {
            println!("Total unique speakers: {}", speaker_stats.unique_speakers);
            println!(
                "Avg samples per speaker: {:.1}",
                stats.total_samples as f64 / speaker_stats.unique_speakers as f64
            );

            println!("\n--- Gender Distribution ---");
            for (gender, count) in &speaker_stats.gender_distribution {
                println!("  {}: {} speakers", gender, count);
            }

            println!("\n--- Age Statistics ---");
            println!(
                "  Mean age: {:.1} years",
                speaker_stats.age_distribution.mean
            );
            println!(
                "  Age range: {:.0} - {:.0} years",
                speaker_stats.age_distribution.min, speaker_stats.age_distribution.max
            );

            println!("\n--- Top 5 speakers by sample count ---");
            let mut speaker_counts: Vec<_> = speaker_stats.samples_per_speaker.iter().collect();
            speaker_counts.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

            for (i, (speaker_id, count)) in speaker_counts.iter().take(5).enumerate() {
                if let Some(dur_stats) = speaker_stats.duration_per_speaker.get(*speaker_id) {
                    println!(
                        "  {}. {}: {} samples, avg duration: {:.2}s",
                        i + 1,
                        speaker_id,
                        count,
                        dur_stats.mean
                    );
                }
            }
        }
    }

    // Example 6: Correlation analysis
    println!("\n\nExample 6: Correlation analysis");
    println!("────────────────────────────────\n");
    {
        let config = StatisticsConfig {
            include_correlations: true,
            include_speaker_stats: false,
            include_language_stats: false,
            ..Default::default()
        };
        let stats = DatasetStatistics::from_samples_with_config(&samples, config);

        if let Some(ref corr) = stats.correlations {
            println!("=== Correlation Coefficients ===");
            println!("(Values range from -1 to +1)");
            println!("  -1: Perfect negative correlation");
            println!("   0: No correlation");
            println!("  +1: Perfect positive correlation\n");

            println!("Duration ↔ Text Length: {:.3}", corr.duration_text_length);
            println!("Duration ↔ Quality:     {:.3}", corr.duration_quality);
            println!("Text Length ↔ Quality:  {:.3}", corr.text_length_quality);
            println!("SNR ↔ Quality:          {:.3}", corr.snr_quality);

            println!("\n--- Interpretation ---");
            if corr.duration_text_length > 0.5 {
                println!("✓ Strong positive correlation between duration and text length");
            }
            if corr.snr_quality > 0.5 {
                println!("✓ Higher SNR correlates with better quality");
            }
        }
    }

    // Example 7: Histogram analysis
    println!("\n\nExample 7: Histogram analysis");
    println!("──────────────────────────────\n");
    {
        let stats = DatasetStatistics::from_samples(&samples);

        println!("--- Duration Histogram ---");
        println!("Bin width: {:.2}s", stats.duration_histogram.bin_width);
        println!("Number of bins: {}", stats.duration_histogram.bins.len());

        if let Some(mode) = stats.duration_histogram.mode() {
            println!("Mode (most common duration): {:.2}s", mode);
        }

        // Print histogram (simplified)
        println!("\nDistribution:");
        for (i, (&bin_start, &count)) in stats
            .duration_histogram
            .bins
            .iter()
            .zip(stats.duration_histogram.counts.iter())
            .enumerate()
        {
            if count > 0 {
                let bin_end = bin_start + stats.duration_histogram.bin_width;
                let bar = "█".repeat((count as f64 / 5.0).ceil() as usize);
                println!("  [{:.1}s - {:.1}s): {} {}", bin_start, bin_end, count, bar);

                // Only show first 15 bins to avoid clutter
                if i >= 14 {
                    let remaining = stats.duration_histogram.bins.len() - i - 1;
                    if remaining > 0 {
                        println!("  ... ({} more bins)", remaining);
                    }
                    break;
                }
            }
        }
    }

    // Example 8: Quality gate analysis
    println!("\n\nExample 8: Quality gate analysis");
    println!("─────────────────────────────────\n");
    {
        let stats = DatasetStatistics::from_samples(&samples);

        // Define quality thresholds
        let min_acceptable_quality = 0.8;
        let min_acceptable_snr = 20.0;

        // Count samples meeting quality gates
        let high_quality_samples = samples
            .iter()
            .filter(|s| {
                s.quality.overall_quality.unwrap_or(0.0) >= min_acceptable_quality
                    && s.quality.snr.unwrap_or(0.0) >= min_acceptable_snr
            })
            .count();

        let pass_rate = (high_quality_samples as f64 / samples.len() as f64) * 100.0;

        println!("Quality Gate Thresholds:");
        println!("  Minimum overall quality: {:.2}", min_acceptable_quality);
        println!("  Minimum SNR: {:.1} dB", min_acceptable_snr);
        println!("\nResults:");
        println!(
            "  Samples passing: {} / {}",
            high_quality_samples,
            samples.len()
        );
        println!("  Pass rate: {:.1}%", pass_rate);

        println!("\nActual quality distribution:");
        println!("  Mean: {:.3}", stats.quality_stats.mean);
        println!("  Median: {:.3}", stats.quality_stats.median);
        println!(
            "  Below threshold: {} samples ({:.1}%)",
            samples.len() - high_quality_samples,
            100.0 - pass_rate
        );
    }

    // Example 9: Export to JSON
    println!("\n\nExample 9: Export statistics to JSON");
    println!("──────────────────────────────────────\n");
    {
        let stats = DatasetStatistics::from_samples(&samples);

        match stats.to_json() {
            Ok(json) => {
                println!("✓ Successfully exported statistics to JSON");
                println!("JSON size: {} bytes", json.len());
                println!("\nFirst 200 characters:");
                println!("{}", &json[..200.min(json.len())]);
                println!("...\n");

                // Verify we can re-import
                match DatasetStatistics::from_json(&json) {
                    Ok(reimported) => {
                        println!("✓ Successfully re-imported statistics from JSON");
                        println!(
                            "  Verified: {} samples, {:.2}s total duration",
                            reimported.total_samples, reimported.total_duration
                        );
                    }
                    Err(e) => println!("✗ Failed to re-import: {}", e),
                }
            }
            Err(e) => println!("✗ Failed to export: {}", e),
        }
    }

    // Example 10: Comparing different configurations
    println!("\n\nExample 10: Performance comparison of different configs");
    println!("────────────────────────────────────────────────────────\n");
    {
        use std::time::Instant;

        let configs = vec![
            ("Fast", StatisticsConfig::fast()),
            ("Default", StatisticsConfig::default()),
            ("Comprehensive", StatisticsConfig::comprehensive()),
        ];

        for (name, config) in configs {
            let start = Instant::now();
            let stats = DatasetStatistics::from_samples_with_config(&samples, config);
            let duration = start.elapsed();

            println!("{} config:", name);
            println!(
                "  Computation time: {:.2}ms",
                duration.as_secs_f64() * 1000.0
            );
            println!(
                "  Includes percentiles: {}",
                stats.config.include_percentiles
            );
            println!(
                "  Includes speaker stats: {}",
                stats.speaker_stats.is_some()
            );
            println!(
                "  Includes language stats: {}",
                stats.language_stats.is_some()
            );
            println!("  Includes correlations: {}", stats.correlations.is_some());
            println!();
        }
    }

    println!("=== All examples completed successfully! ===");
}

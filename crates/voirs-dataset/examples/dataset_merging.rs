//! Comprehensive example demonstrating dataset merging capabilities
//!
//! This example shows how to:
//! - Merge multiple datasets with different configurations
//! - Handle ID and speaker conflicts
//! - Apply quality filtering during merge
//! - Generate merge reports and statistics
//!
//! Run with: cargo run --example dataset_merging

use voirs_dataset::{
    datasets::merger::{
        DatasetMerger, IdConflictStrategy, MergeConfig, QualityFilterStrategy, SpeakerMergeStrategy,
    },
    AudioData, DatasetSample, LanguageCode, QualityMetrics, SpeakerInfo,
};

fn create_sample_dataset(
    name: &str,
    count: usize,
    lang: LanguageCode,
    quality_range: (f32, f32),
) -> Vec<DatasetSample> {
    let mut samples = Vec::new();

    for i in 0..count {
        let id = format!("{:03}", i + 1);
        let text = format!("Sample text from {} number {}", name, i + 1);
        let audio = AudioData::silence(1.0 + (i as f32 * 0.1), 22050, 1);

        // Varying quality scores
        let quality =
            quality_range.0 + (quality_range.1 - quality_range.0) * (i as f32 / count as f32);

        let speaker = SpeakerInfo {
            id: format!("speaker_{}", (i % 3) + 1),
            name: Some(format!("Speaker {}", (i % 3) + 1)),
            gender: Some(if i % 2 == 0 {
                "male".to_string()
            } else {
                "female".to_string()
            }),
            age: Some((25 + (i % 3) * 10) as u32),
            accent: None,
            metadata: Default::default(),
        };

        let sample = DatasetSample::new(id, text, audio, lang)
            .with_quality(QualityMetrics {
                overall_quality: Some(quality),
                snr: Some(20.0 + quality * 10.0),
                clipping: Some(0.01 * (1.0 - quality)),
                dynamic_range: Some(40.0 + quality * 10.0),
                spectral_quality: Some(quality),
            })
            .with_speaker(speaker);

        samples.push(sample);
    }

    samples
}

fn main() {
    println!("=== VoiRS Dataset Merging Example ===\n");

    // Example 1: Basic merge with ID conflict resolution
    println!("Example 1: Basic merge with ID renaming");
    println!("─────────────────────────────────────");
    {
        let config = MergeConfig {
            id_conflict_strategy: IdConflictStrategy::RenameWithPrefix,
            speaker_strategy: SpeakerMergeStrategy::PrefixWithDataset,
            ..Default::default()
        };

        let mut merger = DatasetMerger::new(config);

        let dataset1 = create_sample_dataset("LJSpeech", 5, LanguageCode::EnUs, (0.8, 0.95));
        let dataset2 = create_sample_dataset("VCTK", 5, LanguageCode::EnGb, (0.75, 0.9));

        merger.add_dataset("ljspeech", dataset1).unwrap();
        merger.add_dataset("vctk", dataset2).unwrap();

        let summary = merger.summary();
        summary.print();

        let merged = merger.into_samples();
        println!("\nMerged {} samples total\n", merged.len());
    }

    // Example 2: Quality-filtered merge
    println!("\nExample 2: Quality-filtered merge");
    println!("──────────────────────────────────");
    {
        let config = MergeConfig {
            quality_filter: QualityFilterStrategy::MinimumQuality { threshold: 0.85 },
            ..Default::default()
        };

        let mut merger = DatasetMerger::new(config);

        let dataset1 = create_sample_dataset("HighQuality", 10, LanguageCode::EnUs, (0.9, 0.98));
        let dataset2 = create_sample_dataset("MixedQuality", 10, LanguageCode::EnUs, (0.5, 0.95));

        merger.add_dataset("high_quality", dataset1).unwrap();
        merger.add_dataset("mixed_quality", dataset2).unwrap();

        let summary = merger.summary();
        println!(
            "Total samples after filtering: {} (filtered: {})",
            summary.total_samples, summary.total_filtered
        );
        println!("Average quality: {:.3}", summary.average_quality);
    }

    // Example 3: Balanced merge (limiting samples per dataset)
    println!("\nExample 3: Balanced merge");
    println!("─────────────────────────");
    {
        let config = MergeConfig::balanced(3); // Max 3 samples per dataset

        let mut merger = DatasetMerger::new(config);

        let dataset1 = create_sample_dataset("Large", 10, LanguageCode::EnUs, (0.8, 0.95));
        let dataset2 = create_sample_dataset("Small", 2, LanguageCode::EnGb, (0.85, 0.9));

        merger.add_dataset("large_dataset", dataset1).unwrap();
        merger.add_dataset("small_dataset", dataset2).unwrap();

        let summary = merger.summary();
        summary.print();
    }

    // Example 4: Language-specific merge
    println!("\nExample 4: Language-specific merge");
    println!("───────────────────────────────────");
    {
        let config = MergeConfig::language_filter(vec![LanguageCode::EnUs, LanguageCode::Ja]);

        let mut merger = DatasetMerger::new(config);

        let dataset1 = create_sample_dataset("English", 5, LanguageCode::EnUs, (0.8, 0.95));
        let dataset2 = create_sample_dataset("Japanese", 5, LanguageCode::Ja, (0.85, 0.92));
        let dataset3 = create_sample_dataset("German", 5, LanguageCode::De, (0.82, 0.9));

        merger.add_dataset("english", dataset1).unwrap();
        merger.add_dataset("japanese", dataset2).unwrap();
        merger.add_dataset("german", dataset3).unwrap();

        let summary = merger.summary();
        println!(
            "Languages in merged dataset: {:?}",
            summary.language_distribution.keys()
        );
        println!(
            "Total samples: {} (filtered: {})",
            summary.total_samples, summary.total_filtered
        );
    }

    // Example 5: Keep highest quality on conflicts
    println!("\nExample 5: Keep highest quality on ID conflicts");
    println!("────────────────────────────────────────────────");
    {
        let config = MergeConfig {
            id_conflict_strategy: IdConflictStrategy::KeepHighestQuality,
            ..Default::default()
        };

        let mut merger = DatasetMerger::new(config);

        // Create datasets with overlapping IDs but different quality
        let mut dataset1 = create_sample_dataset("Source1", 3, LanguageCode::EnUs, (0.6, 0.7));
        let mut dataset2 = create_sample_dataset("Source2", 3, LanguageCode::EnUs, (0.9, 0.95));

        // Make IDs overlap
        for (i, sample) in dataset2.iter_mut().enumerate() {
            sample.id = format!("{:03}", i + 1);
        }

        merger.add_dataset("source1", dataset1).unwrap();
        merger.add_dataset("source2", dataset2).unwrap();

        let merged = merger.into_samples();
        println!("Merged {} samples (kept highest quality)", merged.len());
        for sample in &merged {
            println!(
                "  ID: {}, Quality: {:.3}",
                sample.id,
                sample.quality.overall_quality.unwrap_or(0.0)
            );
        }
    }

    // Example 6: Advanced merge with combined quality filtering
    println!("\nExample 6: Advanced merge with combined quality criteria");
    println!("──────────────────────────────────────────────────────────");
    {
        let config = MergeConfig {
            quality_filter: QualityFilterStrategy::Combined {
                min_quality: Some(0.8),
                min_snr: Some(25.0),
                max_clipping: Some(0.05),
            },
            deduplicate_text: true,
            preserve_source_info: true,
            ..Default::default()
        };

        let mut merger = DatasetMerger::new(config);

        let dataset1 = create_sample_dataset("Premium", 8, LanguageCode::EnUs, (0.85, 0.98));
        let dataset2 = create_sample_dataset("Standard", 8, LanguageCode::EnUs, (0.7, 0.85));

        merger.add_dataset("premium", dataset1).unwrap();
        merger.add_dataset("standard", dataset2).unwrap();

        let summary = merger.summary();
        summary.print();

        let samples = merger.samples();
        println!("\nSample details:");
        for (i, sample) in samples.iter().take(3).enumerate() {
            println!(
                "  Sample {}: ID={}, Source={:?}, Quality={:.3}",
                i + 1,
                sample.id,
                sample.metadata.get("source_dataset"),
                sample.quality.overall_quality.unwrap_or(0.0)
            );
        }
    }

    println!("\n=== All examples completed successfully! ===");
}

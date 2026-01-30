//! Comprehensive example demonstrating deduplication and audio fingerprinting
//!
//! This example shows how to:
//! - Detect duplicate samples using text matching
//! - Detect similar audio using perceptual fingerprinting
//! - Configure deduplication strategies
//! - Generate deduplication reports
//!
//! Run with: cargo run --example deduplication_fingerprinting

use voirs_dataset::{
    audio::fingerprint::{AudioFingerprint, FingerprintConfig, FingerprintIndex},
    processing::deduplication::{DeduplicationConfig, Deduplicator, DuplicateSelectionStrategy},
    AudioData, DatasetSample, LanguageCode, QualityMetrics,
};

fn create_test_samples() -> Vec<DatasetSample> {
    let mut samples = Vec::new();

    // Sample 1: Original
    let audio1 = AudioData::silence(1.0, 22050, 1);
    let sample1 = DatasetSample::new(
        "001".to_string(),
        "Hello, this is a test sample.".to_string(),
        audio1,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.95),
        snr: Some(30.0),
        clipping: Some(0.01),
        dynamic_range: Some(50.0),
        spectral_quality: Some(0.92),
    });

    // Sample 2: Exact text duplicate
    let audio2 = AudioData::silence(1.1, 22050, 1);
    let sample2 = DatasetSample::new(
        "002".to_string(),
        "Hello, this is a test sample.".to_string(), // Same text
        audio2,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.85),
        snr: Some(25.0),
        clipping: Some(0.02),
        dynamic_range: Some(45.0),
        spectral_quality: Some(0.88),
    });

    // Sample 3: Different text, different audio
    let audio3 = AudioData::silence(1.2, 22050, 1);
    let sample3 = DatasetSample::new(
        "003".to_string(),
        "This is a completely different sample.".to_string(),
        audio3,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.9),
        snr: Some(28.0),
        clipping: Some(0.015),
        dynamic_range: Some(48.0),
        spectral_quality: Some(0.9),
    });

    // Sample 4: Case variation
    let audio4 = AudioData::silence(1.0, 22050, 1);
    let sample4 = DatasetSample::new(
        "004".to_string(),
        "HELLO, THIS IS A TEST SAMPLE.".to_string(), // Same text, different case
        audio4,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.88),
        snr: Some(26.0),
        clipping: Some(0.018),
        dynamic_range: Some(46.0),
        spectral_quality: Some(0.87),
    });

    // Sample 5: Whitespace variation
    let audio5 = AudioData::silence(0.9, 22050, 1);
    let sample5 = DatasetSample::new(
        "005".to_string(),
        "Hello,   this   is   a   test   sample.".to_string(), // Same text, extra whitespace
        audio5,
        LanguageCode::EnUs,
    )
    .with_quality(QualityMetrics {
        overall_quality: Some(0.92),
        snr: Some(29.0),
        clipping: Some(0.012),
        dynamic_range: Some(49.0),
        spectral_quality: Some(0.91),
    });

    samples.push(sample1);
    samples.push(sample2);
    samples.push(sample3);
    samples.push(sample4);
    samples.push(sample5);

    samples
}

fn main() {
    println!("=== VoiRS Deduplication & Fingerprinting Example ===\n");

    // Example 1: Text-only deduplication
    println!("Example 1: Text-only deduplication");
    println!("───────────────────────────────────");
    {
        let config = DeduplicationConfig::text_only();
        let deduplicator = Deduplicator::new(config);

        let samples = create_test_samples();
        println!("Input samples: {}", samples.len());

        let (unique, report) = deduplicator.deduplicate(samples).unwrap();
        println!("Unique samples: {}", unique.len());
        println!("Duplicates removed: {}", report.duplicates_removed);
        println!("Exact text matches: {}", report.stats.exact_text_matches);
    }

    // Example 2: Case-insensitive deduplication
    println!("\nExample 2: Case-insensitive text deduplication");
    println!("───────────────────────────────────────────────");
    {
        let config = DeduplicationConfig {
            check_exact_text: true,
            check_audio_similarity: false,
            case_sensitive_text: false, // Ignore case
            normalize_whitespace: true,
            ..Default::default()
        };
        let deduplicator = Deduplicator::new(config);

        let samples = create_test_samples();
        let (unique, report) = deduplicator.deduplicate(samples).unwrap();

        println!("Unique samples (case-insensitive): {}", unique.len());
        println!("Duplicates removed: {}", report.duplicates_removed);

        println!("\nDuplicate groups:");
        for (i, group) in report.duplicate_groups.iter().enumerate() {
            println!(
                "  Group {}: {} samples, kept '{}', reason: {}",
                i + 1,
                group.sample_ids.len(),
                group.kept_sample_id,
                group.detection_reason
            );
        }
    }

    // Example 3: Keep highest quality strategy
    println!("\nExample 3: Keep highest quality duplicate");
    println!("──────────────────────────────────────────");
    {
        let config = DeduplicationConfig {
            check_exact_text: true,
            check_audio_similarity: false,
            selection_strategy: DuplicateSelectionStrategy::KeepHighestQuality,
            case_sensitive_text: false,
            normalize_whitespace: true,
            ..Default::default()
        };
        let deduplicator = Deduplicator::new(config);

        let samples = create_test_samples();
        let (unique, report) = deduplicator.deduplicate(samples).unwrap();

        println!("Kept samples:");
        for sample in &unique {
            println!(
                "  ID: {}, Quality: {:.3}, Text: {}",
                sample.id,
                sample.quality.overall_quality.unwrap_or(0.0),
                &sample.text[..30.min(sample.text.len())]
            );
        }
    }

    // Example 4: Audio fingerprinting
    println!("\nExample 4: Audio fingerprinting");
    println!("────────────────────────────────");
    {
        let config = FingerprintConfig::for_speech();
        let mut index = FingerprintIndex::new(config.clone());

        // Create some test audio samples
        let audio1 = AudioData::silence(1.0, 22050, 1);
        let audio2 = AudioData::silence(1.0, 22050, 1); // Identical
        let audio3 = AudioData::silence(2.0, 22050, 1); // Different duration

        index.add("sample1".to_string(), &audio1).unwrap();
        index.add("sample2".to_string(), &audio2).unwrap();
        index.add("sample3".to_string(), &audio3).unwrap();

        println!("Index size: {} samples", index.len());

        // Find duplicates
        let duplicates = index.find_duplicates(0.95);
        println!("Found {} duplicate pairs", duplicates.len());
        for (id1, id2, similarity) in &duplicates {
            println!("  '{}' <-> '{}': {:.3} similarity", id1, id2, similarity);
        }

        // Find similar to a query
        let query = AudioFingerprint::from_audio(&audio1, &config).unwrap();
        let similar = index.find_similar(&query, 0.90);
        println!("\nSimilar to sample1 (threshold: 0.90):");
        for (id, similarity) in &similar {
            println!("  {}: {:.3} similarity", id, similarity);
        }
    }

    // Example 5: Fingerprint configurations
    println!("\nExample 5: Fingerprint configuration presets");
    println!("─────────────────────────────────────────────");
    {
        let speech_config = FingerprintConfig::for_speech();
        println!("Speech config:");
        println!("  Frequency bands: {}", speech_config.num_bands);
        println!("  Window size: {}", speech_config.window_size);
        println!("  Min frequency: {} Hz", speech_config.min_freq);
        println!("  Max frequency: {} Hz", speech_config.max_freq);

        let music_config = FingerprintConfig::for_music();
        println!("\nMusic config:");
        println!("  Frequency bands: {}", music_config.num_bands);
        println!("  Window size: {}", music_config.window_size);

        let fast_config = FingerprintConfig::fast();
        println!("\nFast config (lower quality, faster):");
        println!("  Frequency bands: {}", fast_config.num_bands);
        println!("  Window size: {}", fast_config.window_size);
    }

    // Example 6: Combined deduplication report
    println!("\nExample 6: Detailed deduplication report");
    println!("─────────────────────────────────────────");
    {
        let config = DeduplicationConfig::strict(); // High threshold
        let deduplicator = Deduplicator::new(config);

        let samples = create_test_samples();
        let (unique, report) = deduplicator.deduplicate(samples).unwrap();

        println!("\n");
        report.print();

        println!("\nFinal unique samples:");
        for (i, sample) in unique.iter().enumerate() {
            println!(
                "  {}: ID={}, Quality={:.3}",
                i + 1,
                sample.id,
                sample.quality.overall_quality.unwrap_or(0.0)
            );
        }
    }

    // Example 7: Different selection strategies
    println!("\nExample 7: Comparing selection strategies");
    println!("──────────────────────────────────────────");
    {
        let strategies = vec![
            ("KeepFirst", DuplicateSelectionStrategy::KeepFirst),
            ("KeepLast", DuplicateSelectionStrategy::KeepLast),
            (
                "KeepHighestQuality",
                DuplicateSelectionStrategy::KeepHighestQuality,
            ),
        ];

        for (name, strategy) in strategies {
            let config = DeduplicationConfig {
                check_exact_text: true,
                check_audio_similarity: false,
                selection_strategy: strategy,
                case_sensitive_text: false,
                normalize_whitespace: true,
                ..Default::default()
            };

            let deduplicator = Deduplicator::new(config);
            let samples = create_test_samples();
            let (unique, _) = deduplicator.deduplicate(samples).unwrap();

            println!(
                "  {}: {} unique samples, kept IDs: {:?}",
                name,
                unique.len(),
                unique.iter().map(|s| &s.id).collect::<Vec<_>>()
            );
        }
    }

    println!("\n=== All examples completed successfully! ===");
}

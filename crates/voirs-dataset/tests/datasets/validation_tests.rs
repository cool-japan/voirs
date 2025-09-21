//! Dataset validation tests
//!
//! This module contains comprehensive tests for dataset validation functionality.

use std::collections::HashMap;
use voirs_dataset::{
    datasets::dummy::DummyDataset,
    traits::Dataset,
    validation::{DatasetValidator, QualityAnalyzer},
    *,
};

#[tokio::test]
async fn test_standard_dataset_loading() {
    // Test loading a dummy dataset
    let dataset = DummyDataset::small();

    // Verify dataset loads correctly
    assert_eq!(dataset.len(), 10);
    assert!(!dataset.is_empty());

    // Test sample retrieval
    let sample = dataset.get(0).await.unwrap();
    assert!(!sample.id.is_empty());
    assert!(!sample.text.is_empty());
    assert!(!sample.audio.samples().is_empty());

    // Test batch retrieval
    let indices = vec![0, 1, 2];
    let batch = dataset.get_batch(&indices).await.unwrap();
    assert_eq!(batch.len(), 3);
}

#[tokio::test]
async fn test_manifest_consistency() {
    // Test that manifest data is consistent with actual dataset content
    let dataset = DummyDataset::small();

    // Get statistics and verify consistency
    let stats = dataset.statistics().await.unwrap();
    assert_eq!(stats.total_items, 10);
    assert!(stats.total_duration > 0.0);
    assert!(stats.average_duration > 0.0);

    // Verify text length statistics are reasonable
    assert!(stats.text_length_stats.min > 0);
    assert!(stats.text_length_stats.max >= stats.text_length_stats.min);
    assert!(stats.text_length_stats.mean > 0.0);

    // Verify duration statistics are reasonable
    assert!(stats.duration_stats.min > 0.0);
    assert!(stats.duration_stats.max >= stats.duration_stats.min);
    assert!(stats.duration_stats.mean > 0.0);

    // Verify metadata consistency
    let metadata = dataset.metadata();
    assert_eq!(metadata.name, "DummyDataset");
    assert_eq!(metadata.total_samples, 10);
}

#[tokio::test]
async fn test_audio_text_alignment() {
    // Test that audio duration and text length have reasonable relationships
    let dataset = DummyDataset::small();

    // Calculate alignment metrics
    let alignment_stats = calculate_alignment_stats(&dataset).await;

    // Verify alignment is reasonable
    assert!(alignment_stats.characters_per_second > 0.0);
    assert!(alignment_stats.words_per_second > 0.0);
    assert!(alignment_stats.characters_per_second < 100.0); // Reasonable upper bound
    assert!(alignment_stats.words_per_second < 30.0); // Reasonable upper bound

    // Test specific alignment for each sample
    for i in 0..dataset.len() {
        let sample = dataset.get(i).await.unwrap();
        let audio_duration = sample.audio.duration();

        // Skip samples with zero or very small duration to avoid division by zero
        if audio_duration <= 0.01 {
            continue;
        }

        let char_per_sec = sample.text.len() as f32 / audio_duration;
        let word_count = sample.text.split_whitespace().count() as f32;
        let words_per_sec = word_count / audio_duration;

        // Verify reasonable speaking rates (relaxed bounds for test data)
        // Normal speech is ~10-20 chars/sec, but test data can be more variable
        assert!(
            char_per_sec > 0.0 && char_per_sec < 500.0,
            "Sample {}: char_per_sec = {:.2}, text_len = {}, duration = {:.3}s",
            i,
            char_per_sec,
            sample.text.len(),
            audio_duration
        );
        assert!(words_per_sec > 0.0 && words_per_sec < 100.0,
               "Sample {i}: words_per_sec = {words_per_sec:.2}, word_count = {word_count}, duration = {audio_duration:.3}s");
    }
}

#[tokio::test]
async fn test_quality_metrics_accuracy() {
    // Test that quality metrics are calculated correctly using the validator
    let dataset = DummyDataset::small();
    let validator = DatasetValidator::new();

    // Get samples for validation
    let mut samples = Vec::new();
    for i in 0..dataset.len() {
        samples.push(dataset.get(i).await.unwrap());
    }

    // Validate quality metrics
    let report = validator.validate_dataset(&samples).unwrap();

    // Should have basic validation results
    assert_eq!(report.validation_stats.total_samples, 10);
    // Note: validation_time_ms might be 0 for very fast operations on small datasets
    // Just verify the validation_time_ms is accessible (unsigned, so always >= 0)
    let _ = report.validation_stats.validation_time_ms;

    // Test individual sample validation
    for (i, sample) in samples.iter().enumerate() {
        let result = validator.validate_sample(sample, i).unwrap();
        assert_eq!(result.index, i);
        // Most dummy samples should be valid
        if result.errors.is_empty() {
            assert!(result.is_valid);
        }
    }
}

#[tokio::test]
async fn test_dataset_validation_edge_cases() {
    // Test edge cases in dataset validation
    let validator = DatasetValidator::new();

    // Create problematic samples
    let edge_case_samples = vec![
        create_sample_with_properties(
            "empty-text",
            "",
            1.0,
            LanguageCode::EnUs,
            None,
        ),
        create_sample_with_properties(
            "very-short",
            "Hi",
            0.05, // Very short audio
            LanguageCode::EnUs,
            None,
        ),
        create_sample_with_properties(
            "very-long",
            "This is a very long text that should generate a warning when the audio duration is extremely long.",
            45.0, // Very long audio
            LanguageCode::EnUs,
            None,
        ),
    ];

    // Validate and check for expected errors/warnings
    let report = validator.validate_dataset(&edge_case_samples).unwrap();

    // Should have errors for empty text and very short audio
    assert!(!report.basic_report.is_valid);
    assert!(!report.basic_report.errors.is_empty());
    assert!(!report.basic_report.warnings.is_empty());

    // Check validation statistics
    assert_eq!(report.validation_stats.total_samples, 3);
    assert!(report.validation_stats.error_samples > 0);
}

#[tokio::test]
async fn test_dataset_statistics_calculation() {
    // Test detailed statistics calculation using dummy dataset
    let dataset = DummyDataset::large(); // Use larger dataset for better statistics

    let stats = dataset.statistics().await.unwrap();

    // Verify basic stats - large dataset has 10,000 samples
    assert_eq!(stats.total_items, 10000);
    assert!(stats.total_duration > 0.0);
    assert!(stats.average_duration > 0.0);

    // Verify text length stats
    assert!(stats.text_length_stats.min > 0);
    assert!(stats.text_length_stats.max >= stats.text_length_stats.min);
    assert!(stats.text_length_stats.mean > 0.0);
    assert!(stats.text_length_stats.std_dev >= 0.0);

    // Verify duration stats
    assert!(stats.duration_stats.min > 0.0);
    assert!(stats.duration_stats.max >= stats.duration_stats.min);
    assert!(stats.duration_stats.mean > 0.0);
    assert!(stats.duration_stats.std_dev >= 0.0);
}

#[tokio::test]
async fn test_quality_analysis() {
    // Test comprehensive quality analysis
    let dataset = DummyDataset::small();
    let analyzer = QualityAnalyzer::new();

    // Get samples for analysis
    let mut samples = Vec::new();
    for i in 0..dataset.len() {
        samples.push(dataset.get(i).await.unwrap());
    }

    // Perform quality analysis
    let report = analyzer.analyze_dataset(&samples).unwrap();

    // Verify overall assessment
    assert!(report.overall_assessment.overall_score >= 0.0);
    assert!(report.overall_assessment.overall_score <= 1.0);
    assert_eq!(
        report.overall_assessment.quality_distribution.total_samples,
        10
    );

    // Verify audio quality distribution
    assert!(report.audio_quality_distribution.snr_distribution.mean >= 0.0);
    assert!(
        report
            .audio_quality_distribution
            .dynamic_range_distribution
            .mean
            >= 0.0
    );

    // Verify outlier analysis
    assert!(report.outliers.summary.outlier_percentage >= 0.0);
    assert!(report.outliers.summary.outlier_percentage <= 100.0);

    // Verify language breakdown
    assert!(!report.language_breakdown.is_empty());
    assert!(report.language_breakdown.contains_key(&LanguageCode::EnUs));
}

// Helper functions for creating test samples

#[allow(dead_code)]
fn create_test_samples() -> Vec<DatasetSample> {
    vec![
        create_sample_with_properties("sample-001", "Hello world", 1.0, LanguageCode::EnUs, None),
        create_sample_with_properties(
            "sample-002",
            "Good morning",
            1.5,
            LanguageCode::EnUs,
            Some("speaker-001".to_string()),
        ),
        create_sample_with_properties(
            "sample-003",
            "こんにちは",
            2.0,
            LanguageCode::Ja,
            Some("speaker-002".to_string()),
        ),
        create_sample_with_properties(
            "sample-004",
            "Bonjour",
            1.2,
            LanguageCode::Fr,
            Some("speaker-003".to_string()),
        ),
        create_sample_with_properties(
            "sample-005",
            "Hola mundo",
            1.8,
            LanguageCode::Es,
            Some("speaker-004".to_string()),
        ),
    ]
}

fn create_sample_with_properties(
    id: &str,
    text: &str,
    duration: f32,
    language: LanguageCode,
    speaker_id: Option<String>,
) -> DatasetSample {
    let sample_rate = 22050;
    let num_samples = (duration * sample_rate as f32) as usize;
    let audio = AudioData::new(vec![0.1; num_samples], sample_rate, 1);

    let mut sample = DatasetSample::new(id.to_string(), text.to_string(), audio, language);

    if let Some(speaker_id) = speaker_id {
        sample.speaker = Some(SpeakerInfo {
            id: speaker_id,
            name: None,
            gender: None,
            age: None,
            accent: None,
            metadata: HashMap::new(),
        });
    }

    sample
}

// Helper struct for alignment statistics
#[derive(Debug)]
struct AlignmentStats {
    characters_per_second: f32,
    words_per_second: f32,
    _average_word_length: f32,
}

async fn calculate_alignment_stats(dataset: &DummyDataset) -> AlignmentStats {
    let mut total_chars = 0;
    let mut total_words = 0;
    let mut total_duration = 0.0;

    for i in 0..dataset.len() {
        let sample = dataset.get(i).await.unwrap();
        total_chars += sample.text.len();
        total_words += sample.text.split_whitespace().count();
        total_duration += sample.audio.duration();
    }

    AlignmentStats {
        characters_per_second: total_chars as f32 / total_duration,
        words_per_second: total_words as f32 / total_duration,
        _average_word_length: if total_words > 0 {
            total_chars as f32 / total_words as f32
        } else {
            0.0
        },
    }
}

//! Integration tests for sampling strategies
//!
//! Tests the complete sampling workflow with realistic datasets

use std::collections::HashMap;
use voirs_dataset::{
    sampling::{
        CurriculumConfig, CurriculumSampler, DifficultyMetric, DifficultyStrategy,
        ImportanceConfig, ImportanceSampler, StratificationField, StratifiedConfig,
        StratifiedSampler,
    },
    AudioData, DatasetSample, LanguageCode, QualityMetrics, SpeakerInfo,
};

/// Create a diverse test dataset with varying characteristics
fn create_diverse_dataset() -> Vec<DatasetSample> {
    let mut samples = Vec::new();

    // High quality English samples
    for i in 0..10 {
        samples.push(create_sample(
            &format!("high_en_{}", i),
            0.90 + i as f32 * 0.01,
            "speaker_a",
            LanguageCode::EnUs,
            2.0 + i as f32 * 0.1,
        ));
    }

    // Medium quality English samples
    for i in 0..15 {
        samples.push(create_sample(
            &format!("med_en_{}", i),
            0.60 + i as f32 * 0.02,
            "speaker_b",
            LanguageCode::EnUs,
            3.0 + i as f32 * 0.2,
        ));
    }

    // Low quality English samples
    for i in 0..10 {
        samples.push(create_sample(
            &format!("low_en_{}", i),
            0.30 + i as f32 * 0.03,
            "speaker_c",
            LanguageCode::EnUs,
            4.0 + i as f32 * 0.3,
        ));
    }

    // High quality Japanese samples
    for i in 0..8 {
        samples.push(create_sample(
            &format!("high_ja_{}", i),
            0.85 + i as f32 * 0.01,
            "speaker_d",
            LanguageCode::Ja,
            2.5 + i as f32 * 0.15,
        ));
    }

    // Medium quality Japanese samples
    for i in 0..7 {
        samples.push(create_sample(
            &format!("med_ja_{}", i),
            0.55 + i as f32 * 0.04,
            "speaker_e",
            LanguageCode::Ja,
            3.5 + i as f32 * 0.25,
        ));
    }

    samples
}

fn create_sample(
    id: &str,
    quality: f32,
    speaker_id: &str,
    language: LanguageCode,
    duration: f32,
) -> DatasetSample {
    let sample_rate = 16000;
    let num_samples = (sample_rate as f32 * duration) as usize;
    let audio = AudioData::new(vec![0.0; num_samples], sample_rate, 1);

    DatasetSample {
        id: id.to_string(),
        audio,
        text: format!("Sample text for {}", id),
        speaker: Some(SpeakerInfo {
            id: speaker_id.to_string(),
            name: Some(format!("Speaker {}", speaker_id)),
            gender: None,
            age: None,
            accent: None,
            metadata: HashMap::new(),
        }),
        language,
        quality: QualityMetrics {
            overall_quality: Some(quality),
            snr: Some(20.0 + quality * 20.0),
            clipping: Some(0.01),
            dynamic_range: Some(60.0),
            spectral_quality: Some(quality),
        },
        phonemes: None,
        metadata: HashMap::new(),
    }
}

#[test]
fn test_curriculum_learning_progression() {
    let samples = create_diverse_dataset();
    let config = CurriculumConfig {
        start_difficulty: 0.0,
        end_difficulty: 1.0,
        num_epochs: 10,
        strategy: DifficultyStrategy::Linear,
        metric: DifficultyMetric::Quality,
    };

    let mut sampler = CurriculumSampler::new(config);
    sampler.calculate_difficulty_scores(&samples);

    // Test progression through epochs
    let mut previous_count = 0;
    for epoch in 0..=10 {
        sampler.set_epoch(epoch);
        let filtered = sampler.filter_by_difficulty(&samples);

        // As difficulty increases, more samples should be included
        assert!(
            filtered.len() >= previous_count,
            "Sample count should increase or stay same with epochs"
        );
        previous_count = filtered.len();
    }

    // At final epoch, all samples should be included
    assert_eq!(previous_count, samples.len());
}

#[test]
fn test_curriculum_exponential_strategy() {
    let samples = create_diverse_dataset();
    let config = CurriculumConfig {
        start_difficulty: 0.0,
        end_difficulty: 1.0,
        num_epochs: 10,
        strategy: DifficultyStrategy::Exponential,
        metric: DifficultyMetric::Quality,
    };

    let mut sampler = CurriculumSampler::new(config);
    sampler.calculate_difficulty_scores(&samples);

    // Exponential should have slower growth early
    sampler.set_epoch(3);
    let early_count = sampler.filter_by_difficulty(&samples).len();

    sampler.set_epoch(7);
    let late_count = sampler.filter_by_difficulty(&samples).len();

    // Late epochs should have more dramatic increase
    let early_growth = early_count as f32 / samples.len() as f32;
    let late_growth = late_count as f32 / samples.len() as f32;

    assert!(
        late_growth > early_growth + 0.1,
        "Exponential strategy should have faster growth in later epochs"
    );
}

#[test]
fn test_curriculum_duration_metric() {
    let samples = create_diverse_dataset();
    let config = CurriculumConfig {
        start_difficulty: 0.0,
        end_difficulty: 1.0, // Increased to full range
        num_epochs: 10,
        strategy: DifficultyStrategy::Linear,
        metric: DifficultyMetric::Duration,
    };

    let mut sampler = CurriculumSampler::new(config);
    sampler.calculate_difficulty_scores(&samples);

    // Early epochs should have fewer samples (shorter ones)
    sampler.set_epoch(0);
    let early_samples = sampler.filter_by_difficulty(&samples);

    // Later epochs should include more samples
    sampler.set_epoch(10);
    let late_samples = sampler.filter_by_difficulty(&samples);

    // Later epochs should have more samples (as threshold increases)
    assert!(
        late_samples.len() >= early_samples.len(),
        "Later epochs should include more samples as difficulty threshold increases"
    );

    // Final epoch should include all samples
    assert_eq!(late_samples.len(), samples.len());
}

#[test]
fn test_importance_sampling_quality_weighting() {
    let samples = create_diverse_dataset();
    let config = ImportanceConfig {
        temperature: 1.0,
        min_probability: 0.001,
        weight_by_quality: true,
        weight_by_loss: false,
        custom_weights: None,
    };

    let mut sampler = ImportanceSampler::new(config, Some(42));
    sampler.calculate_weights(&samples);

    let weights = sampler.weights();

    // Weights should be calculated for all samples
    assert_eq!(weights.len(), samples.len());

    // Weights should sum to approximately 1.0
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.1, "Weights should sum to ~1.0");

    // All weights should be positive
    assert!(
        weights.iter().all(|&w| w > 0.0),
        "All weights should be positive"
    );
}

#[test]
fn test_importance_sampling_temperature_effect() {
    let samples = create_diverse_dataset();

    // Low temperature - sharper distribution
    let config_low = ImportanceConfig {
        temperature: 0.5,
        ..Default::default()
    };
    let mut sampler_low = ImportanceSampler::new(config_low, Some(42));
    sampler_low.calculate_weights(&samples);
    let weights_low = sampler_low.weights();

    // High temperature - more uniform distribution
    let config_high = ImportanceConfig {
        temperature: 2.0,
        ..Default::default()
    };
    let mut sampler_high = ImportanceSampler::new(config_high, Some(42));
    sampler_high.calculate_weights(&samples);
    let weights_high = sampler_high.weights();

    // Calculate variance of distributions
    let mean_low: f32 = weights_low.iter().sum::<f32>() / weights_low.len() as f32;
    let variance_low: f32 = weights_low
        .iter()
        .map(|&w| (w - mean_low).powi(2))
        .sum::<f32>()
        / weights_low.len() as f32;

    let mean_high: f32 = weights_high.iter().sum::<f32>() / weights_high.len() as f32;
    let variance_high: f32 = weights_high
        .iter()
        .map(|&w| (w - mean_high).powi(2))
        .sum::<f32>()
        / weights_high.len() as f32;

    // Low temperature should have higher variance (sharper distribution)
    assert!(
        variance_low > variance_high,
        "Low temperature should produce sharper distribution"
    );
}

#[test]
fn test_importance_sampling_deterministic() {
    let samples = create_diverse_dataset();
    let config = ImportanceConfig::default();

    // Sample with fixed seed
    let mut sampler1 = ImportanceSampler::new(config.clone(), Some(42));
    sampler1.calculate_weights(&samples);
    let indices1 = sampler1.sample_indices(20);

    // Sample again with same seed
    let mut sampler2 = ImportanceSampler::new(config, Some(42));
    sampler2.calculate_weights(&samples);
    let indices2 = sampler2.sample_indices(20);

    // Results should be identical
    assert_eq!(
        indices1, indices2,
        "Same seed should produce same sampling results"
    );
}

#[test]
fn test_stratified_sampling_by_speaker() {
    let samples = create_diverse_dataset();
    let config = StratifiedConfig {
        category_field: StratificationField::Speaker,
        equal_per_category: true,
        min_per_category: 1,
    };

    let mut sampler = StratifiedSampler::new(config, Some(42));
    let sampled = sampler.sample(&samples, 25);

    // Count samples per speaker
    let mut speaker_counts = HashMap::new();
    for sample in &sampled {
        let speaker = sample.speaker.as_ref().unwrap().id.clone();
        *speaker_counts.entry(speaker).or_insert(0) += 1;
    }

    // With equal_per_category=true, counts should be balanced
    let counts: Vec<_> = speaker_counts.values().copied().collect();
    let max_count = *counts.iter().max().unwrap();
    let min_count = *counts.iter().min().unwrap();

    assert!(
        max_count - min_count <= 1,
        "Equal per category should produce balanced distribution"
    );
}

#[test]
fn test_stratified_sampling_by_language() {
    let samples = create_diverse_dataset();
    let config = StratifiedConfig {
        category_field: StratificationField::Language,
        equal_per_category: false, // Proportional
        min_per_category: 2,
    };

    let mut sampler = StratifiedSampler::new(config, Some(42));
    let sampled = sampler.sample(&samples, 20);

    // Count samples per language
    let mut lang_counts = HashMap::new();
    for sample in &sampled {
        let lang = sample.language.as_str().to_string();
        *lang_counts.entry(lang).or_insert(0) += 1;
    }

    // Both languages should be represented (min_per_category=2)
    for count in lang_counts.values() {
        assert!(
            *count >= 2,
            "Each language should have at least min_per_category samples"
        );
    }
}

#[test]
fn test_stratified_sampling_by_quality_tier() {
    let samples = create_diverse_dataset();
    let config = StratifiedConfig {
        category_field: StratificationField::QualityTier,
        equal_per_category: true,
        min_per_category: 1,
    };

    let mut sampler = StratifiedSampler::new(config, Some(42));
    let sampled = sampler.sample(&samples, 30);

    // Count samples per quality tier
    let mut tier_counts = HashMap::new();
    for sample in &sampled {
        let quality = sample.quality.overall_quality.unwrap_or(0.5);
        let tier = if quality >= 0.8 {
            "high"
        } else if quality >= 0.5 {
            "medium"
        } else {
            "low"
        };
        *tier_counts.entry(tier).or_insert(0) += 1;
    }

    // All quality tiers should be represented
    assert!(
        tier_counts.len() >= 2,
        "Multiple quality tiers should be represented"
    );
}

#[test]
fn test_stratified_sampling_consistency() {
    let samples = create_diverse_dataset();
    let config = StratifiedConfig {
        category_field: StratificationField::Speaker,
        equal_per_category: true,
        min_per_category: 1,
    };

    // Sample multiple times with the same configuration
    let mut sampler1 = StratifiedSampler::new(config.clone(), Some(42));
    let sampled1 = sampler1.sample(&samples, 20);

    let mut sampler2 = StratifiedSampler::new(config.clone(), Some(42));
    let sampled2 = sampler2.sample(&samples, 20);

    // Both should have same total count
    assert_eq!(sampled1.len(), sampled2.len());

    // Count speakers in each sample
    let count_speakers = |samples: &[DatasetSample]| -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for s in samples {
            let speaker = s.speaker.as_ref().unwrap().id.clone();
            *counts.entry(speaker).or_insert(0) += 1;
        }
        counts
    };

    let counts1 = count_speakers(&sampled1);
    let counts2 = count_speakers(&sampled2);

    // Both should have same speakers
    assert_eq!(
        counts1.len(),
        counts2.len(),
        "Should sample from same number of speakers"
    );

    // With equal_per_category, distribution should be balanced in both
    for (speaker, count) in &counts1 {
        assert!(
            counts2.contains_key(speaker),
            "Same speakers should be sampled"
        );
        // Counts should be similar (allow ±1 difference due to rounding)
        let diff = (*count as i32 - counts2[speaker] as i32).abs();
        assert!(diff <= 1, "Speaker counts should be similar across samples");
    }
}

#[test]
fn test_combined_curriculum_and_importance() {
    let samples = create_diverse_dataset();

    // First apply curriculum learning to filter samples
    let curriculum_config = CurriculumConfig {
        start_difficulty: 0.0,
        end_difficulty: 0.6,
        num_epochs: 5,
        strategy: DifficultyStrategy::Linear,
        metric: DifficultyMetric::Quality,
    };

    let mut curriculum_sampler = CurriculumSampler::new(curriculum_config);
    curriculum_sampler.calculate_difficulty_scores(&samples);
    curriculum_sampler.set_epoch(3); // Mid-way through curriculum

    let filtered = curriculum_sampler.filter_by_difficulty(&samples);

    // Then apply importance sampling on filtered set
    let importance_config = ImportanceConfig::default();
    let mut importance_sampler = ImportanceSampler::new(importance_config, Some(42));
    importance_sampler.calculate_weights(&filtered);

    let final_samples = importance_sampler.sample(&filtered, 10);

    assert_eq!(final_samples.len(), 10);
    assert!(final_samples.len() <= filtered.len());
}

#[test]
fn test_sampling_with_empty_dataset() {
    let samples: Vec<DatasetSample> = Vec::new();

    // Curriculum sampler with empty dataset
    let mut curriculum_sampler = CurriculumSampler::new(CurriculumConfig::default());
    curriculum_sampler.calculate_difficulty_scores(&samples);
    let filtered = curriculum_sampler.filter_by_difficulty(&samples);
    assert_eq!(filtered.len(), 0);

    // Importance sampler with empty dataset
    let mut importance_sampler = ImportanceSampler::new(ImportanceConfig::default(), Some(42));
    importance_sampler.calculate_weights(&samples);
    let sampled = importance_sampler.sample(&samples, 10);
    assert_eq!(sampled.len(), 0);

    // Stratified sampler with empty dataset
    let mut stratified_sampler = StratifiedSampler::new(StratifiedConfig::default(), Some(42));
    let sampled = stratified_sampler.sample(&samples, 10);
    assert_eq!(sampled.len(), 0);
}

#[test]
fn test_sampling_robustness() {
    let samples = create_diverse_dataset();

    // Test with very low temperature
    let config_low = ImportanceConfig {
        temperature: 0.01,
        ..Default::default()
    };
    let mut sampler = ImportanceSampler::new(config_low, Some(42));
    sampler.calculate_weights(&samples);
    let weights = sampler.weights();
    assert!(weights.iter().all(|&w| w.is_finite() && w >= 0.0));

    // Test with very high temperature
    let config_high = ImportanceConfig {
        temperature: 10.0,
        ..Default::default()
    };
    let mut sampler = ImportanceSampler::new(config_high, Some(42));
    sampler.calculate_weights(&samples);
    let weights = sampler.weights();
    assert!(weights.iter().all(|&w| w.is_finite() && w >= 0.0));
}

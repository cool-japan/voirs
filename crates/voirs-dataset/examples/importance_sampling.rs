//! Importance Sampling Example
//!
//! Demonstrates how to use importance sampling to focus on harder or
//! more informative samples during training. This example shows:
//! - Quality-based importance weighting
//! - Temperature-controlled sampling
//! - Custom weights
//! - Stratified sampling by categories

use std::collections::HashMap;
use voirs_dataset::{
    sampling::{
        ImportanceConfig, ImportanceSampler, StratificationField, StratifiedConfig,
        StratifiedSampler,
    },
    AudioData, DatasetSample, LanguageCode, QualityMetrics, SpeakerInfo,
};

fn create_diverse_dataset() -> Vec<DatasetSample> {
    let mut samples = Vec::new();

    // High quality samples (3)
    for i in 0..3 {
        samples.push(create_sample(
            &format!("high_quality_{}", i),
            0.95 - i as f32 * 0.02,
            "speaker_a",
            LanguageCode::EnUs,
        ));
    }

    // Medium quality samples (4)
    for i in 0..4 {
        samples.push(create_sample(
            &format!("medium_quality_{}", i),
            0.70 - i as f32 * 0.05,
            "speaker_b",
            LanguageCode::EnUs,
        ));
    }

    // Low quality samples (3) - more challenging
    for i in 0..3 {
        samples.push(create_sample(
            &format!("low_quality_{}", i),
            0.45 - i as f32 * 0.05,
            "speaker_c",
            LanguageCode::EnUs,
        ));
    }

    // Add some multilingual samples
    samples.push(create_sample(
        "japanese_001",
        0.85,
        "speaker_d",
        LanguageCode::Ja,
    ));
    samples.push(create_sample(
        "japanese_002",
        0.75,
        "speaker_d",
        LanguageCode::Ja,
    ));

    samples
}

fn create_sample(
    id: &str,
    quality: f32,
    speaker_id: &str,
    language: LanguageCode,
) -> DatasetSample {
    let audio = AudioData::new(vec![0.0; 16000], 16000, 1);

    DatasetSample {
        id: id.to_string(),
        audio,
        text: format!("Sample {}", id),
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

fn main() {
    println!("=== Importance Sampling Example ===\n");

    let samples = create_diverse_dataset();
    println!("Created dataset with {} samples", samples.len());

    // Example 1: Quality-based importance sampling
    println!("\n--- Example 1: Quality-based Importance Sampling ---");
    quality_based_sampling(&samples);

    // Example 2: Temperature-controlled sampling
    println!("\n--- Example 2: Temperature-controlled Sampling ---");
    temperature_sampling(&samples);

    // Example 3: Stratified sampling by speaker
    println!("\n--- Example 3: Stratified Sampling by Speaker ---");
    stratified_sampling(&samples);

    // Example 4: Stratified sampling by language
    println!("\n--- Example 4: Stratified Sampling by Language ---");
    language_stratified_sampling(&samples);
}

fn quality_based_sampling(samples: &[DatasetSample]) {
    let config = ImportanceConfig {
        temperature: 1.0,
        min_probability: 0.01,
        weight_by_quality: true,
        weight_by_loss: false,
        custom_weights: None,
    };

    let mut sampler = ImportanceSampler::new(config, Some(42));
    sampler.calculate_weights(samples);

    println!("Importance weights (higher weight = more important/harder):");
    let weights = sampler.weights();
    for (i, sample) in samples.iter().enumerate() {
        let weight = weights.get(i).unwrap_or(&0.0);
        let quality = sample.quality.overall_quality.unwrap_or(0.0);
        println!(
            "  {}: quality={:.2}, weight={:.4}",
            sample.id, quality, weight
        );
    }

    // Sample 10 examples
    println!("\nSampling 10 examples (lower quality samples should appear more often):");
    let indices = sampler.sample_indices(10);
    let mut sample_counts: HashMap<String, usize> = HashMap::new();

    for idx in indices {
        if let Some(sample) = samples.get(idx) {
            *sample_counts.entry(sample.id.clone()).or_insert(0) += 1;
        }
    }

    let mut sorted: Vec<_> = sample_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    for (id, count) in sorted {
        println!("  {} selected {} times", id, count);
    }
}

fn temperature_sampling(samples: &[DatasetSample]) {
    println!("Comparing different temperature values:\n");

    for &temp in &[0.5, 1.0, 2.0] {
        let config = ImportanceConfig {
            temperature: temp,
            min_probability: 0.01,
            weight_by_quality: true,
            weight_by_loss: false,
            custom_weights: None,
        };

        let mut sampler = ImportanceSampler::new(config, Some(42));
        sampler.calculate_weights(samples);

        // Calculate entropy of distribution (higher = more uniform)
        let weights = sampler.weights();
        let entropy: f32 = weights
            .iter()
            .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
            .sum();

        println!("Temperature = {:.1}:", temp);
        println!("  Weight distribution:");
        let max_weight = weights.iter().copied().fold(0.0f32, f32::max);
        let min_weight = weights.iter().copied().fold(1.0f32, f32::min);
        println!("    Max weight: {:.4}", max_weight);
        println!("    Min weight: {:.4}", min_weight);
        println!("    Entropy: {:.4} (higher = more uniform)", entropy);
        println!();
    }

    println!("Temperature effects:");
    println!("  Low (0.5):  Sharp distribution - focuses on hardest samples");
    println!("  Medium (1.0): Balanced distribution");
    println!("  High (2.0):  Soft distribution - more uniform sampling");
}

fn stratified_sampling(samples: &[DatasetSample]) {
    let config = StratifiedConfig {
        category_field: StratificationField::Speaker,
        equal_per_category: true,
        min_per_category: 1,
    };

    let mut sampler = StratifiedSampler::new(config, Some(42));

    println!("Sampling 9 examples with equal samples per speaker:");
    let sampled = sampler.sample(samples, 9);

    let mut speaker_counts: HashMap<String, usize> = HashMap::new();
    for sample in &sampled {
        let speaker_id = sample
            .speaker
            .as_ref()
            .map(|s| s.id.as_str())
            .unwrap_or("unknown");
        *speaker_counts.entry(speaker_id.to_string()).or_insert(0) += 1;
    }

    println!("  Samples per speaker:");
    for (speaker, count) in speaker_counts {
        println!("    {}: {} samples", speaker, count);
    }
}

fn language_stratified_sampling(samples: &[DatasetSample]) {
    let config = StratifiedConfig {
        category_field: StratificationField::Language,
        equal_per_category: false, // Proportional to dataset
        min_per_category: 1,
    };

    let mut sampler = StratifiedSampler::new(config, Some(42));

    println!("Sampling 10 examples with proportional language representation:");
    let sampled = sampler.sample(samples, 10);

    let mut language_counts: HashMap<String, usize> = HashMap::new();
    for sample in &sampled {
        let lang = sample.language.as_str();
        *language_counts.entry(lang.to_string()).or_insert(0) += 1;
    }

    println!("  Samples per language:");
    for (lang, count) in language_counts {
        // Count in original dataset
        let original_count = samples
            .iter()
            .filter(|s| s.language.as_str() == lang)
            .count();
        println!(
            "    {}: {} samples ({}% of dataset)",
            lang,
            count,
            (original_count * 100) / samples.len()
        );
    }
}

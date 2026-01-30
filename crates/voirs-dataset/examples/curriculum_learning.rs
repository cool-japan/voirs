//! Curriculum Learning Example
//!
//! Demonstrates how to use curriculum learning sampling to progressively
//! train on harder samples. This example shows:
//! - Creating a curriculum learning sampler
//! - Calculating difficulty scores
//! - Progressive difficulty-based training
//! - Different difficulty metrics

use std::collections::HashMap;
use voirs_dataset::{
    sampling::{CurriculumConfig, CurriculumSampler, DifficultyMetric, DifficultyStrategy},
    AudioData, DatasetSample, LanguageCode, QualityMetrics,
};

fn create_sample_dataset() -> Vec<DatasetSample> {
    // Create samples with varying quality (difficulty)
    vec![
        // Easy samples (high quality)
        create_sample("easy_001", 0.95, 2.0, "This is clear speech."),
        create_sample("easy_002", 0.92, 2.5, "Another clear sample."),
        create_sample("easy_003", 0.90, 2.3, "Easy to understand."),
        // Medium samples
        create_sample("medium_001", 0.75, 3.5, "Slightly more challenging audio."),
        create_sample("medium_002", 0.70, 4.0, "Some background noise present."),
        create_sample("medium_003", 0.72, 3.8, "Moderate quality sample."),
        // Hard samples (low quality)
        create_sample("hard_001", 0.50, 5.5, "Difficult sample with low quality."),
        create_sample("hard_002", 0.45, 6.0, "Very challenging audio quality."),
        create_sample("hard_003", 0.48, 5.8, "Hard to process sample."),
        // Very hard samples
        create_sample(
            "veryhard_001",
            0.30,
            8.0,
            "Extremely challenging with significant noise and distortion.",
        ),
    ]
}

fn create_sample(id: &str, quality: f32, duration: f32, text: &str) -> DatasetSample {
    let sample_rate = 16000;
    let num_samples = (sample_rate as f32 * duration) as usize;
    let audio = AudioData::new(vec![0.0; num_samples], sample_rate, 1);

    DatasetSample {
        id: id.to_string(),
        audio,
        text: text.to_string(),
        speaker: None,
        language: LanguageCode::EnUs,
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
    println!("=== Curriculum Learning Example ===\n");

    // Create sample dataset
    let samples = create_sample_dataset();
    println!("Created dataset with {} samples", samples.len());

    // Example 1: Linear curriculum learning
    println!("\n--- Example 1: Linear Curriculum Learning ---");
    linear_curriculum_example(&samples);

    // Example 2: Exponential curriculum learning
    println!("\n--- Example 2: Exponential Curriculum Learning ---");
    exponential_curriculum_example(&samples);

    // Example 3: Duration-based curriculum
    println!("\n--- Example 3: Duration-based Curriculum ---");
    duration_curriculum_example(&samples);

    // Example 4: Step-wise curriculum
    println!("\n--- Example 4: Step-wise Curriculum ---");
    stepwise_curriculum_example(&samples);
}

fn linear_curriculum_example(samples: &[DatasetSample]) {
    let config = CurriculumConfig {
        start_difficulty: 0.0,
        end_difficulty: 1.0,
        num_epochs: 10,
        strategy: DifficultyStrategy::Linear,
        metric: DifficultyMetric::Quality,
    };

    let mut sampler = CurriculumSampler::new(config);
    sampler.calculate_difficulty_scores(samples);

    println!("Training with linear difficulty progression:");
    for epoch in 0..10 {
        let threshold = sampler.current_difficulty_threshold();
        let filtered = sampler.filter_by_difficulty(samples);

        println!(
            "  Epoch {}: Difficulty threshold = {:.2}, Training on {} samples",
            epoch,
            threshold,
            filtered.len()
        );

        // Simulate training
        // ... your training code here ...

        sampler.next_epoch();
    }
}

fn exponential_curriculum_example(samples: &[DatasetSample]) {
    let config = CurriculumConfig {
        start_difficulty: 0.0,
        end_difficulty: 1.0,
        num_epochs: 10,
        strategy: DifficultyStrategy::Exponential,
        metric: DifficultyMetric::Quality,
    };

    let mut sampler = CurriculumSampler::new(config);
    sampler.calculate_difficulty_scores(samples);

    println!("Training with exponential difficulty progression:");
    println!("  (Stays on easy samples longer, then ramps up quickly)");

    for epoch in 0..10 {
        let threshold = sampler.current_difficulty_threshold();
        let filtered = sampler.filter_by_difficulty(samples);

        println!(
            "  Epoch {}: Threshold = {:.3}, Samples = {}",
            epoch,
            threshold,
            filtered.len()
        );

        sampler.next_epoch();
    }
}

fn duration_curriculum_example(samples: &[DatasetSample]) {
    let config = CurriculumConfig {
        start_difficulty: 0.0,
        end_difficulty: 1.0,
        num_epochs: 10,
        strategy: DifficultyStrategy::Linear,
        metric: DifficultyMetric::Duration,
    };

    let mut sampler = CurriculumSampler::new(config);
    sampler.calculate_difficulty_scores(samples);

    println!("Training with duration-based curriculum:");
    println!("  (Start with short samples, progress to longer ones)");

    for epoch in [0, 3, 6, 9] {
        sampler.set_epoch(epoch);
        let threshold = sampler.current_difficulty_threshold();
        let filtered = sampler.filter_by_difficulty(samples);

        // Calculate average duration
        let avg_duration: f32 =
            filtered.iter().map(|s| s.audio.duration()).sum::<f32>() / filtered.len() as f32;

        println!(
            "  Epoch {}: Avg duration = {:.1}s, Samples = {}",
            epoch,
            avg_duration,
            filtered.len()
        );
    }
}

fn stepwise_curriculum_example(samples: &[DatasetSample]) {
    let config = CurriculumConfig {
        start_difficulty: 0.0,
        end_difficulty: 1.0,
        num_epochs: 10,
        strategy: DifficultyStrategy::Stepwise,
        metric: DifficultyMetric::Quality,
    };

    let mut sampler = CurriculumSampler::new(config);
    sampler.calculate_difficulty_scores(samples);

    println!("Training with step-wise curriculum:");
    println!("  (Difficulty increases in discrete steps)");

    for epoch in 0..10 {
        let threshold = sampler.current_difficulty_threshold();
        let filtered = sampler.filter_by_difficulty(samples);

        println!(
            "  Epoch {}: Threshold = {:.2}, Samples = {}",
            epoch,
            threshold,
            filtered.len()
        );

        sampler.next_epoch();
    }
}

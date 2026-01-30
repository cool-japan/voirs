//! Advanced sampling strategies for dataset processing
//!
//! This module provides sophisticated sampling strategies including:
//! - Curriculum learning: Start with easy samples and gradually increase difficulty
//! - Importance sampling: Sample based on quality metrics or loss values
//! - Stratified sampling: Ensure balanced representation across categories
//! - Active learning: Sample most informative examples

use crate::{DatasetSample, QualityMetrics, Result};
use scirs2_core::random::{seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sampling strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Random sampling (uniform distribution)
    Random,
    /// Curriculum learning (easy to hard)
    Curriculum(CurriculumConfig),
    /// Importance sampling based on quality
    Importance(ImportanceConfig),
    /// Stratified sampling across categories
    Stratified(StratifiedConfig),
    /// Active learning based on uncertainty
    ActiveLearning(ActiveLearningConfig),
}

/// Curriculum learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumConfig {
    /// Starting difficulty (0.0 = easiest, 1.0 = hardest)
    pub start_difficulty: f32,
    /// Ending difficulty
    pub end_difficulty: f32,
    /// Number of epochs for curriculum
    pub num_epochs: usize,
    /// Difficulty increase strategy
    pub strategy: DifficultyStrategy,
    /// Difficulty metric to use
    pub metric: DifficultyMetric,
}

/// Difficulty increase strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DifficultyStrategy {
    /// Linear increase
    Linear,
    /// Exponential increase
    Exponential,
    /// Step-wise increase
    Stepwise,
    /// Adaptive based on performance
    Adaptive,
}

/// Difficulty metric for curriculum learning
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DifficultyMetric {
    /// Based on audio quality (lower quality = harder)
    Quality,
    /// Based on duration (longer = harder)
    Duration,
    /// Based on speech rate (faster = harder)
    SpeechRate,
    /// Based on spectral complexity
    SpectralComplexity,
    /// Custom metric (provided externally)
    Custom,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            start_difficulty: 0.0,
            end_difficulty: 1.0,
            num_epochs: 10,
            strategy: DifficultyStrategy::Linear,
            metric: DifficultyMetric::Quality,
        }
    }
}

/// Importance sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceConfig {
    /// Temperature for softmax sampling (higher = more uniform)
    pub temperature: f32,
    /// Minimum sampling probability
    pub min_probability: f32,
    /// Weight by quality metrics
    pub weight_by_quality: bool,
    /// Weight by loss (if available)
    pub weight_by_loss: bool,
    /// Custom weights
    pub custom_weights: Option<Vec<f32>>,
}

impl Default for ImportanceConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            min_probability: 0.01,
            weight_by_quality: true,
            weight_by_loss: false,
            custom_weights: None,
        }
    }
}

/// Stratified sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedConfig {
    /// Category field to stratify by
    pub category_field: StratificationField,
    /// Ensure equal samples per category
    pub equal_per_category: bool,
    /// Minimum samples per category
    pub min_per_category: usize,
}

/// Field to use for stratification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StratificationField {
    /// By speaker
    Speaker,
    /// By language
    Language,
    /// By quality tier
    QualityTier,
    /// By duration bucket
    DurationBucket,
}

impl Default for StratifiedConfig {
    fn default() -> Self {
        Self {
            category_field: StratificationField::Speaker,
            equal_per_category: false,
            min_per_category: 1,
        }
    }
}

/// Active learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningConfig {
    /// Uncertainty metric
    pub uncertainty_metric: UncertaintyMetric,
    /// Top-k most uncertain samples to select
    pub top_k: usize,
    /// Diversity weight (0.0 = only uncertainty, 1.0 = only diversity)
    pub diversity_weight: f32,
}

/// Uncertainty metric for active learning
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UncertaintyMetric {
    /// Maximum entropy
    Entropy,
    /// Least confidence
    LeastConfidence,
    /// Margin sampling
    Margin,
    /// Custom uncertainty scores
    Custom,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            uncertainty_metric: UncertaintyMetric::Entropy,
            top_k: 100,
            diversity_weight: 0.3,
        }
    }
}

/// Curriculum learning sampler
pub struct CurriculumSampler {
    config: CurriculumConfig,
    current_epoch: usize,
    difficulty_scores: HashMap<String, f32>,
}

impl CurriculumSampler {
    /// Create new curriculum sampler
    pub fn new(config: CurriculumConfig) -> Self {
        Self {
            config,
            current_epoch: 0,
            difficulty_scores: HashMap::new(),
        }
    }

    /// Calculate difficulty scores for samples
    pub fn calculate_difficulty_scores(&mut self, samples: &[DatasetSample]) {
        for sample in samples {
            let difficulty = match self.config.metric {
                DifficultyMetric::Quality => self.quality_difficulty(&sample.quality),
                DifficultyMetric::Duration => self.duration_difficulty(sample.audio.duration()),
                DifficultyMetric::SpeechRate => {
                    self.speech_rate_difficulty(&sample.text, sample.audio.duration())
                }
                DifficultyMetric::SpectralComplexity => {
                    self.spectral_complexity_difficulty(sample.audio.samples())
                }
                DifficultyMetric::Custom => {
                    // Use custom difficulty if available in metadata
                    sample
                        .metadata
                        .get("difficulty")
                        .and_then(|v| v.as_f64().map(|f| f as f32))
                        .unwrap_or(0.5)
                }
            };

            self.difficulty_scores.insert(sample.id.clone(), difficulty);
        }
    }

    /// Calculate difficulty based on quality metrics
    fn quality_difficulty(&self, quality: &QualityMetrics) -> f32 {
        // Lower quality = higher difficulty
        let overall_quality = quality.overall_quality.unwrap_or(0.5);
        1.0 - overall_quality
    }

    /// Calculate difficulty based on duration
    fn duration_difficulty(&self, duration: f32) -> f32 {
        // Longer duration = higher difficulty
        // Normalize to [0, 1] assuming max 30 seconds
        (duration / 30.0).min(1.0)
    }

    /// Calculate difficulty based on speech rate
    fn speech_rate_difficulty(&self, text: &str, duration: f32) -> f32 {
        if duration < 0.1 {
            return 0.5;
        }

        // Words per second
        let word_count = text.split_whitespace().count();
        let wps = word_count as f32 / duration;

        // Normalize: typical speech is 2-3 wps, fast is 4+ wps
        ((wps - 2.0) / 3.0).clamp(0.0, 1.0)
    }

    /// Calculate difficulty based on spectral complexity
    fn spectral_complexity_difficulty(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.5;
        }

        // Use variance as proxy for complexity
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        // Normalize to [0, 1]
        (variance / 0.1).min(1.0)
    }

    /// Get current difficulty threshold
    pub fn current_difficulty_threshold(&self) -> f32 {
        let progress = if self.config.num_epochs > 0 {
            (self.current_epoch as f32 / self.config.num_epochs as f32).min(1.0)
        } else {
            1.0
        };

        match self.config.strategy {
            DifficultyStrategy::Linear => {
                self.config.start_difficulty
                    + progress * (self.config.end_difficulty - self.config.start_difficulty)
            }
            DifficultyStrategy::Exponential => {
                let exp_progress = progress.powi(2);
                self.config.start_difficulty
                    + exp_progress * (self.config.end_difficulty - self.config.start_difficulty)
            }
            DifficultyStrategy::Stepwise => {
                let steps = 5;
                let step = (progress * steps as f32).floor() / steps as f32;
                self.config.start_difficulty
                    + step * (self.config.end_difficulty - self.config.start_difficulty)
            }
            DifficultyStrategy::Adaptive => {
                // For adaptive, would need performance feedback
                // For now, use linear
                self.config.start_difficulty
                    + progress * (self.config.end_difficulty - self.config.start_difficulty)
            }
        }
    }

    /// Filter samples by current difficulty
    pub fn filter_by_difficulty(&self, samples: &[DatasetSample]) -> Vec<DatasetSample> {
        let threshold = self.current_difficulty_threshold();

        samples
            .iter()
            .filter(|sample| {
                if let Some(&difficulty) = self.difficulty_scores.get(&sample.id) {
                    difficulty <= threshold
                } else {
                    true // Include if no difficulty score
                }
            })
            .cloned()
            .collect()
    }

    /// Advance to next epoch
    pub fn next_epoch(&mut self) {
        self.current_epoch += 1;
    }

    /// Reset to first epoch
    pub fn reset(&mut self) {
        self.current_epoch = 0;
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Set current epoch (for testing or manual control)
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }
}

/// Importance sampling sampler
pub struct ImportanceSampler {
    config: ImportanceConfig,
    weights: Vec<f32>,
    seed: u64,
}

impl ImportanceSampler {
    /// Create new importance sampler
    pub fn new(config: ImportanceConfig, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| fastrand::u64(..));

        Self {
            config,
            weights: Vec::new(),
            seed,
        }
    }

    /// Calculate importance weights for samples
    pub fn calculate_weights(&mut self, samples: &[DatasetSample]) {
        let mut raw_weights = Vec::new();

        if let Some(custom_weights) = &self.config.custom_weights {
            raw_weights = custom_weights.clone();
        } else if self.config.weight_by_quality {
            for sample in samples {
                // Higher importance for lower quality (harder samples)
                let quality = sample.quality.overall_quality.unwrap_or(0.5);
                let importance = 1.0 - quality;
                raw_weights.push(importance);
            }
        } else {
            // Uniform weights
            raw_weights = vec![1.0; samples.len()];
        }

        // Apply temperature and normalize
        self.weights = self.apply_temperature_and_normalize(&raw_weights);
    }

    /// Apply temperature scaling and normalization
    fn apply_temperature_and_normalize(&self, weights: &[f32]) -> Vec<f32> {
        if weights.is_empty() {
            return Vec::new();
        }

        // Apply temperature
        let scaled: Vec<f32> = weights
            .iter()
            .map(|&w| (w / self.config.temperature).exp())
            .collect();

        // Normalize
        let sum: f32 = scaled.iter().sum();
        let normalized: Vec<f32> = if sum > 1e-8 {
            scaled.iter().map(|&w| w / sum).collect()
        } else {
            vec![1.0 / weights.len() as f32; weights.len()]
        };

        // Apply minimum probability
        normalized
            .iter()
            .map(|&p| p.max(self.config.min_probability))
            .collect()
    }

    /// Sample indices based on importance weights
    pub fn sample_indices(&mut self, num_samples: usize) -> Vec<usize> {
        if self.weights.is_empty() {
            return Vec::new();
        }

        let mut indices = Vec::new();
        let cumulative = self.cumulative_weights();

        // Use fastrand with the seed for deterministic sampling
        fastrand::seed(self.seed);

        for _ in 0..num_samples {
            let random_value = fastrand::f32();
            let idx = cumulative
                .iter()
                .position(|&cum| random_value <= cum)
                .unwrap_or(cumulative.len() - 1);
            indices.push(idx);
        }

        // Update seed for next sampling
        self.seed = fastrand::u64(..);

        indices
    }

    /// Calculate cumulative weights
    fn cumulative_weights(&self) -> Vec<f32> {
        let mut cumulative = Vec::new();
        let mut sum = 0.0;

        for &weight in &self.weights {
            sum += weight;
            cumulative.push(sum);
        }

        // Normalize to ensure last value is 1.0
        if sum > 1e-8 {
            for val in &mut cumulative {
                *val /= sum;
            }
        }

        cumulative
    }

    /// Sample dataset
    pub fn sample(&mut self, samples: &[DatasetSample], num_samples: usize) -> Vec<DatasetSample> {
        let indices = self.sample_indices(num_samples);
        indices
            .into_iter()
            .filter_map(|idx| samples.get(idx).cloned())
            .collect()
    }

    /// Get importance weights
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

/// Stratified sampler
pub struct StratifiedSampler {
    config: StratifiedConfig,
    seed: u64,
}

impl StratifiedSampler {
    /// Create new stratified sampler
    pub fn new(config: StratifiedConfig, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| fastrand::u64(..));

        Self { config, seed }
    }

    /// Group samples by category
    fn group_by_category(&self, samples: &[DatasetSample]) -> HashMap<String, Vec<usize>> {
        let mut groups = HashMap::new();

        for (idx, sample) in samples.iter().enumerate() {
            let category = match self.config.category_field {
                StratificationField::Speaker => sample
                    .speaker
                    .as_ref()
                    .map(|s| s.id.clone())
                    .unwrap_or_else(|| "unknown".to_string()),
                StratificationField::Language => sample.language.as_str().to_string(),
                StratificationField::QualityTier => {
                    let quality = sample.quality.overall_quality.unwrap_or(0.5);
                    if quality >= 0.8 {
                        "high"
                    } else if quality >= 0.5 {
                        "medium"
                    } else {
                        "low"
                    }
                    .to_string()
                }
                StratificationField::DurationBucket => {
                    let duration = sample.audio.duration();
                    if duration < 2.0 {
                        "short"
                    } else if duration < 5.0 {
                        "medium"
                    } else {
                        "long"
                    }
                    .to_string()
                }
            };

            groups.entry(category).or_insert_with(Vec::new).push(idx);
        }

        groups
    }

    /// Sample with stratification
    pub fn sample(&mut self, samples: &[DatasetSample], num_samples: usize) -> Vec<DatasetSample> {
        let groups = self.group_by_category(samples);
        let num_categories = groups.len();

        if num_categories == 0 {
            return Vec::new();
        }

        fastrand::seed(self.seed);
        let mut result = Vec::new();

        if self.config.equal_per_category {
            // Equal samples per category
            let per_category = num_samples / num_categories;

            for indices in groups.values() {
                let mut category_indices = indices.clone();
                fastrand::shuffle(&mut category_indices);

                for &idx in category_indices.iter().take(per_category) {
                    if let Some(sample) = samples.get(idx) {
                        result.push(sample.clone());
                    }
                }
            }
        } else {
            // Proportional sampling
            let total_samples: usize = groups.values().map(|v| v.len()).sum();

            for indices in groups.values() {
                let proportion = indices.len() as f32 / total_samples as f32;
                let category_samples =
                    ((num_samples as f32 * proportion) as usize).max(self.config.min_per_category);

                let mut category_indices = indices.clone();
                fastrand::shuffle(&mut category_indices);

                for &idx in category_indices.iter().take(category_samples) {
                    if let Some(sample) = samples.get(idx) {
                        result.push(sample.clone());
                    }
                }
            }
        }

        // Update seed for next sampling
        self.seed = fastrand::u64(..);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioData;

    fn create_test_samples(count: usize) -> Vec<DatasetSample> {
        (0..count)
            .map(|i| {
                let audio = AudioData::new(vec![0.0; 1000], 16000, 1);
                DatasetSample {
                    id: format!("sample_{}", i),
                    audio,
                    text: format!("Test sample {}", i),
                    speaker: None,
                    language: crate::LanguageCode::EnUs,
                    quality: QualityMetrics {
                        overall_quality: Some(i as f32 / count as f32),
                        ..Default::default()
                    },
                    phonemes: None,
                    metadata: HashMap::new(),
                }
            })
            .collect()
    }

    #[test]
    fn test_curriculum_config_default() {
        let config = CurriculumConfig::default();
        assert_eq!(config.start_difficulty, 0.0);
        assert_eq!(config.end_difficulty, 1.0);
        assert_eq!(config.num_epochs, 10);
    }

    #[test]
    fn test_curriculum_sampler_difficulty_threshold() {
        let config = CurriculumConfig {
            num_epochs: 10,
            strategy: DifficultyStrategy::Linear,
            ..Default::default()
        };
        let mut sampler = CurriculumSampler::new(config);

        // Epoch 0
        assert_eq!(sampler.current_difficulty_threshold(), 0.0);

        // Epoch 5 (halfway)
        sampler.current_epoch = 5;
        assert!((sampler.current_difficulty_threshold() - 0.5).abs() < 0.01);

        // Epoch 10 (end)
        sampler.current_epoch = 10;
        assert_eq!(sampler.current_difficulty_threshold(), 1.0);
    }

    #[test]
    fn test_curriculum_sampler_calculate_scores() {
        let config = CurriculumConfig::default();
        let mut sampler = CurriculumSampler::new(config);
        let samples = create_test_samples(5);

        sampler.calculate_difficulty_scores(&samples);

        assert_eq!(sampler.difficulty_scores.len(), 5);
    }

    #[test]
    fn test_curriculum_sampler_filter() {
        let config = CurriculumConfig {
            num_epochs: 10,
            ..Default::default()
        };
        let mut sampler = CurriculumSampler::new(config);
        let samples = create_test_samples(10);

        sampler.calculate_difficulty_scores(&samples);

        // At epoch 0, should only include easiest samples
        let filtered = sampler.filter_by_difficulty(&samples);
        assert!(filtered.len() < samples.len());

        // At epoch 10, should include all samples
        sampler.current_epoch = 10;
        let filtered = sampler.filter_by_difficulty(&samples);
        assert_eq!(filtered.len(), samples.len());
    }

    #[test]
    fn test_importance_config_default() {
        let config = ImportanceConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.min_probability, 0.01);
        assert!(config.weight_by_quality);
    }

    #[test]
    fn test_importance_sampler_calculate_weights() {
        let config = ImportanceConfig::default();
        let mut sampler = ImportanceSampler::new(config, Some(42));
        let samples = create_test_samples(5);

        sampler.calculate_weights(&samples);

        assert_eq!(sampler.weights.len(), 5);
        // Weights should sum to ~1.0 (allowing for min_probability adjustments)
        let sum: f32 = sampler.weights.iter().sum();
        assert!(sum > 0.9 && sum < 1.5);
    }

    #[test]
    fn test_importance_sampler_sample_indices() {
        let config = ImportanceConfig::default();
        let mut sampler = ImportanceSampler::new(config, Some(42));
        sampler.weights = vec![0.2, 0.3, 0.5];

        let indices = sampler.sample_indices(10);

        assert_eq!(indices.len(), 10);
        // All indices should be valid
        assert!(indices.iter().all(|&i| i < 3));
    }

    #[test]
    fn test_importance_sampler_sample() {
        let config = ImportanceConfig::default();
        let mut sampler = ImportanceSampler::new(config, Some(42));
        let samples = create_test_samples(10);

        sampler.calculate_weights(&samples);
        let sampled = sampler.sample(&samples, 5);

        assert_eq!(sampled.len(), 5);
    }

    #[test]
    fn test_stratified_config_default() {
        let config = StratifiedConfig::default();
        assert!(!config.equal_per_category);
        assert_eq!(config.min_per_category, 1);
    }

    #[test]
    fn test_stratified_sampler_sample() {
        let config = StratifiedConfig::default();
        let mut sampler = StratifiedSampler::new(config, Some(42));
        let samples = create_test_samples(10);

        let sampled = sampler.sample(&samples, 5);

        assert!(!sampled.is_empty());
        assert!(sampled.len() <= 10);
    }

    #[test]
    fn test_stratified_sampler_equal_per_category() {
        let config = StratifiedConfig {
            equal_per_category: true,
            ..Default::default()
        };
        let mut sampler = StratifiedSampler::new(config, Some(42));

        // Create samples with different quality tiers
        let mut samples = Vec::new();
        for i in 0..9 {
            let audio = AudioData::new(vec![0.0; 1000], 16000, 1);
            let quality_tier = i / 3; // 0-2, 3-5, 6-8
            samples.push(DatasetSample {
                id: format!("sample_{}", i),
                audio,
                text: "test".to_string(),
                speaker: None,
                language: crate::LanguageCode::EnUs,
                quality: QualityMetrics {
                    overall_quality: Some(quality_tier as f32 * 0.4),
                    ..Default::default()
                },
                phonemes: None,
                metadata: HashMap::new(),
            });
        }

        let sampled = sampler.sample(&samples, 6);

        // Should have roughly equal samples from each quality tier
        assert!(!sampled.is_empty());
    }
}

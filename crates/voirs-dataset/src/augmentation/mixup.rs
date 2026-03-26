//! MixUp and Advanced Mixing Augmentation for Audio
//!
//! Implementation of MixUp and related mixing augmentation techniques for speech and audio.
//! These techniques improve model robustness and generalization by creating synthetic training
//! samples through interpolation.
//!
//! References:
//! - MixUp: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
//! - CutMix: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)

use crate::{AudioData, DatasetSample, LanguageCode, QualityMetrics, Result, SpeakerInfo};
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Mixing strategy for audio augmentation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixingStrategy {
    /// Standard MixUp: linear interpolation of entire samples
    MixUp,
    /// CutMix: replace random segments from one sample with another
    CutMix,
    /// TimeMix: mix along time axis with variable ratios
    TimeMix,
    /// FrequencyMix: mix in frequency domain (for spectrograms)
    FrequencyMix,
    /// AdaptiveMix: adaptive mixing based on sample characteristics
    AdaptiveMix,
}

/// MixUp configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixUpConfig {
    /// Mixing strategy to use
    pub strategy: MixingStrategy,
    /// Alpha parameter for Beta distribution (controls mixing ratio)
    /// Higher alpha = more balanced mixing, lower = more extreme mixing
    pub alpha: f32,
    /// Minimum mixing ratio (lambda_min)
    pub lambda_min: f32,
    /// Maximum mixing ratio (lambda_max)
    pub lambda_max: f32,
    /// Enable label mixing (for supervised learning)
    pub mix_labels: bool,
    /// Enable metadata mixing
    pub mix_metadata: bool,
    /// Preserve speaker identity (don't mix different speakers)
    pub preserve_speaker: bool,
    /// Preserve language (don't mix different languages)
    pub preserve_language: bool,
}

impl Default for MixUpConfig {
    fn default() -> Self {
        Self {
            strategy: MixingStrategy::MixUp,
            alpha: 0.4,
            lambda_min: 0.2,
            lambda_max: 0.8,
            mix_labels: true,
            mix_metadata: false,
            preserve_speaker: true,
            preserve_language: true,
        }
    }
}

impl MixUpConfig {
    /// Create configuration for balanced mixing
    pub fn balanced() -> Self {
        Self {
            alpha: 1.0, // Beta(1,1) = uniform distribution
            lambda_min: 0.3,
            lambda_max: 0.7,
            ..Default::default()
        }
    }

    /// Create configuration for aggressive mixing
    pub fn aggressive() -> Self {
        Self {
            alpha: 0.2, // More extreme mixing ratios
            lambda_min: 0.1,
            lambda_max: 0.9,
            preserve_speaker: false,
            preserve_language: false,
            ..Default::default()
        }
    }

    /// Create configuration for conservative mixing
    pub fn conservative() -> Self {
        Self {
            alpha: 2.0, // Centered around 0.5
            lambda_min: 0.4,
            lambda_max: 0.6,
            preserve_speaker: true,
            preserve_language: true,
            ..Default::default()
        }
    }

    /// Create configuration for CutMix
    pub fn cutmix() -> Self {
        Self {
            strategy: MixingStrategy::CutMix,
            alpha: 1.0,
            lambda_min: 0.2,
            lambda_max: 0.8,
            mix_labels: true,
            mix_metadata: false,
            preserve_speaker: false,
            preserve_language: true,
        }
    }
}

/// MixUp augmenter
pub struct MixUpAugmentor {
    config: MixUpConfig,
}

impl MixUpAugmentor {
    /// Create new MixUp augmenter
    pub fn new(config: MixUpConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(MixUpConfig::default())
    }

    /// Mix two audio samples using configured strategy
    pub fn mix_samples(
        &self,
        sample1: &DatasetSample,
        sample2: &DatasetSample,
    ) -> Result<DatasetSample> {
        // Check compatibility
        if self.config.preserve_speaker && sample1.speaker != sample2.speaker {
            return Ok(sample1.clone()); // Skip mixing incompatible samples
        }

        if self.config.preserve_language && sample1.language != sample2.language {
            return Ok(sample1.clone()); // Skip mixing different languages
        }

        // Sample mixing ratio
        let lambda = self.sample_lambda();

        // Mix based on strategy
        match self.config.strategy {
            MixingStrategy::MixUp => self.apply_mixup(sample1, sample2, lambda),
            MixingStrategy::CutMix => self.apply_cutmix(sample1, sample2, lambda),
            MixingStrategy::TimeMix => self.apply_timemix(sample1, sample2, lambda),
            MixingStrategy::FrequencyMix => self.apply_mixup(sample1, sample2, lambda), // Fallback to MixUp
            MixingStrategy::AdaptiveMix => self.apply_adaptive_mix(sample1, sample2),
        }
    }

    /// Sample mixing ratio from Beta distribution
    fn sample_lambda(&self) -> f32 {
        let mut rng = thread_rng();

        // Use Beta distribution to sample lambda
        // For simplicity, we approximate Beta distribution
        let lambda = if self.config.alpha == 1.0 {
            // Beta(1,1) = Uniform(0,1)
            rng.random_range(0.0..1.0)
        } else {
            // Approximate Beta distribution using two gamma samples
            // For alpha == beta, we can use a simpler approximation
            let u1: f32 = rng.random();
            let u2: f32 = rng.random();
            let x = u1.powf(1.0 / self.config.alpha);
            let y = u2.powf(1.0 / self.config.alpha);
            x / (x + y)
        };

        // Clamp to configured range
        lambda.clamp(self.config.lambda_min, self.config.lambda_max)
    }

    /// Apply standard MixUp: linear interpolation
    fn apply_mixup(
        &self,
        sample1: &DatasetSample,
        sample2: &DatasetSample,
        lambda: f32,
    ) -> Result<DatasetSample> {
        // Mix audio
        let mixed_audio = self.mix_audio(&sample1.audio, &sample2.audio, lambda)?;

        // Mix text if enabled
        let mixed_text = if self.config.mix_labels {
            format!("{} [MIX] {}", sample1.text, sample2.text)
        } else {
            sample1.text.clone()
        };

        // Create mixed sample
        let mixed_sample = DatasetSample {
            id: format!("{}_mixup_{}", sample1.id, sample2.id),
            audio: mixed_audio,
            text: mixed_text,
            speaker: sample1.speaker.clone(),
            language: sample1.language,
            quality: self.mix_quality_metrics(&sample1.quality, &sample2.quality, lambda),
            phonemes: sample1.phonemes.clone(),
            metadata: if self.config.mix_metadata {
                self.mix_metadata(&sample1.metadata, &sample2.metadata)
            } else {
                sample1.metadata.clone()
            },
        };

        Ok(mixed_sample)
    }

    /// Apply CutMix: replace segments
    fn apply_cutmix(
        &self,
        sample1: &DatasetSample,
        sample2: &DatasetSample,
        lambda: f32,
    ) -> Result<DatasetSample> {
        let mut rng = thread_rng();

        // Determine cut region
        let audio1 = sample1.audio.samples();
        let audio2 = sample2.audio.samples();
        let len = audio1.len().min(audio2.len());

        let cut_size = ((1.0 - lambda) * len as f32) as usize;
        let cut_start = if len > cut_size {
            rng.random_range(0..=(len - cut_size))
        } else {
            0
        };
        let cut_end = (cut_start + cut_size).min(len);

        // Create mixed audio
        let mut mixed_samples = Vec::with_capacity(len);
        for i in 0..len {
            if i >= cut_start && i < cut_end {
                mixed_samples.push(if i < audio2.len() { audio2[i] } else { 0.0 });
            } else {
                mixed_samples.push(if i < audio1.len() { audio1[i] } else { 0.0 });
            }
        }

        let mixed_audio = AudioData::new(
            mixed_samples,
            sample1.audio.sample_rate(),
            sample1.audio.channels(),
        );

        // Create mixed sample
        Ok(DatasetSample {
            id: format!("{}_cutmix_{}", sample1.id, sample2.id),
            audio: mixed_audio,
            text: sample1.text.clone(),
            speaker: sample1.speaker.clone(),
            language: sample1.language,
            quality: sample1.quality.clone(),
            phonemes: sample1.phonemes.clone(),
            metadata: sample1.metadata.clone(),
        })
    }

    /// Apply TimeMix: mix with time-varying ratio
    fn apply_timemix(
        &self,
        sample1: &DatasetSample,
        sample2: &DatasetSample,
        lambda: f32,
    ) -> Result<DatasetSample> {
        let mut rng = thread_rng();

        let audio1 = sample1.audio.samples();
        let audio2 = sample2.audio.samples();
        let len = audio1.len().min(audio2.len());

        // Create time-varying mixing ratio
        let num_segments = 10;
        let segment_size = len / num_segments;

        let mut mixed_samples = Vec::with_capacity(len);
        for i in 0..len {
            let segment = i / segment_size.max(1);
            // Vary lambda slightly per segment
            let segment_lambda = (lambda + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0);

            let s1 = if i < audio1.len() { audio1[i] } else { 0.0 };
            let s2 = if i < audio2.len() { audio2[i] } else { 0.0 };
            mixed_samples.push(segment_lambda * s1 + (1.0 - segment_lambda) * s2);
        }

        let mixed_audio = AudioData::new(
            mixed_samples,
            sample1.audio.sample_rate(),
            sample1.audio.channels(),
        );

        Ok(DatasetSample {
            id: format!("{}_timemix_{}", sample1.id, sample2.id),
            audio: mixed_audio,
            text: format!("{} [TIMEMIX] {}", sample1.text, sample2.text),
            speaker: sample1.speaker.clone(),
            language: sample1.language,
            quality: self.mix_quality_metrics(&sample1.quality, &sample2.quality, lambda),
            phonemes: sample1.phonemes.clone(),
            metadata: sample1.metadata.clone(),
        })
    }

    /// Apply adaptive mixing based on sample characteristics
    fn apply_adaptive_mix(
        &self,
        sample1: &DatasetSample,
        sample2: &DatasetSample,
    ) -> Result<DatasetSample> {
        // Compute adaptive lambda based on sample characteristics
        let rms1 = sample1.audio.rms().unwrap_or(0.5);
        let rms2 = sample2.audio.rms().unwrap_or(0.5);

        // Weight mixing toward quieter sample to prevent overpowering
        let lambda = rms2 / (rms1 + rms2 + 1e-8);
        let lambda = lambda.clamp(self.config.lambda_min, self.config.lambda_max);

        self.apply_mixup(sample1, sample2, lambda)
    }

    /// Mix two audio samples with given ratio
    fn mix_audio(&self, audio1: &AudioData, audio2: &AudioData, lambda: f32) -> Result<AudioData> {
        let samples1 = audio1.samples();
        let samples2 = audio2.samples();

        // Handle different lengths by padding/truncating
        let max_len = samples1.len().max(samples2.len());
        let mut mixed = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let s1 = if i < samples1.len() { samples1[i] } else { 0.0 };
            let s2 = if i < samples2.len() { samples2[i] } else { 0.0 };
            mixed.push(lambda * s1 + (1.0 - lambda) * s2);
        }

        Ok(AudioData::new(
            mixed,
            audio1.sample_rate(),
            audio1.channels(),
        ))
    }

    /// Mix quality metrics
    fn mix_quality_metrics(
        &self,
        q1: &QualityMetrics,
        q2: &QualityMetrics,
        lambda: f32,
    ) -> QualityMetrics {
        QualityMetrics {
            snr: match (q1.snr, q2.snr) {
                (Some(a), Some(b)) => Some(lambda * a + (1.0 - lambda) * b),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
            clipping: match (q1.clipping, q2.clipping) {
                (Some(a), Some(b)) => Some(lambda * a + (1.0 - lambda) * b),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
            dynamic_range: match (q1.dynamic_range, q2.dynamic_range) {
                (Some(a), Some(b)) => Some(lambda * a + (1.0 - lambda) * b),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
            spectral_quality: match (q1.spectral_quality, q2.spectral_quality) {
                (Some(a), Some(b)) => Some(lambda * a + (1.0 - lambda) * b),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
            overall_quality: match (q1.overall_quality, q2.overall_quality) {
                (Some(a), Some(b)) => Some(lambda * a + (1.0 - lambda) * b),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
        }
    }

    /// Mix metadata
    fn mix_metadata(
        &self,
        m1: &HashMap<String, Value>,
        m2: &HashMap<String, Value>,
    ) -> HashMap<String, Value> {
        let mut mixed = m1.clone();
        for (k, v) in m2 {
            mixed
                .entry(k.clone())
                .or_insert_with(|| Value::String(format!("{:?}_mixed", v)));
        }
        mixed
    }
}

/// Batch MixUp augmentation
pub struct BatchMixUpAugmentor {
    augmenter: MixUpAugmentor,
}

impl BatchMixUpAugmentor {
    /// Create new batch augmenter
    pub fn new(config: MixUpConfig) -> Self {
        Self {
            augmenter: MixUpAugmentor::new(config),
        }
    }

    /// Mix samples in batch (each sample mixed with a random other sample)
    pub fn mix_batch(&self, samples: &[DatasetSample]) -> Result<Vec<DatasetSample>> {
        let mut rng = thread_rng();
        let mut mixed_samples = Vec::new();

        for sample in samples {
            // Choose a random other sample to mix with
            let other_idx = rng.random_range(0..samples.len());
            let other_sample = &samples[other_idx];

            let mixed = self.augmenter.mix_samples(sample, other_sample)?;
            mixed_samples.push(mixed);
        }

        Ok(mixed_samples)
    }

    /// Mix samples with explicit pairs
    pub fn mix_pairs(
        &self,
        pairs: &[(DatasetSample, DatasetSample)],
    ) -> Result<Vec<DatasetSample>> {
        pairs
            .iter()
            .map(|(s1, s2)| self.augmenter.mix_samples(s1, s2))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sample(id: &str, frequency: f32) -> DatasetSample {
        let sample_rate = 16000;
        let duration_secs = 0.1;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;

        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let audio = AudioData::new(samples, sample_rate, 1);
        DatasetSample {
            id: id.to_string(),
            audio,
            text: format!("Test sample {}", id),
            speaker: Some(SpeakerInfo {
                id: "speaker1".to_string(),
                name: Some("Speaker 1".to_string()),
                gender: None,
                age: None,
                accent: None,
                metadata: HashMap::new(),
            }),
            language: LanguageCode::EnUs,
            quality: QualityMetrics {
                snr: Some(30.0),
                clipping: Some(0.01),
                dynamic_range: Some(60.0),
                spectral_quality: Some(0.95),
                overall_quality: Some(0.9),
            },
            phonemes: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_mixup_config_default() {
        let config = MixUpConfig::default();
        assert_eq!(config.strategy, MixingStrategy::MixUp);
        assert_eq!(config.alpha, 0.4);
        assert!(config.preserve_speaker);
    }

    #[test]
    fn test_mixup_config_balanced() {
        let config = MixUpConfig::balanced();
        assert_eq!(config.alpha, 1.0);
        assert_eq!(config.lambda_min, 0.3);
        assert_eq!(config.lambda_max, 0.7);
    }

    #[test]
    fn test_mixup_config_aggressive() {
        let config = MixUpConfig::aggressive();
        assert_eq!(config.alpha, 0.2);
        assert!(!config.preserve_speaker);
    }

    #[test]
    fn test_mixup_config_conservative() {
        let config = MixUpConfig::conservative();
        assert_eq!(config.alpha, 2.0);
        assert!(config.preserve_speaker);
    }

    #[test]
    fn test_mixup_config_cutmix() {
        let config = MixUpConfig::cutmix();
        assert_eq!(config.strategy, MixingStrategy::CutMix);
    }

    #[test]
    fn test_mixup_augmentor_creation() {
        let augmenter = MixUpAugmentor::default_config();
        assert_eq!(augmenter.config.strategy, MixingStrategy::MixUp);
    }

    #[test]
    fn test_mix_samples_basic() {
        let config = MixUpConfig::default();
        let augmenter = MixUpAugmentor::new(config);

        let sample1 = create_test_sample("sample1", 440.0);
        let sample2 = create_test_sample("sample2", 880.0);

        let mixed = augmenter.mix_samples(&sample1, &sample2).unwrap();

        assert!(mixed.id.contains("mixup"));
        assert_eq!(mixed.audio.sample_rate(), sample1.audio.sample_rate());
        assert!(!mixed.audio.samples().is_empty());
    }

    #[test]
    fn test_mix_samples_preserve_speaker() {
        let config = MixUpConfig {
            preserve_speaker: true,
            ..Default::default()
        };
        let augmenter = MixUpAugmentor::new(config);

        let sample1 = create_test_sample("sample1", 440.0);
        let mut sample2 = create_test_sample("sample2", 880.0);
        sample2.speaker = Some(SpeakerInfo {
            id: "speaker2".to_string(),
            name: Some("Speaker 2".to_string()),
            gender: None,
            age: None,
            accent: None,
            metadata: HashMap::new(),
        });

        let mixed = augmenter.mix_samples(&sample1, &sample2).unwrap();

        // Should return original when speakers don't match
        assert_eq!(mixed.id, sample1.id);
    }

    #[test]
    fn test_mix_samples_preserve_language() {
        let config = MixUpConfig {
            preserve_language: true,
            ..Default::default()
        };
        let augmenter = MixUpAugmentor::new(config);

        let mut sample1 = create_test_sample("sample1", 440.0);
        let mut sample2 = create_test_sample("sample2", 880.0);
        sample2.language = LanguageCode::Ja;

        let mixed = augmenter.mix_samples(&sample1, &sample2).unwrap();

        // Should return original when languages don't match
        assert_eq!(mixed.id, sample1.id);
    }

    #[test]
    fn test_cutmix_strategy() {
        let config = MixUpConfig::cutmix();
        let augmenter = MixUpAugmentor::new(config);

        let sample1 = create_test_sample("sample1", 440.0);
        let sample2 = create_test_sample("sample2", 880.0);

        let mixed = augmenter.mix_samples(&sample1, &sample2).unwrap();

        assert!(mixed.id.contains("cutmix"));
        assert_eq!(mixed.audio.samples().len(), sample1.audio.samples().len());
    }

    #[test]
    fn test_timemix_strategy() {
        let config = MixUpConfig {
            strategy: MixingStrategy::TimeMix,
            ..Default::default()
        };
        let augmenter = MixUpAugmentor::new(config);

        let sample1 = create_test_sample("sample1", 440.0);
        let sample2 = create_test_sample("sample2", 880.0);

        let mixed = augmenter.mix_samples(&sample1, &sample2).unwrap();

        assert!(mixed.id.contains("timemix"));
        assert!(!mixed.audio.samples().is_empty());
    }

    #[test]
    fn test_adaptive_mix_strategy() {
        let config = MixUpConfig {
            strategy: MixingStrategy::AdaptiveMix,
            ..Default::default()
        };
        let augmenter = MixUpAugmentor::new(config);

        let sample1 = create_test_sample("sample1", 440.0);
        let sample2 = create_test_sample("sample2", 880.0);

        let mixed = augmenter.mix_samples(&sample1, &sample2).unwrap();

        assert!(mixed.id.contains("mixup"));
        assert!(!mixed.audio.samples().is_empty());
    }

    #[test]
    fn test_lambda_sampling() {
        let config = MixUpConfig::balanced();
        let lambda_min = config.lambda_min;
        let lambda_max = config.lambda_max;
        let augmenter = MixUpAugmentor::new(config);

        // Sample lambda multiple times
        for _ in 0..100 {
            let lambda = augmenter.sample_lambda();
            assert!(lambda >= lambda_min);
            assert!(lambda <= lambda_max);
        }
    }

    #[test]
    fn test_quality_metrics_mixing() {
        let config = MixUpConfig::default();
        let augmenter = MixUpAugmentor::new(config);

        let q1 = QualityMetrics {
            snr: Some(30.0),
            clipping: Some(0.0),
            dynamic_range: Some(60.0),
            spectral_quality: Some(0.9),
            overall_quality: Some(0.9),
        };

        let q2 = QualityMetrics {
            snr: Some(20.0),
            clipping: Some(0.1),
            dynamic_range: Some(50.0),
            spectral_quality: Some(0.8),
            overall_quality: Some(0.8),
        };

        let mixed = augmenter.mix_quality_metrics(&q1, &q2, 0.5);

        assert!(mixed.snr.is_some());
        assert!(mixed.snr.unwrap() > 20.0 && mixed.snr.unwrap() < 30.0);
    }

    #[test]
    fn test_batch_mixup() {
        let config = MixUpConfig::default();
        let batch_augmenter = BatchMixUpAugmentor::new(config);

        let samples = vec![
            create_test_sample("sample1", 440.0),
            create_test_sample("sample2", 880.0),
            create_test_sample("sample3", 1320.0),
        ];

        let mixed_batch = batch_augmenter.mix_batch(&samples).unwrap();

        assert_eq!(mixed_batch.len(), 3);
        for mixed in &mixed_batch {
            assert!(!mixed.audio.samples().is_empty());
        }
    }

    #[test]
    fn test_mix_pairs() {
        let config = MixUpConfig::default();
        let batch_augmenter = BatchMixUpAugmentor::new(config);

        let pairs = vec![
            (
                create_test_sample("sample1", 440.0),
                create_test_sample("sample2", 880.0),
            ),
            (
                create_test_sample("sample3", 1320.0),
                create_test_sample("sample4", 1760.0),
            ),
        ];

        let mixed = batch_augmenter.mix_pairs(&pairs).unwrap();

        assert_eq!(mixed.len(), 2);
        for m in &mixed {
            assert!(!m.audio.samples().is_empty());
        }
    }

    #[test]
    fn test_mix_different_lengths() {
        let config = MixUpConfig::default();
        let augmenter = MixUpAugmentor::new(config);

        let mut sample1 = create_test_sample("sample1", 440.0);
        let sample2 = create_test_sample("sample2", 880.0);

        // Make sample1 shorter
        let short_samples: Vec<f32> = sample1.audio.samples().iter().take(500).copied().collect();
        sample1.audio = AudioData::new(short_samples, 16000, 1);

        let mixed = augmenter.mix_samples(&sample1, &sample2).unwrap();

        // Should handle different lengths gracefully
        assert!(!mixed.audio.samples().is_empty());
    }

    #[test]
    fn test_mix_labels() {
        let config = MixUpConfig {
            mix_labels: true,
            ..Default::default()
        };
        let augmenter = MixUpAugmentor::new(config);

        let sample1 = create_test_sample("sample1", 440.0);
        let sample2 = create_test_sample("sample2", 880.0);

        let mixed = augmenter.mix_samples(&sample1, &sample2).unwrap();

        assert!(mixed.text.contains("sample1"));
        assert!(mixed.text.contains("sample2"));
        assert!(mixed.text.contains("[MIX]"));
    }

    #[test]
    fn test_no_mix_labels() {
        let config = MixUpConfig {
            mix_labels: false,
            ..Default::default()
        };
        let augmenter = MixUpAugmentor::new(config);

        let sample1 = create_test_sample("sample1", 440.0);
        let sample2 = create_test_sample("sample2", 880.0);

        let mixed = augmenter.mix_samples(&sample1, &sample2).unwrap();

        assert_eq!(mixed.text, sample1.text);
    }
}

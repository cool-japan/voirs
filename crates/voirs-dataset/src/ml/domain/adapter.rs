//! Domain adapter implementation
//!
//! This module contains the implementation of domain adaptation functionality
//! including shift detection, feature alignment, transfer learning, and data mixing.

use crate::{DatasetError, DatasetSample, Result};
use std::collections::HashMap;

use super::config::{
    AdaptationStrategy, DataMixingConfig, DistanceMetric, DomainAdaptationConfig, DomainConfig,
    ShiftDetectionConfig, ShiftDetectionMethod, StatisticalTest, TransferLearningConfig,
};
use super::types::{
    AdaptationRecommendation, AdaptationResult, AudioStatistics, CompatibilityReport,
    DomainAdapter, DomainShift, DomainStatistics, Priority, RecommendationType, SpeakerStatistics,
    TextStatistics, VocabularyStatistics,
};

/// Domain adapter implementation
pub struct DomainAdapterImpl {
    config: DomainAdaptationConfig,
    shift_detector: ShiftDetector,
    feature_aligner: FeatureAligner,
    transfer_learner: TransferLearner,
    data_mixer: DataMixer,
}

/// Domain shift detector
struct ShiftDetector {
    config: ShiftDetectionConfig,
}

/// Feature aligner
struct FeatureAligner {
    strategy: AdaptationStrategy,
}

/// Transfer learner
struct TransferLearner {
    config: TransferLearningConfig,
}

/// Data mixer
struct DataMixer {
    config: DataMixingConfig,
}

impl DomainAdapterImpl {
    /// Create a new domain adapter
    pub fn new(config: DomainAdaptationConfig) -> Result<Self> {
        Ok(Self {
            shift_detector: ShiftDetector::new(&config.shift_detection)?,
            feature_aligner: FeatureAligner::new(&config.adaptation_strategy)?,
            transfer_learner: TransferLearner::new(&config.transfer_learning)?,
            data_mixer: DataMixer::new(&config.data_mixing)?,
            config,
        })
    }

    /// Validate configuration
    pub fn validate_config(&self) -> Result<()> {
        if self.config.data_mixing.source_weight + self.config.data_mixing.target_weight > 1.0 {
            return Err(DatasetError::Configuration(
                "Source and target weights cannot exceed 1.0".to_string(),
            ));
        }

        if self.config.shift_detection.significance_threshold <= 0.0
            || self.config.shift_detection.significance_threshold >= 1.0
        {
            return Err(DatasetError::Configuration(
                "Shift detection threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Create a default adapter instance
    pub fn new_default() -> Result<Self> {
        Self::new(DomainAdaptationConfig::default())
    }
}

impl ShiftDetector {
    fn new(config: &ShiftDetectionConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    async fn detect_shift(
        &self,
        source_samples: &[DatasetSample],
        target_samples: &[DatasetSample],
    ) -> Result<DomainShift> {
        // Extract features for comparison
        let source_features = self.extract_features(source_samples).await?;
        let target_features = self.extract_features(target_samples).await?;

        // Calculate shift magnitude based on method
        let magnitude = match &self.config.method {
            ShiftDetectionMethod::Statistical { test, alpha } => {
                self.statistical_test(&source_features, &target_features, test, *alpha)
                    .await?
            }
            ShiftDetectionMethod::Distance { metric, .. } => {
                self.distance_based_detection(&source_features, &target_features, metric)
                    .await?
            }
            ShiftDetectionMethod::Density { .. } => {
                self.density_based_detection(&source_features, &target_features)
                    .await?
            }
            ShiftDetectionMethod::ModelBased { .. } => {
                self.model_based_detection(&source_features, &target_features)
                    .await?
            }
            _ => {
                // Fallback for other methods
                self.basic_statistical_detection(&source_features, &target_features)
                    .await?
            }
        };

        let mut metadata = HashMap::new();
        metadata.insert(
            "detection_time".to_string(),
            chrono::Utc::now().to_rfc3339(),
        );
        metadata.insert("method".to_string(), format!("{:?}", self.config.method));

        Ok(DomainShift {
            magnitude,
            direction: vec![0.0; 10], // Placeholder for feature direction vector
            affected_features: vec!["audio".to_string(), "text".to_string()],
            confidence: 0.9,
            timestamp: chrono::Utc::now(),
            metadata,
        })
    }

    async fn extract_features(&self, samples: &[DatasetSample]) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        for sample in samples {
            // Extract audio features
            features.push(sample.audio.duration());
            features.push(sample.audio.samples().len() as f32);
            if let Some(rms) = sample.audio.rms() {
                features.push(rms);
            }

            // Extract text features
            features.push(sample.text.len() as f32);
            features.push(sample.text.split_whitespace().count() as f32);

            // Extract quality features
            if let Some(snr) = sample.quality.snr {
                features.push(snr);
            }
            if let Some(overall) = sample.quality.overall_quality {
                features.push(overall);
            }
        }

        Ok(features)
    }

    async fn statistical_test(
        &self,
        source_features: &[f32],
        target_features: &[f32],
        test_type: &StatisticalTest,
        _alpha: f64,
    ) -> Result<f32> {
        match test_type {
            StatisticalTest::KolmogorovSmirnov => {
                // Simplified KS test implementation
                let source_mean: f32 =
                    source_features.iter().sum::<f32>() / source_features.len() as f32;
                let target_mean: f32 =
                    target_features.iter().sum::<f32>() / target_features.len() as f32;
                Ok((source_mean - target_mean).abs() / source_mean.max(target_mean))
            }
            StatisticalTest::MannWhitneyU => {
                // Simplified Mann-Whitney U test
                self.mann_whitney_u_test(source_features, target_features)
                    .await
            }
            StatisticalTest::ChiSquare => {
                // Simplified Chi-square test
                self.chi_square_test(source_features, target_features).await
            }
            StatisticalTest::AndersonDarling => {
                // Simplified Anderson-Darling test
                self.anderson_darling_test(source_features, target_features)
                    .await
            }
        }
    }

    async fn mann_whitney_u_test(&self, source: &[f32], target: &[f32]) -> Result<f32> {
        // Simplified implementation
        let source_median = self.calculate_median(source);
        let target_median = self.calculate_median(target);
        Ok((source_median - target_median).abs() / source_median.max(target_median))
    }

    async fn chi_square_test(&self, source: &[f32], target: &[f32]) -> Result<f32> {
        // Simplified implementation
        let source_var = self.calculate_variance(source);
        let target_var = self.calculate_variance(target);
        Ok((source_var - target_var).abs() / source_var.max(target_var))
    }

    async fn anderson_darling_test(&self, source: &[f32], target: &[f32]) -> Result<f32> {
        // Simplified implementation
        let source_skew = self.calculate_skewness(source);
        let target_skew = self.calculate_skewness(target);
        Ok((source_skew - target_skew).abs())
    }

    fn calculate_median(&self, data: &[f32]) -> f32 {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }

    fn calculate_variance(&self, data: &[f32]) -> f32 {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        variance
    }

    fn calculate_skewness(&self, data: &[f32]) -> f32 {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = self.calculate_variance(data);
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        let skewness = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f32>()
            / data.len() as f32;
        skewness
    }

    async fn distance_based_detection(
        &self,
        source_features: &[f32],
        target_features: &[f32],
        metric: &DistanceMetric,
    ) -> Result<f32> {
        match metric {
            DistanceMetric::Wasserstein => {
                self.wasserstein_distance(source_features, target_features)
                    .await
            }
            DistanceMetric::JensenShannon => {
                self.jensen_shannon_divergence(source_features, target_features)
                    .await
            }
            DistanceMetric::KLDivergence => {
                self.kl_divergence(source_features, target_features).await
            }
            DistanceMetric::MMD => self.mmd_distance(source_features, target_features).await,
        }
    }

    async fn wasserstein_distance(&self, source: &[f32], target: &[f32]) -> Result<f32> {
        // Simplified 1D Wasserstein distance
        let mut source_sorted = source.to_vec();
        let mut target_sorted = target.to_vec();
        source_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        target_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_len = source_sorted.len().min(target_sorted.len());
        let distance = (0..min_len)
            .map(|i| (source_sorted[i] - target_sorted[i]).abs())
            .sum::<f32>()
            / min_len as f32;

        Ok(distance)
    }

    async fn jensen_shannon_divergence(&self, source: &[f32], target: &[f32]) -> Result<f32> {
        // Simplified JS divergence
        let source_mean = source.iter().sum::<f32>() / source.len() as f32;
        let target_mean = target.iter().sum::<f32>() / target.len() as f32;
        Ok((source_mean - target_mean).abs() / (source_mean + target_mean).max(1e-8))
    }

    async fn kl_divergence(&self, source: &[f32], target: &[f32]) -> Result<f32> {
        // Simplified KL divergence
        let source_var = self.calculate_variance(source);
        let target_var = self.calculate_variance(target);
        Ok((source_var / target_var.max(1e-8)).ln().abs())
    }

    async fn mmd_distance(&self, source: &[f32], target: &[f32]) -> Result<f32> {
        // Simplified MMD with RBF kernel
        let source_mean = source.iter().sum::<f32>() / source.len() as f32;
        let target_mean = target.iter().sum::<f32>() / target.len() as f32;
        let gamma = 1.0;
        Ok((-gamma * (source_mean - target_mean).powi(2)).exp())
    }

    async fn density_based_detection(
        &self,
        source_features: &[f32],
        target_features: &[f32],
    ) -> Result<f32> {
        // Simplified density-based detection using histogram comparison
        let source_hist = self.create_histogram(source_features, 10);
        let target_hist = self.create_histogram(target_features, 10);

        let distance = source_hist
            .iter()
            .zip(target_hist.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / source_hist.len() as f32;

        Ok(distance)
    }

    async fn model_based_detection(
        &self,
        source_features: &[f32],
        target_features: &[f32],
    ) -> Result<f32> {
        // Simplified model-based detection using statistical moments
        let source_moments = self.calculate_moments(source_features);
        let target_moments = self.calculate_moments(target_features);

        let distance = source_moments
            .iter()
            .zip(target_moments.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / source_moments.len() as f32;

        Ok(distance)
    }

    async fn basic_statistical_detection(
        &self,
        source_features: &[f32],
        target_features: &[f32],
    ) -> Result<f32> {
        // Basic statistical comparison
        let source_mean = source_features.iter().sum::<f32>() / source_features.len() as f32;
        let target_mean = target_features.iter().sum::<f32>() / target_features.len() as f32;
        Ok((source_mean - target_mean).abs() / source_mean.max(target_mean))
    }

    fn create_histogram(&self, data: &[f32], bins: usize) -> Vec<f32> {
        if data.is_empty() {
            return vec![0.0; bins];
        }

        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        if range == 0.0 {
            let mut hist = vec![0.0; bins];
            hist[0] = data.len() as f32;
            return hist;
        }

        let mut hist = vec![0.0; bins];
        for &value in data {
            let bin = ((value - min_val) / range * (bins - 1) as f32) as usize;
            let bin = bin.min(bins - 1);
            hist[bin] += 1.0;
        }

        // Normalize
        let total = data.len() as f32;
        hist.iter_mut().for_each(|x| *x /= total);

        hist
    }

    fn calculate_moments(&self, data: &[f32]) -> Vec<f32> {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = self.calculate_variance(data);
        let skewness = self.calculate_skewness(data);
        let kurtosis = self.calculate_kurtosis(data);

        vec![mean, variance, skewness, kurtosis]
    }

    fn calculate_kurtosis(&self, data: &[f32]) -> f32 {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = self.calculate_variance(data);
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        let kurtosis = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f32>()
            / data.len() as f32;
        kurtosis - 3.0 // Excess kurtosis
    }
}

impl FeatureAligner {
    fn new(strategy: &AdaptationStrategy) -> Result<Self> {
        Ok(Self {
            strategy: strategy.clone(),
        })
    }

    async fn align_features(
        &self,
        source_samples: &[DatasetSample],
        target_samples: &[DatasetSample],
    ) -> Result<(Vec<DatasetSample>, Vec<DatasetSample>)> {
        // Simplified feature alignment
        match &self.strategy {
            AdaptationStrategy::None => Ok((source_samples.to_vec(), target_samples.to_vec())),
            _ => {
                // Apply basic normalization for all other strategies
                let aligned_source = self.normalize_samples(source_samples).await?;
                let aligned_target = self.normalize_samples(target_samples).await?;
                Ok((aligned_source, aligned_target))
            }
        }
    }

    async fn normalize_samples(&self, samples: &[DatasetSample]) -> Result<Vec<DatasetSample>> {
        // Basic normalization - in practice this would be more sophisticated
        let mut normalized = samples.to_vec();

        // Normalize audio RMS if available
        if let Some(max_rms) = samples
            .iter()
            .filter_map(|s| s.audio.rms())
            .fold(None, |max, rms| {
                max.map_or(Some(rms), |m: f32| Some(m.max(rms)))
            })
        {
            for sample in &mut normalized {
                if let Some(_current_rms) = sample.audio.rms() {
                    let scale_factor = 0.5 / max_rms; // Normalize to 0.5 max RMS
                    let mut scaled_samples = sample.audio.samples().to_vec();
                    for sample_val in &mut scaled_samples {
                        *sample_val *= scale_factor;
                    }
                    sample.audio = crate::AudioData::new(
                        scaled_samples,
                        sample.audio.sample_rate(),
                        sample.audio.channels(),
                    );
                }
            }
        }

        Ok(normalized)
    }
}

impl TransferLearner {
    fn new(config: &TransferLearningConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    async fn apply_transfer_learning(
        &mut self,
        source_samples: &[DatasetSample],
        target_samples: &[DatasetSample],
    ) -> Result<()> {
        // Simplified transfer learning implementation
        tracing::info!(
            "Applying transfer learning from {} source samples to {} target samples",
            source_samples.len(),
            target_samples.len()
        );

        // In practice, this would involve:
        // 1. Loading the source model
        // 2. Applying the fine-tuning strategy
        // 3. Training on target data
        // 4. Saving the adapted model

        match &self.config.fine_tuning_strategy {
            super::config::FineTuningStrategy::Freeze => {
                tracing::info!("Using freeze strategy - keeping transferred layers frozen");
            }
            super::config::FineTuningStrategy::GradualUnfreeze { .. } => {
                tracing::info!("Using gradual unfreeze strategy");
            }
            _ => {
                tracing::info!("Using default fine-tuning strategy");
            }
        }

        Ok(())
    }
}

impl DataMixer {
    fn new(config: &DataMixingConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    async fn mix_data(
        &self,
        domain_samples: &[(DomainConfig, Vec<DatasetSample>)],
    ) -> Result<Vec<DatasetSample>> {
        let mut mixed_samples = Vec::new();

        for (domain_config, samples) in domain_samples {
            let weight = if domain_config.name.contains("source") {
                self.config.source_weight
            } else {
                self.config.target_weight
            };

            let num_samples = (samples.len() as f32 * weight) as usize;
            let selected_samples = &samples[..num_samples.min(samples.len())];
            mixed_samples.extend_from_slice(selected_samples);
        }

        // Shuffle the mixed samples
        use scirs2_core::random::seq::SliceRandom;
        let mut rng = scirs2_core::random::thread_rng();
        mixed_samples.shuffle(&mut rng);

        Ok(mixed_samples)
    }
}

#[async_trait::async_trait]
impl DomainAdapter for DomainAdapterImpl {
    async fn detect_domain_shift(
        &self,
        source_samples: &[DatasetSample],
        target_samples: &[DatasetSample],
    ) -> Result<DomainShift> {
        self.shift_detector
            .detect_shift(source_samples, target_samples)
            .await
    }

    async fn adapt_domain(
        &self,
        source_samples: &[DatasetSample],
        target_samples: &[DatasetSample],
    ) -> Result<AdaptationResult> {
        // 1. Detect domain shift
        let shift = self
            .detect_domain_shift(source_samples, target_samples)
            .await?;

        if !shift.is_significant(0.3) {
            return Ok(AdaptationResult::success(0.0, source_samples.to_vec()));
        }

        // 2. Align features
        let (aligned_source, aligned_target) = self
            .feature_aligner
            .align_features(source_samples, target_samples)
            .await?;

        // 3. Mix data
        let domain_samples = vec![
            (self.config.source_domain.clone(), aligned_source),
            (self.config.target_domain.clone(), aligned_target),
        ];
        let mixed_samples = self.data_mixer.mix_data(&domain_samples).await?;

        let improvement = shift.magnitude * 0.5; // Simplified improvement calculation
        Ok(AdaptationResult::success(improvement, mixed_samples))
    }

    async fn apply_preprocessing(
        &self,
        samples: &[DatasetSample],
        domain_config: &DomainConfig,
    ) -> Result<Vec<DatasetSample>> {
        let mut processed_samples = samples.to_vec();

        // Apply preprocessing steps from domain config
        for step in &domain_config.preprocessing {
            processed_samples = self
                .apply_preprocessing_step(processed_samples, step)
                .await?;
        }

        Ok(processed_samples)
    }

    async fn mix_domains(
        &self,
        domain_samples: &[(DomainConfig, Vec<DatasetSample>)],
    ) -> Result<Vec<DatasetSample>> {
        self.data_mixer.mix_data(domain_samples).await
    }

    async fn transfer_learning(
        &mut self,
        source_samples: &[DatasetSample],
        target_samples: &[DatasetSample],
    ) -> Result<()> {
        self.transfer_learner
            .apply_transfer_learning(source_samples, target_samples)
            .await
    }

    async fn get_domain_statistics(&self, samples: &[DatasetSample]) -> Result<DomainStatistics> {
        let total_items = samples.len();
        let total_duration: f32 = samples.iter().map(|s| s.duration()).sum();
        let average_duration = if total_items > 0 {
            total_duration / total_items as f32
        } else {
            0.0
        };

        // Create basic dataset statistics
        let dataset_stats = crate::DatasetStatistics {
            total_items,
            total_duration,
            average_duration,
            language_distribution: std::collections::HashMap::new(),
            speaker_distribution: std::collections::HashMap::new(),
            text_length_stats: crate::LengthStatistics {
                min: 0,
                max: 0,
                mean: 0.0,
                median: 0,
                std_dev: 0.0,
            },
            duration_stats: crate::DurationStatistics {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
            },
        };

        // Calculate audio statistics
        let mut audio_stats = AudioStatistics::new();
        for sample in samples {
            let sample_rate = sample.audio.sample_rate();
            let channels = sample.audio.channels();
            *audio_stats
                .sample_rate_distribution
                .entry(sample_rate)
                .or_insert(0.0) += 1.0;
            *audio_stats
                .channel_distribution
                .entry(channels)
                .or_insert(0.0) += 1.0;
        }

        // Normalize distributions
        let total_samples = samples.len() as f32;
        for count in audio_stats.sample_rate_distribution.values_mut() {
            *count /= total_samples;
        }
        for count in audio_stats.channel_distribution.values_mut() {
            *count /= total_samples;
        }

        // Calculate text statistics
        let text_stats = self.calculate_text_statistics(samples).await?;

        // Calculate speaker statistics
        let speaker_stats = self.calculate_speaker_statistics(samples).await?;

        Ok(DomainStatistics::new(
            dataset_stats,
            audio_stats,
            text_stats,
            speaker_stats,
        ))
    }

    async fn validate_compatibility(
        &self,
        source_domain: &DomainConfig,
        target_domain: &DomainConfig,
    ) -> Result<CompatibilityReport> {
        let similarity_score = source_domain.similarity_score(target_domain);

        // Calculate component-wise compatibility
        let audio_compatibility = self.calculate_audio_compatibility(source_domain, target_domain);
        let text_compatibility = self.calculate_text_compatibility(source_domain, target_domain);
        let speaker_compatibility =
            self.calculate_speaker_compatibility(source_domain, target_domain);

        let mut report = CompatibilityReport::new(
            similarity_score,
            audio_compatibility,
            text_compatibility,
            speaker_compatibility,
        );

        // Generate recommendations
        if audio_compatibility < 0.7 {
            report.add_recommendation(AdaptationRecommendation::new(
                RecommendationType::AudioPreprocessing,
                Priority::High,
                "Audio characteristics differ significantly. Consider resampling and normalization.".to_string(),
                0.3,
            ));
        }

        if text_compatibility < 0.6 {
            report.add_recommendation(AdaptationRecommendation::new(
                RecommendationType::TextNormalization,
                Priority::Medium,
                "Text styles differ. Consider text normalization and style adaptation.".to_string(),
                0.2,
            ));
        }

        Ok(report)
    }
}

impl DomainAdapterImpl {
    async fn apply_preprocessing_step(
        &self,
        samples: Vec<DatasetSample>,
        step: &super::config::PreprocessingStep,
    ) -> Result<Vec<DatasetSample>> {
        match step {
            super::config::PreprocessingStep::AudioNormalization(_) => {
                // Apply audio normalization
                self.normalize_audio_samples(samples).await
            }
            super::config::PreprocessingStep::Resampling { target_rate, .. } => {
                // Apply resampling
                self.resample_audio(samples, *target_rate).await
            }
            _ => {
                // For other preprocessing steps, return as-is for now
                Ok(samples)
            }
        }
    }

    async fn normalize_audio_samples(
        &self,
        mut samples: Vec<DatasetSample>,
    ) -> Result<Vec<DatasetSample>> {
        // Find peak across all samples
        let peak = samples
            .iter()
            .map(|s| {
                s.audio
                    .samples()
                    .iter()
                    .fold(0.0f32, |max, &val| max.max(val.abs()))
            })
            .fold(0.0, f32::max);

        if peak > 0.0 {
            let scale_factor = 0.9 / peak; // Normalize to 90% of full scale
            for sample in &mut samples {
                let mut scaled_samples = sample.audio.samples().to_vec();
                for sample_val in &mut scaled_samples {
                    *sample_val *= scale_factor;
                }
                sample.audio = crate::AudioData::new(
                    scaled_samples,
                    sample.audio.sample_rate(),
                    sample.audio.channels(),
                );
            }
        }

        Ok(samples)
    }

    async fn resample_audio(
        &self,
        mut samples: Vec<DatasetSample>,
        target_rate: u32,
    ) -> Result<Vec<DatasetSample>> {
        for sample in &mut samples {
            if sample.audio.sample_rate() != target_rate {
                sample.audio = sample.audio.resample(target_rate)?;
            }
        }
        Ok(samples)
    }

    async fn calculate_text_statistics(&self, samples: &[DatasetSample]) -> Result<TextStatistics> {
        let mut text_stats = TextStatistics::new();

        // Calculate basic text statistics
        let mut total_length = 0;
        let mut min_length = usize::MAX;
        let mut max_length = 0;
        let mut word_counts = HashMap::new();

        for sample in samples {
            let length = sample.text.len();
            total_length += length;
            min_length = min_length.min(length);
            max_length = max_length.max(length);

            // Count words
            for word in sample.text.split_whitespace() {
                *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }

        let avg_length = if !samples.is_empty() {
            total_length as f32 / samples.len() as f32
        } else {
            0.0
        };

        text_stats.length_distribution = (min_length, max_length, avg_length);

        // Create vocabulary statistics
        let mut frequent_words: Vec<_> = word_counts.into_iter().collect();
        frequent_words.sort_by(|a, b| b.1.cmp(&a.1));
        frequent_words.truncate(100); // Top 100 words

        text_stats.vocabulary_stats = VocabularyStatistics {
            unique_words: frequent_words.len(),
            frequent_words,
            word_frequency_distribution: HashMap::new(), // Simplified
            oov_rate: 0.05,                              // Placeholder
        };

        Ok(text_stats)
    }

    async fn calculate_speaker_statistics(
        &self,
        samples: &[DatasetSample],
    ) -> Result<SpeakerStatistics> {
        let mut speaker_stats = SpeakerStatistics::new();

        let mut speaker_ids = std::collections::HashSet::new();
        for sample in samples {
            if let Some(speaker_id) = sample.speaker_id() {
                speaker_ids.insert(speaker_id.to_string());
            }
        }

        speaker_stats.unique_speakers = speaker_ids.len();

        // Create speaker distribution
        for speaker_id in speaker_ids {
            let count = samples
                .iter()
                .filter(|s| s.speaker_id() == Some(&speaker_id))
                .count();
            let frequency = count as f32 / samples.len() as f32;
            speaker_stats
                .speaker_distribution
                .insert(speaker_id, frequency);
        }

        Ok(speaker_stats)
    }

    fn calculate_audio_compatibility(&self, source: &DomainConfig, target: &DomainConfig) -> f32 {
        let sr_overlap = (source
            .audio_characteristics
            .sample_rate_range
            .1
            .min(target.audio_characteristics.sample_rate_range.1)
            - source
                .audio_characteristics
                .sample_rate_range
                .0
                .max(target.audio_characteristics.sample_rate_range.0))
            as f32;
        let sr_union = (source
            .audio_characteristics
            .sample_rate_range
            .1
            .max(target.audio_characteristics.sample_rate_range.1)
            - source
                .audio_characteristics
                .sample_rate_range
                .0
                .min(target.audio_characteristics.sample_rate_range.0))
            as f32;

        if sr_union > 0.0 {
            (sr_overlap / sr_union).max(0.0)
        } else {
            1.0
        }
    }

    fn calculate_text_compatibility(&self, source: &DomainConfig, target: &DomainConfig) -> f32 {
        // Check language compatibility
        let language_overlap = source
            .languages
            .iter()
            .filter(|lang| target.languages.contains(lang))
            .count() as f32;
        let language_union =
            (source.languages.len() + target.languages.len()) as f32 - language_overlap;

        if language_union > 0.0 {
            language_overlap / language_union
        } else {
            1.0
        }
    }

    fn calculate_speaker_compatibility(&self, source: &DomainConfig, target: &DomainConfig) -> f32 {
        // Calculate gender distribution similarity
        1.0 - ((source.speaker_characteristics.gender_distribution.male
            - target.speaker_characteristics.gender_distribution.male)
            .abs()
            + (source.speaker_characteristics.gender_distribution.female
                - target.speaker_characteristics.gender_distribution.female)
                .abs())
            / 2.0
    }
}

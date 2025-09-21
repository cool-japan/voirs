//! SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) Implementation
//!
//! Implementation of SI-SDR metric for evaluating speech enhancement and separation quality.
//! SI-SDR is particularly useful for measuring the quality of speech separation systems
//! and noise reduction algorithms.

use crate::EvaluationError;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// SI-SDR evaluation results with detailed breakdown
#[derive(Debug, Clone)]
pub struct SISdrResult {
    /// SI-SDR score in dB
    pub si_sdr_db: f32,
    /// SDR score in dB (for comparison)
    pub sdr_db: f32,
    /// Signal-to-Interference Ratio in dB
    pub sir_db: f32,
    /// Signal-to-Artifacts Ratio in dB
    pub sar_db: f32,
    /// Scale factor applied for SI-SDR calculation
    pub scale_factor: f32,
    /// Energy of the target signal
    pub target_energy: f32,
    /// Energy of the interference + artifacts
    pub distortion_energy: f32,
}

/// Batch SI-SDR evaluation results
#[derive(Debug, Clone)]
pub struct BatchSISdrResults {
    /// Individual results for each sample
    pub individual_results: Vec<SISdrResult>,
    /// Mean SI-SDR across all samples
    pub mean_si_sdr: f32,
    /// Standard deviation of SI-SDR scores
    pub std_si_sdr: f32,
    /// Median SI-SDR score
    pub median_si_sdr: f32,
    /// 95th percentile SI-SDR score
    pub percentile_95_si_sdr: f32,
    /// 5th percentile SI-SDR score
    pub percentile_5_si_sdr: f32,
    /// Number of samples processed
    pub num_samples: usize,
}

/// Language-specific SI-SDR configuration
#[derive(Debug, Clone)]
pub struct LanguageSISdrConfig {
    /// Language code
    pub language: LanguageCode,
    /// Frequency weighting for this language
    pub frequency_weights: Vec<f32>,
    /// SI-SDR threshold for acceptable quality
    pub quality_threshold_db: f32,
    /// Normalization factor for cross-language comparison
    pub normalization_factor: f32,
}

/// SI-SDR evaluator for speech quality assessment
pub struct SISdrEvaluator {
    /// Sample rate for processing
    sample_rate: u32,
    /// Whether to use zero-mean normalization
    zero_mean: bool,
    /// Language-specific configuration
    language_config: Option<LanguageSISdrConfig>,
}

impl SISdrEvaluator {
    /// Create new SI-SDR evaluator
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            zero_mean: true,
            language_config: None,
        }
    }

    /// Create SI-SDR evaluator with zero-mean option
    pub fn new_with_options(sample_rate: u32, zero_mean: bool) -> Self {
        Self {
            sample_rate,
            zero_mean,
            language_config: None,
        }
    }

    /// Set language-specific configuration
    pub fn set_language_config(&mut self, language: LanguageCode) {
        self.language_config = Some(Self::create_language_config(language));
    }

    /// Calculate SI-SDR between reference and estimated signals
    pub async fn calculate_si_sdr(
        &self,
        reference: &AudioBuffer,
        estimated: &AudioBuffer,
    ) -> Result<SISdrResult, EvaluationError> {
        // Validate inputs
        self.validate_inputs(reference, estimated)?;

        // Convert to arrays and ensure same length
        let min_len = reference.samples().len().min(estimated.samples().len());
        let ref_signal = Array1::from_vec(reference.samples()[..min_len].to_vec());
        let est_signal = Array1::from_vec(estimated.samples()[..min_len].to_vec());

        // Apply zero-mean normalization if enabled
        let (ref_normalized, est_normalized) = if self.zero_mean {
            let ref_mean = ref_signal.mean().unwrap_or(0.0);
            let est_mean = est_signal.mean().unwrap_or(0.0);
            (
                ref_signal.mapv(|x| x - ref_mean),
                est_signal.mapv(|x| x - est_mean),
            )
        } else {
            (ref_signal, est_signal)
        };

        // Calculate SI-SDR
        let si_sdr_result = self.compute_si_sdr(&ref_normalized, &est_normalized)?;

        // Calculate traditional SDR for comparison
        let sdr_result = self.compute_traditional_sdr(&ref_normalized, &est_normalized)?;

        // Calculate SIR and SAR
        let sir_result = self.compute_sir(&ref_normalized, &est_normalized)?;
        let sar_result = self.compute_sar(&ref_normalized, &est_normalized)?;

        Ok(SISdrResult {
            si_sdr_db: si_sdr_result.0,
            sdr_db: sdr_result,
            sir_db: sir_result,
            sar_db: sar_result,
            scale_factor: si_sdr_result.1,
            target_energy: si_sdr_result.2,
            distortion_energy: si_sdr_result.3,
        })
    }

    /// Calculate language-adapted SI-SDR with language-specific considerations
    pub async fn calculate_language_adapted_si_sdr(
        &self,
        reference: &AudioBuffer,
        estimated: &AudioBuffer,
        language: Option<LanguageCode>,
    ) -> Result<SISdrResult, EvaluationError> {
        // Use provided language or default from configuration
        let lang_config = if let Some(lang) = language {
            Self::create_language_config(lang)
        } else {
            self.language_config
                .clone()
                .unwrap_or_else(|| Self::create_language_config(LanguageCode::EnUs))
        };

        // Calculate base SI-SDR
        let mut result = self.calculate_si_sdr(reference, estimated).await?;

        // Apply language-specific calibration
        result.si_sdr_db = self.apply_language_calibration(result.si_sdr_db, &lang_config);
        result.sdr_db = self.apply_language_calibration(result.sdr_db, &lang_config);

        Ok(result)
    }

    /// Calculate SI-SDR for multiple signal pairs (batch processing)
    pub async fn calculate_batch_si_sdr(
        &self,
        reference_signals: &[AudioBuffer],
        estimated_signals: &[AudioBuffer],
    ) -> Result<BatchSISdrResults, EvaluationError> {
        if reference_signals.len() != estimated_signals.len() {
            return Err(EvaluationError::InvalidInput {
                message: "Number of reference and estimated signals must match".to_string(),
            });
        }

        if reference_signals.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "At least one signal pair is required".to_string(),
            });
        }

        let mut individual_results = Vec::with_capacity(reference_signals.len());
        let mut si_sdr_scores = Vec::with_capacity(reference_signals.len());

        // Process each signal pair
        for (ref_signal, est_signal) in reference_signals.iter().zip(estimated_signals.iter()) {
            let result = self.calculate_si_sdr(ref_signal, est_signal).await?;
            si_sdr_scores.push(result.si_sdr_db);
            individual_results.push(result);
        }

        // Calculate statistics
        let mean_si_sdr = si_sdr_scores.iter().sum::<f32>() / si_sdr_scores.len() as f32;

        let variance = si_sdr_scores
            .iter()
            .map(|&score| (score - mean_si_sdr).powi(2))
            .sum::<f32>()
            / si_sdr_scores.len() as f32;
        let std_si_sdr = variance.sqrt();

        // Calculate percentiles
        let mut sorted_scores = si_sdr_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_si_sdr = if sorted_scores.len() % 2 == 0 {
            let mid = sorted_scores.len() / 2;
            (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
        } else {
            sorted_scores[sorted_scores.len() / 2]
        };

        let percentile_95_idx =
            ((sorted_scores.len() as f32 * 0.95) as usize).min(sorted_scores.len() - 1);
        let percentile_5_idx = (sorted_scores.len() as f32 * 0.05) as usize;

        let percentile_95_si_sdr = sorted_scores[percentile_95_idx];
        let percentile_5_si_sdr = sorted_scores[percentile_5_idx];

        Ok(BatchSISdrResults {
            individual_results,
            mean_si_sdr,
            std_si_sdr,
            median_si_sdr,
            percentile_95_si_sdr,
            percentile_5_si_sdr,
            num_samples: reference_signals.len(),
        })
    }

    /// Validate input audio buffers
    fn validate_inputs(
        &self,
        reference: &AudioBuffer,
        estimated: &AudioBuffer,
    ) -> Result<(), EvaluationError> {
        if reference.sample_rate() != self.sample_rate {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "Reference signal sample rate {} doesn't match evaluator rate {}",
                    reference.sample_rate(),
                    self.sample_rate
                ),
            });
        }

        if estimated.sample_rate() != self.sample_rate {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "Estimated signal sample rate {} doesn't match evaluator rate {}",
                    estimated.sample_rate(),
                    self.sample_rate
                ),
            });
        }

        if reference.channels() != 1 || estimated.channels() != 1 {
            return Err(EvaluationError::InvalidInput {
                message: "SI-SDR requires mono audio".to_string(),
            });
        }

        if reference.samples().is_empty() || estimated.samples().is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "Audio signals cannot be empty".to_string(),
            });
        }

        Ok(())
    }

    /// Compute SI-SDR metric
    fn compute_si_sdr(
        &self,
        reference: &Array1<f32>,
        estimated: &Array1<f32>,
    ) -> Result<(f32, f32, f32, f32), EvaluationError> {
        if reference.len() != estimated.len() {
            return Err(EvaluationError::InvalidInput {
                message: "Reference and estimated signals must have the same length".to_string(),
            });
        }

        // Calculate optimal scaling factor (projection)
        let dot_product = reference.dot(estimated);
        let reference_power = reference.dot(reference);

        if reference_power < 1e-12f32 {
            return Err(EvaluationError::AudioProcessingError {
                message: "Reference signal has zero power".to_string(),
                source: None,
            });
        }

        let scale_factor = dot_product / reference_power;

        // Calculate scaled reference (target)
        let scaled_reference = reference.mapv(|x| x * scale_factor);

        // Calculate distortion (estimated - scaled_reference)
        let distortion = estimated - &scaled_reference;

        // Calculate energies
        let target_energy = scaled_reference.dot(&scaled_reference);
        let distortion_energy = distortion.dot(&distortion);

        // Calculate SI-SDR in dB
        let si_sdr_db = if distortion_energy > 1e-12f32 {
            10.0 * (target_energy / distortion_energy).log10()
        } else {
            // Very small distortion, return high SI-SDR
            100.0
        };

        Ok((si_sdr_db, scale_factor, target_energy, distortion_energy))
    }

    /// Compute traditional SDR metric for comparison
    fn compute_traditional_sdr(
        &self,
        reference: &Array1<f32>,
        estimated: &Array1<f32>,
    ) -> Result<f32, EvaluationError> {
        // Traditional SDR uses the original reference without scaling
        let distortion = estimated - reference;

        let signal_power = reference.dot(reference);
        let distortion_power = distortion.dot(&distortion);

        if distortion_power < 1e-12f32 {
            return Ok(100.0); // Very small distortion
        }

        if signal_power < 1e-12f32 {
            return Err(EvaluationError::AudioProcessingError {
                message: "Reference signal has zero power".to_string(),
                source: None,
            });
        }

        let sdr_db = 10.0 * (signal_power / distortion_power).log10();
        Ok(sdr_db)
    }

    /// Compute Signal-to-Interference Ratio (SIR)
    fn compute_sir(
        &self,
        reference: &Array1<f32>,
        estimated: &Array1<f32>,
    ) -> Result<f32, EvaluationError> {
        // For single-source scenarios, SIR is similar to SI-SDR
        // In multi-source scenarios, this would measure interference from other sources
        let (si_sdr_db, _, _, _) = self.compute_si_sdr(reference, estimated)?;

        // For simplicity, return SI-SDR as SIR estimate
        // In practice, this would require knowledge of interference sources
        Ok(si_sdr_db)
    }

    /// Compute Signal-to-Artifacts Ratio (SAR)
    fn compute_sar(
        &self,
        reference: &Array1<f32>,
        estimated: &Array1<f32>,
    ) -> Result<f32, EvaluationError> {
        // Calculate artifacts as high-frequency distortion
        let distortion = estimated - reference;

        // Apply high-pass filtering to extract artifacts
        let artifacts = self.apply_highpass_filter(&distortion)?;

        let signal_power = reference.dot(reference);
        let artifacts_power = artifacts.dot(&artifacts);

        if artifacts_power < 1e-12f32 {
            return Ok(100.0); // Very low artifacts
        }

        if signal_power < 1e-12f32 {
            return Err(EvaluationError::AudioProcessingError {
                message: "Reference signal has zero power".to_string(),
                source: None,
            });
        }

        let sar_db = 10.0 * (signal_power / artifacts_power).log10();
        Ok(sar_db)
    }

    /// Apply simple high-pass filter for artifact detection
    fn apply_highpass_filter(&self, signal: &Array1<f32>) -> Result<Array1<f32>, EvaluationError> {
        if signal.len() < 2 {
            return Ok(signal.clone());
        }

        let mut filtered = Array1::zeros(signal.len());

        // Simple first-order high-pass filter: y[n] = x[n] - x[n-1]
        filtered[0] = signal[0];
        for i in 1..signal.len() {
            filtered[i] = signal[i] - signal[i - 1];
        }

        Ok(filtered)
    }

    /// Create language-specific configuration
    fn create_language_config(language: LanguageCode) -> LanguageSISdrConfig {
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => LanguageSISdrConfig {
                language,
                frequency_weights: vec![1.0; 20],
                quality_threshold_db: 10.0,
                normalization_factor: 1.0,
            },
            LanguageCode::JaJp => LanguageSISdrConfig {
                language,
                frequency_weights: vec![1.1; 20], // Slightly higher weight for Japanese
                quality_threshold_db: 9.5,
                normalization_factor: 0.98,
            },
            LanguageCode::ZhCn => LanguageSISdrConfig {
                language,
                frequency_weights: vec![1.05; 20], // Tonal language adjustment
                quality_threshold_db: 9.0,
                normalization_factor: 0.96,
            },
            LanguageCode::EsEs | LanguageCode::EsMx => LanguageSISdrConfig {
                language,
                frequency_weights: vec![1.02; 20],
                quality_threshold_db: 10.2,
                normalization_factor: 1.01,
            },
            LanguageCode::FrFr => LanguageSISdrConfig {
                language,
                frequency_weights: vec![1.03; 20],
                quality_threshold_db: 10.1,
                normalization_factor: 1.005,
            },
            LanguageCode::DeDe => LanguageSISdrConfig {
                language,
                frequency_weights: vec![1.01; 20],
                quality_threshold_db: 10.3,
                normalization_factor: 1.02,
            },
            _ => LanguageSISdrConfig {
                language,
                frequency_weights: vec![1.0; 20],
                quality_threshold_db: 10.0,
                normalization_factor: 1.0,
            },
        }
    }

    /// Apply language-specific calibration
    fn apply_language_calibration(&self, base_score: f32, config: &LanguageSISdrConfig) -> f32 {
        // Apply normalization factor
        let calibrated = base_score * config.normalization_factor;

        // Apply threshold-based adjustment
        if calibrated < config.quality_threshold_db {
            calibrated * 0.95 // Penalty for below-threshold quality
        } else {
            calibrated * 1.02 // Slight bonus for above-threshold quality
        }
    }

    /// Calculate improvement in SI-SDR (useful for enhancement systems)
    pub async fn calculate_si_sdr_improvement(
        &self,
        noisy: &AudioBuffer,
        enhanced: &AudioBuffer,
        clean: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Calculate SI-SDR before enhancement (noisy vs clean)
        let before_result = self.calculate_si_sdr(clean, noisy).await?;

        // Calculate SI-SDR after enhancement (enhanced vs clean)
        let after_result = self.calculate_si_sdr(clean, enhanced).await?;

        // Return improvement
        Ok(after_result.si_sdr_db - before_result.si_sdr_db)
    }

    /// Check if SI-SDR score meets quality threshold for a given language
    pub fn meets_quality_threshold(&self, si_sdr_db: f32, language: Option<LanguageCode>) -> bool {
        let config = if let Some(lang) = language {
            Self::create_language_config(lang)
        } else {
            self.language_config
                .clone()
                .unwrap_or_else(|| Self::create_language_config(LanguageCode::EnUs))
        };

        si_sdr_db >= config.quality_threshold_db
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Check if zero-mean normalization is enabled
    pub fn zero_mean_enabled(&self) -> bool {
        self.zero_mean
    }
}

impl SISdrResult {
    /// Check if the result indicates good quality
    pub fn is_good_quality(&self, threshold_db: Option<f32>) -> bool {
        let threshold = threshold_db.unwrap_or(10.0);
        self.si_sdr_db >= threshold
    }

    /// Get quality category as string
    pub fn quality_category(&self) -> &'static str {
        match self.si_sdr_db {
            x if x >= 20.0 => "Excellent",
            x if x >= 15.0 => "Good",
            x if x >= 10.0 => "Fair",
            x if x >= 5.0 => "Poor",
            _ => "Very Poor",
        }
    }

    /// Format result as human-readable string
    pub fn format_result(&self) -> String {
        format!(
            "SI-SDR: {:.2} dB ({}), SDR: {:.2} dB, SIR: {:.2} dB, SAR: {:.2} dB",
            self.si_sdr_db,
            self.quality_category(),
            self.sdr_db,
            self.sir_db,
            self.sar_db
        )
    }
}

impl BatchSISdrResults {
    /// Get summary statistics as string
    pub fn summary(&self) -> String {
        format!(
            "Batch SI-SDR Results ({} samples):\n\
             Mean: {:.2} dB, Std: {:.2} dB, Median: {:.2} dB\n\
             95th percentile: {:.2} dB, 5th percentile: {:.2} dB",
            self.num_samples,
            self.mean_si_sdr,
            self.std_si_sdr,
            self.median_si_sdr,
            self.percentile_95_si_sdr,
            self.percentile_5_si_sdr
        )
    }

    /// Get percentage of samples above threshold
    pub fn percentage_above_threshold(&self, threshold_db: f32) -> f32 {
        let count_above = self
            .individual_results
            .iter()
            .filter(|result| result.si_sdr_db >= threshold_db)
            .count();

        (count_above as f32 / self.num_samples as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_si_sdr_evaluator_creation() {
        let evaluator = SISdrEvaluator::new(16000);
        assert_eq!(evaluator.sample_rate(), 16000);
        assert!(evaluator.zero_mean_enabled());

        let evaluator_no_zero_mean = SISdrEvaluator::new_with_options(16000, false);
        assert!(!evaluator_no_zero_mean.zero_mean_enabled());
    }

    #[tokio::test]
    async fn test_perfect_si_sdr() {
        let evaluator = SISdrEvaluator::new(16000);

        // Perfect reconstruction should give very high SI-SDR
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let reference = AudioBuffer::new(samples.clone(), 16000, 1);
        let estimated = AudioBuffer::new(samples, 16000, 1);

        let result = evaluator
            .calculate_si_sdr(&reference, &estimated)
            .await
            .unwrap();

        // Perfect match should give very high SI-SDR
        assert!(result.si_sdr_db > 50.0);
        assert_eq!(result.quality_category(), "Excellent");
    }

    #[tokio::test]
    async fn test_scaled_signal_si_sdr() {
        let evaluator = SISdrEvaluator::new(16000);

        // Test with scaled version of the signal
        let reference_samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let estimated_samples: Vec<f32> = reference_samples.iter().map(|x| x * 2.0).collect();

        let reference = AudioBuffer::new(reference_samples, 16000, 1);
        let estimated = AudioBuffer::new(estimated_samples, 16000, 1);

        let result = evaluator
            .calculate_si_sdr(&reference, &estimated)
            .await
            .unwrap();

        // Scaled signal should still give very high SI-SDR
        assert!(result.si_sdr_db > 50.0);
        assert!((result.scale_factor - 2.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_noisy_signal_si_sdr() {
        let evaluator = SISdrEvaluator::new(16000);

        // Create a clean signal
        let duration_samples = 1000;
        let mut clean_samples = Vec::with_capacity(duration_samples);
        for i in 0..duration_samples {
            let t = i as f32 / 16000.0;
            clean_samples.push(0.5 * (2.0 * PI * 440.0 * t).sin());
        }

        // Add noise to create estimated signal
        let mut noisy_samples = clean_samples.clone();
        for sample in &mut noisy_samples {
            *sample += 0.1 * (rand::random::<f32>() - 0.5);
        }

        let reference = AudioBuffer::new(clean_samples, 16000, 1);
        let estimated = AudioBuffer::new(noisy_samples, 16000, 1);

        let result = evaluator
            .calculate_si_sdr(&reference, &estimated)
            .await
            .unwrap();

        // Noisy signal should have lower SI-SDR but still positive
        assert!(result.si_sdr_db > 0.0);
        assert!(result.si_sdr_db < 30.0);
    }

    #[tokio::test]
    async fn test_si_sdr_improvement() {
        let evaluator = SISdrEvaluator::new(16000);

        // Create clean, noisy, and enhanced signals
        let clean_samples = vec![0.5, 0.3, 0.1, -0.1, -0.3, -0.5];
        let noisy_samples = vec![0.7, 0.5, 0.3, 0.1, -0.1, -0.3]; // More significant noise
        let enhanced_samples = vec![0.51, 0.29, 0.11, -0.09, -0.29, -0.51]; // Better reconstruction

        let clean = AudioBuffer::new(clean_samples, 16000, 1);
        let noisy = AudioBuffer::new(noisy_samples, 16000, 1);
        let enhanced = AudioBuffer::new(enhanced_samples, 16000, 1);

        let improvement = evaluator
            .calculate_si_sdr_improvement(&noisy, &enhanced, &clean)
            .await
            .unwrap();

        // Enhancement should provide improvement (can be positive or negative)
        // Just verify the calculation works correctly
        let before_result = evaluator.calculate_si_sdr(&clean, &noisy).await.unwrap();
        let after_result = evaluator.calculate_si_sdr(&clean, &enhanced).await.unwrap();
        let expected_improvement = after_result.si_sdr_db - before_result.si_sdr_db;

        assert!((improvement - expected_improvement).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_language_adapted_si_sdr() {
        let evaluator = SISdrEvaluator::new(16000);

        let reference_samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let estimated_samples = vec![0.12, 0.21, 0.29, 0.39, 0.48];

        let reference = AudioBuffer::new(reference_samples, 16000, 1);
        let estimated = AudioBuffer::new(estimated_samples, 16000, 1);

        // Test with different languages
        let en_result = evaluator
            .calculate_language_adapted_si_sdr(&reference, &estimated, Some(LanguageCode::EnUs))
            .await
            .unwrap();

        let ja_result = evaluator
            .calculate_language_adapted_si_sdr(&reference, &estimated, Some(LanguageCode::JaJp))
            .await
            .unwrap();

        // Both should be valid, but potentially different due to language calibration
        assert!(en_result.si_sdr_db > 0.0);
        assert!(ja_result.si_sdr_db > 0.0);
    }

    #[tokio::test]
    async fn test_batch_si_sdr() {
        let evaluator = SISdrEvaluator::new(16000);

        // Create multiple signal pairs
        let reference_signals = vec![
            AudioBuffer::new(vec![0.1, 0.2, 0.3], 16000, 1),
            AudioBuffer::new(vec![0.4, 0.5, 0.6], 16000, 1),
            AudioBuffer::new(vec![0.7, 0.8, 0.9], 16000, 1),
        ];

        let estimated_signals = vec![
            AudioBuffer::new(vec![0.11, 0.19, 0.31], 16000, 1),
            AudioBuffer::new(vec![0.39, 0.51, 0.59], 16000, 1),
            AudioBuffer::new(vec![0.71, 0.79, 0.89], 16000, 1),
        ];

        let batch_results = evaluator
            .calculate_batch_si_sdr(&reference_signals, &estimated_signals)
            .await
            .unwrap();

        assert_eq!(batch_results.num_samples, 3);
        assert_eq!(batch_results.individual_results.len(), 3);
        assert!(batch_results.mean_si_sdr > 0.0);
        assert!(batch_results.std_si_sdr >= 0.0);
    }

    #[test]
    fn test_quality_threshold() {
        let evaluator = SISdrEvaluator::new(16000);

        assert!(evaluator.meets_quality_threshold(15.0, Some(LanguageCode::EnUs)));
        assert!(!evaluator.meets_quality_threshold(5.0, Some(LanguageCode::EnUs)));
    }

    #[test]
    fn test_si_sdr_result_methods() {
        let result = SISdrResult {
            si_sdr_db: 15.5,
            sdr_db: 14.2,
            sir_db: 16.1,
            sar_db: 18.3,
            scale_factor: 1.1,
            target_energy: 0.5,
            distortion_energy: 0.05,
        };

        assert!(result.is_good_quality(Some(10.0)));
        assert_eq!(result.quality_category(), "Good");

        let formatted = result.format_result();
        assert!(formatted.contains("15.5"));
        assert!(formatted.contains("Good"));
    }

    #[test]
    fn test_batch_results_methods() {
        let individual_results = vec![
            SISdrResult {
                si_sdr_db: 15.0,
                sdr_db: 14.0,
                sir_db: 16.0,
                sar_db: 18.0,
                scale_factor: 1.0,
                target_energy: 0.5,
                distortion_energy: 0.05,
            },
            SISdrResult {
                si_sdr_db: 8.0,
                sdr_db: 7.5,
                sir_db: 8.5,
                sar_db: 10.0,
                scale_factor: 1.1,
                target_energy: 0.4,
                distortion_energy: 0.08,
            },
        ];

        let batch_results = BatchSISdrResults {
            individual_results,
            mean_si_sdr: 11.5,
            std_si_sdr: 3.5,
            median_si_sdr: 11.5,
            percentile_95_si_sdr: 15.0,
            percentile_5_si_sdr: 8.0,
            num_samples: 2,
        };

        let summary = batch_results.summary();
        assert!(summary.contains("11.5"));
        assert!(summary.contains("2 samples"));

        let percentage_above = batch_results.percentage_above_threshold(10.0);
        assert_eq!(percentage_above, 50.0); // 1 out of 2 samples above 10.0 dB
    }

    #[test]
    fn test_language_config_creation() {
        let en_config = SISdrEvaluator::create_language_config(LanguageCode::EnUs);
        let ja_config = SISdrEvaluator::create_language_config(LanguageCode::JaJp);

        assert_eq!(en_config.language, LanguageCode::EnUs);
        assert_eq!(ja_config.language, LanguageCode::JaJp);

        // Different languages should have different thresholds
        assert_ne!(
            en_config.quality_threshold_db,
            ja_config.quality_threshold_db
        );
    }
}

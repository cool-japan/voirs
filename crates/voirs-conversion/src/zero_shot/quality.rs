//! Quality assessment for zero-shot voice conversion

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Quality assessor for zero-shot conversion
pub struct QualityAssessor {
    /// Quality metrics
    metrics: HashMap<String, Box<dyn QualityMetric>>,

    /// Quality thresholds
    thresholds: QualityThresholds,

    /// Assessment history
    history: Arc<RwLock<Vec<QualityAssessment>>>,

    /// Configuration
    config: QualityAssessmentConfig,
}

/// Quality metric trait
pub trait QualityMetric: Send + Sync {
    /// Assess quality
    fn assess_quality(&self, original: &[f32], converted: &[f32], sample_rate: u32) -> Result<f32>;

    /// Get metric name
    fn name(&self) -> &str;

    /// Get metric range
    fn range(&self) -> (f32, f32);
}

/// Quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum acceptable quality
    pub min_acceptable: f32,

    /// Good quality threshold
    pub good_quality: f32,

    /// Excellent quality threshold
    pub excellent_quality: f32,

    /// Per-metric thresholds
    pub metric_thresholds: HashMap<String, f32>,
}

/// Quality assessment result
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall quality score
    pub overall_score: f32,

    /// Individual metric scores
    pub metric_scores: HashMap<String, f32>,

    /// Quality classification
    pub classification: QualityClassification,

    /// Assessment timestamp
    pub timestamp: Instant,

    /// Assessment confidence
    pub confidence: f32,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Quality classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityClassification {
    /// Poor quality
    Poor,

    /// Acceptable quality
    Acceptable,

    /// Good quality
    Good,

    /// Excellent quality
    Excellent,
}

/// Quality assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessmentConfig {
    /// Enabled quality metrics
    pub enabled_metrics: Vec<String>,

    /// Assessment mode
    pub assessment_mode: AssessmentMode,

    /// Real-time assessment
    pub realtime_assessment: bool,

    /// Assessment frequency
    pub assessment_frequency: AssessmentFrequency,
}

/// Assessment mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssessmentMode {
    /// Fast assessment
    Fast,

    /// Comprehensive assessment
    Comprehensive,

    /// Custom assessment
    Custom,
}

/// Assessment frequency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssessmentFrequency {
    /// Assess every conversion
    Every,

    /// Periodic assessment
    Periodic,

    /// On-demand assessment
    OnDemand,
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityAssessor {
    /// Creates a new quality assessor with default configuration.
    ///
    /// Initializes the assessor with:
    /// - SNR and spectral distance metrics
    /// - Quality thresholds (min: 0.6, good: 0.75, excellent: 0.9)
    /// - Real-time assessment enabled
    /// - Fast assessment mode
    /// - Empty assessment history
    ///
    /// # Returns
    ///
    /// A new [`QualityAssessor`] instance ready to evaluate voice conversion quality.
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            thresholds: QualityThresholds {
                min_acceptable: 0.6,
                good_quality: 0.75,
                excellent_quality: 0.9,
                metric_thresholds: HashMap::new(),
            },
            history: Arc::new(RwLock::new(Vec::new())),
            config: QualityAssessmentConfig {
                enabled_metrics: vec!["snr".to_string(), "spectral_distance".to_string()],
                assessment_mode: AssessmentMode::Fast,
                realtime_assessment: true,
                assessment_frequency: AssessmentFrequency::Every,
            },
        }
    }

    /// Assesses the overall quality of converted audio compared to the original.
    ///
    /// Computes a weighted combination of Signal-to-Noise Ratio (SNR) and spectral distance
    /// to produce an overall quality score. The score is normalized to the range [0.0, 1.0],
    /// where higher values indicate better quality.
    ///
    /// # Arguments
    ///
    /// * `original` - Original audio samples as f32 values
    /// * `converted` - Converted audio samples as f32 values
    /// * `sample_rate` - Audio sample rate in Hz (e.g., 16000, 22050, 44100)
    ///
    /// # Returns
    ///
    /// A `Result` containing the overall quality score (0.0 to 1.0), or an error if assessment fails.
    /// The score combines SNR (60% weight) and inverted spectral distance (40% weight).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use voirs_conversion::zero_shot::quality::QualityAssessor;
    /// let assessor = QualityAssessor::new();
    /// let original = vec![0.0f32; 16000];
    /// let converted = vec![0.0f32; 16000];
    /// let quality = assessor.assess_overall_quality(&original, &converted, 16000)?;
    /// println!("Quality score: {}", quality);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn assess_overall_quality(
        &self,
        original: &[f32],
        converted: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        // Simplified quality assessment
        let snr = self.calculate_snr(original, converted)?;
        let spectral_dist = self.calculate_spectral_distance(original, converted, sample_rate)?;

        // Combine metrics
        let quality_score = (snr * 0.6 + (1.0 - spectral_dist) * 0.4).clamp(0.0, 1.0);

        Ok(quality_score)
    }

    fn calculate_snr(&self, original: &[f32], converted: &[f32]) -> Result<f32> {
        if original.len() != converted.len() {
            return Ok(0.0);
        }

        let signal_power: f32 = original.iter().map(|x| x * x).sum();
        let noise_power: f32 = original
            .iter()
            .zip(converted.iter())
            .map(|(o, c)| (o - c).powi(2))
            .sum();

        if noise_power == 0.0 {
            return Ok(1.0);
        }

        let snr_db = 10.0 * (signal_power / noise_power).log10();
        Ok((snr_db / 40.0).clamp(0.0, 1.0)) // Normalize to 0-1 range
    }

    fn calculate_spectral_distance(
        &self,
        original: &[f32],
        converted: &[f32],
        sample_rate: u32,
    ) -> Result<f32> {
        // Simplified spectral distance calculation
        // In practice, would use FFT and proper spectral analysis
        let orig_energy: f32 = original.iter().map(|x| x * x).sum();
        let conv_energy: f32 = converted.iter().map(|x| x * x).sum();

        let energy_diff = (orig_energy - conv_energy).abs() / orig_energy.max(1e-10);
        Ok(energy_diff.min(1.0))
    }
}

impl Default for QualityAssessment {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            metric_scores: HashMap::new(),
            classification: QualityClassification::Poor,
            timestamp: Instant::now(),
            confidence: 0.0,
            recommendations: Vec::new(),
        }
    }
}
